#!/usr/bin/env python3
"""
SAM 3D Body 羽毛球动作分析服务器 v2
- 三阶段分析：引拍 / 击球 / 随挥
- 正面 + 背面 3D 视角
- 挥拍动作链分析
"""
import sys, os, json, base64, time, traceback, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sam-3d-body'))

import torch, cv2, numpy as np
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

# ============ 加载模型 ============
device = torch.device('cpu')
print("Loading SAM 3D Body model...")
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

CKPT = os.path.join(os.path.dirname(__file__), 'sam-3d-body', 'checkpoints', 'sam-3d-body-dinov3')
model, model_cfg = load_sam_3d_body(
    os.path.join(CKPT, 'model.ckpt'), device=device,
    mhr_path=os.path.join(CKPT, 'assets', 'mhr_model.pt')
)
estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=model_cfg)
print("✅ SAM 3D Body loaded!")

# ============ MHR70 关键点 ============
KP = {
    'pelvis':0,'l_hip':1,'r_hip':2,'l_knee':4,'r_knee':5,
    'l_ankle':7,'r_ankle':8,'neck':12,'l_shoulder':16,'r_shoulder':17,
    'l_elbow':18,'r_elbow':19,'l_wrist':20,'r_wrist':21,
    'r_mid':29,'l_mid':28,
}

def angle_3d(a,b,c):
    ba,bc=a-b,c-b
    m=np.linalg.norm(ba)*np.linalg.norm(bc)
    return float(np.degrees(np.arccos(np.clip(np.dot(ba,bc)/max(m,1e-8),-1,1)))) if m>1e-8 else None

def analyze_single(k3d):
    """单帧 3D 分析"""
    a={}
    a['r_elbow']=angle_3d(k3d[KP['r_shoulder']],k3d[KP['r_elbow']],k3d[KP['r_wrist']])
    a['r_shoulder_abd']=angle_3d(k3d[KP['r_hip']],k3d[KP['r_shoulder']],k3d[KP['r_elbow']])
    a['r_wrist']=angle_3d(k3d[KP['r_elbow']],k3d[KP['r_wrist']],k3d[KP['r_mid']])
    a['l_elbow']=angle_3d(k3d[KP['l_shoulder']],k3d[KP['l_elbow']],k3d[KP['l_wrist']])
    a['l_shoulder_up']=angle_3d(k3d[KP['l_hip']],k3d[KP['l_shoulder']],k3d[KP['l_elbow']])
    a['r_knee']=angle_3d(k3d[KP['r_hip']],k3d[KP['r_knee']],k3d[KP['r_ankle']])
    a['l_knee']=angle_3d(k3d[KP['l_hip']],k3d[KP['l_knee']],k3d[KP['l_ankle']])
    sv=k3d[KP['r_shoulder']]-k3d[KP['l_shoulder']]
    hv=k3d[KP['r_hip']]-k3d[KP['l_hip']]
    sh,hh=np.array([sv[0],sv[2]]),np.array([hv[0],hv[2]])
    d,m=np.dot(sh,hh),np.linalg.norm(sh)*np.linalg.norm(hh)
    a['torso_twist']=float(np.degrees(np.arccos(np.clip(d/max(m,1e-8),-1,1))))
    sm=(k3d[KP['r_shoulder']]+k3d[KP['l_shoulder']])/2
    hm=(k3d[KP['r_hip']]+k3d[KP['l_hip']])/2
    tv=sm-hm;d2,m2=np.dot(tv,np.array([0,1,0])),np.linalg.norm(tv)
    a['body_tilt']=float(np.degrees(np.arccos(np.clip(d2/max(m2,1e-8),-1,1))))
    a['is_overhead']=float(k3d[KP['r_wrist']][1]>k3d[KP['r_shoulder']][1])
    # 手腕速度代理：手腕与肘的距离变化
    a['arm_extension']=float(np.linalg.norm(k3d[KP['r_wrist']]-k3d[KP['r_shoulder']]))
    return {k:round(v,1)if v is not None else None for k,v in a.items()}

def render_back_view(k3d, vertices):
    """从背面渲染 3D 关键点（绕 Y 轴旋转 180°）"""
    rot180=np.array([[-1,0,0],[0,1,0],[0,0,-1]])
    back_k3d=k3d@rot180.T
    # 返回背面关键点坐标
    return back_k3d.tolist()

def feedback_phase(angles, phase_name):
    """单阶段诊断"""
    fb,s=[],100
    v=angles.get('r_elbow')
    if phase_name=='hit':
        if v and v<120:fb.append({'l':'bad','m':f'{phase_name}:右肘弯曲严重({v}°)'});s-=25
        elif v and v<150:fb.append({'l':'warn','m':f'{phase_name}:右肘未完全伸展({v}°)'});s-=10
        else:fb.append({'l':'good','m':f'{phase_name}:右肘伸展良好({v}°)'})
    v=angles.get('torso_twist')
    if v and v<15:fb.append({'l':'bad','m':f'{phase_name}:转体不足({v}°)'});s-=15
    elif v and v<30:fb.append({'l':'warn','m':f'{phase_name}:转体偏小({v}°)'});s-=5
    return s,fb

def run_inference(img_b64):
    """解码 + 推理"""
    if ',' in img_b64:img_b64=img_b64.split(',',1)[1]
    img=cv2.imdecode(np.frombuffer(base64.b64decode(img_b64),np.uint8),cv2.IMREAD_COLOR)
    if img is None:return None
    out=estimator.process_one_image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    r=out[0]if isinstance(out,list)else out
    k3d=r['pred_keypoints_3d'].cpu().numpy()if hasattr(r['pred_keypoints_3d'],'cpu')else r['pred_keypoints_3d']
    verts=r['pred_vertices'].cpu().numpy()if hasattr(r['pred_vertices'],'cpu')else r['pred_vertices']
    return k3d,verts

@app.route('/analyze',methods=['POST'])
def analyze_route():
    try:
        d=request.get_json()
        if not d:return jsonify({'error':'No data'}),400

        results={}
        phases=['backswing','hit','followThrough']

        for phase in phases:
            key_map={'backswing':'backswingSS','hit':'image','followThrough':'followThroughSS'}
            key=key_map[phase]
            img_b64=d.get(key)
            if not img_b64:continue

            t0=time.time()
            res=run_inference(img_b64)
            if res is None:
                results[phase]={'error':'Bad image'}
                continue
            k3d,verts=res
            elapsed=time.time()-t0

            ang=analyze_single(k3d)
            s,fb=feedback_phase(ang,{'backswing':'引拍','hit':'击球','followThrough':'随挥'}[phase])
            back_k3d=render_back_view(k3d,verts)

            results[phase]={
                'elapsed':round(elapsed,1),'angles':ang,
                'score':max(0,min(100,s)),'feedback':fb,
                'back_keypoints_3d':back_k3d
            }

        # === 动作链分析：引拍→击球→随挥 ===
        chain_feedback=[]
        if 'backswing' in results and 'hit' in results and 'followThrough' in results:
            bs=results['backswing']['angles']
            ht=results['hit']['angles']
            ft=results['followThrough']['angles']

            # 引拍到击球：肘角度变化
            if bs.get('r_elbow') and ht.get('r_elbow'):
                elbow_change=ht['r_elbow']-bs['r_elbow']
                if elbow_change>20:
                    chain_feedback.append({'l':'good','m':f'引拍→击球肘伸展+{elbow_change:.0f}°，手臂发力充分'})
                elif elbow_change<0:
                    chain_feedback.append({'l':'bad','m':f'引拍→击球肘角度反而变小({elbow_change:.0f}°)，发力方向错误'})

            # 击球到随挥：手臂继续前送
            if ht.get('r_elbow') and ft.get('r_elbow'):
                follow_change=ft['r_elbow']-ht['r_elbow']
                if follow_change>-10:
                    chain_feedback.append({'l':'good','m':f'击球→随挥手臂保持伸展，动作连贯'})
                else:
                    chain_feedback.append({'l':'warn','m':f'击球→随挥手臂过早收回，缺少完整的随挥动作'})

            # 扭转角变化
            if bs.get('torso_twist') and ht.get('torso_twist'):
                twist_delta=ht['torso_twist']-bs['torso_twist']
                if twist_delta>5:
                    chain_feedback.append({'l':'good','m':f'引拍→击球转体+{twist_delta:.0f}°，核心发力显著'})
                elif twist_delta<-5:
                    chain_feedback.append({'l':'bad','m':f'转体方向异常({twist_delta:.0f}°)，可能提前打开身体'})

        # 综合评分
        scores=[r.get('score',0)for r in results.values()if'error'not in r]
        total_score=round(sum(scores)/max(len(scores),1))

        return jsonify({
            'status':'ok',
            'phases':results,
            'chain_feedback':chain_feedback,
            'total_score':total_score,
            'grade':'great'if total_score>=80 else'ok'if total_score>=55 else'poor'
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error':str(e)}),500

@app.route('/health',methods=['GET'])
def health():
    return jsonify({'status':'ok'})

if __name__=='__main__':
    print("\n🏸 SAM 3D Body 羽毛球 v2 — 三阶段 + 背面视图")
    print("POST http://localhost:8765/analyze\n")
    app.run(host='0.0.0.0',port=8765,debug=False)
