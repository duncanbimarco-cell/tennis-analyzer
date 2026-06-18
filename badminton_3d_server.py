#!/usr/bin/env python3
"""
SAM 3D Body 羽毛球动作分析服务器
接受击球关键帧图片，返回 3D 全身关节角度分析
"""
import sys, os, json, base64, time, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sam-3d-body'))

import torch, cv2, numpy as np
from flask import Flask, request, jsonify

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

# ============ MHR70 关键点映射 ============
KP = {  # keypoint_name → index
    'pelvis':0, 'l_hip':1, 'r_hip':2, 'l_knee':4, 'r_knee':5,
    'l_ankle':7, 'r_ankle':8, 'neck':12, 'l_shoulder':16, 'r_shoulder':17,
    'l_elbow':18, 'r_elbow':19, 'l_wrist':20, 'r_wrist':21,
    'r_mid':29,  # right middle finger base
}


def angle_3d(a, b, c):
    ba, bc = a - b, c - b
    m = np.linalg.norm(ba) * np.linalg.norm(bc)
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / max(m, 1e-8), -1, 1)))) if m > 1e-8 else None


def analyze(k3d):
    """3D 羽毛球动作分析"""
    a = {}
    # 右臂
    a['r_elbow'] = angle_3d(k3d[KP['r_shoulder']], k3d[KP['r_elbow']], k3d[KP['r_wrist']])
    a['r_shoulder_up'] = angle_3d(k3d[KP['r_hip']], k3d[KP['r_shoulder']], k3d[KP['r_elbow']])
    a['r_wrist'] = angle_3d(k3d[KP['r_elbow']], k3d[KP['r_wrist']], k3d[KP['r_mid']])
    # 左臂
    a['l_elbow'] = angle_3d(k3d[KP['l_shoulder']], k3d[KP['l_elbow']], k3d[KP['l_wrist']])
    a['l_shoulder_up'] = angle_3d(k3d[KP['l_hip']], k3d[KP['l_shoulder']], k3d[KP['l_elbow']])
    # 下肢
    a['r_knee'] = angle_3d(k3d[KP['r_hip']], k3d[KP['r_knee']], k3d[KP['r_ankle']])
    a['l_knee'] = angle_3d(k3d[KP['l_hip']], k3d[KP['l_knee']], k3d[KP['l_ankle']])
    # 肩髋扭转
    sv = k3d[KP['r_shoulder']] - k3d[KP['l_shoulder']]
    hv = k3d[KP['r_hip']] - k3d[KP['l_hip']]
    sh, hh = np.array([sv[0], sv[2]]), np.array([hv[0], hv[2]])
    d, m = np.dot(sh, hh), np.linalg.norm(sh) * np.linalg.norm(hh)
    a['torso_twist'] = float(np.degrees(np.arccos(np.clip(d / max(m, 1e-8), -1, 1))))
    # 身体倾斜
    sm = (k3d[KP['r_shoulder']] + k3d[KP['l_shoulder']]) / 2
    hm = (k3d[KP['r_hip']] + k3d[KP['l_hip']]) / 2
    tv = sm - hm
    d2, m2 = np.dot(tv, np.array([0, 1, 0])), np.linalg.norm(tv)
    a['body_tilt'] = float(np.degrees(np.arccos(np.clip(d2 / max(m2, 1e-8), -1, 1))))
    a['is_overhead'] = float(k3d[KP['r_wrist']][1] > k3d[KP['r_shoulder']][1])
    return {k: round(v, 1) if v is not None else None for k, v in a.items()}


def feedback(a):
    """诊断"""
    fb, s = [], 100
    if (v := a.get('r_elbow')) and v < 120:
        fb.append({'l':'bad','m':f'右肘弯曲严重({v}°)，力量传导效率低'}); s -= 25
    elif v and v < 150:
        fb.append({'l':'warn','m':f'右肘未完全伸展({v}°)，建议锁肘'}); s -= 10
    elif v:
        fb.append({'l':'good','m':f'右肘伸展良好({v}°)'})

    if (v := a.get('torso_twist')) and v < 15:
        fb.append({'l':'bad','m':f'转体不足({v}°)，缺乏核心发力'}); s -= 15
    elif v and v < 30:
        fb.append({'l':'warn','m':f'转体偏小({v}°)，加大侧身'}); s -= 5
    elif v:
        fb.append({'l':'good','m':f'转体充分({v}°)'})

    if (v := a.get('l_shoulder_up')) and v < 60:
        fb.append({'l':'bad','m':'非持拍手未抬起'}); s -= 10
    elif v:
        fb.append({'l':'good','m':'非持拍手抬起，姿势标准'})

    if (v := a.get('r_knee')) and v > 150:
        fb.append({'l':'warn','m':f'膝关节偏直({v}°)，缺下肢蓄力'}); s -= 5

    if a.get('is_overhead'):
        fb.append({'l':'good','m':'手腕高于肩，过顶击球'})

    g = 'great' if s >= 80 else 'ok' if s >= 55 else 'poor'
    return {'score': max(0, min(100, s)), 'grade': g, 'feedback': fb}


@app.route('/analyze', methods=['POST'])
def analyze_route():
    try:
        d = request.get_json()
        if not d or 'image' not in d:
            return jsonify({'error': 'No image'}), 400
        b64 = d['image']
        if ',' in b64:
            b64 = b64.split(',', 1)[1]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Bad image'}), 400

        t0 = time.time()
        out = estimator.process_one_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        elapsed = time.time() - t0

        r = out[0] if isinstance(out, list) else out
        k3d = r['pred_keypoints_3d'].cpu().numpy() if hasattr(r['pred_keypoints_3d'], 'cpu') else r['pred_keypoints_3d']

        ang = analyze(k3d)
        fb = feedback(ang)

        return jsonify({
            'status': 'ok', 'elapsed': round(elapsed, 1),
            'angles': ang, 'score': fb['score'],
            'grade': fb['grade'], 'feedback': fb['feedback']
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print("\n🏸 SAM 3D Body 羽毛球分析服务器")
    print("POST http://localhost:8765/analyze\n")
    app.run(host='0.0.0.0', port=8765, debug=False)
