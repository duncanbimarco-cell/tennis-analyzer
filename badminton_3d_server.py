#!/usr/bin/env python3
"""
羽毛球 3D 动作分析服务器
使用 SAM 3D Body 对击球关键帧进行精确 3D 人体网格重建

启动方式:
    python badminton_3d_server.py

API:
    POST /analyze  - 上传击球关键帧图片，返回 3D 分析结果
    GET  /health    - 健康检查
"""

import os
import sys
import json
import base64
import io
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

# 添加 SAM 3D Body 到 Python path
SAM3D_PATH = Path(__file__).parent / "sam-3d-body"
if SAM3D_PATH.exists():
    sys.path.insert(0, str(SAM3D_PATH))

CHECKPOINT_DIR = SAM3D_PATH / "checkpoints" / "sam-3d-body-dinov3"

from flask import Flask, request, jsonify

app = Flask(__name__)

# 全局模型实例（懒加载）
estimator = None
device = None
MODEL_LOADED = False


def get_device():
    """获取最佳可用设备：MPS > CUDA > CPU"""
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model():
    """加载 SAM 3D Body 模型（首次调用时加载）"""
    global estimator, device, MODEL_LOADED

    if MODEL_LOADED:
        return True

    try:
        from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

        device = get_device()
        print(f"[SAM3D] 使用设备: {device}")

        checkpoint_path = CHECKPOINT_DIR / "model.ckpt"
        mhr_path = CHECKPOINT_DIR / "assets" / "mhr_model.pt"

        if not checkpoint_path.exists():
            print(f"[SAM3D] ⚠️ 模型文件不存在: {checkpoint_path}")
            print(f"[SAM3D] 请先下载模型: hf download facebook/sam-3d-body-dinov3 --local-dir {CHECKPOINT_DIR}")
            return False

        print(f"[SAM3D] 加载模型 checkpoint: {checkpoint_path}")
        model, model_cfg = load_sam_3d_body(
            str(checkpoint_path),
            device=device,
            mhr_path=str(mhr_path),
        )

        # 创建 estimator（不包含 detector/segmentor，用 MediaPipe 替代）
        estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
            human_detector=None,   # 用前端 MediaPipe 的 bbox 替代
            human_segmentor=None,
            fov_estimator=None,
        )

        MODEL_LOADED = True
        print("[SAM3D] ✅ 模型加载成功")
        return True

    except Exception as e:
        print(f"[SAM3D] ❌ 模型加载失败: {e}")
        traceback.print_exc()
        return False


def extract_key_angles(outputs):
    """
    从 SAM 3D Body 输出中提取羽毛球关键角度。

    SAM 3D Body 使用 MHR (Momentum Human Rig) 骨骼，关键关节索引：
    - 右肩: 2, 右肘: 3, 右腕: 4
    - 左肩: 5, 左肘: 6, 左腕: 7
    - 右髋: 9, 右膝: 10, 右踝: 11
    - 左髋: 12, 左膝: 13, 左踝: 14
    （具体索引取决于 MHR 骨骼定义，此处为示例）
    """
    try:
        # 获取 3D 关节位置
        if hasattr(outputs, 'pred_vertices'):
            joints_3d = outputs.pred_joints_3d  # (N_joints, 3)
        elif isinstance(outputs, dict) and 'joints_3d' in outputs:
            joints_3d = outputs['joints_3d']
        elif isinstance(outputs, dict) and 'pred_keypoints_3d' in outputs:
            joints_3d = outputs['pred_keypoints_3d']
        else:
            # 尝试直接获取
            joints_3d = None

        if joints_3d is None:
            return {"error": "无法提取 3D 关节", "available_keys": str(dir(outputs))}

        joints = joints_3d[0] if len(joints_3d.shape) == 3 else joints_3d  # (N, 3)

        def angle_3d(a, b, c):
            """计算 3D 空间中三点夹角"""
            ba = a - b
            bc = c - b
            cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
            return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

        # MHR 骨骼常用索引（需根据实际模型确认）
        # 这里使用常见的 SMPL/Human36M 映射
        angles = {}

        # 持拍手（右）：肩→肘→腕
        if joints.shape[0] > 4:
            angles["r_elbow_3d"] = round(angle_3d(joints[2], joints[3], joints[4]), 1)
        # 非持拍手（左）
        if joints.shape[0] > 7:
            angles["l_elbow_3d"] = round(angle_3d(joints[5], joints[6], joints[7]), 1)
        # 右膝：髋→膝→踝
        if joints.shape[0] > 11:
            angles["r_knee_3d"] = round(angle_3d(joints[9], joints[10], joints[11]), 1)
        # 左膝
        if joints.shape[0] > 14:
            angles["l_knee_3d"] = round(angle_3d(joints[12], joints[13], joints[14]), 1)

        # 肩髋扭转：双肩中点 vs 双髋中点
        if joints.shape[0] > 14:
            sh_mid = (joints[2] + joints[5]) / 2
            hip_mid = (joints[9] + joints[12]) / 2
            # 水平面投影的扭转角
            sh_vec = joints[2] - joints[5]
            hip_vec = joints[9] - joints[12]
            sh_vec_h = np.array([sh_vec[0], 0, sh_vec[2]])
            hip_vec_h = np.array([hip_vec[0], 0, hip_vec[2]])
            cos_t = np.dot(sh_vec_h, hip_vec_h) / (np.linalg.norm(sh_vec_h) * np.linalg.norm(hip_vec_h) + 1e-8)
            angles["torsion_3d"] = round(float(np.degrees(np.arccos(np.clip(cos_t, -1, 1)))), 1)

            # 身体倾斜
            torso = sh_mid - hip_mid
            vertical = np.array([0, 1, 0])
            cos_tilt = np.dot(torso, vertical) / (np.linalg.norm(torso) + 1e-8)
            angles["tilt_3d"] = round(float(np.degrees(np.arccos(np.clip(cos_tilt, -1, 1)))), 1)

        # 手腕高度（用于判断过顶/低手）
        if joints.shape[0] > 4:
            angles["r_wrist_height"] = round(float(joints[4][1]), 3)  # Y坐标
            angles["r_shoulder_height"] = round(float(joints[2][1]), 3)

        return angles

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


# ==================== Flask API ====================

@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "device": str(device) if device else "not initialized",
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    分析击球关键帧

    请求体 JSON:
    {
        "images": ["base64_image_1", "base64_image_2", ...],
        "bboxes": [[x1,y1,x2,y2], ...],  // 可选，人体边界框
        "return_mesh": false  // 是否返回完整网格（数据量大）
    }

    返回:
    {
        "success": true,
        "results": [
            {
                "angles": {  // 3D 角度
                    "r_elbow_3d": 165.3,
                    "l_elbow_3d": 120.5,
                    "torsion_3d": 35.2,
                    ...
                },
                "joints_3d": [...],  // 3D 关节坐标（可选）
                "inference_time_ms": 123
            },
            ...
        ]
    }
    """
    if not MODEL_LOADED:
        ok = load_model()
        if not ok:
            return jsonify({
                "success": False,
                "error": "模型未加载。请先下载 model checkpoint 到 " + str(CHECKPOINT_DIR),
                "hint": "hf download facebook/sam-3d-body-dinov3 --local-dir " + str(CHECKPOINT_DIR),
            }), 503

    try:
        data = request.get_json(force=True)
        images_b64 = data.get("images", [])
        bboxes = data.get("bboxes", [])
        return_mesh = data.get("return_mesh", False)

        if not images_b64:
            return jsonify({"success": False, "error": "缺少 images 字段"}), 400

        results = []

        for i, img_b64 in enumerate(images_b64):
            # Base64 -> numpy image
            if img_b64.startswith("data:"):
                img_b64 = img_b64.split(",", 1)[1]
            img_bytes = base64.b64decode(img_b64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                results.append({"error": f"图片 {i} 解码失败"})
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 使用指定的 bbox 或默认全图
            bbox = None
            if i < len(bboxes) and bboxes[i]:
                bbox = bboxes[i]

            t0 = time.time()

            # SAM 3D Body 推理
            outputs = estimator.process_one_image(
                img_rgb,
                bbox_thr=0.8,
                use_mask=False,
            )

            inference_time = round((time.time() - t0) * 1000)

            # 提取角度
            angles = extract_key_angles(outputs)

            result = {
                "angles": angles,
                "inference_time_ms": inference_time,
            }

            # 可选：返回 3D 关节
            if return_mesh and hasattr(outputs, 'pred_joints_3d'):
                joints = outputs.pred_joints_3d
                if len(joints.shape) == 3:
                    joints = joints[0]
                result["joints_3d"] = joints.tolist()

            results.append(result)

        return jsonify({
            "success": True,
            "results": results,
            "device": str(device),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }), 500


def main():
    print("=" * 60)
    print("🏸 羽毛球 3D 动作分析服务器 (SAM 3D Body)")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"模型路径: {CHECKPOINT_DIR}")

    # 检查模型文件
    ckpt = CHECKPOINT_DIR / "model.ckpt"
    if ckpt.exists():
        print(f"✅ 模型文件存在: {ckpt}")
        print("正在预加载模型...")
        load_model()
    else:
        print(f"⚠️ 模型文件不存在: {ckpt}")
        print(f"请运行以下命令下载模型:")
        print(f"  hf download facebook/sam-3d-body-dinov3 --local-dir {CHECKPOINT_DIR}")
        print(f"服务器将以无模型模式启动（/analyze 不可用，/health 可用）")

    print(f"\n📡 服务器启动: http://localhost:8765")
    print(f"   GET  /health   - 健康检查")
    print(f"   POST /analyze  - 3D 分析击球帧")

    app.run(host="0.0.0.0", port=8765, debug=False)


if __name__ == "__main__":
    main()
