"""
数据处理模块
对原始 IMU 数据进行滤波、姿态估计和特征提取。

特征提取策略（针对网球挥拍场景）:
- 挥拍仰角: 从加速度峰值附近的方向向量估算
- 上旋量: 陀螺仪 Z 轴积分
- 拍头速度: 加速度幅值积分
- 轨迹: 简化重建用于可视化
"""

import numpy as np
from dataclasses import dataclass, field

from ble_receiver import SensorSample, SwingRecord
from impact_analyzer import ImpactAnalyzer

GRAVITY = 9.81


@dataclass
class ProcessedSwing:
    """处理后的挥拍数据"""
    raw: SwingRecord
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    accel_magnitudes: np.ndarray = field(default_factory=lambda: np.array([]))
    gyro_magnitudes: np.ndarray = field(default_factory=lambda: np.array([]))
    world_accel: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    velocity: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    trajectory: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    # 挥拍特征
    max_accel: float = 0.0
    max_gyro: float = 0.0
    peak_speed: float = 0.0
    swing_angle_deg: float = 0.0       # 挥拍仰角 (度)
    trajectory_length: float = 0.0
    gyro_upward_integral: float = 0.0  # 陀螺仪Z轴上旋积分 (度)
    swing_plane: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))
    # 击球震动特征
    impact_detected: bool = False
    impact_idx: int = 0
    impact_peak_g: float = 0.0
    vibration_dominant_freq: float = 0.0
    vibration_decay_rate: float = 0.0
    estimated_tension_lbs: float = 0.0
    recommended_tension_min: float = 0.0
    recommended_tension_max: float = 0.0


class SwingProcessor:
    """挥拍数据处理器"""

    def __init__(self):
        pass

    def process(self, swing: SwingRecord) -> ProcessedSwing:
        """处理一次挥拍记录"""
        n = swing.sample_count
        if n < 10:
            raise ValueError(f"采样点太少 ({n})")

        timestamps = np.array([s.timestamp_us for s in swing.samples], dtype=np.float64)
        timestamps_ms = (timestamps - timestamps[0]) / 1000.0

        accel = np.array([[s.accel_x, s.accel_y, s.accel_z] for s in swing.samples])
        gyro = np.array([[s.gyro_x, s.gyro_y, s.gyro_z] for s in swing.samples])

        accel_mag = np.sqrt(np.sum(accel ** 2, axis=1))
        gyro_mag = np.sqrt(np.sum(gyro ** 2, axis=1))

        dt = np.diff(timestamps_ms) / 1000.0
        dt = np.clip(dt, 0.001, 0.05)
        dt_full = np.concatenate([dt, [dt[-1]]])

        # ===== 1. 挥拍仰角 =====
        # 方法: 找到加速度峰值前后的区间，用该区间加速度均值方向估算挥拍方向
        peak_idx = int(np.argmax(accel_mag))

        # 取峰值附近 ±15% 的窗口（排除开头静止段）
        window = max(5, int(n * 0.15))
        start_idx = max(0, peak_idx - window)
        end_idx = min(n, peak_idx + window)

        # 窗口内平均加速度（含重力），减去静止时重力估算
        # 静止加速度 ≈ [0, 0, -9.8] (Z向上为正)，假设传感器大致水平
        accel_window = accel[start_idx:end_idx]
        mean_accel = np.mean(accel_window, axis=0)

        # 去除重力分量 (假设重力投影在各轴)
        # 用开头几帧估计重力方向
        rest_frames = min(20, n // 4)
        gravity_est = np.mean(accel[:rest_frames], axis=0)
        # 归一化重力方向
        g_mag = np.linalg.norm(gravity_est)
        if g_mag > 1:
            gravity_est = gravity_est / g_mag * GRAVITY

        motion_accel = mean_accel - gravity_est

        # 仰角: 运动加速度向量与水平面夹角
        horizontal = np.sqrt(motion_accel[0]**2 + motion_accel[1]**2)
        vertical = motion_accel[2]
        swing_angle_deg = float(np.degrees(np.arctan2(vertical, max(horizontal, 0.01))))

        # ===== 2. 拍头速度 (加速度幅值积分，标量) =====
        # 去除重力后积分运动加速度幅值
        motion_mag = np.abs(accel_mag - GRAVITY)
        # 只在加速度高于阈值时积分（过滤噪声）
        threshold = 3.0
        active = motion_mag > threshold
        speed_scalar = np.zeros(n)
        for i in range(1, n):
            if active[i]:
                speed_scalar[i] = speed_scalar[i - 1] + motion_mag[i] * dt_full[i]
            else:
                speed_scalar[i] = max(0, speed_scalar[i - 1] * 0.95)  # 衰减

        peak_speed = float(np.max(speed_scalar))

        # ===== 3. 陀螺仪 Z 轴上旋积分 =====
        gyro_z = np.array([s.gyro_z for s in swing.samples])
        gyro_z_positive = np.maximum(gyro_z, 0)
        gyro_upward = float(np.trapz(gyro_z_positive, timestamps_ms / 1000.0))

        # ===== 4. 简化轨迹重建（用于 3D 可视化） =====
        # 使用运动加速度积分，叠加在估算的挥拍平面上
        forward_accel = motion_mag * np.cos(np.radians(swing_angle_deg))
        upward_accel = motion_mag * np.sin(np.radians(swing_angle_deg))

        # 前向位移 (Y轴)
        vel_fwd = np.zeros(n)
        pos_fwd = np.zeros(n)
        for i in range(1, n):
            if active[i]:
                vel_fwd[i] = vel_fwd[i - 1] + forward_accel[i] * dt_full[i]
            else:
                vel_fwd[i] = vel_fwd[i - 1] * 0.9
            pos_fwd[i] = pos_fwd[i - 1] + vel_fwd[i] * dt_full[i]

        # 向上位移 (Z轴)
        vel_up = np.zeros(n)
        pos_up = np.zeros(n)
        for i in range(1, n):
            if active[i]:
                vel_up[i] = vel_up[i - 1] + upward_accel[i] * dt_full[i]
            else:
                vel_up[i] = vel_up[i - 1] * 0.9
            pos_up[i] = pos_up[i - 1] + vel_up[i] * dt_full[i]

        # 侧向位移 (X轴，来自原始加速度X分量)
        vel_lat = np.zeros(n)
        pos_lat = np.zeros(n)
        for i in range(1, n):
            vel_lat[i] = vel_lat[i - 1] + accel[i, 0] * dt_full[i]
            pos_lat[i] = pos_lat[i - 1] + vel_lat[i] * dt_full[i]

        trajectory = np.column_stack([pos_lat, pos_fwd, pos_up])

        # 去趋势
        trajectory = self._detrend(trajectory, timestamps_ms)
        velocity = np.column_stack([vel_lat, vel_fwd, vel_up])
        world_accel = np.column_stack([accel[:, 0], forward_accel, upward_accel])

        # 轨迹长度
        segment_lengths = np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1))
        trajectory_length = float(np.sum(segment_lengths))

        # PCA 挥拍平面
        if n >= 3:
            centered = trajectory - np.mean(trajectory, axis=0)
            try:
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                swing_plane = vh[2] if vh.shape[0] >= 3 else np.array([0, 1, 0])
            except np.linalg.LinAlgError:
                swing_plane = np.array([0, 1, 0])
        else:
            swing_plane = np.array([0, 1, 0])

        # ===== 击球震动分析 =====
        analyzer = ImpactAnalyzer()
        impact = analyzer.analyze(accel, timestamps_ms, swing_angle_deg, peak_speed)

        return ProcessedSwing(
            raw=swing,
            timestamps=timestamps_ms,
            accel_magnitudes=accel_mag,
            gyro_magnitudes=gyro_mag,
            world_accel=world_accel,
            velocity=velocity,
            trajectory=trajectory,
            max_accel=float(np.max(accel_mag)),
            max_gyro=float(np.max(gyro_mag)),
            peak_speed=peak_speed,
            swing_angle_deg=swing_angle_deg,
            trajectory_length=trajectory_length,
            gyro_upward_integral=gyro_upward,
            swing_plane=swing_plane,
            impact_detected=impact.detected,
            impact_idx=impact.impact_idx,
            impact_peak_g=impact.impact_peak_g,
            vibration_dominant_freq=impact.vibration_dominant_freq,
            vibration_decay_rate=impact.decay_rate,
            estimated_tension_lbs=impact.estimated_tension_lbs,
            recommended_tension_min=impact.recommended_tension_min,
            recommended_tension_max=impact.recommended_tension_max,
        )

    def _detrend(self, data: np.ndarray, t: np.ndarray) -> np.ndarray:
        """去除线性趋势"""
        result = np.zeros_like(data)
        for axis in range(data.shape[1]):
            series = data[:, axis]
            if len(t) > 1:
                A = np.vstack([t, np.ones_like(t)]).T
                slope, intercept = np.linalg.lstsq(A, series, rcond=None)[0]
                result[:, axis] = series - (slope * t + intercept)
            else:
                result[:, axis] = series
        return result


def compute_gyro_upward_score(swing: ProcessedSwing) -> float:
    """从陀螺仪旋积分计算上旋分数 0~100"""
    gyro_total = swing.gyro_upward_integral
    # 100度总旋转 → 满分; 50度 → 75分; 20度 → 50分
    score = min(100, gyro_total / 100.0 * 100)
    return round(float(score), 1)
