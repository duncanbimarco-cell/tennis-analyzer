"""
上旋评分模块
根据挥拍数据多个维度综合评估上旋动作质量。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple

from data_processor import ProcessedSwing, compute_gyro_upward_score
from impact_analyzer import compute_impact_score


# ==================== 评分权重 ====================
# 各项满分 100，按权重加权
WEIGHTS = {
    "swing_angle": 0.35,             # 挥拍仰角 (30°~50°)
    "peak_speed": 0.23,              # 拍头峰值速度
    "trajectory_smoothness": 0.18,   # 轨迹平滑度
    "gyro_upward": 0.14,             # 旋转"刷球"强度 (陀螺仪)
    "impact_quality": 0.10,          # 击球震动质量
}


@dataclass
class TopspinScore:
    """上旋评分结果"""
    total: float = 0.0              # 综合分数 0~100
    details: Dict[str, float] = field(default_factory=dict)
    grade: str = "D"                # S/A/B/C/D
    feedback: str = ""              # 评语反馈

    @property
    def grade_emoji(self) -> str:
        return {"S": "[S] 完美", "A": "[A] 优秀", "B": "[B] 良好", "C": "[C] 一般", "D": "[D] 需改进"}[self.grade]


class TopspinScorer:
    """上旋动作评分器"""

    # 理想参数范围
    IDEAL_ANGLE_MIN, IDEAL_ANGLE_MAX = 30.0, 50.0    # 仰角 (度)
    ANGLE_TOLERANCE = 15.0   # 可接受额外范围
    IDEAL_SPEED_MIN = 20.0   # 最低理想速度 (km/h)，约 5.5 m/s
    IDEAL_SPEED_MAX = 80.0   # 最高理想速度

    def score(self, swing: ProcessedSwing) -> TopspinScore:
        """对一次挥拍进行综合评分"""
        scores = {}
        feedback_parts = []

        # 1. 挥拍仰角评分 (40%)
        scores["swing_angle"] = self._score_angle(swing.swing_angle_deg)
        if swing.swing_angle_deg < self.IDEAL_ANGLE_MIN:
            feedback_parts.append(f"仰角偏低 ({swing.swing_angle_deg:.0f}°)，挥拍轨迹太平，缺少从下往上的发力")
        elif swing.swing_angle_deg > self.IDEAL_ANGLE_MAX + self.ANGLE_TOLERANCE:
            feedback_parts.append(f"仰角偏高 ({swing.swing_angle_deg:.0f}°)，注意不要过度向上")
        else:
            feedback_parts.append(f"仰角优秀 ({swing.swing_angle_deg:.0f}°)")

        # 2. 拍头速度评分 (25%)
        speed_kmh = swing.peak_speed * 3.6
        scores["peak_speed"] = self._score_speed(speed_kmh)
        if speed_kmh < self.IDEAL_SPEED_MIN:
            feedback_parts.append(f"拍速偏低 ({speed_kmh:.0f} km/h)，需要加快挥拍")
        else:
            feedback_parts.append(f"拍速达标 ({speed_kmh:.0f} km/h)")

        # 3. 轨迹平滑度评分 (20%)
        scores["trajectory_smoothness"] = self._score_smoothness(swing)
        smoothness = scores["trajectory_smoothness"]
        if smoothness < 60:
            feedback_parts.append("轨迹不够平滑，注意挥拍连贯性")
        else:
            feedback_parts.append("挥拍轨迹流畅")

        # 4. 旋转"刷球"强度评分 (14%)
        scores["gyro_upward"] = compute_gyro_upward_score(swing)
        if scores["gyro_upward"] < 50:
            feedback_parts.append("拍头旋转不足，注意击球时手腕向上翻拍")
        else:
            feedback_parts.append("拍头旋转充分，刷球效果好")

        # 5. 击球震动质量评分 (10%)
        impact_result = swing  # ProcessedSwing contains all impact fields
        scores["impact_quality"] = compute_impact_score_from_swing(swing)
        imp = scores["impact_quality"]
        if not swing.impact_detected:
            feedback_parts.append("未检测到清晰击球点")
        elif imp >= 70:
            feedback_parts.append(f"击球质量优秀 (主频{swing.vibration_dominant_freq:.0f}Hz)")
        elif imp >= 40:
            feedback_parts.append(f"击球质量一般 (主频{swing.vibration_dominant_freq:.0f}Hz)")
        else:
            feedback_parts.append("击球点或震动异常，检查是否击中甜区")

        # 加权总评
        total = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)

        # 等级
        if total >= 90:
            grade = "S"
        elif total >= 75:
            grade = "A"
        elif total >= 60:
            grade = "B"
        elif total >= 40:
            grade = "C"
        else:
            grade = "D"

        return TopspinScore(
            total=round(total, 1),
            details={k: round(v, 1) for k, v in scores.items()},
            grade=grade,
            feedback=" | ".join(feedback_parts),
        )

    def _score_angle(self, angle: float) -> float:
        """仰角评分: 在理想范围内满分，偏离则递减"""
        if self.IDEAL_ANGLE_MIN <= angle <= self.IDEAL_ANGLE_MAX:
            return 100.0
        elif angle < self.IDEAL_ANGLE_MIN:
            dist = self.IDEAL_ANGLE_MIN - angle
            return max(0, 100 - dist * 4)  # 每偏离 1° 扣 4 分
        else:
            dist = angle - self.IDEAL_ANGLE_MAX
            return max(0, 100 - dist * 2)  # 偏高扣分稍轻

    def _score_speed(self, speed_kmh: float) -> float:
        """速度评分"""
        if speed_kmh < 5:
            return 0
        elif speed_kmh < self.IDEAL_SPEED_MIN:
            return speed_kmh / self.IDEAL_SPEED_MIN * 100
        elif speed_kmh < self.IDEAL_SPEED_MAX:
            return 100
        else:
            return 100  # 不因太快扣分

    def _score_smoothness(self, swing: ProcessedSwing) -> float:
        """轨迹平滑度: 基于轨迹的 jerk (加加速度) 分析"""
        if len(swing.trajectory) < 10:
            return 50.0

        # 计算加速度的变化率 (jerk)
        accel = np.diff(swing.trajectory, axis=0)
        if len(accel) < 2:
            return 50.0
        jerk = np.diff(accel, axis=0)

        jerk_mag = np.sqrt(np.sum(jerk ** 2, axis=1))
        jerk_std = float(np.std(jerk_mag) + 1e-9)

        # 标准差越小越平滑，映射到 0~100
        score = max(0, 100 - jerk_std * 2)
        return min(100, score)


def compute_impact_score_from_swing(swing: ProcessedSwing) -> float:
    """从 ProcessedSwing 提取击球信息并评分"""
    from impact_analyzer import ImpactResult, compute_impact_score
    result = ImpactResult(
        detected=swing.impact_detected,
        impact_peak_g=swing.impact_peak_g,
        vibration_dominant_freq=swing.vibration_dominant_freq,
        decay_rate=swing.vibration_decay_rate,
    )
    return compute_impact_score(result)
