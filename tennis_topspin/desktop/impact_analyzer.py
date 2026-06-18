"""
击球震动分析模块
检测击球瞬间的震动特征，分析频谱和衰减，估算拍线磅数并给出推荐。

物理原理:
  - 弦床可近似为简谐振子，震动主频 f ∝ √(T/m)
    T = 弦张力（磅数），m = 弦质量
  - 基准: 50 lbs ≈ 370 Hz (聚酯弦，16L 线径)
  - 每 ±1 lb 约改变 6-8 Hz
  - 衰减率反映弦床能量耗散，与磅数和击球点位置相关

专利核心:
  单 IMU 传感器同时测量挥拍质量 + 弦床张力 + 个性化磅数推荐
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

GRAVITY = 9.81


@dataclass
class ImpactResult:
    """击球震动分析结果"""
    detected: bool = False
    impact_idx: int = 0                # 击球点在样本中的索引
    impact_time_ms: float = 0.0        # 击球时间 (ms)
    impact_peak_g: float = 0.0         # 击球加速度峰值 (g)
    vibration_dominant_freq: float = 0.0    # 震动主频 (Hz)
    vibration_secondary_freq: float = 0.0  # 次频 (Hz)
    decay_rate: float = 0.0            # 衰减系数 (越大衰减越快)
    spectral_centroid: float = 0.0     # 频谱质心 (Hz)
    estimated_tension_lbs: float = 0.0 # 估算当前磅数
    recommended_tension_min: float = 0.0  # 推荐磅数下限
    recommended_tension_max: float = 0.0  # 推荐磅数上限
    recommendation_reason: str = ""    # 推荐理由


class ImpactAnalyzer:
    """
    击球震动分析器

    用法:
        analyzer = ImpactAnalyzer()
        result = analyzer.analyze(accel_data, timestamps, swing_angle, peak_speed)
    """

    # 弦床物理模型参数
    BASE_FREQ_HZ = 370.0        # 50 lbs 基准主频 (聚酯弦)
    BASE_TENSION_LBS = 50.0     # 基准磅数
    FREQ_PER_LB = 7.0           # 每磅改变频率 (Hz)
    MIN_FREQ_HZ = 200.0         # 有效震动频率下限
    MAX_FREQ_HZ = 600.0         # 有效震动频率上限
    TENSION_MIN = 30.0          # 最低合理磅数
    TENSION_MAX = 70.0          # 最高合理磅数

    # 击球检测参数
    JERK_THRESHOLD = 500.0       # jerk 阈值 (m/s³)，检测击球尖峰
    VIBRATION_WINDOW_MS = 120.0  # 击球后震动分析窗口
    HIGH_PASS_CUTOFF = 20.0      # 高通滤波截止频率 (Hz)，去除挥拍运动

    def analyze(self, accel: np.ndarray, timestamps: np.ndarray,
                swing_angle_deg: float = 0.0,
                peak_speed_ms: float = 0.0) -> ImpactResult:
        """
        分析一次挥拍中的击球震动。

        Args:
            accel: (N, 3) 原始加速度数据 [ax, ay, az]
            timestamps: (N,) 时间序列 (ms)
            swing_angle_deg: 挥拍仰角
            peak_speed_ms: 挥拍峰值速度 (m/s)

        Returns:
            ImpactResult
        """
        n = len(accel)
        if n < 30:
            return ImpactResult()

        # 1. 计算加速度幅值
        accel_mag = np.sqrt(np.sum(accel ** 2, axis=1))

        # 2. 检测击球点 (用 jerk = 加速度的导数)
        sample_period_ms = np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 5.0
        jerk = np.abs(np.diff(accel_mag)) / (sample_period_ms / 1000.0)

        # 找 jerk 峰值 (在挥拍后半段，对应球拍触球瞬间)
        search_start = int(n * 0.3)  # 跳过开头静止段
        search_end = int(n * 0.9)    # 跳过末尾
        if search_end <= search_start:
            search_start, search_end = 0, n

        jerk_region = jerk[search_start:search_end]
        if len(jerk_region) < 5:
            return ImpactResult()

        impact_local_idx = int(np.argmax(jerk_region))
        impact_idx = search_start + impact_local_idx
        impact_time_ms = float(timestamps[impact_idx])

        # 判断是否是有效击球 (jerk 足够大)
        peak_jerk = float(jerk_region[impact_local_idx])
        if peak_jerk < self.JERK_THRESHOLD:
            return ImpactResult(detected=True, impact_idx=impact_idx,
                               impact_time_ms=impact_time_ms)

        # 击球加速度峰值 (g)
        impact_window = slice(max(0, impact_idx - 2), min(n, impact_idx + 5))
        impact_peak_g = float(np.max(accel_mag[impact_window]) / GRAVITY)

        # 3. 提取击球后震动信号（从 Y 轴提取，包含主要震动分量）
        vibration_ms = self.VIBRATION_WINDOW_MS
        vib_end_idx = min(n, int(impact_idx + vibration_ms / sample_period_ms))
        if vib_end_idx <= impact_idx + 5:
            return ImpactResult(detected=True, impact_idx=impact_idx,
                               impact_time_ms=impact_time_ms,
                               impact_peak_g=impact_peak_g)

        # 使用 Y 轴（前向）加速度，包含最清晰的击球震动
        vib_signal = accel[impact_idx:vib_end_idx, 1].copy()
        vib_time = timestamps[impact_idx:vib_end_idx] - impact_time_ms

        # 去除低频趋势 (多项式拟合去趋势 → 残留高频震动)
        vib_filtered = self._detrend_signal(vib_signal, vib_time)

        # 4. 频率分析 (零交叉法 + FFT 双重验证)
        sample_rate = 1000.0 / sample_period_ms
        zc_freq = self._zero_crossing_freq(vib_filtered, sample_rate)
        dom_freq, sec_freq, centroid = self._fft_analysis(vib_filtered, sample_rate)

        # 优先使用零交叉法（短窗口更可靠），FFT 作为补充
        if zc_freq > 0 and dom_freq > 0:
            # 两者接近则取平均，偏差大则信任零交叉
            if abs(zc_freq - dom_freq) < 15:
                dom_freq = round((zc_freq + dom_freq) / 2, 1)
            else:
                dom_freq = zc_freq
        elif zc_freq > 0:
            dom_freq = zc_freq

        # 5. 衰减率分析
        decay = self._compute_decay(vib_filtered, vib_time)

        # 6. 估算磅数
        # 弦床频率 ∝ √(张力), 基准 50lbs → 参考频率取决于采样率
        if dom_freq > 10:
            if sample_rate >= 500:
                # 高频采样: 真实弦床频率 300-500Hz
                ref_freq = 370.0  # 50lbs 对应 ~370Hz
                freq_per_lb = 7.0
            else:
                # 200Hz 采样: Nyquist=100Hz, 频率被压缩
                # 50lbs 对应约 60Hz, 每磅约 1.2Hz
                ref_freq = 60.0
                freq_per_lb = 1.4

            estimated_tension = self.BASE_TENSION_LBS + (dom_freq - ref_freq) / freq_per_lb
            estimated_tension = round(np.clip(estimated_tension, self.TENSION_MIN, self.TENSION_MAX), 1)
        else:
            estimated_tension = 0.0

        # 7. 推荐磅数
        rec_min, rec_max, reason = self._recommend(
            dom_freq, decay, swing_angle_deg, peak_speed_ms, estimated_tension
        )

        return ImpactResult(
            detected=True,
            impact_idx=impact_idx,
            impact_time_ms=impact_time_ms,
            impact_peak_g=impact_peak_g,
            vibration_dominant_freq=dom_freq,
            vibration_secondary_freq=sec_freq,
            decay_rate=decay,
            spectral_centroid=centroid,
            estimated_tension_lbs=estimated_tension,
            recommended_tension_min=rec_min,
            recommended_tension_max=rec_max,
            recommendation_reason=reason,
        )

    def _detrend_signal(self, signal: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """多项式去趋势，提取高频震动残差"""
        n = len(signal)
        if n < 5:
            return signal - np.mean(signal)
        # 用 3 阶多项式拟合低频趋势
        t_norm = (timestamps - timestamps[0]) / max(timestamps[-1] - timestamps[0], 1)
        coeffs = np.polyfit(t_norm, signal, 3)
        trend = np.polyval(coeffs, t_norm)
        return signal - trend

    def _zero_crossing_freq(self, signal: np.ndarray, sample_rate: float) -> float:
        """用零交叉法估算信号主频，短窗口下比 FFT 更可靠"""
        n = len(signal)
        if n < 6:
            return 0.0

        # 确保信号在零附近（去除 DC 偏置）
        s = signal - np.mean(signal)

        # 只分析非零段（信号衰减到太小则停止）
        threshold = np.max(np.abs(s)) * 0.1
        crossings = 0
        valid_samples = 0
        for i in range(1, n):
            if max(abs(s[i]), abs(s[i - 1])) < threshold and i > n // 3:
                break  # 信号衰减殆尽
            if s[i] * s[i - 1] < 0:
                crossings += 1
            valid_samples += 1

        if crossings < 3 or valid_samples < 3:
            return 0.0

        # 频率 = 穿越次数 / 2 / 时间
        duration_sec = valid_samples / sample_rate
        if duration_sec <= 0:
            return 0.0
        freq = (crossings / 2) / duration_sec
        return round(float(freq), 1)

    def _highpass_filter(self, signal: np.ndarray, sample_period_ms: float) -> np.ndarray:
        """一阶高通滤波，去除低频挥拍运动"""
        if len(signal) < 2:
            return signal
        sample_rate = 1000.0 / sample_period_ms
        dt = sample_period_ms / 1000.0
        rc = 1.0 / (2 * np.pi * self.HIGH_PASS_CUTOFF)
        alpha = rc / (rc + dt)

        filtered = np.zeros_like(signal)
        filtered[0] = signal[0]
        for i in range(1, len(signal)):
            filtered[i] = alpha * (filtered[i - 1] + signal[i] - signal[i - 1])
        return filtered

    def _fft_analysis(self, signal: np.ndarray, sample_rate: float
                      ) -> Tuple[float, float, float]:
        """FFT 频谱分析，返回 (主频, 次频, 频谱质心)"""
        n = len(signal)
        if n < 8:
            return 0.0, 0.0, 0.0

        # 加窗
        window = np.hanning(n)
        signal_win = signal * window

        # FFT
        fft = np.abs(np.fft.rfft(signal_win))
        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

        # 频率范围: 20Hz 到 Nyquist*0.9 (避免混叠)
        nyquist = sample_rate / 2
        f_min = max(15, freqs[1])  # 跳过 DC
        f_max = nyquist * 0.85

        valid = (freqs >= f_min) & (freqs <= f_max)
        if not np.any(valid):
            return 0.0, 0.0, 0.0

        fft_valid = fft[valid]
        freqs_valid = freqs[valid]

        # 平滑后找峰值
        if len(fft_valid) > 3:
            fft_smooth = np.convolve(fft_valid, np.ones(3) / 3, mode='same')
        else:
            fft_smooth = fft_valid

        # 主频
        peak_idx = int(np.argmax(fft_smooth))
        dom_freq = round(float(freqs_valid[peak_idx]), 1)

        # 次频 (排除主频附近)
        if len(freqs_valid) > 2:
            df = freqs_valid[1] - freqs_valid[0]
        else:
            df = 5.0
        exclude_half = max(1, int(15 / df))
        mask = np.ones_like(fft_smooth, dtype=bool)
        lo = max(0, peak_idx - exclude_half)
        hi = min(len(mask), peak_idx + exclude_half + 1)
        mask[lo:hi] = False
        if np.any(mask) and np.max(fft_smooth[mask]) > 0:
            sec_idx = int(np.argmax(fft_smooth * mask.astype(float)))
            sec_freq = round(float(freqs_valid[sec_idx]), 1)
        else:
            sec_freq = 0.0

        # 频谱质心
        total = np.sum(fft_valid) + 1e-10
        centroid = round(float(np.sum(freqs_valid * fft_valid) / total), 1)

        return dom_freq, sec_freq, centroid

    def _compute_decay(self, signal: np.ndarray, timestamps: np.ndarray) -> float:
        """计算震动衰减率 (指数拟合)"""
        n = len(signal)
        if n < 10:
            return 0.0

        # 取信号包络 (Hilbert 包络或用绝对值+低通代替)
        envelope = np.abs(signal)
        # 简单低通平滑包络
        alpha = 0.15
        env_smooth = np.zeros_like(envelope)
        env_smooth[0] = envelope[0]
        for i in range(1, n):
            env_smooth[i] = alpha * envelope[i] + (1 - alpha) * env_smooth[i - 1]

        # 找包络峰值位置
        peak_idx = int(np.argmax(env_smooth))

        # 取峰值后的衰减段
        if peak_idx >= n - 5:
            return 0.0

        decay_segment = env_smooth[peak_idx:]
        t_decay = timestamps[peak_idx:] - timestamps[peak_idx]
        t_decay = t_decay / 1000.0  # 转秒

        # 对数拟合: log(env) = log(A) - lambda * t
        positive = decay_segment > np.max(decay_segment) * 0.05
        if np.sum(positive) < 5:
            return 0.0

        y = np.log(np.maximum(decay_segment[positive], 1e-9))
        t_fit = t_decay[positive]

        if len(t_fit) > 1:
            A = np.vstack([t_fit, np.ones_like(t_fit)]).T
            slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
            decay_rate = round(float(-slope), 3)  # 正值表示衰减
        else:
            decay_rate = 0.0

        return decay_rate

    def _recommend(self, dom_freq: float, decay: float,
                   swing_angle: float, peak_speed: float,
                   estimated_tension: float) -> Tuple[float, float, str]:
        """综合推荐磅数"""
        speed_kmh = peak_speed * 3.6
        reasons = []

        # 基准: 当前估算磅数
        if estimated_tension <= 0:
            estimated_tension = self.BASE_TENSION_LBS
        base = estimated_tension
        adjustment = 0.0

        # 规则 1: 挥拍速度快 → 升磅增加控制
        if speed_kmh > 40:
            adjustment += 3
            reasons.append(f"挥拍速度快({speed_kmh:.0f}km/h)")
        elif speed_kmh < 20:
            adjustment -= 2
            reasons.append(f"挥拍速度偏慢({speed_kmh:.0f}km/h)")

        # 规则 2: 挥拍仰角低 → 升磅补偿控制
        if 0 < swing_angle < 25:
            adjustment += 2
            reasons.append("挥拍偏平")

        # 规则 3: 震动衰减过快 → 磅数偏高，适当降
        if decay > 0.8:
            adjustment -= 3
            reasons.append("震动衰减过快(磅数可能偏高)")
        elif 0 < decay < 0.3:
            adjustment += 3
            reasons.append("震动衰减过慢(磅数可能偏低)")

        # 规则 4: 主频判断 (频率偏离基准表示磅数异常)
        if dom_freq > 85:
            adjustment -= 2
            reasons.append(f"弦床偏硬({dom_freq:.0f}Hz)")
        elif 0 < dom_freq < 45:
            adjustment += 2
            reasons.append(f"弦床偏软({dom_freq:.0f}Hz)")

        # 计算推荐范围
        rec_center = base + adjustment
        rec_center = np.clip(rec_center, self.TENSION_MIN + 2, self.TENSION_MAX - 2)
        rec_min = round(max(self.TENSION_MIN, rec_center - 2), 1)
        rec_max = round(min(self.TENSION_MAX, rec_center + 2), 1)

        if not reasons:
            reasons.append("当前磅数适合你的挥拍风格")

        return rec_min, rec_max, "; ".join(reasons)


def compute_impact_score(result: ImpactResult) -> float:
    """
    击球质量评分 0~100。
    评估击球点位置（甜区判断）、震动特征正常性。
    """
    if not result.detected:
        return 50.0  # 未检测到击球，给中性分数

    score = 0.0
    weight_total = 0.0

    # 1. 击球力度适中 (20-80g 为正常范围)
    g = result.impact_peak_g
    if 20 <= g <= 80:
        score += 30
    elif 10 <= g <= 100:
        score += 15
    else:
        score += 0
    weight_total += 30

    # 2. 震动主频在合理范围 (根据是否检测到来判断)
    f = result.vibration_dominant_freq
    if f > 15:
        # 有有效震动频率
        if 35 <= f <= 90:
            score += 35  # 在合理震动范围
        elif 20 <= f <= 95:
            score += 18
        else:
            score += 8
    else:
        score += 0  # 无有效震动
    weight_total += 35

    # 3. 衰减率适中 (0.3-0.8)
    d = result.decay_rate
    if 0.3 <= d <= 0.8:
        score += 25
    elif 0.15 <= d <= 1.2:
        score += 12
    else:
        score += 3
    weight_total += 25

    # 4. 击球时间点在合理位置（挥拍后半段，不是过早或过晚）
    # 这里用是否有有效检测来判断
    score += 10
    weight_total += 10

    return round(min(100, score / weight_total * 100), 1)
