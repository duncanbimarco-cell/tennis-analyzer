"""
3D 可视化模块
生成挥拍轨迹图、传感器曲线、震动分析、评分雷达图。
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from data_processor import ProcessedSwing
from scorer import TopspinScore


def plot_swing_report(swing: ProcessedSwing, score: TopspinScore, save_path: str = None):
    """生成挥拍综合报告图 (3×2 共 6 个子图)"""
    fig = plt.figure(figsize=(18, 14))

    # 标题：总分 + 磅数推荐
    title = f'网球上旋挥拍报告 — 综合评分: {score.total:.0f}/100 [{score.grade}]'
    if swing.impact_detected and swing.estimated_tension_lbs > 0:
        title += (f'  |  当前磅数: ~{swing.estimated_tension_lbs:.0f} lbs'
                  f'  推荐: {swing.recommended_tension_min:.0f}~{swing.recommended_tension_max:.0f} lbs')
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.98)

    # ===== 子图布局: 3行 × 2列 =====
    # 左列: 3D轨迹 / 传感器曲线 / 震动波形
    # 右列: 速度曲线 / 评分雷达 / FFT频谱

    # 子图 1: 3D 轨迹 (左上行)
    ax1 = fig.add_subplot(3, 2, 1, projection='3d')
    plot_3d_trajectory(ax1, swing)

    # 子图 2: 速度曲线 (右上)
    ax2 = fig.add_subplot(3, 2, 2)
    plot_speed_curve(ax2, swing)

    # 子图 3: 传感器曲线 (左中)
    ax3 = fig.add_subplot(3, 2, 3)
    plot_sensor_curves(ax3, swing)

    # 子图 4: 评分雷达 (右中)
    ax4 = fig.add_subplot(3, 2, 4)
    plot_score_breakdown(ax4, score)

    # 子图 5: 震动波形 (左下)
    ax5 = fig.add_subplot(3, 2, 5)
    plot_vibration_waveform(ax5, swing)

    # 子图 6: FFT 频谱 (右下)
    ax6 = fig.add_subplot(3, 2, 6)
    plot_fft_spectrum(ax6, swing)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"报告已保存: {save_path}")
    plt.close()


def plot_3d_trajectory(ax, swing: ProcessedSwing):
    """绘制 3D 挥拍轨迹"""
    traj = swing.trajectory
    points = traj - traj[0]
    t = swing.timestamps / max(swing.timestamps) if max(swing.timestamps) > 0 else np.zeros_like(swing.timestamps)

    for i in range(len(points) - 1):
        ax.plot3D(points[i:i + 2, 0], points[i:i + 2, 1], points[i:i + 2, 2],
                   color=plt.cm.viridis(t[i]), lw=2)

    ax.scatter(*points[0], c='green', s=60, marker='o', label='起点', zorder=5)
    ax.scatter(*points[-1], c='red', s=60, marker='*', label='终点', zorder=5)

    # 击球点标记
    if swing.impact_detected and 0 < swing.impact_idx < len(points):
        ax.scatter(*points[swing.impact_idx], c='yellow', s=100, marker='D',
                   edgecolors='orange', linewidths=2, label='击球点', zorder=5)
    else:
        impact_idx = int(len(points) * 0.7)
        ax.scatter(*points[impact_idx], c='yellow', s=80, marker='D',
                   edgecolors='orange', linewidths=1.5, label='假想击球点', zorder=5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'3D 挥拍轨迹 (仰角: {swing.swing_angle_deg:.1f}°)', fontsize=10)
    ax.legend(fontsize=6, loc='upper left')

    max_range = max(np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2]), 0.1)
    mid = np.mean(points, axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


def plot_sensor_curves(ax, swing: ProcessedSwing):
    """绘制加速度和角速度幅值曲线，标注击球点"""
    t = swing.timestamps
    color_accel = '#2196F3'
    color_gyro = '#FF5722'

    ax2_twin = ax.twinx()
    line1, = ax.plot(t, swing.accel_magnitudes, color=color_accel, lw=1.2, alpha=0.7, label='加速度')
    line2, = ax2_twin.plot(t, swing.gyro_magnitudes, color=color_gyro, lw=1.2, alpha=0.7, label='角速度')

    ax.set_xlabel('时间 (ms)')
    ax.set_ylabel('加速度 (m/s²)', color=color_accel)
    ax2_twin.set_ylabel('角速度 (deg/s)', color=color_gyro)
    ax.tick_params(axis='y', labelcolor=color_accel)
    ax2_twin.tick_params(axis='y', labelcolor=color_gyro)

    # 标注峰值
    accel_peak_idx = np.argmax(swing.accel_magnitudes)
    gyro_peak_idx = np.argmax(swing.gyro_magnitudes)
    ax.annotate(f'{swing.max_accel:.0f}', xy=(t[accel_peak_idx], swing.max_accel),
                xytext=(t[accel_peak_idx] + 15, swing.max_accel + 5),
                fontsize=7, color=color_accel,
                arrowprops=dict(arrowstyle='->', color=color_accel, lw=1))
    ax2_twin.annotate(f'{swing.max_gyro:.0f}', xy=(t[gyro_peak_idx], swing.max_gyro),
                       xytext=(t[gyro_peak_idx] + 15, swing.max_gyro + 50),
                       fontsize=7, color=color_gyro,
                       arrowprops=dict(arrowstyle='->', color=color_gyro, lw=1))

    # 击球点竖线
    if swing.impact_detected and 0 < swing.impact_idx < len(t):
        ax.axvline(x=t[swing.impact_idx], color='orange', linestyle='--', alpha=0.7, lw=1.5)
        ax.text(t[swing.impact_idx] + 5, ax.get_ylim()[1] * 0.9, '击球',
                fontsize=7, color='orange', fontweight='bold')

    ax.set_title('传感器数据曲线', fontsize=10)
    lines = [line1, line2]
    ax.legend(lines, [l.get_label() for l in lines], fontsize=7)


def plot_speed_curve(ax, swing: ProcessedSwing):
    """绘制速度曲线"""
    t = swing.timestamps
    speed = np.sqrt(np.sum(swing.velocity ** 2, axis=1))
    speed_kmh = speed * 3.6

    ax.fill_between(t, 0, speed_kmh, color='#4CAF50', alpha=0.3)
    ax.plot(t, speed_kmh, color='#388E3C', lw=2)

    peak_idx = np.argmax(speed_kmh)
    ax.annotate(f'峰值: {speed_kmh[peak_idx]:.1f} km/h',
                xy=(t[peak_idx], speed_kmh[peak_idx]),
                xytext=(t[peak_idx] + 15, speed_kmh[peak_idx] - 8),
                fontsize=8, fontweight='bold', color='#2E7D32',
                arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5))

    # 击球点竖线
    if swing.impact_detected and 0 < swing.impact_idx < len(t):
        ax.axvline(x=t[swing.impact_idx], color='orange', linestyle='--', alpha=0.7, lw=1.5)

    ax.set_xlabel('时间 (ms)')
    ax.set_ylabel('速度 (km/h)')
    ax.set_title('拍头速度曲线', fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_score_breakdown(ax, score: TopspinScore):
    """绘制评分明细雷达图"""
    categories = list(score.details.keys())
    values = list(score.details.values())
    n = len(categories)

    # 短标签
    short_labels = {
        "swing_angle": "仰角\n35%",
        "peak_speed": "拍速\n23%",
        "trajectory_smoothness": "平滑\n18%",
        "gyro_upward": "上旋\n14%",
        "impact_quality": "击球\n10%",
    }

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    values_plot = values + values[:1]

    ax.set_aspect('equal')
    ax.fill(angles, values_plot, color='#2196F3', alpha=0.25)
    ax.plot(angles, values_plot, color='#1565C0', lw=2, marker='o', markersize=8)

    for level in [25, 50, 75]:
        ax.plot(angles, [level] * len(angles), '--', color='#CCC', lw=0.5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [short_labels.get(c, c) for c in categories],
        fontsize=7
    )
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25', '50', '75', '100'], fontsize=6)

    ax.text(0, -15, f'总分: {score.total:.0f}/100\n等级: {score.grade}',
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#FFF9C4', edgecolor='#F9A825'))

    ax.set_title('评分明细', fontsize=10)


def plot_vibration_waveform(ax, swing: ProcessedSwing):
    """绘制击球后震动波形"""
    t = swing.timestamps
    accel_mag = swing.accel_magnitudes

    if swing.impact_detected and swing.impact_idx < len(t):
        idx = swing.impact_idx
        # 显示击球前后 150ms
        sample_period = np.mean(np.diff(t)) if len(t) > 1 else 5.0
        half_window = int(150 / sample_period)
        start = max(0, idx - half_window // 3)
        end = min(len(t), idx + half_window)

        t_seg = t[start:end] - t[idx]
        accel_seg = accel_mag[start:end]

        ax.plot(t_seg, accel_seg, color='#333', lw=1.5)
        ax.axvline(x=0, color='orange', linestyle='--', lw=2, alpha=0.8, label='击球瞬间')
        ax.fill_between(t_seg[t_seg >= 0], 0,
                         accel_seg[t_seg >= 0],
                         color='#FF5722', alpha=0.15, label='震动区域')

        # 标注峰值
        ax.annotate(f'{swing.impact_peak_g:.1f}g',
                    xy=(0, swing.impact_peak_g * 9.81),
                    xytext=(15, swing.impact_peak_g * 9.81 + 5),
                    fontsize=8, fontweight='bold', color='#FF5722',
                    arrowprops=dict(arrowstyle='->', color='#FF5722', lw=1.5))

        ax.set_xlabel('时间 (ms, 0=击球瞬间)')
        ax.set_title(f'击球震动波形 (峰值: {swing.impact_peak_g:.1f}g)', fontsize=10)
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, '未检测到击球震动', ha='center', va='center',
                transform=ax.transAxes, fontsize=11, color='#999')
        ax.set_title('击球震动波形', fontsize=10)

    ax.set_ylabel('加速度 (m/s²)')
    ax.grid(True, alpha=0.3)


def plot_fft_spectrum(ax, swing: ProcessedSwing):
    """绘制震动 FFT 频谱图"""
    if not swing.impact_detected or swing.impact_idx >= len(swing.timestamps):
        ax.text(0.5, 0.5, '无震动频谱数据', ha='center', va='center',
                transform=ax.transAxes, fontsize=11, color='#999')
        ax.set_title('震动频谱分析', fontsize=10)
        return

    idx = swing.impact_idx
    t = swing.timestamps
    accel_mag = swing.accel_magnitudes

    # 取击球后 120ms 窗口
    sample_period = np.mean(np.diff(t)) if len(t) > 1 else 5.0
    sample_rate = 1000.0 / sample_period
    window_samples = int(120 / sample_period)
    end_idx = min(len(t), idx + window_samples)

    if end_idx <= idx + 8:
        ax.text(0.5, 0.5, '震动窗口不足', ha='center', va='center',
                transform=ax.transAxes, fontsize=11, color='#999')
        ax.set_title('震动频谱分析', fontsize=10)
        return

    signal = accel_mag[idx:end_idx]
    signal = signal - np.mean(signal)
    n = len(signal)
    window = np.hanning(n)
    signal_win = signal * window

    fft = np.abs(np.fft.rfft(signal_win))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    # 显示有效频率范围 (根据采样率自适应)
    nyquist = sample_rate / 2
    valid = (freqs >= 10) & (freqs <= nyquist * 0.9)
    if not np.any(valid):
        ax.text(0.5, 0.5, '频谱范围异常', ha='center', va='center',
                transform=ax.transAxes, fontsize=11, color='#999')
        ax.set_title('震动频谱分析', fontsize=10)
        return

    freqs_v = freqs[valid]
    fft_v = fft[valid]

    # 平滑
    if len(fft_v) > 3:
        fft_v = np.convolve(fft_v, np.ones(3) / 3, mode='same')

    ax.fill_between(freqs_v, 0, fft_v, color='#9C27B0', alpha=0.3)
    ax.plot(freqs_v, fft_v, color='#7B1FA2', lw=1.5)

    # 标注主频
    dom_freq = swing.vibration_dominant_freq
    if dom_freq > 0:
        peak_idx = np.argmax(fft_v)
        ax.annotate(f'主频: {dom_freq:.0f} Hz\n≈ {swing.estimated_tension_lbs:.0f} lbs',
                    xy=(freqs_v[peak_idx], fft_v[peak_idx]),
                    xytext=(freqs_v[peak_idx] + 40, fft_v[peak_idx] * 0.9),
                    fontsize=8, fontweight='bold', color='#7B1FA2',
                    arrowprops=dict(arrowstyle='->', color='#7B1FA2', lw=1.5),
                    bbox=dict(boxstyle='round', facecolor='#F3E5F5', alpha=0.8))

        # 推荐磅数范围
        if swing.recommended_tension_min > 0:
            ax.text(0.95, 0.95,
                    f'推荐: {swing.recommended_tension_min:.0f}~{swing.recommended_tension_max:.0f} lbs',
                    transform=ax.transAxes, fontsize=8, fontweight='bold',
                    color='#4CAF50', ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#4CAF50'))

    ax.set_xlabel('频率 (Hz)')
    ax.set_ylabel('幅值')
    ax.set_title(f'震动频谱分析 (主频: {dom_freq:.0f} Hz)', fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_session_history(history: list, save_path: str = None):
    """绘制练习历史趋势图"""
    if not history:
        print("暂无历史数据")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('练习历史趋势', fontsize=14, fontweight='bold')

    indices = list(range(len(history)))
    total_scores = [h['total_score'] for h in history]
    angles = [h['angle'] for h in history]
    speeds = [h['speed_kmh'] for h in history]
    gyro_scores = [h['gyro_score'] for h in history]

    # 总分趋势
    ax = axes[0, 0]
    ax.plot(indices, total_scores, 'o-', color='#2196F3', lw=2, markersize=8)
    ax.fill_between(indices, 0, total_scores, alpha=0.2, color='#2196F3')
    ax.axhline(y=75, color='green', linestyle='--', alpha=0.5, label='优秀线')
    ax.axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='及格线')
    ax.set_xlabel('挥拍次数')
    ax.set_ylabel('总分')
    ax.set_title('综合评分趋势')
    ax.legend(fontsize=7)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # 仰角趋势
    ax = axes[0, 1]
    ax.plot(indices, angles, 'o-', color='#FF9800', lw=2, markersize=8)
    ax.fill_between(indices, 30, 50, alpha=0.15, color='green', label='理想范围 30°~50°')
    ax.axhline(y=30, color='green', linestyle='--', alpha=0.7)
    ax.axhline(y=50, color='green', linestyle='--', alpha=0.7)
    ax.set_xlabel('挥拍次数')
    ax.set_ylabel('仰角 (°)')
    ax.set_title('挥拍仰角趋势')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 速度趋势
    ax = axes[1, 0]
    ax.plot(indices, speeds, 'o-', color='#4CAF50', lw=2, markersize=8)
    ax.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='最低速度 20 km/h')
    ax.set_xlabel('挥拍次数')
    ax.set_ylabel('速度 (km/h)')
    ax.set_title('拍头速度趋势')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 上旋得分趋势
    ax = axes[1, 1]
    ax.plot(indices, gyro_scores, 'o-', color='#9C27B0', lw=2, markersize=8)
    ax.set_xlabel('挥拍次数')
    ax.set_ylabel('旋转得分')
    ax.set_title('上旋质量趋势')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"历史趋势已保存: {save_path}")
    plt.close()
