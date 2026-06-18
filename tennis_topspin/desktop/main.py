#!/usr/bin/env python3
"""
网球上旋挥拍练习系统 - 桌面应用主入口

用法:
  # 实时 BLE 模式 (需要 ESP32 硬件)
  python main.py --ble

  # 模拟模式 (用于测试和演示)
  python main.py --demo

  # 从本地缓存 CSV 文件加载数据分析
  python main.py --file swing_data.csv
"""

import argparse
import asyncio
import csv
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免弹出窗口阻塞

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# 使用支持中文的字体
_font_path = '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
fm.fontManager.addfont(_font_path)
_prop = fm.FontProperties(fname=_font_path)
_font_name = _prop.get_name()
plt.rcParams['font.family'] = _font_name
plt.rcParams['font.monospace'] = _font_name

import numpy as np

from ble_receiver import SwingReceiver, SwingRecord, SensorSample
from data_processor import SwingProcessor, ProcessedSwing
from scorer import TopspinScorer, TopspinScore
from visualizer import plot_swing_report, plot_session_history


# ==================== 配置 ====================
DATA_DIR = Path.home() / "tennis_data"
DATA_DIR.mkdir(exist_ok=True)


# ==================== 挥拍处理流程 ====================
class SwingSession:
    """挥拍练习会话"""

    def __init__(self):
        self.processor = SwingProcessor()
        self.scorer = TopspinScorer()
        self.history = []  # 当前会话的所有挥拍记录
        self.swing_count = 0

    def handle_swing(self, record: SwingRecord) -> dict:
        """处理一次挥拍：从原始数据到评分的完整流程"""
        self.swing_count += 1

        # 1. 处理 IMU 数据 → 轨迹、特征
        try:
            processed = self.processor.process(record)
        except ValueError as e:
            print(f"[错误] 数据处理失败: {e}")
            return None

        # 2. 评分
        score = self.scorer.score(processed)

        # 3. 保存原始数据到 CSV
        self._save_raw_csv(record, self.swing_count)

        # 4. 打印结果
        self._print_result(processed, score)

        # 5. 记录历史
        history_entry = {
            "swing_no": self.swing_count,
            "timestamp": datetime.now().isoformat(),
            "total_score": score.total,
            "grade": score.grade,
            "angle": processed.swing_angle_deg,
            "speed_kmh": processed.peak_speed * 3.6,
            "gyro_score": score.details.get("gyro_upward", 0),
            "smoothness": score.details.get("trajectory_smoothness", 0),
            "impact_score": score.details.get("impact_quality", 0),
            "impact_peak_g": processed.impact_peak_g,
            "vib_freq": processed.vibration_dominant_freq,
            "tension_lbs": processed.estimated_tension_lbs,
            "rec_tension_min": processed.recommended_tension_min,
            "rec_tension_max": processed.recommended_tension_max,
            "samples": record.sample_count,
            "duration_ms": record.total_duration_ms,
        }
        self.history.append(history_entry)

        # 6. 可视化
        try:
            report_path = DATA_DIR / f"swing_{self.swing_count:04d}_report.png"
            plot_swing_report(processed, score, save_path=str(report_path))
        except Exception as e:
            print(f"[警告] 图表生成失败: {e}")

        return history_entry

    def _print_result(self, swing: ProcessedSwing, score: TopspinScore):
        """打印挥拍结果到控制台"""
        print()
        print("=" * 60)
        print(f"  挥拍 #{self.swing_count}  |  {score.grade_emoji}")
        print("=" * 60)
        print(f"  综合评分: {score.total:.0f}/100")
        print()
        print(f"  仰角:     {swing.swing_angle_deg:.1f}°  (理想: 30°~50°)  → {score.details['swing_angle']:.0f}/100")
        print(f"  拍速:     {swing.peak_speed * 3.6:.1f} km/h            → {score.details['peak_speed']:.0f}/100")
        print(f"  平滑度:                         → {score.details['trajectory_smoothness']:.0f}/100")
        print(f"  上旋:                           → {score.details['gyro_upward']:.0f}/100")
        imp_score = score.details.get('impact_quality', 0)
        print(f"  击球:     {swing.impact_peak_g:.0f}g, {swing.vibration_dominant_freq:.0f}Hz → {imp_score:.0f}/100")
        print()
        if swing.impact_detected and swing.estimated_tension_lbs > 0:
            print(f"  当前磅数: ~{swing.estimated_tension_lbs:.0f} lbs")
            print(f"  推荐磅数: {swing.recommended_tension_min:.0f}~{swing.recommended_tension_max:.0f} lbs")
        print()
        print(f"  评语: {score.feedback}")
        print("=" * 60)

    def _save_raw_csv(self, record: SwingRecord, swing_no: int):
        """保存原始传感器数据到 CSV"""
        filename = DATA_DIR / f"swing_{swing_no:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_us", "accel_x", "accel_y", "accel_z",
                             "gyro_x", "gyro_y", "gyro_z"])
            for s in record.samples:
                writer.writerow([
                    s.timestamp_us,
                    round(s.accel_x, 4), round(s.accel_y, 4), round(s.accel_z, 4),
                    round(s.gyro_x, 4), round(s.gyro_y, 4), round(s.gyro_z, 4),
                ])
        print(f"数据已保存: {filename}")

    def save_session_summary(self):
        """保存会话总结 JSON"""
        if not self.history:
            return
        filename = DATA_DIR / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        print(f"\n会话总结已保存: {filename}")

        # 生成历史趋势图
        try:
            plot_session_history(self.history,
                                 save_path=str(filename).replace('.json', '_trend.png'))
        except Exception as e:
            print(f"[警告] 趋势图生成失败: {e}")

    def show_summary(self):
        """显示会话总结"""
        if not self.history:
            print("暂无挥拍数据")
            return

        scores = [h['total_score'] for h in self.history]
        print()
        print("=" * 55)
        print("  练习会话总结")
        print("=" * 55)
        print(f"  总挥拍次数: {len(self.history)}")
        print(f"  平均分:     {np.mean(scores):.1f}")
        print(f"  最高分:     {np.max(scores):.0f} (第 {np.argmax(scores) + 1} 次)")
        print(f"  最低分:     {np.min(scores):.0f} (第 {np.argmin(scores) + 1} 次)")
        print(f"  等级分布:")
        for grade in ['S', 'A', 'B', 'C', 'D']:
            count = sum(1 for h in self.history if h['grade'] == grade)
            bar = '█' * count
            print(f"    {grade}: {bar} {count}")
        print("=" * 55)


# ==================== 模拟数据生成 (用于演示) ====================
def generate_demo_swing(quality: str = "good") -> SwingRecord:
    """
    生成模拟挥拍数据，用于无硬件时演示软件功能。

    模拟真实的网球挥拍 + 击球瞬间弦床震动。

    Args:
        quality: "excellent" | "good" | "flat" | "poor"
    """
    duration_ms = random.uniform(350, 500)
    sample_count = int(duration_ms * 0.2)  # ~200Hz
    impact_progress = 0.7  # 击球点在挥拍 70% 处

    # 不同质量等级的震动特征（模拟不同磅数）
    # 注: 200Hz 采样率下 Nyquist=100Hz，震动频率需 <90Hz 才能正确检测
    # 真实弦床震动 (300-500Hz) 需要 ≥1000Hz 采样，这里用缩比频率演示原理
    # excellent: 高磅数 → 高频率 (85Hz), 快衰减
    # good: 适中磅数 → 中频率 (72Hz), 正常衰减
    # flat: 偏低磅数 → 低频率 (55Hz), 慢衰减
    # poor: 过低磅数 → 很低频率 (38Hz), 很慢衰减
    impact_config = {
        "excellent": {"freq": 72, "amplitude": 90, "decay_tau": 50, "peak_g": 50},
        "good":      {"freq": 58, "amplitude": 70, "decay_tau": 55, "peak_g": 40},
        "flat":      {"freq": 43, "amplitude": 75, "decay_tau": 70, "peak_g": 45},
        "poor":      {"freq": 25, "amplitude": 40, "decay_tau": 100, "peak_g": 22},
    }
    ic = impact_config[quality]

    samples = []
    sample_period_ms = duration_ms / sample_count

    for i in range(sample_count):
        t_us = int(i * sample_period_ms * 1000)
        t_ms = i * sample_period_ms
        progress = i / max(sample_count - 1, 1)

        # 加速度包络
        envelope = np.sin(progress * np.pi) * (1 + 0.3 * np.sin(progress * np.pi * 2))

        # 噪声
        noise_accel = random.uniform(-1.5, 1.5)
        noise_gyro = random.uniform(-3, 3)

        if quality == "excellent":
            accel_forward = 40 * envelope
            accel_up = 35 * envelope
            accel_lateral = 8 * envelope
            gyro_upward = 600 * envelope
        elif quality == "good":
            accel_forward = 35 * envelope
            accel_up = 25 * envelope
            accel_lateral = 6 * envelope
            gyro_upward = 400 * envelope
        elif quality == "flat":
            accel_forward = 45 * envelope
            accel_up = 5 * envelope
            accel_lateral = 5 * envelope
            gyro_upward = 80 * envelope
        else:  # poor
            accel_forward = 15 * envelope
            accel_up = random.uniform(-5, 10)
            accel_lateral = 5 * envelope
            gyro_upward = 50 * envelope
            noise_accel *= 3
            noise_gyro *= 3

        ax = accel_lateral + noise_accel * 0.3
        ay = accel_forward + noise_accel
        az = -9.8 + accel_up + noise_accel * 0.5

        # ===== 击球震动叠加 =====
        # 在击球点之后叠加衰减正弦波模拟弦床震动
        impact_t_ms = impact_progress * duration_ms
        dt_from_impact = t_ms - impact_t_ms

        if dt_from_impact >= 0:
            # 衰减正弦波: A * exp(-t/tau) * sin(2*pi*f*t)
            decay = np.exp(-dt_from_impact / ic["decay_tau"])
            vibration = ic["amplitude"] * decay * np.sin(
                2 * np.pi * ic["freq"] * dt_from_impact / 1000.0
            )
            # 震动主要分布在 Y(前向) 和 Z(上下) 方向
            ay += vibration * 0.7
            az += vibration * 0.5
            # 击球瞬间的冲击尖峰（极窄脉冲）
            if dt_from_impact < sample_period_ms:
                ay += ic["peak_g"] * 9.81 * 0.9
                az += ic["peak_g"] * 9.81 * 0.4

        gx = random.uniform(-30, 30)
        gy = random.uniform(-30, 30)
        gz = gyro_upward + noise_gyro

        samples.append(SensorSample(
            timestamp_us=t_us,
            accel_x=round(ax, 3), accel_y=round(ay, 3), accel_z=round(az, 3),
            gyro_x=round(gx, 3), gyro_y=round(gy, 3), gyro_z=round(gz, 3),
        ))

    return SwingRecord(samples=samples, total_duration_ms=duration_ms)


# ==================== 主程序 ====================
async def run_ble_mode():
    """BLE 实时模式"""
    print("启动 BLE 实时模式...")
    print("请确保 ESP32 设备已上电并运行固件\n")

    session = SwingSession()

    def on_swing(record: SwingRecord):
        session.handle_swing(record)

    receiver = SwingReceiver(on_swing=on_swing)

    if not await receiver.scan_and_connect():
        print("\n连接失败，请检查设备。")
        return

    print("\n等待挥拍数据... (按 Ctrl+C 退出)\n")

    try:
        await receiver.wait_forever()
    except KeyboardInterrupt:
        pass
    finally:
        await receiver.disconnect()
        session.save_session_summary()
        session.show_summary()


def run_demo_mode():
    """演示模式：生成模拟数据测试整个流水线"""
    print("启动演示模式...\n")

    session = SwingSession()

    qualities = ["excellent", "good", "good", "good", "flat", "excellent", "good"]
    print(f"将模拟 {len(qualities)} 次挥拍，展示不同动作质量的评分差异\n")

    for i, quality in enumerate(qualities):
        print(f"\n>>> 模拟挥拍 ({quality}) <<<")
        record = generate_demo_swing(quality)
        session.handle_swing(record)
        time.sleep(1)  # 间隔以便查看图表

    session.save_session_summary()
    session.show_summary()
    print("\n演示完成! 检查 ~/tennis_data/ 目录查看保存的数据和图表。")


def run_file_mode(filepath: str):
    """从 CSV 文件加载挥拍数据进行分析"""
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        return

    samples = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(SensorSample(
                timestamp_us=int(row['timestamp_us']),
                accel_x=float(row['accel_x']),
                accel_y=float(row['accel_y']),
                accel_z=float(row['accel_z']),
                gyro_x=float(row['gyro_x']),
                gyro_y=float(row['gyro_y']),
                gyro_z=float(row['gyro_z']),
            ))

    if not samples:
        print("CSV 文件中没有数据")
        return

    total_us = samples[-1].timestamp_us - samples[0].timestamp_us
    record = SwingRecord(samples=samples, total_duration_ms=total_us / 1000.0)

    session = SwingSession()
    session.handle_swing(record)
    session.show_summary()


def main():
    parser = argparse.ArgumentParser(
        description="网球上旋挥拍练习系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --ble        # 连接 ESP32 实时练习
  %(prog)s --demo       # 演示模式 (无需硬件)
  %(prog)s -f data.csv  # 分析保存的挥拍数据
        """
    )
    parser.add_argument('--ble', action='store_true', help='BLE 实时连接模式')
    parser.add_argument('--demo', action='store_true', help='模拟演示模式 (无需硬件)')
    parser.add_argument('--file', '-f', type=str, help='从 CSV 文件加载数据进行分析')

    args = parser.parse_args()

    if args.ble:
        asyncio.run(run_ble_mode())
    elif args.file:
        run_file_mode(args.file)
    else:
        # 默认进入演示模式
        run_demo_mode()


if __name__ == '__main__':
    main()
