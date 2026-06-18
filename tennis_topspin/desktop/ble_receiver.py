"""
BLE 数据接收模块
负责扫描、连接 ESP32 设备，接收挥拍数据包并重组为完整的挥拍记录。
"""

import asyncio
import struct
from dataclasses import dataclass
from typing import Callable, Optional
from collections import deque

try:
    from bleak import BleakScanner, BleakClient
    from bleak.backends.device import BLEDevice
except ImportError:
    BleakScanner = None
    BleakClient = None
    BLEDevice = None

SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
DEVICE_NAME = "TennisSwing"


@dataclass
class SensorSample:
    """单个传感器采样点"""
    timestamp_us: int
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float


@dataclass
class SwingRecord:
    """一次完整的挥拍记录"""
    samples: list  # list of SensorSample
    total_duration_ms: float

    @property
    def sample_count(self):
        return len(self.samples)

    @property
    def sample_rate(self):
        if self.total_duration_ms > 0:
            return self.sample_count / (self.total_duration_ms / 1000.0)
        return 0


class SwingReceiver:
    """BLE 挥拍数据接收器"""

    def __init__(self, on_swing: Optional[Callable[[SwingRecord], None]] = None):
        self.on_swing = on_swing
        self._client: Optional[BleakClient] = None
        self._device: Optional[BLEDevice] = None
        self._buffer = bytearray()
        self._receiving = False
        self._packet_count = 0
        self._swing_samples = []

    async def scan_and_connect(self) -> bool:
        """扫描并连接到 TennisSwing 设备"""
        if BleakScanner is None:
            raise RuntimeError("bleak 库未安装，请运行: pip install bleak")

        print(f"正在扫描 BLE 设备 '{DEVICE_NAME}' ...")
        device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=10.0)

        if device is None:
            print(f"未找到设备 '{DEVICE_NAME}'，请确认:")
            print("  1. ESP32 已上电")
            print("  2. 固件已烧录并运行")
            return False

        print(f"找到设备: {device.name} ({device.address})")
        print("正在连接...")

        self._device = device
        self._client = BleakClient(device)
        await self._client.connect()
        print("BLE 已连接!")

        # 订阅通知特征值
        await self._client.start_notify(CHARACTERISTIC_UUID, self._handle_notification)
        print("已订阅数据通知，等待挥拍...")
        return True

    def _handle_notification(self, _sender, data: bytearray):
        """处理 BLE 通知数据"""
        if len(data) < 1:
            return

        count = data[0]
        sample_size = 28  # 每个样本 28 字节

        if count == 0:
            # 挥拍结束标记
            self._finish_swing()
            return

        expected_len = 1 + count * sample_size
        if len(data) != expected_len:
            print(f"[警告] 数据包长度异常: {len(data)} (期望 {expected_len})")
            return

        # 解析样本数据
        for i in range(count):
            offset = 1 + i * sample_size
            try:
                timestamp_us, ax, ay, az, gx, gy, gz = struct.unpack_from(
                    '<Iffffff', data, offset
                )
                sample = SensorSample(
                    timestamp_us=timestamp_us,
                    accel_x=ax, accel_y=ay, accel_z=az,
                    gyro_x=gx, gyro_y=gy, gyro_z=gz,
                )
                self._swing_samples.append(sample)
            except struct.error as e:
                print(f"[警告] 解析样本失败: {e}")
                continue

    def _finish_swing(self):
        """完成一次挥拍数据接收"""
        if not self._swing_samples:
            print("[警告] 收到空挥拍记录")
            return

        total_us = (self._swing_samples[-1].timestamp_us -
                    self._swing_samples[0].timestamp_us)
        record = SwingRecord(
            samples=self._swing_samples,
            total_duration_ms=total_us / 1000.0,
        )

        print(f"收到挥拍数据: {record.sample_count} 个采样点, "
              f"时长 {record.total_duration_ms:.0f}ms, "
              f"采样率 {record.sample_rate:.0f}Hz")

        self._swing_samples = []

        if self.on_swing:
            self.on_swing(record)

    async def disconnect(self):
        """断开 BLE 连接"""
        if self._client and self._client.is_connected:
            await self._client.disconnect()
            print("BLE 已断开")

    async def wait_forever(self):
        """保持连接并等待数据"""
        if self._client is None:
            return
        try:
            while self._client.is_connected:
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            pass
