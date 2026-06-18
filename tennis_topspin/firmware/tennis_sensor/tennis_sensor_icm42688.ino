/*
 * 网球上旋传感器 - ESP32-S3 + ICM-42688-P 固件
 *
 * 硬件:
 *   - 主控: ESP32-S3 Mini / Seeed XIAO ESP32S3
 *   - 传感器: ICM-42688-P (六轴, I2C/SPI)
 *   - 电源: 3.7V 锂电池 + TP4056
 *
 * 接线 (I2C 模式, 默认):
 *   ESP32-S3      ICM-42688-P
 *   ---------     -----------
 *   3.3V      ->  VCC
 *   GND       ->  GND
 *   GPIO6(SDA)->  SDA
 *   GPIO7(SCL)->  SCL
 *   GPIO5     ->  INT (可选, 中断引脚)
 *
 * 接线 (SPI 模式, 取消下方 USE_SPI 注释):
 *   ESP32-S3      ICM-42688-P
 *   ---------     -----------
 *   3.3V      ->  VCC
 *   GND       ->  GND
 *   GPIO10    ->  MOSI/SDA
 *   GPIO9     ->  MISO/AD0
 *   GPIO8     ->  SCK/SCL
 *   GPIO7     ->  CS
 *   GPIO5     ->  INT (可选)
 *
 * 库依赖 (Arduino Library Manager 搜索安装):
 *   - ICM42688 by hideakitai
 *
 * BLE 设备名: TennisSwing
 * 输出: 200Hz 采样，挥拍自动检测，BLE 实时传输
 */

#include <ICM42688.h>
#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>
#include <BLE2902.h>

// ==================== 通信协议选择 ====================
// 取消下面这行注释则使用 SPI，否则默认 I2C
// #define USE_SPI

// ==================== 引脚配置 ====================
#ifdef USE_SPI
  #define ICM_CS_PIN   7
  // SPI: MOSI=10, MISO=9, SCK=8 (ESP32-S3 默认 SPI2 引脚)
  ICM42688 imu(SPI, ICM_CS_PIN);
#else
  // I2C: SDA=6, SCL=7 (ESP32-S3 默认 I2C 引脚)
  // XIAO ESP32S3 用户请改为: SDA=44, SCL=43
  ICM42688 imu(Wire, 0x68);
#endif

#define INT_PIN 5  // 可选中断引脚，不用则悬空

// ==================== BLE 配置 ====================
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

// ==================== 采样配置 ====================
#define SAMPLE_RATE_HZ      200
#define SAMPLE_INTERVAL_US  (1000000 / SAMPLE_RATE_HZ)
#define SWING_THRESHOLD     15.0     // 加速度幅值阈值 (m/s²)
#define BUFFER_SIZE         400      // 最大 2 秒缓冲 (200Hz × 2s)
#define POST_SWING_MS       300      // 挥拍结束后继续记录

// ==================== 数据结构 ====================
struct SensorSample {
  uint32_t timestamp_us;
  float accel_x, accel_y, accel_z;
  float gyro_x, gyro_y, gyro_z;
};

// ==================== 全局变量 ====================
BLEServer* pServer = nullptr;
BLECharacteristic* pCharacteristic = nullptr;
bool deviceConnected = false;

SensorSample buffer[BUFFER_SIZE];
int bufferIndex = 0;
bool isSwinging = false;
unsigned long swingStartTime = 0;
unsigned long lastSampleTime = 0;
unsigned long lastAboveThreshold = 0;

// ==================== BLE 回调 ====================
class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) {
    deviceConnected = true;
    Serial.println("BLE 已连接");
  }
  void onDisconnect(BLEServer* pServer) {
    deviceConnected = false;
    Serial.println("BLE 已断开，重新广播...");
    BLEDevice::startAdvertising();
  }
};

// ==================== 初始化 ====================
void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  delay(500);

  // --- 初始化 ICM-42688-P ---
  Serial.println("正在初始化 ICM-42688-P...");
  int status = imu.begin();
  if (status < 0) {
    Serial.printf("传感器初始化失败! 错误码: %d\n", status);
    Serial.println("请检查: 1.接线 2.供电 3.I2C地址(0x68或0x69)");
    while (1) delay(1000);
  }
  Serial.println("传感器就绪 (ICM-42688-P)");

  // 配置传感器参数
  imu.setAccelFS(ICM42688::AccelFS::gpm16);    // ±16g 量程
  imu.setGyroFS(ICM42688::GyroFS::dps500);     // ±500dps 量程
  imu.setAccelODR(ICM42688::ODR::odr200);       // 200Hz 加速度输出
  imu.setGyroODR(ICM42688::ODR::odr200);        // 200Hz 陀螺仪输出
  imu.setFilters(true, true);                     // 开启低通滤波
  Serial.println("ICM-42688-P 配置完成 (200Hz, ±16g, ±500dps)");

  // --- 初始化 BLE ---
  BLEDevice::init("TennisSwing");
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new ServerCallbacks());

  BLEService* pService = pServer->createService(SERVICE_UUID);
  pCharacteristic = pService->createCharacteristic(
    CHARACTERISTIC_UUID,
    BLECharacteristic::PROPERTY_READ |
    BLECharacteristic::PROPERTY_NOTIFY
  );
  pCharacteristic->addDescriptor(new BLE2902());
  pService->start();

  BLEAdvertising* pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  BLEDevice::startAdvertising();
  Serial.println("BLE 广播已启动 | 设备名: TennisSwing");
  Serial.println("等待挥拍...\n");

  lastSampleTime = micros();
}

// ==================== 主循环 ====================
void loop() {
  unsigned long now = micros();

  // 严格 200Hz 采样
  if (now - lastSampleTime < SAMPLE_INTERVAL_US) return;
  lastSampleTime = now;

  // 读取 ICM-42688-P 数据
  imu.getAGT();  // 同时获取加速度、陀螺仪、温度

  float ax = imu.accX();
  float ay = imu.accY();
  float az = imu.accZ();
  float gx = imu.gyrX();
  float gy = imu.gyrY();
  float gz = imu.gyrZ();

  // 计算运动加速度幅值（去除重力后的纯运动量）
  float accelMag = sqrt(ax * ax + ay * ay + az * az);
  float motionAccel = abs(accelMag - 9.81);

  // --- 挥拍检测 ---
  if (!isSwinging && motionAccel > SWING_THRESHOLD) {
    isSwinging = true;
    bufferIndex = 0;
    swingStartTime = now;
    Serial.println("挥拍开始!");
  }

  // --- 记录数据 ---
  if (isSwinging && bufferIndex < BUFFER_SIZE) {
    buffer[bufferIndex].timestamp_us = now - swingStartTime;
    buffer[bufferIndex].accel_x = ax;
    buffer[bufferIndex].accel_y = ay;
    buffer[bufferIndex].accel_z = az;
    buffer[bufferIndex].gyro_x = gx;
    buffer[bufferIndex].gyro_y = gy;
    buffer[bufferIndex].gyro_z = gz;
    bufferIndex++;
  }

  // --- 挥拍结束判断 ---
  if (isSwinging && motionAccel < SWING_THRESHOLD) {
    if (lastAboveThreshold == 0) {
      lastAboveThreshold = now;
    } else if (now - lastAboveThreshold > POST_SWING_MS * 1000UL) {
      isSwinging = false;
      lastAboveThreshold = 0;
      Serial.printf("挥拍结束 | 采样数: %d\n", bufferIndex);
      sendSwingData();
    }
  } else if (motionAccel >= SWING_THRESHOLD) {
    lastAboveThreshold = now;
  }
}

// ==================== BLE 数据传输 ====================
void sendSwingData() {
  if (!deviceConnected) {
    Serial.println("BLE 未连接，数据丢弃");
    return;
  }

  const int samplesPerPacket = 10;  // 每包 10 个样本
  const int sampleSize = 28;        // 每个样本 28 字节

  for (int start = 0; start < bufferIndex; start += samplesPerPacket) {
    int count = min(samplesPerPacket, bufferIndex - start);

    uint8_t packet[1 + samplesPerPacket * sampleSize];
    packet[0] = count;

    for (int i = 0; i < count; i++) {
      SensorSample& s = buffer[start + i];
      int offset = 1 + i * sampleSize;
      memcpy(packet + offset,      &s.timestamp_us, 4);
      memcpy(packet + offset + 4,  &s.accel_x,     4);
      memcpy(packet + offset + 8,  &s.accel_y,     4);
      memcpy(packet + offset + 12, &s.accel_z,     4);
      memcpy(packet + offset + 16, &s.gyro_x,      4);
      memcpy(packet + offset + 20, &s.gyro_y,      4);
      memcpy(packet + offset + 24, &s.gyro_z,      4);
    }

    pCharacteristic->setValue(packet, 1 + count * sampleSize);
    pCharacteristic->notify();
    delay(8);
  }

  // 结束标记
  uint8_t endPacket[1] = {0};
  pCharacteristic->setValue(endPacket, 1);
  pCharacteristic->notify();

  Serial.println("数据传输完成");
}
