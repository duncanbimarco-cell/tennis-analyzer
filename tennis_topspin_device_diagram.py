import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Ellipse
import matplotlib.lines as mlines

# Use a font that supports Chinese characters
font_path = '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
font_name = prop.get_name()
plt.rcParams['font.family'] = font_name
plt.rcParams['font.monospace'] = font_name  # fallback for monospace CJK

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('网球上旋挥拍练习装置 — 完整设计图', fontsize=18, fontweight='bold', y=0.98)

# ========== 图1: 装置整体示意图 ==========
ax1 = axes[0, 0]
ax1.set_xlim(0, 12)
ax1.set_ylim(0, 10)
ax1.set_title('① 装置整体结构', fontsize=13, fontweight='bold')
ax1.axis('off')

# 球拍
racket_handle = FancyBboxPatch((4.2, 1.5), 0.3, 3.5, boxstyle="round,pad=0.05",
                                facecolor='#8B4513', edgecolor='#5C2E00', linewidth=2)
ax1.add_patch(racket_handle)
racket_head = patches.Ellipse((4.35, 6.2), 2.2, 3.2, angle=0,
                               facecolor='none', edgecolor='#333', linewidth=3)
ax1.add_patch(racket_head)
# 拍线
for i in np.linspace(4.9, 7.8, 6):
    ax1.plot([3.25, 5.45], [i, i], color='#999', linewidth=0.8)
for i in np.linspace(3.25, 5.45, 8):
    ax1.plot([i, i], [4.9, 7.8], color='#999', linewidth=0.8)

# 传感器模块（拍柄底部）
sensor_box = FancyBboxPatch((3.9, 1.3), 0.9, 0.9, boxstyle="round,pad=0.08",
                             facecolor='#2196F3', edgecolor='#1565C0', linewidth=2)
ax1.add_patch(sensor_box)
ax1.text(4.35, 1.75, 'IMU\n传感器', ha='center', va='center', fontsize=7,
         color='white', fontweight='bold')

# 标注
ax1.annotate('MPU6050 六轴传感器\n(加速度+陀螺仪)', xy=(4.8, 1.75), xytext=(7.5, 2.5),
            arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2),
            fontsize=8, color='#1565C0', ha='center',
            bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.9))
ax1.annotate('ESP32 主控\n+ 蓝牙发射', xy=(4.5, 1.0), xytext=(7.5, 0.8),
            arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2),
            fontsize=8, color='#1565C0', ha='center',
            bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.9))
ax1.annotate('锂电池供电', xy=(3.8, 1.75), xytext=(1.2, 2.0),
            arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2),
            fontsize=8, color='#1565C0', ha='center',
            bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.9))

# 挥拍方向弧线
swing_arc = patches.Arc((4.35, 5), 9, 11, angle=0, theta1=220, theta2=320,
                         color='#FF5722', linewidth=3, linestyle='--')
ax1.add_patch(swing_arc)
ax1.annotate('挥拍轨迹\n(低→高)', xy=(8.5, 7.5), fontsize=9, color='#FF5722', fontweight='bold',
            ha='center')

# 网球
ball = patches.Circle((8.2, 5.5), 0.35, facecolor='#FFEB3B', edgecolor='#F9A825', linewidth=2)
ax1.add_patch(ball)
# 上旋箭头
ax1.annotate('', xy=(8.7, 5.5), xytext=(7.7, 5.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2, connectionstyle='arc3,rad=0.3'))
ax1.text(8.2, 6.2, '上旋', fontsize=8, color='red', ha='center', fontweight='bold')

# ========== 图2: 数据流架构 ==========
ax2 = axes[0, 1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.set_title('② 数据流架构', fontsize=13, fontweight='bold')
ax2.axis('off')

# 画流程框
boxes = [
    (0.5, 4.5, 2.5, '传感器采集\n加速度XYZ\n角速度XYZ\n(200Hz采样)', '#2196F3'),
    (3.8, 4.5, 2.5, '蓝牙BLE\n无线传输', '#4CAF50'),
    (7.0, 4.5, 2.5, 'Python桌面应用\n数据接收/存储\n轨迹重建/评分', '#FF9800'),
]
for x, y, w, text, color in boxes:
    box = FancyBboxPatch((x, y), w, 1.8, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='#333', linewidth=2, alpha=0.9)
    ax2.add_patch(box)
    ax2.text(x + w/2, y + 0.9, text, ha='center', va='center', fontsize=7.5,
             color='white', fontweight='bold')

# 箭头
for xx in [3.0, 6.3]:
    ax2.annotate('', xy=(xx + 0.7, 5.4), xytext=(xx, 5.4),
                arrowprops=dict(arrowstyle='->', color='#333', lw=3))

# 数据示例
ax2.text(5, 2.5, '数据包格式: [时间戳, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]',
         ha='center', fontsize=8, family='monospace',
         bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='#CCC'))

# 下方结果展示
result_box = FancyBboxPatch((2, 0.2), 6, 1.8, boxstyle="round,pad=0.1",
                             facecolor='white', edgecolor='#FF9800', linewidth=2)
ax2.add_patch(result_box)
ax2.text(5, 1.5, '输出结果', ha='center', fontsize=9, fontweight='bold', color='#FF9800')
ax2.text(5, 0.8, '3D轨迹图  |  挥拍仰角: 35°  |  拍头速度: 28 km/h  |  上旋评分: 85/100',
         ha='center', fontsize=7, family='monospace')

# ========== 图3: 上旋动作评分依据 ==========
ax3 = axes[1, 0]
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.set_title('③ 上旋评分算法示意', fontsize=13, fontweight='bold')
ax3.axis('off')

# 坐标轴
ax3.plot([1, 1], [1, 9], 'k-', lw=1.5)  # Y轴
ax3.plot([1, 9], [1, 1], 'k-', lw=1.5)  # X轴
ax3.text(0.7, 5, '高度\n(Z轴)', ha='center', fontsize=8)
ax3.text(5, 0.5, '水平位移 (X轴)', ha='center', fontsize=8)

# 标准上旋轨迹
x_ideal = np.linspace(1.5, 8.5, 50)
y_ideal = 1.5 + 6.5 * ((x_ideal - 1.5) / 7) ** 1.8  # 低→高，指数增长模拟加速上扬
ax3.plot(x_ideal, y_ideal, 'b-', lw=3, label='标准上旋轨迹', alpha=0.8)

# 填充可接受范围
y_upper = y_ideal + 0.8
y_lower = y_ideal - 0.6
ax3.fill_between(x_ideal, y_lower, y_upper, alpha=0.15, color='blue')

# 错误轨迹1: 太平
y_flat = 1.5 + 0.8 * ((x_ideal - 1.5))
ax3.plot(x_ideal, y_flat[:50], 'r--', lw=1.5, label='过平(无上旋)', alpha=0.7)

# 错误轨迹2: 太陡
y_steep = 1.5 + 9 * ((x_ideal - 1.5) / 7) ** 1.3
ax3.plot(x_ideal, np.clip(y_steep[:50], 0, 10), 'orange', linestyle='--', lw=1.5,
         label='过陡(切球)', alpha=0.7)

# 击球点
ax3.plot(3.5, 3.0, 'o', color='#FFEB3B', markersize=15, markeredgecolor='#F9A825', markeredgewidth=2)
ax3.plot(3.5, 3.0, 'x', color='red', markersize=6, mew=2)
ax3.annotate('击球点', xy=(3.5, 3.0), xytext=(2.2, 2.3), fontsize=8,
            arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))

# 角度标注
ax3.annotate('上旋仰角\n≈ 35°~50°', xy=(6.5, 4.5), xytext=(7.5, 3.5), fontsize=8, color='blue',
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
            bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8))

ax3.legend(loc='upper left', fontsize=7, framealpha=0.9)

# 评分表
score_table = [
    ('挥拍仰角 (30°~50°)', '权重 40%'),
    ('拍头速度 (>20 km/h)', '权重 25%'),
    ('轨迹平滑度', '权重 20%'),
    ('旋转加速度 (陀螺仪Z)', '权重 15%'),
]
for i, (item, weight) in enumerate(score_table):
    ax3.text(6.5, 8.8 - i * 0.6, f'{item}  {weight}', fontsize=7,
             bbox=dict(boxstyle='round', facecolor='#FFF9C4', edgecolor='#F9A825', alpha=0.7))

# ========== 图4: 3D外壳设计概念 ==========
ax4 = axes[1, 1]
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.set_title('④ 3D打印外壳 & 安装方式', fontsize=13, fontweight='bold')
ax4.axis('off')

# 拍柄截面
handle = FancyBboxPatch((1, 3.5), 1.2, 3, boxstyle="round,pad=0.03",
                          facecolor='#8B4513', edgecolor='#5C2E00', linewidth=2)
ax4.add_patch(handle)
ax4.text(1.6, 5, '拍柄', fontsize=9, ha='center', rotation=90)

# 外壳 - 正面视图
case_front = FancyBboxPatch((3.5, 3.5), 2.5, 1.5, boxstyle="round,pad=0.15",
                              facecolor='#ECEFF1', edgecolor='#546E7A', linewidth=2)
ax4.add_patch(case_front)
ax4.text(4.75, 4.6, 'ESP32 + MPU6050', fontsize=7, ha='center', fontweight='bold')
ax4.text(4.75, 4.2, '电池仓', fontsize=6, ha='center', color='#666')
# 指示灯
led = patches.Circle((6.3, 4.7), 0.12, facecolor='#4CAF50', edgecolor='#388E3C', linewidth=1)
ax4.add_patch(led)
ax4.text(6.3, 5.05, 'LED', fontsize=5, ha='center', color='#4CAF50')
# 开关
switch = FancyBboxPatch((6.0, 3.7), 0.6, 0.3, boxstyle="round,pad=0.02",
                          facecolor='#455A64', edgecolor='#263238', linewidth=1)
ax4.add_patch(switch)
ax4.text(6.3, 3.5, '开关', fontsize=5, ha='center')

# 绑带示意
for yy in [3.2, 5.2]:
    band = FancyBboxPatch((3.3, yy), 3.0, 0.15, boxstyle="round,pad=0.02",
                            facecolor='#333', edgecolor='#111', linewidth=1)
    ax4.add_patch(band)
ax4.text(4.75, 2.6, '弹力绑带固定在拍柄上', fontsize=7, ha='center', color='#555')

# 侧视图框
case_side = FancyBboxPatch((7.5, 3.5), 1.2, 1.5, boxstyle="round,pad=0.1",
                             facecolor='#ECEFF1', edgecolor='#546E7A', linewidth=2)
ax4.add_patch(case_side)
ax4.text(8.1, 4.6, '侧\n视', fontsize=7, ha='center', color='#999')
ax4.text(8.1, 4.1, '~2cm', fontsize=6, ha='center', color='#666')
# 厚度标注
ax4.annotate('', xy=(7.5, 3.2), xytext=(8.7, 3.2),
            arrowprops=dict(arrowstyle='<->', color='#666', lw=1))
ax4.text(8.1, 2.9, '厚度约 20mm', fontsize=7, ha='center', color='#666')

# 爆炸视图标注线
ax4.annotate('USB充电口', xy=(3.8, 3.5), xytext=(2.5, 2.0),
            arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5),
            fontsize=7, color='#1565C0')
ax4.annotate('重置按钮', xy=(5.2, 3.5), xytext=(5.5, 2.2),
            arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5),
            fontsize=7, color='#1565C0')

# 重量标注
ax4.text(4.75, 7, '总重量: ~45g\n(不影响挥拍手感)', ha='center', fontsize=8,
         bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#4CAF50'))

# 元件清单
parts = [
    'ESP32 Dev Board ×1',
    'MPU6050 IMU ×1',
    '3.7V 锂电池 ×1',
    'TP4056 充电模块 ×1',
    '3D打印外壳 ×1',
    '弹力绑带 ×2',
]
ax4.text(0.2, 1.2, '元件清单:\n' + '\n'.join(parts), fontsize=7, family='monospace',
         bbox=dict(boxstyle='round', facecolor='#FAFAFA', edgecolor='#CCC'))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('/Users/zhangsanmao/claude code/tennis_topspin_device.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print('图表已保存: tennis_topspin_device.png')
