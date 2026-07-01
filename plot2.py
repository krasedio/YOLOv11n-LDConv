import matplotlib.pyplot as plt
import numpy as np


# 设置字体顺序：首选 Times New Roman，若无中文字形则回退到 SimHei
plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 设置绘图风格
plt.style.use('seaborn-v0_8-white')

# 1. 创建画布
fig, ax = plt.subplots(figsize=(10, 8))

# 2. 定义数据范围
# 这里的 'size' 只是表示一个索引，方便我们在图中标记 1-9
size = np.linspace(0, 10, 100)

params_std = (size**2)

# 红线：AKConv（线性关系）
# AKConv 允许 9x1 或 1x9 这种线性分解，这里我们让 9x9 的 AKConv 参数量也对应 Y 轴的 9
params_ak = size

# 4. 绘制曲线
# 蓝线：标准/可变形卷积 (二次增长)
ax.plot(size, params_std, color='#00A8E1', linewidth=3, label='可变形卷积和标准卷积')
# 红线：AKConv (线性增长)
ax.plot(size, params_ak, color='#E31F2A', linewidth=3, label='线性可变形卷积')

# 5. 设置坐标轴限制
ax.set_xlim(0, 11)
ax.set_ylim(0, 11)

# 6. 绘制辅助线和刻度标记
# X轴刻度点 1 到 9
x_points = np.arange(1, 10)
'''
for x in x_points:
    # 绘制垂直辅助线（红线，只到 LDConv 曲线）
    if x >= 1: # 排除 0
        y_ak = x # LDConv 参数量与 x 相等 (1:1 关系)
        # 绘制红色垂直虚线
        ax.axvline(x=x, ymin=0, ymax=y_ak/11, color='#E31F2A', linestyle='--', linewidth=1.5)
        # 绘制红色水平虚线 (从 Y 轴到 LDConv 曲线)
        ax.plot([0, x], [y_ak, y_ak], color='#E31F2A', linestyle='--', linewidth=1.5)
'''


# 处理蓝线的关键交点 (Kernel Size 1, 2, 3)
'''
critical_x_blue = [1, 2, 3]
for x in critical_x_blue:
    y_std = x**2 # 标准卷积参数量
    if y_std <= 10: # 限制在可视范围内
        # 绘制蓝色垂直虚线
        ax.axvline(x=x, ymin=0, ymax=y_std/11, color='#00A8E1', linestyle='--', linewidth=1.5)
        # 绘制蓝色水平虚线 (从 Y 轴到蓝线)
        ax.plot([0, x], [y_std, y_std], color='#00A8E1', linestyle='--', linewidth=1.5)
'''
# 7. 设置刻度标签 (只需 1-9)
ax.set_xticks(np.arange(1, 10))
ax.set_xticklabels(np.arange(1, 10), fontsize=18,fontfamily='Times New Roman')

ax.set_yticks(np.arange(1, 10))
ax.set_yticklabels(np.arange(1, 10), fontsize=18,fontfamily='Times New Roman')


# 8. 设置坐标轴标签和字体
# ax.set_xlabel('卷积核大小', fontsize=16,  fontweight='bold', loc='right')
# ax.set_ylabel('参数量', fontsize=16, fontweight='bold', loc='top', rotation=0, labelpad=-30)


# 10. 移除顶部和右侧的边框，使刻度只在 X 和 Y 轴
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# 让 Y 轴的标签 rotation=0，并放到顶部
ax.yaxis.set_label_coords(0.056, 0.95)
ax.xaxis.set_label_coords(1.05, -0.01)

# 11. 在图中直接添加图例标签
# 蓝色标签
# ax.text(6.0, 9.8, '可变形卷积和标准卷积', color='#00A8E1', fontsize=14, fontweight='bold', ha='right', va='center')
# 红色标签
# ax.text(8.0, 9.8, '线性可变形卷积', color='#E31F2A', fontsize=14, fontweight='bold', ha='right', va='center')

# plt.tight_layout()
plt.show()