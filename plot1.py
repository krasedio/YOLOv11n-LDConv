import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimSun']        # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False          # 用来正常显示负号


# 1. 定义路径
path_base = r"C:\Users\Admin\Desktop\毕业设计\Codes\yolo\yolo11n\results.csv"
path_akconv = r"C:\Users\Admin\Desktop\毕业设计\Codes\yolo\yolo11n_akconv\results.csv"

# 2. 读取数据并清洗列名
def load_and_clean(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

df_base = load_and_clean(path_base)
df_akconv = load_and_clean(path_akconv)

plt.style.use('seaborn-v0_8-whitegrid')

# ---- Figure 1: 训练损失对比 (Train Loss) ----
plt.figure(figsize=(6, 6))
# 以 Box Loss 为例，你也可以绘制 train/loss (总损失)
plt.plot(df_base['epoch'], df_base['train/box_loss'], label='YOLOv11n', color='gray', linestyle='--')
plt.plot(df_akconv['epoch'], df_akconv['train/box_loss'], label='YOLOv11n-LDConv', color='blue')
#plt.xlabel('轮次')
#plt.ylabel('损失')
plt.legend()
plt.grid(False)
plt.tight_layout()

# ---- Figure 2: 验证损失对比 (Validation Loss) ----
plt.figure(figsize=(6, 6))
plt.plot(df_base['epoch'], df_base['val/box_loss'], label='YOLOv11n', color='gray', linestyle='--')
plt.plot(df_akconv['epoch'], df_akconv['val/box_loss'], label='YOLOv11n-LDConv', color='red')
#plt.xlabel('轮次')
#plt.ylabel('损失')
plt.legend()
plt.grid(False)
plt.tight_layout()


# 最后一次性显示所有窗口
plt.show()

# 打印数值结果
print(f"--- 最终性能统计 ---")
print(f"Baseline - Min Val Loss: {df_base['val/box_loss'].min():.4f} | Max mAP50: {df_base['metrics/mAP50(B)'].max():.4f}")
print(f"AKConv   - Min Val Loss: {df_akconv['val/box_loss'].min():.4f} | Max mAP50: {df_akconv['metrics/mAP50(B)'].max():.4f}")