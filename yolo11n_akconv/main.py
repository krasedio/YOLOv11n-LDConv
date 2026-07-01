import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('results.csv')
df.columns = [c.strip() for c in df.columns]

sns.set_theme(style="whitegrid")

# 绘制 epoch 关于 mAP50 的回归趋势图
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='epoch', y='metrics/mAP50(B)', linewidth=2)

plt.title('mAP50 Convergence Curve', fontsize=15)
plt.xlabel('Training Epochs', fontsize=12)
plt.ylabel('mAP50 Score', fontsize=12)
plt.show()