import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# 1. 读取数据
file_path = "D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/new/all_plots_with_masked_Trait_values_with_PC_scores.csv"
df = pd.read_csv(file_path)

# 2. 定义两两对照的变量对 (Raw/New vs Masked)
pairs = [
    ('PC1', 'PC1_new', 'PC1_m'),
    ('LNC', 'LNC', 'LNC_m'),
    ('LMA', 'LMA', 'LMA_m'),
    ('LPC', 'LPC', 'LPC_m'),
    ('Lignin', 'Lignin', 'Lignin_m'),
    ('Cellulose', 'Cellulose', 'Cellulose_m'),
    ('NSC', 'NSC', 'NSC_m'),
    ('Phenolics', 'Phenolics', 'Phenolics_m'),
    ('EWT', 'EWT', 'EWT_m'),
]

# 适配可能的列名
actual_pairs = []
for name, x_col, y_col in pairs:
  x_real = x_col if x_col in df.columns else (
      'PC1' if x_col == 'PC1_new' else f'{name}_raw'
  )
  actual_pairs.append((name, x_real, y_col))

# 提取数据并剔除缺失值
all_cols = list(
    set([p[1] for p in actual_pairs] + [p[2] for p in actual_pairs])
)
sub_df = df[all_cols].dropna()

# 3. Z-score 标准化
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(sub_df[all_cols]),
    columns=all_cols,
    index=sub_df.index,
)

# 4. 全局样式设置
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style('white')
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 9  # 图片内字体大小
rcParams['axes.labelsize'] = 12  # 坐标轴标签稍大
rcParams['xtick.labelsize'] = 9  # X轴刻度
rcParams['ytick.labelsize'] = 9  # Y轴刻度

# 5. 创建画布 (8 x 6.5 cm)
fig, ax = plt.subplots(figsize=(8 / 2.54, 6.5 / 2.54), facecolor='white')

# 1:1 参考线
ax.plot(
    [-3.5, 3.5],
    [-3.5, 3.5],
    color='#666666',
    linestyle=':',
    linewidth=1.2,
    label='1:1 Ref',
    zorder=1,
)

# 分配 9 种 distinct 颜色
colors = plt.cm.tab10(np.linspace(0, 1, len(actual_pairs)))

# 6. 循环绘制散点与拟合线
for i, (name, x_col, y_col) in enumerate(actual_pairs):
  x = df_scaled[x_col]
  y = df_scaled[y_col]
  color = colors[i]

  # 线性拟合
  slope, intercept, r_val, p_val, _ = linregress(x, y)
  r2 = r_val**2

  # A. 绘制散点
  ax.scatter(
      x,
      y,
      color=color,
      alpha=0.08,
      s=6,
      edgecolors='none',
      rasterized=True,
      zorder=2,
  )

  # B. 绘制拟合线
  x_line = np.linspace(x.min(), x.max(), 100)
  y_line = slope * x_line + intercept

  lw = 2.0 if name == 'PC1' else 1.3
  ls = '--' if name == 'PC1' else '-'

  ax.plot(
      x_line,
      y_line,
      color=color,
      linestyle=ls,
      linewidth=lw,
      label=f'{name} ($R^2$={r2:.2f})',
      zorder=4,
  )

# ── 坐标轴修饰 ─────────────────────────────────────────────────────────
ax.set_xlabel('Unmasked value (Standardized)', labelpad=4)
ax.set_ylabel('Masked value (Standardized)', labelpad=4)

ax.set_xlim(-3.0, 3.0)
ax.set_ylim(-3.0, 3.0)
ax.set_aspect('equal')

# ── 图例调整：放置于图外右侧 ───────────────────────────────────────────
ax.legend(
    bbox_to_anchor=(1.02, 1),  # 定位到坐标轴外右侧顶部
    loc='upper left',  # 以图例左上角对齐 anchor
    fontsize=7,  # 适合外侧放置的紧凑字号
    frameon=False,  # 移除边框更简洁
    handlelength=1.2,  # 线条长度
    handletextpad=0.4,  # 文字与线的间距
    labelspacing=0.3,  # 行间距
)

ax.grid(True, linestyle='--', alpha=0.3)

# 确保外置图例保存/显示时不被截断
plt.tight_layout()

# 保存 pdf 格式建议加上 bbox_inches='tight'
plt.savefig("FigS2_d.pdf", dpi=300, format='pdf', bbox_inches='tight')
plt.show()