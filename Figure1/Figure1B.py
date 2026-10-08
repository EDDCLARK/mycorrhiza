import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from scipy import stats

# =============================================================================
# 1. Nature 风格出版级样式设置
# =============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("white")
rcParams['font.family'] = 'Arial'
# rcParams['font.size'] = 7
# rcParams['axes.labelsize'] = 8
# rcParams['xtick.labelsize'] = 7
# rcParams['ytick.labelsize'] = 7
# rcParams['axes.titlesize'] = 8
# rcParams['axes.linewidth'] = 0.5
# rcParams['xtick.major.width'] = 0.5
# rcParams['ytick.major.width'] = 0.5
# rcParams['xtick.major.size'] = 3
# rcParams['ytick.major.size'] = 3
# rcParams['pdf.fonttype'] = 42
# rcParams['ps.fonttype'] = 42
rcParams['font.size'] = 9  # 图片内字体大小（五号字=10.5，图片内用8-9）
rcParams['axes.labelsize'] = 12  # 坐标轴标签稍大
rcParams['xtick.labelsize'] = 9  # X轴刻度
rcParams['ytick.labelsize'] = 9  # Y轴刻度

# =============================================================================
# 2. 数据读取与 Division 聚合计算
# =============================================================================
csv_path = "D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/new/all_plots_environment_division.csv"
gdf = pd.read_csv(csv_path)
print(gdf.axes)


# 清理核心变量
data_clean = gdf[['PC1_new', 'EcM_prop_x', 'DIVISION_C']].dropna().copy()
data_clean = data_clean.rename(columns={'DIVISION_C': 'division', 'EcM_prop_x': 'prp_EcM'})

# 按 division 聚合计算 median, IQR, count
div_stats = data_clean.groupby('division').agg(
    pc1_mean=('PC1_new', 'mean'),
    pc1_median=('PC1_new', 'median'),
    pc1_std=('PC1_new', 'std'),
    pc1_se=('PC1_new', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
    pc1_q25=('PC1_new', lambda x: np.percentile(x, 25)),
    pc1_q75=('PC1_new', lambda x: np.percentile(x, 75)),
    ecm_mean=('prp_EcM', 'mean'),
    ecm_median=('prp_EcM', 'median'),
    ecm_std=('prp_EcM', 'std'),
    ecm_q25=('prp_EcM', lambda x: np.percentile(x, 25)),
    ecm_q75=('prp_EcM', lambda x: np.percentile(x, 75)),
    ecm_se=('prp_EcM', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
    n=('PC1_new', 'count')
).reset_index()

print(div_stats)



# 过滤掉 count 过小或 std 为 NaN 的类别
div_stats = div_stats.dropna(subset=['pc1_std', 'ecm_std'])

# 计算非对称误差
xerr_left = div_stats['ecm_median'] - div_stats['ecm_q25']
xerr_right = div_stats['ecm_q75'] - div_stats['ecm_median']
yerr_left = div_stats['pc1_median'] - div_stats['pc1_q25']
yerr_right = div_stats['pc1_q75'] - div_stats['pc1_median']

# 计算 Log Scale 映射的点大小 (Size)
min_size = 20
max_size = 120
log_n = np.log10(div_stats['n'])
if log_n.max() != log_n.min():
    div_stats['point_size'] = min_size + (log_n - log_n.min()) / (log_n.max() - log_n.min()) * (max_size - min_size)
else:
    div_stats['point_size'] = 50

# =============================================================================
# 3. 拟合原始全量数据 (All Plots Linear Regression)
# =============================================================================
x_raw = data_clean['prp_EcM']
y_raw = data_clean['PC1_new']

# 计算一元线性回归
slope, intercept, r_value, p_value, std_err = stats.linregress(x_raw, y_raw)

# 格式化 p 值的论文标注格式
if p_value < 0.001:
    p_text = "p < 0.001"
else:
    p_text = f"p = {p_value:.3f}"

reg_str = f"Slope = {slope:.2f}\n{p_text}"

# =============================================================================
# 4. 绘图 (Nature Single Column 宽度)
# =============================================================================
fig, ax = plt.subplots(figsize=(8 / 2.54, 6.5 / 2.54), dpi=300)

# --- A. 底图：所有 Plot 的半透明背景散点 ---
ax.scatter(
    x_raw,
    y_raw,
    s=6,
    color='#D0D5DD',  # 浅灰色背景
    alpha=0.25,
    edgecolor='none',
    zorder=1,
    label='All plots'
)

# --- B. 原始数据的回归拟合线 ---
x_fit = np.linspace(x_raw.min(), x_raw.max(), 100)
y_fit = slope * x_fit + intercept
ax.plot(
    x_fit,
    y_fit,
    color='#1A5276',   # 深蓝色拟合线
    linestyle='--',     # 虚线线型（也可改为 '-' 实线）
    linewidth=1.0,
    zorder=2
)

# --- C. X / Y 双向误差线 (Errorbars) ---
ax.errorbar(
    x=div_stats['ecm_median'],
    y=div_stats['pc1_median'],
    xerr=[xerr_left, xerr_right],
    yerr=[yerr_left, yerr_right],
    fmt='none',
    ecolor='#475467',  # 深灰/石墨色误差线
    elinewidth=0.6,
    capthick=0.6,
    alpha=0.7,
    zorder=3
)

# --- D. 叠加 Division 均值散点（大小按 log(n) 缩放）---
sc = ax.scatter(
    div_stats['ecm_median'],
    div_stats['pc1_median'],
    s=div_stats['point_size'],
    c=div_stats['ecm_median'],
    cmap='Spectral',
    edgecolor='#101828',  # 深色细外边框
    linewidth=0.6,
    alpha=0.9,
    zorder=4
)

# =============================================================================
# 5. 标注数据量 ($n$) 与 回归参数
# =============================================================================
# A. 每个 Division 标注样本量
for _, row in div_stats.iterrows():
    x = row['ecm_median']
    y = row['pc1_median']
    n_val = int(row['division'])    #int(row['n'])

    ax.annotate(
        f'{n_val}',
        xy=(x, y),
        xytext=(2, 2),
        textcoords='offset points',
        fontsize=7,
        color='#1D2939',
        fontweight='normal',
        zorder=5
    )

# B. 标注 Slope、$R^2$ 和 $p$ 值到左下角或合适空白处
ax.text(
    0.05, 0.08,             # 相对坐标 (x=0.05, y=0.08)，可根据图面自由调整
    reg_str,
    transform=ax.transAxes,
    fontsize=8.5,
    fontweight='normal',
    color='#1A5276',
    va='bottom',
    ha='left',
    zorder=5
)

# =============================================================================
# 6. 坐标轴与细节优化
# =============================================================================
sns.despine(ax=ax, top=True, right=True)

ax.set_xlabel('EcM tree dominance', labelpad=3)
ax.set_ylabel('LES (PC1)', labelpad=3)
ax.set_ylim(-6, 4.5)

ax.tick_params(direction='out', which='both')

plt.tight_layout()

# 保存矢量图
fig.savefig("Fig1_b.pdf", dpi=300, format='pdf', bbox_inches='tight')
# plt.show()