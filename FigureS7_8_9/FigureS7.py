import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import gaussian_kde, wasserstein_distance
from scipy.spatial.distance import mahalanobis
import seaborn as sns

# ==========================================
# 1. 设置 Nature 出版级全局绘图样式
# ==========================================
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Arial']
# rcParams['font.size'] = 8
# rcParams['axes.titlesize'] = 8.5
# rcParams['axes.labelsize'] = 8
# rcParams['xtick.labelsize'] = 7
# rcParams['ytick.labelsize'] = 7
# rcParams['legend.fontsize'] = 7
# rcParams['pdf.fonttype'] = 42
# rcParams['ps.fonttype'] = 42
# rcParams['axes.linewidth'] = 0.6
# rcParams['xtick.major.width'] = 0.6
# rcParams['ytick.major.width'] = 0.6

# ==========================================
# 2. 从 CSV 读取数据并按 0.5 划分组别
# ==========================================
csv_path = "D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/new/all_plots_with_SHAP.csv"
df = pd.read_csv(csv_path)

# 按 EcM_prop = 0.5 划分为两类
am_pc1 = df[df['EcM_prop'] < 0.5]['PC1_new'].dropna().values
ecm_pc1 = df[df['EcM_prop'] > 0.5]['PC1_new'].dropna().values

# ==========================================
# 3. 计算功能空间差异指标 (Metrics)
# ==========================================
# 指标 1: Cohen's d (均值差异的效应量)
n_am, n_ecm = len(am_pc1), len(ecm_pc1)
mean_am, mean_ecm = np.mean(am_pc1), np.mean(ecm_pc1)
var_am, var_ecm = np.var(am_pc1, ddof=1), np.var(ecm_pc1, ddof=1)
pooled_std = np.sqrt(((n_am - 1) * var_am + (n_ecm - 1) * var_ecm) / (n_am + n_ecm - 2))
cohen_d = abs((mean_ecm - mean_am) / pooled_std)

# 指标 2: Wasserstein Distance (Earth Mover's Distance, 推土机距离)
emd_dist = wasserstein_distance(am_pc1, ecm_pc1)

# 指标 3: Distribution Overlap Coefficient (OVL, 分布重叠率)
x_grid = np.linspace(min(df['PC1_new'].min(), -5), max(df['PC1_new'].max(), 5), 1000)
kde_am = gaussian_kde(am_pc1)(x_grid)
kde_ecm = gaussian_kde(ecm_pc1)(x_grid)
ovl_coef = np.trapz(np.minimum(kde_am, kde_ecm), x_grid)

# 指标 4: Bhattacharyya Distance (巴氏距离)
bhattacharyya_coef = np.trapz(np.sqrt(kde_am * kde_ecm), x_grid)
bhattacharyya_dist = -np.log(bhattacharyya_coef) if bhattacharyya_coef > 0 else np.inf

print("=" * 50)
print("【AM 与 EcM 主导群落 PC1_new 功能空间差异指标】:")
print(f"  1. Cohen's d (效应量)             : {cohen_d:.4f}")
print(f"  2. Wasserstein Distance (EMD距离)  : {emd_dist:.4f}")
print(f"  3. Overlapping Coefficient (OVL重叠率): {ovl_coef:.4f} ({ovl_coef*100:.1f}%)")
print(f"  4. Bhattacharyya Distance (巴氏距离): {bhattacharyya_dist:.4f}")
print("=" * 50)

# ==========================================
# 4. 绘制 Nature 风格的 KDE 曲线图
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("white")
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 9
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
fig, ax = plt.subplots(figsize=(3.5, 2.8), dpi=300) # 单栏黄金比例尺寸

# 配色选择: AM 采用柔和橙红(#d95f02), EcM 采用墨绿/青蓝(#1b9e77)
color_am = '#d95f02'
color_ecm = '#1b9e77'

# 1. 绘制 KDE 填充与折线
ax.plot(x_grid, kde_am, color=color_am, linewidth=1.5, label=f'AM-dominated')
ax.fill_between(x_grid, kde_am, alpha=0.25, color=color_am)

ax.plot(x_grid, kde_ecm, color=color_ecm, linewidth=1.5, label=f'EcM-dominated')
ax.fill_between(x_grid, kde_ecm, alpha=0.25, color=color_ecm)

# 2. 添加均值虚线
ax.axvline(mean_am, color=color_am, linestyle='--', linewidth=0.8, alpha=0.8)
ax.axvline(mean_ecm, color=color_ecm, linestyle='--', linewidth=0.8, alpha=0.8)

# 3. 在图表内部添加量化指标标注文本框
metrics_text = (
    f"Effect size (Cohen's $d$): {cohen_d:.2f}\n"
    f"Overlap (OVL): {ovl_coef*100:.1f}%\n"
    f"Wasserstein dist.: {emd_dist:.2f}"
)
ax.text(
    0.04, 0.93, metrics_text,
    transform=ax.transAxes,
    fontsize=6.5,
    va='top', ha='left',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cccccc', alpha=0.8, linewidth=0.5)
)

# 4. 坐标轴与美化
ax.set_xlabel('Functional Space (LES)')
ax.set_ylabel('Kernel Density')

# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.grid(True, linestyle='--', alpha=0.2, color='gray')

ax.legend(frameon=True, loc='upper right', handlelength=1.2)

fig.savefig("./FigS5_a.pdf",format="pdf", dpi=300, bbox_inches="tight")
plt.show()