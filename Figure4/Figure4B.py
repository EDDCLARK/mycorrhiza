import warnings
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from matplotlib.colors import TwoSlopeNorm

warnings.filterwarnings('ignore')

# =============================================================================
# 1. 数据读取与预处理
# =============================================================================
data_path = "D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/new/all_plots_environment_supersection.csv"
df_input = pd.read_csv(data_path)

model_vars = [
    'PC1_new', 'MAT_x', 'Tseasonali_x', 'EcM_prop_x', 'SSection', 'geometry_x'
]

df_model = df_input[model_vars].dropna().copy()

# 空间去重 (保持一致性)
dup_mask = df_model.duplicated(subset=['SSection', 'geometry_x'], keep='first')
if dup_mask.sum() > 0:
    df_model = df_model[~dup_mask].copy().reset_index(drop=True)

# 恢复真实物理尺度用于分箱
df_model['MAT_real'] = df_model['MAT_x']
df_model['Tseas_real'] = df_model['Tseasonali_x']

# =============================================================================
# 2. 8x8 离散网格划分与“实际均值”差值计算
# =============================================================================
n_bins_mat = 8
n_bins_tseas = 8

# 进行 8x8 pd.cut 分箱
df_model['MAT_bin'], mat_edges = pd.cut(
    df_model['MAT_real'], bins=n_bins_mat, labels=False, retbins=True, include_lowest=True
)

df_model['Tseas_bin'], tseas_edges = pd.cut(
    df_model['Tseas_real'], bins=n_bins_tseas, labels=False, retbins=True, include_lowest=True
)

mat_centers = (mat_edges[:-1] + mat_edges[1:]) / 2
tseas_centers = (tseas_edges[:-1] + tseas_edges[1:]) / 2

# 初始化 8x8 结果矩阵
diff_pc1_grid = np.full((n_bins_tseas, n_bins_mat), np.nan)
grid_n_ecm = np.zeros((n_bins_tseas, n_bins_mat))
grid_n_am = np.zeros((n_bins_tseas, n_bins_mat))

# 遍历每个网格计算实际观测均值差值
for i in range(n_bins_tseas):  # y 轴 (Tseasonali)
    for j in range(n_bins_mat):  # x 轴 (MAT)
        # 筛选落入当前网格 (i, j) 的所有样地
        sub = df_model[(df_model['Tseas_bin'] == i) & (df_model['MAT_bin'] == j)]

        # 筛选 EcM 主导 (EcM > 0.8) 和 AM 主导 (EcM < 0.2) 的样地
        sub_ecm = sub[sub['EcM_prop_x'] == 1]
        sub_am = sub[sub['EcM_prop_x'] == 0]

        grid_n_ecm[i, j] = len(sub_ecm)
        grid_n_am[i, j] = len(sub_am)

        # 只有当该网格内 EcM 和 AM 都有真实样地存在时，才计算均值差值
        if len(sub_ecm) > 0 and len(sub_am) > 0:
            mean_ecm_pc1 = sub_ecm['PC1_new'].mean()
            mean_am_pc1 = sub_am['PC1_new'].mean()
            diff_pc1_grid[i, j] = mean_ecm_pc1 - mean_am_pc1

# =============================================================================
# 3. 绘图样式与精细渲染 (Fig1_f 风格)
# =============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("white")
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 9
rcParams['axes.labelsize'] = 11
rcParams['xtick.labelsize'] = 8.5
rcParams['ytick.labelsize'] = 8.5

# 颜色归一化 (根据实际数据的最大绝对值对称分布)
valid_vals = diff_pc1_grid[~np.isnan(diff_pc1_grid)]
vmax = np.percentile(np.abs(valid_vals), 98) if len(valid_vals) > 0 else 1.0
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
cmap = plt.cm.Spectral_r  # 倒转 Spectral

# 单列 Nature 标准物理尺寸 (6.5 cm x 6.5 cm)
fig, ax_main = plt.subplots(figsize=(6.5 / 2.54, 6.5 / 2.54))

im = ax_main.imshow(
    diff_pc1_grid,
    origin="lower",  # y轴从低到高
    aspect="auto",
    cmap=cmap,
    norm=norm,
    interpolation="nearest",
)

# 单元格数值标注
for i in range(n_bins_tseas):
    for j in range(n_bins_mat):
        val = diff_pc1_grid[i, j]
        if np.isnan(val):
            continue

        # 文字颜色动态适配背景深浅
        txt_color = "black" if abs(val) < (vmax * 0.5) else "white"

        ax_main.text(
            j, i, f"{val:+.2f}",
            ha="center", va="center",
            fontsize=6.0,
            color=txt_color,
            fontweight="bold"
        )

# 设置坐标轴刻度与实际物理中心值
ax_main.set_xticks(range(n_bins_mat))
ax_main.set_xticklabels([f"{v:.1f}" for v in mat_centers])
ax_main.set_yticks(range(n_bins_tseas))
ax_main.set_yticklabels([f"{v:.0f}" for v in tseas_centers])

ax_main.set_xlabel("MAT (°C)", labelpad=4)
ax_main.set_ylabel("Temperature Seasonality", labelpad=4)

# 次要刻度用于绘制白色网格线
ax_main.set_xticks(np.arange(-0.5, n_bins_mat, 1), minor=True)
ax_main.set_yticks(np.arange(-0.5, n_bins_tseas, 1), minor=True)
ax_main.grid(which="minor", color="white", linewidth=1.2)
ax_main.tick_params(which="minor", length=0)
ax_main.tick_params(axis='x', rotation=45)

# Colorbar 调整
cbar = fig.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=7)
cbar.set_label(r'Observed $\Delta PC1$ (EcM - AM)', fontsize=8.5)

plt.tight_layout()
plt.show()