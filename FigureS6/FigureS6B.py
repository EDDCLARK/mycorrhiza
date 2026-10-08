# =============================================================================
# Main Heatmap Only
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
from matplotlib import rcParams
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# —————————————————————————————————————————— Data ——————————————————————————————————————————————

gdf = pd.read_csv("D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/new/all_plots.csv")

cols = ["AM_prop","PC1_new", "EcM_prop", "MAT", "MAP"]
sub = gdf[cols].dropna().copy()

n_bins_mat = 8
n_bins_map = 8

sub["MAT_bin"] = pd.cut(
    sub["MAT"], bins=n_bins_mat,
    labels=False, include_lowest=True
)

sub["MAP_bin"] = pd.cut(
    sub["MAP"], bins=n_bins_map,
    labels=False, include_lowest=True
)

# 每格统计量
grid = (sub.groupby(["MAT_bin", "MAP_bin"])["AM_prop"]
           .agg(mean="mean", std="std", n="count")
           .reset_index())

mean_mat = np.full((n_bins_map, n_bins_mat), np.nan)
std_mat  = np.full((n_bins_map, n_bins_mat), np.nan)
n_mat    = np.full((n_bins_map, n_bins_mat), np.nan)

for _, row in grid.iterrows():
    i = int(row["MAP_bin"])   # y轴 (MAP)
    j = int(row["MAT_bin"])   # x轴 (MAT)
    mean_mat[i, j] = row["mean"]
    std_mat[i, j]  = row["std"]
    n_mat[i, j]    = row["n"]

# 过滤样本量不足的网格
# mean_mat[n_mat < 100] = np.nan
# std_mat[n_mat < 100]  = np.nan

# 计算分箱中心（刻度标签）
mat_edges = pd.cut(sub["MAT"], bins=n_bins_mat, retbins=True, include_lowest=True)[1]
map_edges = pd.cut(sub["MAP"], bins=n_bins_map, retbins=True, include_lowest=True)[1]

mat_centers = (mat_edges[:-1] + mat_edges[1:]) / 2
map_centers = (map_edges[:-1] + map_edges[1:]) / 2


# ══════════════════════════════════════════════════════════════════════════════
# 绘图样式与设置
# ══════════════════════════════════════════════════════════════════════════════
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("white")
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 9
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9

# ecm
# vmax = np.nanpercentile(np.abs(mean_mat), 98)
# norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
# cmap = plt.cm.Spectral

# pc1
vmax = np.nanpercentile(np.abs(mean_mat), 98)
norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
cmap = plt.cm.Spectral

# ══════════════════════════════════════════════════════════════
# 单子图绘制 (只保留 ax_main)
# ══════════════════════════════════════════════════════════════
fig, ax_main = plt.subplots(figsize=(6.5 / 2.54, 6.5 / 2.54))

im = ax_main.imshow(
    mean_mat,
    origin      = "lower",
    aspect      = "auto",
    cmap        = cmap,
    norm        = norm,
    interpolation = "nearest",
)

# 单元格数值标注
for i in range(n_bins_map):
    for j in range(n_bins_mat):
        val = mean_mat[i, j]
        n   = n_mat[i, j]
        if np.isnan(val):
            continue
        txt_color = "black" if abs(val) < 0.7 and abs(val) > 0.3 else "white"        # ecm:   "black" if abs(val) < 0.7 and abs(val) > 0.3 else "white"
        ax_main.text(                                                                # pc1: "black" if abs(val) < 2 else "white"
            j, i, f"{val:+.2f}",                                                     # ecm no +;        pc1   +
            ha        = "center",
            va        = "center",
            fontsize  = 6.5,
            color     = txt_color,
            fontweight= "normal",
        )

# 设置坐标轴刻度与标签
ax_main.set_xticks(range(n_bins_mat))
ax_main.set_xticklabels([f"{v:.1f}" for v in mat_centers])
ax_main.set_yticks(range(n_bins_map))
ax_main.set_yticklabels([f"{v:.0f}" for v in map_centers])

ax_main.set_xlabel("MAT (°C)", labelpad=4)
ax_main.set_ylabel("MAP (mm)", labelpad=4)

# 次要刻度用于绘制单元格白线网格
ax_main.set_xticks(np.arange(-0.5, n_bins_mat, 1), minor=True)
ax_main.set_yticks(np.arange(-0.5, n_bins_map, 1), minor=True)
ax_main.grid(which="minor", color="white", linewidth=1.2)
ax_main.tick_params(which="minor", length=0)
ax_main.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

fig.savefig("./FigS4_b.pdf",format="pdf", dpi=300, bbox_inches="tight")