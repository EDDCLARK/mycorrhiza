# =============================================================================
# PC2 distribution: EcM dominated VS. AM dominated
# Output         : PDF (vector, Illustrator-editable) + PNG preview
# Requirements   : pip install matplotlib cartopy numpy pandas sklearn
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator
from scipy.stats import gaussian_kde, mannwhitneyu
import warnings
import geopandas as gpd
warnings.filterwarnings('ignore')


# ============================================================================
# —————————————————————————————————————————— data  ——————————————————————————————————————————————
gdf = pd.read_csv("D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/new/all_plots.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 3. 绘图
# ══════════════════════════════════════════════════════════════════════════════
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("white")
rcParams['font.family'] = 'Arial'  # 论文常用字体
rcParams['font.size'] = 9  # 图片内字体大小（五号字=10.5，图片内用8-9）
rcParams['axes.labelsize'] = 12  # 坐标轴标签稍大
rcParams['xtick.labelsize'] = 9  # X轴刻度
rcParams['ytick.labelsize'] = 9  # Y轴刻度
# rcParams['legend.fontsize'] = 8 # 图例稍小
# rcParams['figure.titlesize'] = 8  # 图片标题


fig, ax = plt.subplots(figsize=(8.4 / 2.54, 8.4 / 2.54), facecolor='white')
ax.set_facecolor('white')



ecm = gdf['EcM_prop'].dropna().values
am  = gdf['AM_prop'].dropna().values

# ── 3. 二维直方图分箱（20×20）────────────────────────────────────────────────
bins   = np.linspace(0, 1, 21)          # 21个边界 → 20个格
counts, xedges, yedges = np.histogram2d(ecm, am, bins=[bins, bins])
counts = counts.T                        # 转置：行=AM，列=EcM

# 屏蔽上三角（EcM + AM > 1 区域）
cx = (xedges[:-1] + xedges[1:]) / 2
cy = (yedges[:-1] + yedges[1:]) / 2
XX, YY = np.meshgrid(cx, cy)
counts_masked = np.where(XX + YY > 1.0 + 1e-9, np.nan, counts.astype(float))
counts_masked = np.where(counts_masked == 0, np.nan, counts_masked)  # 0→nan留白

# ── 4. 对数色标 ─────────────────────────────────────────────────────────────
vmin, vmax = 1, 5000

# 自定义色板：白→浅橙→深红，学术感更强
from matplotlib.colors import LinearSegmentedColormap
cmap_colors = [
    (1.00, 1.00, 1.00),   # 白
    (1.00, 0.88, 0.82),   # 极浅粉
    (0.98, 0.68, 0.55),   # 浅橙红
    (0.90, 0.35, 0.25),   # 中红
    (0.70, 0.08, 0.08),   # 深红
]
cmap = LinearSegmentedColormap.from_list('ecm_am', cmap_colors, N=256)
norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

# ── 5. 绘图 ──────────────────────────────────────────────────────────────────

im = ax.pcolormesh(
    xedges, yedges, counts_masked,
    cmap=cmap, norm=norm,
    linewidth=0,          # 格子间不留缝
    rasterized=True,      # PDF体积友好
)

# ── 6. 对角参考线（EcM + AM = 1）────────────────────────────────────────────
ax.plot([0, 1], [1, 0],
        color='#555555', linewidth=0.6, linestyle='--', alpha=0.7, zorder=15)

# ── 7. 坐标轴设置 ────────────────────────────────────────────────────────────
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')

ax.xaxis.set_major_locator(MultipleLocator(0.25))
ax.yaxis.set_major_locator(MultipleLocator(0.25))
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))

ax.tick_params(which='minor', length=1.5, width=0.5)

ax.set_xlabel('EcM Dominance', labelpad=4)
ax.set_ylabel('AM Dominance',  labelpad=4)

# 去掉右、上边框
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)

# ── 8. Colorbar（对数刻度）──────────────────────────────────────────────────
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.08)

cb = fig.colorbar(im, cax=cax)
cb.set_label('Number of plots',labelpad=4)
cb.ax.tick_params(length=2.5, width=0.6)

# 手动设置对数刻度标签（1, 10, 100, 1000, 10000）
import matplotlib.ticker as ticker
cb.set_ticks([1, 10, 100, 1000, 5000])
cb.set_ticklabels(['1', '10', '100', '1000', '5000'])
cb.outline.set_linewidth(0.6)





plt.show()
# ── 12. Export ────────────────────────────────────────────────────────────────
fig.savefig("./FigS2_b.pdf",
            format="pdf", dpi=300, bbox_inches="tight", facecolor="none")
