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



ALPHA_BAR    = 0.45
BIN_WIDTH    = 0.05
bins = np.arange(0, 1.2, BIN_WIDTH)

# ── 直方图 ────────────────────────────────────────────────────────────────────
ax.hist(gdf['EcM_prop'], bins=bins, density=False, color='gray',
        alpha=ALPHA_BAR, edgecolor='black', zorder=2)



print(max(gdf['EcM_prop']))

# # ── 坐标轴设置 ────────────────────────────────────────────────────────────────
ax.set_xlabel(r'EcM Dominance',labelpad=0)
ax.set_ylabel('Number of plots',labelpad=0)
ax.set_xlim(0, 1.05)
# ax.set_ylim(0, 0.5)
ax.set_xticks([0,0.25,0.5,0.75,1])




plt.show()
# ── 12. Export ────────────────────────────────────────────────────────────────
fig.savefig("./FigS2_a.pdf",
            format="pdf", dpi=300, bbox_inches="tight", facecolor="none")
