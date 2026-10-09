# =============================================================================
# leaf traits PCA: PC1 x Leaf economic strategy (PC2)  loadings
# Output         : PDF (vector, Illustrator-editable) + PNG preview
# Requirements   : pip install matplotlib cartopy numpy pandas sklearn
# =============================================================================

import cartopy.crs as ccrs
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import os
from osgeo import gdal
from osgeo import gdalconst
import xarray as xr
# import gloce as gc
import time
from osgeo import gdal
from mpl_toolkits.basemap import Basemap
import matplotlib.ticker as mtick
from scipy.stats import gaussian_kde
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import seaborn as sns
from matplotlib import rcParams, gridspec
import matplotlib.colors as colors
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpcreader
import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
from pyproj import CRS
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
from shapely import wkt
warnings.filterwarnings('ignore')


# ============================================================================

# —————————————————————————————————————————— data  ——————————————————————————————————————————————

df = pd.read_csv("D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/new/all_plots_with_masked_Trait_values_with_PC_scores.csv")
print(df.axes)




df['geometry'] = df['geometry'].apply(wkt.loads)
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")


# ══════════════════════════════════════════════════════════════════════════════
# 3. 绘图
# ══════════════════════════════════════════════════════════════════════════════
'''subplot pattern'''
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("white")
rcParams['font.family'] = 'Arial'  # 论文常用字体
rcParams['font.size'] = 9  # 图片内字体大小（五号字=10.5，图片内用8-9）
rcParams['axes.labelsize'] = 12  # 坐标轴标签稍大
rcParams['xtick.labelsize'] = 9  # X轴刻度
rcParams['ytick.labelsize'] = 9  # Y轴刻度
# rcParams['legend.fontsize'] = 8  # 图例稍小
# rcParams['figure.titlesize'] = 8  # 图片标题

target_proj = ccrs.AlbersEqualArea(
    central_longitude=-96,  # lon_0
    central_latitude=40,  # lat_0
    standard_parallels=(20, 60),  # lat_1, lat_2
    false_easting=0,  # x_0
    false_northing=0,  # y_0
    globe=ccrs.Globe(datum='NAD83', ellipse='GRS80')  # datum=NAD83
)

fig = plt.figure(figsize=(8.4 / 2.54, 6.3 / 2.54))
ax = fig.add_subplot(111, projection=target_proj)

xlim = ax.get_xlim()  # 获取经度范围
ylim = ax.get_ylim()  # 获取纬度范围
print(f"ax1范围 - 经度: {xlim}, 纬度: {ylim}")

# Add map features
# ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)
# ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
# ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':', alpha=0.7)
for spine in ax.spines.values():
    spine.set_visible(False)

shp_path = r"D:/yanbo/mycorrhizal/Phd_Project1/data/version5/S_USA.ECOSYS_ECOMAPDIVISIONS_2025.shp"  # division

reader = shpcreader.Reader(shp_path)
my_feature = cfeature.ShapelyFeature(
    reader.geometries(),
    crs=ccrs.PlateCarree(),
    edgecolor='black',
    facecolor='none',
    linewidth=1,
    linestyle='-',
    alpha=0.7
)
ax.add_feature(my_feature)


# ----------- EcM dminance -------------------------------
# scatter = ax.scatter(gdf.geometry.x, gdf.geometry.y,
#                      c=gdf['EcM_prop'], cmap='Spectral',
#                      s=0.5, alpha=0.8, transform=ccrs.PlateCarree(),
#                      edgecolors='none', linewidth=0.1)


# ── PC1 分级色彩设计 ──────────────────────────────────────────────
# BrBG 发散色板，以零为中心对称分级
vals     = gdf['PC1_m'].dropna()
clim     = np.percentile(np.abs(vals), 98)   # 去除极端值，色阶对称
n_levels = 10                                 # 分级数（偶数，零居中）

cmap     = plt.cm.Spectral_r                      # 红蓝发散：红=acquisitive   蓝=conservative
norm     = mpl.colors.TwoSlopeNorm(
    vmin = -clim,
    vcenter = 0,
    vmax =  clim,
)

scatter = ax.scatter(
    gdf.geometry.x, gdf.geometry.y,
    c           = gdf['PC1_m'],  # A
    cmap        = cmap,
    norm        = norm,
    s           = 0.5,
    alpha       = 0.8,
    transform   = ccrs.PlateCarree(),
    edgecolors  = 'none',
    rasterized  = True,        # 矢量文件中点层光栅化，文件更小
)

# ── Colorbar（图外右侧）──────────────────────────────────────

cax = fig.add_axes([0.12, 0.14, 0.78, 0.02])  # [left, bottom, width, height]
cb = plt.colorbar(
    scatter,
    cax         = cax,
    orientation = "horizontal",
    extend      = "both",          # PC1 both;    ecm  neither
)
# cb.set_label(
# "PC1 value",         # PC1 value; EcM tree dominance
#     labelpad  = 4,
# )
cb.ax.tick_params(width=0.5, length=2.5)
cb.outline.set_linewidth(0.4)



# colors = ['#CCCCCC', '#F3B300',
#             '#509DC2', '#000000']
# ax.text(0.25, 0.2, f'{0.118645:.2%}', transform=ax.transAxes,
#         fontweight='bold',
#         color='#CCCCCC')
# ax.text(0.25, 0.15, f'{0.205004:.2%}', transform=ax.transAxes,
#         fontweight='bold',
#         color='#F3B300')
# ax.text(0.25, 0.1, f'{0.381369:.2%}', transform=ax.transAxes,
#         fontweight='bold',
#         color='#509DC2')
# ax.text(0.25, 0.05, f'{0.294982:.2%}', transform=ax.transAxes,
#         fontweight='bold',
#         color='#000000')



# # Add gridlines
# gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
#                   linestyle='--', linewidth=0.5, alpha=0.5)
# gl.top_labels = False
# gl.right_labels = False
# gl.xlabel_style = {'rotation': 360}

# plt.tight_layout()
plt.show()
# ── 12. Export ────────────────────────────────────────────────────────────────
fig.savefig("./FigS2_c.pdf",format="pdf", dpi=300, bbox_inches="tight")
# fig.savefig("./Fig1_d.pdf",format="pdf", dpi=300, bbox_inches="tight")

