import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
from matplotlib.patches import Patch

warnings.filterwarnings('ignore')

# ============================================================================
# 1. 全局样式设置 (Nature 期刊排版规范)
# ============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 9
rcParams['axes.labelsize'] = 10
rcParams['xtick.labelsize'] = 8.5
rcParams['ytick.labelsize'] = 8.5
rcParams['hatch.color'] = '#333333'
rcParams['hatch.linewidth'] = 0.5

# ============================================================================
# 2. 读取 Shapefile 并转换投影 (Crucial Step!)
# ============================================================================
shp_path = r"D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/new/partial_r_conifer_supersection.shp"
gdf = gpd.read_file(shp_path)

# 如果没指定 CRS，默认指定为 WGS84 (EPSG:4326)
if gdf.crs is None:
    gdf.set_crs(epsg=4326, inplace=True)

# 目标 Albers 投影 (Cartopy 定义)
target_proj = ccrs.AlbersEqualArea(
    central_longitude=-96,
    central_latitude=40,
    standard_parallels=(20, 60),
    false_easting=0,
    false_northing=0,
    globe=ccrs.Globe(datum='NAD83', ellipse='GRS80')
)

# 💡【核心解决空图步骤】：直接将 GeoPandas 数据重投影为 Cartopy 的 proj4 坐标系
gdf_proj = gdf.to_crs(target_proj.proj4_init)

# ============================================================================
# 3. 数据清洗与分组 (N=0, 显著, 不显著)
# ============================================================================
col_mapping = {c.lower(): c for c in gdf_proj.columns}
r_col = col_mapping.get('partial_r', 'Partial_r')
p_col = col_mapping.get('p_value', 'p_value')
n_col = col_mapping.get('n_samples', 'N_samples')

gdf_proj[r_col] = pd.to_numeric(gdf_proj[r_col], errors='coerce')
gdf_proj[p_col] = pd.to_numeric(gdf_proj[p_col], errors='coerce')
gdf_proj[n_col] = pd.to_numeric(gdf_proj[n_col], errors='coerce').fillna(0)

gdf_nodata = gdf_proj[gdf_proj[n_col] == 0]
gdf_valid  = gdf_proj[gdf_proj[n_col] > 0].copy()
gdf_nonsig = gdf_valid[gdf_valid[p_col] >= 0.05]

# ============================================================================
# 4. 构建画布与绘图
# ============================================================================
fig = plt.figure(figsize=(8.4 / 2.54, 6.5 / 2.54), dpi=300)
ax = fig.add_axes([0.02, 0.15, 0.96, 0.82], projection=target_proj)

for spine in ax.spines.values():
    spine.set_visible(False)

# 颜色映射 (以 0 为中心)
valid_r = gdf_valid[r_col].dropna()
max_r = np.percentile(np.abs(valid_r), 98) if len(valid_r) > 0 else 0.5

cmap = plt.cm.RdBu_r
norm = mpl.colors.TwoSlopeNorm(vmin=-max_r, vcenter=0, vmax=max_r)

# ──────────────────────────────────────────────────────────────────────────
# 💡【绘图 1】：有数据的区域 (根据 Partial_r 上色)
if not gdf_valid.empty:
    gdf_valid.plot(
        column=r_col,
        cmap=cmap,
        norm=norm,
        ax=ax,
        linewidth=0.3,
        edgecolor='#333333',
        zorder=2
    )

# 💡【绘图 2】：N_samples == 0 的区域 (深灰色填充)
if not gdf_nodata.empty:
    gdf_nodata.plot(
        ax=ax,
        facecolor='#4A4A4A',
        edgecolor='#333333',
        linewidth=0.3,
        zorder=3
    )

# 💡【绘图 3】：不显著区域 (p >= 0.05) 叠加斜线阴影
if not gdf_nonsig.empty:
    gdf_nonsig.plot(
        ax=ax,
        facecolor='none',  # 透明
        hatch='////',      # 斜线
        edgecolor='#333333',
        linewidth=0.3,
        zorder=4
    )

# 自动定位到数据的包络范围，防止边界错位导致空图
bounds = gdf_proj.total_bounds  # [xmin, ymin, xmax, ymax]
ax.set_extent([bounds[0], bounds[2], bounds[1], bounds[3]], crs=target_proj)

# ============================================================================
# 5. Colorbar & Legend
# ============================================================================
cax = fig.add_axes([0.15, 0.08, 0.70, 0.025])

sm_map = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm_map._A = []

cb = fig.colorbar(sm_map, cax=cax, orientation="horizontal", extend="both")
cb.set_label("Partial correlation ($r$)", labelpad=3, fontsize=8)
cb.ax.tick_params(labelsize=7.5, width=0.5, length=2)
cb.outline.set_linewidth(0.4)

legend_elements = [
    Patch(facecolor='#4A4A4A', edgecolor='#333333', linewidth=0.4, label='Data Shortage'),
    Patch(facecolor='white', edgecolor='#333333', hatch='////', linewidth=0.4, label='Not Significant')
]

ax.legend(
    handles=legend_elements,
    loc='lower left',
    frameon=False,
    framealpha=0.8,
    facecolor='white',
    edgecolor='none',
    # fontsize=7,
    handlelength=1.2,
    handleheight=0.9,
    borderpad=0,
    labelspacing=0.1
)

# 保存与展示
plt.savefig("./Fig2_c_conifer.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.show()