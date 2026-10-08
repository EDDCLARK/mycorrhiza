import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import seaborn as sns

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

# ============================================================================
# 2. 读取 Shapefile 并转换投影
# ============================================================================
shp_path = r"D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/new/partial_r_supersection.shp"
gdf = gpd.read_file(shp_path)

if gdf.crs is None:
    gdf.set_crs(epsg=4326, inplace=True)

target_proj = ccrs.AlbersEqualArea(
    central_longitude=-96,
    central_latitude=40,
    standard_parallels=(20, 60),
    false_easting=0,
    false_northing=0,
    globe=ccrs.Globe(datum='NAD83', ellipse='GRS80')
)

gdf_proj = gdf.to_crs(target_proj.proj4_init)

# ============================================================================
# 3. 匹配 SSection 列名与离散分类配色
# ============================================================================
col_mapping = {c.lower(): c for c in gdf_proj.columns}
div_col = col_mapping.get('ssection', 'SSection')

# 确保转换为字符串并去除缺失值
gdf_proj[div_col] = gdf_proj[div_col].astype(str)

# 获取唯一生态区名称并排序
unique_divisions = sorted(gdf_proj[div_col].unique())
num_categories = len(unique_divisions)

# 根据分类数量自动生成高对比度离散色板
palette = sns.color_palette("tab20", n_colors=num_categories)

# ============================================================================
# 4. 构建画布与地图绘制
# ============================================================================
# 2 列图例纵向占用空间略多一点，因此调高整体画布纵横比
fig = plt.figure(figsize=(10.0 / 2.54, 9.5 / 2.54), dpi=300)

# 💡【布局调整】：底部留出 28% 的空间容纳 2 列图例 [left, bottom, width, height]
ax = fig.add_axes([0.02, 0.28, 0.96, 0.69], projection=target_proj)

for spine in ax.spines.values():
    spine.set_visible(False)

# ── 按 SSection 唯一值绘图 ────────────────────────────────────────────────
gdf_proj.plot(
    column=div_col,
    categorical=True,
    legend=True,
    legend_kwds={
        'loc': 'upper center',
        'bbox_to_anchor': (0.5, -0.05),  # 放置在地图正下方
        'ncol': 2,                       # 💡 改为 2 列展示
        'frameon': False,
        'title': 'Ecological Supersection',
        'title_fontsize': 8.5,
        'fontsize': 7.2,
        'markerscale': 0.8,
        'labelspacing': 0.35,            # 行间距
        'columnspacing': 1.5             # 💡 增加两列之间的间距
    },
    cmap=mpl.colors.ListedColormap(palette),
    ax=ax,
    linewidth=0.4,
    edgecolor='#333333',
    zorder=2
)

# 自动调整边界范围
map_bounds = gdf_proj.total_bounds
ax.set_extent([map_bounds[0], map_bounds[2], map_bounds[1], map_bounds[3]], crs=target_proj)

# 保存与展示
plt.savefig("./FigS3_b.jpg", format="jpg", dpi=300, bbox_inches="tight")
plt.show()