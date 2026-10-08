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
shp_path = r"D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/new/partial_r_division.shp"
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
# 3. 匹配 DIVISION_N 与 DIVISION_C 列名
# ============================================================================
col_mapping = {c.lower(): c for c in gdf_proj.columns}
div_n_col = col_mapping.get('division_n', 'DIVISION_N')
div_c_col = col_mapping.get('division_c', 'DIVISION_C')

gdf_proj[div_n_col] = gdf_proj[div_n_col].astype(str)
gdf_proj[div_c_col] = gdf_proj[div_c_col].astype(str)

# 获取 DIVISION_N 唯一值并生成离散色板
unique_divisions = sorted(gdf_proj[div_n_col].unique())
num_divisions = len(unique_divisions)

palette = sns.color_palette("tab20", n_colors=num_divisions) if num_divisions > 10 else sns.color_palette("Set3",
                                                                                                          n_colors=num_divisions)

# ============================================================================
# 4. 构建画布与地图绘制
# ============================================================================
fig = plt.figure(figsize=(11.0 / 2.54, 8.0 / 2.54), dpi=300)
ax = fig.add_axes([0.02, 0.05, 0.68, 0.90], projection=target_proj)

for spine in ax.spines.values():
    spine.set_visible(False)

# ── 1. 按 DIVISION_N 唯一值填充颜色 ─────────────────────────────────────────
gdf_proj.plot(
    column=div_n_col,
    categorical=True,
    legend=True,
    legend_kwds={
        'loc': 'center left',
        'bbox_to_anchor': (1.02, 0.5),  # 放置于右侧图例栏
        'frameon': False,
        'title': 'Ecological Division',
        'title_fontsize': 8.5,
        'fontsize': 7.5,
        'markerscale': 0.8,
        'labelspacing': 0.3
    },
    cmap=mpl.colors.ListedColormap(palette),
    ax=ax,
    linewidth=0.4,
    edgecolor='#333333',
    zorder=2
)

# ── 2. 在每个 Division 的几何质心上叠加 DIVISION_C 文本 ──────────────────────
for idx, row in gdf_proj.iterrows():
    label_text = row[div_c_col]

    if pd.isna(label_text) or label_text.lower() == 'nan':
        continue

    # 获取投影后的几何质心坐标
    centroid = row.geometry.centroid

    ax.text(
        centroid.x, centroid.y,
        label_text,
        fontsize=6.5,
        fontweight='bold',
        color='#111111',
        ha='center',
        va='center',
        zorder=5,
        bbox=dict(
            boxstyle='round,pad=0.15',
            facecolor='white',
            edgecolor='none',
            alpha=0.65
        )
    )

# 自动调整边界范围
map_bounds = gdf_proj.total_bounds
ax.set_extent([map_bounds[0], map_bounds[2], map_bounds[1], map_bounds[3]], crs=target_proj)

# 保存与展示
plt.savefig("./FigS3_a.jpg", format="jpg", dpi=300, bbox_inches="tight")
plt.show()