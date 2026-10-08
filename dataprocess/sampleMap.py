import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# =============================================================================
# 1. 路径设置与配置
# =============================================================================
data_dir = "D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/grids/randomsamples/downsampled_datasets"


# 网格与下采样比例矩阵
grid_scales = ["0d5deg", "1deg", "2deg"]
n_counts = [3, 5, 7]

# 创建 3x3 画布
fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.2, wspace=0.1)

# 全局美化设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 正常显示中文/英文
plt.rcParams['axes.unicode_minus'] = False

print("🎨 开始绘制 9 个下采样数据集的空间分布图...")

# =============================================================================
# 2. 循环读取 9 个数据并绘制子图
# =============================================================================
for row_idx, n_keep in enumerate(n_counts):
    for col_idx, res_name in enumerate(grid_scales):
        ax = axes[row_idx, col_idx]

        filename = f"all_plots_downsample_{res_name}_n{n_keep}.csv"
        filepath = os.path.join(data_dir, filename)

        if not os.path.exists(filepath):
            ax.text(0.5, 0.5, f"未找到文件:\n{filename}", ha='center', va='center', transform=ax.transAxes)
            continue

        # 读取 CSV 并转为 GeoDataFrame
        df = pd.read_csv(filepath)
        if isinstance(df['geometry_x'].iloc[0], str):
            gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df['geometry_x']), crs="EPSG:4326")
        else:
            gdf = gpd.GeoDataFrame(df, geometry=df['geometry_x'], crs="EPSG:4326")

        # 提取经纬度绘制散点
        lons = gdf.geometry.x
        lats = gdf.geometry.y

        # 绘制样地点（使用半透明小红点）
        ax.scatter(lons, lats, s=1.2, c='#d62728', alpha=0.4, edgecolors='none')

        # 设置子图标题与刻度
        n_plots = len(gdf)
        ax.set_title(f"格网: {res_name} | 上限: N={n_keep}\n(总计样地点数: {n_plots:,})", fontsize=11,
                     fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)

        # 外围坐标轴标签设置
        if row_idx == 2:
            ax.set_xlabel("Longitude (°E)", fontsize=10)
        if col_idx == 0:
            ax.set_ylabel("Latitude (°N)", fontsize=10)

# 设置总标题
fig.suptitle("FIA Forest Plots Spatial Distribution across Downsampling Grids & Cap Levels",
             fontsize=16, fontweight='bold', y=0.98)

# 保存高分辨率图像
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.show()