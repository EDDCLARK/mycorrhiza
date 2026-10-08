import os
import pandas as pd
import geopandas as gpd

# =============================================================================
# 1. 路径设置与数据读取
# =============================================================================
csv_path = "D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/new/all_plots_environment_supersection.csv"
grid_dir = "D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/grids"
output_dir = "D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/grids/randomsamples/downsampled_datasets"

os.makedirs(output_dir, exist_ok=True)

# 1) 读取 CSV 点数据并转为 GeoDataFrame (WGS84 / EPSG:4326)
df_input = pd.read_csv(csv_path)

if isinstance(df_input['geometry_x'].iloc[0], str):
    gdf_plots = gpd.GeoDataFrame(
        df_input,
        geometry=gpd.GeoSeries.from_wkt(df_input['geometry_x']),
        crs="EPSG:4326"
    )
else:
    gdf_plots = gpd.GeoDataFrame(
        df_input,
        geometry=df_input['geometry_x'],
        crs="EPSG:4326"
    )

print(f"✅ 点数据读取完成，总计样地点数: {len(gdf_plots)}")

# 2) 自动化匹配 0.5°、1°、2° 的 shp 文件路径
grid_files = {
    "0d5deg": None,
    "1deg": None,
    "2deg": None
}

for file in os.listdir(grid_dir):
    if file.endswith('.shp'):
        full_path = os.path.join(grid_dir, file)
        if "0d5" in file:
            grid_files["0d5deg"] = full_path
        elif "1" in file and "0d5" not in file:
            grid_files["1deg"] = full_path
        elif "2" in file:
            grid_files["2deg"] = full_path

# 采样数量规则
sample_counts = [3, 5, 7]

# =============================================================================
# 2. 循环处理各个格网与下采样
# =============================================================================
random_seed = 42  # 固定的随机种子，保证结果可重复

for res_name, shp_path in grid_files.items():
    if not shp_path or not os.path.exists(shp_path):
        print(f"⚠️ 警告: 未找到分辨率为 {res_name} 的 shp 文件，跳过此尺度。")
        continue

    print(f"\n" + "="*60)
    print(f"🌍 正在处理格网分辨率: [{res_name}] -> {os.path.basename(shp_path)}")
    print("="*60)

    # 读取格网 shp 并确保坐标系统一
    gdf_grid = gpd.read_file(shp_path)
    if gdf_grid.crs != gdf_plots.crs:
        gdf_grid = gdf_grid.to_crs(gdf_plots.crs)

    # 给格网赋予唯一的临时 ID 标识
    gdf_grid['grid_unique_id'] = gdf_grid.index

    # 空间连接: 判断样地点落入哪个网格 (predicate='within' 或 'intersects')
    joined = gpd.sjoin(gdf_plots, gdf_grid[['grid_unique_id', 'geometry']], how='inner', predicate='within')

    print(f"  - 成功落入网格的样地点数: {len(joined)}")

    # 对不同的采样数进行抽样
    for n_keep in sample_counts:
        # 定义采样逻辑：对每个网格组，取 min(网格内点数, n_keep)
        sampled_gdf = (
            joined.groupby('grid_unique_id', group_keys=False)
            .apply(lambda x: x.sample(n=min(len(x), n_keep), random_state=random_seed))
            .reset_index(drop=True)
        )

        # 剔除空间连接产生的临时列
        cols_to_drop = [c for c in ['grid_unique_id', 'index_right'] if c in sampled_gdf.columns]
        sampled_df = pd.DataFrame(sampled_gdf.drop(columns=cols_to_drop))

        # 保存为 CSV
        out_filename = f"all_plots_downsample_{res_name}_n{n_keep}.csv"
        out_filepath = os.path.join(output_dir, out_filename)
        sampled_df.to_csv(out_filepath, index=False)

        print(f"  ✅ [保留上限 {n_keep} 个/网格] 抽样后数据量: {len(sampled_df):>6d} 行 -> 已保存至: {out_filename}")

print("\n" + "="*60)
print(f"🎉 全部 9 个下采样数据集已成功生成并保存在:\n   {output_dir}")
print("="*60)