import numpy as np
import geopandas as gpd

# Shapefile 文件路径
shp_path = "D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/All_plot_addConifer.shp"

# 1. 读取 Shapefile
gdf = gpd.read_file(shp_path)

# 2. 将数值 -9999 和字符串 "-9999" 统一替换为 np.nan
gdf = gdf.replace([-9999, "-9999", -9999.0], np.nan)

# 3. 移除属性表中包含 NaN 的行
# how='any': 只要任意一列有 NaN 就删除该行
# （注意：geometry 列如果不为空，不会被 dropna 误删）
gdf_clean = gdf.dropna(how='any').copy()

# 4. 覆盖写入更新原 Shapefile
gdf_clean.to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")

print(f"处理完成！原 Shapefile 已更新。清理前行数: {len(gdf)}，清理后行数: {len(gdf_clean)}")