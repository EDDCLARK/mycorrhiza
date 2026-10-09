import geopandas as gpd
import pandas as pd
from shapely import wkt

# 1. 读取 CSV 文件
csv_path = "D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/new/all_plots_with_masked_Trait_values_with_PC_scores.csv"
df = pd.read_csv(csv_path)

# 2. 将 geometry 列转换为 shapely 空间几何对象（如果 geometry 已经是 WKT 文本格式）
if isinstance(df['geometry'].iloc[0], str):
  df['geometry'] = df['geometry'].apply(wkt.loads)

# 3. 构建 GeoDataFrame 并指定坐标系 (常用 WGS84 即 EPSG:4326)
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')

# 4. 导出为 Shapefile (.shp)
shp_path = "D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/new/all_plots_with_masked_Trait_values_with_PC_scores.shp"
gdf.to_file(shp_path, encoding='utf-8')

print(f'成功导出 Shapefile 至: {shp_path}')