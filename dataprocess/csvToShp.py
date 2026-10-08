import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape

# 1. 读取 CSV 文件
csv_path = "D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/All_plot_addConifer.csv"
output_shp_path = "D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/All_plot_addConifer.shp"

df = pd.read_csv(csv_path)

# 2. 解析 .geo 列提取 Shapely Geometry 对象
def parse_geo(geo_str):
    try:
        if pd.isna(geo_str):
            return None
        # 如果列里的字符串是用单引号或包含 json 字符串，转为 dict 字典
        geo_dict = json.loads(geo_str) if isinstance(geo_str, str) else geo_str
        # shapely.geometry.shape 可以直接将 GeoJSON 字典转为 Point 对象
        return shape(geo_dict)
    except Exception as e:
        return None

# 将 .geo 列转换为 Shapely 的 Geometry 几何列
geometries = df['.geo'].apply(parse_geo)

# 3. 创建 GeoDataFrame (默认坐标系通常为 WGS84 / EPSG:4326)
# 同时删除原有的 '.geo' 文本列，避免导出 shp 时出现复杂的 json 文本字段
df_clean = df.drop(columns=['.geo'])
gdf = gpd.GeoDataFrame(df_clean, geometry=geometries, crs="EPSG:4326")

# 4. 过滤掉几何对象为空 (NaN) 的无效行
gdf = gdf[gdf.geometry.notna()]

# 5. 导出为 Shapefile
# 注意：Shapefile 属性字段名最长只支持 10 个字符，GeoPandas 会自动帮你做截断警告
gdf.to_file(output_shp_path, driver="ESRI Shapefile", encoding="utf-8")

print(f"成功导出 Shapefile 文件至: {output_shp_path}")