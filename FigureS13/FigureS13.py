import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
from matplotlib.lines import Line2D
import geopandas as gpd
from libpysal.weights import KNN, lag_spatial

warnings.filterwarnings('ignore')

# =============================================================================
# 1. 全局样式设置 (Nature 期刊风格)
# =============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("white")
rcParams['font.family'] = 'Arial'  # 论文常用字体
rcParams['font.size'] = 9  # 图片内字体大小
rcParams['axes.labelsize'] = 9  # 坐标轴标签
rcParams['xtick.labelsize'] = 7.5  # X轴刻度
rcParams['ytick.labelsize'] = 7.5  # Y轴刻度

# =============================================================================
# 2. 数据读取、去重与标准化
# =============================================================================
data_path = "D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/new/all_plots_environment_supersection.csv"
df_input = pd.read_csv(data_path)

model_vars = [
    'PC1_new', 'C_N_x', 'soilN_x', 'sand_x', 'SLOPE_x', 'totalNdepo_x',
    'MAT_x', 'AridityInd_x', 'STDAGE_x', 'speciesRic_x', 'conifer_pr_x',
    'EcM_prop_x', 'AM_prop_x', 'SSection', 'Tseasonali_x', 'geometry_x'
]

df_model = df_input[model_vars].dropna().copy()

# 空间去重
dup_mask = df_model.duplicated(subset=['SSection', 'geometry_x'], keep='first')
if dup_mask.sum() > 0:
    df_model = df_model[~dup_mask].copy().reset_index(drop=True)

# 标准化连续变量
scale_vars = [
    'C_N_x', 'soilN_x', 'sand_x', 'SLOPE_x', 'totalNdepo_x', 'MAT_x',
    'AridityInd_x', 'STDAGE_x', 'speciesRic_x', 'EcM_prop_x', 'conifer_pr_x', 'Tseasonali_x'
]
scaler = StandardScaler()
df_model[scale_vars] = scaler.fit_transform(df_model[scale_vars])

# =============================================================================
# 3. 计算空间滞后项 (Spatial Lag) 并拟合 SAR-LMM
# =============================================================================
if isinstance(df_model['geometry_x'].iloc[0], str):
    gdf_model = gpd.GeoDataFrame(df_model, geometry=gpd.GeoSeries.from_wkt(df_model['geometry_x']), crs="EPSG:4326")
else:
    gdf_model = gpd.GeoDataFrame(df_model, geometry=df_model['geometry_x'], crs="EPSG:4326")

w = KNN.from_dataframe(gdf_model, k=8)
w.transform = 'R'

# 计算并标准化空间滞后项
df_model['PC1_new_spatial_lag'] = lag_spatial(w, df_model['PC1_new'].values)
scaler_lag = StandardScaler()
df_model['PC1_new_spatial_lag_scaled'] = scaler_lag.fit_transform(df_model[['PC1_new_spatial_lag']])

# SAR-LMM 模型公式
formula_sar = (
    "PC1_new ~ EcM_prop_x * Tseasonali_x + "
    "EcM_prop_x * totalNdepo_x + EcM_prop_x * C_N_x + EcM_prop_x * AridityInd_x + "
    "EcM_prop_x * MAT_x + soilN_x + sand_x + SLOPE_x + "
    "EcM_prop_x * STDAGE_x + EcM_prop_x * speciesRic_x + conifer_pr_x + "
    "PC1_new_spatial_lag_scaled"
)

model_sar = MixedLM.from_formula(
    formula=formula_sar,
    data=df_model,
    groups="SSection",
    re_formula="~EcM_prop_x"
)

result = model_sar.fit(reml=False)

# =============================================================================
# 4. 提取系数并过滤（剔除截距、随机效应及空间滞后项）
# =============================================================================
conf_int = result.conf_int()
summary_df = pd.DataFrame({
    'Estimate': result.params,
    'CI_lower': conf_int[0],
    'CI_upper': conf_int[1],
    'pvalue': result.pvalues
})

# 剔除截距、随机效应以及空间滞后项
drop_pattern = 'Intercept|Group|Var|Cov|PC1_new_spatial_lag_scaled'
summary_df = summary_df.loc[~summary_df.index.str.contains(drop_pattern, case=False)].copy()

# 变量标签映射
label_dict = {
    'EcM_prop_x': 'EcM Dominance',
    'Tseasonali_x': 'T Seasonality',
    'totalNdepo_x': 'N Deposition',
    'C_N_x': 'Soil C:N',
    'AridityInd_x': 'Aridity index',
    'MAT_x': 'MAT',
    'soilN_x': 'Soil N',
    'sand_x': 'Sand Content',
    'SLOPE_x': 'Slope',
    'STDAGE_x': 'Forest Age',
    'speciesRic_x': 'Species Richness',
    'conifer_pr_x': 'Conifer proportion',
    'EcM_prop_x:Tseasonali_x': 'EcM × T Seasonality',
    'EcM_prop_x:totalNdepo_x': 'EcM × N Deposition',
    'EcM_prop_x:C_N_x': 'EcM × Soil C:N',
    'EcM_prop_x:AridityInd_x': 'EcM × Aridity index',
    'EcM_prop_x:MAT_x': 'EcM × MAT',
    'EcM_prop_x:STDAGE_x': 'EcM × Forest Age',
    'EcM_prop_x:speciesRic_x': 'EcM × Species Richness',
}

summary_df['Clean_Label'] = [label_dict.get(var, var) for var in summary_df.index]
summary_df['is_sig'] = summary_df['pvalue'] < 0.05

# 重新排序：非交互项在左，EcM 及其交互项在右
is_ecm_group = summary_df.index.str.contains('EcM_prop_x', case=False)
df_left = summary_df[~is_ecm_group]
df_right = summary_df[is_ecm_group]

plot_df = pd.concat([df_left, df_right]).reset_index(drop=True)

num_left = len(df_left)
total_vars = len(plot_df)

# =============================================================================
# 5. 绘图 (X 轴为变量，Y 轴为效应量)
# =============================================================================
fig, ax = plt.subplots(figsize=(16.8 / 2.54, 7.5 / 2.54), dpi=300)

x_positions = np.arange(total_vars)

# ── 1. 右侧 (EcM 及其交互项) 添加浅红色背景区域 ───────────────────────────────
ax.axvspan(num_left - 0.5, total_vars - 0.5, color='#FDF0F0', zorder=0)

# ── 2. Y=0 参考线 ─────────────────────────────────────────────────────────────
ax.axhline(y=0, color='#666666', linestyle='--', linewidth=0.8, zorder=1)

# ── 3. 绘制垂直置信区间与散点 ──────────────────────────────────────────────────
for i, row in plot_df.iterrows():
    err_color = '#1A1A1A' if row['is_sig'] else '#888888'

    # Y 轴方向置信区间
    ax.plot(
        [i, i], [row['CI_lower'], row['CI_upper']],
        color=err_color, linewidth=1.2, zorder=2
    )

    # 显著：实心；不显著：空心
    if row['is_sig']:
        ax.plot(
            i, row['Estimate'], marker='o', markersize=7.5,
            color='#888888', markeredgecolor='#333333', markeredgewidth=1.0, zorder=3
        )
    else:
        ax.plot(
            i, row['Estimate'], marker='o', markersize=7.5,
            color='white', markeredgecolor='#333333', markeredgewidth=1.2, zorder=3
        )

# ── 4. 坐标轴与标签美化 ──────────────────────────────────────────────────────
ax.set_xticks(x_positions)
ax.set_xticklabels(plot_df['Clean_Label'], rotation=45, ha='right', fontsize=8.5)

ax.set_ylabel('Standardized Effect Size (95% CI)', fontsize=9.5)
ax.set_xlim(-0.6, total_vars - 0.4)

# 移除上方和右侧边框
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['left'].set_linewidth(0.6)
ax.spines['bottom'].set_linewidth(0.6)

# ── 5. 添加图例 ───────────────────────────────────────────────────────────────
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='p < 0.05',
           markerfacecolor='#888888', markeredgecolor='#333333', markersize=5.5),
    Line2D([0], [0], marker='o', color='w', label='p ≥ 0.05',
           markerfacecolor='white', markeredgecolor='#333333', markeredgewidth=1.2, markersize=5.5)
]

ax.legend(
    handles=legend_elements,
    loc='lower left',
    frameon=False,
    framealpha=0.85,
    facecolor='white',
    edgecolor='none'
)

# plt.tight_layout()
fig.savefig("./FigS10.pdf", format="pdf", dpi=300, bbox_inches="tight")
fig.savefig("./FigS10.jpg", format="jpg", dpi=300, bbox_inches="tight")
plt.show()