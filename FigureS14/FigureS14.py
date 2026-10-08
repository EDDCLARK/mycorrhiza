import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import statsmodels.api as sm

# ==========================================
# 0. 变量名映射字典 (统一修改为 EcM tree proportion)
# ==========================================
var_name_map = {
    'EcM_prop_x': 'EcM tree proportion',
    'EcM_prop': 'EcM tree proportion',
    'Tseasonali_x': 'T Seasonality',
    'Tseasonali': 'T Seasonality',
    'totalNdepo_x': 'N Deposition',
    'totalNdepo': 'N Deposition',
    'C_N_x': 'Soil C:N',
    'C_N': 'Soil C:N',
    'AridityInd_x': 'Aridity index',
    'AridityInd': 'Aridity index',
    'MAT_x': 'MAT',
    'MAT': 'MAT',
    'soilN_x': 'Soil N',
    'soilN': 'Soil N',
    'sand_x': 'Sand Content',
    'sand': 'Sand Content',
    'SLOPE_x': 'Slope',
    'SLOPE': 'Slope',
    'STDAGE_x': 'Forest Age',
    'STDAGE': 'Forest Age',
    'speciesRic_x': 'Species Richness',
    'speciesRic': 'Species Richness',
    'conifer_pr_x': 'Conifer proportion',
    'conifer_pr': 'Conifer proportion'
}

# ==========================================
# 1. Nature 出版级样式设置
# ==========================================
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 8
rcParams['axes.titlesize'] = 8.5
rcParams['axes.labelsize'] = 8
rcParams['xtick.labelsize'] = 7
rcParams['ytick.labelsize'] = 7
rcParams['legend.fontsize'] = 6.5
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['axes.linewidth'] = 0.6
rcParams['xtick.major.width'] = 0.6
rcParams['ytick.major.width'] = 0.6

# ==========================================
# 2. 读取数据与列名自动诊断
# ==========================================
csv_path = "D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/new/all_plots_with_SHAP.csv"
df = pd.read_csv(csv_path)

# 定位 EcM 变量列
ecm_col = 'EcM_prop' if 'EcM_prop' in df.columns else 'EcM_prop_x'

# 自动匹配 EcM 的主效应 SHAP 列
shap_ecm_candidates = [c for c in df.columns if
                       ('SHAP' in c or 'shap' in c) and ecm_col in c and 'inter' not in c.lower()]
if not shap_ecm_candidates:
    shap_ecm_candidates = [c for c in df.columns if
                           'SHAP' in c and ('EcM' in c or 'ecm' in c) and 'inter' not in c.lower()]

shap_ecm_col = shap_ecm_candidates[0] if shap_ecm_candidates else None

if not shap_ecm_col:
    raise SystemExit("❌ 错误：未能在 CSV 中匹配到 EcM 的主效应 SHAP 列！")

# ==========================================
# 3. 提取数据并绘图
# ==========================================
plot_df = df.dropna(subset=[ecm_col, shap_ecm_col]).copy()
x_vals = plot_df[ecm_col].values
y_vals = plot_df[shap_ecm_col].values

fig, ax = plt.subplots(figsize=(4.0, 3.2), dpi=300)

# 获取统一显示名称: 'EcM tree proportion'
ecm_display_name = var_name_map.get(ecm_col, 'EcM tree proportion')

# 1. 绘制散点图 (颜色 c 直接设置为 x_vals，即 EcM tree proportion)
scatter = ax.scatter(
    x_vals,
    y_vals,
    c=x_vals,
    cmap='viridis',
    alpha=0.7,
    s=10,
    edgecolor='none'
)

# 添加 Colorbar，名称与 X 轴保持完全一致
cbar = fig.colorbar(scatter, ax=ax, pad=0.03, fraction=0.046)
cbar.set_label(ecm_display_name, fontweight='bold', fontsize=7.5)
cbar.ax.tick_params(labelsize=6.5)

# 2. 0 刻度参考线
ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.6, zorder=1)

# 3. LOWESS 非参数趋势线
lowess = sm.nonparametric.lowess(y_vals, x_vals, frac=0.35)
ax.plot(lowess[:, 0], lowess[:, 1], color='#D55E00', linewidth=1.8, label='LOWESS Trend', zorder=3)

# 4. 坐标轴与标签设置
ax.set_xlabel(ecm_display_name, fontweight='bold')
ax.set_ylabel(f"SHAP value for {ecm_display_name}", fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.grid(True, linestyle='--', alpha=0.2, color='gray')
ax.legend(frameon=False, loc='best')

plt.tight_layout()
fig.savefig("./FigS12.jpg", format="jpg", dpi=300, bbox_inches="tight")
plt.show()