import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import xgboost as xgb
import shap
import seaborn as sns
from sklearn.metrics import r2_score

# 设置全局字体与样式 (符合 Nature/Science 期刊出版规范)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("white")
rcParams['font.family'] = 'Arial'  # 论文常用字体
rcParams['font.size'] = 9  # 图片内字体大小（五号字=10.5，图片内用8-9）
rcParams['axes.labelsize'] = 9  # 坐标轴标签稍大
rcParams['xtick.labelsize'] = 9  # X轴刻度
rcParams['ytick.labelsize'] = 9  # Y轴刻度

# ==========================================
# 1. 数据读取与准备 (沿用你的数据路径与变量)
# ==========================================
df_input = pd.read_csv("D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/new/all_plots_environment_supersection.csv")

target_y = 'PC1_new'
scale_vars = ['C_N_x', 'soilN_x', 'sand_x', 'SLOPE_x', 'totalNdepo_x',
              'MAT_x', 'AridityInd_x', 'STDAGE_x', 'speciesRic_x',
              'EcM_prop_x', 'Tseasonali_x']




model_vars = [target_y] + scale_vars
df_model = df_input[model_vars].dropna().copy()

X = df_model[scale_vars]
y = df_model[target_y]

# ==========================================
# 2. 拟合 XGBoost 回归模型
# ==========================================
model_xgb = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    colsample_bytree=1.0,
    min_child_weight=1,
    subsample=0.7,
    random_state=42,
    n_jobs=-1
)
model_xgb.fit(X, y)

# 2.1 提取 Gini / Feature Importance (基于 Gain 或 Weight)
gini_imp = model_xgb.feature_importances_

# ==========================================
# 3. 计算 SHAP 值与 SHAP Importance
# ==========================================
explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer(X)

# 计算每个变量的平均绝对 SHAP 值 (Mean |SHAP value|)
shap_imp = np.abs(shap_values.values).mean(axis=0)

# ==========================================
# 4. 数据汇总与自定义变量名映射
# ==========================================
df_importance = pd.DataFrame({
    'Variable': scale_vars,
    'Gini_Importance': gini_imp,
    'SHAP_Importance': shap_imp
})

# 定义自定义变量名映射
name_mapping = {
    'MAT_x': 'MAT',
    'AridityInd_x': 'Aridity Index',
    'Tseasonali_x': 'T Seasonality',
    'totalNdepo_x': 'N Deposition',
    'C_N_x': 'Soil C:N',
    'soilN_x': 'Soil N',
    'sand_x': 'Sand Content',
    'SLOPE_x': 'Slope',
    'STDAGE_x': 'Forest Age',
    'speciesRic_x': 'Species Richness',
    'EcM_prop_x': 'EcM Dominance'
}

df_importance['Display_Name'] = df_importance['Variable'].map(name_mapping).fillna(
    df_importance['Variable'].str.replace('_x$', '', regex=True)
)

# 按 SHAP 重要性从大到小排序 (便于作图时从下到上呈阶梯状)
df_importance = df_importance.sort_values(by='SHAP_Importance', ascending=True).reset_index(drop=True)

# ==========================================
# 5. 双水平柱状图绘制 (固定主图比例，不压缩绘图区)
# ==========================================
# fig = plt.figure(figsize=(6 / 2.54, 7.5 / 2.54), dpi=300)
fig, ax = plt.subplots(figsize=(6 / 2.54, 7.5 / 2.54), dpi=300)
# 使用 fig.add_axes 绝对固定主图位置：[left, bottom, width, height]
# left=0.42 给左侧长变量名留足 42% 空间；width=0.53 确保主图宽度绝对固定为 53%
# ax = fig.add_axes([0.42, 0.12, 0.53, 0.83])

y_positions = np.arange(len(df_importance))
h = 0.35  # 柱子高度

color_shap = "#2166AC"  #  SHAP Importance
color_gini = "#92C5DE"  #  Gini Importance

# 归一化/标准显示 (防止 Gini 与 SHAP 数量级不同导致对比困难，这里保留各自原值)
# --- 绘制 1: SHAP Importance (上方柱子: y - h/2) ---
ax.barh(
    y_positions - h/2,
    df_importance['SHAP_Importance'],
    height=h,
    color=color_shap,
    edgecolor="none",
    zorder=3
)

# --- 绘制 2: Gini Importance (下方柱子: y + h/2) ---
ax.barh(
    y_positions + h/2,
    df_importance['Gini_Importance'],
    height=h,
    color=color_gini,
    edgecolor="none",
    zorder=3
)

# ==========================================
# 6. 坐标轴与细节美化 (Nature/Science 风格)
# ==========================================
ax.set_yticks(y_positions)
ax.set_yticklabels(df_importance['Display_Name'], ha='right')

ax.set_xlabel("Feature importance", labelpad=5)

# 动态扩展 X 轴边界
max_val = max(df_importance['SHAP_Importance'].max(), df_importance['Gini_Importance'].max())
ax.set_xlim(0, max_val * 1.1)

# 0 刻度基线
ax.axvline(0, color="#333333", linestyle="-", linewidth=0.8, zorder=4)

# 隐藏多余边框
# for spine in ["top", "right", "left"]:
#     ax.spines[spine].set_visible(False)

ax.spines["bottom"].set_linewidth(0.8)
ax.spines["bottom"].set_color("#333333")

ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", width=0.8, direction="out", colors="#333333")
ax.grid(axis='x', linestyle=':', alpha=0.5, zorder=0)
ax.set_xlim(0, 0.8)

# 图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=color_shap, label='SHAP Importance'),
    Patch(facecolor=color_gini, label='Gini Importance')
]

ax.legend(
    handles=legend_elements,
    frameon=False,
    # fontsize=7,
    loc="lower right",
    handlelength=1.0,
    handleheight=1.0,
    borderpad=0.2,
    labelspacing=0.3
)

# 保存图片时使用 bbox_inches='tight' 配合 add_axes，彻底防止标签裁剪和图片变形
# fig.savefig("./Fig2_b.pdf",format="pdf", dpi=300, bbox_inches="tight")
plt.show()