import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import xgboost as xgb
import shap
import seaborn as sns

# 设置全局字体与样式 (符合 Nature/Science 期刊出版规范)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("white")
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 9

# ==========================================
# 1. 数据读取与模型拟合
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

model_xgb = xgb.XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    colsample_bytree=1.0, min_child_weight=1, subsample=0.7,
    random_state=42, n_jobs=-1
)
model_xgb.fit(X, y)

# ==========================================
# 2. 计算 SHAP 相对贡献度与二分化合并
# ==========================================
explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer(X)
shap_imp = np.abs(shap_values.values).mean(axis=0)

df_importance = pd.DataFrame({
    'Variable': scale_vars,
    'SHAP_Importance': shap_imp
})

# 计算总体与 EcM Dominance 的 SHAP 值
total_shap = df_importance['SHAP_Importance'].sum()
ecm_shap = df_importance.loc[df_importance['Variable'] == 'EcM_prop_x', 'SHAP_Importance'].values[0]

# 二分占比 (%)
ecm_pct = (ecm_shap / total_shap) * 100
others_pct = 100.0 - ecm_pct

# 数据列表
pct_values = [ecm_pct, others_pct]
labels = ['EcM Dominance', 'Others']

# 配色：EcM 高亮深红/蓝，Others 使用中灰色
colors = ['#2166AC', '#E0E0E0']

# ==========================================
# 3. 绘制无图例二分环图 (设置背景透明)
# ==========================================
# 💡 设置 facecolor='none' 确保画布透明
fig, ax = plt.subplots(figsize=(5.0 / 2.54, 5.0 / 2.54), dpi=300, facecolor='none')
ax.set_facecolor('none')

wedges, texts, autotexts = ax.pie(
    pct_values,
    labels=None,  # 移除扇区标记
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    pctdistance=0.75,
    wedgeprops=dict(width=0.35, edgecolor='white', linewidth=1.5)  # 空心圆环控制
)

# 格式化扇区文字（扇区内只保留格式化百分比或留白）
plt.setp(autotexts[0], size=8, weight="bold", color="none")       # EcM 扇区内部文字
plt.setp(autotexts[1], size=8, weight="bold", color="none")     # Others 扇区内部文字

# ------------------------------------------
# 4. 中心直接展示核心占比信息
# ------------------------------------------
ax.text(
    0, 0.10, 'EcM Dominance',
    ha='center', va='center', fontsize=8.5, fontweight='bold', color='#2166AC'
)
ax.text(
    0, -0.12, f'{ecm_pct:.1f}%',
    ha='center', va='center', fontsize=12, fontweight='bold', color='#333333'
)

plt.tight_layout()

# 💡 导出关键参数：transparent=True, facecolor='none', edgecolor='none'
fig.savefig(
    "./Fig2_f.pdf",
    format="pdf",
    dpi=300,
    bbox_inches="tight",
    transparent=True,    # 核心：设置导出透明背景
    facecolor='none',    # 核心：将画布背景设为无色
    edgecolor='none'
)

plt.show()