import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import r2_score
from matplotlib.patches import Patch

# 设置全局字体与样式
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 9

# ==========================================
# 1. 数据读取与准备
# ==========================================
df_input = pd.read_csv("D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/new/all_plots.csv")

target_y = 'PC1_new'
scale_vars = ['C_N', 'soilN', 'sand', 'SLOPE', 'totalNdepo',
              'MAT', 'AridityInd', 'STDAGE', 'speciesRic',
              'EcM_prop', 'Tseasonali','conifer_pr']

model_vars = [target_y] + scale_vars
# 过滤空值并保留原数据的 Index，以便精准合并
df_model = df_input.dropna(subset=model_vars).copy()

X = df_model[scale_vars]
y = df_model[target_y]

# ==========================================
# 2. 训练集与测试集划分
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ==========================================
# 3. 10折交叉验证与超参数调优
# ==========================================
param_grid = {
    'n_estimators': [200,500,750,1000],
    'max_depth': [3, 4, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'min_child_weight': [1, 3, 5]
}

base_xgb = xgb.XGBRegressor(random_state=42, n_jobs=-1)
cv_strategy = KFold(n_splits=10, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=base_xgb,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_


# 获取 10 折交叉验证下的最佳平均 R2 及标准差
best_index = grid_search.best_index_
cv_r2_mean = grid_search.cv_results_['mean_test_score'][best_index]
cv_r2_std = grid_search.cv_results_['std_test_score'][best_index]

# 在独立测试集上评估 R2
y_pred_test = best_model.predict(X_test)
test_r2 = r2_score(y_test, y_pred_test)

# 输出评估结果
print("=" * 50)
print("【XGBoost 最优参数组合】:")
for param, val in best_params.items():
    print(f"  - {param}: {val}")

print("\n【模型性能评估】:")
print(f"  - 训练集 10 折 CV 平均 R²: {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
print(f"  - 独立测试集 R²         : {test_r2:.4f}")
print("=" * 50)


# ==========================================
# 4. 计算并提取全样本的 SHAP 净效应与交互作用
# ==========================================
explainer = shap.TreeExplainer(best_model)

# 4.1 各因子的 SHAP 净效应 (N_samples x N_features)
shap_obj = explainer(X)
shap_values_matrix = shap_obj.values  # numpy array

# 构建各因子 SHAP 净效应 DataFrame (增加 'SHAP_' 前缀)
df_shap_main = pd.DataFrame(
    shap_values_matrix,
    columns=[f"SHAP_{col}" for col in scale_vars],
    index=df_model.index
)

# 4.2 各因子对 EcM_prop 的 SHAP 交互作用 (N_samples x N_features x N_features)
interaction_matrix = explainer.shap_interaction_values(X)

# 获取 EcM_prop 的列索引
ecm_idx = scale_vars.index('EcM_prop')

# 提取所有因子与 EcM_prop 的交互作用矩阵：shape -> (N_samples, N_features)
ecm_interactions = interaction_matrix[:, ecm_idx, :]

# 构建交互作用 DataFrame (增加 'SHAP_inter_EcM_x_' 前缀)
# 当 col == 'EcM_prop' 时，提取的值是 EcM_prop 的纯主效应 (Main Effect)
df_shap_inter = pd.DataFrame(
    ecm_interactions,
    columns=[f"SHAP_inter_EcM_x_{col}" for col in scale_vars],
    index=df_model.index
)

# ==========================================
# 5. 合并回原始数据并导出 CSV
# ==========================================
# 将计算出的 SHAP 列按原 Index 贴回 cleaned 数据
df_output = pd.concat([df_model, df_shap_main, df_shap_inter], axis=1)

# output_path = "D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/new/all_plots_with_SHAP_addConifer.csv"
# df_output.to_csv(output_path, index=False)

print("=" * 60)
print(f"✅ 数据已成功导出至: {output_path}")
print(f"  - 新增 SHAP 净效应列数     : {df_shap_main.shape[1]} 列 (例: SHAP_MAT)")
print(f"  - 新增 EcM 交互作用列数   : {df_shap_inter.shape[1]} 列 (例: SHAP_inter_EcM_x_MAT)")
print("=" * 60)