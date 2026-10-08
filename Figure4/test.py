import warnings
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import seaborn as sns
from matplotlib import rcParams
from matplotlib.colors import TwoSlopeNorm
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

warnings.filterwarnings('ignore')

# =============================================================================
# 1. 数据读取与预处理
# =============================================================================
data_path = "D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/new/all_plots_environment_supersection.csv"
df_input = pd.read_csv(data_path)

model_vars = [
    'PC1_new',
    'C_N_x',
    'soilN_x',
    'sand_x',
    'SLOPE_x',
    'totalNdepo_x',
    'MAT_x',
    'AridityInd_x',
    'STDAGE_x',
    'speciesRic_x',
    'conifer_pr_x',
    'EcM_prop_x',
    'AM_prop_x',
    'SSection',
    'Tseasonali_x',
    'geometry_x',
]

df_model = df_input[model_vars].dropna().copy()

# 空间去重
dup_mask = df_model.duplicated(subset=['SSection', 'geometry_x'], keep='first')
if dup_mask.sum() > 0:
  df_model = df_model[~dup_mask].copy().reset_index(drop=True)

# 备份未标准化的物理尺度（用于后续 8x8 离散分箱）
df_model['MAT_real'] = df_model['MAT_x']
df_model['Tseas_real'] = df_model['Tseasonali_x']
df_model['C_N_real'] = df_model['C_N_x']
df_model['Ndep_real'] = df_model['totalNdepo_x']
df_model['AI_real'] = df_model['AridityInd_x']

# 协变量标准化
scale_vars = [
    'C_N_x',
    'soilN_x',
    'sand_x',
    'SLOPE_x',
    'totalNdepo_x',
    'MAT_x',
    'AridityInd_x',
    'STDAGE_x',
    'speciesRic_x',
    'EcM_prop_x',
    'conifer_pr_x',
    'Tseasonali_x',
]

scaler = StandardScaler()
df_model[scale_vars] = scaler.fit_transform(df_model[scale_vars])

# =============================================================================
# 2. 拟合全变量 LMM 模型，并生成样地级预测值 PC1_pred
# =============================================================================
formula_init = (
    'PC1_new ~ EcM_prop_x * Tseasonali_x + '
    'EcM_prop_x * totalNdepo_x + EcM_prop_x * C_N_x + EcM_prop_x * AridityInd_x'
    ' + '
    'EcM_prop_x * MAT_x + soilN_x + sand_x + SLOPE_x + '
    'EcM_prop_x * STDAGE_x + EcM_prop_x * speciesRic_x + conifer_pr_x'
)

model_random_slope = MixedLM.from_formula(
    formula=formula_init,
    data=df_model,
    groups='SSection',
    re_formula='~EcM_prop_x',
)

result_random_slope = model_random_slope.fit(reml=False)

# 使用 patsy 提取当前样地全部变量（包含各种真实协变量与交互项）的设计矩阵 X
formula_rhs = formula_init.split('~')[1].strip()
exog_all = patsy.dmatrix(formula_rhs, data=df_model, return_type='dataframe')

# 对齐系数向量
fe_params = result_random_slope.fe_params
exog_all = exog_all[fe_params.index]

# 1.计算每一个真实样地的全变量预测值 PC1_pred (Y_hat = X * Beta)
df_model['PC1_pred'] = np.dot(exog_all.values, fe_params.values)

# 2. 提取拟合好的随机效应字典 (以 SSection 为键)
random_effects = result_random_slope.random_effects

# 3. 为每个样地加上其对应 SSection 的随机效应 (Random Intercept + Random Slope)
re_adjustments = []
for idx, row in df_model.iterrows():
  group_id = row['SSection']
  ecm_val = row['EcM_prop_x']

  if group_id in random_effects:
    re_dict = random_effects[group_id]
    # 取出随机截距 (Group/SSection) 和随机斜率 (EcM_prop_x)
    u_intercept = re_dict.get('Group', 0.0)
    u_slope = re_dict.get('EcM_prop_x', 0.0)

    # 随机效应补充值: u_0 + u_1 * EcM
    re_val = u_intercept + u_slope * ecm_val
  else:
    re_val = 0.0

  re_adjustments.append(re_val)

# 4. 得到包含【固定效应 + 随机效应】的全模型预测 PC1
df_model['PC1_pred_with_RE'] = df_model['PC1_pred'] + np.array(re_adjustments)

# =============================================================================
# 3. 8x8 离散网格划分与“0.2/0.8 组别预测 PC1 均值差”计算
# =============================================================================
n_bins_mat = 8
n_bins_tseas = 8

df_model['MAT_bin'], mat_edges = pd.cut(
    df_model['MAT_real'],
    bins=n_bins_mat,
    labels=False,
    retbins=True,
    include_lowest=True,
)

df_model['Tseas_bin'], tseas_edges = pd.cut(
    df_model['Tseas_real'],
    bins=n_bins_tseas,
    labels=False,
    retbins=True,
    include_lowest=True,
)

mat_centers = (mat_edges[:-1] + mat_edges[1:]) / 2
tseas_centers = (tseas_edges[:-1] + tseas_edges[1:]) / 2

# 将标准化后的 0.2 和 0.8 极值阈值计算出来 (由于之前做了 StandardScaler)
# 标准化公式: z = (x - mean) / std
ecm_mean = scaler.mean_[scale_vars.index('EcM_prop_x')]
ecm_scale = scaler.scale_[scale_vars.index('EcM_prop_x')]

ecm_02_threshold = (0.2 - ecm_mean) / ecm_scale
ecm_08_threshold = (0.8 - ecm_mean) / ecm_scale

diff_pc1_grid = np.full((n_bins_tseas, n_bins_mat), np.nan)
grid_n_ecm = np.zeros((n_bins_tseas, n_bins_mat))
grid_n_am = np.zeros((n_bins_tseas, n_bins_mat))

# 遍历每个网格计算 EcM (>=0.8) 与 AM (<=0.2) 样地的【模型预测 PC1 均值差】
for i in range(n_bins_tseas):
  for j in range(n_bins_mat):
    sub = df_model[(df_model['Tseas_bin'] == i) & (df_model['MAT_bin'] == j)]

    # 按照 0.8 (EcM主导) 和 0.2 (AM主导) 划分真实样地
    sub_ecm = sub[sub['EcM_prop_x'] >= ecm_08_threshold]
    sub_am = sub[sub['EcM_prop_x'] <= ecm_02_threshold]

    grid_n_ecm[i, j] = len(sub_ecm)
    grid_n_am[i, j] = len(sub_am)

    # 只有当该网格内同时存在 EcM 主导和 AM 主导样地时，才计算预测均值差
    if len(sub_ecm) > 0 and len(sub_am) > 0:
      mean_ecm_pred_pc1 = sub_ecm['PC1_pred'].mean()
      mean_am_pred_pc1 = sub_am['PC1_pred'].mean()
      diff_pc1_grid[i, j] = mean_ecm_pred_pc1 - mean_am_pred_pc1

# =============================================================================
# 4. 绘图样式与精细渲染 (Fig1_f 风格)
# =============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style('white')
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 9
rcParams['axes.labelsize'] = 11
rcParams['xtick.labelsize'] = 8.5
rcParams['ytick.labelsize'] = 8.5

valid_vals = diff_pc1_grid[~np.isnan(diff_pc1_grid)]
vmax = np.percentile(np.abs(valid_vals), 98) if len(valid_vals) > 0 else 1.0
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
cmap = plt.cm.Spectral_r

fig, ax_main = plt.subplots(figsize=(6.5 / 2.54, 6.5 / 2.54))

im = ax_main.imshow(
    diff_pc1_grid,
    origin='lower',
    aspect='auto',
    cmap=cmap,
    norm=norm,
    interpolation='nearest',
)

# 单元格数值标注
for i in range(n_bins_tseas):
  for j in range(n_bins_mat):
    val = diff_pc1_grid[i, j]
    if np.isnan(val):
      continue

    txt_color = 'black' if abs(val) < (vmax * 0.5) else 'white'

    ax_main.text(
        j,
        i,
        f'{val:+.2f}',
        ha='center',
        va='center',
        fontsize=6.0,
        color=txt_color,
        fontweight='bold',
    )

ax_main.set_xticks(range(n_bins_mat))
ax_main.set_xticklabels([f'{v:.1f}' for v in mat_centers])
ax_main.set_yticks(range(n_bins_tseas))
ax_main.set_yticklabels([f'{v:.0f}' for v in tseas_centers])

ax_main.set_xlabel('MAT (°C)', labelpad=4)
ax_main.set_ylabel('Temperature Seasonality', labelpad=4)

ax_main.set_xticks(np.arange(-0.5, n_bins_mat, 1), minor=True)
ax_main.set_yticks(np.arange(-0.5, n_bins_tseas, 1), minor=True)
ax_main.grid(which='minor', color='white', linewidth=1.2)
ax_main.tick_params(which='minor', length=0)
ax_main.tick_params(axis='x', rotation=45)

cbar = fig.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=7)
cbar.set_label(r'Predicted $\Delta PC1$ ($\overline{EcM}_{0.8} - \overline{AM}_{0.2}$)', fontsize=8.5)

plt.tight_layout()
plt.show()