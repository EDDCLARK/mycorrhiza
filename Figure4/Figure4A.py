import warnings
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

# 记录原始尺度的均值和标准差 (用于后续 Z-score 反推与标准化)
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

means_dict = {v: df_model[v].mean() for v in scale_vars}
stds_dict = {v: df_model[v].std() for v in scale_vars}

scaler = StandardScaler()
df_model[scale_vars] = scaler.fit_transform(df_model[scale_vars])

# =============================================================================
# 2. 拟合初始 LMM 模型
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

# =============================================================================
# 3. 对所有真实观测点分别在 EcM=0.2 与 EcM=0.8 下重新预测 PC1
# =============================================================================
# 转化为 Z-score 标准化后的 0.2 和 0.8
ecm02_scaled = (0.2 - means_dict['EcM_prop_x']) / stds_dict['EcM_prop_x']
ecm08_scaled = (0.8 - means_dict['EcM_prop_x']) / stds_dict['EcM_prop_x']

# 保持所有样地的真实环境/林分协变量不变，仅替换 EcM_prop_x
df_all_pred_02 = df_model.copy()
df_all_pred_02['EcM_prop_x'] = ecm02_scaled

df_all_pred_08 = df_model.copy()
df_all_pred_08['EcM_prop_x'] = ecm08_scaled

# 预测每个样地在两种菌根策略下的 PC1
df_model['pred_PC1_ecm02'] = result_random_slope.predict(df_all_pred_02)
df_model['pred_PC1_ecm08'] = result_random_slope.predict(df_all_pred_08)

# 计算各样地的预测差值 ΔPC1
df_model['delta_PC1'] = df_model['pred_PC1_ecm08'] - df_model['pred_PC1_ecm02']

# =============================================================================
# 4. 8x8 网格划分与网格内 ΔPC1 均值统计
# =============================================================================
n_bins_mat = 8
n_bins_tseas = 8

# 还原真实物理尺度用于划分 Bin
df_model['MAT_real'] = df_model['MAT_x'] * stds_dict['MAT_x'] + means_dict['MAT_x']
df_model['Tseas_real'] = (
    df_model['Tseasonali_x'] * stds_dict['Tseasonali_x']
    + means_dict['Tseasonali_x']
)
# df_model['MAT_real'] = (
#     df_model['AridityInd_x'] * stds_dict['AridityInd_x']
#     + means_dict['AridityInd_x']
# )

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

# 计算各网格内样地的 ΔPC1 均值和样地总数 N
grid_stats = (
    df_model.groupby(['Tseas_bin', 'MAT_bin'])['delta_PC1']
    .agg(mean='mean', count='count')
    .reset_index()
)

diff_pc1_grid = np.full((n_bins_tseas, n_bins_mat), np.nan)
grid_n = np.zeros((n_bins_tseas, n_bins_mat))

for _, row in grid_stats.iterrows():
  i = int(row['Tseas_bin'])
  j = int(row['MAT_bin'])
  diff_pc1_grid[i, j] = row['mean']
  grid_n[i, j] = row['count']

# =============================================================================
# 5. 绘图样式与精细渲染 (Fig1_f 规范)
# =============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style('white')
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 9
rcParams['axes.labelsize'] = 11
rcParams['xtick.labelsize'] = 8.5
rcParams['ytick.labelsize'] = 8.5

# 颜色与发散归一化 (Center at 0)
valid_vals = diff_pc1_grid[~np.isnan(diff_pc1_grid)]
vmax = np.percentile(np.abs(valid_vals), 98) if len(valid_vals) > 0 else 1.0
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
cmap = plt.cm.Spectral_r

# 单列 Nature 标准物理尺寸 (6.5 cm x 6.5 cm)
fig, ax_main = plt.subplots(figsize=(6.5 / 2.54, 6.5 / 2.54))

im = ax_main.imshow(
    diff_pc1_grid,
    origin='lower',  # y轴从低到高
    aspect='auto',
    cmap=cmap,
    norm=norm,
    interpolation='nearest',
)

# 单元格数值标注
for i in range(n_bins_tseas):  # y 轴: Tseasonali
  for j in range(n_bins_mat):  # x 轴: MAT
    val = diff_pc1_grid[i, j]
    n_count = grid_n[i, j]

    if np.isnan(val) or n_count == 0:
      continue

    # 文字颜色动态适配背景亮暗
    txt_color = 'black' if abs(val) < (vmax * 0.5) else 'white'

    ax_main.text(
        j,
        i,
        f'{val:+.2f}',  # 格式化带 + / - 号的两位浮点数
        ha='center',
        va='center',
        fontsize=6.0,
        color=txt_color,
        fontweight='bold',
    )

# 设置坐标轴刻度与实际物理中心值
ax_main.set_xticks(range(n_bins_mat))
ax_main.set_xticklabels([f'{v:.1f}' for v in mat_centers])
ax_main.set_yticks(range(n_bins_tseas))
ax_main.set_yticklabels([f'{v:.0f}' for v in tseas_centers])

ax_main.set_xlabel('MAT (°C)', labelpad=4)
ax_main.set_ylabel('Temperature Seasonality', labelpad=4)

# 次要刻度用于绘制白色小方格网格线
ax_main.set_xticks(np.arange(-0.5, n_bins_mat, 1), minor=True)
ax_main.set_yticks(np.arange(-0.5, n_bins_tseas, 1), minor=True)
ax_main.grid(which='minor', color='white', linewidth=1.2)
ax_main.tick_params(which='minor', length=0)
ax_main.tick_params(axis='x', rotation=45)

# Colorbar 调整
cbar = fig.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=7)
cbar.set_label(r'Mean Predicted $\Delta PC1$', fontsize=8.5)

plt.tight_layout()
plt.show()

