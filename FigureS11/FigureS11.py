import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Patch
import seaborn as sns

# 设置全局字体 (符合 Nature/Science 期刊出版规范)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("white")
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 9
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9

# ==========================================
# 1. 数据读取与准备
# ==========================================
df_input = pd.read_csv(
    "D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/new/all_plots_environment_supersection.csv")

# 模型所需全变量
target_y = 'Phenolics_x'
scale_vars = ['C_N_x', 'soilN_x', 'sand_x', 'SLOPE_x', 'totalNdepo_x',
              'MAT_x', 'AridityInd_x', 'STDAGE_x', 'speciesRic_x',
              'EcM_prop_x', 'Tseasonali_x','conifer_pr_x']

model_vars = [target_y] + scale_vars + ['SSection']

# 剔除空值
df_model = df_input[model_vars].dropna().copy()

# 标准化 (Z-score 转换)：对因变量与自变量统一步骤标准化 (使用 2D 结构)
scaler = StandardScaler()
df_model[[target_y] + scale_vars] = scaler.fit_transform(df_model[[target_y] + scale_vars])

# ==========================================
# 2. 拟合线性混合效应模型 (LMM) 并提取系数与 p 值
# ==========================================
formula = f"{target_y} ~ " + " + ".join(scale_vars)

model_random_slope = MixedLM.from_formula(
    formula=formula,
    data=df_model,
    groups="SSection",
    re_formula="~EcM_prop_x"
)

result_lmm = model_random_slope.fit(reml=False)

lmm_params = result_lmm.params
lmm_pvalues = result_lmm.pvalues

lmm_results = []
for var in scale_vars:
    lmm_results.append({
        'Variable': var,
        'lmm_coef': lmm_params[var],
        'lmm_abs_coef': abs(lmm_params[var]),
        'lmm_p': lmm_pvalues[var]
    })
df_lmm = pd.DataFrame(lmm_results)

# ==========================================
# 3. 计算偏相关系数 (Partial Correlation) 与 p 值
# ==========================================
pcorr_results = []
n_obs = len(df_model)

for x_col in scale_vars:
    covars = [c for c in scale_vars if c != x_col]

    X_cov = np.column_stack([np.ones(n_obs), df_model[covars].values])

    beta_y, _, _, _ = np.linalg.lstsq(X_cov, df_model[target_y].values, rcond=None)
    beta_x, _, _, _ = np.linalg.lstsq(X_cov, df_model[x_col].values, rcond=None)

    res_y = df_model[target_y].values - X_cov @ beta_y
    res_x = df_model[x_col].values - X_cov @ beta_x

    r, p = stats.pearsonr(res_x, res_y)
    pcorr_results.append({
        'Variable': x_col,
        'pcorr_r': r,
        'pcorr_abs_r': abs(r),
        'pcorr_p': p
    })
df_pcorr = pd.DataFrame(pcorr_results)

# ==========================================
# 4. 合并数据与整理显著性标记
# ==========================================
merged_df = pd.merge(df_pcorr, df_lmm, on='Variable')

merged_df = merged_df.sort_values(by='lmm_abs_coef', ascending=True).reset_index(drop=True)

def get_p_symbol(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'

merged_df['pcorr_symbol'] = merged_df['pcorr_p'].apply(get_p_symbol)
merged_df['lmm_symbol'] = merged_df['lmm_p'].apply(get_p_symbol)

name_mapping = {
    'MAT_x': 'MAT',
    'MAP_x': 'MAP',
    'AridityInd_x': 'Aridity Index',
    'Tseasonali_x': 'T Seasonality',
    'totalNdepo_x': 'N Deposition',
    'C_N_x': 'Soil C:N',
    'soilN_x': 'Soil N',
    'sand_x': 'Sand Content',
    'SLOPE_x': 'Slope',
    'STDAGE_x': 'Forest Age',
    'speciesRic_x': 'Species Richness',
    'EcM_prop_x': 'EcM Dominance',
    'conifer_pr_x': 'Conifer Proportion'
}

merged_df['Display_Name'] = merged_df['Variable'].map(name_mapping).fillna(
    merged_df['Variable'].str.replace('_x$', '', regex=True)
)

# ==========================================
# 5. 绘制分组双水平柱状图 (Fig. 1b 风格)
# ==========================================
fig, ax = plt.subplots(figsize=(6 / 2.54, 7.5 / 2.54), dpi=300)

y_positions = np.arange(len(merged_df))
h = 0.35

color_pcorr_pos = "#B2182B"
color_pcorr_neg = "#2166AC"
color_lmm_pos = "#F4A582"
color_lmm_neg = "#92C5DE"

# --- 绘制 1: Partial Correlation (上方柱子: y - h/2) ---
for i, row in merged_df.iterrows():
    c = color_pcorr_pos if row['pcorr_r'] > 0 else color_pcorr_neg
    ax.barh(
        y_positions[i] - h / 2,
        row['pcorr_abs_r'],
        height=h,
        color=c,
        edgecolor="none",
        zorder=3
    )

# --- 绘制 2: LMM Coefficient (下方柱子: y + h/2) ---
for i, row in merged_df.iterrows():
    c = color_lmm_pos if row['lmm_coef'] > 0 else color_lmm_neg
    ax.barh(
        y_positions[i] + h / 2,
        row['lmm_abs_coef'],
        height=h,
        color=c,
        edgecolor="none",
        zorder=3
    )

    cof = float(row['lmm_abs_coef'])

    if row['lmm_abs_coef'] > 0.4:
        ax.text(
            0.35,
            y_positions[i],
            '{:.2f}'.format(cof),
            va='center',
            ha='left',
            fontweight='normal',
            color='#333333'
        )

# ==========================================
# 6. 坐标轴与美化
# ==========================================
ax.set_yticks(y_positions)
ax.set_yticklabels(merged_df['Display_Name'], fontsize=9.5)
ax.set_xlabel(f"Coefficient", fontsize=9.5, labelpad=6)

ax.set_xlim(0, 0.4)
ax.axvline(0, color="#333333", linestyle="-", linewidth=0.8, zorder=4)

ax.spines["bottom"].set_linewidth(0.8)
ax.spines["bottom"].set_color("#333333")

ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", width=0.8, direction="out", colors="#333333")
ax.grid(axis='x', linestyle=':', alpha=0.5, zorder=0)

# ==========================================
# 7. 图例
# ==========================================
legend_elements = [
    Patch(facecolor=color_pcorr_pos, label='Partial corr. (+)'),
    Patch(facecolor=color_pcorr_neg, label='Partial corr. (−)'),
    Patch(facecolor=color_lmm_pos, label='LMM coef. (+)'),
    Patch(facecolor=color_lmm_neg, label='LMM coef. (−)')
]

ax.legend(
    handles=legend_elements,
    frameon=False,
    loc="lower right",
    handlelength=1.0,
    handleheight=1.0,
    borderpad=0,
    labelspacing=0.1
)

plt.show()
fig.savefig("./FigS8_Phenolic.pdf",format="pdf", dpi=300, bbox_inches="tight")