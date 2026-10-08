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
# plt.rcParams["font.family"] = "Arial"
# plt.rcParams["font.size"] = 9
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("white")
rcParams['font.family'] = 'Arial'  # 论文常用字体
rcParams['font.size'] = 9  # 图片内字体大小（五号字=10.5，图片内用8-9）
rcParams['axes.labelsize'] = 9  # 坐标轴标签稍大
rcParams['xtick.labelsize'] = 9  # X轴刻度
rcParams['ytick.labelsize'] = 9  # Y轴刻度
# ==========================================
# 1. 数据读取与准备
# ==========================================
df_input = pd.read_csv(
    "D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/new/all_plots_environment_supersection.csv")

# 模型所需全变量
target_y = 'PC1_new'
scale_vars = ['C_N_x', 'soilN_x', 'sand_x', 'SLOPE_x', 'totalNdepo_x',
              'MAT_x', 'AridityInd_x', 'STDAGE_x', 'speciesRic_x',
              'EcM_prop_x', 'Tseasonali_x']

model_vars = [target_y] + scale_vars + ['SSection']

# 剔除空值
df_model = df_input[model_vars].dropna().copy()

# 标准化 (Z-score 转换)：确保 LMM 固定效应与偏相关系数在相同尺度下对比
scaler = StandardScaler()
df_model[scale_vars] = scaler.fit_transform(df_model[scale_vars])

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

# 提取固定效应系数与 p 值 (排除 Intercept 和随机效应参数 Group Var)
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

    # 构造含截距项的设计矩阵
    X_cov = np.column_stack([np.ones(n_obs), df_model[covars].values])

    # 计算残差 (扣除协变量影响)
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

# 按 LMM 效应绝对值从大到小排序
merged_df = merged_df.sort_values(by='lmm_abs_coef', ascending=True).reset_index(drop=True)


# 显著性星号转换函数
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

# 简化显示名 (去除尾部的 _x)
# merged_df['Display_Name'] = merged_df['Variable'].str.replace('_x$', '', regex=True)


# ----------------- 【在此处定义你的自定义变量名映射】 -----------------
name_mapping = {
    'MAT_x': 'MAT',                      # Mean Annual Temperature
    'MAP_x': 'MAP',                      # Mean Annual Precipitation
    'AridityInd_x': 'Aridity Index',      # 干旱指数
    'Tseasonali_x': 'T Seasonality',  # 温度季节性
    'totalNdepo_x': 'N Deposition',# 氮沉降
    'C_N_x': 'Soil C:N',                 # 碳氮比
    'soilN_x': 'Soil N',                 # 土壤氮
    'sand_x': 'Sand Content',            # 砂粒含量
    'SLOPE_x': 'Slope',                  # 坡度
    'STDAGE_x': 'Forest Age',             # 林龄
    'speciesRic_x': 'Species Richness',  # 物种丰富度
    'EcM_prop_x': 'EcM Dominance'  # EcM 占比
}

# 应用自定义命名映射；如果某个变量没在字典里，则默认去除尾部的 '_x'
merged_df['Display_Name'] = merged_df['Variable'].map(name_mapping).fillna(
    merged_df['Variable'].str.replace('_x$', '', regex=True)
)
# ------------------------------------------------------------------








# ==========================================
# 5. 绘制分组双水平柱状图 (Fig. 1b 风格)
# ==========================================
fig, ax = plt.subplots(figsize=(6 / 2.54, 7.5 / 2.54), dpi=300)

y_positions = np.arange(len(merged_df))
h = 0.35  # 条形宽度

# 颜色配置 (深色表示 Partial Correlation；浅色表示 LMM)
color_pcorr_pos = "#B2182B"  # Partial Correlation (正)
color_pcorr_neg = "#2166AC"  # Partial Correlation (负)
color_lmm_pos = "#F4A582"  # LMM (正)
color_lmm_neg = "#92C5DE"  # LMM (负)

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

    # if row['pcorr_abs_r'] > 0.4:
    #     ax.text(
    #         0.4,
    #         y_positions[i] - h / 2,
    #         row['pcorr_abs_r'] + row['pcorr_symbol'],
    #         va='center',
    #         ha='left',
    #         fontsize=7.5,
    #         fontweight='bold' if row['pcorr_symbol'] != 'ns' else 'normal',
    #         color='#333333'
    #     )
    # else:
    #     # 星号标注
    #     ax.text(
    #         row['pcorr_abs_r'] + 0.008,
    #         y_positions[i] - h / 2,
    #         row['pcorr_symbol'],
    #         va='center',
    #         ha='left',
    #         fontsize=7.5,
    #         fontweight='bold' if row['pcorr_symbol'] != 'ns' else 'normal',
    #         color='#333333'
    #     )



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
            # fontsize=5.5,
            fontweight='normal',
            color='#333333'
        )

    # if row['lmm_abs_coef'] > 0.4:
    #     ax.text(
    #         0.4,
    #         y_positions[i] + h / 2,
    #         row['lmm_symbol'],
    #         va='center',
    #         ha='left',
    #         fontsize=7.5,
    #         fontweight='bold' if row['lmm_symbol'] != 'ns' else 'normal',
    #         color='#333333'
    #     )
    # else:
    #     # 星号标注
    #     ax.text(
    #         row['lmm_abs_coef'] + 0.008,
    #         y_positions[i] + h / 2,
    #         row['lmm_symbol'],
    #         va='center',
    #         ha='left',
    #         fontsize=7.5,
    #         fontweight='bold' if row['lmm_symbol'] != 'ns' else 'normal',
    #         color='#333333'
    #     )



# ==========================================
# 6. 坐标轴与美化
# ==========================================
ax.set_yticks(y_positions)
ax.set_yticklabels(merged_df['Display_Name'], fontsize=9.5)

ax.set_xlabel(f"Coefficient", fontsize=9.5, labelpad=6)

# 动态扩展 X 轴范围
max_val = max(merged_df['pcorr_abs_r'].max(), merged_df['lmm_abs_coef'].max())
ax.set_xlim(0, 0.4)

# 0 刻度纵线
ax.axvline(0, color="#333333", linestyle="-", linewidth=0.8, zorder=4)

# 移除冗余边框
# for spine in ["top", "right", "left"]:
#     ax.spines[spine].set_visible(False)

ax.spines["bottom"].set_linewidth(0.8)
ax.spines["bottom"].set_color("#333333")

ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", width=0.8, direction="out", colors="#333333")

ax.grid(axis='x', linestyle=':', alpha=0.5, zorder=0)

# ==========================================
# 7. 四分图例 (区分模型与正负)
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
    # fontsize=7.5,
    loc="lower right",
    handlelength=1.0,
    handleheight=1.0,
    borderpad=0,
    labelspacing=0.1
)

# plt.tight_layout()
plt.show()
# fig.savefig("./Fig2_a.pdf",format="pdf", dpi=300, bbox_inches="tight")