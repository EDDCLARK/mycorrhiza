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

warnings.filterwarnings('ignore')

# =============================================================================
# 1. 全局样式设置 (Nature 期刊风格)
# =============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("white")
rcParams['font.family'] = 'Arial'  # 论文常用字体
rcParams['font.size'] = 9  # 图片内字体大小（五号字=10.5，图片内用8-9）
rcParams['axes.labelsize'] = 9  # 坐标轴标签稍大
rcParams['xtick.labelsize'] = 7.5  # X轴刻度
rcParams['ytick.labelsize'] = 7.5  # Y轴刻度

# =============================================================================
# 2. 数据读取与模型拟合
# =============================================================================
data_path = "D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/new/all_plots_environment_supersection.csv"
df_input = pd.read_csv(data_path)

model_vars = [
    'PC1_new', 'C_N_x', 'soilN_x', 'sand_x', 'SLOPE_x', 'totalNdepo_x',
    'MAT_x', 'AridityInd_x', 'STDAGE_x', 'speciesRic_x', 'conifer_pr_x',
    'EcM_prop_x', 'AM_prop_x', 'SSection', 'Tseasonali_x'
]

df_model = df_input[model_vars].dropna().copy()

scale_vars = [
    'C_N_x', 'soilN_x', 'sand_x', 'SLOPE_x', 'totalNdepo_x', 'MAT_x',
    'AridityInd_x', 'STDAGE_x', 'speciesRic_x', 'EcM_prop_x', 'conifer_pr_x', 'Tseasonali_x'
]

scaler = StandardScaler()
df_model[scale_vars] = scaler.fit_transform(df_model[scale_vars])

formula = (
    "PC1_new ~ EcM_prop_x * Tseasonali_x + "
    "EcM_prop_x * totalNdepo_x + EcM_prop_x * C_N_x + EcM_prop_x * AridityInd_x + "
    "EcM_prop_x * MAT_x + soilN_x + sand_x + SLOPE_x + "
    "EcM_prop_x * STDAGE_x + EcM_prop_x * speciesRic_x"
)

model_random_slope = MixedLM.from_formula(
    formula=formula,
    data=df_model,
    groups="SSection",
    re_formula="~EcM_prop_x"
)

result = model_random_slope.fit(reml=False)

# =============================================================================
# 3. 提取系数并排序分步（左：非交互项，右：EcM及交互项）
# =============================================================================
conf_int = result.conf_int()
summary_df = pd.DataFrame({
    'Estimate': result.params,
    'CI_lower': conf_int[0],
    'CI_upper': conf_int[1],
    'pvalue': result.pvalues
})

# 剔除截距与随机效应
summary_df = summary_df.loc[~summary_df.index.str.contains('Intercept|Group|Var|Cov', case=False)].copy()

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
    #'conifer_pr_x':'Conifer proportion',
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

# 记录右侧 EcM 区域的起始与终止 X 轴索引
num_left = len(df_left)
total_vars = len(plot_df)

# =============================================================================
# 4. 绘图 (X 轴为变量，Y 轴为效应量)
# =============================================================================
fig, ax = plt.subplots(figsize=(16.8 / 2.54, 7.5 / 2.54), dpi=300)

x_positions = np.arange(total_vars)

# ── 1. 右侧 (EcM 及其交互项) 添加浅红色背景区域 ───────────────────────────────
# 从非交互项和交互项中间的缝隙 (num_left - 0.5) 一直到最右边界 (total_vars - 0.5)
ax.axvspan(num_left - 0.5, total_vars - 0.5, color='#FDF0F0', zorder=0)

# ── 2. Y=0 参考参考线 ────────────────────────────────────────────────────────
ax.axhline(y=0, color='#666666', linestyle='--', linewidth=0.8, zorder=1)

# ── 3. 绘制垂直置信区间与散点 ───────────────────────────────────────────────
for i, row in plot_df.iterrows():
    err_color = '#1A1A1A' if row['is_sig'] else '#888888'



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

    # Y 轴方向置信区间 (Vertical Line)
    ax.plot(
        [i, i], [row['CI_lower'], row['CI_upper']],
        color=err_color, linewidth=1.2, zorder=5
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

# ── 5. 添加区域分组标注文本 ─────────────────────────────────────────────────


# 图例
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
    edgecolor='none',
    # fontsize=7.5
)


fig.savefig("./Fig3_a.pdf",format="pdf", dpi=300, bbox_inches="tight")
plt.show()