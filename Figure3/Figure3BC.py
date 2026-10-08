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

warnings.filterwarnings('ignore')

# =============================================================================
# 1. 全局样式设置 (Nature 期刊标准)
# =============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("white")
rcParams['font.family'] = 'Arial'  # 论文常用字体
rcParams['font.size'] = 9  # 图片内字体大小（五号字=10.5，图片内用8-9）
rcParams['axes.labelsize'] = 12  # 坐标轴标签稍大
rcParams['xtick.labelsize'] = 9  # X轴刻度
rcParams['ytick.labelsize'] = 9  # Y轴刻度

# =============================================================================
# 2. 数据读取、标准化与模型拟合
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

# 提取未归一化的均值和标准差，用于还原真实的物理坐标/标签
# 注意：原代码中使用 MAT_x 来还原 ndepo_raw，这里保持逻辑一致
ecm_mean, ecm_std = df_model['EcM_prop_x'].mean(), df_model['EcM_prop_x'].std()
ndepo_mean, ndepo_std = df_model['MAT_x'].mean(), df_model['MAT_x'].std()

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
# 3. 细化分组（由 3 组增加至 5 组分位数）
# =============================================================================
# X 轴序列 (EcM dominance)
ecm_scaled_seq = np.linspace(df_model['EcM_prop_x'].min(), df_model['EcM_prop_x'].max(), 100)
ecm_raw_seq = ecm_scaled_seq * ecm_std + ecm_mean  # 反标准化为 0~1 的实际比例

# 5 个梯度分位数: 10th, 30th, 50th, 70th, 90th
ndepo_quantiles = [0.10, 0.30, 0.50, 0.70, 0.90]
quantile_labels = ['10th', '30th', '50th', '70th', '90th']

ndepo_scaled_vals = df_model['Tseasonali_x'].quantile(ndepo_quantiles).values
ndepo_raw_vals = ndepo_scaled_vals * ndepo_std + ndepo_mean

# 使用 Seaborn 连续渐变色板 (如 YlOrRd / Rocket / Flare)，体现从低到高浓度的递进




colors = sns.color_palette("Spectral_r", n_colors=len(ndepo_quantiles))

# =============================================================================
# 4. 绘图：交互偏依赖曲线 (5组梯度，带实际数值标记)
# =============================================================================
fig, ax = plt.subplots(figsize=(8.4 / 2.54, 7.5 / 2.54), dpi=300)

fe_params = result.params  # 固定效应参数

for idx, (q_lab, ndepo_val, ndepo_raw, color) in enumerate(
        zip(quantile_labels, ndepo_scaled_vals, ndepo_raw_vals, colors)):
    # 偏依赖预测响应
    pred_y = (
            fe_params['Intercept'] +
            fe_params['EcM_prop_x'] * ecm_scaled_seq +
            fe_params['Tseasonali_x'] * ndepo_val +
            fe_params['EcM_prop_x:Tseasonali_x'] * (ecm_scaled_seq * ndepo_val)
    )

    # 格式化实际数值（保留 1 位小数，可根据变量精度需求更改如 {:.2f}）
    legend_label = f'{q_lab} ({ndepo_raw:.1f} $^{{\circ}}$C)'

    # 绘制拟合线
    ax.plot(
        ecm_raw_seq, pred_y,
        label=legend_label,
        color=color,
        linewidth=1.6,
        zorder=3
    )

# 0 坐标参考线
ax.axhline(y=0, color='#999999', linestyle='--', linewidth=0.7, zorder=1)

# ── 坐标轴与美化 ─────────────────────────────────────────────────────────────
ax.set_xlabel('EcM Dominance')
ax.set_ylabel('LES (PC1)')

# 隐藏右上边框
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['left'].set_linewidth(0.6)
ax.spines['bottom'].set_linewidth(0.6)

# 图例设置
ax.legend(
    title='T Seasonality (Quantile)',  # 可改为带有单位的标题，如 'MAT (°C)'
    title_fontsize=8,
    loc='best',
    frameon=False,
    fontsize=7.5
)

# 保存图片
fig.savefig("./Fig3_c.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.show()