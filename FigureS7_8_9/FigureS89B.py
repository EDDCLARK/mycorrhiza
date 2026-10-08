import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats
from scipy.stats import gaussian_kde, wasserstein_distance
import statsmodels.api as sm

# ==========================================
# 1. 设置 Nature 出版级全局绘图样式
# ==========================================
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 8
rcParams['axes.titlesize'] = 8.5
rcParams['axes.labelsize'] = 8
rcParams['xtick.labelsize'] = 7
rcParams['ytick.labelsize'] = 7
rcParams['legend.fontsize'] = 7
rcParams['pdf.fonttype'] = 42  # 保留矢量文本，方便 Illustrator 后期组图
rcParams['ps.fonttype'] = 42
rcParams['axes.linewidth'] = 0.6
rcParams['xtick.major.width'] = 0.6
rcParams['ytick.major.width'] = 0.6

# ==========================================
# 2. 计算各个 Division 内 AM vs EcM 的差异指标
# ==========================================
# 读取原始环境与分区数据
df_input = pd.read_csv(
    "D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/new/all_plots_environment_supersection.csv")

# 过滤空值
df_clean = df_input[['SSection', 'PC1_new', 'EcM_prop_x']].dropna().copy()

div_metrics = []

for div, sub_df in df_clean.groupby('SSection'):
    # 以 0.5 为界划分 AM 与 EcM 优势群落
    am_pc1 = sub_df[sub_df['EcM_prop_x'] < 0.25]['PC1_new'].values
    ecm_pc1 = sub_df[sub_df['EcM_prop_x'] > 0.75]['PC1_new'].values

    # 样本量安全检查（确保两组都有足够样方计算密度/均值）
    if len(am_pc1) < 100 or len(ecm_pc1) < 100:
        continue

    # 1. Cohen's d (效应量 / 均值标准化差异)
    n_am, n_ecm = len(am_pc1), len(ecm_pc1)
    mean_am, mean_ecm = np.mean(am_pc1), np.mean(ecm_pc1)
    var_am, var_ecm = np.var(am_pc1, ddof=1), np.var(ecm_pc1, ddof=1)
    pooled_std = np.sqrt(((n_am - 1) * var_am + (n_ecm - 1) * var_ecm) / (n_am + n_ecm - 2))
    cohen_d = abs((mean_ecm - mean_am) / pooled_std) if pooled_std > 0 else np.nan

    # 2. Wasserstein Distance (EMD 推土机距离)
    emd_dist = wasserstein_distance(am_pc1, ecm_pc1)

    # 3. Overlapping Coefficient (OVL 重叠率)
    x_grid = np.linspace(
        min(sub_df['PC1_new'].min(), -5),
        max(sub_df['PC1_new'].max(), 5),
        500
    )
    try:
        kde_am = gaussian_kde(am_pc1)(x_grid)
        kde_ecm = gaussian_kde(ecm_pc1)(x_grid)
        ovl_coef = np.trapz(np.minimum(kde_am, kde_ecm), x_grid)
    except Exception:
        ovl_coef = np.nan

    div_metrics.append({
        'SSection': div,
        'N_AM': n_am,
        'N_EcM': n_ecm,
        'Cohen_d': cohen_d,
        'Wasserstein_dist': emd_dist,
        'OVL_overlap': ovl_coef
    })

df_metrics = pd.DataFrame(div_metrics)

# ==========================================
# 3. 合并偏相关分析结果与差异指标
# ==========================================
df_pcorr_summary = pd.read_csv("D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/new/partial_corr_conifer_by_supersections.csv")
# 这里与前面计算出的偏相关 DataFrame (df_pcorr_summary) 按 DIVISION_C 合并
df_merged = pd.merge(df_pcorr_summary, df_metrics, on='SSection', how='inner').dropna()

print("=" * 65)
print("【分区级偏相关系数与 AM-EcM 功能空间差异指标汇总】:")
print(df_merged[['SSection', 'Partial_r', 'Cohen_d', 'Wasserstein_dist', 'OVL_overlap']].to_string(index=False))
print("=" * 65)

# ==========================================
# 4. 绘制 Nature 级散点拟合图
# ==========================================
# 定义三组画图目标：(Y轴变量, X轴标签/Y轴标签, 子图序号)
plots_config = [
    ('Cohen_d', "Cohen's $d$", 'a'),
    ('Wasserstein_dist', 'Wasserstein Distance', 'b'),
    ('OVL_overlap', 'Overlap Coefficient (OVL)', 'c')
]

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), dpi=300)  # 标准双栏宽度 (~180 mm)

for idx, (y_var, y_label, label) in enumerate(plots_config):
    ax = axes[idx]

    x_data = df_merged['Partial_r'].values
    y_data = df_merged[y_var].values
    s_data = df_merged['N_samples'].values  # 散点大小按该分区的总样本量加权展示

    # 点的大小按样本量缩放 (界定在 20 到 100 pt 之间)
    s_scaled = 20 + 80 * (s_data - s_data.min()) / (s_data.max() - s_data.min() + 1e-5)

    # 绘制散点
    scatter = ax.scatter(
        x_data, y_data,
        s=s_scaled,
        color='#2b5c8f',
        alpha=0.6,
        edgecolor='black',
        linewidth=0.5,
        zorder=3
    )

    # 线性拟合与 OLS 统计
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)

    # 生成拟合线及 95% 置信区间
    x_seq = np.linspace(x_data.min(), x_data.max(), 100)
    y_fit = intercept + slope * x_seq

    # 绘制拟合直线
    ax.plot(x_seq, y_fit, color='#d95f02', linewidth=1.4, zorder=4)

    # 计算并绘制 95% 置信区间阴影
    n = len(x_data)
    residuals = y_data - (intercept + slope * x_data)
    residual_std = np.sqrt(np.sum(residuals ** 2) / (n - 2))
    x_mean = np.mean(x_data)
    se_fit = residual_std * np.sqrt(1 / n + (x_seq - x_mean) ** 2 / np.sum((x_data - x_mean) ** 2))

    t_val = stats.t.ppf(0.975, df=n - 2)
    ax.fill_between(
        x_seq,
        y_fit - t_val * se_fit,
        y_fit + t_val * se_fit,
        color='#d95f02', alpha=0.15, zorder=2
    )

    # 零参考线
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    # 设置坐标轴标签
    ax.set_xlabel('$r$ of EcM Dominance')
    ax.set_ylabel(y_label)

    # 标注拟合统计指标 (R² & p-value)
    p_text = f"$p$ < 0.001" if p_value < 0.001 else f"$p$ = {p_value:.3f}"
    stat_str = f"$R^2$ = {r_value ** 2:.2f}\n{p_text}"

    ax.text(
        0.15, 0.22, stat_str,    # 0.65, 0.92
        transform=ax.transAxes,
        fontsize=6.5,
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.8, linewidth=0.5)
    )

    # 标注 panel 序号 (a, b, c)
    # ax.text(-0.15, 1.08, label, transform=ax.transAxes, fontsize=9.5, fontweight='bold', va='top')

    # 精简边框美化
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.grid(True, linestyle='--', alpha=0.2, color='gray')

plt.tight_layout()

fig.savefig("./FigS6_b_conifer.pdf",format="pdf", dpi=300, bbox_inches="tight")
plt.show()