import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 1. 全局样式设置 (Nature 期刊排版规范)
# ============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("white")
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 9
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 8.5
rcParams['ytick.labelsize'] = 8.5

# ============================================================================
# 2. 读取 Shapefile 并进行数据提取
# ============================================================================
shp_path = r"D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/new/partial_r_conifer_supersection.shp"
gdf = gpd.read_file(shp_path)

col_mapping = {c.lower(): c for c in gdf.columns}
r_col = col_mapping.get('partial_r', 'Partial_r')
p_col = col_mapping.get('p_value', 'p_value')
n_col = col_mapping.get('n_samples', 'N_samples')

gdf[r_col] = pd.to_numeric(gdf[r_col], errors='coerce')
gdf[p_col] = pd.to_numeric(gdf[p_col], errors='coerce')
gdf[n_col] = pd.to_numeric(gdf[n_col], errors='coerce').fillna(0)

# 只提取有效样本 (N > 0)
df_valid = gdf[gdf[n_col] > 0].dropna(subset=[r_col, p_col]).copy()

# 划分为三大类：显著正相关、显著负相关、不显著
df_sig_pos = df_valid[(df_valid[p_col] < 0.05) & (df_valid[r_col] > 0)]
df_sig_neg = df_valid[(df_valid[p_col] < 0.05) & (df_valid[r_col] < 0)]
df_nonsig  = df_valid[df_valid[p_col] >= 0.05]

# ============================================================================
# 3. 绘制 Partial r 分布直方图 + KDE 曲线
# ============================================================================
fig, ax = plt.subplots(figsize=(7.5 / 2.54, 5.5 / 2.54), dpi=300)

# 配色与地图保持一致
color_pos = "#B2182B"     # 显著正 (深红)
color_neg = "#2166AC"     # 显著负 (深蓝)
color_nonsig = "#999999"  # 不显著 (中灰)

# 确定统一的直方图 Bin 边界
r_min, r_max = df_valid[r_col].min(), df_valid[r_col].max()
max_abs_r = max(abs(r_min), abs(r_max))
bins = np.linspace(-max_abs_r, max_abs_r, 31)  # 保持关于 0 对称

# --- 1. 绘制不显著区域 (底色带斜线) ---
ax.hist(
    df_nonsig[r_col],
    bins=bins,
    color='white',
    edgecolor='#555555',
    hatch='////',
    linewidth=0.5,
    alpha=0.7,
    zorder=2,
    label=f'Not Sig. (p ≥ 0.05)'
)

# --- 2. 绘制显著负相关区域 ---
ax.hist(
    df_sig_neg[r_col],
    bins=bins,
    color=color_neg,
    edgecolor='none',
    alpha=0.85,
    zorder=3,
    label=f'Sig. (-) (p < 0.05)'
)

# --- 3. 绘制显著正相关区域 ---
ax.hist(
    df_sig_pos[r_col],
    bins=bins,
    color=color_pos,
    edgecolor='none',
    alpha=0.85,
    zorder=3,
    label=f'Sig. (+) (p < 0.05)'
)

# --- 4. 叠加整体 KDE (核密度估计) 曲线 ---
ax_kde = ax.twinx()  # 双 Y 轴，右侧显示密度 scale
sns.kdeplot(
    data=df_valid[r_col],
    ax=ax_kde,
    color='#333333',
    linewidth=1.2,
    linestyle='-',
    zorder=4
)
ax_kde.set_ylabel('')
ax_kde.set_yticks([])  # 隐藏右侧 KDE 的具体刻度，保持简洁
ax_kde.spines['top'].set_visible(False)
ax_kde.spines['right'].set_visible(False)

# --- 5. 参考线 (0 刻度线 & 均值虚线) ---
ax.axvline(0, color='#333333', linestyle='--', linewidth=0.8, zorder=5)

# mean_r = df_valid[r_col].mean()
# ax.axvline(mean_r, color='#D95F02', linestyle=':', linewidth=1.0, zorder=5)
# # 在图中标注 Mean 值
# ax.text(
#     mean_r + 0.02, ax.get_ylim()[1] * 0.88,
#     f'Mean = {mean_r:.2f}',
#     color='#D95F02',
#     fontsize=7.5,
#     fontweight='bold'
# )

# ============================================================================
# 4. 坐标轴美化与图例设置
# ============================================================================
ax.set_xlabel('Partial correlation ($r$)', labelpad=4)
ax.set_ylabel('Frequency (Count)', labelpad=4)
ax.set_xlim(-0.5,1)
# 边界线条精简
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.6)
ax.spines['bottom'].set_linewidth(0.6)

ax.tick_params(axis='both', width=0.6, colors='#333333')
ax.grid(axis='y', linestyle=':', alpha=0.5, zorder=0)

# ============================================================================
# 精细化图例（同时展示样本量 n 和占比 %）
# ============================================================================
n_total = len(df_valid)  # 有效样本总数 (N > 0)

# 计算百分比
pct_pos = (len(df_sig_pos) / n_total) * 100 if n_total > 0 else 0
pct_neg = (len(df_sig_neg) / n_total) * 100 if n_total > 0 else 0
pct_nonsig = (len(df_nonsig) / n_total) * 100 if n_total > 0 else 0

# 构建带 n 和 % 的图例元素
legend_elements = [
    Patch(
        facecolor=color_pos,
        edgecolor='none',
        label=f'Sig. (+) ($n={len(df_sig_pos)}$) {pct_pos:.1f}%'
    ),
    Patch(
        facecolor=color_neg,
        edgecolor='none',
        label=f'Sig. (-) ($n={len(df_sig_neg)}$) {pct_neg:.1f}%'
    ),
    Patch(
        facecolor='white',
        edgecolor='#555555',
        hatch='////',
        label=f'Not Sig. ($n={len(df_nonsig)}$) {pct_nonsig:.1f}%'
    ),
]

ax.legend(
    handles=legend_elements,
    loc='best',
    frameon=False,
    fontsize=7.5,
    handlelength=1.0,
    handleheight=0.8,
    borderpad=0.2,
    labelspacing=0.3
)

plt.tight_layout()
plt.savefig("./Fig2_e_conifer.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.show()