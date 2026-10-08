# =============================================================================
# leaf traits PCA: PC1 x Leaf economic strategy (PC2)  loadings
# Output         : PDF (vector, Illustrator-editable) + PNG preview
# Requirements   : pip install matplotlib cartopy numpy pandas sklearn
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
import geopandas as gpd
warnings.filterwarnings('ignore')


# ============================================================================
# —————————————————————————————————————————— data  ——————————————————————————————————————————————

variables = gpd.read_file("D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/All_plot_addConifer.shp")
print(variables.axes)


df = variables[['LNC', 'LMA', 'LPC', 'Lignin','Cellulose','NSC','Phenolics','EWT']]
# —————————————————————————————————————————— PCA  ——————————————————————————————————————————————
# 标准化
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# PCA
pca = PCA(n_components=3)
scores = pca.fit_transform(df_scaled)

# 计算载荷
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loadings_df = pd.DataFrame(
    loadings,
    columns=[f'PC{i + 1}' for i in range(3)],
    index=df.columns
)

# 计算贡献
contributions = pd.DataFrame(
    loadings_df.values ** 2 / pca.explained_variance_,
    columns=loadings_df.columns,
    index=df.columns
)
loadings_df = loadings_df * -1              # ======================================================= 进行了方向变换 ==========================================


# 得分DataFrame
scores_df = pd.DataFrame(
    scores,
    columns=[f'PC{i + 1}' for i in range(3)],
    index=df.index
)

# 结果汇总
results = {
    'pca': pca,
    'scores': scores_df,
    'loadings': loadings_df,
    'contributions': contributions,
    'explained_variance_ratio': pca.explained_variance_ratio_,
    'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
    'eigenvalues': pca.explained_variance_,
    'scaler': scaler,
    'scaled_data': df_scaled
}
print(results['scores'])
print(results['loadings'])
print(results['contributions'])
print(results['explained_variance_ratio'])

scores_df = pd.DataFrame(results['scores'])[['PC1', 'PC2']]  # 只取 PC1/PC2

trait_names = ['LNC', 'LMA', 'LPC', 'Lignin',
                'Cellulose', 'NSC','Phenolics','EWT']

loadings_df = pd.DataFrame(results['loadings'], index=trait_names)[['PC1', 'PC2']]

scores_norm = scores_df.copy()
for col in ['PC1','PC2']:
    max_abs = scores_norm[col].abs().max()
    scores_norm[col] = scores_norm[col] / max_abs   # 线性映射到 [-1, 1]

# ══════════════════════════════════════════════════════════════════════════════
# 3. 绘图
# ══════════════════════════════════════════════════════════════════════════════
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("white")
rcParams['font.family'] = 'Arial'  # 论文常用字体
rcParams['font.size'] = 9  # 图片内字体大小（五号字=10.5，图片内用8-9）
rcParams['axes.labelsize'] = 12  # 坐标轴标签稍大
rcParams['xtick.labelsize'] = 9  # X轴刻度
rcParams['ytick.labelsize'] = 9  # Y轴刻度
# rcParams['legend.fontsize'] = 8 # 图例稍小
# rcParams['figure.titlesize'] = 8  # 图片标题


fig, ax = plt.subplots(figsize=(8 / 2.54, 6.5 / 2.54), facecolor='white')
ax.set_facecolor('white')

# ── KDE 密度填充 ──────────────────────────────────────────────────────────────
xy = scores_norm.values.T  # (2, n)
kde = gaussian_kde(xy, bw_method=0.25)
xg = np.linspace(-1, 1, 250)
yg = np.linspace(-1, 1, 250)
Xg, Yg = np.meshgrid(xg, yg)
Zg = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)

cmap_colors = ['white','#fffde7', '#ffe082', '#ffb300', '#e65100', '#8b0000']
cmap = LinearSegmentedColormap.from_list('heatmap', cmap_colors, N=256)

levels = np.linspace(Zg.min(), Zg.max(), 18)
ax.contourf(Xg, Yg, Zg, levels=levels, cmap=cmap, alpha=0.85, zorder=1)
ax.contour(Xg, Yg, Zg, levels=levels[::2], colors='#c0392b',
           linewidths=0.4, alpha=0.5, zorder=2)

# ── 虚线十字 ──────────────────────────────────────────────────────────────────
ax.axhline(0, color='#888888', lw=0.8, ls='--', zorder=3)
ax.axvline(0, color='#888888', lw=0.8, ls='--', zorder=3)

# ── 箭头 ──────────────────────────────────────────────────────────────────────
ARROW_SCALE = 0.95
for trait in loadings_df.index:
    x, y = loadings_df.loc[trait, 'PC1'], loadings_df.loc[trait, 'PC2']
    ax.annotate(
        '', xy=(x * ARROW_SCALE, y * ARROW_SCALE), xytext=(0, 0),
        arrowprops=dict(arrowstyle='->', color='black',
                        lw=1.1, mutation_scale=10), zorder=5
    )

# ── 标签 ──────────────────────────────────────────────────────────────────────
LABEL_OFFSET = 0.08  # 标签到箭头头部的偏移量（data units）
for trait in loadings_df.index:
    x, y = loadings_df.loc[trait, 'PC1'], loadings_df.loc[trait, 'PC2']
    # 沿箭头方向再延伸一点放标签
    norm = np.hypot(x, y) or 1
    lx = x * ARROW_SCALE + x / norm * LABEL_OFFSET
    ly = y * ARROW_SCALE + y / norm * LABEL_OFFSET
    ha = 'left' if x >= 0 else 'right'
    ax.text(lx, ly, trait,
            ha=ha, va='center',
            color='#1a1a1a', zorder=6,
            bbox=dict(boxstyle='round,pad=0.18', fc='none', ec='none', alpha=0.6))

# ── 坐标轴 ───────────────────────────────────────────────────────────────────
ax.add_patch(plt.Circle((0, 0), 1, fill=False,  linewidth=0.5, linestyle='--', alpha=1, color='gray'))
# 请将括号内的百分比替换为你实际的方差解释率
ax.set_ylabel('PC2 Scaled (39.3%)', labelpad=0)
ax.set_xlabel('PC1 Scaled (43.5%)', labelpad=0)
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
# for spine in ax.spines.values():
#     spine.set_linewidth(0.8)
#     spine.set_color('#333333')
ax.set_aspect('equal')

plt.tight_layout()
plt.show()


# ── 12. Export ────────────────────────────────────────────────────────────────
fig.savefig("./Fig1_a.pdf",
            format="pdf", dpi=300, bbox_inches="tight", facecolor="none")
