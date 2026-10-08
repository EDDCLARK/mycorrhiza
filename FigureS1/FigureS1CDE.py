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
loadings_df = loadings_df * -1


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

# scores_df = pd.DataFrame(results['scores'])[['PC2', 'PC3']]  # 只取 PC1/PC2

trait_names = ['LNC', 'LMA', 'LPC', 'Lignin',
               'Cellulose', 'NSC', 'Phenolics','EWT']

loadings_df = pd.DataFrame(results['loadings'], index=trait_names)[['PC1','PC2', 'PC3']]

# scores_norm = scores_df.copy()
# for col in ['PC2','PC3']:
#     max_abs = scores_norm[col].abs().max()
#     scores_norm[col] = scores_norm[col] / max_abs   # 线性映射到 [-1, 1]

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


fig, ax = plt.subplots(figsize=(6 / 2.54, 6 / 2.54), facecolor='white')
ax.set_facecolor('white')

# ==============================================================================
print(loadings_df.axes)
bar1 = sns.barplot(loadings_df, x='PC3', y=loadings_df.index, ax=ax,palette=['#ffe082' if x < 0 else '#8b0000' for x in loadings_df['PC3']])

ax.set_xlabel("PC3 (scaled)",labelpad=0)
ax.set_xlim(-1,1)
ax.set_ylabel("")
# ax.set_yticklabels("")


plt.tight_layout()
plt.show()


# ── 12. Export ────────────────────────────────────────────────────────────────
fig.savefig("./FigS1_e.pdf",
            format="pdf", dpi=300, bbox_inches="tight", facecolor="none")
