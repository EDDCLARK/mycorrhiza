import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import statsmodels.api as sm

# ==========================================
# 0. 变量名及单位映射字典 (采用 MathText 防止乱码)
# ==========================================
var_name_map = {
    'EcM_prop_x': 'EcM Dominance',
    'EcM_prop': 'EcM Dominance',
    'Tseasonali_x': r'T Seasonality ($^\circ$C)',
    'Tseasonali': r'T Seasonality ($^\circ$C)',
    'totalNdepo_x': r'N Deposition (kg N ha$^{-1}$ yr$^{-1}$)',
    'totalNdepo': r'N Deposition (kg N ha$^{-1}$ yr$^{-1}$)',
    'C_N_x': 'Soil C:N',
    'C_N': 'Soil C:N',
    'AridityInd_x': 'Aridity Index',
    'AridityInd': 'Aridity Index',
    'MAT_x': r'MAT ($^\circ$C)',
    'MAT': r'MAT ($^\circ$C)',
    'soilN_x': 'Soil N',
    'soilN': 'Soil N',
    'sand_x': 'Sand Content',
    'sand': 'Sand Content',
    'SLOPE_x': 'Slope',
    'SLOPE': 'Slope',
    'STDAGE_x': 'Forest Age',
    'STDAGE': 'Forest Age',
    'speciesRic_x': 'Species Richness',
    'speciesRic': 'Species Richness',
    'conifer_pr_x': 'Conifer proportion',
    'conifer_pr': 'Conifer proportion'
}

# 辅助函数：自动去除名称中的单位括号部分 (用于柱状图/条形图)
def remove_units(name_str):
    return re.sub(r'\s*\(.*?\)', '', name_str).strip()

# ==========================================
# 1. 设置 Nature / Word 跨栏标准样式 (紧凑型)
# ==========================================
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 6.5
rcParams['axes.titlesize'] = 7.0
rcParams['axes.labelsize'] = 6.5
rcParams['xtick.labelsize'] = 5.5
rcParams['ytick.labelsize'] = 5.5
rcParams['legend.fontsize'] = 5.5
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['axes.linewidth'] = 0.5
rcParams['xtick.major.width'] = 0.5
rcParams['ytick.major.width'] = 0.5

# ==========================================
# 2. 读取数据并诊断列名
# ==========================================
csv_path = "D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/new/all_plots_with_SHAP.csv"
df = pd.read_csv(csv_path)

# 定位 EcM 列
ecm_col = 'EcM_prop' if 'EcM_prop' in df.columns else 'EcM_prop_x'

# 自动匹配 EcM 的主效应 SHAP 列
shap_ecm_candidates = [c for c in df.columns if
                       ('SHAP' in c or 'shap' in c) and ecm_col in c and 'inter' not in c.lower()]
if not shap_ecm_candidates:
    shap_ecm_candidates = [c for c in df.columns if
                           'SHAP' in c and ('EcM' in c or 'ecm' in c) and 'inter' not in c.lower()]

shap_ecm_col = shap_ecm_candidates[0] if shap_ecm_candidates else None

# 自动查找所有包含交互作用 (inter) 且包含 EcM 的 SHAP 列
inter_cols = [c for c in df.columns if
              ('inter' in c.lower() or 'interaction' in c.lower()) and ('EcM' in c or 'ecm' in c)]

print("🔍 [诊断信息]:")
print(f"  - 检测到 EcM 属性列: {ecm_col}")
print(f"  - 检测到 EcM 主效应 SHAP 列: {shap_ecm_col}")

if not shap_ecm_col:
    print("\n❌ 错误：未能在 CSV 中匹配到 EcM 的主效应 SHAP 列！")
    raise SystemExit("停止执行：请检查列名。")

# ==========================================
# 3. 自动匹配环境因子并提取数据
# ==========================================
feature_map = {}
for col in inter_cols:
    clean_name = col.replace('SHAP_', '').replace('shap_', '').replace('inter_', '').replace('interaction_', '')
    parts = clean_name.split('_and_') if '_and_' in clean_name else clean_name.split('_x_')
    if len(parts) == 1:
        parts = clean_name.split('_')

    other_parts = [p for p in parts if 'ecm' not in p.lower()]
    if other_parts:
        var_base = other_parts[0]
        matched_raw = None
        for candidate in [var_base, f"{var_base}_x", var_base.replace('_x', '')]:
            if candidate in df.columns:
                matched_raw = candidate
                break

        if matched_raw and matched_raw != ecm_col:
            feature_map[col] = (var_base, matched_raw)

# ==========================================
# 4. 计算交互作用强度并全量排序
# ==========================================
valid_inter_cols = []
for inter_col, (var_name, raw_feat) in feature_map.items():
    sub = df.dropna(subset=[ecm_col, shap_ecm_col, inter_col, raw_feat])
    if len(sub) > 0:
        importance = sub[inter_col].abs().mean()
        display_name = var_name_map.get(raw_feat, var_name_map.get(var_name, var_name))
        display_name_no_unit = remove_units(display_name)

        valid_inter_cols.append({
            'inter_col': inter_col,
            'importance': importance,
            'var_name': var_name,
            'raw_feat': raw_feat,
            'display_name': display_name,
            'display_name_no_unit': display_name_no_unit
        })

# 按交互强度从高到低排序
ranking_df = pd.DataFrame(valid_inter_cols).sort_values(by='importance', ascending=False).reset_index(drop=True)

print(f"\n✅ 交互作用强度排名 (前 4):")
for idx, row in ranking_df.head(4).iterrows():
    print(f"  Rank {idx + 1}: {row['display_name_no_unit']} (Importance: {row['importance']:.4f})")

# ==========================================
# 5. 构建画布 (适配 Word 跨栏宽度: 7.1 in × 2.9 in)
# ==========================================
fig = plt.figure(figsize=(7.1, 2.9), dpi=300)
gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.0, 1.0], wspace=0.38, hspace=0.48)

# 左图: 交互作用排名条形图
ax_left = fig.add_subplot(gs[:, 0])

# 右侧 2x2 网格子图布局
ax_right1 = fig.add_subplot(gs[0, 1])  # Top 1
ax_right2 = fig.add_subplot(gs[0, 2])  # Top 2
ax_right3 = fig.add_subplot(gs[1, 1])  # Top 3
ax_right4 = fig.add_subplot(gs[1, 2])  # Top 4

# ------------------------------------------
# 图 (a): 交互作用强度排名条形图 (无单位)
# ------------------------------------------
plot_rank_df = ranking_df.sort_values(by='importance', ascending=True)
y_positions = np.arange(len(plot_rank_df))

bars = ax_left.barh(y_positions, plot_rank_df['importance'], color='#3B528B', edgecolor='none', height=0.6, alpha=0.85)

n_vars = len(plot_rank_df)
top4_indices = [n_vars - 1 - i for i in range(min(4, n_vars))]
for idx in top4_indices:
    bars[idx].set_color('#440154')

ax_left.set_yticks(y_positions)
# 这里使用去掉了单位的变量名
ax_left.set_yticklabels(plot_rank_df['display_name_no_unit'], fontweight='bold', fontsize=6.0)
ax_left.set_xlabel("Mean |Interaction SHAP|", fontweight='bold', fontsize=6.5, labelpad=2)
ax_left.spines['top'].set_visible(False)
ax_left.spines['right'].set_visible(False)

# ------------------------------------------
# 图 (b) - (e): Top 1 至 Top 4 偏依赖图 / 箱型图
# ------------------------------------------
vmin, vmax = df[ecm_col].min(), df[ecm_col].max()
cmap_style = 'viridis'
right_axes = [ax_right1, ax_right2, ax_right3, ax_right4]

last_scatter = None

for idx in range(min(4, len(ranking_df))):
    ax = right_axes[idx]
    row = ranking_df.iloc[idx]

    display_name = row['display_name']
    raw_feat = row['raw_feat']
    inter_col = row['inter_col']

    sub_df = df.dropna(subset=[ecm_col, shap_ecm_col, inter_col, raw_feat]).copy()

    x_val = sub_df[raw_feat].values.astype(float)
    ecm_val = sub_df[ecm_col].values

    # 单位转换处理
    if 'tseasonali' in raw_feat.lower():
        x_val = x_val / 100.0
    elif 'aridityind' in raw_feat.lower():
        x_val = x_val / 10000.0

    ecm_shap_val = sub_df[shap_ecm_col].values
    inter_shap_val = sub_df[inter_col].values
    y_mod_val = np.sign(ecm_shap_val) * inter_shap_val

    # 判断是否为干旱指数 (Aridity Index) -> 绘制分组箱型图
    if 'aridityind' in raw_feat.lower():
        bins = [-np.inf, 0.2, 0.5, 0.65, np.inf]
        labels = ['Arid', 'Semi-arid', 'Sub-humid', 'Humid']
        categories = pd.cut(x_val, bins=bins, labels=labels)

        box_data = [y_mod_val[categories == cat] for cat in labels]

        bp = ax.boxplot(
            box_data,
            patch_artist=True,
            labels=labels,
            widths=0.45,
            flierprops=dict(marker='o', markersize=0.8, alpha=0.3, markeredgecolor='none', markerfacecolor='gray')
        )

        colors = ['#E45756', '#F28E2B', '#76B7B2', '#4E79A7']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_linewidth(0.5)

        for element in ['whiskers', 'caps']:
            plt.setp(bp[element], color='#333333', linewidth=0.5)
        plt.setp(bp['medians'], color='black', linewidth=0.8)

        ax.axhline(0, color='red', linestyle='--', linewidth=0.6, alpha=0.7)
        ax.set_xticklabels(labels, fontsize=5.0, rotation=15)

    else:
        # 散点图 + LOWESS
        scatter = ax.scatter(
            x_val,
            y_mod_val,
            c=ecm_val,
            cmap=cmap_style,
            vmin=vmin,
            vmax=vmax,
            alpha=0.8,
            s=0.5,
            edgecolor='none'
        )
        last_scatter = scatter

        lowess = sm.nonparametric.lowess(y_mod_val, x_val, frac=0.35)
        ax.plot(lowess[:, 0], lowess[:, 1], color='red', linewidth=1.0, label='LOWESS Trend')
        ax.axhline(0, color='red', linestyle='--', linewidth=0.6, alpha=0.7)

    # 坐标轴美化与紧凑处理 (右侧图表的 X 轴保留带单位的名字)
    ax.set_xlabel(f"{display_name}", fontweight='bold', fontsize=6.0, labelpad=1.5)
    ax.set_ylabel("Modulation effect", fontweight='bold', fontsize=5.5, labelpad=1.5)

    ax.tick_params(axis='both', labelsize=5.0, pad=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.2, color='gray', linewidth=0.4)

# ==========================================
# 6. 添加全局 Colorbar 并导出
# ==========================================
plt.tight_layout(rect=[0, 0, 0.91, 1.0])

if last_scatter is not None:
    cbar_ax = fig.add_axes([0.92, 0.18, 0.012, 0.65])
    cbar = fig.colorbar(last_scatter, cax=cbar_ax)
    ecm_label_name = remove_units(var_name_map.get(ecm_col, 'EcM Proportion'))
    cbar.set_label(ecm_label_name, fontweight='bold', fontsize=6.0, labelpad=2)
    cbar.ax.tick_params(labelsize=5.0)

fig.savefig("./FigS14.pdf", format="pdf", dpi=300, bbox_inches='tight', pad_inches=0.03)
plt.show()