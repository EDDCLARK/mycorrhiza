import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# ==========================================
# 1. 读取数据与准备变量
# ==========================================
df_input = pd.read_csv(
    "D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/new/all_plots_environment_division.csv")

target_y = 'PC1_new'
focal_x = 'EcM_prop_x'

covariates = ['C_N_x', 'soilN_x', 'sand_x', 'SLOPE_x', 'totalNdepo_x',
              'MAT_x', 'AridityInd_x', 'STDAGE_x', 'speciesRic_x', 'Tseasonali_x','conifer_pr_x']

required_cols = [target_y, focal_x, 'DIVISION_C'] + covariates  # DIVISION_C   SSection
df_clean = df_input[required_cols].dropna().copy()


# ==========================================
# 2. 自定义偏相关计算函数 (残差法)
# ==========================================
def calculate_partial_corr(df, x_col, y_col, cov_cols):
    """
    通过 OLS 回归提取残差计算偏相关系数及 p 值
    """
    n = len(df)
    k = len(cov_cols)  # 协变量数量

    # 构建协变量矩阵 Z (带常数项)
    Z = sm.add_constant(df[cov_cols])

    # 1. 对 y 回归求残差 e_y
    model_y = sm.OLS(df[y_col], Z).fit()
    e_y = model_y.resid

    # 2. 对 x 回归求残差 e_x
    model_x = sm.OLS(df[x_col], Z).fit()
    e_x = model_x.resid

    # 3. 计算残差间的 Pearson 相关系数
    r, _ = stats.pearsonr(e_x, e_y)

    # 4. 手动计算偏相关的 t 统计量与 p 值 (df = N - k - 2)
    df_degrees = n - k - 2
    if df_degrees <= 0:
        return np.nan, np.nan

    t_stat = r * np.sqrt(df_degrees / (1 - r ** 2))
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=df_degrees))

    return r, p_val


# ==========================================
# 3. 按 DIVISION_C 分组循环计算
# ==========================================
results = []
divisions = df_clean['DIVISION_C'].unique()  # DIVISION_C   SSection

for div in divisions:
    sub_df = df_clean[df_clean['DIVISION_C'] == div].copy()  # DIVISION_C   SSection
    n_samples = len(sub_df)

    # 自由度安全检查 (N 必须大于 k + 2)
    min_required_n = len(covariates) + 2
    if n_samples <= min_required_n:
        print(f"⚠️ 警告: 分组 {div} 样本量太少 (N={n_samples} <= {min_required_n})，跳过计算。")
        continue

    # 调用自定义偏相关函数
    r_val, p_val = calculate_partial_corr(sub_df, focal_x, target_y, covariates)

    # 显著性标记
    if p_val < 0.001:
        sig_label = '***'
    elif p_val < 0.01:
        sig_label = '**'
    elif p_val < 0.05:
        sig_label = '*'
    else:
        sig_label = 'ns'

    results.append({
        'DIVISION_C': div,   # DIVISION_C   SSection
        'N_samples': n_samples,
        'Partial_r': r_val,
        'p_value': p_val,
        'Significance': sig_label
    })

# 转换为 DataFrame 汇总表并排序
df_pcorr_summary = pd.DataFrame(results)
df_pcorr_summary = df_pcorr_summary.sort_values(by='N_samples', ascending=False).reset_index(drop=True)

# ==========================================
# 4. 打印展示与导出
# ==========================================
print("=" * 65)
print(f"【按 DIVISION_C 分组的 {focal_x} 偏相关分析结果 (残差法)】")
print("=" * 65)
print(df_pcorr_summary.to_string(index=False))
print("=" * 65)

df_pcorr_summary.to_csv("D:/yanbo/mycorrhizal/Phd_Project1/data/version5/maintext/new/partial_corr_conifer_by_divisions.csv", index=False)