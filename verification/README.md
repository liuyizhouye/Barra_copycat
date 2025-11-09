# CNE5 模型有效性验证

本模块实现了 CNE5 模型的所有有效性验证功能，基于《Barra China Equity Model (CNE5) Empirical Notes, July 2012》中的验证框架。

## 概述

在《**Barra China Equity Model (CNE5) Empirical Notes, July 2012**》中，模型的**有效性验证**主要集中在三个层面：

1. **因子收益的解释力与稳健性验证**
2. **特异收益与特异风险预测的准确性验证**
3. **协方差矩阵（系统风险部分）的预测偏差与稳定性验证**

---

## 一、因子层面验证：解释力与稳定性检验

CNE5 在第 5 章中进行了**系统的因子回测与统计显著性分析**，与旧模型 CHE2 对比。其验证指标包括：

### 1. 截面回归显著性（t-statistics）

逐个风格因子的月度因子收益回归，考察因子是否能显著解释个股超额收益。

**函数**：`calculate_factor_t_statistics()`

**返回指标**：
- `t_statistic`: t 统计量，|t| > 2 表示在 5% 水平下显著
- `p_value`: p 值，越小越显著
- `coefficient`: 回归系数

### 2. R² 与 RMS 分解（Cross-sectional dispersion decomposition）

将横截面收益离散度拆分为因子贡献与特异收益贡献，再细分为行业、风格、国家因子。该部分用于评估每类因子对市场波动的解释度及随时间的稳定性。

**函数**：`calculate_r_squared()`

**返回指标**：
- `r_squared`: R² = 1 - SS_res / SS_tot，越高越好，表示因子解释力强
- `rms_total`: 总 RMS 收益
- `{factor}_contribution`: 各因子的 RMS 贡献

### 3. 稳定性与共线性诊断

利用暴露的截面相关系数与 VIF（Variance Inflation Factor）检测风格因子之间的独立性与稳健性，避免过度共线性影响回归稳定性。

**函数**：
- `calculate_style_stability()`: 计算风格稳定性系数（月度相邻期暴露的截面相关）
- `calculate_vif()`: 计算方差膨胀因子，VIF > 5 通常被视为问题信号
- `calculate_factor_correlation()`: 计算因子暴露的截面相关系数

**理想值**：
- 风格稳定性系数：< 0.80 视为偏低，> 0.90 较理想
- VIF：< 5 为正常，> 5 表示存在共线性问题

---

## 二、特异收益与特异风险验证：MRAD 与 Bias 指标体系

特异风险（specific risk）部分是模型性能验证的重点，CNE5 采用以下两类核心指标：

### 1. MRAD（Mean Ratio of Actual to Predicted volatility）

**定义**：

\[
\text{MRAD} = \frac{1}{N}\sum_{i=1}^{N} \frac{\sigma_{\text{realized},i}}{\sigma_{\text{predicted},i}}
\]

表示预测风险与实际波动的偏离程度。MRAD 越接近 1，说明风险预测越准确。

**函数**：`calculate_mrad()`

**验证结果**：文中分别绘制了 CHE2 与 CNE5 在不同时期（2000–2012）下的 MRAD 时间序列。CNE5 在三个版本（S/L/D）中均表现出更接近 1 的均值与更小的波动范围。

**理想值**：接近 1.0  
**范围**：CNE5 验证结果通常在 0.23–0.29 区间

### 2. Bias 统计量（Mean Bias Statistic）

**定义**：平均预测偏差，衡量系统性高估或低估风险的程度。Bias ≈ 1 表示无系统偏差。

**函数**：`calculate_bias_statistic()`

**公式**：Bias = mean_actual / mean_predicted

CNE5 通过引入**贝叶斯收缩（Bayesian Shrinkage）**与**Volatility Regime Adjustment (VRA)** 显著降低了 Bias 偏离。验证图 5.17–5.19 展示了 CHE2 与 CNE5 在短期、长期、日度模型下的对比：CNE5 模型在所有版本中 Bias 接近 1，而 CHE2 通常存在低估风险（Bias < 1）的倾向。

**理想值**：接近 1.0  
**范围**：CNE5 验证结果通常在 0.92–1.03 区间

### 3. 时间段与样本覆盖

- 验证期覆盖 1999–2011 年，样本包含所有 A 股
- 结果统计表（Table 5.9）汇总了不同测试组合（纯因子、随机组合、因子倾斜组合、优化组合、特异风险组合）的平均 MRAD 与 Bias 值，均保持在 0.23–0.29 与 0.92–1.03 区间内，表现出高一致性

---

## 三、协方差矩阵与系统风险验证：OBA 与 VRA 框架

CNE5 引入了两项系统性偏差修正机制，用于验证并改进因子协方差矩阵的预测能力：

### 1. Optimization Bias Adjustment (OBA)

**问题来源**：样本协方差矩阵的特征因子（eigenfactors）存在系统性偏差——低波动特征因子风险被低估，高波动因子风险被高估。

**验证方法**：通过 Monte Carlo 模拟估计各 eigenfactor 的偏差，并在协方差矩阵中调整特征波动率。

**函数**：`validate_oba_effectiveness()`

**返回指标**：
- `eigenvalue_bias_before`: 调整前的特征值偏差
- `eigenvalue_bias_after`: 调整后的特征值偏差
- `improvement`: 改善程度

**效果评估**：CNE5 对比 CHE2 的优化组合风险预测显示，修正后的模型显著降低了优化后组合的低估现象，MRAD 更接近 1。

### 2. Volatility Regime Adjustment (VRA)

**目的**：修正风险预测在市场波动体制转换（如危机期）的系统偏差。

**验证方法**：构造**截面偏差统计量（cross-sectional bias statistic）**，作为当日风险预测偏差的即时度量，对因子波动与特异风险分别计算加权平均调整系数。

**函数**：`validate_vra_effectiveness()`

**返回指标**：
- `adjustment_factor`: 调整因子
- `predicted_vol_before`: 调整前的预测波动
- `predicted_vol_after`: 调整后的预测波动
- `actual_vol`: 实际波动
- `improvement`: 预测误差改善程度

**验证结果**：经 VRA 校准后，模型在高波动期的风险预测显著改善，特异风险与整体因子波动预测的时间一致性提升。

---

## 四、综合验证结论

文末总结了模型有效性的整体结论：

- CNE5 在各类验证指标上（MRAD、Bias、R²、因子显著性）均优于旧模型 CHE2
- OBA 与 VRA 机制有效减少了系统性偏差
- 日度特异风险模型和贝叶斯收缩技术显著提升了风险预测的稳定性与解释力
- 因子解释度在长期保持稳定，适用于不同投资期限（S/L/D 模型）

---

## 使用方法

### 基本使用

```python
import polars as pl
from barra_copycat.cne5 import build_complete_cne5_model
from verification import (
    calculate_all_factor_validations,
    calculate_specific_risk_validations,
    calculate_all_covariance_validations,
    aggregate_validation_results,
)

# 构建 CNE5 模型
model_results = build_complete_cne5_model(
    returns_df,
    mkt_cap_df,
    sector_df,
    # ... 其他参数
)

# 因子层面验证
factor_validations = calculate_all_factor_validations(
    returns_df,
    model_results["factor_returns"],
    model_results["residual_returns"],
    model_results["style_factors"],
    sector_df,
    mkt_cap_df,
)

# 特异风险验证
specific_risk_validations = calculate_specific_risk_validations(
    model_results["residual_returns"],
    model_results["specific_risks"],
)

# 协方差验证
covariance_validations = calculate_all_covariance_validations(
    model_results["factor_returns"],
    model_results["residual_returns"],
    model_results["specific_risks"],
    model_results["factor_covariance"],
    n_assets=len(symbols),
)

# 汇总所有验证结果
all_validations = aggregate_validation_results(
    factor_validations,
    specific_risk_validations,
    covariance_validations,
)
```

### 单独使用各个验证函数

```python
from verification import (
    calculate_factor_t_statistics,
    calculate_r_squared,
    calculate_mrad,
    calculate_bias_statistic,
    validate_oba_effectiveness,
    validate_vra_effectiveness,
)

# 计算 t 统计量
t_stats = calculate_factor_t_statistics(
    returns_df,
    factor_returns_df,
    style_df,
    sector_df,
    mkt_cap_df,
)

# 计算 R²
r_squared = calculate_r_squared(
    returns_df,
    factor_returns_df,
    residual_returns_df,
    style_df,
    sector_df,
    mkt_cap_df,
)

# 计算 MRAD
mrad = calculate_mrad(
    residual_returns_df,
    specific_risk_df,
    window=252,
    annualize=True,
)

# 计算 Bias
bias = calculate_bias_statistic(
    residual_returns_df,
    specific_risk_df,
    window=252,
    annualize=True,
)
```

---

## 验证指标汇总

| 指标 | 定义 | 理想值 | CNE5 典型范围 |
|------|------|--------|---------------|
| **MRAD** | (1/N) * sum(σ_realized / σ_predicted) | 接近 1.0 | 0.23–0.29 |
| **Bias** | mean_actual / mean_predicted | 接近 1.0 | 0.92–1.03 |
| **R²** | 1 - SS_res / SS_tot | 越高越好 | - |
| **t-statistics** | 因子收益回归的 t 统计量 | \|t\| > 2 | - |
| **风格稳定性** | 月度相邻期暴露的截面相关 | > 0.90 | - |
| **VIF** | 方差膨胀因子 | < 5 | - |

---

## 文件结构

```
verification/
├── __init__.py                    # 模块导出
├── factor_validation.py          # 因子层面验证
├── specific_risk_validation.py   # 特异风险验证
├── covariance_validation.py     # 协方差验证
├── utils.py                      # 工具函数
├── Verification.md               # 原始验证规范文档（已合并到本文件）
└── README.md                     # 本文件（完整文档）
```

---

## 参考文档

- 《Barra China Equity Model (CNE5) Empirical Notes, July 2012》
- CNE5 模型实现：`barra_copycat.cne5`
- 验证规范：`verification/Verification.md`（原始文档）
