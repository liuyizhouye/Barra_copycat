# CNE5 模型验证框架

> 基于《Barra China Equity Model (CNE5) Empirical Notes, July 2012》

本模块实现了CNE5模型的**完整验证功能**，包括因子验证、特异风险验证和协方差验证。

---

## 验证框架总览

CNE5模型验证分为三个层面：

1. **因子层面** - 解释力与稳定性检验
2. **特异风险** - MRAD与Bias指标体系（**使用前向验证**）
3. **协方差矩阵** - OBA与VRA框架

---

## 一、因子层面验证

### 1. 截面回归显著性（t-statistics）

评估因子是否能显著解释个股超额收益。

**函数**：`calculate_factor_t_statistics()`

**返回指标**：
- `t_statistic`: t统计量，|t| > 2 表示在5%水平下显著
- `p_value`: p值
- `coefficient`: 回归系数

### 2. R² 与 RMS 分解

将横截面收益离散度拆分为因子贡献与特异收益贡献。

**函数**：`calculate_r_squared()`

**返回指标**：
- `r_squared`: R² = 1 - SS_res / SS_tot
- `rms_total`: 总RMS收益
- `{factor}_contribution`: 各因子的RMS贡献

### 3. 稳定性与共线性诊断

**函数**：
- `calculate_style_stability()` - 风格稳定性系数
- `calculate_vif()` - 方差膨胀因子
- `calculate_factor_correlation()` - 因子相关系数

**理想值**：
- 风格稳定性：> 0.90
- VIF：< 5

---

## 二、特异风险验证（⚠️ 使用前向验证方法）

### 关键改进：前向验证方法

本模块采用**前向验证**（Forward Validation）方法，避免前视偏差（Look-ahead Bias）：

- ✅ 在日期t的预测，使用t+1到t+forward_window的数据验证
- ✅ 真正的out-of-sample验证
- ✅ 符合时间序列预测的学术和工业标准

### 1. MRAD（Mean Ratio of Actual to Predicted Volatility）

$$
\text{MRAD} = \frac{1}{N}\sum_{i=1}^{N} \frac{\sigma_{\text{realized},i}}{\sigma_{\text{predicted},i}}
$$

**函数**：`calculate_mrad()`

**参数**：
- `forward_window`: 前向验证窗口（默认21天）
- `annualize`: 是否年化

**理想值**：接近1.0  
**CNE5典型范围**：0.23–0.29（使用前向验证后，实际值可能有所不同）

**重要说明**：
- ⚠️ 最后`forward_window`天的数据无法验证，会被排除
- ✅ 实际波动使用预测日期**之后**的数据计算

### 2. Bias 统计量

$$
\text{Bias} = \frac{\text{mean}(\sigma_{\text{actual}})}{\text{mean}(\sigma_{\text{predicted}})}
$$

**函数**：`calculate_bias_statistic()`

**参数**：同MRAD

**理想值**：接近1.0  
**CNE5典型范围**：0.92–1.03（使用前向验证后，实际值可能有所不同）

**重要说明**：同样使用前向验证方法

---

## 三、协方差矩阵验证

### 1. OBA（Optimization Bias Adjustment）验证

评估OBA对协方差矩阵特征值偏差的修正效果。

**函数**：`validate_oba_effectiveness()`

**返回指标**：
- `eigenvalue_bias_before`: 调整前的特征值偏差
- `eigenvalue_bias_after`: 调整后的特征值偏差
- `improvement`: 改善程度

### 2. VRA（Volatility Regime Adjustment）验证

评估VRA对波动体制转换的适应能力。

**函数**：`validate_vra_effectiveness()`

**返回指标**：
- `adjustment_factor`: 调整因子
- `predicted_vol_before/after`: 调整前后的预测波动
- `actual_vol`: 实际波动
- `improvement`: 预测误差改善程度

---

## 使用方法

### 基本使用

```python
from cne5.cne5 import build_complete_cne5_model
from verification import (
    calculate_all_factor_validations,
    calculate_specific_risk_validations,
    calculate_all_covariance_validations,
)

# 构建CNE5模型
model_results = build_complete_cne5_model(
    returns_df, mkt_cap_df, sector_df, ...
)

# 1. 因子层面验证
factor_validations = calculate_all_factor_validations(
    returns_df,
    model_results["factor_returns"],
    model_results["residual_returns"],
    model_results["style_factors"],
    sector_df,
    mkt_cap_df,
)

# 2. 特异风险验证（前向验证）
specific_risk_validations = calculate_specific_risk_validations(
    model_results["residual_returns"],
    model_results["specific_risks"],
    forward_window=21,  # 前向验证窗口
)

# 3. 协方差验证
covariance_validations = calculate_all_covariance_validations(
    model_results["factor_returns"],
    model_results["residual_returns"],
    model_results["specific_risks"],
    model_results["factor_covariance"],
    n_assets=len(symbols),
)
```

### 单独使用MRAD/Bias验证

```python
from verification import calculate_mrad, calculate_bias_statistic

# MRAD验证（前向验证）
mrad = calculate_mrad(
    residual_returns_df,
    specific_risk_df,
    forward_window=21,  # 可选：5(周), 21(月), 63(季)
    annualize=True,
)

# Bias验证（前向验证）
bias = calculate_bias_statistic(
    residual_returns_df,
    specific_risk_df,
    forward_window=21,
    annualize=True,
)

print(f"平均MRAD: {mrad['mrad'].mean():.4f}")
print(f"平均Bias: {bias['bias'].mean():.4f}")
```

---

## 验证指标汇总

| 指标 | 定义 | 理想值 | 验证方法 |
|------|------|--------|---------|
| **MRAD** | (1/N) Σ(σ_realized / σ_predicted) | 接近1.0 | ✅ 前向验证 |
| **Bias** | mean_actual / mean_predicted | 接近1.0 | ✅ 前向验证 |
| **R²** | 1 - SS_res / SS_tot | 越高越好 | 截面回归 |
| **t-statistics** | 因子收益回归的t统计量 | \|t\| > 2 | 截面回归 |
| **风格稳定性** | 月度相邻期暴露的截面相关 | > 0.90 | 时间序列 |
| **VIF** | 方差膨胀因子 | < 5 | 多重回归 |

---

## 前向验证方法说明

### 为什么使用前向验证？

传统方法使用历史数据验证历史预测，存在**前视偏差**：
- 🔴 用训练集验证训练集
- 🔴 验证结果过于乐观
- 🔴 不符合时间序列预测规范

### 前向验证方法

- ✅ 预测在日期t，使用t+1到t+forward_window的数据验证
- ✅ 真正的out-of-sample验证
- ✅ 符合学术和工业标准

### 参数选择建议

**`forward_window` 推荐值**：
- **21天**（默认）- 适合月度预测验证
- **5天** - 适合周度预测验证
- **63天** - 适合季度预测验证

**注意**：
- 窗口越大，可验证的日期越少（最后forward_window天无法验证）
- 窗口应该与预测期限相匹配

---

## 模块结构

```
verification/
├── __init__.py                    # 模块导出
├── factor_validation.py          # 因子层面验证
├── specific_risk_validation.py   # 特异风险验证（前向验证）
├── covariance_validation.py     # 协方差验证
└── utils.py                      # 工具函数
```

---

## 参考文档

- 《Barra China Equity Model (CNE5) Empirical Notes, July 2012》
- [CNE5模型规范](../cne5/README.md) - 因子定义和理论
- [项目主README](../README.md) - 项目架构和使用指南
- [特异风险验证代码](specific_risk_validation.py) - MRAD/Bias前向验证实现
