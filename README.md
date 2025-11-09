# Barra Copycat

Barra Copycat 是一个机构级多因子股权风险模型，用于量化交易和系统化投资。该项目实现了类似 Barra 和 Axioma 的特征因子模型，能够创建自定义因子并估计其收益，也可用于构建因子协方差矩阵以支撑带约束的组合优化（如市场中性的因子敞口）。

![mom_factor](https://github.com/user-attachments/assets/f9d2927c-e899-4fd6-944c-8f9a104b410f)

核心依赖：**numpy**（数值计算）和 **polars**（高性能数据处理）。实现完整的 Barra CNE5 多因子风险模型，包括10个风格因子、因子收益估计、协方差矩阵和特异风险模型。

## 安装

使用 pip 安装：

```bash
pip install Barra-copycat
```

---

## 项目架构详解

### 文件结构

```
cne5/
├── __init__.py              # 包初始化文件
├── cne5.py                  # CNE5 模型主模块（入口）
├── cne5_covariance.py       # 因子协方差矩阵（OBA、VRA）
├── cne5_risk.py            # 特异风险模型（贝叶斯收缩、RMS分解等）
├── cne5_factors/           # CNE5 风格因子（每个因子独立文件）
│   ├── __init__.py         # 导出所有因子函数
│   ├── _utils.py           # 共享工具函数（标准化、正交化）
│   ├── size.py             # Size 因子
│   ├── beta.py             # Beta 因子
│   ├── momentum.py         # Momentum 因子
│   ├── residual_volatility.py  # Residual Volatility 因子
│   ├── nonlinear_size.py   # Non-linear Size 因子
│   ├── book_to_price.py    # Book-to-Price 因子
│   ├── liquidity.py       # Liquidity 因子
│   ├── earnings_yield.py  # Earnings Yield 因子
│   ├── growth.py          # Growth 因子
│   └── leverage.py        # Leverage 因子
├── model.py                 # 核心模型：因子收益估计
└── math.py                  # 数学和统计工具
```

### 核心功能

1. **CNE5 风格因子构建**：10个CNE5标准风格因子
2. **因子收益估计**：国家、行业、风格因子收益
3. **因子协方差矩阵**：支持 OBA 和 VRA 调整
4. **特异风险模型**：日频序列法 + 贝叶斯收缩
5. **诊断工具**：RMS分解、风格稳定性、VIF

---

## 文件详细说明与依赖关系

### 核心模块依赖关系图

```
用户代码
    ↓
cne5.py (主入口)
    ├──→ cne5_factors/ (10个风格因子)
    │       ├──→ _utils.py (标准化、正交化)
    │       └──→ math.py (exp_weights, winsorize_xsection等)
    ├──→ model.py (因子收益估计)
    │       └──→ math.py (winsorize)
    ├──→ cne5_covariance.py (协方差矩阵)
    │       └──→ math.py (exp_weights)
    └──→ cne5_risk.py (特异风险)
            └──→ math.py (exp_weights)
```

### 各文件详细说明

#### 1. `cne5.py` - CNE5 模型主入口

**作用**：整合 CNE5 模型的所有功能，提供高级接口。

**主要函数**：
- `build_cne5_style_factors()`: 构建所有10个风格因子
- `estimate_cne5_factor_returns()`: 估计因子收益（国家、行业、风格）
- `build_cne5_risk_model()`: 构建完整风险模型
- `build_complete_cne5_model()`: 一键构建完整 CNE5 模型

**依赖关系**：
- 导入 `cne5_factors/` 中的所有因子函数
- 导入 `model.py` 的 `estimate_factor_returns()`
- 导入 `cne5_covariance.py` 的协方差相关函数
- 导入 `cne5_risk.py` 的风险相关函数

**使用场景**：用户主要使用的模块，通过此模块访问所有 CNE5 功能。

---

#### 2. `cne5_factors/` - CNE5 风格因子模块

**作用**：实现 CNE5 模型的10个风格因子，每个因子独立文件。

##### 2.1 `_utils.py` - 共享工具函数

**作用**：提供 CNE5 因子构建的共享工具函数。

**主要函数**：
- `standardize_cne5()`: CNE5 标准化（市值加权均值=0，等权标准差=1）
- `orthogonalize_factor()`: 回归加权正交化

**依赖关系**：
- 不依赖其他模块（使用 Polars 和 NumPy 原生功能）

**被依赖**：所有因子文件都使用此模块的标准化和正交化函数。

##### 2.2 `size.py` - Size 因子

**作用**：计算 Size 因子（市值对数）。

**依赖关系**：
- 导入 `_utils.py` 的 `standardize_cne5`

##### 2.3 `beta.py` - Beta 因子

**作用**：计算 Beta 因子（市场 Beta）。

**依赖关系**：
- 导入 `_utils.py` 的 `standardize_cne5`
- 导入 `math.py` 的 `exp_weights`（用于 EWMA）

##### 2.4 `momentum.py` - Momentum 因子

**作用**：计算 Momentum 因子（动量）。

**依赖关系**：
- 导入 `_utils.py` 的 `standardize_cne5`
- 导入 `math.py` 的 `exp_weights`

##### 2.5 `residual_volatility.py` - Residual Volatility 因子

**作用**：计算 Residual Volatility 因子（残差波动率）。

**依赖关系**：
- 导入 `_utils.py` 的 `standardize_cne5`, `orthogonalize_factor`
- 导入 `math.py` 的 `exp_weights`
- 依赖 `beta.py` 和 `size.py` 的输出（通过参数传入）

##### 2.6 `nonlinear_size.py` - Non-linear Size 因子

**作用**：计算 Non-linear Size 因子（Size 的立方）。

**依赖关系**：
- 导入 `_utils.py` 的 `standardize_cne5`, `orthogonalize_factor`
- 导入 `math.py` 的 `winsorize_xsection`
- 依赖 `size.py` 的输出（通过参数传入）

##### 2.7 `book_to_price.py` - Book-to-Price 因子

**作用**：计算 Book-to-Price 因子（账面价值/市值）。

**依赖关系**：
- 导入 `_utils.py` 的 `standardize_cne5`

##### 2.8 `liquidity.py` - Liquidity 因子

**作用**：计算 Liquidity 因子（流动性）。

**依赖关系**：
- 导入 `_utils.py` 的 `standardize_cne5`, `orthogonalize_factor`
- 依赖 `size.py` 的输出（通过参数传入）

##### 2.9 `earnings_yield.py` - Earnings Yield 因子

**作用**：计算 Earnings Yield 因子（盈利收益率）。

**依赖关系**：
- 导入 `_utils.py` 的 `standardize_cne5`

##### 2.10 `growth.py` - Growth 因子

**作用**：计算 Growth 因子（成长性）。

**依赖关系**：
- 导入 `_utils.py` 的 `standardize_cne5`

##### 2.11 `leverage.py` - Leverage 因子

**作用**：计算 Leverage 因子（杠杆率）。

**依赖关系**：
- 导入 `_utils.py` 的 `standardize_cne5`

##### 2.12 `__init__.py` - 因子模块导出

**作用**：统一导出所有因子函数，方便 `cne5.py` 导入。

**依赖关系**：
- 导入所有因子文件中的函数

---

#### 3. `model.py` - 因子收益估计核心模块

**作用**：实现因子收益估计的核心算法（加权最小二乘法）。

**主要函数**：
- `_factor_returns()`: 单期因子收益估计（市场、行业、风格）
- `estimate_factor_returns()`: 多期因子收益估计

**核心算法**：
- 使用市值平方根作为权重矩阵
- 行业因子收益约束：所有行业因子收益之和为0
- 风格因子可选正交化到市场+行业

**依赖关系**：
- 导入 `math.py` 的 `winsorize`

**被依赖**：
- `cne5.py` 的 `estimate_cne5_factor_returns()` 调用此模块

**数据流**：
```
returns_df + mkt_cap_df + sector_df + style_df
    ↓
estimate_factor_returns()
    ↓
factor_returns_df (因子收益) + residual_returns_df (残差收益)
```

---

#### 4. `cne5_covariance.py` - 因子协方差矩阵模块

**作用**：实现因子协方差矩阵的估计和调整。

**主要函数**：
- `estimate_factor_covariance()`: 估计因子协方差矩阵（支持样本协方差和 EWMA）
- `optimization_bias_adjustment()`: 优化偏差调整（OBA）
- `volatility_regime_adjustment()`: 波动体制调整（VRA）

**依赖关系**：
- 导入 `math.py` 的 `exp_weights`（用于 EWMA 协方差）

**被依赖**：
- `cne5.py` 的 `build_cne5_risk_model()` 调用此模块

**数据流**：
```
factor_returns_df
    ↓
estimate_factor_covariance()
    ↓
factor_cov (因子协方差矩阵)
    ↓
optimization_bias_adjustment() (可选)
    ↓
volatility_regime_adjustment() (可选)
    ↓
adjusted_factor_cov (调整后的协方差矩阵)
```

---

#### 5. `cne5_risk.py` - 特异风险模型模块

**作用**：实现特异风险（idiosyncratic risk）的估计和诊断工具。

**主要函数**：
- `estimate_specific_risk_ts()`: 日频序列法估计特异风险
- `bayesian_shrinkage()`: 贝叶斯收缩（按市值分位收缩）
- `calculate_rms_decomposition()`: RMS 分解（x-sigma-rho 分解）
- `calculate_style_stability()`: 风格稳定性系数
- `calculate_vif()`: 方差膨胀因子（VIF）

**依赖关系**：
- 导入 `math.py` 的 `exp_weights`（用于 EWMA 特异风险估计）

**被依赖**：
- `cne5.py` 的 `build_cne5_risk_model()` 调用此模块

**数据流**：
```
residual_returns_df
    ↓
estimate_specific_risk_ts()
    ↓
specific_risk_df (原始特异风险)
    ↓
bayesian_shrinkage() (可选)
    ↓
specific_risk_shrunk_df (收缩后的特异风险)
```

---

#### 6. `math.py` - 数学和统计工具模块

**作用**：提供通用的数学和统计工具函数。

**主要函数**：
- `exp_weights()`: 生成指数衰减权重（用于 EWMA）
- `winsorize()` / `winsorize_xsection()`: 去极值处理
- `center_xsection()`: 横截面去中心化/标准化
- `percentiles_xsection()`: 分位数截断
- `norm_xsection()`: 归一化到指定区间

**被依赖**：
- `cne5_factors/beta.py` 使用 `exp_weights`
- `cne5_factors/momentum.py` 使用 `exp_weights`
- `cne5_factors/residual_volatility.py` 使用 `exp_weights`
- `cne5_factors/nonlinear_size.py` 使用 `winsorize_xsection`
- `cne5_covariance.py` 使用 `exp_weights`
- `cne5_risk.py` 使用 `exp_weights`
- `model.py` 使用 `winsorize`

**特点**：底层工具模块，被多个模块广泛使用。

---

### 完整数据流图

```
原始数据
    ↓
[cne5_factors/] 构建风格因子
    ├── size.py → Size 因子
    ├── beta.py → Beta 因子
    ├── momentum.py → Momentum 因子
    ├── residual_volatility.py → Residual Volatility 因子
    ├── nonlinear_size.py → Non-linear Size 因子
    ├── book_to_price.py → Book-to-Price 因子
    ├── liquidity.py → Liquidity 因子
    ├── earnings_yield.py → Earnings Yield 因子
    ├── growth.py → Growth 因子
    └── leverage.py → Leverage 因子
    ↓
style_df (所有风格因子)
    ↓
[model.py] 估计因子收益
    ↓
factor_returns_df (因子收益) + residual_returns_df (残差收益)
    ↓
[cne5_covariance.py] 估计因子协方差矩阵
    ├── estimate_factor_covariance()
    ├── optimization_bias_adjustment() (可选)
    └── volatility_regime_adjustment() (可选)
    ↓
factor_cov (因子协方差矩阵)
    ↓
[cne5_risk.py] 估计特异风险
    ├── estimate_specific_risk_ts()
    └── bayesian_shrinkage() (可选)
    ↓
specific_risk_df (特异风险)
    ↓
完整 CNE5 风险模型
```

### 模块间调用关系总结

1. **用户 → `cne5.py`**：用户主要通过 `cne5.py` 访问所有功能
2. **`cne5.py` → `cne5_factors/`**：构建风格因子
3. **`cne5.py` → `model.py`**：估计因子收益
4. **`cne5.py` → `cne5_covariance.py`**：估计协方差矩阵
5. **`cne5.py` → `cne5_risk.py`**：估计特异风险
6. **所有模块 → `math.py`**：使用数学工具函数
7. **`cne5_factors/` → `_utils.py`**：使用标准化和正交化函数

---

## 核心模块详解

### model.py - 因子收益估计模块

#### `_factor_returns()` - 单期因子收益估计

估计单个时间点的市场、行业、风格和残差收益。

**核心算法**：
- 加权矩阵：使用市值平方根作为权重
- 市场+行业估计（带约束）：行业因子收益之和为0
- 风格估计：可选正交化到市场+行业
- 残差：资产收益减去因子解释部分

#### `estimate_factor_returns()` - 多期因子收益估计

对所有时间点进行因子收益估计，按日循环调用单期估计。

**代码示例**：
```python
from cne5.model import estimate_factor_returns

factor_returns_df, residual_returns_df = estimate_factor_returns(
    returns_df,        # 日度收益数据
    mkt_cap_df,       # 市值数据
    sector_df,        # 行业分类数据
    style_df,         # 风格得分数据
    winsor_factor=0.05,      # 去掉极端5%
    residualize_styles=True   # 正交化风格因子
)
```

### cne5_factors/ - CNE5 风格因子模块

CNE5 模型的10个风格因子，每个因子独立文件：

- `size.py` - Size 因子
- `beta.py` - Beta 因子
- `momentum.py` - Momentum 因子
- `residual_volatility.py` - Residual Volatility 因子
- `nonlinear_size.py` - Non-linear Size 因子
- `book_to_price.py` - Book-to-Price 因子
- `liquidity.py` - Liquidity 因子
- `earnings_yield.py` - Earnings Yield 因子
- `growth.py` - Growth 因子
- `leverage.py` - Leverage 因子

所有因子都遵循 CNE5 标准化规范：市值加权均值=0，等权标准差=1。

### math.py - 数学统计工具模块

横截面标准化、去极值、归一化等工具。

**主要函数**：
- `center_xsection()`：横截面去中心化/标准化
- `winsorize()` / `winsorize_xsection()`：去极值
- `percentiles_xsection()`：分位数截断
- `exp_weights()`：指数衰减权重
- `norm_xsection()`：归一化到指定区间


---

## 使用指南

### 1. 数据准备

需要准备以下数据（Polars DataFrame）：

**行业分类（GICS Level 1）**：
```
symbol	Basic Materials	Communication Services	Consumer Cyclical	...
"A"	0	0	0	...
"AA"	1	0	0	...
```

**收益数据**：
```
date	symbol	asset_returns
2013-01-02	"A"	0.022962
2013-01-02	"AAMC"	-0.073171
...
```

**基本面数据（价值因子）**：
```
date	symbol	book_price	sales_price	cf_price	market_cap
2013-10-30	"AAPL"	0.343017	0.081994	0.007687	4.5701e11
...
```

**注意**：您需要自行准备数据。推荐使用 Yahoo Finance 等数据源。

### 2. 构建 CNE5 风格因子

```python
from cne5.cne5 import build_cne5_style_factors

# 构建所有 CNE5 风格因子
style_df = build_cne5_style_factors(
    returns_df,
    mkt_cap_df,
    risk_free_df,      # 可选
    book_value_df,    # 可选
    turnover_df,      # 可选
    earnings_df,      # 可选
    growth_df,        # 可选
    leverage_df,      # 可选
).collect()
```

### 3. 构建完整 CNE5 模型

```python
from cne5.cne5 import build_complete_cne5_model

# 构建完整的 CNE5 模型（因子 + 收益 + 风险）
results = build_complete_cne5_model(
    returns_df,
    mkt_cap_df,
    sector_df,
    risk_free_df,      # 可选
    book_value_df,    # 可选
    turnover_df,      # 可选
    earnings_df,      # 可选
    growth_df,        # 可选
    leverage_df,      # 可选
)

# 访问结果
style_factors = results['style_factors']
factor_returns = results['factor_returns']      # 包含 country + 行业 + 风格因子收益
factor_covariance = results['factor_covariance']  # 因子协方差矩阵
specific_risks = results['specific_risks']     # 特异风险
```

或者分步构建：

```python
from cne5.cne5 import (
    build_cne5_style_factors,
    estimate_cne5_factor_returns,
    build_cne5_risk_model,
)

# 1. 构建风格因子
style_df = build_cne5_style_factors(...).collect()

# 2. 估计因子收益
factor_returns_df, residual_returns_df = estimate_cne5_factor_returns(
    returns_df, mkt_cap_df, sector_df, style_df
)

# 3. 构建风险模型
factor_cov, specific_risks_df, risk_info = build_cne5_risk_model(
    factor_returns_df, residual_returns_df, mkt_cap_df
)
```

在 M1 MacBook 上，10+ 年日频市场、行业、风格因子收益估计可在 1 分钟内完成。

### 4. 模型验证

即便使用 Yahoo Finance，结果与 Barra 模型接近（10 年）：

![val_factor](https://github.com/user-attachments/assets/28f41989-f802-4c2f-beed-1d2bda24a96d)

![val_comparison](https://github.com/user-attachments/assets/366f49a8-d7e7-46de-bb61-6f656393275a)

---

## 关键设计理念

### 1. 横截面标准化

所有因子得分进行横截面标准化（去均值、除标准差），确保：
- 不同时间的因子在同一尺度
- 因子之间可组合比较

### 2. 市值加权

使用市值加权最小二乘法，给予大盘股更高权重，反映：
- 大盘股的市场影响
- 机构投资者的关注点

### 3. 行业约束

行业因子收益之和为零，确保：
- 市场收益由行业收益线性组合构成
- 避免行业因子共线性

### 4. 可扩展性

通过 Polars LazyFrame：
- 延迟计算
- 链式操作
- 内存高效

---

## 项目依赖

### 必需依赖
- numpy ~1.26.2
- polars ~1.0.0

---

## 总结

- **cne5.py**：CNE5 模型主入口
- **cne5_factors/**：10个CNE5风格因子
- **cne5_covariance.py**：因子协方差矩阵（OBA、VRA）
- **cne5_risk.py**：特异风险模型
- **model.py**：核心因子收益估计算法
- **math.py**：数学和统计工具

可基于本库：
1. 构建完整的 CNE5 风险模型
2. 进行量化分析和因子研究
3. 支持组合优化和风险管理
4. 扩展自定义因子

**注意**：本项目不包含数据获取功能。您需要自行准备金融数据。