# Barra Copycat

Barra Copycat 是一个机构级多因子股权风险模型，用于量化交易和系统化投资。该项目实现了类似 Barra 和 Axioma 的特征因子模型，能够创建自定义因子并估计其收益，也可用于构建因子协方差矩阵以支撑带约束的组合优化（如市场中性的因子敞口）。

![mom_factor](https://github.com/user-attachments/assets/f9d2927c-e899-4fd6-944c-8f9a104b410f)

核心依赖：**numpy**（数值计算）和 **polars**（高性能数据处理）。支持市场、行业和风格因子；内置价值、规模和动量；并提供通用的数学与数据清洗工具函数。

## 安装

使用 pip 安装：

```bash
pip install Barra-copycat
```

---

## 项目架构详解

### 文件结构

```
toraniko/
├── __init__.py          # 空文件，标识为Python包
├── model.py             # 核心模型：因子收益估计
├── styles.py            # 风格因子构建
├── math.py              # 数学和统计工具
├── utils.py             # 数据清洗工具
└── tests/               # 测试文件
    ├── test_model.py    # 测试 model.py
    ├── test_math.py     # 测试 math.py
    └── test_utils.py    # 测试 utils.py
```

### 核心功能

1. **构建风格因子**：动量(Momentum)、价值(Value)、规模(Size)
2. **估计因子收益**：市场、行业、风格
3. **计算残差收益**：用于风险分解
4. **支持组合优化**：提供因子协方差矩阵的计算基础

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
from toraniko.model import estimate_factor_returns

factor_returns_df, residual_returns_df = estimate_factor_returns(
    returns_df,        # 日度收益数据
    mkt_cap_df,       # 市值数据
    sector_df,        # 行业分类数据
    style_df,         # 风格得分数据
    winsor_factor=0.05,      # 去掉极端5%
    residualize_styles=True   # 正交化风格因子
)
```

### styles.py - 风格因子模块

三种经典风格因子：

#### 1. `factor_mom()` - 动量因子

使用指数加权滚动收益计算动量得分。

```python
from toraniko.styles import factor_mom

mom_scores = factor_mom(
    returns_df,
    trailing_days=252,   # 1年回看
    half_life=126,       # 半年半衰期
    lag=20,              # 1个月滞后
    winsor_factor=0.01   # 去掉1%极端值
).collect()
```

![mom_factor](https://github.com/user-attachments/assets/88983248-a982-4c9e-9048-c01f1e7d191a)

#### 2. `factor_sze()` - 规模因子

基于市值对数，实现 Small-Minus-Big（SMB）。

```python
from toraniko.styles import factor_sze

sze_scores = factor_sze(mkt_cap_df).collect()
```

#### 3. `factor_val()` - 价值因子

基于价格比率（book-price, sales-price, cf-price）计算价值得分。

```python
from toraniko.styles import factor_val

val_scores = factor_val(value_df).collect()
```

![value_factor](https://github.com/user-attachments/assets/ca5c1afc-128e-4cd6-9871-6d7eb0e77ebc)

### math.py - 数学统计工具模块

横截面标准化、去极值、归一化等工具。

**主要函数**：
- `center_xsection()`：横截面去中心化/标准化
- `winsorize()` / `winsorize_xsection()`：去极值
- `percentiles_xsection()`：分位数截断
- `exp_weights()`：指数衰减权重
- `norm_xsection()`：归一化到指定区间

### utils.py - 数据清洗工具模块

- `fill_features()`：前向填充缺失值
- `smooth_features()`：滚动均值平滑
- `top_n_by_group()`：分组取前N名

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

### 2. 构建风格因子

```python
from toraniko.styles import factor_mom, factor_sze, factor_val

# 动量因子
mom_scores = factor_mom(returns_df).collect()

# 规模因子
sze_scores = factor_sze(mkt_cap_df).collect()

# 价值因子
val_scores = factor_val(value_df).collect()

# 合并所有风格因子
style_df = mom_scores.join(sze_scores, on=["date", "symbol"]) \
                     .join(val_scores, on=["date", "symbol"])
```

### 3. 估计因子收益

```python
from toraniko.utils import top_n_by_group
from toraniko.model import estimate_factor_returns

# 合并数据并取市值前3000（如 Russell 3000）
ddf = (
    ret_df.join(cap_df, on=["date", "symbol"])
    .join(sector_df, on="symbol")
    .join(style_df, on=["date", "symbol"])
    .drop_nulls()
)
ddf = top_n_by_group(ddf.lazy(), 3000, "market_cap", ("date",), True).collect()

# 估计因子收益
fac_df, eps_df = estimate_factor_returns(
    returns_df, mkt_cap_df, sector_df, style_df,
    winsor_factor=0.1, residualize_styles=False
)

print(fac_df.head())
# date | market | Technology | Healthcare | ... | mom_score | sze_score | val_score
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

### 开发依赖
- pytest ~7.4.4

---

## 测试

项目使用 pytest 进行单元测试：

```bash
pytest toraniko/tests/
```

---

## 总结

- **model.py**：核心算法
- **styles.py**：动量/规模/价值
- **math.py + utils.py**：数据处理
- **测试**：保证代码质量

可基于本库：
1. 进行量化分析
2. 扩展新因子
3. 调参优化
4. 构建完整风险系统

**注意**：本项目不包含数据获取功能。您需要自行准备金融数据。