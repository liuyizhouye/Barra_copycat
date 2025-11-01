# Barra Copycat 项目代码架构详解

本文档详细解释了 Barra Copycat 项目的文件结构、每个模块的功能以及各个函数的作用，帮助您深入理解这个多因子风险模型项目的代码。

## 目录

- [项目概述](#项目概述)
- [文件架构](#文件架构)
- [核心模块详解](#核心模块详解)
  - [model.py - 因子收益估计模块](#modelpy---因子收益估计模块)
  - [styles.py - 风格因子模块](#stylespy---风格因子模块)
  - [math.py - 数学统计工具模块](#mathpy---数学统计工具模块)
  - [utils.py - 数据清洗工具模块](#utilspy---数据清洗工具模块)
- [测试模块](#测试模块)
- [项目依赖](#项目依赖)
- [工作流程](#工作流程)

---

## 项目概述

Barra Copycat 是一个机构级多因子股权风险模型，用于量化交易和系统化投资。该项目实现了类似 Barra 和 Axioma 的特征因子模型，能够：

1. **构建风格因子**：动量(Momentum)、价值(Value)、规模(Size)
2. **估计因子收益**：市场因子、行业因子、风格因子
3. **计算残差收益**：用于风险分解
4. **支持组合优化**：提供因子协方差矩阵的计算基础

核心依赖：**numpy**（数值计算）和 **polars**（高性能数据处理）

---

## 文件架构

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

---

## 核心模块详解

### model.py - 因子收益估计模块

这是项目的**核心模块**，实现了完整的因子收益估计逻辑。

#### 主要函数

##### 1. `_factor_returns()` - 单期因子收益估计

```python
def _factor_returns(
    returns: np.ndarray,
    mkt_caps: np.ndarray,
    sector_scores: np.ndarray,
    style_scores: np.ndarray,
    residualize_styles: bool,
) -> tuple[np.ndarray, np.ndarray]
```

**功能**：估计单个时间点的市场因子、行业因子、风格因子和残差收益

**参数说明**：
- `returns`: 资产收益数组 (n_assets x 1)
- `mkt_caps`: 市值数组 (n_assets x 1)，用于加权
- `sector_scores`: 行业暴露矩阵 (n_assets x n_sectors)
- `style_scores`: 风格暴露矩阵 (n_assets x n_styles)
- `residualize_styles`: 是否将风格因子正交化到市场+行业因子

**返回值**：
- 因子收益数组：(市场因子 + 行业因子 + 风格因子)
- 残差收益数组：(资产收益 - 因子解释部分)

**核心算法**：

1. **加权矩阵构建**：
   ```python
   W = np.diag(np.sqrt(mkt_caps.ravel()))
   ```
   使用市值的平方根作为权重，给予大盘股更高权重

2. **市场+行业因子估计**（带约束）：
   - 约束：所有行业因子收益之和为0
   - 经济含义：市场收益完全由行业收益线性组合构成
   - 使用加权最小二乘法(WLS)求解

3. **风格因子估计**：
   - 根据 `residualize_styles` 参数决定是否正交化
   - 如果 True：基于去除市场+行业后的残差
   - 如果 False：直接基于资产收益

4. **残差计算**：
   ```python
   epsilon = sector_resid_returns - (style_scores @ fac_ret_style)
   ```

**代码示例**：
```python
returns = np.array([[0.01], [0.02], [-0.01]])  # 3个资产的收益
mkt_caps = np.array([[1000], [500], [200]])     # 3个资产的市值
sector_scores = np.array([[1,0], [1,0], [0,1]]) # 属于2个行业
style_scores = np.array([[0.5], [0.3], [-0.2]]) # 风格得分

fac_ret, epsilon = _factor_returns(
    returns, mkt_caps, sector_scores, style_scores, residualize_styles=True
)
```

##### 2. `estimate_factor_returns()` - 多期因子收益估计

```python
def estimate_factor_returns(
    returns_df: pl.DataFrame,
    mkt_cap_df: pl.DataFrame,
    sector_df: pl.DataFrame,
    style_df: pl.DataFrame,
    winsor_factor: float | None = 0.05,
    residualize_styles: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame]
```

**功能**：对所有时间点进行因子收益估计

**参数说明**：
- `returns_df`: Polars DataFrame，列包含 date, symbol, asset_returns
- `mkt_cap_df`: Polars DataFrame，列包含 date, symbol, market_cap
- `sector_df`: Polars DataFrame，列包含 date, symbol, 以及每个行业的一列
- `style_df`: Polars DataFrame，列包含 date, symbol, 以及每个风格的一列
- `winsor_factor`: 去极值阈值（如 0.05 表示去掉极端 5%）
- `residualize_styles`: 是否正交化风格因子

**返回值**：
- 因子收益 DataFrame：每行一个日期，列为各因子
- 残差收益 DataFrame：每行一个日期，列为各资产的残差

**工作流程**：
1. 验证输入 DataFrame 结构
2. 合并所有输入数据（按日期和股票代码）
3. 逐个日期循环：
   - 提取当日数据
   - 对收益去极值（如果指定）
   - 调用 `_factor_returns()` 估计当日因子收益
   - 收集结果
4. 组装最终结果

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

print(factor_returns_df.head())
# date | market | Technology | Healthcare | ... | mom_score | val_score
```

---

### styles.py - 风格因子模块

该模块实现了三种经典风格因子的构建：**动量、规模、价值**。

#### 主要函数

##### 1. `factor_mom()` - 动量因子

```python
def factor_mom(
    returns_df: pl.DataFrame | pl.LazyFrame,
    trailing_days: int = 504,
    half_life: int = 126,
    lag: int = 20,
    winsor_factor: float = 0.01,
) -> pl.LazyFrame
```

**功能**：计算每只股票的滚动动量得分

**参数说明**：
- `returns_df`: 包含 date, symbol, asset_returns 的 DataFrame
- `trailing_days`: 回看天数（默认504天，约2年）
- `half_life`: 指数衰减半衰期（默认126天，约半年）
- `lag`: 滞后期天数（默认20天，约1个月，避免日内效应）
- `winsor_factor`: 去极值阈值

**计算逻辑**：
1. 使用指数衰减权重（越近的数据权重越大）
2. 对滞后后的收益序列应用滚动加权累积收益
3. 横截面去极值
4. 横截面标准化（去均值、除以标准差）

**返回**：LazyFrame，列包含 date, symbol, mom_score

**代码示例**：
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

##### 2. `factor_sze()` - 规模因子

```python
def factor_sze(
    mkt_cap_df: pl.DataFrame | pl.LazyFrame,
    lower_decile: float = 0.2,
    upper_decile: float = 0.8,
) -> pl.LazyFrame
```

**功能**：计算每只股票的规模得分（基于市值）

**参数说明**：
- `mkt_cap_df`: 包含 date, symbol, market_cap 的 DataFrame
- `lower_decile`: 下分位数阈值（默认0.2）
- `upper_decile`: 上分位数阈值（默认0.8）

**计算逻辑**：
1. 对市值取对数
2. **乘以 -1**（实现 Small-Minus-Big，因为小盘股风险溢价更高）
3. 横截面标准化
4. 应用分位数截断（保留极端值，中间值设为0）

**返回**：LazyFrame，列包含 date, symbol, sze_score

**注意**：该因子遵循 Fama-French 的 SMB（Small-Minus-Big）逻辑

##### 3. `factor_val()` - 价值因子

```python
def factor_val(
    value_df: pl.DataFrame | pl.LazyFrame,
    winsorize_features: float | None = None
) -> pl.LazyFrame
```

**功能**：计算每只股票的价值得分（基于价格比率）

**参数说明**：
- `value_df`: 包含 date, symbol, book_price, sales_price, cf_price 的 DataFrame
- `winsorize_features`: 是否对特征去极值

**计算逻辑**：
1. （可选）对特征去极值
2. 对 book_price 和 sales_price 取对数
3. 横截面标准化每个特征（去均值、除以标准差）
4. **取平均**：将三个标准化后的特征取均值
5. 再次横截面标准化

**返回**：LazyFrame，列包含 date, symbol, val_score

**代码示例**：
```python
from toraniko.styles import factor_val

# value_df 需要包含 book_price, sales_price, cf_price 列
val_scores = factor_val(
    value_df,
    winsorize_features=0.05  # 对特征去极值
).collect()
```

---

### math.py - 数学统计工具模块

该模块提供了所有因子构建和数据处理所需的数学和统计工具函数。

#### 主要函数

##### 1. `center_xsection()` - 横截面去中心化/标准化

```python
def center_xsection(
    target_col: str,
    over_col: str,
    standardize: bool = False
) -> pl.Expr
```

**功能**：对某一列按分组去中心化（可选标准化）

**参数说明**：
- `target_col`: 目标列名
- `over_col`: 分组列名（通常是 date）
- `standardize`: 是否除以标准差

**返回值**：Polars 表达式（可以链式调用）

**数学公式**：
- 去中心化：`x - mean(x)`
- 标准化：`(x - mean(x)) / std(x)`

**用途**：将不同时间点的因子得分拉到同一尺度，消除时间趋势

##### 2. `norm_xsection()` - 横截面归一化

```python
def norm_xsection(
    target_col: str,
    over_col: str,
    lower: int | float = 0,
    upper: int | float = 1,
) -> pl.Expr
```

**功能**：将数值缩放到 [lower, upper] 区间

**数学公式**：
```
normalized = (x - min(x)) / (max(x) - min(x)) * (upper - lower) + lower
```

##### 3. `winsorize()` - 去极值（NumPy版本）

```python
def winsorize(
    data: np.ndarray,
    percentile: float = 0.05,
    axis: int = 0
) -> np.ndarray
```

**功能**：将极端值裁剪到分位数

**逻辑**：
- 找到 percentile 和 1-percentile 分位数
- 小于下限的设为下限，大于上限的设为上限

**示例**：
```python
# percentile=0.05 表示将极端5%的值裁剪掉
data = np.array([1, 2, 3, 100, 200])
winsorized = winsorize(data, percentile=0.05)
# 结果：[1, 2, 3, 3, 3]  （如果100和200是极端值）
```

##### 4. `winsorize_xsection()` - 横截面去极值

```python
def winsorize_xsection(
    df: pl.DataFrame | pl.LazyFrame,
    data_cols: tuple[str, ...],
    group_col: str,
    percentile: float = 0.05,
) -> pl.DataFrame | pl.LazyFrame
```

**功能**：对多个列按分组去极值

**与 `winsorize()` 的区别**：这是 Polars 版本，可以处理分组数据

##### 5. `percentiles_xsection()` - 分位数截断

```python
def percentiles_xsection(
    target_col: str,
    over_col: str,
    lower_pct: float,
    upper_pct: float,
    fill_val: float | int = 0.0,
) -> pl.Expr
```

**功能**：保留分位数以外的值，中间值用 fill_val 填充

**用途**：构建 Long-Short 组合（只关注极端值）

**示例**：
```python
# 保留 top 20% 和 bottom 20%，中间设为 0
df.with_columns(
    percentiles_xsection("score", "date", 0.2, 0.8, 0.0).alias("score")
)
```

##### 6. `exp_weights()` - 指数衰减权重

```python
def exp_weights(window: int, half_life: int) -> np.ndarray
```

**功能**：生成指数衰减权重数组

**数学公式**：
```
decay = log(2) / half_life
weights[i] = exp(-decay * (window - 1 - i))
```

**示例**：
```python
weights = exp_weights(window=10, half_life=3)
# 越近的数据权重越大，每3个时间段权重减半
```

---

### utils.py - 数据清洗工具模块

该模块提供了数据清洗和预处理功能，确保进入模型的数质量良好。

#### 主要函数

##### 1. `fill_features()` - 填充缺失值

```python
def fill_features(
    df: pl.DataFrame | pl.LazyFrame,
    features: tuple[str, ...],
    sort_col: str,
    over_col: str
) -> pl.LazyFrame
```

**功能**：对特征列进行前向填充

**处理流程**：
1. 将特征列转为浮点型
2. 将 NaN、Inf、字符串"NaN"转为 NULL
3. 按 sort_col 排序
4. 按 over_col 分组，对每组的 NULL 值前向填充

**用途**：处理财务报表数据的缺失值（季度报告的数据可以使用到下一季度）

**示例**：
```python
df = pl.DataFrame({
    "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
    "symbol": ["AAPL", "AAPL", "AAPL"],
    "book_value": [100, None, None]
})

filled = fill_features(df, ("book_value",), "date", "symbol").collect()
# book_value: [100, 100, 100]  # 后两行被填充
```

##### 2. `smooth_features()` - 平滑特征

```python
def smooth_features(
    df: pl.DataFrame | pl.LazyFrame,
    features: tuple[str, ...],
    sort_col: str,
    over_col: str,
    window_size: int,
) -> pl.LazyFrame
```

**功能**：用滚动均值平滑特征

**逻辑**：
- 按时间排序
- 按股票分组
- 计算每个特征的滚动窗口均值

**用途**：降低财务数据的噪音

##### 3. `top_n_by_group()` - 分组取前N名

```python
def top_n_by_group(
    df: pl.DataFrame | pl.LazyFrame,
    n: int,
    rank_var: str,
    group_var: tuple[str, ...],
    filter: bool = True,
) -> pl.LazyFrame
```

**功能**：每个分组内按某列排名，取前n名

**参数说明**：
- `n`: 取前几名
- `rank_var`: 排序列名
- `group_var`: 分组列名（如 "date"）
- `filter`: 
  - True: 只返回前n名
  - False: 返回所有行，添加 rank_mask 列标识是否在前n名

**用途**：构建等权重投资组合（如 Russell 3000）

**示例**：
```python
# 每天取市值最大的3000只股票
top_stocks = top_n_by_group(
    df,
    n=3000,
    rank_var="market_cap",
    group_var=("date",),
    filter=True
).collect()
```

---

## 测试模块

项目使用 `pytest` 进行单元测试，测试文件位于 `toraniko/tests/` 目录下。

### test_model.py

测试 `_factor_returns()` 和 `estimate_factor_returns()` 的：
- **输出维度正确性**
- **行业约束满足性**（行业因子和为0）
- **可重复性**（相同输入相同输出）
- **市值加权有效性**
- **残差正交性**

### test_math.py

测试数学工具函数的：
- **去中心化正确性**
- **标准化正确性**
- **归一化范围**
- **去极值阈值**

### test_utils.py

测试数据清洗函数的：
- **填充逻辑**
- **平滑效果**
- **排名正确性**

---

## 项目依赖

### 必需依赖

- **numpy ~1.26.2**: 数值计算核心库
- **polars ~1.0.0**: 高性能数据处理库

### 开发依赖

- **pytest ~7.4.4**: 单元测试框架

安装命令：
```bash
pip install -r requirements.txt
```

---

## 工作流程

### 1. 数据准备

您需要准备以下数据（以 Polars DataFrame 格式）：

```python
# 1. 收益数据
returns_df = pl.DataFrame({
    "date": [...],
    "symbol": [...],
    "asset_returns": [...]
})

# 2. 市值数据
mkt_cap_df = pl.DataFrame({
    "date": [...],
    "symbol": [...],
    "market_cap": [...]
})

# 3. 行业分类数据（One-hot编码）
sector_df = pl.DataFrame({
    "date": [...],
    "symbol": [...],
    "Technology": [...],      # 0或1
    "Healthcare": [...],
    ...
})

# 4. （可选）基本面数据（用于价值因子）
value_df = pl.DataFrame({
    "date": [...],
    "symbol": [...],
    "book_price": [...],
    "sales_price": [...],
    "cf_price": [...]
})
```

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
from toraniko.model import estimate_factor_returns

# 估计所有时间的因子收益
factor_returns_df, residual_returns_df = estimate_factor_returns(
    returns_df,
    mkt_cap_df,
    sector_df,
    style_df,
    winsor_factor=0.05,
    residualize_styles=True
)

print(factor_returns_df)
# date | market | Technology | Healthcare | ... | mom_score | sze_score | val_score

print(residual_returns_df)
# date | AAPL | MSFT | GOOGL | ...  （各股票的残差收益）
```

### 4. 后续应用

- **风险分解**：分析资产收益的因子暴露和特异性风险
- **组合优化**：基于因子协方差矩阵构建最优组合
- **风险模型**：预测投资组合的 VaR、跟踪误差等

---

## 关键设计理念

### 1. 横截面标准化

所有因子得分都进行横截面标准化（去均值、除标准差），确保：
- 不同时间的因子得分在同一尺度
- 因子之间可以组合比较

### 2. 市值加权

因子收益估计使用市值加权最小二乘法，给予大盘股更高权重，反映：
- 大盘股对市场的真实影响
- 符合机构投资者的关注点

### 3. 行业约束

行业因子收益之和为零，确保：
- 市场收益完全由行业因子线性组合构成
- 避免行业因子的共线性问题

### 4. 可扩展性

通过 Polars LazyFrame 实现：
- 延迟计算，优化性能
- 链式操作，代码优雅
- 内存高效，处理大规模数据

---

## 总结

Barra Copycat 是一个完整且优雅的多因子风险模型实现：

- **模型模块**（model.py）：核心估计算法
- **因子模块**（styles.py）：经典风格因子
- **工具模块**（math.py + utils.py）：数据处理流水线
- **测试完善**：确保代码质量

通过理解这些模块的功能和交互，您可以：
1. 使用现有功能进行量化分析
2. 扩展新的风格因子
3. 优化模型参数
4. 构建完整的风险管理系统

希望这份文档帮助您深入理解项目代码！
