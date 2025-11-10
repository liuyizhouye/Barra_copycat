# CNE5 模型规范文档

> 基于《Barra China Equity Model (CNE5) Model Description, July 2012》

本文档为CNE5模型的**理论规范**，包含因子定义、计算方法和模型参数。

**注意**：本文档仅说明理论规范，代码实现和使用方法请参考项目主 [README.md](../README.md)。

---

## 一、模型总体结构

### 三类因子

CNE5 模型包含三类因子：

1. **国家因子（Country Factor）**
   - 单一国家模型中的市值加权国家投资组合
   - 使行业因子变成"美元中性"组合（100%多行业，100%空国家）

2. **行业因子（Industry Factors）**
   - 基于GICS体系，针对中国市场定制
   - 约束条件：所有行业因子收益之和为0

3. **风格因子（Style Factors）** - 10个
   - Size, Beta, Momentum, Residual Volatility, Non-linear Size
   - Book-to-Price, Liquidity, Earnings Yield, Growth, Leverage

---

## 二、标准化与正交化规范

### 标准化方法

所有风格因子采用CNE5标准化：
- **市值加权均值** = 0
- **等权标准差** = 1

### 正交化关系

为降低多重共线性，部分因子需要正交化：

| 因子 | 正交化目标 |
|------|-----------|
| Residual Volatility | Beta + Size |
| Liquidity | Size |
| Non-linear Size | Size |

---

## 三、风格因子定义

### 1) Size

* **定义**：1.0 · LNCAP
* **LNCAP**：公司总市值的自然对数

### 2) Beta

* **定义**：1.0 · BETA
* **计算**：对超额收益做时间序列回归
  - 窗口：252交易日
  - 半衰期：63日
  - 基准：市值加权超额收益

### 3) Momentum

* **定义**：1.0 · RSTR
* **计算**：滞后21日后，过去504日的超额对数收益指数加权和
  - 滞后期：L=21日
  - 回看期：T=504日
  - 半衰期：126日

### 4) Residual Volatility

* **定义**：0.74 · DASTD + 0.16 · CMRA + 0.10 · HSIGMA
* **DASTD**：过去252日日度超额收益的波动率，半衰期42日
* **CMRA**：12个月累计超额对数收益的区间范围
* **HSIGMA**：回归残差的波动率（252日窗口，半衰期63日）
* **正交化**：对Beta与Size正交

### 5) Non-linear Size

* **定义**：1.0 · NLSIZE
* **计算**：标准化后Size暴露立方，相对Size回归加权正交化，再winsorize和标准化

### 6) Book-to-Price

* **定义**：1.0 · BTOP
* **BTOP**：最新账面普通股权益 / 当前市值

### 7) Liquidity

* **定义**：0.35 · STOM + 0.35 · STOQ + 0.30 · STOA
* **STOM**：过去21交易日日度换手率之和的对数
* **STOQ**：逐月STOM的log-exp平均，T=3
* **STOA**：逐月STOM的log-exp平均，T=12
* **正交化**：对Size正交

### 8) Earnings Yield

* **定义**：0.68 · EPFWD + 0.21 · CETOP + 0.11 · ETOP
* **EPFWD**：未来12个月预测盈利 / 当前市值
* **CETOP**：过去12个月现金盈利 / 现价
* **ETOP**：过去12个月盈利 / 当前市值

### 9) Growth

* **定义**：0.18 · EGRLF + 0.11 · EGRSF + 0.24 · EGRO + 0.47 · SGRO
* **EGRLF**：分析师长期（3-5年）盈利增速预测
* **EGRSF**：分析师短期（1年）盈利增速预测
* **EGRO**：过去5年EPS回归斜率/平均EPS
* **SGRO**：过去5年销售额回归斜率/平均销售额

### 10) Leverage

* **定义**：0.38 · MLEV + 0.35 · DTOA + 0.27 · BLEV
* **MLEV**：市值杠杆 = (ME+PE+LD)/ME
* **DTOA**：负债/资产 = TD/TA
* **BLEV**：账面杠杆 = (BE+PE+LD)/BE

---

## 四、因子收益估计

### 截面回归方法

使用加权最小二乘法（WLS）估计因子收益：

$$
r_n = \sum_k X_{nk} f_k + u_n
$$

### 回归权重

回归权重与市值平方根成正比：

$$
v_n \propto \sqrt{\text{MarketCap}_n}
$$

目的：降低小盘股噪声，提升估计稳定性。

---

## 五、因子协方差矩阵

### 基础估计方法

1. **样本协方差**：使用历史窗口（通常252天）
2. **EWMA协方差**：指数加权移动平均（半衰期通常63天）

### 偏差修正机制

#### 1. 优化偏差调整（OBA）

**问题**：样本协方差的特征因子存在系统性偏差

**方法**：蒙特卡洛模拟
1. 对特征值按大小排序匹配
2. 估计各特征因子的偏差系数
3. 调整特征值后重构协方差矩阵

#### 2. 波动体制调整（VRA）

**问题**：波动非平稳性导致风险预测偏差

**方法**：截面偏差统计量
1. 计算实际波动与预测波动的比率
2. 使用指数加权平均平滑调整因子
3. 同时调整因子波动和特异风险

---

## 六、特异风险模型

### 估计方法

#### 1. 日频序列法

使用EWMA估计个股特异收益的波动率：
- 窗口：252交易日
- 半衰期：63交易日
- 年化公式：$\sigma_{\text{annual}} = \sigma_{\text{daily}} \times \sqrt{252}$

#### 2. 贝叶斯收缩

按市值分位数分桶（通常10桶），向桶内均值收缩：

$$
\sigma_{\text{shrunk}} = \lambda \cdot \sigma_{\text{individual}} + (1-\lambda) \cdot \sigma_{\text{bucket}}
$$

收缩强度：

$$
\lambda = \frac{1}{1 + \left(\frac{\sigma_{\text{individual}} - \mu_{\text{bucket}}}{\sigma_{\text{bucket}}}\right)^2}
$$

### 调整机制

特异风险同样应用VRA（波动体制调整）。

---

## 七、模型版本

CNE5提供三个版本，暴露与因子收益相同，差别在协方差和特异风险的响应度：

| 版本 | 适用期限 | 特点 |
|------|---------|------|
| CNE5S | 短期（月度） | 响应快，适合短期预测 |
| CNE5L | 长期（季度+） | 平滑，适合长期配置 |
| CNE5D | 日度 | 最及时，适合日度风控 |

---

## 八、诊断指标

### 风格稳定性系数

月度相邻期暴露的截面相关性：
- **< 0.80**：偏低，因子不稳定
- **> 0.90**：理想，因子稳定

### 方差膨胀因子（VIF）

衡量因子间共线性：
- **< 5**：正常
- **> 5**：存在共线性问题

---

## 参考文档

- 《Barra China Equity Model (CNE5) Model Description, July 2012》
- 《Barra China Equity Model (CNE5) Empirical Notes, July 2012》
- [项目主 README.md](../README.md) - 代码实现和使用指南
- [验证模块 README.md](../verification/README.md) - 模型验证方法
