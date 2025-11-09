"""CNE5 因子协方差矩阵实现。

包括：
- 因子协方差矩阵估计
- 优化偏差调整（OBA）
- 波动体制调整（VRA）
"""

from typing import Literal, Optional, Dict, Tuple

import numpy as np
import polars as pl

from cne5.math import exp_weights


def estimate_factor_covariance(
    factor_returns_df: pl.DataFrame,
    window: int = 252,
    half_life: Optional[int] = None,
    method: Literal["sample", "ewma"] = "ewma",
) -> np.ndarray:
    """估计因子协方差矩阵。

    参数
    ----------
    factor_returns_df: Polars DataFrame，包含列：| date | country | sector1 | ... | style1 | ... |
    window: 估计窗口长度(交易日)，默认252
    half_life: 半衰期(交易日)，如果为 None 则使用等权，默认 None
    method: 估计方法，"sample" 或 "ewma"，默认 "ewma"

    返回
    -------
    numpy array，因子协方差矩阵 (n_factors x n_factors)
    """
    # 获取因子列（排除 date）
    factor_cols = [col for col in factor_returns_df.columns if col != "date"]
    factor_data = factor_returns_df.select(factor_cols).to_numpy()

    if method == "sample":
        # 简单样本协方差
        if len(factor_data) < window:
            return np.cov(factor_data.T)
        return np.cov(factor_data[-window:].T)

    elif method == "ewma":
        # 指数加权移动平均协方差
        if half_life is None:
            # 等权
            if len(factor_data) < window:
                return np.cov(factor_data.T)
            return np.cov(factor_data[-window:].T)

        # 使用指数权重
        weights = exp_weights(window, half_life)
        weights = weights / weights.sum()  # 归一化

        if len(factor_data) < window:
            return np.cov(factor_data.T)

        # 计算加权协方差
        data = factor_data[-window:]
        n_factors = data.shape[1]

        # 加权均值
        weighted_mean = np.average(data, axis=0, weights=weights)

        # 加权协方差
        cov_matrix = np.zeros((n_factors, n_factors))
        for i in range(n_factors):
            for j in range(n_factors):
                cov_matrix[i, j] = np.average(
                    (data[:, i] - weighted_mean[i]) * (data[:, j] - weighted_mean[j]),
                    weights=weights,
                )

        return cov_matrix

    else:
        raise ValueError(f"未知的方法: {method}")


def optimization_bias_adjustment(
    factor_cov: np.ndarray,
    n_assets: int,
    n_simulations: int = 1000,
) -> np.ndarray:
    """优化偏差调整（OBA）。

    使用蒙特卡洛方法估计特征因子的偏差，并调整因子协方差矩阵。

    参数
    ----------
    factor_cov: 因子协方差矩阵 (n_factors x n_factors)
    n_assets: 资产数量
    n_simulations: 蒙特卡洛模拟次数，默认1000

    返回
    -------
    numpy array，调整后的因子协方差矩阵
    """
    n_factors = factor_cov.shape[0]

    # 特征值分解
    eigenvals, eigenvecs = np.linalg.eigh(factor_cov)
    eigenvals = np.maximum(eigenvals, 1e-10)  # 避免负值

    # 蒙特卡洛模拟估计偏差
    np.random.seed(42)  # 可复现
    bias_factors = np.ones(n_factors)

    for i in range(n_simulations):
        # 生成随机因子收益
        random_returns = np.random.multivariate_normal(
            np.zeros(n_factors), factor_cov, size=n_assets
        )

        # 计算样本协方差
        sample_cov = np.cov(random_returns.T)

        # 特征值分解
        sample_eigenvals, _ = np.linalg.eigh(sample_cov)

        # 计算偏差（简化版本）
        for j in range(n_factors):
            if eigenvals[j] > 1e-10:
                bias_ratio = sample_eigenvals[j] / eigenvals[j]
                bias_factors[j] = (bias_factors[j] * i + bias_ratio) / (i + 1)

    # 调整特征值
    adjusted_eigenvals = eigenvals / bias_factors

    # 重构协方差矩阵
    adjusted_cov = eigenvecs @ np.diag(adjusted_eigenvals) @ eigenvecs.T

    return adjusted_cov


def volatility_regime_adjustment(
    factor_returns_df: pl.DataFrame,
    residual_returns_df: pl.DataFrame,
    factor_cov: np.ndarray,
    specific_risks: Dict[str, float],
    window: int = 60,
    half_life: int = 20,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """波动体制调整（VRA）。

    基于截面偏差统计量调整因子波动和特异风险。

    参数
    ----------
    factor_returns_df: Polars DataFrame，因子收益
    residual_returns_df: Polars DataFrame，残差收益
    factor_cov: 因子协方差矩阵
    specific_risks: 特异风险字典 {symbol: risk}
    window: 回看窗口，默认60
    half_life: 半衰期，默认20

    返回
    -------
    元组：(调整后的因子协方差矩阵, 调整后的特异风险字典)
    """
    # 计算截面偏差统计量
    # 简化实现：使用最近的实际波动与预测波动的比率

    # 获取最近的因子收益
    factor_cols = [col for col in factor_returns_df.columns if col != "date"]
    recent_factor_returns = (
        factor_returns_df.sort("date").tail(window).select(factor_cols).to_numpy()
    )

    # 计算实际波动
    actual_vol = np.std(recent_factor_returns, axis=0)

    # 计算预测波动（从协方差矩阵对角线）
    predicted_vol = np.sqrt(np.diag(factor_cov))

    # 计算调整因子（避免除零）
    adjustment_factor = np.where(
        predicted_vol > 1e-10, actual_vol / predicted_vol, 1.0
    )

    # 使用指数加权平均平滑调整因子
    weights = exp_weights(window, half_life)
    weights = weights / weights.sum()

    # 计算加权平均调整因子
    if len(adjustment_factor) > len(weights):
        adjustment_factor = adjustment_factor[-len(weights):]
    weighted_adjustment = np.average(adjustment_factor, weights=weights[-len(adjustment_factor):])

    # 调整因子协方差矩阵
    adjusted_factor_cov = factor_cov * (weighted_adjustment ** 2)

    # 调整特异风险
    adjusted_specific_risks = {
        symbol: risk * weighted_adjustment for symbol, risk in specific_risks.items()
    }

    return adjusted_factor_cov, adjusted_specific_risks

