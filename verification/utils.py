"""验证模块的通用工具函数。"""

import numpy as np
import polars as pl


def calculate_portfolio_risk(
    weights: np.ndarray,
    factor_cov: np.ndarray,
    factor_exposures: np.ndarray,
    specific_risks: np.ndarray,
) -> float:
    """计算组合风险。

    参数
    ----------
    weights: 组合权重 (n_assets,)
    factor_cov: 因子协方差矩阵 (n_factors, n_factors)
    factor_exposures: 因子暴露矩阵 (n_assets, n_factors)
    specific_risks: 特异风险 (n_assets,)

    返回
    -------
    组合总风险（年化）
    """
    # 因子风险
    factor_risk = weights @ factor_exposures @ factor_cov @ factor_exposures.T @ weights
    
    # 特异风险
    specific_risk = np.sum((weights ** 2) * (specific_risks ** 2))
    
    # 总风险
    total_risk = np.sqrt(factor_risk + specific_risk)
    
    return total_risk


def calculate_realized_volatility(
    returns: np.ndarray,
    window: int = 252,
    annualize: bool = True,
) -> float:
    """计算已实现波动率。

    参数
    ----------
    returns: 收益序列
    window: 窗口长度
    annualize: 是否年化

    返回
    -------
    已实现波动率
    """
    if len(returns) < window:
        return np.nan
    
    recent_returns = returns[-window:]
    vol = np.std(recent_returns)
    
    if annualize:
        vol *= np.sqrt(252)
    
    return vol


def aggregate_validation_results(
    factor_validations: dict,
    specific_risk_validations: dict,
    covariance_validations: dict,
) -> dict:
    """汇总所有验证结果。

    参数
    ----------
    factor_validations: 因子验证结果
    specific_risk_validations: 特异风险验证结果
    covariance_validations: 协方差验证结果

    返回
    -------
    汇总的验证结果字典
    """
    return {
        "factor_validations": factor_validations,
        "specific_risk_validations": specific_risk_validations,
        "covariance_validations": covariance_validations,
        "summary": {
            "factor_t_statistics_mean": (
                factor_validations["t_statistics"]["t_statistic"].mean()
                if "t_statistics" in factor_validations
                else np.nan
            ),
            "r_squared_mean": (
                factor_validations["r_squared"]["r_squared"].mean()
                if "r_squared" in factor_validations
                else np.nan
            ),
            "mrad_mean": (
                specific_risk_validations["mrad"]["mrad"].mean()
                if "mrad" in specific_risk_validations
                else np.nan
            ),
            "bias_mean": (
                specific_risk_validations["bias"]["bias"].mean()
                if "bias" in specific_risk_validations
                else np.nan
            ),
        },
    }

