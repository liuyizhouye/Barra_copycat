"""协方差矩阵与系统风险验证：OBA 与 VRA 框架。

包括：
- Optimization Bias Adjustment (OBA) 验证
- Volatility Regime Adjustment (VRA) 验证
"""

import numpy as np
import polars as pl

from cne5.cne5_covariance import (
    estimate_factor_covariance,
    optimization_bias_adjustment,
    volatility_regime_adjustment,
)


def validate_oba_effectiveness(
    factor_returns_df: pl.DataFrame,
    n_assets: int,
    window: int = 252,
    half_life: int | None = 63,
    n_simulations: int = 1000,
) -> dict:
    """验证 OBA (Optimization Bias Adjustment) 的有效性。

    通过对比调整前后的协方差矩阵特征值偏差，评估 OBA 的效果。

    参数
    ----------
    factor_returns_df: Polars DataFrame，因子收益
    n_assets: 资产数量
    window: 协方差估计窗口，默认252
    half_life: 半衰期，默认63
    n_simulations: OBA 模拟次数，默认1000

    返回
    -------
    字典，包含验证结果：
    {
        'eigenvalue_bias_before': 调整前的特征值偏差,
        'eigenvalue_bias_after': 调整后的特征值偏差,
        'improvement': 改善程度,
        'factor_cov_before': 调整前的协方差矩阵,
        'factor_cov_after': 调整后的协方差矩阵,
    }
    """
    # 估计原始协方差矩阵
    factor_cov_before = estimate_factor_covariance(
        factor_returns_df,
        window=window,
        half_life=half_life,
        method="ewma" if half_life is not None else "sample",
    )
    
    # 应用 OBA
    factor_cov_after = optimization_bias_adjustment(
        factor_cov_before,
        n_assets=n_assets,
        n_simulations=n_simulations,
    )
    
    # 计算特征值
    eigenvals_before, _ = np.linalg.eigh(factor_cov_before)
    eigenvals_after, _ = np.linalg.eigh(factor_cov_after)
    
    # 计算偏差（简化：使用特征值的相对变化）
    # 实际应用中，可以通过 Monte Carlo 模拟估计真实偏差
    eigenvalue_bias_before = np.std(eigenvals_before) / np.mean(eigenvals_before) if np.mean(eigenvals_before) > 1e-10 else np.nan
    eigenvalue_bias_after = np.std(eigenvals_after) / np.mean(eigenvals_after) if np.mean(eigenvals_after) > 1e-10 else np.nan
    
    improvement = (eigenvalue_bias_before - eigenvalue_bias_after) / eigenvalue_bias_before if eigenvalue_bias_before > 1e-10 else np.nan
    
    return {
        "eigenvalue_bias_before": eigenvalue_bias_before,
        "eigenvalue_bias_after": eigenvalue_bias_after,
        "improvement": improvement,
        "factor_cov_before": factor_cov_before,
        "factor_cov_after": factor_cov_after,
        "eigenvals_before": eigenvals_before,
        "eigenvals_after": eigenvals_after,
    }


def validate_vra_effectiveness(
    factor_returns_df: pl.DataFrame,
    residual_returns_df: pl.DataFrame,
    specific_risk_df: pl.DataFrame,
    factor_cov: np.ndarray,
    window: int = 60,
    half_life: int = 20,
) -> pl.DataFrame:
    """验证 VRA (Volatility Regime Adjustment) 的有效性。

    通过对比调整前后的风险预测准确性，评估 VRA 的效果。

    参数
    ----------
    factor_returns_df: Polars DataFrame，因子收益
    residual_returns_df: Polars DataFrame，残差收益
    specific_risk_df: Polars DataFrame，特异风险预测
    factor_cov: 因子协方差矩阵
    window: VRA 回看窗口，默认60
    half_life: VRA 半衰期，默认20

    返回
    -------
    Polars DataFrame，包含各日期的 VRA 调整效果
    列：| date | adjustment_factor | predicted_vol_before | predicted_vol_after | actual_vol |
    """
    # 获取因子列
    factor_cols = [col for col in factor_returns_df.columns if col != "date"]
    dates = factor_returns_df["date"].unique().sort().to_list()
    
    # 获取最新日期的特异风险
    latest_date = specific_risk_df["date"].max()
    latest_risks = (
        specific_risk_df.filter(pl.col("date") == latest_date)
        .select(["symbol", "specific_risk_shrunk" if "specific_risk_shrunk" in specific_risk_df.columns else "specific_risk"])
        .to_dicts()
    )
    specific_risks_dict = {
        item["symbol"]: item["specific_risk_shrunk" if "specific_risk_shrunk" in specific_risk_df.columns else "specific_risk"]
        for item in latest_risks
    }
    
    results = []
    for date in dates[-window:]:  # 只验证最近 window 个日期
        # 获取到当前日期的因子收益
        factor_data = factor_returns_df.filter(pl.col("date") <= date).sort("date")
        
        # 计算实际波动（最近 window 个值）
        recent_factor_returns = (
            factor_data.tail(window).select(factor_cols).to_numpy()
        )
        
        if len(recent_factor_returns) < window:
            continue
        
        actual_vol = np.std(recent_factor_returns, axis=0)
        
        # 计算预测波动（从协方差矩阵）
        predicted_vol_before = np.sqrt(np.diag(factor_cov))
        
        # 应用 VRA
        factor_cov_adjusted, specific_risks_adjusted = volatility_regime_adjustment(
            factor_returns_df.filter(pl.col("date") <= date),
            residual_returns_df.filter(pl.col("date") <= date),
            factor_cov,
            specific_risks_dict,
            window=window,
            half_life=half_life,
        )
        
        predicted_vol_after = np.sqrt(np.diag(factor_cov_adjusted))
        
        # 计算调整因子
        adjustment_factor = np.mean(predicted_vol_after / predicted_vol_before) if np.mean(predicted_vol_before) > 1e-10 else 1.0
        
        # 计算预测误差改善
        error_before = np.mean(np.abs(actual_vol - predicted_vol_before))
        error_after = np.mean(np.abs(actual_vol - predicted_vol_after))
        improvement = (error_before - error_after) / error_before if error_before > 1e-10 else 0.0
        
        results.append({
            "date": date,
            "adjustment_factor": adjustment_factor,
            "predicted_vol_before": np.mean(predicted_vol_before),
            "predicted_vol_after": np.mean(predicted_vol_after),
            "actual_vol": np.mean(actual_vol),
            "error_before": error_before,
            "error_after": error_after,
            "improvement": improvement,
        })
    
    return pl.DataFrame(results)


def calculate_all_covariance_validations(
    factor_returns_df: pl.DataFrame,
    residual_returns_df: pl.DataFrame,
    specific_risk_df: pl.DataFrame,
    factor_cov: np.ndarray,
    n_assets: int,
    window: int = 252,
    half_life: int | None = 63,
) -> dict:
    """计算所有协方差矩阵验证指标。

    返回包含所有验证结果的字典。
    """
    return {
        "oba_validation": validate_oba_effectiveness(
            factor_returns_df, n_assets, window, half_life
        ),
        "vra_validation": validate_vra_effectiveness(
            factor_returns_df,
            residual_returns_df,
            specific_risk_df,
            factor_cov,
        ),
    }

