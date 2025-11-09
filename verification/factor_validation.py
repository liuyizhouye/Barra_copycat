"""因子层面验证：解释力与稳定性检验。

包括：
- 截面回归显著性（t-statistics）
- R² 与 RMS 分解
- 稳定性与共线性诊断
"""

from typing import Optional

import numpy as np
import polars as pl

try:
    from scipy import stats
except ImportError:
    # 如果没有 scipy，使用 numpy 的近似实现
    stats = None

from cne5.cne5_risk import (
    calculate_rms_decomposition,
    calculate_style_stability,
    calculate_vif,
)


def calculate_factor_t_statistics(
    returns_df: pl.DataFrame,
    factor_returns_df: pl.DataFrame,
    style_df: pl.DataFrame,
    sector_df: pl.DataFrame,
    mkt_cap_df: pl.DataFrame,
) -> pl.DataFrame:
    """计算因子收益的截面回归显著性（t-statistics）。

    对每个风格因子，进行月度因子收益回归，考察因子是否能显著解释个股超额收益。

    参数
    ----------
    returns_df: Polars DataFrame，包含列：| date | symbol | asset_returns |
    factor_returns_df: Polars DataFrame，因子收益，包含列：| date | country | sector1 | ... | style1 | ... |
    style_df: Polars DataFrame，风格因子暴露
    sector_df: Polars DataFrame，行业因子暴露
    mkt_cap_df: Polars DataFrame，市值数据

    返回
    -------
    Polars DataFrame，包含各因子的 t-statistics 和 p-values
    列：| date | factor_name | t_statistic | p_value | coefficient |
    """
    # 合并数据
    df = (
        returns_df.join(mkt_cap_df, on=["date", "symbol"])
        .join(style_df, on=["date", "symbol"])
        .join(sector_df, on=["date", "symbol"])
    )

    dates = df["date"].unique().to_list()
    style_cols = [col for col in style_df.columns if col not in ["date", "symbol"]]
    sector_cols = [col for col in sector_df.columns if col not in ["date", "symbol"]]
    all_factors = ["country"] + sector_cols + style_cols

    results = []
    for date in dates:
        date_data = df.filter(pl.col("date") == date)
        returns = date_data["asset_returns"].to_numpy()
        weights = np.sqrt(date_data["market_cap"].to_numpy())  # 市值平方根权重

        for factor_name in all_factors:
            # 获取因子暴露
            if factor_name == "country":
                exposures = np.ones(len(returns))
            elif factor_name in sector_cols:
                exposures = date_data[factor_name].to_numpy()
            else:
                exposures = date_data[factor_name].to_numpy()

            # 获取因子收益
            factor_return = (
                factor_returns_df.filter(pl.col("date") == date)[factor_name].to_numpy()[0]
            )

            # 加权回归：returns = alpha + beta * exposures + epsilon
            # 使用加权最小二乘
            X = np.vstack([np.ones(len(exposures)), exposures]).T
            W = np.diag(weights)
            
            try:
                # 加权最小二乘：beta = (X'WX)^(-1) X'Wy
                XWX = X.T @ W @ X
                XWy = X.T @ W @ returns
                beta = np.linalg.solve(XWX, XWy)
                
                # 计算残差
                y_pred = X @ beta
                residuals = returns - y_pred
                
                # 计算标准误
                n = len(returns)
                k = 2  # 截距 + 斜率
                mse = np.sum(weights * residuals ** 2) / (np.sum(weights) * (n - k))
                
                # 计算系数标准误
                var_beta = mse * np.linalg.inv(XWX)
                se_beta = np.sqrt(np.diag(var_beta))
                
                # t-statistic for slope (index 1)
                t_stat = beta[1] / se_beta[1] if se_beta[1] > 1e-10 else 0.0
                
                # p-value (two-tailed)
                if stats is not None:
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - k))
                else:
                    # 如果没有 scipy，使用简单的阈值判断
                    # 对于大样本，t 分布接近正态分布
                    # |t| > 2 时 p < 0.05 (近似)
                    if abs(t_stat) > 2.576:  # 99% 置信水平
                        p_value = 0.01
                    elif abs(t_stat) > 1.96:  # 95% 置信水平
                        p_value = 0.05
                    elif abs(t_stat) > 1.645:  # 90% 置信水平
                        p_value = 0.10
                    else:
                        p_value = 0.5
                
                results.append({
                    "date": date,
                    "factor_name": factor_name,
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "coefficient": beta[1],
                })
            except (np.linalg.LinAlgError, ValueError):
                results.append({
                    "date": date,
                    "factor_name": factor_name,
                    "t_statistic": np.nan,
                    "p_value": np.nan,
                    "coefficient": np.nan,
                })

    return pl.DataFrame(results)


def calculate_r_squared(
    returns_df: pl.DataFrame,
    factor_returns_df: pl.DataFrame,
    residual_returns_df: pl.DataFrame,
    style_df: pl.DataFrame,
    sector_df: pl.DataFrame,
    mkt_cap_df: pl.DataFrame,
) -> pl.DataFrame:
    """计算 R² 与 RMS 分解。

    将横截面收益离散度拆分为因子贡献与特异收益贡献，
    再细分为行业、风格、国家因子。

    参数
    ----------
    returns_df: Polars DataFrame，包含列：| date | symbol | asset_returns |
    factor_returns_df: Polars DataFrame，因子收益
    residual_returns_df: Polars DataFrame，残差收益
    style_df: Polars DataFrame，风格因子暴露
    sector_df: Polars DataFrame，行业因子暴露
    mkt_cap_df: Polars DataFrame，市值数据

    返回
    -------
    Polars DataFrame，包含 R² 和 RMS 分解结果
    """
    # 使用已有的 RMS 分解函数
    rms_decomp = calculate_rms_decomposition(
        returns_df, factor_returns_df, style_df, sector_df, mkt_cap_df
    )

    # 计算 R²
    dates = returns_df["date"].unique().to_list()
    results = []
    
    for date in dates:
        date_returns = returns_df.filter(pl.col("date") == date)["asset_returns"].to_numpy()
        date_residuals = (
            residual_returns_df.filter(pl.col("date") == date)
            .select(pl.exclude("date"))
            .to_numpy()
            .ravel()
        )
        
        # 只取有残差的股票
        valid_mask = ~np.isnan(date_residuals)
        if np.sum(valid_mask) == 0:
            continue
            
        returns_valid = date_returns[valid_mask]
        residuals_valid = date_residuals[valid_mask]
        
        # 总方差
        ss_tot = np.sum(returns_valid ** 2)
        # 残差方差
        ss_res = np.sum(residuals_valid ** 2)
        
        # R² = 1 - SS_res / SS_tot
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else np.nan
        
        # 获取 RMS 分解结果
        rms_row = rms_decomp.filter(pl.col("date") == date)
        if len(rms_row) > 0:
            rms_dict = rms_row.to_dicts()[0]
            results.append({
                "date": date,
                "r_squared": r_squared,
                **rms_dict,
            })
        else:
            results.append({
                "date": date,
                "r_squared": r_squared,
            })

    return pl.DataFrame(results)


def calculate_factor_correlation(
    style_df: pl.DataFrame,
    sector_df: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """计算因子暴露的截面相关系数，用于稳定性与共线性诊断。

    参数
    ----------
    style_df: Polars DataFrame，风格因子暴露
    sector_df: 可选，行业因子暴露

    返回
    -------
    Polars DataFrame，包含因子间的相关系数矩阵（按日期）
    """
    dates = style_df["date"].unique().to_list()
    style_cols = [col for col in style_df.columns if col not in ["date", "symbol"]]
    
    results = []
    for date in dates:
        date_data = style_df.filter(pl.col("date") == date)
        
        # 计算风格因子间的相关系数矩阵
        style_matrix = date_data.select(style_cols).to_numpy()
        corr_matrix = np.corrcoef(style_matrix.T)
        
        # 转换为长格式
        for i, factor1 in enumerate(style_cols):
            for j, factor2 in enumerate(style_cols):
                if i <= j:  # 只保存上三角（包括对角线）
                    results.append({
                        "date": date,
                        "factor1": factor1,
                        "factor2": factor2,
                        "correlation": corr_matrix[i, j],
                    })
    
    return pl.DataFrame(results)


def calculate_all_factor_validations(
    returns_df: pl.DataFrame,
    factor_returns_df: pl.DataFrame,
    residual_returns_df: pl.DataFrame,
    style_df: pl.DataFrame,
    sector_df: pl.DataFrame,
    mkt_cap_df: pl.DataFrame,
) -> dict:
    """计算所有因子层面验证指标。

    返回包含所有验证结果的字典。
    """
    return {
        "t_statistics": calculate_factor_t_statistics(
            returns_df, factor_returns_df, style_df, sector_df, mkt_cap_df
        ),
        "r_squared": calculate_r_squared(
            returns_df, factor_returns_df, residual_returns_df, style_df, sector_df, mkt_cap_df
        ),
        "style_stability": calculate_style_stability(style_df),
        "vif": calculate_vif(style_df, sector_df),
        "factor_correlation": calculate_factor_correlation(style_df, sector_df),
    }

