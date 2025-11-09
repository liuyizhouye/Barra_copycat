"""CNE5 模型主模块。

整合 CNE5 模型的所有功能，包括：
- 10个风格因子构建
- 国家因子、行业因子、风格因子收益估计
- 因子协方差矩阵估计
- 优化偏差调整（OBA）
- 波动体制调整（VRA）
- 特异风险模型（日频序列法 + 贝叶斯收缩）
- RMS 分解
- 风格稳定性系数
- VIF（方差膨胀因子）
"""

import numpy as np
import polars as pl

from barra_copycat.cne5_factors import (
    factor_size_cne5,
    factor_beta_cne5,
    factor_momentum_cne5,
    factor_residual_volatility_cne5,
    factor_nonlinear_size_cne5,
    factor_book_to_price_cne5,
    factor_liquidity_cne5,
    factor_earnings_yield_cne5,
    factor_growth_cne5,
    factor_leverage_cne5,
)
from barra_copycat.model import estimate_factor_returns
from barra_copycat.cne5_covariance import (
    estimate_factor_covariance,
    optimization_bias_adjustment,
    volatility_regime_adjustment,
)
from barra_copycat.cne5_risk import (
    estimate_specific_risk_ts,
    bayesian_shrinkage,
    calculate_rms_decomposition,
    calculate_style_stability,
    calculate_vif,
)


def build_cne5_style_factors(
    returns_df: pl.DataFrame | pl.LazyFrame,
    mkt_cap_df: pl.DataFrame | pl.LazyFrame,
    risk_free_df: pl.DataFrame | pl.LazyFrame | None = None,
    # 基本面数据（可选）
    book_value_df: pl.DataFrame | pl.LazyFrame | None = None,
    turnover_df: pl.DataFrame | pl.LazyFrame | None = None,
    earnings_df: pl.DataFrame | pl.LazyFrame | None = None,
    growth_df: pl.DataFrame | pl.LazyFrame | None = None,
    leverage_df: pl.DataFrame | pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    """构建所有 CNE5 风格因子。

    参数
    ----------
    returns_df: Polars DataFrame，包含列：| date | symbol | asset_returns |
    mkt_cap_df: Polars DataFrame，包含列：| date | symbol | market_cap |
    risk_free_df: 可选，无风险利率数据，包含列：| date | risk_free_rate |
    book_value_df: 可选，账面价值数据，包含列：| date | symbol | book_value |
    turnover_df: 可选，换手率数据，包含列：| date | symbol | turnover |
    earnings_df: 可选，盈利数据，包含列：| date | symbol | epfwd | cetop | etop |
    growth_df: 可选，成长数据，包含列：| date | symbol | egrlf | egrsf | egro | sgro |
    leverage_df: 可选，杠杆数据，包含列：| date | symbol | mlev | dtoa | blev |

    返回
    -------
    Polars LazyFrame，包含所有10个风格因子：| date | symbol | size | beta | momentum | ... |
    """
    # 1. Size 因子
    size_df = factor_size_cne5(mkt_cap_df)

    # 2. Beta 因子
    beta_df = factor_beta_cne5(returns_df, mkt_cap_df, risk_free_df)

    # 3. Momentum 因子
    momentum_df = factor_momentum_cne5(returns_df, mkt_cap_df, risk_free_df)

    # 4. Residual Volatility 因子（需要 beta 和 size）
    residual_vol_df = factor_residual_volatility_cne5(
        returns_df, mkt_cap_df, beta_df, size_df, risk_free_df
    )

    # 5. Non-linear Size 因子
    nonlinear_size_df = factor_nonlinear_size_cne5(size_df, mkt_cap_df)

    # 合并基础因子
    style_df = (
        size_df.join(beta_df, on=["date", "symbol"])
        .join(momentum_df, on=["date", "symbol"])
        .join(residual_vol_df, on=["date", "symbol"])
        .join(nonlinear_size_df, on=["date", "symbol"])
    )

    # 6. Book-to-Price 因子（如果提供数据）
    if book_value_df is not None:
        btop_df = factor_book_to_price_cne5(book_value_df, mkt_cap_df)
        style_df = style_df.join(btop_df, on=["date", "symbol"])

    # 7. Liquidity 因子（如果提供数据）
    if turnover_df is not None:
        liquidity_df = factor_liquidity_cne5(turnover_df, mkt_cap_df, size_df)
        style_df = style_df.join(liquidity_df, on=["date", "symbol"])

    # 8. Earnings Yield 因子（如果提供数据）
    if earnings_df is not None:
        earnings_yield_df = factor_earnings_yield_cne5(earnings_df, mkt_cap_df)
        style_df = style_df.join(earnings_yield_df, on=["date", "symbol"])

    # 9. Growth 因子（如果提供数据）
    if growth_df is not None:
        growth_factor_df = factor_growth_cne5(growth_df, mkt_cap_df)
        style_df = style_df.join(growth_factor_df, on=["date", "symbol"])

    # 10. Leverage 因子（如果提供数据）
    if leverage_df is not None:
        leverage_factor_df = factor_leverage_cne5(leverage_df, mkt_cap_df)
        style_df = style_df.join(leverage_factor_df, on=["date", "symbol"])

    return style_df


def estimate_cne5_factor_returns(
    returns_df: pl.DataFrame,
    mkt_cap_df: pl.DataFrame,
    sector_df: pl.DataFrame,
    style_df: pl.DataFrame,
    winsor_factor: float | None = 0.05,
    residualize_styles: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """估计 CNE5 因子收益（国家、行业、风格）。

    注意：在 CNE5 模型中，"市场"因子实际上就是"国家"因子（Country factor），
    它代表以市值加权的国家投资组合。

    参数
    ----------
    returns_df: Polars DataFrame，包含列：| date | symbol | asset_returns |
    mkt_cap_df: Polars DataFrame，包含列：| date | symbol | market_cap |
    sector_df: Polars DataFrame，包含列：| date | symbol | 以及每个行业一列
    style_df: Polars DataFrame，包含列：| date | symbol | 以及每个风格一列
    winsor_factor: 去极值的比例
    residualize_styles: 布尔值，指示风格收益是否应正交化到国家+行业收益

    返回
    -------
    按日期展开的Polars DataFrame元组：(因子收益, 残差收益)
    因子收益包含：country (国家因子，原 market) + 行业因子 + 风格因子
    """
    # 使用现有的 estimate_factor_returns 函数
    factor_returns_df, residual_returns_df = estimate_factor_returns(
        returns_df,
        mkt_cap_df,
        sector_df,
        style_df,
        winsor_factor=winsor_factor,
        residualize_styles=residualize_styles,
    )

    # 将 "market" 重命名为 "country" 以符合 CNE5 术语
    factor_returns_df = factor_returns_df.rename({"market": "country"})

    return factor_returns_df, residual_returns_df


def build_cne5_model(
    returns_df: pl.DataFrame,
    mkt_cap_df: pl.DataFrame,
    sector_df: pl.DataFrame,
    risk_free_df: pl.DataFrame | None = None,
    # 基本面数据（可选）
    book_value_df: pl.DataFrame | None = None,
    turnover_df: pl.DataFrame | None = None,
    earnings_df: pl.DataFrame | None = None,
    growth_df: pl.DataFrame | None = None,
    leverage_df: pl.DataFrame | None = None,
    winsor_factor: float | None = 0.05,
    residualize_styles: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """构建完整的 CNE5 模型：风格因子 + 因子收益估计。

    参数
    ----------
    returns_df: Polars DataFrame，包含列：| date | symbol | asset_returns |
    mkt_cap_df: Polars DataFrame，包含列：| date | symbol | market_cap |
    sector_df: Polars DataFrame，包含列：| date | symbol | 以及每个行业一列
    risk_free_df: 可选，无风险利率数据
    book_value_df: 可选，账面价值数据
    turnover_df: 可选，换手率数据
    earnings_df: 可选，盈利数据
    growth_df: 可选，成长数据
    leverage_df: 可选，杠杆数据
    winsor_factor: 去极值的比例
    residualize_styles: 布尔值，指示风格收益是否应正交化到国家+行业收益

    返回
    -------
    元组：(风格因子DataFrame, 因子收益DataFrame, 残差收益DataFrame)
    """
    # 构建风格因子
    style_df = build_cne5_style_factors(
        returns_df,
        mkt_cap_df,
        risk_free_df,
        book_value_df,
        turnover_df,
        earnings_df,
        growth_df,
        leverage_df,
    ).collect()

    # 估计因子收益
    factor_returns_df, residual_returns_df = estimate_cne5_factor_returns(
        returns_df,
        mkt_cap_df,
        sector_df,
        style_df,
        winsor_factor=winsor_factor,
        residualize_styles=residualize_styles,
    )

    return style_df, factor_returns_df, residual_returns_df


def build_cne5_risk_model(
    factor_returns_df: pl.DataFrame,
    residual_returns_df: pl.DataFrame,
    mkt_cap_df: pl.DataFrame,
    apply_oba: bool = True,
    apply_vra: bool = True,
    apply_bayesian_shrinkage: bool = True,
    cov_window: int = 252,
    cov_half_life: int | None = 63,
    specific_risk_window: int = 252,
    specific_risk_half_life: int = 63,
    n_oba_simulations: int = 1000,
) -> tuple[np.ndarray, pl.DataFrame, dict]:
    """构建完整的 CNE5 风险模型。

    参数
    ----------
    factor_returns_df: Polars DataFrame，因子收益
    residual_returns_df: Polars DataFrame，残差收益
    mkt_cap_df: Polars DataFrame，市值数据
    apply_oba: 是否应用优化偏差调整，默认 True
    apply_vra: 是否应用波动体制调整，默认 True
    apply_bayesian_shrinkage: 是否应用贝叶斯收缩，默认 True
    cov_window: 协方差估计窗口，默认252
    cov_half_life: 协方差估计半衰期，默认63
    specific_risk_window: 特异风险估计窗口，默认252
    specific_risk_half_life: 特异风险估计半衰期，默认63
    n_oba_simulations: OBA 蒙特卡洛模拟次数，默认1000

    返回
    -------
    元组：(因子协方差矩阵, 特异风险DataFrame, 风险模型信息字典)
    """
    # 1. 估计因子协方差矩阵
    factor_cov = estimate_factor_covariance(
        factor_returns_df,
        window=cov_window,
        half_life=cov_half_life,
        method="ewma" if cov_half_life is not None else "sample",
    )

    # 2. 应用 OBA（如果需要）
    if apply_oba:
        # 需要资产数量，从残差收益 DataFrame 估算
        n_assets = len([col for col in residual_returns_df.columns if col != "date"])
        factor_cov = optimization_bias_adjustment(
            factor_cov, n_assets=n_assets, n_simulations=n_oba_simulations
        )

    # 3. 估计特异风险（日频序列法）
    specific_risk_df = estimate_specific_risk_ts(
        residual_returns_df,
        window=specific_risk_window,
        half_life=specific_risk_half_life,
    )

    # 4. 应用贝叶斯收缩（如果需要）
    if apply_bayesian_shrinkage:
        specific_risk_df = bayesian_shrinkage(specific_risk_df, mkt_cap_df)

    # 5. 应用 VRA（如果需要）
    if apply_vra:
        # 转换为字典格式
        latest_date = specific_risk_df["date"].max()
        latest_risks = (
            specific_risk_df.filter(pl.col("date") == latest_date)
            .select(["symbol", "specific_risk_shrunk" if apply_bayesian_shrinkage else "specific_risk"])
            .to_dicts()
        )
        specific_risks_dict = {
            item["symbol"]: item["specific_risk_shrunk" if apply_bayesian_shrinkage else "specific_risk"]
            for item in latest_risks
        }

        factor_cov, specific_risks_dict = volatility_regime_adjustment(
            factor_returns_df,
            residual_returns_df,
            factor_cov,
            specific_risks_dict,
        )

        # 更新 DataFrame
        risk_col = "specific_risk_shrunk" if apply_bayesian_shrinkage else "specific_risk"
        for item in latest_risks:
            symbol = item["symbol"]
            if symbol in specific_risks_dict:
                specific_risk_df = specific_risk_df.with_columns(
                    pl.when((pl.col("date") == latest_date) & (pl.col("symbol") == symbol))
                    .then(specific_risks_dict[symbol])
                    .otherwise(pl.col(risk_col))
                    .alias(risk_col)
                )

    # 构建信息字典
    info = {
        "factor_cov_shape": factor_cov.shape,
        "n_factors": factor_cov.shape[0],
        "oba_applied": apply_oba,
        "vra_applied": apply_vra,
        "bayesian_shrinkage_applied": apply_bayesian_shrinkage,
    }

    return factor_cov, specific_risk_df, info


def build_complete_cne5_model(
    returns_df: pl.DataFrame,
    mkt_cap_df: pl.DataFrame,
    sector_df: pl.DataFrame,
    risk_free_df: pl.DataFrame | None = None,
    # 基本面数据（可选）
    book_value_df: pl.DataFrame | None = None,
    turnover_df: pl.DataFrame | None = None,
    earnings_df: pl.DataFrame | None = None,
    growth_df: pl.DataFrame | None = None,
    leverage_df: pl.DataFrame | None = None,
    # 因子收益估计参数
    winsor_factor: float | None = 0.05,
    residualize_styles: bool = True,
    # 风险模型参数
    apply_oba: bool = True,
    apply_vra: bool = True,
    apply_bayesian_shrinkage: bool = True,
    cov_window: int = 252,
    cov_half_life: int | None = 63,
    specific_risk_window: int = 252,
    specific_risk_half_life: int = 63,
) -> dict:
    """构建完整的 CNE5 模型：因子 + 收益 + 风险。

    参数
    ----------
    returns_df: Polars DataFrame，包含列：| date | symbol | asset_returns |
    mkt_cap_df: Polars DataFrame，包含列：| date | symbol | market_cap |
    sector_df: Polars DataFrame，包含列：| date | symbol | 以及每个行业一列
    risk_free_df: 可选，无风险利率数据
    book_value_df: 可选，账面价值数据
    turnover_df: 可选，换手率数据
    earnings_df: 可选，盈利数据
    growth_df: 可选，成长数据
    leverage_df: 可选，杠杆数据
    winsor_factor: 去极值的比例
    residualize_styles: 布尔值，指示风格收益是否应正交化到国家+行业收益
    apply_oba: 是否应用优化偏差调整
    apply_vra: 是否应用波动体制调整
    apply_bayesian_shrinkage: 是否应用贝叶斯收缩
    cov_window: 协方差估计窗口
    cov_half_life: 协方差估计半衰期
    specific_risk_window: 特异风险估计窗口
    specific_risk_half_life: 特异风险估计半衰期

    返回
    -------
    字典，包含所有模型结果：
    {
        'style_factors': style_df,
        'factor_returns': factor_returns_df,
        'residual_returns': residual_returns_df,
        'factor_covariance': factor_cov,
        'specific_risks': specific_risk_df,
        'risk_info': info,
    }
    """
    # 1. 构建风格因子
    style_df = build_cne5_style_factors(
        returns_df,
        mkt_cap_df,
        risk_free_df,
        book_value_df,
        turnover_df,
        earnings_df,
        growth_df,
        leverage_df,
    ).collect()

    # 2. 估计因子收益
    factor_returns_df, residual_returns_df = estimate_cne5_factor_returns(
        returns_df,
        mkt_cap_df,
        sector_df,
        style_df,
        winsor_factor=winsor_factor,
        residualize_styles=residualize_styles,
    )

    # 3. 构建风险模型
    factor_cov, specific_risk_df, risk_info = build_cne5_risk_model(
        factor_returns_df,
        residual_returns_df,
        mkt_cap_df,
        apply_oba=apply_oba,
        apply_vra=apply_vra,
        apply_bayesian_shrinkage=apply_bayesian_shrinkage,
        cov_window=cov_window,
        cov_half_life=cov_half_life,
        specific_risk_window=specific_risk_window,
        specific_risk_half_life=specific_risk_half_life,
    )

    return {
        "style_factors": style_df,
        "factor_returns": factor_returns_df,
        "residual_returns": residual_returns_df,
        "factor_covariance": factor_cov,
        "specific_risks": specific_risk_df,
        "risk_info": risk_info,
    }
