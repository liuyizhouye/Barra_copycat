"""CNE5 Residual Volatility 因子: 0.74·DASTD + 0.16·CMRA + 0.10·HSIGMA。"""

from typing import Union, Optional

import numpy as np
import polars as pl
import polars.exceptions as pl_exc

from cne5.math import exp_weights
from cne5.cne5_factors._utils import standardize_cne5, orthogonalize_factor


def factor_residual_volatility_cne5(
    returns_df: Union[pl.DataFrame, pl.LazyFrame],
    mkt_cap_df: Union[pl.DataFrame, pl.LazyFrame],
    beta_df: Union[pl.DataFrame, pl.LazyFrame],
    size_df: Union[pl.DataFrame, pl.LazyFrame],
    risk_free_df: Optional[Union[pl.DataFrame, pl.LazyFrame]] = None,
    dastd_window: int = 252,
    dastd_half_life: int = 42,
    cmra_months: int = 12,
    hsigma_window: int = 252,
    hsigma_half_life: int = 63,
) -> pl.LazyFrame:
    """CNE5 Residual Volatility 因子: 0.74·DASTD + 0.16·CMRA + 0.10·HSIGMA。

    参数
    ----------
    returns_df: Polars DataFrame，包含列：| date | symbol | asset_returns |
    mkt_cap_df: Polars DataFrame，包含列：| date | symbol | market_cap |
    beta_df: Polars DataFrame，包含列：| date | symbol | beta |
    size_df: Polars DataFrame，包含列：| date | symbol | size |
    risk_free_df: 可选，无风险利率数据
    dastd_window: DASTD 窗口长度，默认252
    dastd_half_life: DASTD 半衰期，默认42
    cmra_months: CMRA 月数，默认12
    hsigma_window: HSIGMA 窗口长度，默认252
    hsigma_half_life: HSIGMA 半衰期，默认63

    返回
    -------
    Polars LazyFrame，包含列：| date | symbol | residual_volatility |
    """
    try:
        df = returns_df.lazy() if isinstance(returns_df, pl.DataFrame) else returns_df
        df = df.join(
            mkt_cap_df.lazy() if isinstance(mkt_cap_df, pl.DataFrame) else mkt_cap_df,
            on=["date", "symbol"],
        )

        # 计算超额收益
        if risk_free_df is not None:
            rf_df = risk_free_df.lazy() if isinstance(risk_free_df, pl.DataFrame) else risk_free_df
            df = df.join(rf_df, on="date")
            df = df.with_columns(
                (pl.col("asset_returns") - pl.col("risk_free_rate")).alias("excess_returns")
            )
        else:
            df = df.with_columns(pl.col("asset_returns").alias("excess_returns"))

        # 计算市场组合超额收益
        df = df.with_columns(
            (
                (pl.col("excess_returns") * pl.col("market_cap")).sum().over("date")
                / pl.col("market_cap").sum().over("date")
            ).alias("market_excess_returns")
        )

        # DASTD: 过去252日日度超额收益的波动率，半衰期42日
        dastd_weights = exp_weights(dastd_window, dastd_half_life)

        def calc_dastd(values: np.ndarray) -> float:
            if len(values) < dastd_window:
                return np.nan
            w = dastd_weights[-len(values):]
            weighted_mean = np.average(values[-dastd_window:], weights=w)
            weighted_var = np.average((values[-dastd_window:] - weighted_mean) ** 2, weights=w)
            return np.sqrt(weighted_var * 252)  # 年化

        df = df.sort("date").with_columns(
            pl.col("excess_returns")
            .rolling_map(calc_dastd, window_size=dastd_window)
            .over("symbol")
            .alias("dastd")
        )

        # CMRA: 累计区间范围 (以月为单位)
        # 每月21日，构造T=1..12月的累计超额对数收益 Z(T)，CMRA = max Z - min Z
        def calc_cmra(values: np.ndarray) -> float:
            if len(values) < 21 * cmra_months:
                return np.nan
            
            # 计算对数收益
            log_returns = np.log(1 + values[-21 * cmra_months:])
            
            # 计算 T=1..12 月的累计对数收益 Z(T)
            # Z(T) = sum of log returns from start to end of month T
            monthly_cumulative = []
            for t in range(1, cmra_months + 1):
                # 第 t 个月的累计收益：从开始到第 t 个月结束
                end_idx = t * 21
                if end_idx <= len(log_returns):
                    z_t = np.sum(log_returns[:end_idx])
                    monthly_cumulative.append(z_t)
            
            if len(monthly_cumulative) == 0:
                return np.nan
            
            # CMRA = max Z - min Z
            return np.max(monthly_cumulative) - np.min(monthly_cumulative)

        df = df.with_columns(
            pl.col("excess_returns")
            .rolling_map(calc_cmra, window_size=21 * cmra_months)
            .over("symbol")
            .alias("cmra")
        )

        # HSIGMA: 来自回归残差的波动率
        hsigma_weights = exp_weights(hsigma_window, hsigma_half_life)

        def calc_hsigma(returns: np.ndarray, market_returns: np.ndarray) -> float:
            if len(returns) < hsigma_window:
                return np.nan
            y = returns[-hsigma_window:]
            X = market_returns[-hsigma_window:].reshape(-1, 1)
            w = hsigma_weights[-len(y):]

            # 加权回归
            W = np.diag(w)
            try:
                XWX = X.T @ W @ X
                XWy = X.T @ W @ y
                if XWX[0, 0] < 1e-10:
                    return np.nan
                beta = XWy[0] / XWX[0, 0]
                residuals = y - X @ beta
                # 加权残差标准差
                weighted_mean = np.average(residuals, weights=w)
                weighted_var = np.average((residuals - weighted_mean) ** 2, weights=w)
                return np.sqrt(weighted_var * 252)  # 年化
            except (np.linalg.LinAlgError, ZeroDivisionError):
                return np.nan

        # 需要先收集数据以获取市场收益序列
        df_collected = df.collect()
        
        # 获取市场收益字典（按日期索引）
        market_returns_dict = (
            df_collected.group_by("date")
            .agg(pl.col("market_excess_returns").first())
            .to_dicts()
        )
        market_returns_by_date = {item["date"]: item["market_excess_returns"] for item in market_returns_dict}

        def calc_hsigma_group(group: pl.DataFrame) -> pl.DataFrame:
            """按股票分组计算 HSIGMA。"""
            excess_returns = group.sort("date")["excess_returns"].to_numpy()
            dates = group.sort("date")["date"].to_list()
            
            hsigma_values = []
            for i in range(len(group)):
                if i < hsigma_window - 1:
                    hsigma_values.append(np.nan)
                else:
                    # 取最后 hsigma_window 个值
                    returns = excess_returns[i - hsigma_window + 1 : i + 1]
                    # 获取对应的市场收益
                    market_vals = np.array([
                        market_returns_by_date.get(d, np.nan)
                        for d in dates[i - hsigma_window + 1 : i + 1]
                    ])
                    
                    if np.any(np.isnan(market_vals)) or len(market_vals) != hsigma_window:
                        hsigma_values.append(np.nan)
                        continue
                    
                    hsigma_values.append(calc_hsigma(returns, market_vals))
            
            return group.with_columns(pl.Series("hsigma", hsigma_values))

        df = (
            df_collected.lazy()
            .group_by("symbol")
            .map_groups(calc_hsigma_group, schema=df_collected.schema)
        )

        # 组合: 0.74·DASTD + 0.16·CMRA + 0.10·HSIGMA
        df = df.with_columns(
            (0.74 * pl.col("dastd") + 0.16 * pl.col("cmra") + 0.10 * pl.col("hsigma")).alias(
                "residual_volatility"
            )
        )

        # 对 Beta 和 Size 正交化
        df = df.join(
            beta_df.lazy() if isinstance(beta_df, pl.DataFrame) else beta_df,
            on=["date", "symbol"],
        )
        df = df.join(
            size_df.lazy() if isinstance(size_df, pl.DataFrame) else size_df,
            on=["date", "symbol"],
        )

        df = orthogonalize_factor(df, "residual_volatility", ["beta", "size"], "market_cap")

        # 标准化
        df = standardize_cne5(df, "residual_volatility", "market_cap")
        return df.select("date", "symbol", "residual_volatility")

    except AttributeError as e:
        raise TypeError("输入必须是 Polars DataFrame 或 LazyFrame") from e
    except pl_exc.ColumnNotFoundError as e:
        raise ValueError("缺少必需的列") from e
