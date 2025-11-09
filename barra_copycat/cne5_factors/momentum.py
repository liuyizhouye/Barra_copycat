"""CNE5 Momentum 因子: 滞后1个月后，过去504日超额对数收益的指数加权和。"""

import numpy as np
import polars as pl
import polars.exceptions as pl_exc

from barra_copycat.math import exp_weights
from barra_copycat.cne5_factors._utils import standardize_cne5


def factor_momentum_cne5(
    returns_df: pl.DataFrame | pl.LazyFrame,
    mkt_cap_df: pl.DataFrame | pl.LazyFrame,
    risk_free_df: pl.DataFrame | pl.LazyFrame | None = None,
    lag: int = 21,
    trailing_days: int = 504,
    half_life: int = 126,
) -> pl.LazyFrame:
    """CNE5 Momentum 因子: 滞后1个月后，过去504日超额对数收益的指数加权和。

    参数
    ----------
    returns_df: Polars DataFrame，包含列：| date | symbol | asset_returns |
    mkt_cap_df: Polars DataFrame，包含列：| date | symbol | market_cap |
    risk_free_df: 可选，无风险利率数据
    lag: 滞后天数，默认21(1个月)
    trailing_days: 回看期长度，默认504
    half_life: 半衰期，默认126

    返回
    -------
    Polars LazyFrame，包含列：| date | symbol | momentum |
    """
    try:
        weights = exp_weights(trailing_days, half_life)

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

        # 滞后并计算对数超额收益
        df = df.sort("date").with_columns(
            pl.col("excess_returns").shift(lag).over("symbol").alias("lagged_excess")
        )

        # 计算指数加权和: sum(w_t * ln(1 + r_t))
        def weighted_log_sum(values: np.ndarray) -> float:
            if len(values) < trailing_days:
                return np.nan
            log_returns = np.log(1 + values[-trailing_days:])
            w = weights[-len(log_returns):]
            return np.sum(w * log_returns)

        df = df.with_columns(
            pl.col("lagged_excess")
            .rolling_map(weighted_log_sum, window_size=trailing_days)
            .over("symbol")
            .alias("momentum")
        )

        df = standardize_cne5(df, "momentum", "market_cap")
        return df.select("date", "symbol", "momentum")

    except AttributeError as e:
        raise TypeError("输入必须是 Polars DataFrame 或 LazyFrame") from e
    except pl_exc.ColumnNotFoundError as e:
        raise ValueError("缺少必需的列") from e
