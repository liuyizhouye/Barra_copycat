"""风格因子实现。"""

import numpy as np
import polars as pl
import polars.exceptions as pl_exc

from toraniko.math import (
    exp_weights,
    center_xsection,
    percentiles_xsection,
    winsorize_xsection,
)

###
# 注意：这些函数不会为您处理 NaN 或 null 的情况，也不会对具有病态分布的数据做出调整。
# 垃圾进，垃圾出。您需要检查数据并使用 math 和 utils 模块中的函数来确保您的特征
# 是合理且行为良好的，然后才尝试从中构建因子！
###


def factor_mom(
    returns_df: pl.DataFrame | pl.LazyFrame,
    trailing_days: int = 504,
    half_life: int = 126,
    lag: int = 20,
    winsor_factor: float = 0.01,
) -> pl.LazyFrame:
    """使用资产收益估计每只股票的滚动动量因子得分。

    参数
    ----------
    returns_df: Polars DataFrame，包含列：| date | symbol | asset_returns |
    trailing_days: 回看期长度，用于测量动量
    half_life: 指数加权衰减率，单位天
    lag: 滞后天数，当前日期收益观察值的滞后期（20个交易日等于一个月）

    返回
    -------
    Polars LazyFrame，包含列：| date | symbol | mom_score |
    """
    weights = exp_weights(trailing_days, half_life)

    def weighted_cumprod(values: np.ndarray) -> float:
        return (np.cumprod(1 + (values * weights[-len(values) :])) - 1)[-1]  # type: ignore

    try:
        df = (
            returns_df.lazy()
            .sort(by="date")
            .with_columns(pl.col("asset_returns").shift(lag).over("symbol").alias("asset_returns"))
            .with_columns(
                pl.col("asset_returns")
                .rolling_map(weighted_cumprod, window_size=trailing_days)
                .over(pl.col("symbol"))
                .alias("mom_score")
            )
        ).collect()
        df = winsorize_xsection(df, ("mom_score",), "date", percentile=winsor_factor)
        return df.lazy().select(
            pl.col("date"),
            pl.col("symbol"),
            center_xsection("mom_score", "date", True).alias("mom_score"),
        )
    except AttributeError as e:
        raise TypeError("`returns_df` 必须是 Polars DataFrame 或 LazyFrame，但缺少属性") from e
    except pl_exc.ColumnNotFoundError as e:
        raise ValueError("`returns_df` 必须包含 'date'、'symbol' 和 'asset_returns' 列") from e


def factor_sze(
    mkt_cap_df: pl.DataFrame | pl.LazyFrame,
    lower_decile: float = 0.2,
    upper_decile: float = 0.8,
) -> pl.LazyFrame:
    """使用资产市值估计每只股票的滚动规模因子得分。

    参数
    ----------
    mkt_cap_df: Polars DataFrame，包含列：| date | symbol | market_cap |
    lower_decile: 下分位数阈值
    upper_decile: 上分位数阈值

    返回
    -------
    Polars LazyFrame，包含列：| date | symbol | sze_score |
    """
    try:
        return (
            mkt_cap_df.lazy()
            # 我们的因子是 Fama-French 的 SMB（小盘股减去大盘股），因为规模风险溢价
            # 存在于小公司而非大公司。因此我们乘以-1
            .with_columns(pl.col("market_cap").log().alias("sze_score") * -1)
            .with_columns(
                "date",
                "symbol",
                (center_xsection("sze_score", "date", True)).alias("sze_score"),
            )
            .with_columns(percentiles_xsection("sze_score", "date", lower_decile, upper_decile, 0.0).alias("sze_score"))
            .select("date", "symbol", "sze_score")
        )
    except AttributeError as e:
        raise TypeError("`mkt_cap_df` 必须是 Polars DataFrame 或 LazyFrame，但缺少属性") from e
    except pl_exc.ColumnNotFoundError as e:
        raise ValueError("`mkt_cap_df` 必须包含 'date'、'symbol' 和 'market_cap' 列") from e


def factor_val(value_df: pl.DataFrame | pl.LazyFrame, winsorize_features: float | None = None) -> pl.LazyFrame:
    """使用价格比率估计每只股票的滚动价值因子得分。

    参数
    ----------
    value_df: Polars DataFrame，包含列：| date | symbol | book_price | sales_price | cf_price
    winsorize_features: 可选浮点数，指示是否应对特征进行去极值处理。如果为 None 则不应用

    返回
    -------
    Polars LazyFrame，包含：| date | symbol | val_score |
    """
    try:
        if winsorize_features is not None:
            value_df = winsorize_xsection(value_df, ("book_price", "sales_price", "cf_price"), "date")
        return (
            value_df.lazy()
            .with_columns(
                pl.col("book_price").log().alias("book_price"),
                pl.col("sales_price").log().alias("sales_price"),
            )
            .with_columns(
                center_xsection("book_price", "date", True).alias("book_price"),
                center_xsection("sales_price", "date", True).alias("sales_price"),
                center_xsection("cf_price", "date", True).alias("cf_price"),
            )
            .with_columns(
                # 注意：在此之前必须已正确处理 NaNs
                pl.mean_horizontal(
                    pl.col("book_price"),
                    pl.col("sales_price"),
                    pl.col("cf_price"),
                ).alias("val_score")
            )
            .select(
                pl.col("date"),
                pl.col("symbol"),
                center_xsection("val_score", "date", True).alias("val_score"),
            )
        )
    except AttributeError as e:
        raise TypeError("`value_df` 必须是 Polars DataFrame 或 LazyFrame，但缺少属性") from e
    except pl_exc.ColumnNotFoundError as e:
        raise ValueError(
            "`value_df` 必须包含 'date'、'symbol'、'book_price'、'sales_price' 和 'cf_price' 列"
        ) from e
