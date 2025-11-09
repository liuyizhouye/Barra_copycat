"""CNE5 Liquidity 因子: 0.35·STOM + 0.35·STOQ + 0.30·STOA。"""

from typing import Union

import polars as pl
import polars.exceptions as pl_exc

from cne5.cne5_factors._utils import standardize_cne5, orthogonalize_factor


def factor_liquidity_cne5(
    turnover_df: Union[pl.DataFrame, pl.LazyFrame],
    mkt_cap_df: Union[pl.DataFrame, pl.LazyFrame],
    size_df: Union[pl.DataFrame, pl.LazyFrame],
    turnover_col: str = "turnover",
) -> pl.LazyFrame:
    """CNE5 Liquidity 因子: 0.35·STOM + 0.35·STOQ + 0.30·STOA。

    参数
    ----------
    turnover_df: Polars DataFrame，包含列：| date | symbol | turnover |
    mkt_cap_df: Polars DataFrame，包含列：| date | symbol | market_cap |
    size_df: Polars DataFrame，包含列：| date | symbol | size |
    turnover_col: 换手率列名

    返回
    -------
    Polars LazyFrame，包含列：| date | symbol | liquidity |
    """
    try:
        df = turnover_df.lazy() if isinstance(turnover_df, pl.DataFrame) else turnover_df
        df = df.join(
            mkt_cap_df.lazy() if isinstance(mkt_cap_df, pl.DataFrame) else mkt_cap_df,
            on=["date", "symbol"],
        )

        # STOM: 过去21交易日日度换手率之和的对数
        df = df.sort("date").with_columns(
            pl.col(turnover_col)
            .rolling_sum(window_size=21)
            .over("symbol")
            .log()
            .alias("stom")
        )

        # STOQ: 把逐月 STOM 做 log-exp 平均，T=3
        # 简化：计算3个月滚动平均
        df = df.with_columns(
            pl.col("stom").rolling_mean(window_size=63).over("symbol").alias("stoq")
        )

        # STOA: 同上，T=12
        df = df.with_columns(
            pl.col("stom").rolling_mean(window_size=252).over("symbol").alias("stoa")
        )

        # 组合: 0.35·STOM + 0.35·STOQ + 0.30·STOA
        df = df.with_columns(
            (0.35 * pl.col("stom") + 0.35 * pl.col("stoq") + 0.30 * pl.col("stoa")).alias(
                "liquidity"
            )
        )

        # 对 Size 正交化
        df = df.join(
            size_df.lazy() if isinstance(size_df, pl.DataFrame) else size_df,
            on=["date", "symbol"],
        )
        df = orthogonalize_factor(df, "liquidity", ["size"], "market_cap")

        # 标准化
        df = standardize_cne5(df, "liquidity", "market_cap")
        return df.select("date", "symbol", "liquidity")

    except AttributeError as e:
        raise TypeError("输入必须是 Polars DataFrame 或 LazyFrame") from e
    except pl_exc.ColumnNotFoundError as e:
        raise ValueError("缺少必需的列") from e
