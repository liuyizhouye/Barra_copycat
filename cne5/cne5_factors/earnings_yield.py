"""CNE5 Earnings Yield 因子: 0.68·EPFWD + 0.21·CETOP + 0.11·ETOP。"""

from typing import Union

import polars as pl
import polars.exceptions as pl_exc

from cne5.cne5_factors._utils import standardize_cne5


def factor_earnings_yield_cne5(
    earnings_df: Union[pl.DataFrame, pl.LazyFrame],
    mkt_cap_df: Union[pl.DataFrame, pl.LazyFrame],
    epfwd_col: str = "epfwd",
    cetop_col: str = "cetop",
    etop_col: str = "etop",
) -> pl.LazyFrame:
    """CNE5 Earnings Yield 因子: 0.68·EPFWD + 0.21·CETOP + 0.11·ETOP。

    参数
    ----------
    earnings_df: Polars DataFrame，包含列：| date | symbol | epfwd | cetop | etop |
    mkt_cap_df: Polars DataFrame，包含列：| date | symbol | market_cap |
    epfwd_col: 预测盈利/价列名
    cetop_col: 现金盈利/价列名
    etop_col: 追踪盈利/价列名

    返回
    -------
    Polars LazyFrame，包含列：| date | symbol | earnings_yield |
    """
    try:
        df = earnings_df.lazy() if isinstance(earnings_df, pl.DataFrame) else earnings_df
        df = df.join(
            mkt_cap_df.lazy() if isinstance(mkt_cap_df, pl.DataFrame) else mkt_cap_df,
            on=["date", "symbol"],
        )

        # 组合: 0.68·EPFWD + 0.21·CETOP + 0.11·ETOP
        df = df.with_columns(
            (
                0.68 * pl.col(epfwd_col)
                + 0.21 * pl.col(cetop_col)
                + 0.11 * pl.col(etop_col)
            ).alias("earnings_yield")
        )

        df = standardize_cne5(df, "earnings_yield", "market_cap")
        return df.select("date", "symbol", "earnings_yield")

    except AttributeError as e:
        raise TypeError("输入必须是 Polars DataFrame 或 LazyFrame") from e
    except pl_exc.ColumnNotFoundError as e:
        raise ValueError("缺少必需的列") from e
