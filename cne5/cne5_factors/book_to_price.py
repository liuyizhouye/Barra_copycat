"""CNE5 Book-to-Price 因子: 最新账面普通股权益 / 当前市值。"""

import polars as pl
import polars.exceptions as pl_exc

from cne5.cne5_factors._utils import standardize_cne5


def factor_book_to_price_cne5(
    book_value_df: pl.DataFrame | pl.LazyFrame,
    mkt_cap_df: pl.DataFrame | pl.LazyFrame,
    book_value_col: str = "book_value",
) -> pl.LazyFrame:
    """CNE5 Book-to-Price 因子: 最新账面普通股权益 / 当前市值。

    参数
    ----------
    book_value_df: Polars DataFrame，包含列：| date | symbol | book_value |
    mkt_cap_df: Polars DataFrame，包含列：| date | symbol | market_cap |
    book_value_col: 账面价值列名

    返回
    -------
    Polars LazyFrame，包含列：| date | symbol | book_to_price |
    """
    try:
        df = book_value_df.lazy() if isinstance(book_value_df, pl.DataFrame) else book_value_df
        df = df.join(
            mkt_cap_df.lazy() if isinstance(mkt_cap_df, pl.DataFrame) else mkt_cap_df,
            on=["date", "symbol"],
        )

        df = df.with_columns(
            (pl.col(book_value_col) / pl.col("market_cap")).alias("book_to_price")
        )

        df = standardize_cne5(df, "book_to_price", "market_cap")
        return df.select("date", "symbol", "book_to_price")

    except AttributeError as e:
        raise TypeError("输入必须是 Polars DataFrame 或 LazyFrame") from e
    except pl_exc.ColumnNotFoundError as e:
        raise ValueError("缺少必需的列") from e
