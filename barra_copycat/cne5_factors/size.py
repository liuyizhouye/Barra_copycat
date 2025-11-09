"""CNE5 Size 因子: LNCAP (市值自然对数)。"""

import polars as pl
import polars.exceptions as pl_exc

from barra_copycat.cne5_factors._utils import standardize_cne5


def factor_size_cne5(
    mkt_cap_df: pl.DataFrame | pl.LazyFrame,
    mkt_cap_col: str = "market_cap",
) -> pl.LazyFrame:
    """CNE5 Size 因子: LNCAP (市值自然对数)。

    参数
    ----------
    mkt_cap_df: Polars DataFrame，包含列：| date | symbol | market_cap |
    mkt_cap_col: 市值列名

    返回
    -------
    Polars LazyFrame，包含列：| date | symbol | size |
    """
    try:
        df_lazy = mkt_cap_df.lazy() if isinstance(mkt_cap_df, pl.DataFrame) else mkt_cap_df
        df_lazy = df_lazy.with_columns(pl.col(mkt_cap_col).log().alias("size"))
        df_lazy = standardize_cne5(df_lazy, "size", mkt_cap_col)
        return df_lazy.select("date", "symbol", "size")
    except AttributeError as e:
        raise TypeError("`mkt_cap_df` 必须是 Polars DataFrame 或 LazyFrame") from e
    except pl_exc.ColumnNotFoundError as e:
        raise ValueError(f"`mkt_cap_df` 必须包含 'date'、'symbol' 和 '{mkt_cap_col}' 列") from e
