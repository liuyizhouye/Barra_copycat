"""CNE5 Non-linear Size 因子: Size 的立方，相对 Size 回归加权正交化。"""

import polars as pl
import polars.exceptions as pl_exc

from cne5.math import winsorize_xsection
from cne5.cne5_factors._utils import standardize_cne5, orthogonalize_factor


def factor_nonlinear_size_cne5(
    size_df: pl.DataFrame | pl.LazyFrame,
    mkt_cap_df: pl.DataFrame | pl.LazyFrame,
) -> pl.LazyFrame:
    """CNE5 Non-linear Size 因子: Size 的立方，相对 Size 回归加权正交化。

    参数
    ----------
    size_df: Polars DataFrame，包含列：| date | symbol | size |
    mkt_cap_df: Polars DataFrame，包含列：| date | symbol | market_cap |

    返回
    -------
    Polars LazyFrame，包含列：| date | symbol | nonlinear_size |
    """
    try:
        df = size_df.lazy() if isinstance(size_df, pl.DataFrame) else size_df
        df = df.join(
            mkt_cap_df.lazy() if isinstance(mkt_cap_df, pl.DataFrame) else mkt_cap_df,
            on=["date", "symbol"],
        )

        # 先标准化 Size
        df = standardize_cne5(df, "size", "market_cap")

        # 立方
        df = df.with_columns((pl.col("size") ** 3).alias("nonlinear_size"))

        # 相对 Size 回归加权正交化
        df = orthogonalize_factor(df, "nonlinear_size", ["size"], "market_cap")

        # Winsorize 和标准化
        df = df.collect()
        df = winsorize_xsection(df, ("nonlinear_size",), "date", percentile=0.05)
        df = standardize_cne5(df, "nonlinear_size", "market_cap")

        return df.lazy().select("date", "symbol", "nonlinear_size")

    except AttributeError as e:
        raise TypeError("输入必须是 Polars DataFrame 或 LazyFrame") from e
    except pl_exc.ColumnNotFoundError as e:
        raise ValueError("缺少必需的列") from e
