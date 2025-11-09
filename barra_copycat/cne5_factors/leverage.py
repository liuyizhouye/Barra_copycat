"""CNE5 Leverage 因子: 0.38·MLEV + 0.35·DTOA + 0.27·BLEV。"""

import polars as pl
import polars.exceptions as pl_exc

from barra_copycat.cne5_factors._utils import standardize_cne5


def factor_leverage_cne5(
    leverage_df: pl.DataFrame | pl.LazyFrame,
    mkt_cap_df: pl.DataFrame | pl.LazyFrame,
    mlev_col: str = "mlev",
    dtoa_col: str = "dtoa",
    blev_col: str = "blev",
) -> pl.LazyFrame:
    """CNE5 Leverage 因子: 0.38·MLEV + 0.35·DTOA + 0.27·BLEV。

    参数
    ----------
    leverage_df: Polars DataFrame，包含列：| date | symbol | mlev | dtoa | blev |
    mkt_cap_df: Polars DataFrame，包含列：| date | symbol | market_cap |
    mlev_col: 市值杠杆列名
    dtoa_col: 负债/资产列名
    blev_col: 账面杠杆列名

    返回
    -------
    Polars LazyFrame，包含列：| date | symbol | leverage |
    """
    try:
        df = leverage_df.lazy() if isinstance(leverage_df, pl.DataFrame) else leverage_df
        df = df.join(
            mkt_cap_df.lazy() if isinstance(mkt_cap_df, pl.DataFrame) else mkt_cap_df,
            on=["date", "symbol"],
        )

        # 组合: 0.38·MLEV + 0.35·DTOA + 0.27·BLEV
        df = df.with_columns(
            (
                0.38 * pl.col(mlev_col) + 0.35 * pl.col(dtoa_col) + 0.27 * pl.col(blev_col)
            ).alias("leverage")
        )

        df = standardize_cne5(df, "leverage", "market_cap")
        return df.select("date", "symbol", "leverage")

    except AttributeError as e:
        raise TypeError("输入必须是 Polars DataFrame 或 LazyFrame") from e
    except pl_exc.ColumnNotFoundError as e:
        raise ValueError("缺少必需的列") from e
