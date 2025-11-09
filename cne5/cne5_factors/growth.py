"""CNE5 Growth 因子: 0.18·EGRLF + 0.11·EGRSF + 0.24·EGRO + 0.47·SGRO。"""

import polars as pl
import polars.exceptions as pl_exc

from cne5.cne5_factors._utils import standardize_cne5


def factor_growth_cne5(
    growth_df: pl.DataFrame | pl.LazyFrame,
    mkt_cap_df: pl.DataFrame | pl.LazyFrame,
    egrlf_col: str = "egrlf",
    egrsf_col: str = "egrsf",
    egro_col: str = "egro",
    sgro_col: str = "sgro",
) -> pl.LazyFrame:
    """CNE5 Growth 因子: 0.18·EGRLF + 0.11·EGRSF + 0.24·EGRO + 0.47·SGRO。

    参数
    ----------
    growth_df: Polars DataFrame，包含列：| date | symbol | egrlf | egrsf | egro | sgro |
    mkt_cap_df: Polars DataFrame，包含列：| date | symbol | market_cap |
    egrlf_col: 长期盈利增速列名
    egrsf_col: 短期盈利增速列名
    egro_col: 过去5年EPS回归斜率/平均EPS列名
    sgro_col: 过去5年销售额回归斜率/平均销售额列名

    返回
    -------
    Polars LazyFrame，包含列：| date | symbol | growth |
    """
    try:
        df = growth_df.lazy() if isinstance(growth_df, pl.DataFrame) else growth_df
        df = df.join(
            mkt_cap_df.lazy() if isinstance(mkt_cap_df, pl.DataFrame) else mkt_cap_df,
            on=["date", "symbol"],
        )

        # 组合: 0.18·EGRLF + 0.11·EGRSF + 0.24·EGRO + 0.47·SGRO
        df = df.with_columns(
            (
                0.18 * pl.col(egrlf_col)
                + 0.11 * pl.col(egrsf_col)
                + 0.24 * pl.col(egro_col)
                + 0.47 * pl.col(sgro_col)
            ).alias("growth")
        )

        df = standardize_cne5(df, "growth", "market_cap")
        return df.select("date", "symbol", "growth")

    except AttributeError as e:
        raise TypeError("输入必须是 Polars DataFrame 或 LazyFrame") from e
    except pl_exc.ColumnNotFoundError as e:
        raise ValueError("缺少必需的列") from e
