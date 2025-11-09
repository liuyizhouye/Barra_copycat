"""CNE5 因子工具函数。

共享的标准化和正交化函数。
"""

from typing import Union, List

import numpy as np
import polars as pl


def standardize_cne5(
    df: Union[pl.DataFrame, pl.LazyFrame],
    factor_col: str,
    mkt_cap_col: str,
    date_col: str = "date",
) -> pl.LazyFrame:
    """CNE5 标准化: 市值加权均值=0, 等权标准差=1。

    参数
    ----------
    df: Polars DataFrame 或 LazyFrame
    factor_col: 要标准化的因子列名
    mkt_cap_col: 市值列名
    date_col: 日期列名

    返回
    -------
    Polars LazyFrame，包含标准化后的因子
    """
    df_lazy = df.lazy() if isinstance(df, pl.DataFrame) else df

    # 第一步: 市值加权去中心化 (cap-weighted mean = 0)
    # 计算市值加权均值
    df_lazy = df_lazy.with_columns(
        (
            (pl.col(factor_col) * pl.col(mkt_cap_col)).sum().over(date_col)
            / pl.col(mkt_cap_col).sum().over(date_col)
        ).alias("_cap_weighted_mean")
    )

    # 去中心化
    df_lazy = df_lazy.with_columns(
        (pl.col(factor_col) - pl.col("_cap_weighted_mean")).alias("_centered")
    )

    # 第二步: 等权标准化 (equal-weighted std = 1)
    df_lazy = df_lazy.with_columns(
        pl.col("_centered").std().over(date_col).alias("_ew_std")
    )

    df_lazy = df_lazy.with_columns(
        (pl.col("_centered") / pl.col("_ew_std")).alias(factor_col)
    )

    return df_lazy.drop(["_cap_weighted_mean", "_centered", "_ew_std"])


def orthogonalize_factor(
    df: Union[pl.DataFrame, pl.LazyFrame],
    target_col: str,
    orthogonal_to_cols: List[str],
    mkt_cap_col: str,
    date_col: str = "date",
) -> pl.LazyFrame:
    """对因子进行回归加权正交化。

    参数
    ----------
    df: Polars DataFrame 或 LazyFrame
    target_col: 要正交化的因子列名
    orthogonal_to_cols: 正交化目标列名列表
    mkt_cap_col: 市值列名(用于加权)
    date_col: 日期列名

    返回
    -------
    Polars LazyFrame，包含正交化后的因子
    """
    df_lazy = df.lazy() if isinstance(df, pl.DataFrame) else df

    def orthogonalize_group(group: pl.DataFrame) -> pl.DataFrame:
        """对单个日期组进行正交化。"""
        y = group[target_col].to_numpy()
        X = group[orthogonal_to_cols].to_numpy()
        weights = np.sqrt(group[mkt_cap_col].to_numpy())

        # 加权最小二乘回归
        W = np.diag(weights)
        try:
            # 计算回归系数: (X'WX)^(-1) X'Wy
            XWX = X.T @ W @ X
            XWy = X.T @ W @ y
            beta = np.linalg.solve(XWX, XWy)
            # 残差 = y - X*beta
            residual = y - X @ beta
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，返回原始值
            residual = y

        return group.with_columns(pl.Series(target_col, residual))

    # 按日期分组进行正交化
    if isinstance(df, pl.DataFrame):
        result = df_lazy.group_by(date_col).map_groups(
            orthogonalize_group, schema=df_lazy.collect_schema()
        )
    else:
        result = df_lazy.group_by(date_col).map_groups(
            orthogonalize_group, schema=df_lazy.collect_schema()
        )

    return result
