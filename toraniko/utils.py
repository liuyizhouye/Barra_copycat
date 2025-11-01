"""工具函数，主要用于数据清洗。"""

import numpy as np
import polars as pl


def fill_features(
    df: pl.DataFrame | pl.LazyFrame, features: tuple[str, ...], sort_col: str, over_col: str
) -> pl.LazyFrame:
    """将特征列转换为数值类型（float），将 NaN 和 inf 值转换为 null，
    然后对 `features` 的每一列进行前向填充 null 值，按 `sort_col` 排序并按 `over_col` 分组。

    参数
    ----------
    df: Polars DataFrame 或 LazyFrame，包含列 `sort_col`、`over_col` 和每个 `features`
    features: 字符串集合，指示 `df` 中的哪些列是特征值
    sort_col: `df` 的列，指示如何排序
    over_col: `df` 的列，指示如何分组

    返回
    -------
    Polars LazyFrame，包含原始列和清洗后的特征数据
    """
    try:
        # 急切检查所有 `features`、`sort_col`、`over_col` 是否存在：在延迟上下文中无法捕获 ColumnNotFoundError
        assert all(c in df.columns for c in features + (sort_col, over_col))
        return (
            df.lazy()
            .with_columns([pl.col(f).cast(float).alias(f) for f in features])
            .with_columns(
                [
                    pl.when(
                        (pl.col(f).abs() == np.inf)
                        | (pl.col(f) == np.nan)
                        | (pl.col(f).is_null())
                        | (pl.col(f).cast(str) == "NaN")
                    )
                    .then(None)
                    .otherwise(pl.col(f))
                    .alias(f)
                    for f in features
                ]
            )
            .sort(by=sort_col)
            .with_columns([pl.col(f).forward_fill().over(over_col).alias(f) for f in features])
        )
    except AttributeError as e:
        raise TypeError("`df` 必须是 Polars DataFrame 或 LazyFrame，但缺少必需属性") from e
    except AssertionError as e:
        raise ValueError(f"`df` 必须包含以下所有列：{[over_col, sort_col] + list(features)}") from e


def smooth_features(
    df: pl.DataFrame | pl.LazyFrame,
    features: tuple[str, ...],
    sort_col: str,
    over_col: str,
    window_size: int,
) -> pl.LazyFrame:
    """通过计算每列的滚动均值来平滑 `df` 的 `features` 列，
    按 `sort_col` 排序并按 `over_col` 分组，使用 `window_size` 个滞后周期作为移动平均窗口。

    参数
    ----------
    df: Polars DataFrame 或 LazyFrame，包含列 `sort_col`、`over_col` 和每个 `features`
    features: 字符串集合，指示 `df` 中的哪些列是特征值
    sort_col: `df` 的列，指示如何排序
    over_col: `df` 的列，指示如何分组
    window_size: 整数，移动平均的时间周期数

    返回
    -------
    Polars LazyFrame，包含原始列，每个 `features` 替换为移动平均
    """
    try:
        # 急切检查 `over_col`、`sort_col`、`features` 是否存在：在延迟上下文中无法捕获 pl.ColumnNotFoundError
        assert all(c in df.columns for c in features + (over_col, sort_col))
        return (
            df.lazy()
            .sort(by=sort_col)
            .with_columns([pl.col(f).rolling_mean(window_size=window_size).over(over_col).alias(f) for f in features])
        )
    except AttributeError as e:
        raise TypeError("`df` 必须是 Polars DataFrame 或 LazyFrame，但缺少必需属性") from e
    except AssertionError as e:
        raise ValueError(f"`df` 必须包含以下所有列：{[over_col, sort_col] + list(features)}") from e


def top_n_by_group(
    df: pl.DataFrame | pl.LazyFrame,
    n: int,
    rank_var: str,
    group_var: tuple[str, ...],
    filter: bool = True,
) -> pl.LazyFrame:
    """根据 `rank_var` 降序标记每个 `group_var` 分组中的前 `n` 行。

    如果 `filter` 为 True，返回的 DataFrame 仅包含过滤后的数据。如果 `filter` 为 False，
    返回的 DataFrame 包含所有数据，并附加一个 'rank_mask' 列指示该行是否在过滤器中。

    参数
    ----------
    df: Polars DataFrame 或 LazyFrame
    n: 整数，指示每个组中要取的前几行
    rank_var: 排名依据的列名
    group_var: 用于分组和排序的列名元组
    filter: 布尔值，指示返回多少数据

    返回
    -------
    Polars LazyFrame，包含原始列和可选的过滤器列
    """
    try:
        # 急切检查 `rank_var`、`group_var` 是否存在：在延迟上下文中无法捕获 ColumnNotFoundError
        assert all(c in df.columns for c in (rank_var,) + group_var)
        rdf = (
            df.lazy()
            .sort(by=list(group_var) + [rank_var])
            .with_columns(pl.col(rank_var).rank(descending=True).over(group_var).cast(int).alias("rank"))
        )
        match filter:
            case True:
                return rdf.filter(pl.col("rank") <= n).drop("rank").sort(by=list(group_var) + [rank_var])
            case False:
                return (
                    rdf.with_columns(
                        pl.when(pl.col("rank") <= n).then(pl.lit(1)).otherwise(pl.lit(0)).alias("rank_mask")
                    )
                    .drop("rank")
                    .sort(by=list(group_var) + [rank_var])
                )
    except AssertionError as e:
        raise ValueError(f"`df` 缺少一个或多个必需列：'{rank_var}' 和 '{group_var}'") from e
    except AttributeError as e:
        raise TypeError("`df` 必须是 Polars DataFrame 或 LazyFrame，但缺少必需属性") from e
