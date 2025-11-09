"""模型中使用的数学和统计操作。"""

import numpy as np
import polars as pl


def center_xsection(target_col: str, over_col: str, standardize: bool = False) -> pl.Expr:
    """按 `over_col` 分组对 `target_col` 列进行横截面去中心化（可选标准化）。

    返回一个 Polars 表达式，因此可以在 `select` 或 `with_columns` 调用中链式使用，
    无需设置新的中间 DataFrame 或物化延迟求值。

    参数
    ----------
    target_col: 要标准化的列名
    over_col: 应应用标准化的分组列，进行横截面处理
    standardize: 布尔值，指示是否也应标准化目标列

    返回
    -------
    Polars Expr
    """
    expr = pl.col(target_col) - pl.col(target_col).drop_nulls().drop_nans().mean().over(over_col)
    if standardize:
        return expr / pl.col(target_col).drop_nulls().drop_nans().std().over(over_col)
    return expr


def norm_xsection(
    target_col: str,
    over_col: str,
    lower: int | float = 0,
    upper: int | float = 1,
) -> pl.Expr:
    """按 `over_col` 分组将 `target_col` 列进行横截面归一化，重缩放到区间 [`lower`, `upper`]。

    返回一个 Polars 表达式，因此可以在 `select` 或 `with_columns` 调用中链式使用，
    无需设置新的中间 DataFrame 或物化延迟求值。

    NaN 值不会传播到最大值和最小值计算中，但 NaN 值会在归一化中保留。

    参数
    ----------
    target_col: 要归一化的列名
    over_col: 用于分组的列名
    lower: 重缩放区间的下界，默认为 0 以构造百分比
    upper: 重缩放区间的上界，默认为 1 以构造百分比

    返回
    -------
    Polars Expr
    """
    min_col = pl.col(target_col).drop_nans().min().over(over_col)
    max_col = pl.col(target_col).drop_nans().max().over(over_col)

    norm_col = (
        pl.when(pl.col(target_col).is_nan())
        .then(pl.col(target_col))  # 保留 NaN 值
        .when(max_col != min_col)  # 确保 min != max 以避免除零
        .then((pl.col(target_col) - min_col) / (max_col - min_col) * (upper - lower) + lower)
        .otherwise(lower)
    )

    return norm_col


def winsorize(data: np.ndarray, percentile: float = 0.05, axis: int = 0) -> np.ndarray:
    """将 2D numpy 数组的每个向量按给定的 `percentile` 对称分位数进行去极值处理。

    返回一个 numpy 数组，不是 DataFrame。

    参数
    ----------
    data: 要进行去极值处理的 numpy 数组
    percentile: 浮点数，指示应用去极值的分位数
    axis: 整数，指示应用去极值的轴（即如果数据是 2D 的方向）

    返回
    -------
    numpy array
    """
    try:
        if not 0 <= percentile <= 1:
            raise ValueError("`percentile` 必须在 0 到 1 之间")
    except AttributeError as e:
        raise TypeError("`percentile` 必须是数字类型，如 int 或 float") from e

    fin_data = np.where(np.isfinite(data), data, np.nan)

    # 计算每列的下限和上限分位数
    lower_bounds = np.nanpercentile(fin_data, percentile * 100, axis=axis, keepdims=True)
    upper_bounds = np.nanpercentile(fin_data, (1 - percentile) * 100, axis=axis, keepdims=True)

    # 将数据裁剪到边界内
    return np.clip(data, lower_bounds, upper_bounds)


def winsorize_xsection(
    df: pl.DataFrame | pl.LazyFrame,
    data_cols: tuple[str, ...],
    group_col: str,
    percentile: float = 0.05,
) -> pl.DataFrame | pl.LazyFrame:
    """按 `group_col` 分组对 `df` 的 `data_cols` 列进行横截面去极值处理，
    使用由 `percentile` 给定的对称分位数。

    参数
    ----------
    df: Polars DataFrame 或 LazyFrame，包含要去极值的特征数据
    data_cols: 字符串集合，指示要接受去极值的列
    group_col: `df` 的分组列，用作横截面分组
    percentile: 浮点数，指示对称去极值阈值

    返回
    -------
    Polars DataFrame 或 LazyFrame
    """

    def winsorize_group(group: pl.DataFrame) -> pl.DataFrame:
        for col in data_cols:
            winsorized_data = winsorize(group[col].to_numpy(), percentile=percentile)
            group = group.with_columns(pl.Series(col, winsorized_data).alias(col))
        return group

    match df:
        case pl.DataFrame():
            grouped = df.group_by(group_col).map_groups(winsorize_group)
        case pl.LazyFrame():
            grouped = df.group_by(group_col).map_groups(winsorize_group, schema=df.collect_schema())
        case _:
            raise TypeError("`df` 必须是 Polars DataFrame 或 LazyFrame")
    return grouped


def percentiles_xsection(
    target_col: str,
    over_col: str,
    lower_pct: float,
    upper_pct: float,
    fill_val: float | int = 0.0,
) -> pl.Expr:
    """横截面标记每个 `over_col` 分组中落在 `lower_pct` 或 `upper_pct` 分位数之外
    的 `target_col` 的所有值。这本质上是一种反去极值处理，适用于构建做多-做空组合。
    在分位数截断点之间的值用 `fill_val` 填充。

    返回一个 Polars 表达式，因此可以在 `select` 或 `with_columns` 调用中链式使用，
    无需设置新的中间 DataFrame 或物化延迟求值。

    参数
    ----------
    target_col: 列名，非分位数阈值内的值将被掩蔽
    over_col: 列名，横截面应用掩蔽
    lower_pct: 浮点数，要保留值的最低分位数
    upper_pct: 浮点数，要保留值的最高分位数
    fill_val: 用于掩蔽的数值

    返回
    -------
    Polars Expr
    """
    return (
        pl.when(
            (pl.col(target_col) <= pl.col(target_col).drop_nans().quantile(lower_pct).over(over_col))
            | (pl.col(target_col) >= pl.col(target_col).drop_nans().quantile(upper_pct).over(over_col))
        )
        .then(pl.col(target_col))
        .otherwise(fill_val)
    )


def exp_weights(window: int, half_life: int) -> np.ndarray:
    """在 `window` 个滞后值上生成指数衰减权重，每 `half_life` 个索引权重减半。

    参数
    ----------
    window: 滞后回看期中的点数
    half_life: 整数，衰减率

    返回
    -------
    numpy array
    """
    try:
        assert isinstance(window, int)
        if not window > 0:
            raise ValueError("`window` 必须是严格正整数")
    except (AttributeError, AssertionError) as e:
        raise TypeError("`window` 必须是整数类型") from e
    try:
        assert isinstance(half_life, int)
        if not half_life > 0:
            raise ValueError("`half_life` 必须是严格正整数")
    except (AttributeError, AssertionError) as e:
        raise TypeError("`half_life` 必须是整数类型") from e
    decay = np.log(2) / half_life
    return np.exp(-decay * np.arange(window))[::-1]
