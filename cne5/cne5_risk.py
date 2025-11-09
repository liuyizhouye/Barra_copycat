"""CNE5 特异风险模型实现。

包括：
- 特异风险模型（日频序列法）
- 贝叶斯收缩
"""

from typing import Optional

import numpy as np
import polars as pl

from cne5.math import exp_weights


def estimate_specific_risk_ts(
    residual_returns_df: pl.DataFrame,
    window: int = 252,
    half_life: int = 63,
) -> pl.DataFrame:
    """使用日频序列法估计特异风险。

    参数
    ----------
    residual_returns_df: Polars DataFrame，包含列：| date | symbol1 | symbol2 | ... |
    window: 估计窗口长度，默认252
    half_life: 半衰期，默认63

    返回
    -------
    Polars DataFrame，包含列：| date | symbol | specific_risk |
    """
    symbol_cols = [col for col in residual_returns_df.columns if col != "date"]
    dates = residual_returns_df["date"].to_list()

    weights = exp_weights(window, half_life) if half_life is not None else None

    results = []
    for i, date in enumerate(dates):
        # 获取到当前日期的数据
        data_up_to_date = residual_returns_df.filter(pl.col("date") <= date).sort("date")

        date_results = {"date": date}
        for symbol in symbol_cols:
            # 获取该股票的历史残差收益
            returns_series = (
                data_up_to_date.select(["date", symbol])
                .drop_nulls()
                .sort("date")[symbol]
                .to_numpy()
            )

            if len(returns_series) < window:
                date_results[symbol] = np.nan
                continue

            # 取最后 window 个值
            recent_returns = returns_series[-window:]

            if weights is not None:
                # 指数加权标准差
                w = weights[-len(recent_returns):]
                w = w / w.sum()
                weighted_mean = np.average(recent_returns, weights=w)
                weighted_var = np.average((recent_returns - weighted_mean) ** 2, weights=w)
                risk = np.sqrt(weighted_var * 252)  # 年化
            else:
                # 简单标准差
                risk = np.std(recent_returns) * np.sqrt(252)  # 年化

            date_results[symbol] = risk

        results.append(date_results)

    # 转换为长格式
    risk_df = pl.DataFrame(results)
    risk_df = risk_df.melt(
        id_vars=["date"], variable_name="symbol", value_name="specific_risk"
    )

    return risk_df


def bayesian_shrinkage(
    specific_risk_df: pl.DataFrame,
    mkt_cap_df: pl.DataFrame,
    n_buckets: int = 10,
) -> pl.DataFrame:
    """贝叶斯收缩：按市值分位将股票分成桶，向桶内均值收缩。

    参数
    ----------
    specific_risk_df: Polars DataFrame，包含列：| date | symbol | specific_risk |
    mkt_cap_df: Polars DataFrame，包含列：| date | symbol | market_cap |
    n_buckets: 市值分位数桶数，默认10

    返回
    -------
    Polars DataFrame，包含列：| date | symbol | specific_risk_shrunk |
    """
    # 合并数据
    df = specific_risk_df.join(mkt_cap_df, on=["date", "symbol"])

    # 按日期分组处理
    def shrink_group(group: pl.DataFrame) -> pl.DataFrame:
        if group.is_empty() or group["specific_risk"].null_count() == len(group):
            return group.select("date", "symbol").with_columns(
                pl.lit(np.nan).alias("specific_risk_shrunk")
            )

        # 计算市值分位数
        try:
            group = group.with_columns(
                pl.col("market_cap")
                .qcut(n_buckets, labels=[f"bucket_{i}" for i in range(n_buckets)])
                .alias("size_bucket")
            )
        except Exception:
            # 如果分位数失败，使用简单分组
            group = group.with_columns(
                (pl.col("market_cap").rank() // (len(group) / n_buckets + 1)).cast(int).alias("size_bucket")
            )

        # 计算每个桶内的均值和标准差
        bucket_stats = (
            group.group_by("size_bucket")
            .agg(
                pl.col("specific_risk").mean().alias("bucket_mean"),
                pl.col("specific_risk").std().alias("bucket_std"),
            )
            .filter(pl.col("bucket_std").is_not_null())
        )

        if bucket_stats.is_empty():
            return group.select("date", "symbol").with_columns(
                pl.col("specific_risk").alias("specific_risk_shrunk")
            )

        # 合并统计信息
        group = group.join(bucket_stats, on="size_bucket", how="left")

        # 计算收缩强度（离均值越远，收缩强度越大）
        # 使用贝叶斯收缩公式：shrinkage = 1 / (1 + (individual_std / bucket_std)^2)
        group = group.with_columns(
            (
                pl.when(pl.col("bucket_std") > 1e-10)
                .then(
                    1.0
                    / (
                        1.0
                        + (
                            (pl.col("specific_risk") - pl.col("bucket_mean"))
                            / pl.col("bucket_std")
                        ).pow(2)
                    )
                )
                .otherwise(1.0)
            ).alias("shrinkage_factor")
        )

        # 收缩：shrunk = shrinkage * individual + (1 - shrinkage) * bucket_mean
        group = group.with_columns(
            (
                pl.when(pl.col("bucket_mean").is_not_null())
                .then(
                    pl.col("shrinkage_factor") * pl.col("specific_risk")
                    + (1 - pl.col("shrinkage_factor")) * pl.col("bucket_mean")
                )
                .otherwise(pl.col("specific_risk"))
            ).alias("specific_risk_shrunk")
        )

        return group.select("date", "symbol", "specific_risk_shrunk")

    result = df.group_by("date").map_groups(shrink_group, schema=df.collect_schema())

    return result.select("date", "symbol", "specific_risk_shrunk")


def calculate_rms_decomposition(
    returns_df: pl.DataFrame,
    factor_returns_df: pl.DataFrame,
    style_df: pl.DataFrame,
    sector_df: pl.DataFrame,
    mkt_cap_df: pl.DataFrame,
) -> pl.DataFrame:
    """计算 RMS（均方根）收益的"x-sigma-rho"截面分解。

    参数
    ----------
    returns_df: Polars DataFrame，包含列：| date | symbol | asset_returns |
    factor_returns_df: Polars DataFrame，因子收益
    style_df: Polars DataFrame，风格因子暴露
    sector_df: Polars DataFrame，行业因子暴露
    mkt_cap_df: Polars DataFrame，市值数据

    返回
    -------
    Polars DataFrame，包含 RMS 分解结果
    """
    # 合并数据
    df = (
        returns_df.join(mkt_cap_df, on=["date", "symbol"])
        .join(style_df, on=["date", "symbol"])
        .join(sector_df, on=["date", "symbol"])
    )

    dates = df["date"].unique().to_list()
    style_cols = [col for col in style_df.columns if col not in ["date", "symbol"]]
    sector_cols = [col for col in sector_df.columns if col not in ["date", "symbol"]]

    results = []
    for date in dates:
        date_data = df.filter(pl.col("date") == date)
        returns = date_data["asset_returns"].to_numpy()

        # 计算 RMS
        rms_total = np.sqrt(np.mean(returns ** 2))

        # 计算各因子的贡献
        factor_contributions = {}
        for factor_name in ["country"] + sector_cols + style_cols:
            if factor_name == "country":
                # 国家因子暴露为1
                exposures = np.ones(len(returns))
            elif factor_name in sector_cols:
                exposures = date_data[factor_name].to_numpy()
            else:
                exposures = date_data[factor_name].to_numpy()

            # 获取因子收益
            factor_return = (
                factor_returns_df.filter(pl.col("date") == date)[factor_name].to_numpy()[0]
            )

            # 计算 sigma (RMS 离散度)
            sigma = np.sqrt(np.mean(exposures ** 2))

            # 计算 rho (截面相关)
            if sigma > 1e-10:
                rho = np.corrcoef(returns, exposures)[0, 1]
            else:
                rho = 0.0

            # 贡献 = sigma * rho * factor_return
            contribution = sigma * rho * factor_return
            factor_contributions[factor_name] = contribution

        results.append(
            {
                "date": date,
                "rms_total": rms_total,
                **{f"{k}_contribution": v for k, v in factor_contributions.items()},
            }
        )

    return pl.DataFrame(results)


def calculate_style_stability(
    style_df: pl.DataFrame,
) -> pl.DataFrame:
    """计算风格稳定性系数：月度相邻期暴露的截面相关。

    参数
    ----------
    style_df: Polars DataFrame，包含列：| date | symbol | style1 | style2 | ... |

    返回
    -------
    Polars DataFrame，包含各风格因子的稳定性系数
    """
    style_cols = [col for col in style_df.columns if col not in ["date", "symbol"]]
    dates = sorted(style_df["date"].unique().to_list())

    results = []
    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        curr_date = dates[i]

        prev_data = style_df.filter(pl.col("date") == prev_date).sort("symbol")
        curr_data = style_df.filter(pl.col("date") == curr_date).sort("symbol")

        # 合并确保相同的股票
        merged = prev_data.join(curr_data, on="symbol", suffix="_curr")

        stability = {}
        for col in style_cols:
            prev_values = merged[col].to_numpy()
            curr_values = merged[f"{col}_curr"].to_numpy()

            if len(prev_values) > 1 and np.std(prev_values) > 1e-10 and np.std(curr_values) > 1e-10:
                corr = np.corrcoef(prev_values, curr_values)[0, 1]
            else:
                corr = np.nan

            stability[col] = corr

        results.append({"date": curr_date, **stability})

    return pl.DataFrame(results)


def calculate_vif(
    style_df: pl.DataFrame,
    sector_df: pl.DataFrame,
    date: Optional[str] = None,
) -> pl.DataFrame:
    """计算方差膨胀因子（VIF），衡量因子间共线性。

    参数
    ----------
    style_df: Polars DataFrame，风格因子暴露
    sector_df: Polars DataFrame，行业因子暴露
    date: 可选，指定日期，如果为 None 则计算所有日期

    返回
    -------
    Polars DataFrame，包含各因子的 VIF 值
    """
    # 合并因子暴露
    df = style_df.join(sector_df, on=["date", "symbol"])

    if date is not None:
        df = df.filter(pl.col("date") == date)

    dates = df["date"].unique().to_list()
    factor_cols = [
        col
        for col in df.columns
        if col not in ["date", "symbol"]
    ]

    results = []
    for dt in dates:
        date_data = df.filter(pl.col("date") == dt)
        X = date_data.select(factor_cols).to_numpy()

        # 计算 VIF
        vif_values = {}
        for i, factor_name in enumerate(factor_cols):
            # 将该因子作为因变量，其他因子作为自变量
            y = X[:, i]
            X_other = np.delete(X, i, axis=1)

            try:
                # 多元回归
                X_with_const = np.hstack([np.ones((len(X_other), 1)), X_other])
                beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
                y_pred = X_with_const @ beta
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)

                if ss_res > 1e-10:
                    r_squared = 1 - (ss_res / ss_tot)
                    vif = 1 / (1 - r_squared) if r_squared < 1 - 1e-10 else np.inf
                else:
                    vif = np.inf
            except np.linalg.LinAlgError:
                vif = np.nan

            vif_values[factor_name] = vif

        results.append({"date": dt, **vif_values})

    return pl.DataFrame(results)

