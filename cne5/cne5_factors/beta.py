"""CNE5 Beta 因子: 对超额收益做时间序列回归。"""

from typing import Union, Optional

import numpy as np
import polars as pl
import polars.exceptions as pl_exc

from cne5.math import exp_weights
from cne5.cne5_factors._utils import standardize_cne5


def factor_beta_cne5(
    returns_df: Union[pl.DataFrame, pl.LazyFrame],
    mkt_cap_df: Union[pl.DataFrame, pl.LazyFrame],
    risk_free_df: Optional[Union[pl.DataFrame, pl.LazyFrame]] = None,
    window: int = 252,
    half_life: int = 63,
) -> pl.LazyFrame:
    """CNE5 Beta 因子: 对超额收益做时间序列回归。

    参数
    ----------
    returns_df: Polars DataFrame，包含列：| date | symbol | asset_returns |
    mkt_cap_df: Polars DataFrame，包含列：| date | symbol | market_cap |
    risk_free_df: 可选，无风险利率数据，包含列：| date | risk_free_rate |
    window: 估计窗口长度(交易日)，默认252
    half_life: 半衰期(交易日)，默认63

    返回
    -------
    Polars LazyFrame，包含列：| date | symbol | beta |
    """
    try:
        # 合并数据
        df = returns_df.lazy() if isinstance(returns_df, pl.DataFrame) else returns_df
        df = df.join(
            mkt_cap_df.lazy() if isinstance(mkt_cap_df, pl.DataFrame) else mkt_cap_df,
            on=["date", "symbol"],
        )

        # 计算市场超额收益(市值加权)
        if risk_free_df is not None:
            rf_df = risk_free_df.lazy() if isinstance(risk_free_df, pl.DataFrame) else risk_free_df
            df = df.join(rf_df, on="date")
            df = df.with_columns(
                (pl.col("asset_returns") - pl.col("risk_free_rate")).alias("excess_returns")
            )
        else:
            df = df.with_columns(pl.col("asset_returns").alias("excess_returns"))

        # 计算市场组合超额收益(市值加权)
        df = df.with_columns(
            (
                (pl.col("excess_returns") * pl.col("market_cap")).sum().over("date")
                / pl.col("market_cap").sum().over("date")
            ).alias("market_excess_returns")
        )

        # 计算 Beta: 使用指数加权滚动回归
        # 需要先收集数据以获取市场收益序列（与每个股票对齐）
        df_collected = df.sort("date").collect()
        
        # 获取市场收益字典（按日期索引）
        market_returns_dict = (
            df_collected.group_by("date")
            .agg(pl.col("market_excess_returns").first())
            .to_dicts()
        )
        market_returns_by_date = {item["date"]: item["market_excess_returns"] for item in market_returns_dict}
        
        weights = exp_weights(window, half_life)

        def calc_beta_group(group: pl.DataFrame) -> pl.DataFrame:
            """按股票分组计算 Beta。"""
            excess_returns = group.sort("date")["excess_returns"].to_numpy()
            dates = group.sort("date")["date"].to_list()
            
            beta_values = []
            for i in range(len(group)):
                if i < window - 1:
                    beta_values.append(np.nan)
                else:
                    # 取最后 window 个值
                    y = excess_returns[i - window + 1 : i + 1]
                    # 获取对应的市场收益
                    market_vals = np.array([
                        market_returns_by_date.get(d, np.nan)
                        for d in dates[i - window + 1 : i + 1]
                    ])
                    
                    if np.any(np.isnan(market_vals)) or len(market_vals) != window:
                        beta_values.append(np.nan)
                        continue
                    
                    X = market_vals.reshape(-1, 1)
                    w = weights

                    # 加权最小二乘
                    W = np.diag(w)
                    try:
                        XWX = X.T @ W @ X
                        XWy = X.T @ W @ y
                        if XWX[0, 0] < 1e-10:
                            beta_values.append(np.nan)
                        else:
                            beta = XWy[0] / XWX[0, 0]
                            beta_values.append(beta)
                    except (np.linalg.LinAlgError, ZeroDivisionError):
                        beta_values.append(np.nan)
            
            return group.with_columns(pl.Series("beta", beta_values))

        df = (
            df_collected.lazy()
            .group_by("symbol")
            .map_groups(calc_beta_group, schema=df_collected.schema)
        )

        df = standardize_cne5(df, "beta", "market_cap")
        return df.select("date", "symbol", "beta")

    except AttributeError as e:
        raise TypeError("输入必须是 Polars DataFrame 或 LazyFrame") from e
    except pl_exc.ColumnNotFoundError as e:
        raise ValueError("缺少必需的列") from e
