"""因子模型完整实现。"""

import numpy as np
import polars as pl
import polars.exceptions as pl_exc

from barra_copycat.math import winsorize


def _factor_returns(
    returns: np.ndarray,
    mkt_caps: np.ndarray,
    sector_scores: np.ndarray,
    style_scores: np.ndarray,
    residualize_styles: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """估计单个时间点的市场因子、行业因子、风格因子和残差收益，对秩缺陷稳健。

    参数
    ----------
    returns: 资产收益数组 (形状: n_assets x 1)
    mkt_caps: 资产市值数组 (形状: n_assets x 1)
    sector_scores: 用于估计行业收益的资产得分矩阵 (形状: n_assets x m_sectors)
    style_scores: 用于估计风格因子收益的资产得分矩阵 (形状: n_assets x m_styles)
    residualize_styles: 布尔值，指示是否将风格因子正交化到市场+行业因子

    返回
    -------
    数组元组: (市场/行业/风格因子收益, 残差收益)
    """
    n_assets = returns.shape[0]
    m_sectors, m_styles = sector_scores.shape[1], style_scores.shape[1]

    # 用作资产特异性方差的逆的代理变量
    W = np.diag(np.sqrt(mkt_caps.ravel()))

    # 估计行业因子收益，约束条件是所有行业因子收益之和为0
    # 经济学含义：市场收益完全由行业收益线性组合构成
    beta_sector = np.hstack([np.ones(n_assets).reshape(-1, 1), sector_scores])
    a = np.concatenate([np.array([0]), (-1 * np.ones(m_sectors - 1))])
    Imat = np.identity(m_sectors)
    R_sector = np.vstack([Imat, a])
    # 变量变换以添加约束条件
    B_sector = beta_sector @ R_sector

    V_sector, _, _, _ = np.linalg.lstsq(B_sector.T @ W @ B_sector, B_sector.T @ W, rcond=None)
    # 变量变换以恢复所有行业
    g = V_sector @ returns
    fac_ret_sector = R_sector @ g

    sector_resid_returns = returns - (B_sector @ g)

    # 估计风格因子收益，无约束条件
    V_style, _, _, _ = np.linalg.lstsq(style_scores.T @ W @ style_scores, style_scores.T @ W, rcond=None)
    if residualize_styles:
        fac_ret_style = V_style @ sector_resid_returns
    else:
        fac_ret_style = V_style @ returns

    # 合并因子收益
    fac_ret = np.concatenate([fac_ret_sector, fac_ret_style])

    # 计算最终残差
    epsilon = sector_resid_returns - (style_scores @ fac_ret_style)

    return fac_ret, epsilon


def estimate_factor_returns(
    returns_df: pl.DataFrame,
    mkt_cap_df: pl.DataFrame,
    sector_df: pl.DataFrame,
    style_df: pl.DataFrame,
    winsor_factor: float | None = 0.05,
    residualize_styles: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame] | pl.DataFrame:
    """使用输入的资产因子得分估计所有时间段的因子收益和残差收益。

    参数
    ----------
    returns_df: Polars DataFrame，包含列：| date | symbol | asset_returns |
    mkt_cap_df: Polars DataFrame，包含列：| date | symbol | market_cap |
    sector_df: Polars DataFrame，包含列：| date | symbol | 以及每个行业一列
    style_df: Polars DataFrame，包含列：| date | symbol | 以及每个风格一列
    winsor_factor: 去极值的比例
    residualize_styles: 布尔值，指示风格收益是否应正交化到市场+行业收益

    返回
    -------
    按日期展开的Polars DataFrame元组：(因子收益, 残差收益)
    """
    returns, residuals = [], []
    try:
        sectors = sorted(sector_df.select(pl.exclude("date", "symbol")).columns)
    except AttributeError as e:
        raise TypeError("`sector_df` 必须是 Polars DataFrame，但缺少必需属性") from e
    except pl_exc.ColumnNotFoundError as e:
        raise ValueError("`sector_df` 必须包含 'date' 和 'symbol' 列以及每个行业列") from e
    try:
        styles = sorted(style_df.select(pl.exclude("date", "symbol")).columns)
    except AttributeError as e:
        raise TypeError("`style_df` 必须是 Polars DataFrame，但缺少必需属性") from e
    except pl_exc.ColumnNotFoundError as e:
        raise ValueError("`style_df` 必须包含 'date' 和 'symbol' 列以及每个风格列") from e
    try:
        returns_df = (
            returns_df.join(mkt_cap_df, on=["date", "symbol"])
            .join(sector_df, on=["date", "symbol"])
            .join(style_df, on=["date", "symbol"])
        )
        dates = returns_df["date"].unique().to_list()
        # 逐日迭代
        # 这可以通过 Polars 的 `.map_groups` 方法提高效率
        for dt in dates:
            ddf = returns_df.filter(pl.col("date") == dt).sort("symbol")
            r = ddf["asset_returns"].to_numpy()
            if winsor_factor is not None:
                r = winsorize(r, winsor_factor)
            f, e = _factor_returns(
                r,
                ddf["market_cap"].to_numpy(),
                ddf.select(sectors).to_numpy(),
                ddf.select(styles).to_numpy(),
                residualize_styles,
            )
            returns.append(f)
            residuals.append(dict(zip(ddf["symbol"].to_list(), e)))
    except AttributeError as e:
        raise TypeError(
            "`returns_df` 和 `mkt_cap_df` 必须是 Polars DataFrame，但缺少某些属性"
        ) from e
    except pl_exc.ColumnNotFoundError as e:
        raise ValueError(
            "`returns_df` 必须包含 'date'、'symbol' 和 'asset_returns' 列；"
            "`mkt_cap_df` 必须包含 'date'、'symbol' 和 'market_cap' 列"
        ) from e
    ret_df = pl.DataFrame(np.array(returns))
    ret_df.columns = ["market"] + sectors + styles
    ret_df = ret_df.with_columns(pl.Series(dates).alias("date"))
    eps_df = pl.DataFrame(residuals).with_columns(pl.Series(dates).alias("date"))
    return ret_df, eps_df
