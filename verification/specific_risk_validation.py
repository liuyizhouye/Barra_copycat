"""特异收益与特异风险验证：MRAD 与 Bias 指标体系。

包括：
- MRAD (Mean Ratio of Actual to Predicted volatility)
- Bias 统计量 (Mean Bias Statistic)
"""

import numpy as np
import polars as pl


def calculate_mrad(
    residual_returns_df: pl.DataFrame,
    specific_risk_df: pl.DataFrame,
    window: int = 252,
    annualize: bool = True,
) -> pl.DataFrame:
    """计算 MRAD (Mean Ratio of Actual to Predicted volatility)。

    MRAD = (1/N) * sum(σ_realized,i / σ_predicted,i)

    表示预测风险与实际波动的偏离程度。MRAD 越接近 1，说明风险预测越准确。

    参数
    ----------
    residual_returns_df: Polars DataFrame，残差收益，包含列：| date | symbol1 | symbol2 | ... |
    specific_risk_df: Polars DataFrame，特异风险预测，包含列：| date | symbol | specific_risk | 或 | specific_risk_shrunk |
    window: 计算实际波动率的窗口长度，默认252
    annualize: 是否年化，默认True

    返回
    -------
    Polars DataFrame，包含各日期的 MRAD 值
    列：| date | mrad | n_assets | mean_ratio | std_ratio |
    """
    symbol_cols = [col for col in residual_returns_df.columns if col != "date"]
    dates = residual_returns_df["date"].unique().sort().to_list()

    results = []
    for i, date in enumerate(dates):
        # 获取到当前日期的残差收益数据
        data_up_to_date = residual_returns_df.filter(pl.col("date") <= date).sort("date")
        
        # 获取当前日期的预测风险
        current_risks = specific_risk_df.filter(pl.col("date") == date)
        
        # 确定使用哪个风险列
        risk_col = "specific_risk_shrunk" if "specific_risk_shrunk" in current_risks.columns else "specific_risk"
        
        ratios = []
        for symbol in symbol_cols:
            # 获取该股票的历史残差收益
            returns_series = (
                data_up_to_date.select(["date", symbol])
                .drop_nulls()
                .sort("date")[symbol]
                .to_numpy()
            )
            
            if len(returns_series) < window:
                continue
            
            # 计算实际波动率（最近 window 个值）
            recent_returns = returns_series[-window:]
            actual_vol = np.std(recent_returns)
            if annualize:
                actual_vol *= np.sqrt(252)
            
            # 获取预测风险
            predicted_risk = (
                current_risks.filter(pl.col("symbol") == symbol)[risk_col].to_numpy()
            )
            
            if len(predicted_risk) > 0 and predicted_risk[0] > 1e-10:
                ratio = actual_vol / predicted_risk[0]
                ratios.append(ratio)
        
        if len(ratios) > 0:
            mrad = np.mean(ratios)
            mean_ratio = mrad
            std_ratio = np.std(ratios)
            
            results.append({
                "date": date,
                "mrad": mrad,
                "n_assets": len(ratios),
                "mean_ratio": mean_ratio,
                "std_ratio": std_ratio,
            })

    return pl.DataFrame(results)


def calculate_bias_statistic(
    residual_returns_df: pl.DataFrame,
    specific_risk_df: pl.DataFrame,
    window: int = 252,
    annualize: bool = True,
) -> pl.DataFrame:
    """计算 Bias 统计量 (Mean Bias Statistic)。

    平均预测偏差，衡量系统性高估或低估风险的程度。Bias≈1 表示无系统偏差。

    参数
    ----------
    residual_returns_df: Polars DataFrame，残差收益
    specific_risk_df: Polars DataFrame，特异风险预测
    window: 计算实际波动率的窗口长度，默认252
    annualize: 是否年化，默认True

    返回
    -------
    Polars DataFrame，包含各日期的 Bias 值
    列：| date | bias | n_assets | mean_predicted | mean_actual |
    """
    symbol_cols = [col for col in residual_returns_df.columns if col != "date"]
    dates = residual_returns_df["date"].unique().sort().to_list()

    results = []
    for i, date in enumerate(dates):
        # 获取到当前日期的残差收益数据
        data_up_to_date = residual_returns_df.filter(pl.col("date") <= date).sort("date")
        
        # 获取当前日期的预测风险
        current_risks = specific_risk_df.filter(pl.col("date") == date)
        risk_col = "specific_risk_shrunk" if "specific_risk_shrunk" in current_risks.columns else "specific_risk"
        
        predicted_risks = []
        actual_risks = []
        
        for symbol in symbol_cols:
            # 获取该股票的历史残差收益
            returns_series = (
                data_up_to_date.select(["date", symbol])
                .drop_nulls()
                .sort("date")[symbol]
                .to_numpy()
            )
            
            if len(returns_series) < window:
                continue
            
            # 计算实际波动率
            recent_returns = returns_series[-window:]
            actual_vol = np.std(recent_returns)
            if annualize:
                actual_vol *= np.sqrt(252)
            
            # 获取预测风险
            predicted_risk = (
                current_risks.filter(pl.col("symbol") == symbol)[risk_col].to_numpy()
            )
            
            if len(predicted_risk) > 0 and predicted_risk[0] > 1e-10:
                predicted_risks.append(predicted_risk[0])
                actual_risks.append(actual_vol)
        
        if len(predicted_risks) > 0:
            mean_predicted = np.mean(predicted_risks)
            mean_actual = np.mean(actual_risks)
            
            # Bias = mean_actual / mean_predicted
            bias = mean_actual / mean_predicted if mean_predicted > 1e-10 else np.nan
            
            results.append({
                "date": date,
                "bias": bias,
                "n_assets": len(predicted_risks),
                "mean_predicted": mean_predicted,
                "mean_actual": mean_actual,
            })

    return pl.DataFrame(results)


def calculate_specific_risk_validations(
    residual_returns_df: pl.DataFrame,
    specific_risk_df: pl.DataFrame,
    window: int = 252,
    annualize: bool = True,
) -> dict:
    """计算所有特异风险验证指标。

    返回包含所有验证结果的字典。
    """
    return {
        "mrad": calculate_mrad(residual_returns_df, specific_risk_df, window, annualize),
        "bias": calculate_bias_statistic(residual_returns_df, specific_risk_df, window, annualize),
    }

