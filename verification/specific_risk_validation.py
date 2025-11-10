"""特异收益与特异风险验证：MRAD 与 Bias 指标体系。

包括：
- MRAD (Mean Ratio of Actual to Predicted volatility)
- Bias 统计量 (Mean Bias Statistic)

注意：使用前向验证方法，避免前视偏差。
"""

import numpy as np
import polars as pl


def calculate_mrad(
    residual_returns_df: pl.DataFrame,
    specific_risk_df: pl.DataFrame,
    forward_window: int = 21,
    annualize: bool = True,
) -> pl.DataFrame:
    """计算 MRAD (Mean Ratio of Actual to Predicted volatility) - 前向验证版本。

    MRAD = (1/N) * sum(σ_realized,i / σ_predicted,i)

    表示预测风险与实际波动的偏离程度。MRAD 越接近 1，说明风险预测越准确。

    本函数使用前向验证方法：在日期t的预测，使用t之后的forward_window天数据验证，
    避免前视偏差（look-ahead bias）。

    参数
    ----------
    residual_returns_df: Polars DataFrame，残差收益，包含列：| date | symbol1 | symbol2 | ... |
    specific_risk_df: Polars DataFrame，特异风险预测，包含列：| date | symbol | specific_risk | 或 | specific_risk_shrunk |
    forward_window: 前向验证窗口长度（交易日），默认21（约1个月）
    annualize: 是否年化，默认True

    返回
    -------
    Polars DataFrame，包含各日期的 MRAD 值
    列：| date | mrad | n_assets | mean_ratio | std_ratio |

    注意
    ----
    - 最后forward_window天的数据无法验证，会被排除
    - 实际波动使用预测日期之后的数据计算，确保out-of-sample验证
    """
    symbol_cols = [col for col in residual_returns_df.columns if col != "date"]
    dates = sorted(residual_returns_df["date"].unique().to_list())

    results = []
    for i, date in enumerate(dates):
        # 跳过最后forward_window天（无法前向验证）
        if i >= len(dates) - forward_window:
            continue
        
        # 获取当前日期的预测风险
        current_risks = specific_risk_df.filter(pl.col("date") == date)
        if len(current_risks) == 0:
            continue
        
        # 确定使用哪个风险列
        risk_col = "specific_risk_shrunk" if "specific_risk_shrunk" in current_risks.columns else "specific_risk"
        
        ratios = []
        for symbol in symbol_cols:
            # 获取预测值
            predicted_risk_data = current_risks.filter(pl.col("symbol") == symbol)[risk_col].to_numpy()
            if len(predicted_risk_data) == 0 or predicted_risk_data[0] <= 1e-10:
                continue
            predicted_risk = predicted_risk_data[0]
            
            # 关键：使用date之后的forward_window天计算实际波动（前向验证）
            future_dates = dates[i+1 : i+1+forward_window]
            future_data = (
                residual_returns_df.filter(pl.col("date").is_in(future_dates))
                .select(["date", symbol])
                .drop_nulls()
                .sort("date")[symbol]
                .to_numpy()
            )
            
            # 需要有足够的前向数据
            if len(future_data) < forward_window * 0.8:  # 至少80%的数据点
                continue
            
            # 计算前向实际波动
            actual_vol = np.std(future_data)
            if annualize:
                # 根据实际窗口长度调整年化因子
                actual_vol *= np.sqrt(252 / len(future_data))
            
            ratio = actual_vol / predicted_risk
            ratios.append(ratio)
        
        if len(ratios) > 0:
            mrad = np.mean(ratios)
            std_ratio = np.std(ratios)
            
            results.append({
                "date": date,
                "mrad": mrad,
                "n_assets": len(ratios),
                "mean_ratio": mrad,
                "std_ratio": std_ratio,
            })

    return pl.DataFrame(results)


def calculate_bias_statistic(
    residual_returns_df: pl.DataFrame,
    specific_risk_df: pl.DataFrame,
    forward_window: int = 21,
    annualize: bool = True,
) -> pl.DataFrame:
    """计算 Bias 统计量 (Mean Bias Statistic) - 前向验证版本。

    平均预测偏差，衡量系统性高估或低估风险的程度。Bias≈1 表示无系统偏差。

    本函数使用前向验证方法：在日期t的预测，使用t之后的forward_window天数据验证，
    避免前视偏差（look-ahead bias）。

    参数
    ----------
    residual_returns_df: Polars DataFrame，残差收益
    specific_risk_df: Polars DataFrame，特异风险预测
    forward_window: 前向验证窗口长度（交易日），默认21（约1个月）
    annualize: 是否年化，默认True

    返回
    -------
    Polars DataFrame，包含各日期的 Bias 值
    列：| date | bias | n_assets | mean_predicted | mean_actual |

    注意
    ----
    - 最后forward_window天的数据无法验证，会被排除
    - 实际波动使用预测日期之后的数据计算，确保out-of-sample验证
    """
    symbol_cols = [col for col in residual_returns_df.columns if col != "date"]
    dates = sorted(residual_returns_df["date"].unique().to_list())

    results = []
    for i, date in enumerate(dates):
        # 跳过最后forward_window天（无法前向验证）
        if i >= len(dates) - forward_window:
            continue
        
        # 获取当前日期的预测风险
        current_risks = specific_risk_df.filter(pl.col("date") == date)
        if len(current_risks) == 0:
            continue
        
        risk_col = "specific_risk_shrunk" if "specific_risk_shrunk" in current_risks.columns else "specific_risk"
        
        predicted_risks = []
        actual_risks = []
        
        for symbol in symbol_cols:
            # 获取预测值
            predicted_risk_data = current_risks.filter(pl.col("symbol") == symbol)[risk_col].to_numpy()
            if len(predicted_risk_data) == 0 or predicted_risk_data[0] <= 1e-10:
                continue
            predicted_risk = predicted_risk_data[0]
            
            # 关键：使用date之后的forward_window天计算实际波动（前向验证）
            future_dates = dates[i+1 : i+1+forward_window]
            future_data = (
                residual_returns_df.filter(pl.col("date").is_in(future_dates))
                .select(["date", symbol])
                .drop_nulls()
                .sort("date")[symbol]
                .to_numpy()
            )
            
            # 需要有足够的前向数据
            if len(future_data) < forward_window * 0.8:  # 至少80%的数据点
                continue
            
            # 计算前向实际波动
            actual_vol = np.std(future_data)
            if annualize:
                # 根据实际窗口长度调整年化因子
                actual_vol *= np.sqrt(252 / len(future_data))
            
            predicted_risks.append(predicted_risk)
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
    forward_window: int = 21,
    annualize: bool = True,
) -> dict:
    """计算所有特异风险验证指标（使用前向验证方法）。

    参数
    ----------
    residual_returns_df: Polars DataFrame，残差收益
    specific_risk_df: Polars DataFrame，特异风险预测
    forward_window: 前向验证窗口长度（交易日），默认21
    annualize: 是否年化，默认True

    返回
    -------
    字典，包含所有验证结果：
    {
        'mrad': MRAD验证结果DataFrame,
        'bias': Bias验证结果DataFrame,
    }
    """
    return {
        "mrad": calculate_mrad(residual_returns_df, specific_risk_df, forward_window, annualize),
        "bias": calculate_bias_statistic(residual_returns_df, specific_risk_df, forward_window, annualize),
    }

