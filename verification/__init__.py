"""CNE5 模型有效性验证模块。

包含三类验证：
1. 因子层面验证：解释力与稳定性检验
2. 特异收益与特异风险验证：MRAD 与 Bias 指标体系
3. 协方差矩阵与系统风险验证：OBA 与 VRA 框架
"""

from verification.factor_validation import (
    calculate_factor_t_statistics,
    calculate_r_squared,
    calculate_factor_correlation,
    calculate_all_factor_validations,
)
from verification.specific_risk_validation import (
    calculate_mrad,
    calculate_bias_statistic,
    calculate_specific_risk_validations,
)
from verification.covariance_validation import (
    validate_oba_effectiveness,
    validate_vra_effectiveness,
    calculate_all_covariance_validations,
)
from verification.utils import (
    calculate_portfolio_risk,
    calculate_realized_volatility,
    aggregate_validation_results,
)

__all__ = [
    "calculate_factor_t_statistics",
    "calculate_r_squared",
    "calculate_factor_correlation",
    "calculate_all_factor_validations",
    "calculate_mrad",
    "calculate_bias_statistic",
    "calculate_specific_risk_validations",
    "validate_oba_effectiveness",
    "validate_vra_effectiveness",
    "calculate_all_covariance_validations",
    "calculate_portfolio_risk",
    "calculate_realized_volatility",
    "aggregate_validation_results",
]

