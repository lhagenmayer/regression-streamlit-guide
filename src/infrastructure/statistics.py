"""
Statistical functions for the Linear Regression Guide.

This module provides a unified interface to all statistical functions,
organized into specialized sub-modules for better maintainability.

Sub-modules:
- stats_modeling: OLS regression model fitting
- stats_calculation: Core statistical calculations
- stats_diagnostics: Model diagnostic tests
- stats_utils: Utility functions for statistics
"""

# Modeling
from .stats_modeling import (
    execute_ols_regression_modeling,
    execute_multiple_ols_regression_modeling,
    create_design_matrix,
)

# Calculations
from .stats_calculation import (
    perform_regression_statistics_calculation,
    perform_simple_regression_stats_calculation,
    perform_multiple_regression_stats_calculation,
    perform_t_test,
    perform_confidence_interval_calculation,
    perform_basic_stats_calculation,
    get_data_ranges,
)

# Diagnostics
from .stats_diagnostics import (
    perform_residual_diagnostics_calculation,
    perform_normality_tests,
    perform_heteroskedasticity_tests,
    perform_variance_inflation_factors_calculation,
    perform_sensitivity_analysis_calculation,
)

# Utilities
from .stats_utils import (
    create_model_summary_dataframe,
    get_model_coefficients,
    get_model_summary_stats,
    get_model_diagnostics,
    format_statistical_value,
)

# Legacy aliases (short forms for convenience)
fit_ols_model = execute_ols_regression_modeling
fit_multiple_ols_model = execute_multiple_ols_regression_modeling
calculate_basic_stats = perform_basic_stats_calculation

__all__ = [
    # Modeling
    'execute_ols_regression_modeling',
    'execute_multiple_ols_regression_modeling',
    'create_design_matrix',
    'fit_ols_model',
    'fit_multiple_ols_model',
    # Calculations
    'perform_regression_statistics_calculation',
    'perform_simple_regression_stats_calculation',
    'perform_multiple_regression_stats_calculation',
    'perform_t_test',
    'perform_confidence_interval_calculation',
    'perform_basic_stats_calculation',
    'calculate_basic_stats',
    'get_data_ranges',
    # Diagnostics
    'perform_residual_diagnostics_calculation',
    'perform_normality_tests',
    'perform_heteroskedasticity_tests',
    'perform_variance_inflation_factors_calculation',
    'perform_sensitivity_analysis_calculation',
    # Utilities
    'create_model_summary_dataframe',
    'get_model_coefficients',
    'get_model_summary_stats',
    'get_model_diagnostics',
    'format_statistical_value',
]