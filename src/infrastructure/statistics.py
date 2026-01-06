"""
Statistical functions for the Linear Regression Guide.

This module provides a unified interface to all statistical functions,
organized into specialized sub-modules for better maintainability.
"""

# Re-export all statistical functions from specialized modules
from .stats_modeling import (
    execute_ols_regression_modeling,
    execute_multiple_ols_regression_modeling,
    create_design_matrix
)

from .stats_calculation import (
    perform_regression_statistics_calculation,
    perform_simple_regression_stats_calculation,
    perform_multiple_regression_stats_calculation,
    perform_t_test,
    perform_confidence_interval_calculation,
    perform_basic_stats_calculation,
    get_data_ranges
)

from .stats_diagnostics import (
    perform_residual_diagnostics_calculation,
    perform_normality_tests,
    perform_heteroskedasticity_tests,
    perform_variance_inflation_factors_calculation,
    perform_sensitivity_analysis_calculation
)

from .stats_utils import (
    create_model_summary_dataframe,
    get_model_coefficients,
    get_model_summary_stats,
    get_model_diagnostics,
    format_statistical_value
)

# ============================================================================
# LEGACY FUNCTION ALIASES (for backward compatibility)
# ============================================================================
# These aliases map old function names to new implementations

# Modeling aliases
fit_ols_model = execute_ols_regression_modeling
fit_multiple_ols_model = execute_multiple_ols_regression_modeling

# Calculation aliases
compute_regression_statistics = perform_regression_statistics_calculation
compute_simple_regression_stats = perform_simple_regression_stats_calculation
compute_multiple_regression_stats = perform_multiple_regression_stats_calculation
calculate_basic_stats = perform_basic_stats_calculation

# Diagnostic aliases
compute_residual_diagnostics = perform_residual_diagnostics_calculation
calculate_variance_inflation_factors = perform_variance_inflation_factors_calculation
calculate_sensitivity_analysis = perform_sensitivity_analysis_calculation

# ============================================================================
# PUBLIC API
# ============================================================================
__all__ = [
    # Modeling (new names)
    'execute_ols_regression_modeling',
    'execute_multiple_ols_regression_modeling',
    'create_design_matrix',

    # Calculations (new names)
    'perform_regression_statistics_calculation',
    'perform_simple_regression_stats_calculation',
    'perform_multiple_regression_stats_calculation',
    'perform_t_test',
    'perform_confidence_interval_calculation',
    'perform_basic_stats_calculation',
    'get_data_ranges',

    # Diagnostics (new names)
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

    # =====================================================
    # LEGACY ALIASES (for backward compatibility)
    # =====================================================
    'fit_ols_model',
    'fit_multiple_ols_model',
    'compute_regression_statistics',
    'compute_simple_regression_stats',
    'compute_multiple_regression_stats',
    'calculate_basic_stats',
    'compute_residual_diagnostics',
    'calculate_variance_inflation_factors',
    'calculate_sensitivity_analysis',
]