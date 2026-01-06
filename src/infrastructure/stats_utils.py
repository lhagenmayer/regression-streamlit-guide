"""
Statistical utility functions for the Linear Regression Guide.

This module contains helper functions for statistical operations.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, Any, List
from ..config import get_logger

logger = get_logger(__name__)


def create_model_summary_dataframe(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    feature_names: Optional[list] = None
) -> pd.DataFrame:
    """
    Create a summary DataFrame from model results.

    Args:
        model: Fitted statsmodels model
        feature_names: Optional feature names

    Returns:
        DataFrame with model summary
    """
    if feature_names is None:
        feature_names = [f"X{i}" for i in range(len(model.params))]

    summary_data = []
    for i, (param, std_err, t_val, p_val) in enumerate(zip(
        model.params, model.bse, model.tvalues, model.pvalues
    )):
        name = "Intercept" if i == 0 else feature_names[i-1]
        summary_data.append({
            "Parameter": name,
            "Coefficient": param,
            "Std_Error": std_err,
            "t_Statistic": t_val,
            "p_Value": p_val,
            "Significant": p_val < 0.05
        })

    return pd.DataFrame(summary_data)


def get_model_coefficients(model: sm.regression.linear_model.RegressionResultsWrapper) -> Dict[str, Any]:
    """
    Extract model coefficients and related statistics.

    Args:
        model: Fitted statsmodels model

    Returns:
        Dictionary with coefficient information
    """
    return {
        "coefficients": model.params.values,
        "std_errors": model.bse.values,
        "t_statistics": model.tvalues.values,
        "p_values": model.pvalues.values,
        "conf_int_lower": model.conf_int().iloc[:, 0].values,
        "conf_int_upper": model.conf_int().iloc[:, 1].values,
    }


def get_model_summary_stats(model: sm.regression.linear_model.RegressionResultsWrapper) -> Dict[str, Any]:
    """
    Extract model summary statistics.

    Args:
        model: Fitted statsmodels model

    Returns:
        Dictionary with model statistics
    """
    return {
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "f_statistic": model.fvalue,
        "f_p_value": model.f_pvalue,
        "aic": model.aic,
        "bic": model.bic,
        "n_obs": model.nobs,
        "df_model": model.df_model,
        "df_resid": model.df_resid,
    }


def get_model_diagnostics(model: sm.regression.linear_model.RegressionResultsWrapper) -> Dict[str, Any]:
    """
    Extract model diagnostic information.

    Args:
        model: Fitted statsmodels model

    Returns:
        Dictionary with diagnostic information
    """
    return {
        "residuals": model.resid.values,
        "fitted_values": model.fittedvalues.values,
        "mse": model.mse_resid,
        "rmse": np.sqrt(model.mse_resid),
        "mae": np.mean(np.abs(model.resid)),
        "condition_number": model.condition_number,
        "durbin_watson": sm.stats.durbin_watson(model.resid),
    }


def format_statistical_value(value: float, precision: int = 4) -> str:
    """
    Format statistical values for display.

    Args:
        value: Numeric value to format
        precision: Number of decimal places

    Returns:
        Formatted string
    """
    if abs(value) < 0.001 or abs(value) > 10000:
        return f"{value:.{precision}e}"
    else:
        return f"{value:.{precision}f}"