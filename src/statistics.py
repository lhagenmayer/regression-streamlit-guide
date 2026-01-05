"""
Statistical computations and mathematical operations for the Linear Regression Guide.

This module centralizes all statistical calculations, model fitting, and mathematical
operations to provide a clean separation of concerns and better maintainability.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.graphics.gofplots import ProbPlot
from typing import Dict, Tuple, Any, Optional, Union, List
import streamlit as st

from .logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)


# ============================================================================
# MODEL FITTING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fit_ols_model(
    X: np.ndarray, y: np.ndarray
) -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, np.ndarray]:
    """
    Fit OLS regression model with caching.

    Args:
        X: Design matrix (with constant column)
        y: Response variable

    Returns:
        Tuple of (fitted model, predictions)
    """
    logger.debug(f"Fitting OLS model with X shape {X.shape}, y shape {y.shape}")
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)
    return model, predictions


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fit_multiple_ols_model(
    X: np.ndarray, y: np.ndarray
) -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, np.ndarray]:
    """
    Fit multiple OLS regression model with caching.

    Args:
        X: Design matrix (with constant column for multiple predictors)
        y: Response variable

    Returns:
        Tuple of (fitted model, predictions)
    """
    logger.debug(f"Fitting multiple OLS model with X shape {X.shape}, y shape {y.shape}")
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)
    return model, predictions


# ============================================================================
# STATISTICAL COMPUTATION FUNCTIONS
# ============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def compute_regression_statistics(
    y: np.ndarray, y_pred: np.ndarray, X: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive regression statistics with caching.

    Args:
        y: Actual values
        y_pred: Predicted values
        X: Design matrix

    Returns:
        Dictionary with regression statistics
    """
    y_mean = np.mean(y)
    sse = np.sum((y - y_pred) ** 2)
    sst = np.sum((y - y_mean) ** 2)
    ssr = np.sum((y_pred - y_mean) ** 2)
    r_squared = ssr / sst if sst != 0 else 0
    n, p = X.shape
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p)

    return {
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "sse": sse,
        "ssr": ssr,
        "sst": sst,
        "y_mean": y_mean
    }


@st.cache_data(ttl=300)  # Cache for 5 minutes
def compute_simple_regression_stats(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    X: np.ndarray,
    y: np.ndarray,
    n: int
) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for simple linear regression.

    Args:
        model: Fitted OLS model
        X: Design matrix
        y: Response variable
        n: Sample size

    Returns:
        Dictionary with all computed statistics
    """
    y_pred = model.predict(X)
    y_mean = np.mean(y)

    # Basic regression statistics
    b0, b1 = model.params[0], model.params[1]
    sse = np.sum((y - y_pred) ** 2)
    sst = np.sum((y - y_mean) ** 2)
    ssr = sst - sse
    mse = sse / (n - 2)
    msr = ssr / 1
    se_regression = np.sqrt(mse)
    sb1, sb0 = model.bse[1], model.bse[0]
    t_val = model.tvalues[1]
    f_val = model.fvalue
    df_resid = int(model.df_resid)

    # Descriptive statistics
    x_mean, y_mean_val = np.mean(X[:, 1]), np.mean(y)  # X[:, 1] excludes constant
    cov_xy = np.sum((X[:, 1] - x_mean) * (y - y_mean_val)) / (n - 1)
    var_x = np.var(X[:, 1], ddof=1)
    var_y = np.var(y, ddof=1)
    corr_xy = cov_xy / (np.sqrt(var_x) * np.sqrt(var_y))

    return {
        "model": model,
        "y_pred": y_pred,
        "y_mean": y_mean,
        "b0": b0,
        "b1": b1,
        "sse": sse,
        "sst": sst,
        "ssr": ssr,
        "mse": mse,
        "msr": msr,
        "se_regression": se_regression,
        "sb1": sb1,
        "sb0": sb0,
        "t_val": t_val,
        "f_val": f_val,
        "df_resid": df_resid,
        "x_mean": x_mean,
        "y_mean_val": y_mean_val,
        "cov_xy": cov_xy,
        "var_x": var_x,
        "var_y": var_y,
        "corr_xy": corr_xy,
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj
    }


@st.cache_data(ttl=300)  # Cache for 5 minutes
def compute_multiple_regression_stats(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    X: np.ndarray,
    y: np.ndarray
) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for multiple linear regression.

    Args:
        model: Fitted OLS model
        X: Design matrix
        y: Response variable

    Returns:
        Dictionary with all computed statistics
    """
    y_pred = model.predict(X)
    y_mean = np.mean(y)

    # Basic regression statistics
    sst = np.sum((y - y_mean) ** 2)
    sse = np.sum(model.resid ** 2)
    ssr = sst - sse

    return {
        "model": model,
        "y_pred": y_pred,
        "y_mean": y_mean,
        "sst": sst,
        "sse": sse,
        "ssr": ssr,
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "f_statistic": model.fvalue,
        "f_pvalue": model.f_pvalue
    }


# ============================================================================
# HYPOTHESIS TESTING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def perform_t_test(coefficient: float, std_error: float, df: int) -> Dict[str, Any]:
    """
    Perform t-test for coefficient significance.

    Args:
        coefficient: Coefficient value
        std_error: Standard error of coefficient
        df: Degrees of freedom

    Returns:
        Dictionary with t-test results
    """
    t_statistic = coefficient / std_error
    p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))

    return {
        "t_statistic": t_statistic,
        "p_value": p_value,
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
        "significant_0001": p_value < 0.001
    }


# ============================================================================
# DIAGNOSTIC FUNCTIONS
# ============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def compute_residual_diagnostics(residuals: np.ndarray) -> Dict[str, Any]:
    """
    Compute residual diagnostics for regression model validation.

    Args:
        residuals: Model residuals

    Returns:
        Dictionary with diagnostic statistics
    """
    # Basic statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals, ddof=1)

    # Normality tests
    shapiro_stat, shapiro_p = stats.shapiro(residuals) if len(residuals) >= 3 else (None, None)

    # Heteroskedasticity test (Breusch-Pagan approximation)
    n = len(residuals)
    if n > 10:
        # Simple approximation: check if variance changes with fitted values
        fitted_values = np.abs(residuals)  # Simple proxy
        bp_stat = np.sum((fitted_values - np.mean(fitted_values))**2) / np.var(fitted_values)
        bp_p = 1 - stats.chi2.cdf(bp_stat, 1)
    else:
        bp_stat, bp_p = None, None

    return {
        "mean_residual": mean_residual,
        "std_residual": std_residual,
        "shapiro_stat": shapiro_stat,
        "shapiro_p": shapiro_p,
        "normal_distribution": shapiro_p is None or shapiro_p > 0.05,
        "bp_stat": bp_stat,
        "bp_p": bp_p,
        "homoskedastic": bp_p is None or bp_p > 0.05
    }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_model_summary_dataframe(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    feature_names: Optional[list] = None
) -> pd.DataFrame:
    """
    Create a summary DataFrame from model results.

    Args:
        model: Fitted OLS model
        feature_names: Optional list of feature names

    Returns:
        DataFrame with model summary
    """
    if feature_names is None:
        if len(model.params) == 2:
            feature_names = ["Intercept", "X"]
        else:
            feature_names = ["Intercept"] + [f"X{i}" for i in range(1, len(model.params))]

    summary_data = {
        "Coefficient": feature_names,
        "Estimate": model.params,
        "Std. Error": model.bse,
        "t value": model.tvalues,
        "Pr(>|t|)": model.pvalues
    }

    return pd.DataFrame(summary_data)


def create_design_matrix(*columns: np.ndarray) -> np.ndarray:
    """
    Create a design matrix by stacking columns and adding constant term.

    Args:
        *columns: Variable columns to include in the design matrix

    Returns:
        Design matrix with constant column
    """
    if len(columns) == 1:
        # Simple regression: single column
        return sm.add_constant(columns[0])
    else:
        # Multiple regression: stack multiple columns
        X = np.column_stack(columns)
        return sm.add_constant(X)


def get_model_coefficients(model: sm.regression.linear_model.RegressionResultsWrapper) -> Dict[str, Any]:
    """
    Extract all model coefficients and related statistics.

    Args:
        model: Fitted OLS model

    Returns:
        Dictionary with all coefficient information
    """
    return {
        "params": model.params,
        "bse": model.bse,
        "tvalues": model.tvalues,
        "pvalues": model.pvalues,
        "conf_int": model.conf_int(),
    }


def get_model_summary_stats(model: sm.regression.linear_model.RegressionResultsWrapper) -> Dict[str, Any]:
    """
    Extract model summary statistics.

    Args:
        model: Fitted OLS model

    Returns:
        Dictionary with summary statistics
    """
    return {
        "rsquared": model.rsquared,
        "rsquared_adj": model.rsquared_adj,
        "fvalue": model.fvalue,
        "f_pvalue": model.f_pvalue,
        "df_resid": model.df_resid,
        "df_model": model.df_model,
        "mse_resid": model.mse_resid,
        "mse_model": model.mse_model,
        "mse_total": model.mse_total,
    }


def get_model_diagnostics(model: sm.regression.linear_model.RegressionResultsWrapper) -> Dict[str, Any]:
    """
    Extract model diagnostic information.

    Args:
        model: Fitted OLS model

    Returns:
        Dictionary with diagnostic information
    """
    return {
        "resid": model.resid,
        "resid_response": model.resid_response,
        "fittedvalues": model.fittedvalues,
        "mse_resid": model.mse_resid,
        "ssr": model.ssr,
        "ess": model.ess,
        "uncentered_tss": model.uncentered_tss,
        "centered_tss": model.centered_tss,
    }


def calculate_prediction_interval(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    X_new: np.ndarray,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate prediction intervals for new observations.

    Args:
        model: Fitted OLS model
        X_new: New design matrix values
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound) arrays
    """
    predictions = model.get_prediction(X_new)
    pred_summary = predictions.summary_frame(alpha=alpha)
    return pred_summary['obs_ci_lower'].values, pred_summary['obs_ci_upper'].values


def get_data_ranges(*arrays: np.ndarray) -> List[List[float]]:
    """
    Get min/max ranges for multiple data arrays.

    Args:
        *arrays: Data arrays to get ranges for

    Returns:
        List of [min, max] pairs for each array
    """
    return [[arr.min(), arr.max()] for arr in arrays]


def calculate_basic_stats(data: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
    """
    Calculate basic statistical summaries for data.

    Args:
        data: Input data array or series

    Returns:
        Dictionary with basic statistics
    """
    if isinstance(data, (np.ndarray, pd.Series)) and len(data) > 0:
        return {
            "mean": float(data.mean()),
            "std": float(data.std()),
            "var": float(data.var()),
            "min": float(data.min()),
            "max": float(data.max()),
            "count": len(data)
        }
    else:
        return {
            "mean": 0.0,
            "std": 0.0,
            "var": 0.0,
            "min": 0.0,
            "max": 0.0,
            "count": 0
        }


@st.cache_data(ttl=300)  # Cache for 5 minutes
def perform_normality_tests(residuals: np.ndarray) -> Dict[str, Any]:
    """
    Perform normality tests on residuals.

    Args:
        residuals: Model residuals

    Returns:
        Dictionary with normality test results
    """
    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = stats.shapiro(residuals) if len(residuals) >= 3 else (None, None)

    # Jarque-Bera test
    jb_stat, jb_pval, _, _ = jarque_bera(residuals)

    # Q-Q plot data
    qq = ProbPlot(residuals)
    qq_data = qq.theoretical_quantiles, qq.sample_quantiles

    return {
        "shapiro_stat": shapiro_stat,
        "shapiro_p": shapiro_p,
        "jb_stat": jb_stat,
        "jb_pval": jb_pval,
        "qq_data": qq_data,
        "normal_shapiro": shapiro_p is None or shapiro_p > 0.05,
        "normal_jb": jb_pval > 0.05
    }


@st.cache_data(ttl=300)  # Cache for 5 minutes
def perform_heteroskedasticity_tests(residuals: np.ndarray, X: np.ndarray) -> Dict[str, Any]:
    """
    Perform heteroskedasticity tests on residuals.

    Args:
        residuals: Model residuals
        X: Design matrix

    Returns:
        Dictionary with heteroskedasticity test results
    """
    # Breusch-Pagan test
    bp_stat, bp_pval, _, _ = het_breuschpagan(residuals, X)

    # White test (simplified approximation)
    # For White test, we need squared residuals regressed on all variables and cross terms
    # This is a simplified version
    white_X = np.column_stack([X, X**2])  # Add squared terms
    white_model = sm.OLS(residuals**2, white_X).fit()
    white_stat = white_model.fvalue
    white_p = white_model.f_pvalue

    return {
        "bp_stat": bp_stat,
        "bp_p": bp_pval,
        "white_stat": white_stat,
        "white_p": white_p,
        "homoskedastic_bp": bp_pval > 0.05,
        "homoskedastic_white": white_p > 0.05
    }


def format_statistical_value(value: float, precision: int = 4) -> str:
    """
    Format statistical values for display.

    Args:
        value: Numerical value
        precision: Decimal places

    Returns:
        Formatted string
    """
    if abs(value) < 0.001:
        return "0.0000"
    elif abs(value) < 0.01:
        return ".4f"
    else:
        return ".4f"


@st.cache_data(ttl=300)  # Cache for 5 minutes
def calculate_variance_inflation_factors(X: np.ndarray, variable_names: List[str] = None) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factors (VIF) for multicollinearity detection.

    Args:
        X: Design matrix (without constant column)
        variable_names: Optional list of variable names

    Returns:
        DataFrame with VIF values for each variable
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    if variable_names is None:
        variable_names = [f"Variable {i+1}" for i in range(X.shape[1])]

    vif_values = [variance_inflation_factor(X, i) for i in range(X.shape[1])]

    return pd.DataFrame({
        "Variable": variable_names,
        "VIF": vif_values
    })


@st.cache_data(ttl=300)  # Cache for 5 minutes
def calculate_sensitivity_analysis(
    model_params: np.ndarray,
    var1_range: np.ndarray,
    var2_fixed_value: float,
    var1_name: str = "Variable 1",
    var2_name: str = "Variable 2"
) -> Dict[str, np.ndarray]:
    """
    Calculate sensitivity analysis for multiple regression.

    Args:
        model_params: Model parameters [intercept, beta1, beta2]
        var1_range: Range of values for variable 1
        var2_fixed_value: Fixed value for variable 2
        var1_name: Name of variable 1
        var2_name: Name of variable 2

    Returns:
        Dictionary with sensitivity data
    """
    response = (
        model_params[0]
        + model_params[1] * var1_range
        + model_params[2] * var2_fixed_value
    )

    return {
        "var1_range": var1_range,
        "response": response,
        "var1_name": var1_name,
        "var2_name": var2_name,
        "var2_fixed_value": var2_fixed_value
    }