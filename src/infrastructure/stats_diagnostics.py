"""
Statistical diagnostics functions for the Linear Regression Guide.

This module contains functions for model diagnostics and validation tests.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.graphics.gofplots import ProbPlot
from typing import Dict, Any, List, Optional
from ..config import get_logger

logger = get_logger(__name__)


def perform_residual_diagnostics_calculation(residuals: np.ndarray) -> Dict[str, Any]:
    """
    Perform residual diagnostics for regression model validation.

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


def perform_normality_tests(residuals: np.ndarray) -> Dict[str, Any]:
    """
    Perform normality tests on residuals.

    Args:
        residuals: Model residuals

    Returns:
        Dictionary with normality test results
    """
    results = {}

    if len(residuals) >= 3:
        # Shapiro-Wilk test
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        results["shapiro_wilk"] = {
            "statistic": shapiro_stat,
            "p_value": shapiro_p,
            "normal": shapiro_p > 0.05
        }

    if len(residuals) >= 8:
        # Jarque-Bera test
        jb_stat, jb_p = jarque_bera(residuals)
        results["jarque_bera"] = {
            "statistic": jb_stat,
            "p_value": jb_p,
            "normal": jb_p > 0.05
        }

    # Kolmogorov-Smirnov test against normal distribution
    # Standardize residuals first
    if len(residuals) > 1:
        std_residuals = (residuals - np.mean(residuals)) / np.std(residuals, ddof=1)
        ks_stat, ks_p = stats.kstest(std_residuals, 'norm')
        results["kolmogorov_smirnov"] = {
            "statistic": ks_stat,
            "p_value": ks_p,
            "normal": ks_p > 0.05
        }

    return results


def perform_heteroskedasticity_tests(residuals: np.ndarray, X: np.ndarray) -> Dict[str, Any]:
    """
    Perform heteroskedasticity tests.

    Args:
        residuals: Model residuals
        X: Design matrix

    Returns:
        Dictionary with heteroskedasticity test results
    """
    results = {}

    if X.shape[1] > 1 and len(residuals) > 10:
        # Breusch-Pagan test
        try:
            bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X)
            results["breusch_pagan"] = {
                "statistic": bp_stat,
                "p_value": bp_p,
                "homoskedastic": bp_p > 0.05
            }
        except Exception as e:
            logger.warning(f"Breusch-Pagan test failed: {e}")

    # White test (simplified version)
    if X.shape[1] > 1:
        # Simple approximation using squared residuals
        white_stat = np.sum(residuals**4) / (np.sum(residuals**2)**2 / len(residuals))
        white_p = 1 - stats.chi2.cdf(white_stat, X.shape[1])
        results["white_test"] = {
            "statistic": white_stat,
            "p_value": white_p,
            "homoskedastic": white_p > 0.05
        }

    return results


def perform_variance_inflation_factors_calculation(X: np.ndarray, variable_names: List[str] = None) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factors (VIF) for multicollinearity detection.

    Args:
        X: Design matrix (without intercept)
        variable_names: Names of variables

    Returns:
        DataFrame with VIF values
    """
    if variable_names is None:
        variable_names = [f"X{i}" for i in range(X.shape[1])]

    vif_data = []

    for i, name in enumerate(variable_names):
        # Regress X_i on all other X variables
        X_others = np.delete(X, i, axis=1)
        y_i = X[:, i]

        if X_others.shape[1] > 0:
            # Add intercept
            X_others = np.column_stack([np.ones(X_others.shape[0]), X_others])
            model = sm.OLS(y_i, X_others).fit()
            r_squared = model.rsquared
            vif = 1 / (1 - r_squared) if r_squared < 1 else float('inf')
        else:
            vif = 1.0  # No multicollinearity possible with single variable

        vif_data.append({
            "Variable": name,
            "VIF": vif,
            "Multicollinearity": "High" if vif > 10 else "Moderate" if vif > 5 else "Low"
        })

    return pd.DataFrame(vif_data)


def perform_sensitivity_analysis_calculation(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    n_bootstraps: int = 100
) -> Dict[str, Any]:
    """
    Perform sensitivity analysis using bootstrapping.

    Args:
        X: Design matrix
        y: Response variable
        coefficients: Original coefficients
        n_bootstraps: Number of bootstrap samples

    Returns:
        Dictionary with sensitivity analysis results
    """
    n = len(y)
    bootstrap_coeffs = []

    for _ in range(n_bootstraps):
        # Bootstrap sample
        indices = np.random.choice(n, n, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]

        # Fit model (simplified)
        try:
            beta_boot = np.linalg.inv(X_boot.T @ X_boot) @ X_boot.T @ y_boot
            bootstrap_coeffs.append(beta_boot)
        except:
            continue

    if not bootstrap_coeffs:
        return {"error": "Bootstrap failed"}

    bootstrap_coeffs = np.array(bootstrap_coeffs)

    # Calculate statistics
    coeff_means = np.mean(bootstrap_coeffs, axis=0)
    coeff_stds = np.std(bootstrap_coeffs, axis=0)
    coeff_ci_lower = np.percentile(bootstrap_coeffs, 2.5, axis=0)
    coeff_ci_upper = np.percentile(bootstrap_coeffs, 97.5, axis=0)

    return {
        "bootstrap_coefficients": bootstrap_coeffs,
        "coefficient_means": coeff_means,
        "coefficient_stds": coeff_stds,
        "confidence_intervals_lower": coeff_ci_lower,
        "confidence_intervals_upper": coeff_ci_upper,
        "n_bootstraps": len(bootstrap_coeffs)
    }