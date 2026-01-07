"""
Data loading module for the Linear Regression Guide.

This module uses NATIVE numpy calculations (no sklearn/statsmodels for core OLS)
so students can verify all results with a calculator.
"""

from typing import Dict, Any, Optional
import numpy as np

# Use our native OLS implementation for transparent, verifiable calculations
from ..infrastructure.native_ols import OLS, add_constant, OLSResult


def _map_dataset_name(display_name: str, regression_type: str) -> str:
    """Map UI display names to internal dataset names."""
    if regression_type == 'multiple':
        multiple_mappings = {
            "🏙️ Städte-Umsatzstudie (75 Städte)": "Cities",
            "🏠 Häuserpreise mit Pool (1000 Häuser)": "Houses",
            "🏪 Elektronikmarkt (simuliert)": "Electronics",
        }
        return multiple_mappings.get(display_name, "Cities")
    else:
        simple_mappings = {
            "🏙️ Advertising Study (75 Städte)": "advertising",
            "🏪 Elektronikmarkt (simuliert)": "electronics",
            "🏙️ Städte-Umsatzstudie (75 Städte)": "advertising",
        }
        return simple_mappings.get(display_name, "advertising")


def load_multiple_regression_data(
    dataset_choice: str,
    n: int,
    noise_level: float,
    seed: int
) -> Dict[str, Any]:
    """
    Load and prepare multiple regression data with NATIVE OLS calculations.
    
    All calculations use explicit formulas that can be verified with a calculator:
    - b = (X'X)^(-1) X'y
    - R² = 1 - SSE/SST
    - t = b / SE(b)
    """
    from .data_generators.multiple_regression_generator import generate_multiple_regression_data

    internal_name = _map_dataset_name(dataset_choice, 'multiple')
    
    try:
        raw_data = generate_multiple_regression_data(internal_name, n, noise_level, seed)
    except ValueError:
        raw_data = generate_multiple_regression_data("Cities", n, noise_level, seed)

    # Extract arrays
    x1 = np.array(raw_data.get("x2_preis", raw_data.get("x1", np.random.randn(n))))
    x2 = np.array(raw_data.get("x3_werbung", raw_data.get("x2", np.random.randn(n))))
    y = np.array(raw_data.get("y_mult", raw_data.get("y", np.random.randn(n))))
    
    # =========================================================
    # FIT NATIVE OLS - All formulas explicit for verification
    # =========================================================
    X = add_constant(np.column_stack([x1, x2]))  # Design matrix [1, x1, x2]
    model: OLSResult = OLS(y, X).fit()
    
    # Build coefficients info from model
    mult_coeffs = {
        "params": list(model.params),
        "bse": list(model.bse),
        "tvalues": list(model.tvalues),
        "pvalues": list(model.pvalues),
    }
    
    # Build summary
    mult_summary = {
        "rsquared": float(model.rsquared),
        "rsquared_adj": float(model.rsquared_adj),
        "fvalue": float(model.fvalue),
        "f_pvalue": float(model.f_pvalue),
    }
    
    # Build diagnostics
    mult_diagnostics = {
        "resid": model.resid,
        "sse": float(model.ssr),  # Note: our SSE is stored as ssr (statsmodels convention)
    }
    
    return {
        "x2_preis": x1,
        "x3_werbung": x2,
        "y_mult": y,
        "y_pred_mult": model.fittedvalues,
        "model_mult": model,
        "mult_coeffs": mult_coeffs,
        "mult_summary": mult_summary,
        "mult_diagnostics": mult_diagnostics,
        "x1_name": raw_data.get("x1_name", "Variable 1"),
        "x2_name": raw_data.get("x2_name", "Variable 2"),
        "y_name": raw_data.get("y_name", "Zielvariable"),
    }


def load_simple_regression_data(
    dataset_choice: str,
    x_variable: Optional[str],
    n: int,
    true_intercept: float = 0,
    true_beta: float = 0,
    noise_level: float = 0,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Load and prepare simple regression data with NATIVE OLS calculations.
    
    All calculations use explicit formulas that can be verified with a calculator:
    
    OLS Coefficients:
        b1 = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²  = Cov(x,y) / Var(x)
        b0 = ȳ - b1 * x̄
    
    Statistics:
        R² = 1 - SSE/SST = SSR/SST
        SE(b) = s / √(Σ(xi - x̄)²)  where s = √(SSE/(n-2))
        t = b / SE(b)
    """
    from .data_generators.simple_regression_generator import generate_simple_regression_data

    internal_name = _map_dataset_name(dataset_choice, 'simple')
    
    try:
        raw_data = generate_simple_regression_data(internal_name, n, noise_level, seed)
    except Exception:
        # Fallback: generate synthetic data
        np.random.seed(seed)
        x = np.random.uniform(2, 10, n)
        y = true_intercept + true_beta * x + np.random.normal(0, noise_level, n)
        raw_data = {"x": x, "y": y, "x_label": "X", "y_label": "Y"}

    # Extract arrays
    x = np.array(raw_data.get("x", np.random.randn(n)))
    y = np.array(raw_data.get("y", np.random.randn(n)))
    n_obs = len(x)
    
    # =========================================================
    # MANUAL CALCULATIONS (for educational transparency)
    # Students can verify each step with a calculator
    # =========================================================
    
    # Step 1: Basic statistics
    x_mean = float(np.mean(x))
    y_mean_val = float(np.mean(y))
    
    # Step 2: Deviations from mean
    x_dev = x - x_mean  # (xi - x̄)
    y_dev = y - y_mean_val  # (yi - ȳ)
    
    # Step 3: Covariance and Variance (using sample formulas with n-1)
    cov_xy = float(np.sum(x_dev * y_dev) / (n_obs - 1))  # Cov(x,y)
    var_x = float(np.sum(x_dev ** 2) / (n_obs - 1))  # Var(x)
    var_y = float(np.sum(y_dev ** 2) / (n_obs - 1))  # Var(y)
    
    # Step 4: Correlation
    corr_xy = cov_xy / (np.sqrt(var_x) * np.sqrt(var_y)) if var_x > 0 and var_y > 0 else 0
    
    # =========================================================
    # FIT NATIVE OLS
    # =========================================================
    X = add_constant(x)
    model: OLSResult = OLS(y, X).fit()
    
    b0 = float(model.params[0])  # Intercept
    b1 = float(model.params[1])  # Slope
    y_pred = model.fittedvalues
    residuals = model.resid
    
    # Sum of squares (from model)
    sse = float(model.ssr)  # SSE = Σ(yi - ŷi)²
    sst = float(model.centered_tss)  # SST = Σ(yi - ȳ)²
    ssr = float(model.ess)  # SSR = Σ(ŷi - ȳ)²
    mse = float(model.mse_resid)  # MSE = SSE/(n-2)
    se_regression = float(np.sqrt(mse)) if mse > 0 else 0
    
    return {
        "x": x,
        "y": y,
        "y_pred": y_pred,
        "residuals": residuals,
        "model": model,
        "b0": b0,
        "b1": b1,
        "x_label": raw_data.get("x_label", "X"),
        "y_label": raw_data.get("y_label", "Y"),
        "x_unit": raw_data.get("x_unit", ""),
        "y_unit": raw_data.get("y_unit", ""),
        "context_title": raw_data.get("context_title", dataset_choice),
        "context_description": raw_data.get("context_description", ""),
        # Statistics for educational display
        "n": n_obs,
        "x_mean": x_mean,
        "y_mean_val": y_mean_val,
        "cov_xy": cov_xy,
        "var_x": var_x,
        "var_y": var_y,
        "corr_xy": corr_xy,
        "sse": sse,
        "sst": sst,
        "ssr": ssr,
        "mse": mse,
        "se_regression": se_regression,
    }


def compute_simple_regression_model(
    x, y, x_label: str, y_label: str, n: int
) -> Dict[str, Any]:
    """
    Compute simple regression model using native OLS.
    """
    x = np.array(x)
    y = np.array(y)
    
    X = add_constant(x)
    model = OLS(y, X).fit()
    
    return {
        'model': model,
        'x': x,
        'y': y,
        'x_label': x_label,
        'y_label': y_label,
        'y_pred': model.fittedvalues,
        'b0': float(model.params[0]),
        'b1': float(model.params[1]),
        'residuals': model.resid,
    }
