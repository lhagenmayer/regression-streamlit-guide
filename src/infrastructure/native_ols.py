"""
Native OLS implementation for the Linear Regression Guide.

This module provides a pure numpy implementation of Ordinary Least Squares
regression, without relying on sklearn or statsmodels for core calculations.
This is educational - showing the math behind regression.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class OLSResult:
    """
    Result of OLS regression - mimics statsmodels RegressionResultsWrapper.
    
    This is a native implementation showing all formulas explicitly.
    """
    # Coefficients
    params: np.ndarray  # [b0, b1, ...] - intercept and slopes
    
    # Standard errors
    bse: np.ndarray  # Standard errors of coefficients
    
    # T-statistics and p-values
    tvalues: np.ndarray
    pvalues: np.ndarray
    
    # Confidence intervals
    conf_int_lower: np.ndarray
    conf_int_upper: np.ndarray
    
    # Model statistics
    rsquared: float
    rsquared_adj: float
    fvalue: float
    f_pvalue: float
    
    # Residuals and predictions
    resid: np.ndarray
    fittedvalues: np.ndarray
    
    # Data
    nobs: int
    df_model: int  # k (number of predictors)
    df_resid: int  # n - k - 1
    
    # Sum of squares
    ess: float  # Explained (SSR)
    ssr: float  # Residual (SSE) - note: statsmodels calls SSE "ssr"
    centered_tss: float  # Total (SST)
    
    # Standard error of regression
    mse_resid: float  # MSE = SSE / (n-k-1)
    mse_model: float  # MSR = SSR / k
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fitted model."""
        if X.ndim == 1:
            X = np.column_stack([np.ones(len(X)), X])
        return X @ self.params
    
    def summary(self) -> 'OLSSummary':
        """Return summary object for display."""
        return OLSSummary(self)


class OLSSummary:
    """Summary display for OLS results."""
    
    def __init__(self, result: OLSResult):
        self.result = result
    
    def as_text(self) -> str:
        """Generate R-style text summary."""
        r = self.result
        lines = [
            "=" * 60,
            "                    OLS Regression Results",
            "=" * 60,
            f"Observations:           {r.nobs:10d}",
            f"R-squared:              {r.rsquared:10.4f}",
            f"Adj. R-squared:         {r.rsquared_adj:10.4f}",
            f"F-statistic:            {r.fvalue:10.4f}",
            f"Prob (F-statistic):     {r.f_pvalue:10.4g}",
            "-" * 60,
            f"{'Coef':<12} {'Estimate':>12} {'Std.Err':>10} {'t':>8} {'P>|t|':>10}",
            "-" * 60,
        ]
        
        names = ['const'] + [f'x{i}' for i in range(1, len(r.params))]
        for i, name in enumerate(names):
            lines.append(
                f"{name:<12} {r.params[i]:>12.4f} {r.bse[i]:>10.4f} "
                f"{r.tvalues[i]:>8.3f} {r.pvalues[i]:>10.4g}"
            )
        
        lines.append("=" * 60)
        return "\n".join(lines)


def fit_ols(X: np.ndarray, y: np.ndarray) -> OLSResult:
    """
    Fit OLS regression using pure numpy - showing all formulas.
    
    This implements OLS from first principles:
    b = (X'X)^(-1) X'y
    
    Args:
        X: Design matrix with intercept column (n x (k+1))
        y: Response variable (n,)
    
    Returns:
        OLSResult with all statistics computed natively
    """
    n = len(y)
    k = X.shape[1] - 1  # Number of predictors (excluding intercept)
    
    # =========================================================
    # STEP 1: Compute OLS coefficients
    # b = (X'X)^(-1) X'y
    # =========================================================
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    Xty = X.T @ y
    params = XtX_inv @ Xty  # b = [b0, b1, b2, ...]
    
    # =========================================================
    # STEP 2: Compute predictions and residuals
    # =========================================================
    y_pred = X @ params
    residuals = y - y_pred
    
    # =========================================================
    # STEP 3: Sum of Squares
    # SST = Σ(yi - ȳ)²  [Total]
    # SSE = Σ(yi - ŷi)² [Error/Residual]
    # SSR = Σ(ŷi - ȳ)² [Regression/Explained]
    # =========================================================
    y_mean = np.mean(y)
    SST = np.sum((y - y_mean) ** 2)
    SSE = np.sum(residuals ** 2)
    SSR = np.sum((y_pred - y_mean) ** 2)
    
    # =========================================================
    # STEP 4: Mean Square Errors
    # MSE = SSE / (n - k - 1)  [Error variance estimate]
    # MSR = SSR / k            [Regression mean square]
    # =========================================================
    df_resid = n - k - 1
    df_model = k
    MSE = SSE / df_resid if df_resid > 0 else 0
    MSR = SSR / k if k > 0 else 0
    
    # =========================================================
    # STEP 5: R-squared and Adjusted R-squared
    # R² = 1 - SSE/SST = SSR/SST
    # R²_adj = 1 - (1-R²)(n-1)/(n-k-1)
    # =========================================================
    rsquared = 1 - SSE / SST if SST > 0 else 0
    rsquared_adj = 1 - (1 - rsquared) * (n - 1) / df_resid if df_resid > 0 else 0
    
    # =========================================================
    # STEP 6: Standard Errors of Coefficients
    # SE(bi) = sqrt(MSE * (X'X)^(-1)_ii)
    # =========================================================
    s2 = MSE  # Variance estimate
    var_b = s2 * XtX_inv
    bse = np.sqrt(np.diag(var_b))
    
    # =========================================================
    # STEP 7: T-statistics and P-values
    # t = b / SE(b)
    # p = 2 * P(T > |t|) with df = n - k - 1
    # =========================================================
    tvalues = params / bse
    pvalues = 2 * (1 - stats.t.cdf(np.abs(tvalues), df_resid))
    
    # =========================================================
    # STEP 8: Confidence Intervals (95%)
    # CI = b ± t_{α/2, df} * SE(b)
    # =========================================================
    t_crit = stats.t.ppf(0.975, df_resid)
    conf_int_lower = params - t_crit * bse
    conf_int_upper = params + t_crit * bse
    
    # =========================================================
    # STEP 9: F-statistic
    # F = MSR / MSE
    # =========================================================
    fvalue = MSR / MSE if MSE > 0 else 0
    f_pvalue = 1 - stats.f.cdf(fvalue, df_model, df_resid) if df_resid > 0 else 1
    
    return OLSResult(
        params=params,
        bse=bse,
        tvalues=tvalues,
        pvalues=pvalues,
        conf_int_lower=conf_int_lower,
        conf_int_upper=conf_int_upper,
        rsquared=rsquared,
        rsquared_adj=rsquared_adj,
        fvalue=fvalue,
        f_pvalue=f_pvalue,
        resid=residuals,
        fittedvalues=y_pred,
        nobs=n,
        df_model=df_model,
        df_resid=df_resid,
        ess=SSR,
        ssr=SSE,  # Note: statsmodels calls SSE "ssr"
        centered_tss=SST,
        mse_resid=MSE,
        mse_model=MSR,
    )


def add_constant(x: np.ndarray) -> np.ndarray:
    """
    Add intercept column to feature matrix.
    
    Args:
        x: Feature array (n,) or (n, k)
    
    Returns:
        Design matrix with ones column prepended (n, k+1)
    """
    if x.ndim == 1:
        return np.column_stack([np.ones(len(x)), x])
    else:
        return np.column_stack([np.ones(x.shape[0]), x])


# Convenience wrapper matching statsmodels API
class OLS:
    """
    OLS class matching statsmodels API for drop-in replacement.
    
    Usage:
        model = OLS(y, X).fit()
        print(model.rsquared)
    """
    
    def __init__(self, y: np.ndarray, X: np.ndarray):
        self.y = np.asarray(y)
        self.X = np.asarray(X)
    
    def fit(self) -> OLSResult:
        """Fit the OLS model."""
        return fit_ols(self.X, self.y)
