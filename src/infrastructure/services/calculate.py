"""
Step 2: CALCULATE

This module handles all statistical calculations.
It computes regression models and statistics from data.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np

from ...config import get_logger
from ..data.generators import DataResult, MultipleRegressionDataResult

logger = get_logger(__name__)


@dataclass
class RegressionResult:
    """Result from regression calculation."""
    # Coefficients
    intercept: float
    slope: float
    
    # Predictions
    y_pred: np.ndarray
    residuals: np.ndarray
    
    # Model fit statistics
    r_squared: float
    r_squared_adj: float
    
    # Standard errors
    se_intercept: float
    se_slope: float
    
    # Test statistics
    t_intercept: float
    t_slope: float
    p_intercept: float
    p_slope: float
    
    # Sum of squares
    sse: float  # Sum of Squared Errors
    sst: float  # Total Sum of Squares
    ssr: float  # Regression Sum of Squares
    mse: float  # Mean Squared Error
    
    # Sample info
    n: int
    df: int  # Degrees of freedom
    
    # Extra stats
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultipleRegressionResult:
    """Result from multiple regression calculation."""
    # Coefficients
    intercept: float
    coefficients: List[float]
    
    # Predictions
    y_pred: np.ndarray
    residuals: np.ndarray
    
    # Model fit
    r_squared: float
    r_squared_adj: float
    f_statistic: float
    f_pvalue: float
    
    # Standard errors and tests for each coefficient
    se_coefficients: List[float]
    t_values: List[float]
    p_values: List[float]
    
    # Sum of squares
    sse: float
    sst: float
    ssr: float
    
    # Sample info
    n: int
    k: int  # Number of predictors
    
    extra: Dict[str, Any] = field(default_factory=dict)


class StatisticsCalculator:
    """
    Step 2: CALCULATE
    
    Computes regression models and statistics.
    Uses transparent, verifiable formulas (no black-box libraries).
    
    Example:
        calc = StatisticsCalculator()
        result = calc.simple_regression(data.x, data.y)
    """
    
    def __init__(self):
        logger.info("StatisticsCalculator initialized")
    
    def simple_regression(
        self, 
        x: np.ndarray, 
        y: np.ndarray
    ) -> RegressionResult:
        """
        Compute simple linear regression: ŷ = b₀ + b₁x
        
        All formulas are explicit for educational transparency:
        - b₁ = Cov(x,y) / Var(x) = Σ(xᵢ-x̄)(yᵢ-ȳ) / Σ(xᵢ-x̄)²
        - b₀ = ȳ - b₁x̄
        - R² = 1 - SSE/SST
        
        Args:
            x: Predictor variable
            y: Response variable
        
        Returns:
            RegressionResult with all statistics
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(x)
        
        logger.info(f"Computing simple regression: n={n}")
        
        # =========================================================
        # STEP 1: Basic Statistics
        # =========================================================
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        # Deviations from mean
        x_dev = x - x_mean
        y_dev = y - y_mean
        
        # =========================================================
        # STEP 2: OLS Coefficients
        # b₁ = Σ(xᵢ-x̄)(yᵢ-ȳ) / Σ(xᵢ-x̄)²
        # b₀ = ȳ - b₁x̄
        # =========================================================
        ss_xx = np.sum(x_dev ** 2)  # Σ(xᵢ-x̄)²
        ss_xy = np.sum(x_dev * y_dev)  # Σ(xᵢ-x̄)(yᵢ-ȳ)
        
        b1 = ss_xy / ss_xx if ss_xx > 0 else 0  # Slope
        b0 = y_mean - b1 * x_mean  # Intercept
        
        # =========================================================
        # STEP 3: Predictions & Residuals
        # =========================================================
        y_pred = b0 + b1 * x
        residuals = y - y_pred
        
        # =========================================================
        # STEP 4: Sum of Squares
        # =========================================================
        sse = np.sum(residuals ** 2)  # SSE = Σ(yᵢ - ŷᵢ)²
        sst = np.sum(y_dev ** 2)  # SST = Σ(yᵢ - ȳ)²
        ssr = sst - sse  # SSR = SST - SSE
        
        # =========================================================
        # STEP 5: R² and Adjusted R²
        # =========================================================
        r_squared = 1 - (sse / sst) if sst > 0 else 0
        r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - 2) if n > 2 else 0
        
        # =========================================================
        # STEP 6: Standard Errors
        # =========================================================
        df = n - 2  # Degrees of freedom
        mse = sse / df if df > 0 else 0
        se_regression = np.sqrt(mse)
        
        # SE(b₁) = s / √(Σ(xᵢ-x̄)²)
        se_b1 = se_regression / np.sqrt(ss_xx) if ss_xx > 0 else 0
        
        # SE(b₀) = s * √(1/n + x̄²/Σ(xᵢ-x̄)²)
        se_b0 = se_regression * np.sqrt(1/n + x_mean**2 / ss_xx) if ss_xx > 0 else 0
        
        # =========================================================
        # STEP 7: t-Statistics and p-Values
        # =========================================================
        t_b0 = b0 / se_b0 if se_b0 > 0 else 0
        t_b1 = b1 / se_b1 if se_b1 > 0 else 0
        
        # p-values using t-distribution
        from scipy import stats
        p_b0 = 2 * (1 - stats.t.cdf(abs(t_b0), df)) if df > 0 else 1
        p_b1 = 2 * (1 - stats.t.cdf(abs(t_b1), df)) if df > 0 else 1
        
        logger.info(f"Regression complete: R²={r_squared:.4f}, b₀={b0:.4f}, b₁={b1:.4f}")
        
        return RegressionResult(
            intercept=b0,
            slope=b1,
            y_pred=y_pred,
            residuals=residuals,
            r_squared=r_squared,
            r_squared_adj=r_squared_adj,
            se_intercept=se_b0,
            se_slope=se_b1,
            t_intercept=t_b0,
            t_slope=t_b1,
            p_intercept=p_b0,
            p_slope=p_b1,
            sse=sse,
            sst=sst,
            ssr=ssr,
            mse=mse,
            n=n,
            df=df,
            extra={
                "x_mean": x_mean,
                "y_mean": y_mean,
                "se_regression": se_regression,
                "correlation": np.sqrt(r_squared) * (1 if b1 >= 0 else -1),
            }
        )
    
    def multiple_regression(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        y: np.ndarray
    ) -> MultipleRegressionResult:
        """
        Compute multiple regression: ŷ = b₀ + b₁x₁ + b₂x₂
        
        Uses matrix formula: b = (X'X)⁻¹X'y
        
        Args:
            x1: First predictor
            x2: Second predictor
            y: Response variable
        
        Returns:
            MultipleRegressionResult with all statistics
        """
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)
        k = 2  # Number of predictors
        
        logger.info(f"Computing multiple regression: n={n}, k={k}")
        
        # =========================================================
        # STEP 1: Build Design Matrix
        # X = [1, x₁, x₂]
        # =========================================================
        X = np.column_stack([np.ones(n), x1, x2])
        
        # =========================================================
        # STEP 2: OLS via Matrix Formula
        # b = (X'X)⁻¹X'y
        # =========================================================
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX)
        Xty = X.T @ y
        b = XtX_inv @ Xty
        
        b0, b1, b2 = b[0], b[1], b[2]
        
        # =========================================================
        # STEP 3: Predictions & Residuals
        # =========================================================
        y_pred = X @ b
        residuals = y - y_pred
        y_mean = np.mean(y)
        
        # =========================================================
        # STEP 4: Sum of Squares
        # =========================================================
        sse = np.sum(residuals ** 2)
        sst = np.sum((y - y_mean) ** 2)
        ssr = sst - sse
        
        # =========================================================
        # STEP 5: R² and Adjusted R²
        # =========================================================
        r_squared = 1 - (sse / sst) if sst > 0 else 0
        r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - k - 1) if n > k + 1 else 0
        
        # =========================================================
        # STEP 6: Standard Errors
        # Var(b) = σ²(X'X)⁻¹
        # =========================================================
        df = n - k - 1
        mse = sse / df if df > 0 else 0
        var_b = mse * XtX_inv
        se_b = np.sqrt(np.diag(var_b))
        
        # =========================================================
        # STEP 7: t-Statistics and p-Values
        # =========================================================
        t_values = b / se_b
        
        from scipy import stats
        p_values = [2 * (1 - stats.t.cdf(abs(t), df)) if df > 0 else 1 for t in t_values]
        
        # =========================================================
        # STEP 8: F-Statistic
        # F = (SSR/k) / (SSE/(n-k-1))
        # =========================================================
        msr = ssr / k if k > 0 else 0
        f_stat = msr / mse if mse > 0 else 0
        f_pvalue = 1 - stats.f.cdf(f_stat, k, df) if df > 0 else 1
        
        logger.info(f"Multiple regression complete: R²={r_squared:.4f}")
        
        return MultipleRegressionResult(
            intercept=b0,
            coefficients=[b1, b2],
            y_pred=y_pred,
            residuals=residuals,
            r_squared=r_squared,
            r_squared_adj=r_squared_adj,
            f_statistic=f_stat,
            f_pvalue=f_pvalue,
            se_coefficients=list(se_b),
            t_values=list(t_values),
            p_values=p_values,
            sse=sse,
            sst=sst,
            ssr=ssr,
            n=n,
            k=k,
            extra={"y_mean": y_mean, "mse": mse}
        )
    
    def basic_stats(self, data: np.ndarray) -> Dict[str, float]:
        """
        Compute basic descriptive statistics.
        
        Args:
            data: Numeric array
        
        Returns:
            Dictionary with mean, std, min, max, etc.
        """
        data = np.asarray(data, dtype=float)
        return {
            "n": len(data),
            "mean": float(np.mean(data)),
            "std": float(np.std(data, ddof=1)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "median": float(np.median(data)),
            "q25": float(np.percentile(data, 25)),
            "q75": float(np.percentile(data, 75)),
        }
