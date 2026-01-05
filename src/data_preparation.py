"""
Data preparation and orchestration for regression models.

This module coordinates data generation for simple and multiple regression,
handling caching and parameter management.
"""

import streamlit as st
from typing import Dict, Any, Tuple
import numpy as np

from data import (
    generate_multiple_regression_data,
    generate_simple_regression_data,
    generate_electronics_market_data,
)
from statistics import (
    create_design_matrix,
    fit_ols_model,
    compute_simple_regression_stats,
    get_model_coefficients,
    get_model_summary_stats,
    get_model_diagnostics,
    calculate_basic_stats,
)
from session_state import (
    check_params_changed,
    update_cached_params,
    cache_model_data,
    get_cached_model_data,
    update_current_model,
    cache_simple_data_temp,
    get_simple_data_temp,
)
from logger import get_logger

logger = get_logger(__name__)


def prepare_multiple_regression_data(
    dataset_choice: str,
    n: int,
    noise_level: float,
    seed: int
) -> Dict[str, Any]:
    """
    Prepare data for multiple regression with caching.
    
    Args:
        dataset_choice: Name of the selected dataset
        n: Number of observations
        noise_level: Standard deviation of noise
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing prepared data and model results
    """
    logger.info(f"Preparing multiple regression data for {dataset_choice}")
    
    # Create parameter tuple for cache comparison
    mult_params = (dataset_choice, n, noise_level, seed)
    
    # Check if we need to regenerate data
    needs_regenerate = check_params_changed(mult_params, "last_mult_params")
    cached_data = get_cached_model_data("mult_model_cache")
    
    if needs_regenerate or cached_data is None:
        with st.spinner("🔄 Lade Datensatz für Multiple Regression..."):
            logger.debug("Generating new multiple regression data")
            
            # Generate raw data
            mult_data = generate_multiple_regression_data(
                dataset_choice, n, noise_level, seed
            )
            
            # Extract data components
            x2_preis = mult_data["x2_preis"]
            x3_werbung = mult_data["x3_werbung"]
            y_mult = mult_data["y_mult"]
            x1_name = mult_data["x1_name"]
            x2_name = mult_data["x2_name"]
            y_name = mult_data["y_name"]
            
            # Create design matrix and fit model
            X_mult = create_design_matrix(x2_preis, x3_werbung)
            model_mult, y_pred_mult = fit_ols_model(X_mult, y_mult)
            
            # Cache the processed data
            cached_data = {
                "x2_preis": x2_preis,
                "x3_werbung": x3_werbung,
                "y_mult": y_mult,
                "x1_name": x1_name,
                "x2_name": x2_name,
                "y_name": y_name,
                "X_mult": X_mult,
                "model_mult": model_mult,
                "y_pred_mult": y_pred_mult,
            }
            
            cache_model_data(cached_data, "mult_model_cache")
            update_cached_params(mult_params, "last_mult_params")
            
            # Store current model and feature names for R output display
            update_current_model(model_mult, ["hp", "drat", "wt"])
            
            logger.info("Multiple regression data prepared and cached")
    else:
        logger.debug("Using cached multiple regression data")
        # Update current model for R output (even when using cache)
        update_current_model(
            cached_data["model_mult"], 
            ["hp", "drat", "wt"]
        )
    
    return cached_data


def prepare_simple_regression_data(
    dataset_choice: str,
    x_variable: str = None,
    n: int = 12,
    true_intercept: float = 0.6,
    true_beta: float = 0.52,
    noise_level: float = 0.4,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Prepare data for simple regression with caching.
    
    Args:
        dataset_choice: Name of the selected dataset
        x_variable: Selected X variable (for non-simulated datasets)
        n: Number of observations
        true_intercept: True intercept value (for simulated data)
        true_beta: True slope value (for simulated data)
        noise_level: Standard deviation of noise (for simulated data)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing prepared data
    """
    logger.info(f"Preparing simple regression data for {dataset_choice}")
    
    # Handle electronics market simulation
    if dataset_choice == "🏪 Elektronikmarkt (simuliert)":
        simple_params = (dataset_choice, n, true_intercept, true_beta, noise_level, seed)
        
        if check_params_changed(simple_params, "last_simple_params"):
            with st.spinner("🔄 Generiere Daten..."):
                logger.debug("Generating new electronics market data")
                
                electronics_data = generate_electronics_market_data(
                    n, true_intercept, true_beta, noise_level, seed
                )
                
                cache_simple_data_temp(electronics_data)
                update_cached_params(simple_params, "last_simple_params")
                
                return electronics_data
        else:
            # Use cached data
            cached_data = get_simple_data_temp()
            if cached_data:
                return cached_data
            else:
                # Fallback: regenerate
                logger.warning("Cache miss, regenerating data")
                electronics_data = generate_electronics_market_data(
                    n, true_intercept, true_beta, noise_level, seed
                )
                cache_simple_data_temp(electronics_data)
                return electronics_data
    
    # Handle other datasets
    else:
        if x_variable:
            with st.spinner("🔄 Lade Datensatz..."):
                logger.debug(f"Loading dataset with x_variable={x_variable}")
                simple_data = generate_simple_regression_data(
                    dataset_choice, x_variable, n, seed=seed
                )
                return simple_data
        else:
            # Fallback: create minimal dataset
            logger.warning("No x_variable specified, creating fallback dataset")
            fallback_data = generate_electronics_market_data(12, 0.6, 0.52, 0.4, 42)
            return fallback_data


def compute_simple_model(
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    y_label: str
) -> Dict[str, Any]:
    """
    Compute simple regression model with caching.
    
    Args:
        x: Predictor variable
        y: Response variable
        x_label: Label for X variable
        y_label: Label for Y variable
    
    Returns:
        Dictionary containing model and statistics
    """
    logger.info("Computing simple regression model")
    
    # Build parameter tuple for cache validation
    basic_stats_x = calculate_basic_stats(x)
    basic_stats_y = calculate_basic_stats(y)
    
    simple_model_key = (
        basic_stats_x["count"],
        basic_stats_x["mean"],
        basic_stats_y["mean"],
    )
    
    # Check if we need to recompute the model
    cached_data = get_cached_model_data("simple_model_cache")
    needs_recompute = (
        cached_data is None
        or "model_key" not in cached_data
        or cached_data.get("model_key") != simple_model_key
    )
    
    if needs_recompute:
        with st.spinner("📊 Berechne Regressionsmodell..."):
            logger.debug("Computing new simple regression model")
            
            # Create design matrix and fit model
            X = create_design_matrix(x)
            model, y_pred = fit_ols_model(X, y)
            
            # Use centralized statistical computations
            n = len(x)
            stats_results = compute_simple_regression_stats(model, X, y, n)
            
            # Get additional centralized model information
            simple_coeffs = get_model_coefficients(model)
            simple_summary = get_model_summary_stats(model)
            simple_diagnostics = get_model_diagnostics(model)
            
            # Cache all computed values
            cached_data = {
                "model_key": simple_model_key,
                "X": X,
                "model": model,
                "y_pred": y_pred,
                "x": x,
                "y": y,
                **stats_results,  # Unpack all statistics
            }
            
            cache_model_data(cached_data, "simple_model_cache")
            
            # Store current model and feature names for R output display
            update_current_model(model, [x_label])
            
            logger.info("Simple regression model computed and cached")
    else:
        logger.debug("Using cached simple regression model")
        # Update current model for R output (even when using cache)
        update_current_model(cached_data["model"], [x_label])
    
    return cached_data
