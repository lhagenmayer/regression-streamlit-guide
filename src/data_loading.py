"""
Data loading and preparation module for the Linear Regression Guide.

This module handles all data loading, validation, caching, and model fitting
for both simple and multiple regression analyses.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from data import (
    generate_multiple_regression_data,
    generate_simple_regression_data,
    generate_electronics_market_data,
)
from statistics import (
    create_design_matrix,
    fit_multiple_ols_model,
    fit_ols_model,
    get_model_coefficients,
    get_model_summary_stats,
    get_model_diagnostics,
    compute_simple_regression_stats,
    calculate_basic_stats,
)
from logger import get_logger

logger = get_logger(__name__)


def load_multiple_regression_data(
    dataset_choice: str,
    n: int,
    noise_level: float,
    seed: int
) -> Dict[str, Any]:
    """
    Load and prepare multiple regression data with caching.
    
    Args:
        dataset_choice: Name of the dataset to load
        n: Number of observations
        noise_level: Noise level for data generation
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing all prepared data and model results
    
    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If data loading or model fitting fails
    """
    # Validate parameters
    _validate_multiple_regression_params(n, noise_level, seed)
    
    # Create parameter tuple for cache comparison
    params = (dataset_choice, n, noise_level, seed)
    
    # Check if we need to regenerate data
    if st.session_state.last_mult_params != params or st.session_state.mult_model_cache is None:
        try:
            with st.spinner("🔄 Lade Datensatz für Multiple Regression..."):
                # Generate data
                mult_data = generate_multiple_regression_data(
                    dataset_choice, n, noise_level, seed
                )
                
                # Extract variables
                x2_preis = mult_data["x2_preis"]
                x3_werbung = mult_data["x3_werbung"]
                y_mult = mult_data["y_mult"]
                
                # Create design matrix and fit model
                X_mult = create_design_matrix(x2_preis, x3_werbung)
                model_mult, y_pred_mult = fit_multiple_ols_model(X_mult, y_mult)
                
                # Get model statistics
                mult_coeffs = get_model_coefficients(model_mult)
                mult_summary = get_model_summary_stats(model_mult)
                mult_diagnostics = get_model_diagnostics(model_mult)
                
                # Cache all results
                cache_data = {
                    "x2_preis": x2_preis,
                    "x3_werbung": x3_werbung,
                    "y_mult": y_mult,
                    "x1_name": mult_data["x1_name"],
                    "x2_name": mult_data["x2_name"],
                    "y_name": mult_data["y_name"],
                    "X_mult": X_mult,
                    "model_mult": model_mult,
                    "y_pred_mult": y_pred_mult,
                    "mult_coeffs": mult_coeffs,
                    "mult_summary": mult_summary,
                    "mult_diagnostics": mult_diagnostics,
                }
                
                st.session_state.mult_model_cache = cache_data
                st.session_state.last_mult_params = params
                
                # Store current model for R output display
                st.session_state.current_model = model_mult
                st.session_state.current_feature_names = ["hp", "drat", "wt"]
                
                logger.info(f"Multiple regression data loaded successfully: {dataset_choice}")
                return cache_data
                
        except Exception as e:
            logger.error(f"Error generating multiple regression data: {e}")
            st.error(f"❌ Fehler beim Laden der Daten: {str(e)}")
            st.info("💡 Bitte versuchen Sie andere Parameter oder laden Sie die Seite neu.")
            if "error_count" not in st.session_state:
                st.session_state.error_count = 0
            st.session_state.error_count += 1
            raise
    else:
        # Use cached data
        try:
            cached = st.session_state.mult_model_cache
            
            # Recompute statistics from cached model if not cached
            if "mult_coeffs" not in cached:
                model_mult = cached["model_mult"]
                cached["mult_coeffs"] = get_model_coefficients(model_mult)
                cached["mult_summary"] = get_model_summary_stats(model_mult)
                cached["mult_diagnostics"] = get_model_diagnostics(model_mult)
            
            # Update current model for R output display
            st.session_state.current_model = cached["model_mult"]
            st.session_state.current_feature_names = ["hp", "drat", "wt"]
            
            logger.debug("Using cached multiple regression data")
            return cached
            
        except Exception as e:
            logger.error(f"Error loading cached multiple regression data: {e}")
            st.error(f"❌ Fehler beim Laden der Cache-Daten: {str(e)}")
            st.info("💡 Die Daten werden neu generiert...")
            # Clear cache to force regeneration
            st.session_state.mult_model_cache = None
            st.session_state.last_mult_params = None
            st.rerun()


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
    Load and prepare simple regression data with caching.
    
    Args:
        dataset_choice: Name of the dataset to load
        x_variable: X variable to use (for multi-variable datasets)
        n: Number of observations
        true_intercept: True intercept parameter (for simulated data)
        true_beta: True slope parameter (for simulated data)
        noise_level: Noise level for data generation
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing all prepared data and model results
    
    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If data loading or model fitting fails
    """
    # Handle simulated dataset
    if dataset_choice == "🏪 Elektronikmarkt (simuliert)":
        simple_params = (dataset_choice, n, true_intercept, true_beta, noise_level, seed)
        
        if st.session_state.last_simple_params != simple_params:
            with st.spinner("🔄 Generiere Daten..."):
                electronics_data = generate_electronics_market_data(
                    n, true_intercept, true_beta, noise_level, seed
                )
                
                # Cache temporary data
                st.session_state.simple_data_temp = electronics_data
                st.session_state.last_simple_params = simple_params
                
                return electronics_data
        else:
            # Use cached data
            if st.session_state.simple_data_temp:
                return st.session_state.simple_data_temp
            else:
                # Fallback: regenerate
                electronics_data = generate_electronics_market_data(
                    n, true_intercept, true_beta, noise_level, seed
                )
                st.session_state.simple_data_temp = electronics_data
                return electronics_data
    else:
        # Handle real datasets
        try:
            with st.spinner("🔄 Lade Datensatz..."):
                simple_data = generate_simple_regression_data(
                    dataset_choice, x_variable, n, seed=seed
                )
            return simple_data
        except Exception as e:
            logger.error(f"Error loading simple regression data: {e}")
            st.error(f"❌ Fehler beim Laden des Datensatzes: {str(e)}")
            st.info("💡 Verwenden Sie den Elektronikmarkt-Datensatz als Fallback.")
            
            # Use fallback data
            fallback_data = generate_electronics_market_data(12, 0.6, 0.52, 0.4, 42)
            if "error_count" not in st.session_state:
                st.session_state.error_count = 0
            st.session_state.error_count += 1
            return fallback_data


def compute_simple_regression_model(
    x, y, x_label: str, y_label: str, n: int
) -> Dict[str, Any]:
    """
    Compute simple regression model with caching.
    
    Args:
        x: X variable data
        y: Y variable data
        x_label: Label for X variable
        y_label: Label for Y variable
        n: Number of observations
    
    Returns:
        Dictionary containing model and all computed statistics
    """
    # Build parameter tuple for cache validation
    simple_model_key = (
        x_label,
        calculate_basic_stats(x)["count"],
        calculate_basic_stats(x)["mean"],
        calculate_basic_stats(y)["mean"],
    )
    
    # Check if we need to recompute the model
    needs_recompute = (
        st.session_state.simple_model_cache is None
        or "model_key" not in st.session_state.simple_model_cache
        or st.session_state.simple_model_cache.get("model_key") != simple_model_key
    )
    
    if needs_recompute:
        try:
            with st.spinner("📊 Berechne Regressionsmodell..."):
                df = pd.DataFrame({x_label: x, y_label: y})
                
                X = create_design_matrix(x)
                model, y_pred = fit_ols_model(X, y)
                
                # Use centralized statistical computations
                stats_results = compute_simple_regression_stats(model, X, y, n)
                
                # Get additional centralized model information
                simple_coeffs = get_model_coefficients(model)
                simple_summary = get_model_summary_stats(model)
                simple_diagnostics = get_model_diagnostics(model)
                
                # Cache all computed values
                cache_data = {
                    "model_key": simple_model_key,
                    "df": df,
                    "X": X,
                    "model": model,
                    "y_pred": y_pred,
                    "x": x,
                    "y": y,
                    **stats_results,  # Unpack all statistics
                    "simple_coeffs": simple_coeffs,
                    "simple_summary": simple_summary,
                    "simple_diagnostics": simple_diagnostics,
                }
                
                st.session_state.simple_model_cache = cache_data
                
                # Store current model for R output display
                st.session_state.current_model = model
                st.session_state.current_feature_names = [x_label]
                
                logger.info("Simple regression model computed successfully")
                return cache_data
                
        except Exception as e:
            logger.error(f"Error computing simple regression model: {e}")
            st.error(f"❌ Fehler bei der Berechnung des Regressionsmodells: {str(e)}")
            st.info("💡 Bitte überprüfen Sie Ihre Daten oder versuchen Sie andere Parameter.")
            if "error_count" not in st.session_state:
                st.session_state.error_count = 0
            st.session_state.error_count += 1
            raise
    else:
        # Use cached model results
        try:
            cached = st.session_state.simple_model_cache
            
            # Store current model for R output display
            st.session_state.current_model = cached["model"]
            st.session_state.current_feature_names = [x_label]
            
            logger.debug("Using cached simple regression model")
            return cached
            
        except Exception as e:
            logger.error(f"Error loading cached simple regression model: {e}")
            st.error(f"❌ Fehler beim Laden der Cache-Daten: {str(e)}")
            st.info("💡 Die Daten werden neu berechnet...")
            # Clear cache to force regeneration
            st.session_state.simple_model_cache = None
            if "error_count" not in st.session_state:
                st.session_state.error_count = 0
            st.session_state.error_count += 1
            st.rerun()


def _validate_multiple_regression_params(n: int, noise_level: float, seed: int) -> None:
    """
    Validate multiple regression parameters.
    
    Args:
        n: Number of observations
        noise_level: Noise level
        seed: Random seed
    
    Raises:
        ValueError: If any parameter is invalid
    """
    try:
        if n <= 0:
            st.error("❌ Fehler: Die Anzahl der Beobachtungen muss positiv sein.")
            st.stop()
        if noise_level < 0:
            st.error("❌ Fehler: Das Rauschen kann nicht negativ sein.")
            st.stop()
        if seed <= 0 or seed >= 10000:
            st.warning("⚠️ Warnung: Der Random Seed sollte zwischen 1 und 9999 liegen.")
    except Exception as e:
        logger.error(f"Input validation error: {e}")
        st.error(f"❌ Fehler bei der Eingabevalidierung: {str(e)}")
        st.stop()
