"""
Session state management for the Linear Regression Guide.

This module handles all session state initialization, cache management,
and parameter tracking for the Streamlit application.
"""

import streamlit as st
from typing import Optional, Dict, Any, Tuple
import numpy as np

from logger import get_logger

logger = get_logger(__name__)


def initialize_session_state() -> None:
    """
    Initialize all session state variables for the application.
    
    This should be called once at application startup to set up
    all required session state variables with their default values.
    """
    logger.debug("Initializing session state")
    
    # Tab navigation state
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = 0
    
    # Multiple regression caching
    if "last_mult_params" not in st.session_state:
        st.session_state.last_mult_params = None
    if "mult_model_cache" not in st.session_state:
        st.session_state.mult_model_cache = None
    
    # Simple regression caching
    if "last_simple_params" not in st.session_state:
        st.session_state.last_simple_params = None
    if "simple_model_cache" not in st.session_state:
        st.session_state.simple_model_cache = None
    if "simple_data_temp" not in st.session_state:
        st.session_state.simple_data_temp = {}
    
    # Current model state for R output display
    if "current_model" not in st.session_state:
        st.session_state.current_model = None
    if "current_feature_names" not in st.session_state:
        st.session_state.current_feature_names = None
    
    logger.info("Session state initialized successfully")


def check_params_changed(
    current_params: Tuple, 
    cache_key: str = "last_mult_params"
) -> bool:
    """
    Check if parameters have changed since last computation.
    
    Args:
        current_params: Tuple of current parameter values
        cache_key: Session state key for cached parameters
    
    Returns:
        True if parameters have changed, False otherwise
    """
    cached_params = st.session_state.get(cache_key)
    return cached_params != current_params


def update_cached_params(
    params: Tuple, 
    cache_key: str = "last_mult_params"
) -> None:
    """
    Update cached parameters in session state.
    
    Args:
        params: Tuple of parameter values to cache
        cache_key: Session state key for caching
    """
    st.session_state[cache_key] = params
    logger.debug(f"Updated cached params for {cache_key}")


def cache_model_data(
    data: Dict[str, Any], 
    cache_key: str = "mult_model_cache"
) -> None:
    """
    Cache model data in session state.
    
    Args:
        data: Dictionary containing model data to cache
        cache_key: Session state key for caching
    """
    st.session_state[cache_key] = data
    logger.debug(f"Cached model data for {cache_key}")


def get_cached_model_data(
    cache_key: str = "mult_model_cache"
) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached model data from session state.
    
    Args:
        cache_key: Session state key for cached data
    
    Returns:
        Cached model data or None if not available
    """
    return st.session_state.get(cache_key)


def update_current_model(
    model: Any, 
    feature_names: list
) -> None:
    """
    Update the current model and feature names for R output display.
    
    Args:
        model: The fitted regression model
        feature_names: List of feature names
    """
    st.session_state.current_model = model
    st.session_state.current_feature_names = feature_names
    logger.debug(f"Updated current model with {len(feature_names)} features")


def get_current_model() -> Tuple[Optional[Any], Optional[list]]:
    """
    Get the current model and feature names from session state.
    
    Returns:
        Tuple of (model, feature_names) or (None, None) if not available
    """
    return (
        st.session_state.get("current_model"),
        st.session_state.get("current_feature_names")
    )


def cache_simple_data_temp(data: Dict[str, Any]) -> None:
    """
    Cache temporary simple regression data.
    
    Args:
        data: Dictionary containing temporary data
    """
    st.session_state.simple_data_temp = data
    logger.debug("Cached temporary simple regression data")


def get_simple_data_temp() -> Dict[str, Any]:
    """
    Get cached temporary simple regression data.
    
    Returns:
        Dictionary of cached temporary data
    """
    return st.session_state.get("simple_data_temp", {})


def clear_cache(cache_keys: Optional[list] = None) -> None:
    """
    Clear specified cache keys or all caches if none specified.
    
    Args:
        cache_keys: List of cache keys to clear, or None to clear all
    """
    if cache_keys is None:
        cache_keys = [
            "last_mult_params",
            "mult_model_cache",
            "last_simple_params",
            "simple_model_cache",
            "simple_data_temp",
        ]
    
    for key in cache_keys:
        if key in st.session_state:
            st.session_state[key] = None if "params" in key or "model" in key else {}
            logger.debug(f"Cleared cache for {key}")
