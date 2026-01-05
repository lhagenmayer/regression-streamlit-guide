"""
Configuration constants for the Linear Regression Guide.

This module contains all centralized configuration constants including colors,
font sizes, data generation parameters, and UI defaults used throughout the application.
"""

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

# Color scheme for visualizations and UI elements
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#2c3e50",
    "correct": "#2ecc71",
    "violated": "#e74c3c",
    "warning": "#f39c12",
    "neutral": "#95a5a6",
    "residual_pos": "#27ae60",
    "residual_neg": "#c0392b",
    "regression_line": "#e74c3c",
    "data_points": "#3498db",
    "confidence_band": "#85c1e9",
}

# Font sizes for different plot elements
FONT_SIZES = {
    "axis_label_3d": 14,
    "axis_label_2d": 12,
    "title_3d": 14,
    "title_2d": 13,
    "tick_3d": 11,
    "legend": 10,
}

# ============================================================================
# UI DEFAULTS & SLIDER CONFIGURATION
# ============================================================================

# Default random seed for reproducibility
DEFAULT_SEED = 42

# Seed input constraints
SEED_MIN = 1
SEED_MAX = 999
SEED_DEFAULT = 42

# ============================================================================
# MULTIPLE REGRESSION DATASETS
# ============================================================================

# Cities dataset parameters
CITIES_DATASET = {
    "n_default": 75,
    "n_min": 20,
    "n_max": 150,
    "n_step": 5,
    "price_mean": 5.69,
    "price_std": 0.52,
    "price_min": 4.83,
    "price_max": 6.49,
    "advertising_mean": 1.84,
    "advertising_std": 0.83,
    "advertising_min": 0.50,
    "advertising_max": 3.10,
    "noise_std": 3.5,
    "y_min": 62.4,
    "y_max": 91.2,
    "y_std_target": 6.49,
    "y_mean_target": 77.37,
}

# Houses dataset parameters
HOUSES_DATASET = {
    "n_default": 1000,
    "n_min": 100,
    "n_max": 2000,
    "n_step": 100,
    "area_mean": 25.21,
    "area_std": 2.92,
    "area_min": 20.03,
    "area_max": 30.00,
    "pool_probability": 0.204,
    "noise_std": 20.0,
    "noise_min": 5.0,
    "noise_max": 40.0,
    "noise_default": 20.0,
    "noise_step": 5.0,
}

# Electronics dataset parameters
ELECTRONICS_DATASET = {
    "n_default": 50,
    "n_min": 20,
    "n_max": 100,
    "n_step": 5,
    "noise_std": 0.35,
    "noise_min": 0.1,
    "noise_max": 1.0,
    "noise_default": 0.35,
    "noise_step": 0.05,
}

# ============================================================================
# SIMPLE REGRESSION PARAMETERS
# ============================================================================

SIMPLE_REGRESSION = {
    "n_default": 12,
    "n_min": 8,
    "n_max": 50,
    "n_step": 1,
    "intercept_min": -1.0,
    "intercept_max": 3.0,
    "intercept_default": 0.6,
    "intercept_step": 0.1,
    "slope_min": 0.1,
    "slope_max": 1.5,
    "slope_default": 0.52,
    "slope_step": 0.01,
    "noise_min": 0.1,
    "noise_max": 1.5,
    "noise_default": 0.4,
    "noise_step": 0.05,
}

# ============================================================================
# 3D VISUALIZATION CONTROLS
# ============================================================================

VISUALIZATION_3D = {
    "x1_slider_min": 4.5,
    "x1_slider_max": 7.0,
    "x1_slider_default": 5.5,
    "x1_slider_step": 0.1,
    "x2_slider_min": 0.5,
    "x2_slider_max": 3.5,
    "x2_slider_default": 2.0,
    "x2_slider_step": 0.1,
    "houses_x1_min": 20.0,
    "houses_x1_max": 30.0,
    "houses_x1_default": 25.0,
    "houses_x1_step": 0.5,
}
