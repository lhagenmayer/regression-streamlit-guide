"""
Simple regression tab for the Linear Regression Guide.

This module renders the simple linear regression analysis tab.
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any

from ...config import COLUMN_LAYOUTS, CAMERA_PRESETS
from ..plots import (
    create_plotly_scatter_with_line,
    create_plotly_residual_plot,
    create_plotly_bar,
    create_r_output_figure,
    get_signif_stars,
)
from ...data import get_simple_regression_content
from ...config import get_logger

logger = get_logger(__name__)


def render_simple_regression_tab(
    model_data: Dict[str, Any],
    x_label: str,
    y_label: str,
    x_unit: str,
    y_unit: str,
    context_title: str,
    context_description: str,
    show_formulas: bool = True,
    show_true_line: bool = False,
    has_true_line: bool = False,
    true_intercept: float = 0,
    true_beta: float = 0
) -> None:
    """
    Render the simple regression analysis tab.
    
    Args:
        model_data: Dictionary containing model and statistics
        x_label: Label for X variable
        y_label: Label for Y variable
        x_unit: Unit for X variable
        y_unit: Unit for Y variable
        context_title: Title for the context
        context_description: Description of the context
        show_formulas: Whether to show mathematical formulas
        show_true_line: Whether to show the true regression line
        has_true_line: Whether dataset has a true line
        true_intercept: True intercept (for simulated data)
        true_beta: True slope (for simulated data)
    """
    # This is a large tab that will be populated from app.py content
    # For now, we'll create a simple placeholder
    st.markdown('<p class="main-header">📈 Leitfaden zur Einfachen Linearen Regression</p>', 
                unsafe_allow_html=True)
    st.markdown("### Von der Frage zur validierten Erkenntnis")
    
    # Extract commonly used variables from model_data
    model = model_data["model"]
    x = model_data["x"]
    y = model_data["y"]
    y_pred = model_data["y_pred"]
    b0 = model_data["b0"]
    b1 = model_data["b1"]
    
    st.info(
        f"""
        **{context_title}**
        
        {context_description}
        """
    )
    
    # Show basic regression results
    st.markdown("### 📊 Regressionsergebnisse")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Intercept (β₀)", f"{b0:.4f}")
    with col2:
        st.metric("Steigung (β₁)", f"{b1:.4f}")
    with col3:
        st.metric("R²", f"{model.rsquared:.4f}")
    
    # Create scatter plot with regression line
    fig = create_plotly_scatter_with_line(
        x=x,
        y=y,
        y_pred=y_pred,
        x_label=x_label,
        y_label=y_label,
        title=f"Regression: {y_label} vs {x_label}"
    )
    st.plotly_chart(fig, key="simple_regression_scatter", use_container_width=True)
    
    # Create residual plot
    fig_resid = create_plotly_residual_plot(
        y_pred, model.resid, title="Residual Plot"
    )
    st.plotly_chart(fig_resid, key="simple_regression_residuals", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 📈 Detaillierte Analyse")
    st.info("Die vollständige detaillierte Analyse wird in einer zukünftigen Version verfügbar sein.")
    
    logger.info("Simple regression tab rendered")
