"""
OLS Estimation section for Simple Linear Regression.

This module explains Ordinary Least Squares estimation
and how to fit the regression line to data.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from ...ui_config import CSS_STYLES
from ...data import generate_simple_regression_data
from ...statistics import compute_simple_regression
from ...logger import get_logger

logger = get_logger(__name__)


def render_estimation(params: dict) -> None:
    """
    Render the OLS estimation section.

    Args:
        params: Dictionary containing UI parameters
    """
    logger.debug("Rendering OLS estimation section")

    # =========================================================
    # KAPITEL 3: DIE METHODE (OLS)
    # =========================================================
    st.markdown("---")
    st.markdown(
        '<p class="section-header">3.0 Die Methode: Schätzung mittels OLS</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    Die **Methode der kleinsten Quadrate (Ordinary Least Squares, OLS)** findet die optimale Gerade.

    **Das Kernprinzip:** Wähle jene Gerade, welche die **Summe der quadrierten vertikalen Abweichungen**
    (Residuen) zwischen Datenpunkten und Gerade **minimiert**.
    """
    )

    # OLS Visualisierung
    col_ols1, col_ols2 = st.columns([2, 1])

    with col_ols1:
        # Generate data and compute OLS
        data = generate_simple_regression_data(
            n=params.get('n', 100),
            true_intercept=params.get('true_intercept', 2.0),
            true_beta=params.get('true_beta', 0.5),
            noise_level=params.get('noise_level', 1.0),
            seed=params.get('seed', 42)
        )

        x, y = data['x'], data['y']
        model_results = compute_simple_regression(x, y)
        y_pred = model_results['predictions']
        b0, b1 = model_results['coefficients']

        # Create OLS plot with plotly
        fig_ols = go.Figure()

        # Data points
        fig_ols.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    size=10, color="#1f77b4", opacity=0.7, line=dict(width=2, color="white")
                ),
                name="Datenpunkte",
            )
        )

        # OLS regression line
        fig_ols.add_trace(
            go.Scatter(
                x=x,
                y=y_pred,
                mode="lines",
                line=dict(color="red", width=3),
                name=f"OLS-Gerade: ŷ = {b0:.3f} + {b1:.3f}x",
            )
        )

        # True line if shown
        if params.get('show_true_line', False):
            y_true = params.get('true_intercept', 2.0) + params.get('true_beta', 0.5) * x
            fig_ols.add_trace(
                go.Scatter(
                    x=x,
                    y=y_true,
                    mode="lines",
                    line=dict(color="green", width=2, dash="dash"),
                    name=f"Wahre Gerade: y = {params.get('true_intercept', 2.0):.1f} + {params.get('true_beta', 0.5):.1f}x",
                )
            )

        fig_ols.update_layout(
            title="OLS-Schätzung: Anpassung der Regressionsgeraden",
            xaxis_title="X (unabhängige Variable)",
            yaxis_title="Y (abhängige Variable)",
            showlegend=True,
            height=400
        )

        st.plotly_chart(fig_ols, width='stretch')