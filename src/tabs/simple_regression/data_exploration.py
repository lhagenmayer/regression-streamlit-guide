"""
Data Exploration section for Simple Linear Regression.

This module handles data visualization and initial exploration
before building the regression model.
"""

import streamlit as st
from ...plots import create_scatter_plot
from ...data import generate_simple_regression_data
from ...logger import get_logger

logger = get_logger(__name__)


def render_data_exploration(params: dict) -> None:
    """
    Render the data exploration section.

    Args:
        params: Dictionary containing UI parameters
    """
    logger.debug("Rendering data exploration section")

    st.subheader("🔍 Datenexploration")

    # Generate data based on parameters
    data = generate_simple_regression_data(
        n=params.get('n', 100),
        true_intercept=params.get('true_intercept', 2.0),
        true_beta=params.get('true_beta', 0.5),
        noise_level=params.get('noise_level', 1.0),
        seed=params.get('seed', 42)
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Streudiagramm")
        fig = create_scatter_plot(data['x'], data['y'])
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown("### 📈 Datenübersicht")
        st.dataframe(data[['x', 'y']].head(10))
        st.metric("Anzahl Beobachtungen", len(data))
        st.metric("Korrelationskoeffizient",
                 ".3f")