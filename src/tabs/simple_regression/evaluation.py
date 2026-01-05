"""
Model Evaluation section for Simple Linear Regression.

This module covers model evaluation metrics and goodness-of-fit measures.
"""

import streamlit as st
from statistics import compute_model_evaluation_metrics
from plots import create_residuals_plot
from logger import get_logger

logger = get_logger(__name__)


def render_evaluation(params: dict) -> None:
    """
    Render the model evaluation section.

    Args:
        params: Dictionary containing UI parameters
    """
    logger.debug("Rendering model evaluation section")

    st.subheader("📊 Modellevaluation")

    st.markdown("""
    ### 🎯 Gütemaße für Regression

    **Bestimmtheitsmaß (R²)**:
    $$
    R^2 = 1 - \\frac{ SS_{res} }{ SS_{tot} } = \\frac{ SS_{reg} }{ SS_{tot} }
    $$

    **Residuenquadratsumme**:
    $$
    SS_{res} = \\sum_{i=1}^n e_i^2
    $$

    **Totale Quadratsumme**:
    $$
    SS_{tot} = \\sum_{i=1}^n (y_i - \\bar{y})^2
    $$
    """)

    # Evaluation metrics display
    if st.checkbox("📈 Zeige Evaluierungsmetriken", value=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("**R²**", "0.85", "85% der Varianz erklärt")
        with col2:
            st.metric("**RMSE**", "0.32", "Wurzel der mittleren quadratischen Abweichung")
        with col3:
            st.metric("**MAE**", "0.25", "Mittlere absolute Abweichung")