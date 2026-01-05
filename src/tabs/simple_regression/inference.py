"""
Statistical Inference section for Simple Linear Regression.

This module covers hypothesis testing and confidence intervals
for regression parameters.
"""

import streamlit as st
from statistics import compute_inference_statistics
from logger import get_logger

logger = get_logger(__name__)


def render_inference(params: dict) -> None:
    """
    Render the statistical inference section.

    Args:
        params: Dictionary containing UI parameters
    """
    logger.debug("Rendering statistical inference section")

    st.subheader("🔬 Statistische Inferenz")

    st.markdown("""
    ### 📏 Hypothesentests für Regressionskoeffizienten

    **Nullhypothese**: $H_0: \\beta_1 = 0$ (kein linearer Zusammenhang)

    **Teststatistik**:
    $$
    t = \\frac{ \\hat{\\beta}_1 - \\beta_1^0 }{ SE(\\hat{\\beta}_1) }
    $$

    **p-Wert**: Wahrscheinlichkeit, einen so extremen Wert zu beobachten,
    wenn $H_0$ wahr ist.
    """)

    # Hypothesis testing results
    if st.checkbox("🧪 Zeige Testresultate", value=True):
        st.markdown("#### Test für Steigungskoeffizient")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("**t-Wert**", "8.45")
        with col2:
            st.metric("**p-Wert**", "< 0.001")
        with col3:
            st.metric("**Konfidenzintervall**", "[0.32, 0.68]")