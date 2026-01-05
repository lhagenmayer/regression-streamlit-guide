"""
The Linear Model section for Simple Linear Regression.

This module explains the theoretical foundation of linear regression
and the mathematical model.
"""

import streamlit as st
import numpy as np
from ...ui_config import CSS_STYLES
from ...plots import create_scatter_plot, create_regression_line_plot
from ...data import generate_simple_regression_data
from ...logger import get_logger

logger = get_logger(__name__)


def render_model(params: dict) -> None:
    """
    Render the linear model explanation section.

    Args:
        params: Dictionary containing UI parameters
    """
    logger.debug("Rendering linear model section")

    # =========================================================
    # KAPITEL 2: DAS FUNDAMENT
    # =========================================================
    st.markdown("---")
    st.markdown(
        '<p class="section-header">2.0 Das Fundament: Das einfache lineare Regressionsmodell</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    Das Verständnis des einfachen linearen Regressionsmodells ist der entscheidende erste Schritt.
    Die Rollen der Variablen werden klar definiert:

    - **Abhängige Variable (Y):** Die Zielvariable – was wir erklären/vorhersagen wollen
    - **Unabhängige Variable (X):** Der Prädiktor – was die Veränderung erklärt
    """
    )

    col_model1, col_model2 = st.columns([1.2, 1])

    with col_model1:
        st.markdown("### Das grundlegende Modell:")
        if params.get('show_formulas', True):
            st.latex(r"y_i = \beta_0 + \beta_1 \cdot x_i + \varepsilon_i")

        st.markdown(
            """
        | Symbol | Bedeutung |
        |--------|-----------|
        | **β₀** | Wahrer Achsenabschnitt (unbekannt) |
        | **β₁** | Wahre Steigung (unbekannt) – Änderung in Y pro Einheit X |
        | **εᵢ** | Zufällige Störgrösse – alle anderen Einflüsse |
        """
        )

    with col_model2:
        context_title = params.get('context_title', 'Umsatzprognose')
        context_description = params.get('context_description', 'X = Werbebudget, Y = Umsatz')

        st.warning(
            f"""
        ### 🎯 Praxisbeispiel: {context_title}

        {context_description}
        """
        )

    # Erste Visualisierung: Die Rohdaten
    st.markdown(
        f'<p class="subsection-header">📊 Unsere Daten: {params.get("n", 100)} Beobachtungen</p>',
        unsafe_allow_html=True,
    )