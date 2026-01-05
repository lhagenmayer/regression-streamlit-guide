"""
Introduction & Problem Statement for Simple Linear Regression.

This module introduces the concept of simple linear regression,
explaining the basic problem and motivation.
"""

import streamlit as st
from ...ui_config import CSS_STYLES
from ...logger import get_logger

logger = get_logger(__name__)


def render_intro() -> None:
    """Render the introduction section for simple linear regression."""
    logger.debug("Rendering simple regression introduction")

    st.markdown(
        f'<p style="{CSS_STYLES["main_header"]}">📖 Umfassender Leitfaden zur Linearen Regression</p>',
        unsafe_allow_html=True,
    )
    st.markdown("### Von der Frage zur validierten Erkenntnis – Ein interaktiver Lernpfad")

    st.markdown("---")
    st.markdown(
        '<p class="section-header">1.0 Einleitung: Die Analyse von Zusammenhängen</p>',
        unsafe_allow_html=True,
    )

    col_intro1, col_intro2 = st.columns([2, 1])

    with col_intro1:
        st.markdown(
            """
        Von der Vorhersage von Unternehmensumsätzen bis hin zur Aufdeckung wissenschaftlicher
        Zusammenhänge – die Fähigkeit, Beziehungen in Daten zu quantifizieren, ist eine
        **Kernkompetenz** in der modernen Analyse.

        Die **Regressionsanalyse** ist das universelle Werkzeug für diese Aufgabe. Sie geht über
        die blosse Feststellung *ob* Variablen zusammenhängen hinaus und erklärt präzise,
        **wie** sie sich gegenseitig beeinflussen.

        > ⚠️ **Wichtig:** Die Regression allein beweist keine Kausalität! Sie quantifiziert die
        > Stärke einer *potenziellen* Ursache-Wirkungs-Beziehung, die durch das Studiendesign
        > gestützt werden muss.
        """
        )

    with col_intro2:
        st.info(
            """
        **Korrelation vs. Regression:**

        | Korrelation | Regression |
        |-------------|------------|
        | *Ungerichtet* | *Gerichtet* |
        | Wie stark? | Um wieviel? |
        | r ∈ [-1, 1] | ŷ = b₀ + b₁x |
        """
        )