"""
ANOVA for Group Comparisons in Simple Linear Regression.

This module explains how to use ANOVA to compare regression models
and test for group differences.
"""

import streamlit as st
from statistics import compute_anova_table
from logger import get_logger

logger = get_logger(__name__)


def render_anova(params: dict) -> None:
    """
    Render the ANOVA analysis section.

    Args:
        params: Dictionary containing UI parameters
    """
    logger.debug("Rendering ANOVA section")

    st.subheader("📊 ANOVA für Gruppenvergleiche")

    st.markdown("""
    ### 🎯 ANOVA-Tabelle für Regression

    Die Gesamtvariation wird zerlegt in:

    | Quelle          | SS          | df    | MS          | F          |
    |----------------|-------------|-------|-------------|------------|
    | Regression     | SS_reg      | 1     | MS_reg      | F-stat     |
    | Residuen       | SS_res      | n-2   | MS_res      |            |
    | **Total**      | **SS_tot**  | **n-1**|             |            |

    **F-Statistik**: $F = \\frac{ MS_{reg} }{ MS_{res} }$
    """)

    # ANOVA table display
    if st.checkbox("📋 Zeige ANOVA-Tabelle", value=True):
        # Placeholder ANOVA table
        anova_data = {
            "Quelle": ["Regression", "Residuen", "Total"],
            "SS": [125.6, 23.4, 149.0],
            "df": [1, 48, 49],
            "MS": [125.6, 0.49, None],
            "F": [256.7, None, None]
        }

        st.dataframe(anova_data, width='stretch')