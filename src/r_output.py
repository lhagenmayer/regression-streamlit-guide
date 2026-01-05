"""
R-style output display for regression models.

This module provides functions to render R-style statistical output
for regression models in the Streamlit application.
"""

import streamlit as st
from typing import Optional, List, Any

from .plots import create_r_output_figure
from .logger import get_logger
from .perplexity_api import interpret_model, is_api_configured

logger = get_logger(__name__)


def render_r_output_section(
    model: Optional[Any] = None,
    feature_names: Optional[List[str]] = None,
    figsize: tuple = (18, 13)
) -> None:
    """
    Render the R output section with model summary and explanation.
    
    Args:
        model: Fitted regression model (statsmodels)
        feature_names: List of feature names for the model
        figsize: Figure size as (width, height)
    """
    logger.debug(f"Rendering R output section (model={'present' if model else 'absent'})")
    
    st.markdown("---")
    
    # Create two columns: R output on left, explanation/interpretation on right
    col_r_output, col_r_explanation = st.columns([3, 2])
    
    with col_r_output:
        st.markdown("### 📊 R Output (Automatisch aktualisiert)")
        
        # Display R output based on current model
        try:
            if model is not None and feature_names is not None:
                fig_r = create_r_output_figure(
                    model, 
                    feature_names=feature_names, 
                    figsize=figsize
                )
                st.plotly_chart(fig_r, use_container_width=True)
            else:
                st.info("ℹ️ Wählen Sie einen Datensatz und Parameter aus, um das R Output zu sehen.")
        except Exception as e:
            st.warning(f"R Output konnte nicht geladen werden: {str(e)}")
            logger.error(f"Error rendering R output: {e}", exc_info=True)
    
    with col_r_explanation:
        _render_r_output_explanation()
        
        # Add interpretation button if model is available
        if model is not None and feature_names is not None:
            st.markdown("---")
            _render_interpretation_section(model, feature_names)
    
    st.markdown("---")


def _render_r_output_explanation() -> None:
    """
    Render the explanation panel for R output sections.
    
    This is a private helper function that provides detailed explanations
    of each section in the R output.
    """
    with st.expander("📖 Erklärung der R Output Abschnitte", expanded=False):
        st.markdown("""
        #### Erklärung der Abschnitte (kurz, präzise)
        • **Call**: zeigt die verwendete Modellformel und das Datenset; nützlich zur Reproduzierbarkeit.

        • **Residuals**: fünf‑Zahlen‑Zusammenfassung der Residuen (Min, 1Q, Median, 3Q, Max) zur schnellen Beurteilung von Schiefe/Ausreißern.

        • **Coefficients**: vier Spalten: Estimate, Std. Error, t value, Pr(>|t|); jede Zeile ist ein Prädiktor (Intercept inklusive). Signifikanzsterne werden darunter erklärt.

        • **Residual standard error und degrees of freedom**: Schätzung der Fehlerstreuung und Freiheitsgrade für Tests.

        • **Multiple R-squared / Adjusted R-squared**: erklärte Varianz und bereinigte Version (bestraft unnötige Prädiktoren).

        • **F-statistic**: globaler Test, ob mindestens ein Prädiktor das Modell signifikant verbessert; p‑value dazu wird angezeigt.

        ---
        #### Wichtige Hinweise, Entscheidungen und Risiken
        • **Interpretation der Koeffizienten**: Ein Estimate ist die geschätzte Änderung in der Zielvariable pro Einheitenänderung des Prädiktors bei konstanten anderen Variablen; Pr(>|t|) gibt die zweiseitige p‑Wert‑Signifikanz an.

        • **Achtung bei Multikollinearität**: hohe Standardfehler oder aliasing können Koeffizienten unzuverlässig machen; summary() zeigt aliased coefficients nicht, Details in summary.lm‑Dokumentation.
        """)


def render_r_output_from_session_state() -> None:
    """
    Render R output using model and feature names from session state.
    
    This is a convenience function that retrieves the current model
    from session state and renders the R output section.
    """
    logger.debug("Rendering R output from session state")
    
    # Get model from session state
    model = st.session_state.get("current_model")
    feature_names = st.session_state.get("current_feature_names")
    
    # Render the R output section
    render_r_output_section(model=model, feature_names=feature_names)


def _render_interpretation_section(model: Any, feature_names: List[str]) -> None:
    """
    Render the interpretation section with button to get AI interpretation.
    
    Args:
        model: Fitted regression model
        feature_names: List of feature names
    """
    st.markdown("### 🤖 AI-Interpretation")
    
    # Check if API is configured
    if not is_api_configured():
        st.warning(
            "⚠️ Perplexity API nicht konfiguriert. "
            "Setzen Sie die Umgebungsvariable `PERPLEXITY_API_KEY` um diese Funktion zu nutzen."
        )
        with st.expander("ℹ️ Wie konfiguriere ich die API?"):
            st.markdown("""
            **So erhalten Sie einen API-Schlüssel:**
            
            1. Besuchen Sie [Perplexity API](https://www.perplexity.ai/settings/api)
            2. Erstellen Sie ein Konto oder melden Sie sich an
            3. Generieren Sie einen API-Schlüssel
            4. Setzen Sie die Umgebungsvariable:
            
            ```bash
            export PERPLEXITY_API_KEY="your-api-key-here"
            ```
            
            Oder fügen Sie sie zu Ihrer `.streamlit/secrets.toml` hinzu:
            
            ```toml
            PERPLEXITY_API_KEY = "your-api-key-here"
            ```
            """)
        return
    
    # Initialize session state for interpretation
    if "interpretation_result" not in st.session_state:
        st.session_state.interpretation_result = None
    if "interpretation_loading" not in st.session_state:
        st.session_state.interpretation_loading = False
    
    # Button to trigger interpretation
    if st.button("🔍 Interpretation generieren", type="primary", use_container_width=True):
        st.session_state.interpretation_loading = True
        
        with st.spinner("🤔 Analysiere Modell mit Perplexity AI..."):
            result = interpret_model(model, feature_names)
            st.session_state.interpretation_result = result
            st.session_state.interpretation_loading = False
    
    # Display interpretation if available
    if st.session_state.interpretation_result is not None:
        result = st.session_state.interpretation_result
        
        if result.get("success"):
            st.markdown("#### 📝 Interpretation:")
            st.markdown(result.get("interpretation", ""))
            
            # Add a small note about the source
            st.caption("_Generiert von Perplexity AI_")
            
            # Option to clear interpretation
            if st.button("🔄 Neue Interpretation", use_container_width=True):
                st.session_state.interpretation_result = None
                st.rerun()
        else:
            st.error(f"❌ {result.get('error', 'Unbekannter Fehler')}")
            
            # Option to retry
            if st.button("🔄 Erneut versuchen", use_container_width=True):
                st.session_state.interpretation_result = None
                st.rerun()
