"""
Datasets overview tab for the Linear Regression Guide.

This module renders the datasets information tab showing
available datasets and their characteristics.
"""

import streamlit as st
import pandas as pd

from ..logger import get_logger

logger = get_logger(__name__)


def render() -> None:
    """
    Render the datasets overview tab.
    
    Displays information about all available datasets including:
    - Electronics market (simulated)
    - Cities sales study
    - House prices with pool
    """
    logger.debug("Rendering datasets tab")
    
    st.markdown('<p class="main-header">📚 Datensätze-Übersicht</p>', unsafe_allow_html=True)
    st.markdown("### Verfügbare Datensätze für Regression-Analysen")
    
    st.markdown("---")
    
    # Dataset 1: Elektronikmarkt
    _render_electronics_dataset()
    
    st.markdown("---")
    
    # Dataset 2: Städte-Umsatzstudie
    _render_cities_dataset()
    
    st.markdown("---")
    
    # Dataset 3: Häuserpreise
    _render_houses_dataset()
    
    st.markdown("---")
    
    # Comparison table
    _render_comparison_table()


def _render_electronics_dataset() -> None:
    """Render information about the electronics market dataset."""
    st.markdown("## 🏪 Elektronikmarkt (simuliert)")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(
            """
        **Beschreibung:** Ein simulierter Datensatz zur Analyse des Zusammenhangs zwischen
        Verkaufsfläche und Umsatz von Elektronikfachmärkten.

        **Verwendung:** Ideal für **einfache lineare Regression**

        **Variablen:**
        - **X (Prädiktor):** Verkaufsfläche (in 100 qm)
        - **Y (Zielvariable):** Umsatz (in Mio. €)

        **Besonderheit:** Die wahren Parameter (β₀, β₁) sind bekannt, da simuliert.
        Perfekt zum Lernen und Verstehen der Grundkonzepte!
        """
        )
    
    with col2:
        st.info(
            """
        **Stichprobengrösse:**
        - Anpassbar: 8-50 Beobachtungen

        **Parameter:**
        - Wahrer Intercept (β₀)
        - Wahre Steigung (β₁)
        - Rauschen-Level (σ)
        - Random Seed
        """
        )


def _render_cities_dataset() -> None:
    """Render information about the cities sales dataset."""
    st.markdown("## 🏙️ Städte-Umsatzstudie (75 Städte)")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(
            """
        **Beschreibung:** Reale Daten einer Handelskette, die in 75 Städten den Zusammenhang
        zwischen Produktpreis, Werbeausgaben und Umsatz untersucht.

        **Verwendung:**
        - **Einfache Regression:** Nur ein Prädiktor (entweder Preis ODER Werbung)
        - **Multiple Regression:** Beide Prädiktoren gleichzeitig

        **Variablen:**
        - **X₁:** Produktpreis (in CHF)
        - **X₂:** Werbeausgaben (in 1'000 CHF)
        - **Y:** Umsatz (in 1'000 CHF)

        **Didaktischer Wert:** Zeigt den Unterschied zwischen einfacher und multipler Regression!
        Bei einfacher Regression fehlt ein wichtiger Prädiktor → höherer Fehlerterm.
        """
        )
    
    with col2:
        st.info(
            """
        **Stichprobengrösse:**
        - n = 75 Städte (fixiert)

        **Statistiken:**
        - Preis: μ=5.69, σ=0.52
        - Werbung: μ=1.84, σ=0.83
        - Umsatz: μ=77.37, σ=6.49
        """
        )


def _render_houses_dataset() -> None:
    """Render information about the house prices dataset."""
    st.markdown("## 🏠 Häuserpreise mit Pool (1000 Häuser)")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(
            """
        **Beschreibung:** Eine Studie von 1000 Hausverkäufen in einer Universitätsstadt,
        die den Einfluss von Wohnfläche und Pool-Vorhandensein auf den Preis untersucht.

        **Verwendung:**
        - **Einfache Regression:** Nur ein Prädiktor (Wohnfläche ODER Pool)
        - **Multiple Regression:** Beide Prädiktoren gleichzeitig

        **Variablen:**
        - **X₁:** Wohnfläche (in sqft/10)
        - **X₂:** Pool (Dummy-Variable: 0 = kein Pool, 1 = Pool vorhanden)
        - **Y:** Hauspreis (in USD)

        **Besonderheit:** Enthält eine **Dummy-Variable** (Pool) - ideal zum Verstehen
        kategorialer Variablen in der Regression! 20.4% der Häuser haben einen Pool.
        """
        )
    
    with col2:
        st.info(
            """
        **Stichprobengrösse:**
        - n = 1000 Häuser (fixiert)

        **Statistiken:**
        - Wohnfläche: μ=25.21, σ=2.92
        - Pool: 20.4% haben Pool
        - Preis: μ=247.66, σ=42.19
        """
        )


def _render_comparison_table() -> None:
    """Render a comparison table of all datasets."""
    st.markdown("### 💡 Welchen Datensatz soll ich wählen?")
    
    comparison_df = pd.DataFrame(
        {
            "Datensatz": ["🏪 Elektronikmarkt", "🏙️ Städte-Umsatzstudie", "🏠 Häuserpreise"],
            "Ideal für": [
                "Anfänger & Grundkonzepte",
                "Vergleich einfach vs. multipel",
                "Dummy-Variablen",
            ],
            "Stichprobe": ["Klein (n=8-50)", "Mittel (n=75)", "Gross (n=1000)"],
            "Prädiktoren": ["1 (nur Fläche)", "2 (Preis, Werbung)", "2 (Fläche, Pool)"],
            "Wahre Parameter": ["✅ Bekannt", "❌ Unbekannt", "❌ Unbekannt"],
        }
    )
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
