"""
Sidebar components for parameter selection and configuration.

This module contains all sidebar UI components for dataset selection,
parameter configuration, and display options.
"""

import streamlit as st
from dataclasses import dataclass
from typing import Optional

from ..config import (
    DEFAULT_SEED,
    SEED_MIN,
    SEED_MAX,
    CITIES_DATASET,
    HOUSES_DATASET,
    ELECTRONICS_DATASET,
    SIMPLE_REGRESSION,
    UI_DEFAULTS,
)
from ..config import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetSelection:
    """Configuration for selected datasets."""
    simple_dataset: str
    multiple_dataset: str


@dataclass
class SimpleRegressionParams:
    """Parameters for simple regression data generation."""
    n: int
    true_intercept: float
    true_beta: float
    noise_level: float
    seed: int
    x_variable: Optional[str] = None


@dataclass
class MultipleRegressionParams:
    """Parameters for multiple regression data generation."""
    n: int
    noise_level: float
    seed: int


@dataclass
class DisplayOptions:
    """Display options for the UI."""
    show_formulas: bool
    show_true_line: bool


def render_sidebar_header() -> None:
    """Render the sidebar header."""
    st.sidebar.markdown("# 🎛️ Parameter")


def render_dataset_selection() -> DatasetSelection:
    """
    Render dataset selection dropdowns.
    
    Returns:
        DatasetSelection object with selected datasets
    """
    logger.debug("Rendering dataset selection")
    
    st.sidebar.markdown("---")
    with st.sidebar.expander("📊 Datensatz", expanded=True):
        simple_dataset = st.selectbox(
            "Datensatz wählen (Einfache Regression):",
            [
                "🏪 Elektronikmarkt (simuliert)",
                "🏙️ Städte-Umsatzstudie (75 Städte)",
                "🏠 Häuserpreise mit Pool (1000 Häuser)",
                "🇨🇭 Schweizer Kantone (sozioökonomisch)",
                "🌤️ Schweizer Wetterstationen",
            ],
            index=0,
            help="Wählen Sie zwischen simulierten Datensätzen, Schweizer Daten oder globalen API-Datensätzen.",
        )
        
        multiple_dataset = st.selectbox(
            "Datensatz wählen (Multiple Regression):",
            [
                "🏙️ Städte-Umsatzstudie (75 Städte)",
                "🏠 Häuserpreise mit Pool (1000 Häuser)",
                "🏪 Elektronikmarkt (erweitert)",
                "🇨🇭 Schweizer Kantone (sozioökonomisch)",
                "🌤️ Schweizer Wetterstationen",
            ],
            index=0,
            help="Wählen Sie einen Datensatz für multiple Regression (2+ Prädiktoren).",
            key="mult_dataset",
        )
    
    return DatasetSelection(
        simple_dataset=simple_dataset,
        multiple_dataset=multiple_dataset
    )


def render_multiple_regression_params(dataset_choice: str) -> MultipleRegressionParams:
    """
    Render parameter controls for multiple regression.
    
    Args:
        dataset_choice: Selected dataset name
    
    Returns:
        MultipleRegressionParams object with selected parameters
    """
    logger.debug(f"Rendering multiple regression params for {dataset_choice}")
    
    st.sidebar.markdown("---")
    with st.sidebar.expander("🎛️ Daten-Parameter (Multiple Regression)", expanded=False):
        if dataset_choice == "🏙️ Städte-Umsatzstudie (75 Städte)":
            st.markdown("**Stichproben-Eigenschaften:**")
            n = st.slider(
                "Anzahl Städte (n)",
                min_value=CITIES_DATASET["n_min"],
                max_value=CITIES_DATASET["n_max"],
                value=CITIES_DATASET["n_default"],
                step=CITIES_DATASET["n_step"],
                help="Grösse der Stichprobe",
                key="n_mult_staedte",
            )
            
            st.markdown("**Zufallskomponente:**")
            noise_level = st.slider(
                "Rauschen (σ)",
                min_value=CITIES_DATASET["noise_min"],
                max_value=CITIES_DATASET["noise_max"],
                value=CITIES_DATASET["noise_std"],
                step=CITIES_DATASET["noise_step"],
                help="Standardabweichung der Störgrösse",
                key="noise_mult_staedte",
            )
            seed = st.number_input(
                "Random Seed",
                min_value=SEED_MIN,
                max_value=SEED_MAX,
                value=DEFAULT_SEED,
                help="Zufallsseed für Reproduzierbarkeit",
                key="seed_mult_staedte",
            )
            
        elif dataset_choice == "🏠 Häuserpreise mit Pool (1000 Häuser)":
            st.markdown("**Stichproben-Eigenschaften:**")
            n = st.slider(
                "Anzahl Häuser (n)",
                min_value=HOUSES_DATASET["n_min"],
                max_value=HOUSES_DATASET["n_max"],
                value=HOUSES_DATASET["n_default"],
                step=HOUSES_DATASET["n_step"],
                help="Grösse der Stichprobe",
                key="n_mult_haeuser",
            )
            
            st.markdown("**Zufallskomponente:**")
            noise_level = st.slider(
                "Rauschen (σ)",
                min_value=HOUSES_DATASET["noise_min"],
                max_value=HOUSES_DATASET["noise_max"],
                value=HOUSES_DATASET["noise_default"],
                step=HOUSES_DATASET["noise_step"],
                help="Standardabweichung der Störgrösse",
                key="noise_mult_haeuser",
            )
            seed = st.number_input(
                "Random Seed",
                min_value=1,
                max_value=999,
                value=42,
                help="Zufallsseed für Reproduzierbarkeit",
                key="seed_mult_haeuser",
            )
            
        else:  # Elektronikmarkt
            st.markdown("**Stichproben-Eigenschaften:**")
            n = st.slider(
                "Anzahl Beobachtungen (n)",
                min_value=ELECTRONICS_DATASET["n_min"],
                max_value=ELECTRONICS_DATASET["n_max"],
                value=ELECTRONICS_DATASET["n_default"],
                step=ELECTRONICS_DATASET["n_step"],
                help="Grösse der Stichprobe",
                key="n_mult_elektro",
            )
            
            st.markdown("**Zufallskomponente:**")
            noise_level = st.slider(
                "Rauschen (σ)",
                min_value=ELECTRONICS_DATASET["noise_min"],
                max_value=ELECTRONICS_DATASET["noise_max"],
                value=ELECTRONICS_DATASET["noise_default"],
                step=ELECTRONICS_DATASET["noise_step"],
                help="Standardabweichung der Störgrösse",
                key="noise_mult_elektro",
            )
            seed = st.number_input(
                "Random Seed",
                min_value=1,
                max_value=999,
                value=42,
                help="Zufallsseed für Reproduzierbarkeit",
                key="seed_mult_elektro",
            )
    
    return MultipleRegressionParams(n=n, noise_level=noise_level, seed=seed)


def render_simple_regression_params(
    dataset_choice: str,
    has_true_line: bool = False
) -> SimpleRegressionParams:
    """
    Render parameter controls for simple regression.
    
    Args:
        dataset_choice: Selected dataset name
        has_true_line: Whether the dataset has true parameters
    
    Returns:
        SimpleRegressionParams object with selected parameters
    """
    logger.debug(f"Rendering simple regression params for {dataset_choice}")
    
    st.sidebar.markdown("---")
    with st.sidebar.expander("🎛️ Daten-Parameter (Einfache Regression)", expanded=False):
        x_variable = None
        
        if dataset_choice == "🏪 Elektronikmarkt (simuliert)":
            # X-Variable als Dropdown (nur eine Option verfügbar)
            x_variable_options = ["Verkaufsfläche (100qm)"]
            x_variable = st.selectbox(
                "X-Variable (Prädiktor):",
                x_variable_options,
                index=0,
                help="Beim simulierten Datensatz ist nur die Verkaufsfläche als Prädiktor verfügbar.",
            )
            
            st.markdown("**Stichproben-Eigenschaften:**")
            n = st.slider(
                "Anzahl Beobachtungen (n)",
                min_value=SIMPLE_REGRESSION["n_min"],
                max_value=SIMPLE_REGRESSION["n_max"],
                value=SIMPLE_REGRESSION["n_default"],
                step=SIMPLE_REGRESSION["n_step"],
                help="Grösse der Stichprobe (mehr Beobachtungen = präzisere Schätzungen)",
            )
            
            st.markdown("**Wahre Parameter (bekannt bei Simulation):**")
            true_intercept = st.slider(
                "Wahrer β₀ (Intercept)",
                min_value=SIMPLE_REGRESSION["intercept_min"],
                max_value=SIMPLE_REGRESSION["intercept_max"],
                value=SIMPLE_REGRESSION["intercept_default"],
                step=SIMPLE_REGRESSION["intercept_step"],
                help="Y-Achsenabschnitt: Wert von Y wenn X=0",
            )
            true_beta = st.slider(
                "Wahre Steigung β₁",
                min_value=SIMPLE_REGRESSION["slope_min"],
                max_value=SIMPLE_REGRESSION["slope_max"],
                value=SIMPLE_REGRESSION["slope_default"],
                step=SIMPLE_REGRESSION["slope_step"],
                help="Steigung: Änderung in Y pro Einheit X",
            )
            
            st.markdown("**Zufallskomponente:**")
            noise_level = st.slider(
                "Rauschen (σ)",
                min_value=SIMPLE_REGRESSION["noise_min"],
                max_value=SIMPLE_REGRESSION["noise_max"],
                value=SIMPLE_REGRESSION["noise_default"],
                step=SIMPLE_REGRESSION["noise_step"],
                help="Standardabweichung der Störgrösse (mehr Rauschen = schlechteres R²)",
            )
            seed = st.number_input(
                "Random Seed",
                min_value=SEED_MIN,
                max_value=SEED_MAX,
                value=DEFAULT_SEED,
                help="Zufallsseed für Reproduzierbarkeit",
            )
            
        elif dataset_choice == "🏙️ Städte-Umsatzstudie (75 Städte)":
            # X-Variable als Dropdown (zwei Optionen verfügbar)
            x_variable_options = ["Werbung (CHF1000)", "Preis (CHF)"]
            x_variable = st.selectbox(
                "X-Variable (Prädiktor):",
                x_variable_options,
                index=0,
                help="Einfache Regression: Nur EIN Prädiktor → grösserer Fehlerterm (didaktisch wertvoll!)",
            )
            st.sidebar.markdown("**Stichproben-Info:**")
            st.sidebar.info("n = 75 Städte (fixiert)")
            n = 75
            true_intercept = 0
            true_beta = 0
            noise_level = 0
            seed = 42
            
        elif dataset_choice == "🏠 Häuserpreise mit Pool (1000 Häuser)":
            # X-Variable als Dropdown (zwei Optionen verfügbar)
            x_variable_options = ["Wohnfläche (sqft/10)", "Pool (0/1)"]
            x_variable = st.selectbox(
                "X-Variable (Prädiktor):",
                x_variable_options,
                index=0,
                help="Einfache Regression: Nur EIN Prädiktor. Pool ist eine Dummy-Variable (0 = kein Pool, 1 = Pool).",
            )
            st.sidebar.markdown("**Stichproben-Info:**")
            st.sidebar.info("n = 1000 Häuser (fixiert)")
            n = 1000
            true_intercept = 0
            true_beta = 0
            noise_level = 0
            seed = 42
            
        else:
            # Default values for other datasets
            n = 12
            true_intercept = 0
            true_beta = 0
            noise_level = 0
            seed = 42
    
    return SimpleRegressionParams(
        n=n,
        true_intercept=true_intercept,
        true_beta=true_beta,
        noise_level=noise_level,
        seed=seed,
        x_variable=x_variable
    )


def render_display_options(has_true_line: bool = False, key_suffix: str = "") -> DisplayOptions:
    """
    Render display options controls.
    
    Args:
        has_true_line: Whether to show the true line option
        key_suffix: Suffix for widget keys to avoid collisions
    
    Returns:
        DisplayOptions object with selected options
    """
    logger.debug(f"Rendering display options (has_true_line={has_true_line})")
    
    st.sidebar.markdown("---")
    with st.sidebar.expander("🔧 Anzeigeoptionen", expanded=False):
        show_formulas = st.checkbox(
            "Formeln anzeigen",
            value=UI_DEFAULTS["show_formulas"],
            help="Zeige mathematische Formeln in der Anleitung",
            key=f"show_formulas{key_suffix}",
        )
        show_true_line = (
            st.checkbox(
                "Wahre Linie zeigen",
                value=UI_DEFAULTS["show_true_line"],
                help="Zeige die wahre Regressionslinie (nur bei Simulation)",
            )
            if has_true_line
            else False
        )
    
    return DisplayOptions(show_formulas=show_formulas, show_true_line=show_true_line)
