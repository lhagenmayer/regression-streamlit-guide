"""
🎓 Umfassender Leitfaden zur Linearen Regression
=================================================
Ein didaktisches Tool zum Verstehen der einfachen linearen Regression.
Alle Konzepte auf einer Seite mit logischem roten Faden.

Starten mit: streamlit run app.py

This is a refactored version with better code organization.
"""

import warnings
import streamlit as st

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Import from our modules
import sys
import os

# Setup proper Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Project root

# Add both src and project root to path
for path_dir in [current_dir, parent_dir]:
    if path_dir not in sys.path:
        sys.path.insert(0, path_dir)

# Use relative imports (this works with Streamlit)
# For direct execution, the PYTHONPATH setup above should make it work
from .config import UI_DEFAULTS
from .config import get_logger
from .ui import inject_accessibility_styles
from .ui import render_r_output_section
from .utils import initialize_session_state
from .ui import (
    render_sidebar_header,
    render_dataset_selection,
    render_multiple_regression_params,
    render_simple_regression_params,
    render_display_options,
)
from .data import (
    load_multiple_regression_data,
    load_simple_regression_data,
    compute_simple_regression_model,
)
from .ui import (
    render_simple_regression_tab,
    render_multiple_regression_tab,
    render_datasets_tab,
)

# Initialize logger for the app
logger = get_logger(__name__)

# Log application startup
logger.info("Starting Linear Regression Guide application (refactored version)")

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="📖 Leitfaden Lineare Regression",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject accessibility improvements
inject_accessibility_styles()

# ---------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------
initialize_session_state()

# Add warning if there have been multiple errors
if st.session_state.get("error_count", 0) > 3:
    st.warning("⚠️ Es sind mehrere Fehler aufgetreten. Bitte erwägen Sie, die Seite neu zu laden.")
    if st.button("🔄 Seite neu laden und Cache leeren"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .subsection-header {
        font-size: 1.4rem;
        font-weight: bold;
        color: #34495e;
        margin-top: 1.5rem;
    }
    .concept-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
    .formula-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .interpretation-box {
        background-color: #d4edda;
        border: 1px solid #28a745;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# SIDEBAR - PARAMETER CONFIGURATION
# ---------------------------------------------------------
render_sidebar_header()

# Dataset selection
dataset_selection = render_dataset_selection()
dataset_choice = dataset_selection.simple_dataset
dataset_choice_mult = dataset_selection.multiple_dataset

# Multiple regression parameters
mult_params_obj = render_multiple_regression_params(dataset_choice_mult)
n_mult = mult_params_obj.n
noise_mult_level = mult_params_obj.noise_level
seed_mult = mult_params_obj.seed

# ---------------------------------------------------------
# LOAD MULTIPLE REGRESSION DATA
# ---------------------------------------------------------
try:
    mult_data = load_multiple_regression_data(
        dataset_choice_mult, n_mult, noise_mult_level, seed_mult
    )
except Exception as e:
    logger.error(f"Failed to load multiple regression data: {e}")
    st.error("Failed to load multiple regression data")
    st.stop()

# Display options for multiple regression (moved here after data loading)
st.sidebar.markdown("---")
with st.sidebar.expander("🔧 Anzeigeoptionen (Multiple)", expanded=False):
    show_formulas_mult = st.checkbox(
        "Formeln anzeigen",
        value=UI_DEFAULTS["show_formulas"],
        help="Zeige mathematische Formeln in der Anleitung",
        key="show_formulas_mult",
    )

# ---------------------------------------------------------
# SIMPLE REGRESSION PARAMETERS
# ---------------------------------------------------------
# Determine if dataset has true line
has_true_line = (dataset_choice == "🏪 Elektronikmarkt (simuliert)")

# Render simple regression parameters
simple_params_obj = render_simple_regression_params(dataset_choice, has_true_line)
n = simple_params_obj.n
true_intercept = simple_params_obj.true_intercept
true_beta = simple_params_obj.true_beta
noise_level = simple_params_obj.noise_level
seed = simple_params_obj.seed
x_variable = simple_params_obj.x_variable

# Display options for simple regression
display_opts = render_display_options(has_true_line, key_suffix="_simple")
show_formulas = display_opts.show_formulas
show_true_line = display_opts.show_true_line

# App Status Indicator
st.sidebar.markdown("---")
error_count = st.session_state.get("error_count", 0)
if error_count == 0:
    st.sidebar.success("✅ App läuft stabil")
elif error_count <= 2:
    st.sidebar.info(f"ℹ️ {error_count} kleine Fehler aufgetreten")
else:
    st.sidebar.warning(f"⚠️ {error_count} Fehler - erwägen Sie Neuladen")

# ---------------------------------------------------------
# LOAD SIMPLE REGRESSION DATA
# ---------------------------------------------------------
try:
    simple_data = load_simple_regression_data(
        dataset_choice, x_variable, n, true_intercept, true_beta, noise_level, seed
    )
    
    x = simple_data["x"]
    y = simple_data["y"]
    x_label = simple_data["x_label"]
    y_label = simple_data["y_label"]
    x_unit = simple_data.get("x_unit", "")
    y_unit = simple_data.get("y_unit", "")
    context_title = simple_data.get("context_title", "")
    context_description = simple_data.get("context_description", "")
    
except Exception as e:
    logger.error(f"Failed to load simple regression data: {e}")
    st.error("Failed to load simple regression data")
    st.stop()

# ---------------------------------------------------------
# COMPUTE SIMPLE REGRESSION MODEL
# ---------------------------------------------------------
try:
    model_data = compute_simple_regression_model(x, y, x_label, y_label, n)
except Exception as e:
    logger.error(f"Failed to compute simple regression model: {e}")
    st.error("Failed to compute regression model")
    st.stop()

# ---------------------------------------------------------
# R OUTPUT DISPLAY - Always visible above tabs
# ---------------------------------------------------------
try:
    render_r_output_section(
        model=st.session_state.get("current_model"),
        feature_names=st.session_state.get("current_feature_names"),
        figsize=(18, 13)
    )
except Exception as e:
    logger.error(f"Error rendering R output: {e}")
    st.warning("⚠️ R-Ausgabe konnte nicht dargestellt werden.")
    st.info("Die Regression wurde trotzdem berechnet und kann in den Tabs eingesehen werden.")

# ---------------------------------------------------------
# MAIN CONTENT - Tab-based Navigation
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📈 Einfache Regression", "📊 Multiple Regression", "📚 Datensätze"])

# TAB 1: SIMPLE REGRESSION
# Note: For now, we'll use the refactored tab module
# In a future iteration, the full detailed content from the original app.py
# can be moved into the simple_regression.py module
with tab1:
    render_simple_regression_tab(
        model_data=model_data,
        x_label=x_label,
        y_label=y_label,
        x_unit=x_unit,
        y_unit=y_unit,
        context_title=context_title,
        context_description=context_description,
        show_formulas=show_formulas,
        show_true_line=show_true_line,
        has_true_line=has_true_line,
        true_intercept=true_intercept,
        true_beta=true_beta,
    )

# TAB 2: MULTIPLE REGRESSION
with tab2:
    render_multiple_regression_tab(
        model_data=mult_data,
        dataset_choice=dataset_choice_mult,
        show_formulas=show_formulas_mult,
    )

# TAB 3: DATASETS
with tab3:
    render_datasets_tab()

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: gray; font-size: 12px; padding: 20px;'>
    📖 Umfassender Leitfaden zur Linearen Regression |
    Von der Frage zur validierten Erkenntnis |
    Erstellt mit Streamlit & statsmodels
</div>
""",
    unsafe_allow_html=True,
)

logger.info("Application rendering complete")
