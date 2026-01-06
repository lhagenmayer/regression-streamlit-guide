#!/usr/bin/env python3
"""
Entry point for the Linear Regression Guide application.

This application is designed to run with Streamlit.

Usage:
    streamlit run run.py

Note: Direct execution with 'python run.py' is not supported.
"""

import sys
import os

# Detect execution context
is_streamlit_execution = any('streamlit' in arg.lower() for arg in sys.argv)

if not is_streamlit_execution:
    print("🚀 Linear Regression Guide")
    print("=" * 30)
    print()
    print("❌ Direct execution not supported.")
    print("   This application requires Streamlit to run.")
    print()
    print("✅ Correct usage:")
    print("   streamlit run run.py")
    print()
    print("📚 For more information, see README.md")
    print()
    sys.exit(1)

# When run through Streamlit, execute the main app
import warnings
import streamlit as st

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

# Import application modules
from src.config import UI_DEFAULTS, get_logger
from src.ui import inject_accessibility_styles
from src.ui import render_r_output_section
from src.utils import initialize_session_state
from src.ui import (
    render_sidebar_header,
    render_dataset_selection,
    render_multiple_regression_params,
    render_simple_regression_params,
    render_display_options,
)
from src.data import (
    load_multiple_regression_data,
    load_simple_regression_data,
    compute_simple_regression_model,
)
from src.ui import (
    render_simple_regression_tab,
    render_multiple_regression_tab,
    render_datasets_tab,
)

# Initialize logger
logger = get_logger(__name__)

# Initialize session state
initialize_session_state()

# Inject accessibility styles
inject_accessibility_styles()

# Main application
def main():
    try:
        # Set page config
        st.set_page_config(
            page_title="🎓 Linear Regression Guide",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state=UI_DEFAULTS["sidebar_expanded"]
        )

        logger.info("Application starting...")

    # Sidebar
    with st.sidebar:
        render_sidebar_header()

        # Dataset selection
        dataset_choice_simple = render_dataset_selection("simple")
        dataset_choice_mult = render_dataset_selection("multiple")

        # Parameters
        simple_params = render_simple_regression_params()
        mult_params = render_multiple_regression_params()

        # Display options
        show_formulas_simple, show_formulas_mult = render_display_options()

    # Main content
    st.title("🎓 Umfassender Leitfaden zur Linearen Regression")
    st.markdown("Von der Frage zur validierten Erkenntnis")

    # Load data
    try:
        simple_data = load_simple_regression_data(
            dataset_choice=dataset_choice_simple,
            **simple_params
        )
        mult_data = load_multiple_regression_data(
            dataset_choice=dataset_choice_mult,
            **mult_params
        )
    except Exception as e:
        st.error(f"❌ Fehler beim Laden der Daten: {e}")
        logger.error(f"Data loading error: {e}")
        return

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📈 Einfache Regression", "📊 Multiple Regression", "📋 Datensätze"])

    # TAB 1: SIMPLE REGRESSION
    with tab1:
        render_simple_regression_tab(
            model_data=simple_data,
            dataset_choice=dataset_choice_simple,
            show_formulas=show_formulas_simple,
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

    # Footer
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

    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"❌ Anwendung-Fehler: {e}")

        # Show error details in development mode
        import traceback
        with st.expander("Fehlerdetails (für Entwickler)"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
