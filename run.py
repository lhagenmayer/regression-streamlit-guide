#!/usr/bin/env python3
"""
Entry point for the Linear Regression Guide application.

This application is designed to run with Streamlit.
"""

import sys
import os
import warnings
import streamlit as st

# Detect execution context
def _is_running_with_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except (ImportError, AttributeError):
        return any('streamlit' in arg.lower() for arg in sys.argv)

if __name__ == "__main__" and not _is_running_with_streamlit():
    print("❌ Direct execution not supported. Use: streamlit run run.py")
    sys.exit(1)

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

# Import application modules
from src.config import UI_DEFAULTS, get_logger
from src.ui import inject_accessibility_styles, render_r_output_section
from src.utils import initialize_session_state
from src.ui import (
    render_sidebar_header,
    render_dataset_selection,
    render_multiple_regression_params,
    render_simple_regression_params,
    render_display_options,
    render_simple_regression_tab,
    render_multiple_regression_tab,
    render_datasets_tab,
)
from src.data import (
    load_multiple_regression_data,
    load_simple_regression_data,
)

logger = get_logger(__name__)

def main():
    # 1. MUST BE THE FIRST STREAMLIT COMMAND
    st.set_page_config(
        page_title="🎓 Linear Regression Guide",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state=UI_DEFAULTS["sidebar_expanded"]
    )

    # 2. Initializations after page config
    initialize_session_state()
    inject_accessibility_styles()

    logger.info("Application starting...")

    # 3. Sidebar
    with st.sidebar:
        render_sidebar_header()
        dataset_selection = render_dataset_selection()
        
        mult_params = render_multiple_regression_params(dataset_selection.multiple_dataset)
        
        has_true_line = (dataset_selection.simple_dataset == "🏪 Elektronikmarkt (simuliert)")
        simple_params = render_simple_regression_params(dataset_selection.simple_dataset, has_true_line)
        
        display_opts_simple = render_display_options(has_true_line, key_suffix="_simple")
        display_opts_mult = render_display_options(has_true_line=False, key_suffix="_mult")

    # 4. Main content title
    st.title("🎓 Umfassender Leitfaden zur Linearen Regression")
    st.markdown("Von der Frage zur validierten Erkenntnis")

    # 5. Load data with error handling
    try:
        simple_data = load_simple_regression_data(
            dataset_choice=dataset_selection.simple_dataset,
            x_variable=simple_params.x_variable,
            n=simple_params.n,
            true_intercept=simple_params.true_intercept,
            true_beta=simple_params.true_beta,
            noise_level=simple_params.noise_level,
            seed=simple_params.seed
        )
        mult_data = load_multiple_regression_data(
            dataset_choice=dataset_selection.multiple_dataset,
            n=mult_params.n,
            noise_level=mult_params.noise_level,
            seed=mult_params.seed
        )
    except Exception as e:
        st.error(f"❌ Fehler beim Laden der Daten: {e}")
        logger.error(f"Data loading error: {e}")
        return

    # 6. Global R output section
    try:
        from src.utils import update_current_model
        update_current_model(simple_data['model'], [simple_params.x_variable or "X"])
    except Exception as e:
        logger.warning(f"Could not update model state: {e}")

    render_r_output_section(
        model=st.session_state.get("current_model"),
        feature_names=st.session_state.get("current_feature_names"),
        figsize=(18, 13)
    )

    # 7. Render tabs
    tab1, tab2, tab3 = st.tabs(["📈 Einfache Regression", "📊 Multiple Regression", "📋 Datensätze"])

    with tab1:
        render_simple_regression_tab(
            model_data=simple_data,
            x_label=simple_data.get('x_label', 'X'),
            y_label=simple_data.get('y_label', 'Y'),
            x_unit=simple_data.get('x_unit', ''),
            y_unit=simple_data.get('y_unit', ''),
            context_title=simple_data.get('context_title', ''),
            context_description=simple_data.get('context_description', ''),
            show_formulas=display_opts_simple.show_formulas,
            show_true_line=display_opts_simple.show_true_line,
            has_true_line=has_true_line,
            true_intercept=simple_params.true_intercept,
            true_beta=simple_params.true_beta,
        )

    with tab2:
        render_multiple_regression_tab(
            model_data=mult_data,
            dataset_choice=dataset_selection.multiple_dataset,
            show_formulas=display_opts_mult.show_formulas,
        )

    with tab3:
        render_datasets_tab()

    # 8. Footer
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: gray; font-size: 12px;'>📖 Linear Regression Guide | statsmodels & Streamlit</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Kritischer Fehler: {e}")
        import traceback
        st.expander("Details").code(traceback.format_exc())
