"""
UI configuration for the Linear Regression Guide.

This module contains page configuration, CSS styles, and UI setup
functions for the Streamlit application.
"""

import streamlit as st
from ..config import get_logger
from .accessibility import inject_accessibility_styles

logger = get_logger(__name__)


def setup_page_config() -> None:
    """
    Configure Streamlit page settings.
    
    Sets up the page title, icon, layout, and initial sidebar state.
    Must be called before any other Streamlit commands.
    """
    logger.debug("Setting up page configuration")
    
    st.set_page_config(
        page_title="📖 Leitfaden Lineare Regression",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    logger.info("Page configuration complete")


def inject_custom_css() -> None:
    """
    Inject custom CSS styles for the application.
    
    Defines styles for headers, sections, and special content boxes.
    """
    logger.debug("Injecting custom CSS")
    
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
    
    logger.info("Custom CSS injected successfully")


def setup_ui() -> None:
    """
    Set up all UI configurations for the application.
    
    This is a convenience function that calls all UI setup functions
    in the correct order. Call this once at application startup.
    """
    logger.info("Setting up UI")
    
    # Page configuration must come first
    setup_page_config()
    
    # Inject custom styles
    inject_custom_css()
    
    # Inject accessibility improvements
    inject_accessibility_styles()
    
    logger.info("UI setup complete")


def render_footer() -> None:
    """
    Render the application footer with credits and information.
    """
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
