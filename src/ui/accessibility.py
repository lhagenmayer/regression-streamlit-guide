"""
Accessibility utilities for the Linear Regression Guide.

This module provides accessibility enhancements for the Streamlit application.
"""

import streamlit as st


def inject_accessibility_styles() -> None:
    """
    Inject accessibility styles into the Streamlit application.

    This function adds CSS styles to improve accessibility and user experience.
    """
    # Basic accessibility styles
    accessibility_css = """
    <style>
    /* Improve focus indicators */
    .stButton > button:focus,
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus {
        outline: 2px solid #4A90E2 !important;
        outline-offset: 2px !important;
    }

    /* Better contrast for text */
    .stMarkdown p, .stMarkdown li {
        line-height: 1.6;
    }

    /* Improve spacing */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        margin-top: 1.5em;
        margin-bottom: 0.5em;
    }

    /* Better button styling */
    .stButton > button {
        border-radius: 4px;
        font-weight: 500;
    }
    </style>
    """

    st.markdown(accessibility_css, unsafe_allow_html=True)