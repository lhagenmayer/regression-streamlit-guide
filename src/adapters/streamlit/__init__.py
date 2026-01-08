"""
Streamlit Adapter Module.

Provides Streamlit-specific rendering using the universal ContentBuilder.
"""

from .app import run_streamlit_app

__all__ = ["run_streamlit_app"]
