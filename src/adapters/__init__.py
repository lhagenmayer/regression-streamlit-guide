"""
Adapters Module - Framework-specific frontends.

This module provides the bridge between our framework-agnostic content
and specific UI frameworks (Streamlit, Flask).

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                        PLATFORM-AGNOSTIC CORE                        │
    │         ContentBuilder → EducationalContent (JSON-serializable)      │
    └─────────────────────────────────────┬───────────────────────────────┘
                                          │
                     ┌────────────────────┴────────────────────┐
                     ↓                                         ↓
    ┌────────────────────────────────┐     ┌────────────────────────────────┐
    │     StreamlitContentRenderer    │     │      HTMLContentRenderer        │
    │        (st.markdown, etc.)      │     │        (HTML/Jinja2)           │
    └────────────────────────────────┘     └────────────────────────────────┘
                     ↓                                         ↓
    ┌────────────────────────────────┐     ┌────────────────────────────────┐
    │         Streamlit App           │     │          Flask App             │
    └────────────────────────────────┘     └────────────────────────────────┘
"""

from .detector import FrameworkDetector, Framework
from .base import BaseRenderer, RenderContext

# Lazy imports for framework-specific components
def get_streamlit_app():
    """Get Streamlit app function."""
    from .streamlit.app import run_streamlit_app
    return run_streamlit_app

def get_flask_app():
    """Get Flask app creator."""
    from .flask_app import create_flask_app, run_flask
    return create_flask_app, run_flask

def get_ai_streamlit_component():
    """Get Streamlit AI interpretation component."""
    from .ai_components import AIInterpretationStreamlit
    return AIInterpretationStreamlit

def get_ai_html_component():
    """Get HTML/Flask AI interpretation component."""
    from .ai_components import AIInterpretationHTML
    return AIInterpretationHTML

__all__ = [
    # Core
    "FrameworkDetector",
    "Framework",
    "BaseRenderer",
    "RenderContext",
    # App factories
    "get_streamlit_app",
    "get_flask_app",
    # AI components
    "get_ai_streamlit_component",
    "get_ai_html_component",
]
