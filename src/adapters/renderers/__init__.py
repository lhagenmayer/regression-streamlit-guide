"""
Renderers - Framework-specific interpreters for ContentStructure.

Each renderer takes the same EducationalContent and renders it
using framework-specific calls.

Streamlit: st.markdown(), st.metric(), st.plotly_chart()
Flask: Jinja2 templates, HTML generation
"""

from .streamlit_renderer import StreamlitContentRenderer
from .html_renderer import HTMLContentRenderer

__all__ = ["StreamlitContentRenderer", "HTMLContentRenderer"]
