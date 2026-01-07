"""
Framework Adapters - Frontend-agnostic rendering layer.

Supports:
- Streamlit (interactive web app)
- Flask (traditional web app)

Auto-detection chooses the right framework at runtime.
"""

from .detector import FrameworkDetector, Framework
from .base import BaseRenderer, RenderContext

__all__ = [
    "FrameworkDetector",
    "Framework", 
    "BaseRenderer",
    "RenderContext",
]
