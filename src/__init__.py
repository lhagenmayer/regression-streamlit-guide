"""
📊 Linear Regression Guide
==========================

Clean Architecture educational app:
    - core/domain      → Pure Python entities & interfaces
    - core/application → Use Cases & DTOs
    - infrastructure   → Data fetching, calculations, plots
    - api              → REST endpoints
    - adapters         → Flask, Streamlit UI

Usage:
    streamlit run src/app.py
"""

from .infrastructure import RegressionPipeline

__all__ = ["RegressionPipeline"]
__version__ = "3.0.0"
