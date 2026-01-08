"""
📊 Linear Regression Guide
==========================

A 4-step pipeline for educational regression analysis:
    1. GET      → Fetch/generate data
    2. CALCULATE → Compute statistics  
    3. PLOT     → Create visualizations
    4. DISPLAY  → Render in UI

Usage:
    streamlit run src/app.py
"""

from .pipeline import RegressionPipeline

__all__ = ["RegressionPipeline"]
__version__ = "2.0.0"
