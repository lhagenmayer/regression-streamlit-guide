"""
Simple Regression Tab - Complete educational module for simple linear regression.

This module orchestrates all sections of the simple regression educational content,
from introduction to advanced diagnostics.
"""

import streamlit as st
from .intro import render_intro
from .data_exploration import render_data_exploration
from .model import render_model
from .estimation import render_estimation
from .evaluation import render_evaluation
from .inference import render_inference
from .anova import render_anova
from .diagnostics import render_diagnostics
from .conclusion import render_conclusion


def render(params: dict) -> None:
    """
    Render the complete simple regression educational module.

    Args:
        params: Dictionary containing all UI parameters from sidebar
    """
    st.header("📈 Einfache Lineare Regression")

    # Educational progression through sections
    render_intro()
    render_data_exploration(params)
    render_model(params)
    render_estimation(params)
    render_evaluation(params)
    render_inference(params)
    render_anova(params)
    render_diagnostics(params)
    render_conclusion()