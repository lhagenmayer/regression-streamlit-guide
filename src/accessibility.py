"""
Accessibility utilities for the Linear Regression Guide.

This module provides functions to enhance accessibility including ARIA labels,
keyboard navigation, and screen reader support.
"""

import streamlit as st
from typing import Optional


def inject_accessibility_styles():
    """
    Inject CSS for improved accessibility including focus indicators,
    color contrast, and responsive design.
    """
    st.markdown(
        """
        <style>
        /* Focus indicators for keyboard navigation */
        button:focus,
        input:focus,
        select:focus,
        textarea:focus,
        [role="button"]:focus {
            outline: 3px solid #1f77b4 !important;
            outline-offset: 2px !important;
            box-shadow: 0 0 0 3px rgba(31, 119, 180, 0.3) !important;
        }

        /* High contrast for interactive elements */
        .stButton > button {
            border: 2px solid #1f77b4;
            transition: all 0.2s ease;
        }

        .stButton > button:hover {
            background-color: #1f77b4;
            color: white;
            border-color: #0d3d66;
        }

        /* Better color contrast for text */
        .main-header {
            font-size: 2.8rem;
            font-weight: bold;
            color: #0d3d66;
            text-align: center;
            margin-bottom: 0.5rem;
        }

        .section-header {
            font-size: 1.8rem;
            font-weight: bold;
            color: #0d3d66;
            border-bottom: 3px solid #1f77b4;
            padding-bottom: 0.5rem;
            margin-top: 2rem;
        }

        .subsection-header {
            font-size: 1.4rem;
            font-weight: bold;
            color: #2c3e50;
            margin-top: 1.5rem;
        }

        /* Improved readability for content boxes */
        .concept-box {
            background-color: #f8f9fa;
            border-left: 4px solid #1f77b4;
            padding: 15px;
            margin: 10px 0;
            border-radius: 0 8px 8px 0;
            color: #2c3e50;
        }

        .formula-box {
            background-color: #fff9e6;
            border: 2px solid #f39c12;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            color: #2c3e50;
            font-weight: 500;
        }

        .warning-box {
            background-color: #fff3cd;
            border-left: 4px solid #f39c12;
            padding: 15px;
            margin: 10px 0;
            border-radius: 0 8px 8px 0;
            color: #856404;
        }

        .success-box {
            background-color: #d4edda;
            border-left: 4px solid #2ecc71;
            padding: 15px;
            margin: 10px 0;
            border-radius: 0 8px 8px 0;
            color: #155724;
        }

        .info-box {
            background-color: #d1ecf1;
            border-left: 4px solid #17a2b8;
            padding: 15px;
            margin: 10px 0;
            border-radius: 0 8px 8px 0;
            color: #0c5460;
        }

        /* Responsive design for mobile */
        @media (max-width: 768px) {
            .main-header {
                font-size: 2rem;
            }
            .section-header {
                font-size: 1.5rem;
            }
            .subsection-header {
                font-size: 1.2rem;
            }
        }

        /* Loading indicator styling */
        .stSpinner {
            text-align: center;
            color: #1f77b4;
        }

        /* Better table contrast */
        .stDataFrame {
            border: 1px solid #dee2e6;
        }

        /* Tab navigation improvements */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 12px 24px;
            background-color: #f8f9fa;
            border-radius: 8px 8px 0 0;
            border: 2px solid transparent;
        }

        .stTabs [aria-selected="true"] {
            background-color: #e8f4fd;
            border-color: #1f77b4;
            font-weight: 600;
        }

        /* Slider improvements */
        .stSlider [role="slider"] {
            background-color: #1f77b4;
        }

        .stSlider [role="slider"]:focus {
            box-shadow: 0 0 0 3px rgba(31, 119, 180, 0.3);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def add_aria_label(element_type: str, label: str, description: Optional[str] = None) -> str:
    """
    Generate ARIA attributes for better screen reader support.

    Args:
        element_type: Type of element (button, slider, select, etc.)
        label: Accessible label for the element
        description: Optional longer description

    Returns:
        HTML string with ARIA attributes
    """
    aria_attrs = f'aria-label="{label}"'
    if description:
        aria_attrs += f' aria-describedby="desc-{label.replace(" ", "-")}"'
    return aria_attrs


def create_screen_reader_text(text: str, element_id: Optional[str] = None):
    """
    Create screen reader only text for additional context.

    Args:
        text: Text for screen readers
        element_id: Optional ID for the element
    """
    id_attr = f'id="{element_id}"' if element_id else ""
    st.markdown(
        f"""
        <span {id_attr} style="
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0,0,0,0);
            white-space: nowrap;
            border-width: 0;
        ">{text}</span>
        """,
        unsafe_allow_html=True,
    )


def add_plot_accessibility_description(
    plot_type: str, title: str, x_label: str, y_label: str, key_insights: Optional[str] = None
):
    """
    Add accessible description for plots and charts.

    Args:
        plot_type: Type of plot (scatter, line, 3D, etc.)
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        key_insights: Optional key insights from the plot
    """
    description = f"{plot_type} chart titled '{title}'. "
    description += f"X-axis shows {x_label}, Y-axis shows {y_label}. "
    if key_insights:
        description += f"Key insight: {key_insights}"

    create_screen_reader_text(description)


def show_loading_indicator(message: str = "Loading..."):
    """
    Show an accessible loading indicator.

    Args:
        message: Loading message to display
    """
    return st.spinner(message)


def create_skip_link(target_id: str, label: str = "Skip to main content"):
    """
    Create a skip link for keyboard navigation.

    Args:
        target_id: ID of the target element
        label: Text for the skip link
    """
    st.markdown(
        f"""
        <a href="#{target_id}" style="
            position: absolute;
            top: -40px;
            left: 0;
            background: #1f77b4;
            color: white;
            padding: 8px;
            text-decoration: none;
            z-index: 100;
        " onfocus="this.style.top='0'" onblur="this.style.top='-40px'">{label}</a>
        """,
        unsafe_allow_html=True,
    )


def create_accessible_button(label: str, key: str, help_text: Optional[str] = None) -> bool:
    """
    Create an accessible button with proper ARIA attributes.

    Args:
        label: Button label
        key: Unique key for the button
        help_text: Optional help text

    Returns:
        True if button was clicked
    """
    if help_text:
        create_screen_reader_text(help_text, element_id=f"help-{key}")
    return st.button(label, key=key, help=help_text)


def create_accessible_slider(
    label: str,
    min_value: float,
    max_value: float,
    value: float,
    step: float,
    key: str,
    help_text: Optional[str] = None,
    format_str: str = "%f",
):
    """
    Create an accessible slider with proper labels and help text.

    Args:
        label: Slider label
        min_value: Minimum value
        max_value: Maximum value
        value: Default value
        step: Step size
        key: Unique key
        help_text: Optional help text
        format_str: Format string for display

    Returns:
        Selected value
    """
    # Add context for screen readers
    description = f"{label}. Range from {min_value} to {max_value}, step {step}."
    if help_text:
        description += f" {help_text}"
    create_screen_reader_text(description, element_id=f"desc-{key}")

    return st.slider(
        label,
        min_value=min_value,
        max_value=max_value,
        value=value,
        step=step,
        key=key,
        help=help_text,
        format=format_str,
    )


def create_accessible_selectbox(
    label: str,
    options: list,
    index: int = 0,
    key: Optional[str] = None,
    help_text: Optional[str] = None,
):
    """
    Create an accessible selectbox with proper labels.

    Args:
        label: Selectbox label
        options: List of options
        index: Default index
        key: Unique key
        help_text: Optional help text

    Returns:
        Selected option
    """
    description = f"{label}. {len(options)} options available."
    if help_text:
        description += f" {help_text}"
    if key:
        create_screen_reader_text(description, element_id=f"desc-{key}")

    return st.selectbox(label, options, index=index, key=key, help=help_text)


def announce_to_screen_reader(message: str):
    """
    Announce a message to screen readers using ARIA live region.

    Args:
        message: Message to announce
    """
    st.markdown(
        f"""
        <div role="status" aria-live="polite" aria-atomic="true" style="
            position: absolute;
            left: -10000px;
            width: 1px;
            height: 1px;
            overflow: hidden;
        ">{message}</div>
        """,
        unsafe_allow_html=True,
    )
