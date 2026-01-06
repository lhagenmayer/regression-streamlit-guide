"""
Multiple regression tab for the Linear Regression Guide.

This module renders the multiple regression analysis tab.
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, Any

from ...config import COLUMN_LAYOUTS, CAMERA_PRESETS
from ..plots import (
    create_plotly_3d_surface,
    create_plotly_residual_plot,
    create_plotly_bar,
    create_regression_mesh,
)
from ...data import get_multiple_regression_formulas, get_multiple_regression_descriptions
from ...config import get_logger

logger = get_logger(__name__)


def render_multiple_regression_tab(
    model_data: Dict[str, Any],
    dataset_choice: str,
    show_formulas: bool = True
) -> None:
    """
    Render the multiple regression analysis tab.
    
    Args:
        model_data: Dictionary containing model data and statistics
        dataset_choice: Name of selected dataset
        show_formulas: Whether to show mathematical formulas
    """
    st.markdown(
        '<p class="main-header">📊 Leitfaden zur Multiplen Regression</p>', 
        unsafe_allow_html=True
    )
    st.markdown("### Von der einfachen zur multiplen Regression – Mehrere Prädiktoren gleichzeitig")
    
    # Extract variables from model_data
    x2_preis = model_data["x2_preis"]
    x3_werbung = model_data["x3_werbung"]
    y_mult = model_data["y_mult"]
    x1_name = model_data["x1_name"]
    x2_name = model_data["x2_name"]
    y_name = model_data["y_name"]
    model_mult = model_data["model_mult"]
    y_pred_mult = model_data["y_pred_mult"]
    mult_coeffs = model_data["mult_coeffs"]
    mult_summary = model_data["mult_summary"]
    mult_diagnostics = model_data["mult_diagnostics"]
    
    # M1: VON DER LINIE ZUR EBENE
    st.markdown("---")
    st.markdown(
        '<p class="section-header">M1. Von der Linie zur Ebene: Der konzeptionelle Sprung</p>',
        unsafe_allow_html=True,
    )
    
    col_m1_1, col_m1_2 = st.columns(COLUMN_LAYOUTS["moderately_wide"])
    
    with col_m1_1:
        st.markdown(
            """
        Bei der **einfachen linearen Regression** haben wir gesehen, wie eine Gerade den Zusammenhang
        zwischen **einer** unabhängigen Variable X und der abhängigen Variable Y beschreibt.

        In der Praxis hängt aber eine Zielvariable oft von **mehreren Faktoren** ab:
        - Umsatz ← Preis, Werbung, Standort, Saison, ...
        - Gehalt ← Ausbildung, Erfahrung, Branche, ...
        - Aktienkurs ← Zinsen, Inflation, Gewinn, ...

        Die **multiple Regression** erweitert die einfache Regression, um diese Komplexität zu modellieren.
        """
        )
        
        st.info(
            """
        **🔑 Der zentrale Unterschied:**

        | Aspekt | Einfache Regression | Multiple Regression |
        |--------|---------------------|---------------------|
        | **Prädiktoren** | 1 Variable (X) | K Variablen (X₁, X₂, ..., Xₖ) |
        | **Geometrie** | Gerade in 2D | Ebene/Hyperebene in (K+1)D |
        | **Gleichung** | ŷ = b₀ + b₁x | ŷ = b₀ + b₁x₁ + b₂x₂ + ... + bₖxₖ |
        | **Interpretation** | "Pro Einheit X" | "Bei Konstanthaltung der anderen" |
        """
        )
    
    with col_m1_2:
        # 3D Visualisierung: Ebene statt Linie
        try:
            with st.spinner("🎨 Erstelle 3D-Visualisierung..."):
                # Erstelle Mesh für die Ebene using helper function
                X1_mesh, X2_mesh, Y_mesh = create_regression_mesh(
                    x2_preis, x3_werbung, mult_coeffs["params"]
                )
                
                # Create plotly 3D surface plot
                fig_3d_plane = create_plotly_3d_surface(
                    X1_mesh,
                    X2_mesh,
                    Y_mesh,
                    x2_preis,
                    x3_werbung,
                    y_mult,
                    x1_label=x1_name,
                    x2_label=x2_name,
                    y_label=y_name,
                    title="Multiple Regression: Ebene statt Gerade",
                )
                
                fig_3d_plane.update_layout(
                    scene=dict(
                        xaxis_title=x1_name,
                        yaxis_title=x2_name,
                        zaxis_title=y_name,
                        camera=CAMERA_PRESETS["default"],
                    )
                )
                
                st.plotly_chart(fig_3d_plane, key="multiple_regression_3d_plane", use_container_width=True)
        except Exception as e:
            logger.error(f"Error creating 3D visualization: {e}")
            st.warning("⚠️ 3D-Visualisierung konnte nicht erstellt werden.")
    
    # M2: DAS GRUNDMODELL
    st.markdown("---")
    st.markdown(
        '<p class="section-header">M2. Das Grundmodell der Multiplen Regression</p>',
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
    Das multiple Regressionsmodell erweitert die einfache lineare Regression um **K unabhängige Variablen**.
    """
    )
    
    if show_formulas:
        st.markdown("### 📐 Das allgemeine Modell")
        formulas = get_multiple_regression_formulas(dataset_choice)
        st.latex(formulas["general"])
        
        st.markdown(f"### 📊 Unser Beispiel: {dataset_choice}")
        if "specific" in formulas:
            st.latex(formulas["specific"])
    
    col_m2_1, col_m2_2 = st.columns([1, 1])
    
    with col_m2_1:
        st.markdown("### 📋 Modellkomponenten")
        st.markdown(
            """
        | Symbol | Bedeutung | Beispiel |
        |--------|-----------|----------|
        | **yᵢ** | Zielvariable (abhängig) | Umsatz in Stadt i |
        | **xₖᵢ** | k-ter Prädiktor (unabhängig) | Preis, Werbung in Stadt i |
        | **β₀** | Achsenabschnitt (Intercept) | Basis-Umsatz ohne Einflüsse |
        | **βₖ** | Partieller Regressionskoeffizient | Effekt von xₖ **ceteris paribus** |
        | **εᵢ** | Störgrösse | Alle anderen Einflüsse |
        """
        )
        
        st.success(
            f"""
        **🎯 Unser geschätztes Modell:**

        Umsatz = {mult_coeffs["params"][0]:.2f}
                 {mult_coeffs["params"][1]:+.2f} · Preis
                 {mult_coeffs["params"][2]:+.2f} · Werbung
        """
        )
    
    with col_m2_2:
        st.markdown("### 🔬 Partielle Koeffizienten")
        st.markdown(
            f"""
        **β₁ (Preis) = {mult_coeffs["params"][1]:.3f}**

        → Pro CHF Preiserhöhung sinkt der Umsatz um {abs(mult_coeffs["params"][1]):.2f} Tausend CHF,
        **wenn Werbung konstant gehalten wird**.

        **β₂ (Werbung) = {mult_coeffs["params"][2]:.3f}**

        → Pro 1000 CHF mehr Werbung steigt der Umsatz um {mult_coeffs["params"][2]:.2f} Tausend CHF,
        **wenn Preis konstant gehalten wird**.
        """
        )
        
        st.warning(
            """
        **⚠️ Wichtig: Ceteris Paribus**

        Die Interpretation "bei Konstanthaltung der anderen Variablen" ist zentral!

        Anders als bei der einfachen Regression misst βₖ den **isolierten Effekt**
        einer Variable.
        """
        )
    
    # Daten anzeigen
    st.markdown("### 📊 Die Daten")
    df_mult = pd.DataFrame({x1_name: x2_preis, x2_name: x3_werbung, y_name: y_mult})
    st.dataframe(
        df_mult.head(15).style.format(
            {"Preis (CHF)": "{:.2f}", "Werbung (CHF1000)": "{:.2f}", "Umsatz (1000 CHF)": "{:.2f}"}
        ),
        use_container_width=True,
    )
    
    st.markdown("---")
    st.markdown("### 📈 Weitere Analysen")
    st.info("Weitere detaillierte Analysen werden in einer zukünftigen Version verfügbar sein.")
    
    logger.info("Multiple regression tab rendered")
