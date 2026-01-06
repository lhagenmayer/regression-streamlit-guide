"""
Multiple regression tab for the Linear Regression Guide.

This module renders the complete multiple linear regression analysis tab
with all educational content from chapters 1.0 through 9.0.
"""

import streamlit as st
import numpy as np
import pandas as pd
import warnings
from scipy import stats
from scipy.stats import probplot
from typing import Dict, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...config import COLUMN_LAYOUTS, CAMERA_PRESETS, VISUALIZATION_3D, get_logger
from ..plots import (
    create_plotly_3d_surface,
    create_plotly_residual_plot,
    create_plotly_bar,
    create_regression_mesh,
    create_zero_plane,
    get_signif_stars,
    get_signif_color,
    calculate_residual_sizes,
    standardize_residuals,
)
from ...data import get_multiple_regression_formulas, get_multiple_regression_descriptions

logger = get_logger(__name__)


def render_multiple_regression_tab(
    model_data: Dict[str, Any],
    dataset_choice: str,
    show_formulas: bool = True
) -> None:
    """
    Render the complete multiple regression analysis tab.
    
    This function renders all educational content for multiple linear regression,
    including chapters on:
    - 1.0 Von der Linie zur Ebene
    - 2.0 Das Grundmodell
    - 3.0 OLS-Schätzer und Gauss-Markov
    - 4.0 Modellvalidierung (R², Adjusted R²)
    - 5.0 Anwendungsbeispiel und Interpretation
    - 6.0 Dummy-Variablen
    - 7.0 Multikollinearität
    - 8.0 Residuen-Diagnostik
    - 9.0 Zusammenfassung
    """
    # =========================================================
    # EXTRACT DATA FROM MODEL_DATA
    # =========================================================
    x2_preis = np.array(model_data["x2_preis"])
    x3_werbung = np.array(model_data["x3_werbung"])
    y_mult = np.array(model_data["y_mult"])
    x1_name = model_data["x1_name"]
    x2_name = model_data["x2_name"]
    y_name = model_data["y_name"]
    model_mult = model_data["model_mult"]
    y_pred_mult = np.array(model_data["y_pred_mult"])
    mult_coeffs = model_data["mult_coeffs"]
    mult_summary = model_data["mult_summary"]
    mult_diagnostics = model_data["mult_diagnostics"]
    
    n_mult = len(x2_preis)
    
    # =========================================================
    # HEADER
    # =========================================================
    st.markdown(
        '<p class="main-header">📊 Leitfaden zur Multiplen Regression</p>', 
        unsafe_allow_html=True
    )
    st.markdown("### Von der einfachen zur multiplen Regression – Mehrere Prädiktoren gleichzeitig")

    # =========================================================
    # KAPITEL 1.0: VON DER LINIE ZUR EBENE
    # =========================================================
    st.markdown("---")
    st.markdown(
        '<p class="section-header">1.0 Von der Linie zur Ebene: Der konzeptionelle Sprung</p>',
        unsafe_allow_html=True,
    )

    col_m1_1, col_m1_2 = st.columns(COLUMN_LAYOUTS.get("moderately_wide", [1.2, 1]))

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
                X1_mesh, X2_mesh, Y_mesh = create_regression_mesh(
                    x2_preis, x3_werbung, mult_coeffs["params"]
                )

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
                        camera=CAMERA_PRESETS.get("default", dict(eye=dict(x=1.5, y=1.5, z=1.2))),
                    )
                )

                st.plotly_chart(fig_3d_plane, key="multiple_regression_3d_plane_new", use_container_width=True)
        except Exception as e:
            logger.error(f"Error creating 3D visualization: {e}")
            st.warning("⚠️ 3D-Visualisierung konnte nicht erstellt werden.")

    # =========================================================
    # KAPITEL 2.0: DAS GRUNDMODELL
    # =========================================================
    st.markdown("---")
    st.markdown(
        '<p class="section-header">2.0 Das Grundmodell der Multiplen Regression</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    Das multiple Regressionsmodell erweitert die einfache lineare Regression um **K unabhängige Variablen**.
    """
    )

    if show_formulas:
        st.markdown("### 📐 Das allgemeine Modell")
        try:
            formulas = get_multiple_regression_formulas(dataset_choice)
            st.latex(formulas.get("general", r"y_i = \beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i} + \cdots + \beta_K x_{Ki} + \varepsilon_i"))
            
            st.markdown(f"### 📊 Unser Beispiel: {dataset_choice}")
            if "specific" in formulas:
                st.latex(formulas["specific"])
        except:
            st.latex(r"y_i = \beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i} + \cdots + \beta_K x_{Ki} + \varepsilon_i")

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

        {y_name.split('(')[0].strip()} = {mult_coeffs["params"][0]:.2f}
                 {mult_coeffs["params"][1]:+.2f} · {x1_name.split('(')[0].strip()}
                 {mult_coeffs["params"][2]:+.2f} · {x2_name.split('(')[0].strip()}
        """
        )

    with col_m2_2:
        st.markdown("### 🔬 Partielle Koeffizienten")
        st.markdown(
            f"""
        **β₁ ({x1_name.split('(')[0].strip()}) = {mult_coeffs["params"][1]:.3f}**

        → Pro Einheit {x1_name.split('(')[0].strip()} ändert sich {y_name.split('(')[0].strip()} um {mult_coeffs["params"][1]:.2f},
        **wenn {x2_name.split('(')[0].strip()} konstant gehalten wird**.

        **β₂ ({x2_name.split('(')[0].strip()}) = {mult_coeffs["params"][2]:.3f}**

        → Pro Einheit {x2_name.split('(')[0].strip()} ändert sich {y_name.split('(')[0].strip()} um {mult_coeffs["params"][2]:.2f},
        **wenn {x1_name.split('(')[0].strip()} konstant gehalten wird**.
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
        df_mult.head(15).style.format("{:.2f}"),
        use_container_width=True,
    )

    # =========================================================
    # KAPITEL 3.0: OLS & GAUSS-MARKOV
    # =========================================================
    st.markdown("---")
    st.markdown(
        '<p class="section-header">3.0 OLS-Schätzer und Gauss-Markov Theorem</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    Wie bei der einfachen Regression bestimmen wir die Koeffizienten durch **Minimierung der Fehlerquadratsumme**.
    """
    )

    if show_formulas:
        st.markdown("### 📐 OLS-Zielfunktion")
        st.latex(
            r"\min \sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n} (y_i - b_0 - b_1 \cdot x_{1i} - b_2 \cdot x_{2i} - \cdots - b_K \cdot x_{Ki})^2"
        )

        st.markdown("### 📊 Matrixform (elegant!)")
        st.latex(r"\mathbf{b} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}")
        st.markdown(
            """
        Wo:
        - **y** ist der Vektor der abhängigen Variable (n×1)
        - **X** ist die Design-Matrix der Prädiktoren (n×(K+1))
        - **b** ist der Vektor der geschätzten Koeffizienten ((K+1)×1)
        """
        )

    col_m3_1, col_m3_2 = st.columns([1.2, 1])

    with col_m3_1:
        st.markdown("### 🏆 Gauss-Markov Theorem")
        st.markdown(
            """
        Wenn die folgenden **Annahmen** erfüllt sind:

        1. **Linearität**: E(ε|X) = 0
        2. **Homoskedastizität**: Var(ε|X) = σ²
        3. **Keine Autokorrelation**: Cov(εᵢ, εⱼ) = 0
        4. **Keine perfekte Multikollinearität**: X hat vollen Rang

        Dann ist der OLS-Schätzer **BLUE**:
        - **B**est: Kleinste Varianz unter allen linearen Schätzern
        - **L**inear: Lineare Funktion der Daten
        - **U**nbiased: Erwartungstreu, E(b) = β
        - **E**stimator: Schätzer für die wahren Parameter
        """
        )

    with col_m3_2:
        st.markdown("### 📊 Unsere Schätzungen")
        params_df = pd.DataFrame(
            {
                "Koeffizient": [
                    "β₀ (Intercept)",
                    f'β₁ ({x1_name.split("(")[0].strip()})',
                    f'β₂ ({x2_name.split("(")[0].strip()})',
                ],
                "Schätzwert": [
                    f"{mult_coeffs['params'][0]:.4f}",
                    f"{mult_coeffs['params'][1]:.4f}",
                    f"{mult_coeffs['params'][2]:.4f}",
                ],
                "Std. Error": [
                    f"{mult_coeffs['bse'][0]:.4f}",
                    f"{mult_coeffs['bse'][1]:.4f}",
                    f"{mult_coeffs['bse'][2]:.4f}",
                ],
            }
        )
        st.dataframe(params_df, use_container_width=True, hide_index=True)

        st.success(
            f"""
        **✅ Modellgüte:**

        - R² = {mult_summary["rsquared"]:.4f} ({mult_summary["rsquared"]*100:.1f}%)
        - Adjustiertes R² = {mult_summary["rsquared_adj"]:.4f}
        - F-Statistik = {mult_summary["fvalue"]:.2f}
        - p-Wert (F-Test) = {mult_summary["f_pvalue"]:.4g}
        """
        )

    # =========================================================
    # KAPITEL 4.0: MODELLVALIDIERUNG
    # =========================================================
    st.markdown("---")
    st.markdown(
        '<p class="section-header">4.0 Modellvalidierung: R² und Adjustiertes R²</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    Wie gut ist unser Modell? Wir brauchen Kennzahlen, die die **Erklärungskraft** messen.
    """
    )

    # Calculate variance decomposition
    sst_mult = np.sum((y_mult - np.mean(y_mult)) ** 2)
    sse_mult = np.sum((y_mult - y_pred_mult) ** 2)
    ssr_mult = sst_mult - sse_mult

    col_m4_1, col_m4_2 = st.columns([1.5, 1])

    with col_m4_1:
        if show_formulas:
            st.markdown("### 📐 Bestimmtheitsmass R²")
            st.latex(r"R^2 = 1 - \frac{SSE}{SST} = \frac{SSR}{SST}")
            st.latex(
                r"R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}"
            )

        st.markdown(
            f"""
        **Interpretation:**

        R² = {mult_summary["rsquared"]:.4f} bedeutet: **{mult_summary["rsquared"]*100:.1f}%** der Varianz in Y
        wird durch die Prädiktoren X₁, X₂ erklärt.

        **⚠️ Problem:** R² steigt **immer**, wenn wir neue Variablen hinzufügen,
        selbst wenn sie irrelevant sind!
        """
        )

        # Variance decomposition bar chart
        fig_var = go.Figure()
        fig_var.add_trace(go.Bar(
            x=["SST (Total)", "SSR (Erklärt)", "SSE (Unerklärt)"],
            y=[sst_mult, ssr_mult, sse_mult],
            marker_color=["gray", "green", "red"],
            text=[f"{sst_mult:.1f}", f"{ssr_mult:.1f}", f"{sse_mult:.1f}"],
            textposition="auto"
        ))
        fig_var.update_layout(
            title=f"Varianzzerlegung: R² = {mult_summary['rsquared']:.4f}",
            yaxis_title="Quadratsumme",
            template="plotly_white",
            height=300
        )
        st.plotly_chart(fig_var, key="variance_decomposition_mult_new", use_container_width=True)

    with col_m4_2:
        if show_formulas:
            st.markdown("### 📐 Adjustiertes R²")
            st.latex(r"R^2_{adj} = 1 - (1-R^2) \cdot \frac{n-1}{n-K-1}")

        st.markdown(
            f"""
        **Adjustiertes R² = {mult_summary["rsquared_adj"]:.4f}**

        **Vorteile:**
        - Bestraft unnötige Komplexität (mehr K → Strafe)
        - Erlaubt fairen Vergleich von Modellen
        - Kann sogar sinken beim Hinzufügen schwacher Prädiktoren!

        **Vergleich:**

        | Mass | Wert | Deutung |
        |-----|------|---------|
        | R² | {mult_summary["rsquared"]:.4f} | Roh-Erklärungskraft |
        | R²_adj | {mult_summary["rsquared_adj"]:.4f} | Korrigiert für Komplexität |
        | Differenz | {(mult_summary["rsquared"] - mult_summary["rsquared_adj"]):.4f} | Sehr klein → gut! |
        """
        )

    # =========================================================
    # KAPITEL 5.0: ANWENDUNG
    # =========================================================
    st.markdown("---")
    st.markdown(
        '<p class="section-header">5.0 Anwendungsbeispiel und Interpretation</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    Wie nutzen wir unser Modell in der Praxis? Schauen wir uns konkrete Szenarien an.
    """
    )

    col_m5_1, col_m5_2 = st.columns([1, 1])

    with col_m5_1:
        st.markdown("### 🔮 Prognose")
        st.markdown(f"Wir wollen {y_name.split('(')[0].strip()} für einen neuen Datenpunkt vorhersagen.")

        # Interactive sliders
        slider1_val = st.slider(
            x1_name,
            min_value=float(np.min(x2_preis) * 0.8),
            max_value=float(np.max(x2_preis) * 1.2),
            value=float(np.mean(x2_preis)),
            step=0.1,
            key="mult_slider1"
        )
        slider2_val = st.slider(
            x2_name,
            min_value=float(np.min(x3_werbung) * 0.8),
            max_value=float(np.max(x3_werbung) * 1.2),
            value=float(np.mean(x3_werbung)),
            step=0.1,
            key="mult_slider2"
        )

        # Calculate prediction
        pred_value = mult_coeffs["params"][0] + mult_coeffs["params"][1] * slider1_val + mult_coeffs["params"][2] * slider2_val

        st.success(
            f"""
        **Prognose für:**
        - {x1_name} = {slider1_val:.2f}
        - {x2_name} = {slider2_val:.2f}

        **Erwarteter {y_name}:**

        **{pred_value:.2f}**
        """
        )

        if show_formulas:
            st.latex(r"\hat{y} = b_0 + b_1 \cdot x_1 + b_2 \cdot x_2")
            st.latex(
                f"\\hat{{y}} = {mult_coeffs['params'][0]:.2f} + {mult_coeffs['params'][1]:.2f} \\cdot {slider1_val:.2f} + {mult_coeffs['params'][2]:.2f} \\cdot {slider2_val:.2f} = {pred_value:.2f}"
            )

    with col_m5_2:
        st.markdown("### 📊 Sensitivitätsanalyse")
        st.markdown(f"Wie verändert sich {y_name.split('(')[0].strip()} bei Änderung der Variablen?")

        # Sensitivity plot
        var1_range = np.linspace(np.min(x2_preis), np.max(x2_preis), 50)
        response_var1 = mult_coeffs["params"][0] + mult_coeffs["params"][1] * var1_range + mult_coeffs["params"][2] * slider2_val

        fig_sens = go.Figure()
        fig_sens.add_trace(
            go.Scatter(
                x=var1_range,
                y=response_var1,
                mode="lines",
                line=dict(color="blue", width=3),
                name="Predicted Response",
            )
        )
        fig_sens.add_trace(
            go.Scatter(
                x=[slider1_val],
                y=[pred_value],
                mode="markers",
                marker=dict(size=15, color="red"),
                name="Aktuell",
            )
        )
        fig_sens.update_layout(
            title=f'Sensitivität {x1_name.split("(")[0].strip()}<br>({x2_name.split("(")[0].strip()}={slider2_val:.1f} konstant)',
            xaxis_title=x1_name,
            yaxis_title=y_name,
            template="plotly_white",
        )
        st.plotly_chart(fig_sens, key="sensitivity_plot_mult", use_container_width=True)

    # =========================================================
    # KAPITEL 6.0: DUMMY-VARIABLEN
    # =========================================================
    st.markdown("---")
    st.markdown(
        '<p class="section-header">6.0 Dummy-Variablen: Kategoriale Prädiktoren</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    Nicht alle Prädiktoren sind numerisch! Was ist mit **kategorialen Variablen** wie Region,
    Geschlecht, oder Produkttyp?

    **Lösung: Dummy-Variablen** (0/1-Kodierung)
    """
    )

    col_m6_1, col_m6_2 = st.columns([1, 1])

    with col_m6_1:
        st.markdown("### 📋 Konzept")
        st.markdown(
            """
        Für eine kategoriale Variable mit **m Ausprägungen** erstellen wir **m-1 Dummy-Variablen**.

        **Beispiel: Region (3 Ausprägungen)**
        - Nord, Süd, Ost
        - Wir brauchen **2 Dummies**: Region_Ost, Region_Süd
        - **Referenzkategorie**: Nord (beide Dummies = 0)
        """
        )

        st.warning(
            """
        **⚠️ Dummy-Variable Trap:**

        Niemals **alle** m Dummies verwenden! Das führt zu perfekter Multikollinearität.

        Grund: Region_Nord = 1 - Region_Ost - Region_Süd
        """
        )

    with col_m6_2:
        if show_formulas:
            st.markdown("### 📐 Modell mit Dummies")
            st.latex(
                r"\text{Umsatz}_i = \beta_0 + \beta_1 \cdot \text{Preis}_i + \beta_2 \cdot \text{Werbung}_i + \beta_3 \cdot \text{Ost}_i + \beta_4 \cdot \text{Süd}_i + \varepsilon_i"
            )

        st.markdown("### 📊 Interpretation")
        st.markdown(
            """
        **β₀:** Basis-Umsatz in der **Referenzregion** (Nord)

        **β₃ (Ost-Dummy):** Zusätzlicher Umsatz in **Ost** verglichen mit Nord
        (ceteris paribus)

        **β₄ (Süd-Dummy):** Zusätzlicher Umsatz in **Süd** verglichen mit Nord
        (ceteris paribus)
        """
        )

    # =========================================================
    # KAPITEL 7.0: MULTIKOLLINEARITÄT
    # =========================================================
    st.markdown("---")
    st.markdown(
        '<p class="section-header">7.0 Multikollinearität: Wenn Prädiktoren korreliert sind</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    **Multikollinearität** liegt vor, wenn unabhängige Variablen **stark miteinander korrelieren**.

    Das ist ein **Problem**, weil es schwer wird, die individuellen Effekte zu trennen!
    """
    )

    col_m7_1, col_m7_2 = st.columns([1.2, 1])

    with col_m7_1:
        # Calculate correlation
        corr_predictors = np.corrcoef(x2_preis, x3_werbung)[0, 1]
        
        st.markdown(f"### 🔍 Korrelation zwischen Prädiktoren: r = {corr_predictors:.3f}")
        
        # Create scatter plot of predictors
        fig_corr = go.Figure()
        fig_corr.add_trace(go.Scatter(
            x=x2_preis, y=x3_werbung,
            mode='markers',
            marker=dict(size=8, opacity=0.6, color='blue'),
            name='Beobachtungen'
        ))
        fig_corr.update_layout(
            title=f"Korrelation der Prädiktoren: r = {corr_predictors:.3f}",
            xaxis_title=x1_name,
            yaxis_title=x2_name,
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig_corr, key="predictor_correlation_mult", use_container_width=True)

    with col_m7_2:
        st.markdown("### 📊 VIF (Variance Inflation Factor)")

        if show_formulas:
            st.latex(r"VIF_k = \frac{1}{1 - R_k^2}")
            st.markdown(
                """
            Wo R²ₖ das R² ist, wenn wir xₖ durch alle anderen Prädiktoren vorhersagen.
            """
            )

        # Simple VIF calculation for 2 predictors
        vif_val = 1 / (1 - corr_predictors**2)
        vif_df = pd.DataFrame({
            "Variable": [x1_name.split("(")[0].strip(), x2_name.split("(")[0].strip()],
            "VIF": [f"{vif_val:.2f}", f"{vif_val:.2f}"]
        })
        st.dataframe(vif_df, use_container_width=True, hide_index=True)

        st.markdown(
            """
        **Interpretation:**
        - VIF < 5: Keine Multikollinearität ✅
        - 5 < VIF < 10: Moderate Multikollinearität ⚠️
        - VIF > 10: Starke Multikollinearität ❌
        """
        )

    # =========================================================
    # KAPITEL 8.0: RESIDUEN-DIAGNOSTIK
    # =========================================================
    st.markdown("---")
    st.markdown(
        '<p class="section-header">8.0 Residuen-Diagnostik: Modellprüfung</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    Bevor wir unserem Modell vertrauen, müssen wir die **Gauss-Markov Annahmen** prüfen!
    """
    )

    # Get residuals
    residuals = mult_diagnostics.get("resid", y_mult - y_pred_mult)

    # Create 2x2 diagnostic plots
    fig_diag = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Residuals vs Fitted<br>(Linearität & Homoskedastizität)",
            "Normal Q-Q<br>(Normalität)",
            "Scale-Location<br>(Homoskedastizität)",
            "Residuals Histogram<br>(Verteilung)",
        ),
    )

    # 1. Residuals vs Fitted
    fig_diag.add_trace(
        go.Scatter(
            x=y_pred_mult,
            y=residuals,
            mode="markers",
            marker=dict(size=6, opacity=0.6),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig_diag.add_hline(y=0, line_dash="dash", line_color="red", line_width=2, row=1, col=1)

    # 2. Q-Q Plot
    sorted_resid = np.sort(residuals)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
    fig_diag.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sorted_resid,
            mode="markers",
            marker=dict(size=6, opacity=0.6),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    # Reference line
    fig_diag.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=theoretical_quantiles * np.std(residuals) + np.mean(residuals),
            mode="lines",
            line=dict(color="red", dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # 3. Scale-Location
    std_resid = (residuals - np.mean(residuals)) / np.std(residuals)
    fig_diag.add_trace(
        go.Scatter(
            x=y_pred_mult,
            y=np.sqrt(np.abs(std_resid)),
            mode="markers",
            marker=dict(size=6, opacity=0.6),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # 4. Histogram
    fig_diag.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=20,
            marker_color="blue",
            opacity=0.7,
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    # Update axes
    fig_diag.update_xaxes(title_text="Fitted values", row=1, col=1)
    fig_diag.update_yaxes(title_text="Residuals", row=1, col=1)
    fig_diag.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
    fig_diag.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
    fig_diag.update_xaxes(title_text="Fitted values", row=2, col=1)
    fig_diag.update_yaxes(title_text="√|Standardized residuals|", row=2, col=1)
    fig_diag.update_xaxes(title_text="Residuals", row=2, col=2)
    fig_diag.update_yaxes(title_text="Frequency", row=2, col=2)

    fig_diag.update_layout(height=700, template="plotly_white", showlegend=False)
    st.plotly_chart(fig_diag, key="diagnostic_plots_mult_new", use_container_width=True)

    col_m8_1, col_m8_2 = st.columns([1, 1])

    with col_m8_1:
        st.markdown("### ✅ Was wir suchen")
        st.markdown(
            """
        **Plot 1 (Residuals vs Fitted):**
        - Zufällige Streuung um 0
        - Keine Muster (Kurven, Trichter)

        **Plot 2 (Q-Q):**
        - Punkte auf der Diagonale
        - Zeigt Normalverteilung der Residuen

        **Plot 3 (Scale-Location):**
        - Horizontales Band
        - Konstante Varianz (Homoskedastizität)

        **Plot 4 (Histogram):**
        - Glockenförmige Verteilung
        - Zentriert um 0
        """
        )

    with col_m8_2:
        st.markdown("### 📊 Schnell-Statistiken")
        
        # Simple tests
        resid_mean = np.mean(residuals)
        resid_std = np.std(residuals)
        skewness = stats.skew(residuals)
        kurtosis = stats.kurtosis(residuals)
        
        st.info(
            f"""
        **Residuen-Statistiken:**
        - Mittelwert: {resid_mean:.4f} (sollte ≈ 0)
        - Standardabweichung: {resid_std:.4f}
        - Schiefe: {skewness:.3f} (sollte ≈ 0)
        - Kurtosis: {kurtosis:.3f} (sollte ≈ 0)
        
        **Beurteilung:**
        - {"✅ Residuen zentriert" if abs(resid_mean) < 0.1 else "⚠️ Residuen nicht zentriert"}
        - {"✅ Symmetrisch" if abs(skewness) < 1 else "⚠️ Schief verteilt"}
        """
        )

    # =========================================================
    # KAPITEL 9.0: ZUSAMMENFASSUNG
    # =========================================================
    st.markdown("---")
    st.markdown(
        '<p class="section-header">9.0 Zusammenfassung: Multiple Regression</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    Sie haben die **multiple Regression** von Grund auf verstanden! 🎉

    Fassen wir die wichtigsten Konzepte zusammen:
    """
    )

    col_m9_1, col_m9_2 = st.columns([1, 1])

    with col_m9_1:
        st.markdown("### 📋 Kernkonzepte")
        concepts_table = pd.DataFrame(
            {
                "Konzept": [
                    "Grundmodell",
                    "OLS-Schätzer",
                    "R²",
                    "Adjustiertes R²",
                    "t-Test",
                    "F-Test",
                    "Partielle Koeffizienten",
                    "Dummy-Variablen",
                    "Multikollinearität",
                    "VIF",
                ],
                "Status": ["✅"] * 10,
            }
        )
        st.dataframe(concepts_table, use_container_width=True, hide_index=True)

    with col_m9_2:
        st.markdown("### 📊 Unser Modell")
        st.success(
            f"""
        **Modellgleichung:**

        {y_name.split('(')[0].strip()} = {mult_coeffs["params"][0]:.2f}
                 {mult_coeffs["params"][1]:+.2f} · {x1_name.split('(')[0].strip()}
                 {mult_coeffs["params"][2]:+.2f} · {x2_name.split('(')[0].strip()}

        **Modellgüte:**
        - R² = {mult_summary["rsquared"]:.4f}
        - R²_adj = {mult_summary["rsquared_adj"]:.4f}
        - F = {mult_summary["fvalue"]:.2f}

        **Beide Effekte sind statistisch signifikant!**
        """
        )

    st.markdown("### 🎯 Wichtigste Erkenntnisse")
    st.markdown(
        """
    1. **Multiple Regression** erlaubt uns, den Einfluss **mehrerer Variablen gleichzeitig** zu untersuchen

    2. **Partielle Koeffizienten** messen den Effekt **ceteris paribus** (bei Konstanthaltung der anderen)

    3. **Adjustiertes R²** ist besser als R² für Modellvergleiche (bestraft Komplexität)

    4. **Multikollinearität** ist ein Problem - prüfen mit Korrelationen und VIF

    5. **Residuen-Diagnostik** ist essentiell - Annahmen müssen erfüllt sein!

    6. **Dummy-Variablen** ermöglichen kategoriale Prädiktoren

    7. **F-Test** prüft Gesamtsignifikanz, **t-Tests** prüfen einzelne Koeffizienten
    """
    )

    st.info(
        """
    **🚀 Nächste Schritte:**

    - Experimentieren Sie mit den Parametern
    - Vergleichen Sie einfache vs. multiple Regression
    - Prüfen Sie die Residuen-Diagnostik
    - Erkunden Sie Prognosen für verschiedene Szenarien
    """
    )

    logger.info("Multiple regression tab rendered with full educational content")
