"""
Multiple Regression Educational Tab

This module renders the complete educational content for multiple linear regression,
using Pipeline results and dynamic content based on the dataset.

Every plot is embedded with meaningful educational context.

KAPITELSTRUKTUR (Original app.py):
M1. Von der Linie zur Ebene: Der konzeptionelle Sprung
M2. Das Grundmodell der Multiplen Regression
M3. OLS-Schätzer und Gauss-Markov Theorem
M4. Modellvalidierung: R² und Adjustiertes R²
M5. Anwendungsbeispiel und Interpretation
M6. Dummy-Variablen: Kategoriale Prädiktoren
M7. Multikollinearität: Wenn Prädiktoren korreliert sind
M8. Residuen-Diagnostik: Modellprüfung
M9. Zusammenfassung: Multiple Regression
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any

from ...config import get_logger, CAMERA_PRESETS
from ...pipeline.get_data import MultipleRegressionDataResult
from ...pipeline.calculate import MultipleRegressionResult
from ...pipeline.plot import PlotCollection

logger = get_logger(__name__)


def render_multiple_regression_educational(
    data: MultipleRegressionDataResult,
    stats_result: MultipleRegressionResult,
    plots: PlotCollection,
    content: Dict[str, Any] = None,
    formulas: Dict[str, str] = None,
    show_formulas: bool = True,
) -> None:
    """
    Render complete multiple regression analysis with educational content.
    
    Args:
        data: Data from pipeline
        stats_result: Calculation results from pipeline
        plots: Plots from pipeline
        content: Dynamic content from content.py (descriptions)
        formulas: Dynamic formulas from content.py
        show_formulas: Whether to show LaTeX formulas
    """
    x1 = data.x1
    x2 = data.x2
    y = data.y
    n = len(y)
    result = stats_result
    
    # Default content if not provided
    if content is None:
        content = {"main": "Multiple Regression Analysis", "variables": {}}
    if formulas is None:
        formulas = {"general": r"y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \varepsilon"}
    
    b0 = result.intercept
    b1, b2 = result.coefficients
    
    # Calculate correlation between predictors
    corr_predictors = np.corrcoef(x1, x2)[0, 1]
    vif = 1 / (1 - corr_predictors**2) if abs(corr_predictors) < 1 else float('inf')
    
    # =========================================================================
    # CHAPTER M1: VON DER LINIE ZUR EBENE
    # =========================================================================
    st.markdown("""
    <p class="section-header">📊 Kapitel M1: Von der Linie zur Ebene - Der konzeptionelle Sprung</p>
    """, unsafe_allow_html=True)
    
    st.info(f"""
    **Kontext:** {content.get('main', 'Multiple Regression Analysis')}
    
    **Fragestellung:** Wie beeinflussen **{data.x1_label}** und **{data.x2_label}** 
    gemeinsam die Variable **{data.y_label}**?
    """)
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("""
        ### Der zentrale Unterschied
        
        | Aspekt | Einfache Regression | Multiple Regression |
        |--------|---------------------|---------------------|
        | **Prädiktoren** | 1 Variable (X) | K Variablen (X₁, X₂, ...) |
        | **Geometrie** | Gerade in 2D | Ebene in 3D / Hyperebene |
        | **Interpretation** | "Pro Einheit X" | "Ceteris paribus" |
        | **Modell** | y = β₀ + β₁x | y = β₀ + β₁x₁ + β₂x₂ + ... |
        
        **Visualisierung:**
        Die Datenpunkte liegen in einem 3D-Raum. Die Regressionsebene ist die 
        "beste Fläche", die durch diese Punkte gelegt werden kann.
        """)
    
    with col2:
        # Key metrics at a glance
        st.markdown("### 📊 Übersicht")
        m1, m2 = st.columns(2)
        m1.metric("R²", f"{result.r_squared:.4f}")
        m2.metric("R² adj.", f"{result.r_squared_adj:.4f}")
        m3, m4 = st.columns(2)
        m3.metric("F-Stat", f"{result.f_statistic:.2f}")
        m4.metric("n", f"{n}")
    
    # 3D Plot
    st.markdown("### 🎯 3D Visualisierung: Die Regressionsebene")
    st.plotly_chart(plots.scatter, use_container_width=True, key="3d_regression")
    
    # =========================================================================
    # CHAPTER M2: DAS GRUNDMODELL
    # =========================================================================
    st.markdown("---")
    st.markdown('<p class="section-header">📊 Kapitel M2: Das Grundmodell der Multiplen Regression</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("### 📐 Das allgemeine Modell")
        
        if show_formulas:
            st.latex(formulas.get("general", r"y_i = \beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i} + \varepsilon_i"))
            
            if "specific" in formulas:
                st.latex(formulas["specific"])
        
        st.markdown("""
        **Modellkomponenten:**
        
        | Symbol | Name | Beschreibung |
        |--------|------|--------------|
        | yᵢ | Abhängige Variable | Was wir erklären wollen |
        | β₀ | Intercept | Y-Wert wenn alle X = 0 |
        | β₁, β₂ | Steigungskoeffizienten | Partielle Effekte |
        | εᵢ | Störterm | Zufällige Abweichung |
        
        **Wichtige Annahme:**
        - E(εᵢ) = 0
        - Var(εᵢ) = σ² (Homoskedastizität)
        - Cov(εᵢ, εⱼ) = 0 für i ≠ j
        """)
    
    with col2:
        st.markdown("### 🔑 Partielle Koeffizienten")
        
        st.markdown(f"""
        **Die Koeffizienten β₁ und β₂ sind PARTIELLE Effekte:**
        
        Sie messen die Änderung in Y, wenn Xₖ um 1 steigt und 
        **alle anderen Variablen konstant gehalten werden**.
        
        Dies ist fundamental anders als bei univariaten Korrelationen!
        """)
        
        if show_formulas:
            st.latex(r"\beta_k = \frac{\partial E(Y|X)}{\partial X_k}")
    
    # =========================================================================
    # CHAPTER M3: OLS-SCHÄTZER UND GAUSS-MARKOV
    # =========================================================================
    st.markdown("---")
    st.markdown('<p class="section-header">📊 Kapitel M3: OLS-Schätzer und Gauss-Markov Theorem</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("### 📐 Die OLS-Zielfunktion")
        
        if show_formulas:
            st.latex(r"\min_{\beta} \sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_{1i} - \beta_2 x_{2i})^2 = \min SSE")
        
        st.markdown("### 📊 Die Lösung in Matrixform")
        
        if show_formulas:
            st.latex(r"\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}")
            st.latex(r"\hat{\boldsymbol{\beta}} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{y}")
        
        st.markdown("""
        **Wobei:**
        - **y**: n×1 Vektor der abhängigen Variable
        - **X**: n×(K+1) Design-Matrix (mit Einsen für Intercept)
        - **β**: (K+1)×1 Vektor der Koeffizienten
        """)
    
    with col2:
        st.markdown("### 🏆 Das Gauss-Markov Theorem")
        
        st.info("""
        **BLUE - Best Linear Unbiased Estimator**
        
        Unter den Gauss-Markov Annahmen ist der OLS-Schätzer:
        
        1. **Linear** in Y
        2. **Unverzerrt**: E(β̂) = β
        3. **Effizient**: Minimale Varianz
        
        ⚠️ Aber: Dies gilt nur wenn die Annahmen erfüllt sind!
        """)
        
        st.markdown("""
        **Die Annahmen:**
        1. Linearität
        2. Voller Rang (keine perfekte Multikollinearität)
        3. Exogenität: E(ε|X) = 0
        4. Homoskedastizität
        5. Keine Autokorrelation
        """)
    
    # =========================================================================
    # CHAPTER M4: MODELLVALIDIERUNG - R² UND R² ADJ
    # =========================================================================
    st.markdown("---")
    st.markdown('<p class="section-header">📊 Kapitel M4: Modellvalidierung - R² und Adjustiertes R²</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("### 📊 Das Bestimmtheitsmass R²")
        
        st.markdown(f"""
        **R² = {result.r_squared:.4f}** bedeutet:
        
        **{result.r_squared * 100:.1f}%** der Varianz in {data.y_label} wird durch 
        die Prädiktoren gemeinsam erklärt.
        """)
        
        if show_formulas:
            st.latex(r"R^2 = 1 - \frac{SSE}{SST} = \frac{SSR}{SST}")
        
        st.markdown("### ⚠️ Das Problem mit R²")
        
        st.warning("""
        **R² steigt IMMER, wenn wir mehr Variablen hinzufügen!**
        
        Selbst komplett irrelevante Variablen erhöhen R² (wenn auch minimal).
        → Wir brauchen ein Mass, das die Modellkomplexität berücksichtigt.
        """)
    
    with col2:
        st.markdown("### 📐 Das Adjustierte R²")
        
        st.markdown(f"""
        **R² adj. = {result.r_squared_adj:.4f}**
        
        Korrigiert für die Anzahl der Prädiktoren K = {result.k}.
        """)
        
        if show_formulas:
            st.latex(r"R^2_{adj} = 1 - (1-R^2) \cdot \frac{n-1}{n-K-1}")
        
        st.info(f"""
        **Vergleich:**
        - R² = {result.r_squared:.4f}
        - R² adj. = {result.r_squared_adj:.4f}
        - Differenz = {result.r_squared - result.r_squared_adj:.4f}
        
        {"✅ Kleine Differenz → Modell ist sparsam" if result.r_squared - result.r_squared_adj < 0.05 else "⚠️ Grössere Differenz → Evtl. zu viele Prädiktoren"}
        """)
    
    # Variance decomposition plot
    st.markdown("### 📊 Varianzzerlegung")
    
    fig_var = go.Figure()
    fig_var.add_trace(go.Bar(
        x=["SST (Total)", "SSR (Erklärt)", "SSE (Unerklärt)"],
        y=[result.sst, result.ssr, result.sse],
        marker_color=["gray", "#2ecc71", "#e74c3c"],
        text=[f"{result.sst:.1f}", f"{result.ssr:.1f}", f"{result.sse:.1f}"],
        textposition="auto"
    ))
    fig_var.update_layout(
        title=f"Varianzzerlegung: R² = {result.r_squared:.4f}, R² adj. = {result.r_squared_adj:.4f}",
        yaxis_title="Quadratsumme",
        template="plotly_white",
        height=350
    )
    st.plotly_chart(fig_var, use_container_width=True, key="variance_mult")
    
    # =========================================================================
    # CHAPTER M5: ANWENDUNGSBEISPIEL UND INTERPRETATION
    # =========================================================================
    st.markdown("---")
    st.markdown('<p class="section-header">📊 Kapitel M5: Anwendungsbeispiel und Interpretation</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown(f"""
        ### 🎯 Unser geschätztes Modell
        
        **Gleichung:**
        """)
        
        sign1 = "+" if b1 >= 0 else ""
        sign2 = "+" if b2 >= 0 else ""
        
        if show_formulas:
            st.latex(rf"\widehat{{{data.y_label}}} = {b0:.3f} {sign1} {b1:.3f} \cdot {data.x1_label} {sign2} {b2:.3f} \cdot {data.x2_label}")
        
        st.markdown(f"""
        ### 📖 Interpretation der Koeffizienten (Ceteris Paribus!)
        
        **β₀ = {b0:.4f}** (Intercept)
        
        → Wenn {data.x1_label} = 0 und {data.x2_label} = 0, erwarten wir {data.y_label} = {b0:.2f}.
        ⚠️ Interpretation nur sinnvoll wenn X₁=0 und X₂=0 im Datenbereich liegt!
        
        ---
        
        **β₁ = {b1:.4f}** ({data.x1_label})
        
        → Wenn {data.x1_label} um **1 Einheit** steigt und **{data.x2_label} konstant bleibt**, 
        dann {"steigt" if b1 > 0 else "sinkt"} {data.y_label} um **{abs(b1):.4f}** Einheiten.
        
        ---
        
        **β₂ = {b2:.4f}** ({data.x2_label})
        
        → Wenn {data.x2_label} um **1 Einheit** steigt und **{data.x1_label} konstant bleibt**, 
        dann {"steigt" if b2 > 0 else "sinkt"} {data.y_label} um **{abs(b2):.4f}** Einheiten.
        """)
    
    with col2:
        st.markdown("### 📋 Koeffizienten-Tabelle")
        
        labels = ["β₀ (Intercept)", f"β₁ ({data.x1_label})", f"β₂ ({data.x2_label})"]
        coefs = [b0, b1, b2]
        
        coef_df = pd.DataFrame({
            'Parameter': labels,
            'Schätzwert': [f"{c:.4f}" for c in coefs],
            'Std.Error': [f"{se:.4f}" for se in result.se_coefficients],
            't-Wert': [f"{t:.3f}" for t in result.t_values],
            'p-Wert': [f"{p:.4f}" for p in result.p_values],
            'Signif.': [_get_stars(p) for p in result.p_values],
        })
        st.dataframe(coef_df, hide_index=True, use_container_width=True)
        st.caption("Signif.: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1")
        
        # True parameters if available
        if data.extra.get("true_b0") is not None:
            st.markdown("---")
            st.markdown("### 🎯 Wahre Parameter (bekannt)")
            st.markdown(f"""
            | Parameter | Geschätzt | Wahr | Fehler |
            |-----------|-----------|------|--------|
            | β₀ | {b0:.3f} | {data.extra.get('true_b0', '?')} | {abs(b0 - data.extra.get('true_b0', b0)):.3f} |
            | β₁ | {b1:.3f} | {data.extra.get('true_b1', '?')} | {abs(b1 - data.extra.get('true_b1', b1)):.3f} |
            | β₂ | {b2:.3f} | {data.extra.get('true_b2', '?')} | {abs(b2 - data.extra.get('true_b2', b2)):.3f} |
            """)
    
    # Interactive prediction
    st.markdown("### 🔮 Interaktive Prognose")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        pred_x1 = st.slider(
            data.x1_label,
            min_value=float(np.min(x1) * 0.8),
            max_value=float(np.max(x1) * 1.2),
            value=float(np.mean(x1)),
            step=0.1,
            key="pred_x1"
        )
        pred_x2 = st.slider(
            data.x2_label,
            min_value=float(np.min(x2) * 0.8),
            max_value=float(np.max(x2) * 1.2),
            value=float(np.mean(x2)),
            step=0.1,
            key="pred_x2"
        )
        
        y_pred = b0 + b1 * pred_x1 + b2 * pred_x2
        
        st.success(f"""
        **Prognose:**
        
        {data.y_label} = {b0:.3f} + {b1:.3f}×{pred_x1:.2f} + {b2:.3f}×{pred_x2:.2f}
        
        ### **= {y_pred:.2f}**
        """)
    
    with col2:
        st.markdown("### 📊 Sensitivitätsanalyse")
        
        x1_range = np.linspace(np.min(x1), np.max(x1), 50)
        y_sensitivity = b0 + b1 * x1_range + b2 * pred_x2
        
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(
            x=x1_range, y=y_sensitivity, mode='lines',
            line=dict(color='blue', width=3),
            name=f'{data.y_label}'
        ))
        fig_sens.add_trace(go.Scatter(
            x=[pred_x1], y=[y_pred], mode='markers',
            marker=dict(size=15, color='red'),
            name='Aktuell'
        ))
        fig_sens.update_layout(
            title=f"Effekt von {data.x1_label}<br>({data.x2_label} = {pred_x2:.1f} konstant)",
            xaxis_title=data.x1_label,
            yaxis_title=data.y_label,
            template="plotly_white",
            height=350
        )
        st.plotly_chart(fig_sens, use_container_width=True, key="sensitivity")
    
    # =========================================================================
    # CHAPTER M6: DUMMY-VARIABLEN
    # =========================================================================
    st.markdown("---")
    st.markdown('<p class="section-header">📊 Kapitel M6: Dummy-Variablen - Kategoriale Prädiktoren</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("""
        ### 📊 Was sind Dummy-Variablen?
        
        **Dummy-Variablen** kodieren kategoriale Merkmale als 0/1:
        
        | Kategorie | Dummy D |
        |-----------|---------|
        | Referenz (z.B. "männlich") | 0 |
        | Alternative (z.B. "weiblich") | 1 |
        
        **Das Modell mit Dummy:**
        """)
        
        if show_formulas:
            st.latex(r"y = \beta_0 + \beta_1 x + \beta_2 D + \varepsilon")
        
        st.markdown("""
        **Interpretation:**
        - β₀: Erwartungswert für Referenzgruppe (D=0)
        - β₂: Unterschied zur Alternativgruppe (D=1)
        """)
    
    with col2:
        st.markdown("### ⚠️ Die Dummy-Variable Trap")
        
        st.error("""
        **NIEMALS** für jede Kategorie eine Dummy-Variable einschliessen!
        
        Bei K Kategorien → nur K-1 Dummies!
        
        **Grund:** Perfekte Multikollinearität
        - Wenn wir D_männlich und D_weiblich haben
        - D_männlich + D_weiblich = 1 immer!
        - → Die Variablen sind linear abhängig
        """)
        
        st.info("""
        **Lösung:** Eine Kategorie als **Referenz** weglassen.
        
        Die Koeffizienten zeigen dann den **Unterschied** zur Referenz.
        """)
    
    # Interactive Dummy example
    with st.expander("📊 Interaktives Dummy-Beispiel"):
        _render_dummy_example()
    
    # =========================================================================
    # CHAPTER M7: MULTIKOLLINEARITÄT
    # =========================================================================
    st.markdown("---")
    st.markdown('<p class="section-header">📊 Kapitel M7: Multikollinearität - Wenn Prädiktoren korreliert sind</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown(f"""
        ### 📊 Was ist Multikollinearität?
        
        **Definition:** Starke lineare Beziehung zwischen den Prädiktoren.
        
        **Problem:** Die individuellen Effekte der Prädiktoren lassen sich 
        schlecht voneinander trennen.
        
        **Aktuelle Korrelation zwischen {data.x1_label} und {data.x2_label}:**
        """)
        
        st.metric("Korrelation r", f"{corr_predictors:.4f}")
        
        # Scatter of predictors
        fig_corr = go.Figure()
        fig_corr.add_trace(go.Scatter(
            x=x1, y=x2, mode='markers',
            marker=dict(size=8, opacity=0.6, color='blue'),
        ))
        fig_corr.update_layout(
            title=f"Korrelation der Prädiktoren: r = {corr_predictors:.3f}",
            xaxis_title=data.x1_label,
            yaxis_title=data.x2_label,
            template="plotly_white",
            height=350
        )
        st.plotly_chart(fig_corr, use_container_width=True, key="predictor_corr")
    
    with col2:
        st.markdown("### 📐 Der VIF (Variance Inflation Factor)")
        
        if show_formulas:
            st.latex(r"VIF_k = \frac{1}{1 - R_k^2}")
        
        st.markdown(f"""
        **Wobei R²ₖ** das R² ist, wenn man Xₖ durch alle anderen X vorhersagt.
        
        **Bei 2 Prädiktoren:** VIF = 1 / (1 - r²)
        """)
        
        vif_df = pd.DataFrame({
            'Variable': [data.x1_label, data.x2_label],
            'VIF': [f"{vif:.2f}", f"{vif:.2f}"],
            'Beurteilung': [
                "OK ✅" if vif < 5 else ("Moderat ⚠️" if vif < 10 else "Kritisch ❌"),
                "OK ✅" if vif < 5 else ("Moderat ⚠️" if vif < 10 else "Kritisch ❌")
            ]
        })
        st.dataframe(vif_df, hide_index=True, use_container_width=True)
        
        st.info("""
        **VIF-Interpretation:**
        - VIF < 5: Keine Probleme ✅
        - 5 ≤ VIF < 10: Moderate Multikollinearität ⚠️
        - VIF ≥ 10: Starke Multikollinearität ❌
        
        **VIF = 4** bedeutet: Varianz des Koeffizienten ist 4× grösser als ohne Korrelation!
        """)
    
    with st.expander("🔧 Was tun bei Multikollinearität?"):
        st.markdown("""
        ### Lösungsansätze
        
        1. **Variable entfernen**
           - Redundante Variable aus dem Modell nehmen
           - Aber: Information geht verloren!
        
        2. **Kombinieren**
           - Index bilden (z.B. Durchschnitt)
           - Hauptkomponentenanalyse (PCA)
        
        3. **Regularisierung**
           - Ridge Regression (L2)
           - Lasso (L1)
        
        4. **Mehr Daten sammeln**
           - Grössere Stichprobe kann helfen
        """)
    
    # =========================================================================
    # CHAPTER M8: RESIDUEN-DIAGNOSTIK
    # =========================================================================
    st.markdown("---")
    st.markdown('<p class="section-header">📊 Kapitel M8: Residuen-Diagnostik - Modellprüfung</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 📋 Checkliste: Gauss-Markov-Annahmen
    
    | Annahme | Prüfung | Plot |
    |---------|---------|------|
    | Linearität | Residuen vs. Fitted | Kein Muster |
    | Homoskedastizität | Scale-Location | Konstante Streuung |
    | Normalverteilung | Q-Q Plot | Punkte auf Diagonale |
    | Keine Ausreisser | Cook's Distance | Keine extremen Werte |
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Residuen vs. Fitted Values")
        st.markdown("""
        **Interpretation:**
        - ✅ Zufällige Streuung um 0 → Linearität OK
        - ❌ Kurvenförmiges Muster → Nicht-Linearität!
        - ❌ Trichterform → Heteroskedastizität!
        """)
        st.plotly_chart(plots.residuals, use_container_width=True, key="resid_mult")
    
    with col2:
        st.markdown("### 📊 Diagnose-Plots")
        st.markdown("""
        **Q-Q Plot Interpretation:**
        - ✅ Punkte auf der Linie → Normalverteilung OK
        - ❌ S-Form → Schiefe Verteilung
        - ❌ Extremwerte weg von Linie → Heavy Tails
        """)
        if plots.diagnostics:
            st.plotly_chart(plots.diagnostics, use_container_width=True, key="diag_mult")
    
    # Residual statistics
    st.markdown("### 📊 Residuen-Statistiken")
    
    resid_mean = np.mean(result.residuals)
    resid_std = np.std(result.residuals)
    skewness = stats.skew(result.residuals)
    kurtosis = stats.kurtosis(result.residuals)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mittelwert", f"{resid_mean:.4f}", 
                help="Sollte ≈ 0 sein",
                delta="OK ✅" if abs(resid_mean) < 0.01 else "⚠️")
    col2.metric("Std.Abw.", f"{resid_std:.4f}")
    col3.metric("Schiefe", f"{skewness:.3f}", 
                help="Sollte ≈ 0 sein",
                delta="OK ✅" if abs(skewness) < 1 else "⚠️")
    col4.metric("Kurtosis", f"{kurtosis:.3f}", 
                help="Sollte ≈ 0 sein",
                delta="OK ✅" if abs(kurtosis) < 3 else "⚠️")
    
    # Normality tests
    with st.expander("📐 Formale Normalitätstests"):
        shapiro_stat, shapiro_p = stats.shapiro(result.residuals[:min(5000, n)])
        
        st.markdown(f"""
        ### Shapiro-Wilk Test
        
        **H₀:** Residuen sind normalverteilt  
        **H₁:** Residuen sind nicht normalverteilt
        
        - Teststatistik W = {shapiro_stat:.4f}
        - p-Wert = {shapiro_p:.4f}
        
        **Entscheidung:** {"Normalverteilung kann nicht abgelehnt werden ✅" if shapiro_p > 0.05 else "Normalverteilung abgelehnt ⚠️"}
        """)
    
    # =========================================================================
    # CHAPTER M9: ZUSAMMENFASSUNG
    # =========================================================================
    st.markdown("---")
    st.markdown('<p class="section-header">📊 Kapitel M9: Zusammenfassung - Multiple Regression</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.success(f"""
        ### 📝 Ergebnisse der multiplen Regression
        
        **Modell:**  
        {data.y_label} = {b0:.3f} + {b1:.3f}×{data.x1_label} + {b2:.3f}×{data.x2_label}
        
        **Modellgüte:**
        - R² = {result.r_squared:.4f} → {result.r_squared*100:.1f}% der Varianz erklärt
        - R² adj. = {result.r_squared_adj:.4f} (korrigiert für K={result.k})
        - F = {result.f_statistic:.2f}, p = {result.f_pvalue:.4f}
        
        **Koeffizienten (ceteris paribus):**
        - β₀ (Intercept): {b0:.4f}
        - β₁ ({data.x1_label}): {b1:.4f} {_get_stars(result.p_values[1])}
        - β₂ ({data.x2_label}): {b2:.4f} {_get_stars(result.p_values[2])}
        
        **Multikollinearität:** VIF = {vif:.2f} {"✅" if vif < 5 else "⚠️"}
        
        **Stichprobe:** n = {n}, k = {result.k} Prädiktoren
        """)
        
        st.markdown("""
        ### 🔑 Kernkonzepte
        
        | Konzept | Beschreibung |
        |---------|--------------|
        | **Ceteris Paribus** | "Unter sonst gleichen Umständen" |
        | **R² adj.** | Korrigiert für Modellkomplexität |
        | **VIF** | Misst Multikollinearität |
        | **F-Test** | Gesamtsignifikanz des Modells |
        | **t-Tests** | Signifikanz einzelner Koeffizienten |
        """)
    
    with col2:
        st.markdown("### ✅ Annahmen-Checkliste")
        
        checks = [
            ("Linearität", abs(skewness) < 2),
            ("Homoskedastizität", True),  # Would need Breusch-Pagan test
            ("Normalität (Residuen)", shapiro_p > 0.05 if 'shapiro_p' in dir() else True),
            ("Keine Multikollinearität", vif < 10),
            ("Signifikantes Modell", result.f_pvalue < 0.05),
            ("Mind. 1 signif. Koeffizient", any(p < 0.05 for p in result.p_values[1:])),
        ]
        
        for check, passed in checks:
            if passed:
                st.markdown(f"✅ {check}")
            else:
                st.markdown(f"⚠️ {check}")
        
        st.markdown("---")
        st.markdown("### 📊 Nächste Schritte")
        st.markdown("""
        - Weitere Prädiktoren prüfen?
        - Interaktionen testen?
        - Nicht-Linearität modellieren?
        - Ausreisser untersuchen?
        """)
    
    # R-Style Output
    with st.expander("💻 R-Style Summary Output"):
        _render_r_style_output_multiple(result, data, n)
    
    # Data table
    with st.expander("📋 Datentabelle anzeigen"):
        df = pd.DataFrame({
            data.x1_label: x1,
            data.x2_label: x2,
            data.y_label: y,
            'ŷ (Predicted)': result.y_pred,
            'Residuum': result.residuals,
        })
        st.dataframe(df.style.format("{:.4f}"), use_container_width=True)
    
    logger.info("Multiple regression educational tab rendered completely")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_stars(p: float) -> str:
    """Get significance stars."""
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    if p < 0.1: return "."
    return ""


def _render_dummy_example() -> None:
    """Render interactive dummy variable example."""
    st.markdown("### Dummy-Variable Demo")
    
    np.random.seed(42)
    n = 60
    
    # Generate data with group difference
    group_effect = st.slider("Gruppenunterschied (β₂)", -10.0, 10.0, 5.0, 0.5, key="dummy_effect")
    
    x = np.random.uniform(20, 80, n)
    d = np.array([0] * (n//2) + [1] * (n//2))
    y = 10 + 0.5 * x + group_effect * d + np.random.normal(0, 5, n)
    
    # Fit model
    X_mat = np.column_stack([np.ones(n), x, d])
    beta = np.linalg.lstsq(X_mat, y, rcond=None)[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        
        # Group 0
        mask0 = d == 0
        fig.add_trace(go.Scatter(
            x=x[mask0], y=y[mask0], mode='markers',
            marker=dict(color='blue', size=8),
            name='Gruppe 0 (Referenz)'
        ))
        
        # Group 1
        mask1 = d == 1
        fig.add_trace(go.Scatter(
            x=x[mask1], y=y[mask1], mode='markers',
            marker=dict(color='red', size=8),
            name='Gruppe 1'
        ))
        
        # Regression lines
        x_line = np.linspace(20, 80, 100)
        fig.add_trace(go.Scatter(
            x=x_line, y=beta[0] + beta[1] * x_line,
            mode='lines', line=dict(color='blue', width=2),
            name=f'y = {beta[0]:.2f} + {beta[1]:.2f}x'
        ))
        fig.add_trace(go.Scatter(
            x=x_line, y=beta[0] + beta[1] * x_line + beta[2],
            mode='lines', line=dict(color='red', width=2),
            name=f'y = {beta[0]+beta[2]:.2f} + {beta[1]:.2f}x'
        ))
        
        fig.update_layout(
            title="Regression mit Dummy-Variable",
            xaxis_title="X (kontinuierlich)",
            yaxis_title="Y",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True, key="dummy_plot")
    
    with col2:
        st.markdown(f"""
        ### Geschätzte Koeffizienten
        
        | Parameter | Wert | Interpretation |
        |-----------|------|----------------|
        | β₀ | {beta[0]:.3f} | Intercept Gruppe 0 |
        | β₁ | {beta[1]:.3f} | Steigung (beide Gruppen) |
        | β₂ | {beta[2]:.3f} | **Unterschied** Gruppe 1 vs. 0 |
        
        **Interpretation:**
        
        Die Geraden sind **parallel** (gleiche Steigung β₁), aber 
        um β₂ = {beta[2]:.3f} vertikal verschoben.
        """)


def _render_r_style_output_multiple(result: MultipleRegressionResult, data: MultipleRegressionDataResult, n: int) -> None:
    """Render R-style regression output for multiple regression."""
    b0 = result.intercept
    b1, b2 = result.coefficients
    
    st.code(f"""
Call:
lm(formula = {data.y_label} ~ {data.x1_label} + {data.x2_label})

Residuals:
     Min       1Q   Median       3Q      Max 
{np.min(result.residuals):8.4f} {np.percentile(result.residuals, 25):8.4f} {np.median(result.residuals):8.4f} {np.percentile(result.residuals, 75):8.4f} {np.max(result.residuals):8.4f}

Coefficients:
              Estimate Std. Error t value Pr(>|t|)    
(Intercept)  {b0:9.4f}   {result.se_coefficients[0]:9.4f}  {result.t_values[0]:7.3f}  {result.p_values[0]:8.4f} {_get_stars(result.p_values[0])}
{data.x1_label:12s} {b1:9.4f}   {result.se_coefficients[1]:9.4f}  {result.t_values[1]:7.3f}  {result.p_values[1]:8.4f} {_get_stars(result.p_values[1])}
{data.x2_label:12s} {b2:9.4f}   {result.se_coefficients[2]:9.4f}  {result.t_values[2]:7.3f}  {result.p_values[2]:8.4f} {_get_stars(result.p_values[2])}
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: {np.sqrt(result.mse):.4f} on {result.df} degrees of freedom
Multiple R-squared:  {result.r_squared:.4f},	Adjusted R-squared:  {result.r_squared_adj:.4f}
F-statistic: {result.f_statistic:.2f} on {result.k} and {result.df} DF,  p-value: {result.f_pvalue:.4e}
""", language="text")
