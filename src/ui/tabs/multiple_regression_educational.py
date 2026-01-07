"""
Multiple Regression Educational Tab

This module renders the complete educational content for multiple linear regression,
using Pipeline results and dynamic content based on the dataset.

Every plot is embedded with meaningful educational context.
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
    
    # =========================================================
    # CHAPTER 1: EINLEITUNG
    # =========================================================
    st.markdown("""
    <p class="section-header">📊 Multiple Lineare Regression</p>
    """, unsafe_allow_html=True)
    
    st.info(f"""
    **Kontext:** {content.get('main', 'Multiple Regression Analysis')}
    
    **Fragestellung:** Wie beeinflussen **{data.x1_label}** und **{data.x2_label}** 
    gemeinsam die Variable **{data.y_label}**?
    """)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", f"{result.r_squared:.4f}", help="Erklärte Varianz")
    col2.metric("R² adj.", f"{result.r_squared_adj:.4f}", help="Korrigiert für Anzahl Prädiktoren")
    col3.metric("F-Statistik", f"{result.f_statistic:.2f}", help="Gesamtsignifikanz")
    col4.metric("n", f"{n}", help="Stichprobengrösse")
    
    # =========================================================
    # CHAPTER 2: VON DER LINIE ZUR EBENE
    # =========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">1️⃣ Von der Linie zur Ebene: Der konzeptionelle Sprung</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("""
        **Der zentrale Unterschied:**
        
        | Aspekt | Einfache Regression | Multiple Regression |
        |--------|---------------------|---------------------|
        | **Prädiktoren** | 1 Variable (X) | K Variablen (X₁, X₂, ...) |
        | **Geometrie** | Gerade in 2D | Ebene in 3D / Hyperebene |
        | **Interpretation** | "Pro Einheit X" | "Bei Konstanthaltung der anderen" |
        
        **Was zeigt der 3D-Plot?**
        
        Die **Regressionsebene** zeigt, wie sich {data.y_label} in Abhängigkeit von 
        beiden Prädiktoren ({data.x1_label} und {data.x2_label}) verändert.
        Die Punkte sind die tatsächlichen Beobachtungen.
        """)
    
    with col2:
        if show_formulas:
            st.markdown("### 📐 Das Modell")
            st.latex(formulas.get("general", r"y_i = \beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i} + \varepsilon_i"))
            
            if "specific" in formulas:
                st.latex(formulas["specific"])
    
    # 3D Plot
    st.plotly_chart(plots.scatter, use_container_width=True, key="3d_regression")
    
    # =========================================================
    # CHAPTER 3: DIE KOEFFIZIENTEN
    # =========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">2️⃣ Die geschätzten Koeffizienten</p>', unsafe_allow_html=True)
    
    b0 = result.intercept
    b1, b2 = result.coefficients
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown(f"""
        ### 🎯 Unser geschätztes Modell
        
        **Gleichung:**
        """)
        
        sign1 = "+" if b1 >= 0 else ""
        sign2 = "+" if b2 >= 0 else ""
        
        if show_formulas:
            st.latex(rf"\hat{{y}} = {b0:.3f} {sign1} {b1:.3f} \cdot x_1 {sign2} {b2:.3f} \cdot x_2")
        
        st.markdown(f"""
        ### 📖 Interpretation (Ceteris Paribus!)
        
        **β₁ = {b1:.4f}** ({data.x1_label})
        
        → Wenn {data.x1_label} um 1 steigt und **{data.x2_label} konstant bleibt**, 
        ändert sich {data.y_label} um **{b1:.4f}**.
        
        ---
        
        **β₂ = {b2:.4f}** ({data.x2_label})
        
        → Wenn {data.x2_label} um 1 steigt und **{data.x1_label} konstant bleibt**, 
        ändert sich {data.y_label} um **{b2:.4f}**.
        """)
        
        st.warning("""
        ⚠️ **Wichtig: Ceteris Paribus**
        
        Anders als bei der einfachen Regression messen die Koeffizienten den 
        **isolierten Effekt** bei Konstanthaltung der anderen Variablen!
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
            | Parameter | Geschätzt | Wahr |
            |-----------|-----------|------|
            | β₀ | {b0:.3f} | {data.extra.get('true_b0', '?')} |
            | β₁ | {b1:.3f} | {data.extra.get('true_b1', '?')} |
            | β₂ | {b2:.3f} | {data.extra.get('true_b2', '?')} |
            """)
    
    # =========================================================
    # CHAPTER 4: MODELLGÜTE
    # =========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">3️⃣ Modellgüte: R² und adjustiertes R²</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown(f"""
        **R² = {result.r_squared:.4f}** bedeutet:
        
        **{result.r_squared * 100:.1f}%** der Varianz in {data.y_label} wird durch 
        die Prädiktoren gemeinsam erklärt.
        
        **⚠️ Problem mit R²:** Es steigt IMMER, wenn wir mehr Variablen hinzufügen, 
        selbst wenn diese irrelevant sind!
        
        **Lösung: Adjustiertes R² = {result.r_squared_adj:.4f}**
        
        Das adjustierte R² bestraft unnötige Komplexität und kann sogar sinken, 
        wenn wir schwache Prädiktoren hinzufügen.
        """)
        
        if show_formulas:
            st.latex(r"R^2_{adj} = 1 - (1-R^2) \cdot \frac{n-1}{n-K-1}")
        
        # Variance decomposition plot
        fig_var = go.Figure()
        fig_var.add_trace(go.Bar(
            x=["SST (Total)", "SSR (Erklärt)", "SSE (Unerklärt)"],
            y=[result.sst, result.ssr, result.sse],
            marker_color=["gray", "#2ecc71", "#e74c3c"],
            text=[f"{result.sst:.1f}", f"{result.ssr:.1f}", f"{result.sse:.1f}"],
            textposition="auto"
        ))
        fig_var.update_layout(
            title=f"Varianzzerlegung: R² = {result.r_squared:.4f}",
            yaxis_title="Quadratsumme",
            template="plotly_white",
            height=300
        )
        st.plotly_chart(fig_var, use_container_width=True, key="variance_mult")
    
    with col2:
        st.markdown("### 📊 Vergleich der R²-Masse")
        
        comparison_df = pd.DataFrame({
            'Mass': ['R²', 'R² adj.', 'Differenz'],
            'Wert': [f"{result.r_squared:.4f}", f"{result.r_squared_adj:.4f}", 
                    f"{result.r_squared - result.r_squared_adj:.4f}"],
            'Interpretation': [
                'Roh-Erklärungskraft',
                'Korrigiert für K=' + str(result.k),
                'Klein = gut!'
            ]
        })
        st.dataframe(comparison_df, hide_index=True, use_container_width=True)
        
        # F-Test
        st.markdown("---")
        st.markdown("### 📊 F-Test (Gesamtsignifikanz)")
        st.metric("F-Statistik", f"{result.f_statistic:.2f}")
        st.metric("p-Wert", f"{result.f_pvalue:.4f}")
        
        if result.f_pvalue < 0.05:
            st.success("✅ Modell ist insgesamt signifikant")
        else:
            st.warning("⚠️ Modell ist nicht signifikant")
    
    # =========================================================
    # CHAPTER 5: MULTIKOLLINEARITÄT
    # =========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">4️⃣ Multikollinearität: Korrelation der Prädiktoren</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.5, 1])
    
    # Calculate correlation between predictors
    corr_predictors = np.corrcoef(x1, x2)[0, 1]
    
    with col1:
        st.markdown(f"""
        **Was ist Multikollinearität?**
        
        Wenn die Prädiktoren stark miteinander korrelieren, wird es schwierig, 
        ihre individuellen Effekte zu trennen. Die Koeffizienten werden instabil.
        
        **Korrelation zwischen {data.x1_label} und {data.x2_label}:**
        
        **r = {corr_predictors:.4f}**
        """)
        
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
        st.markdown("### 📊 VIF (Variance Inflation Factor)")
        
        # Simple VIF calculation for 2 predictors
        vif = 1 / (1 - corr_predictors**2) if abs(corr_predictors) < 1 else float('inf')
        
        vif_df = pd.DataFrame({
            'Variable': [data.x1_label, data.x2_label],
            'VIF': [f"{vif:.2f}", f"{vif:.2f}"],
            'Beurteilung': [
                "OK ✅" if vif < 5 else ("Moderat ⚠️" if vif < 10 else "Hoch ❌"),
                "OK ✅" if vif < 5 else ("Moderat ⚠️" if vif < 10 else "Hoch ❌")
            ]
        })
        st.dataframe(vif_df, hide_index=True, use_container_width=True)
        
        st.markdown("""
        **Interpretation VIF:**
        - VIF < 5: Keine Probleme ✅
        - 5 ≤ VIF < 10: Moderate Multikollinearität ⚠️
        - VIF ≥ 10: Starke Multikollinearität ❌
        """)
        
        if show_formulas:
            st.latex(r"VIF_k = \frac{1}{1 - R_k^2}")
    
    # =========================================================
    # CHAPTER 6: RESIDUENANALYSE
    # =========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">5️⃣ Residuenanalyse: Modellannahmen prüfen</p>', unsafe_allow_html=True)
    
    st.markdown("""
    **Die Gauss-Markov-Annahmen müssen erfüllt sein:**
    1. Linearität: E(ε|X) = 0
    2. Homoskedastizität: Var(ε|X) = σ² (konstant)
    3. Keine Autokorrelation
    4. Normalverteilung der Residuen (für Inferenz)
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Residuen vs. Fitted")
        st.markdown("""
        **Was wir suchen:** Zufällige Streuung um 0, kein Muster!
        """)
        st.plotly_chart(plots.residuals, use_container_width=True, key="resid_mult")
    
    with col2:
        st.markdown("### Diagnose-Plots")
        st.markdown("""
        **Q-Q Plot:** Normalverteilung prüfen (Punkte auf Diagonale)
        """)
        if plots.diagnostics:
            st.plotly_chart(plots.diagnostics, use_container_width=True, key="diag_mult")
    
    # Residual statistics
    resid_mean = np.mean(result.residuals)
    resid_std = np.std(result.residuals)
    skewness = stats.skew(result.residuals)
    kurtosis = stats.kurtosis(result.residuals)
    
    st.markdown("### 📊 Residuen-Statistiken")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mittelwert", f"{resid_mean:.4f}", help="Sollte ≈ 0 sein")
    col2.metric("Std.Abw.", f"{resid_std:.4f}")
    col3.metric("Schiefe", f"{skewness:.3f}", help="Sollte ≈ 0 sein")
    col4.metric("Kurtosis", f"{kurtosis:.3f}", help="Sollte ≈ 0 sein")
    
    # =========================================================
    # CHAPTER 7: PROGNOSE
    # =========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">6️⃣ Prognose: Das Modell anwenden</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🔮 Interaktive Prognose")
        
        # Sliders for prediction
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
        
        # Calculate prediction
        y_pred = b0 + b1 * pred_x1 + b2 * pred_x2
        
        st.success(f"""
        **Prognose für:**
        - {data.x1_label} = {pred_x1:.2f}
        - {data.x2_label} = {pred_x2:.2f}
        
        **Erwarteter {data.y_label}:**
        
        ### **{y_pred:.2f}**
        """)
    
    with col2:
        st.markdown("### 📊 Sensitivitätsanalyse")
        
        # Show how y changes with x1 (keeping x2 constant)
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
            title=f"Sensitivität: {data.x1_label}<br>({data.x2_label} = {pred_x2:.1f} konstant)",
            xaxis_title=data.x1_label,
            yaxis_title=data.y_label,
            template="plotly_white",
            height=350
        )
        st.plotly_chart(fig_sens, use_container_width=True, key="sensitivity")
    
    # =========================================================
    # CHAPTER 8: ZUSAMMENFASSUNG
    # =========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">7️⃣ Zusammenfassung</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.success(f"""
        ### 📝 Ergebnisse der multiplen Regression
        
        **Modell:**  
        {data.y_label} = {b0:.3f} + {b1:.3f}×{data.x1_label} + {b2:.3f}×{data.x2_label}
        
        **Güte:**
        - R² = {result.r_squared:.4f} → {result.r_squared*100:.1f}% der Varianz erklärt
        - R² adj. = {result.r_squared_adj:.4f}
        - F = {result.f_statistic:.2f}, p = {result.f_pvalue:.4f}
        
        **Koeffizienten (ceteris paribus):**
        - β₁ ({data.x1_label}): {b1:.4f} {"✅" if result.p_values[1] < 0.05 else ""}
        - β₂ ({data.x2_label}): {b2:.4f} {"✅" if result.p_values[2] < 0.05 else ""}
        
        **Stichprobe:** n = {n}, k = {result.k} Prädiktoren
        """)
    
    with col2:
        st.markdown("### 📋 Checkliste")
        
        checks = [
            ("Linearität", True),
            ("Homoskedastizität", abs(skewness) < 2),
            ("Normalität Residuen", abs(kurtosis) < 7),
            ("Keine Multikollinearität", vif < 10),
            ("Signifikantes Modell", result.f_pvalue < 0.05),
        ]
        
        for check, passed in checks:
            if passed:
                st.markdown(f"✅ {check}")
            else:
                st.markdown(f"⚠️ {check}")
    
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
    
    logger.info("Multiple regression educational tab rendered")


def _get_stars(p: float) -> str:
    """Get significance stars."""
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    if p < 0.1: return "."
    return ""
