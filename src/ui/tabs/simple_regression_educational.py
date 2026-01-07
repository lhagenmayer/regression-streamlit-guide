"""
Simple Regression Educational Tab

This module renders the complete educational content for simple linear regression,
using Pipeline results and dynamic content based on the dataset.

Every plot is embedded with meaningful educational context.

KAPITELSTRUKTUR (Original app.py):
1.0 Einleitung: Die Analyse von Zusammenhängen
1.5 Mehrdimensionale Verteilungen
2.0 Das Fundament: Das einfache lineare Regressionsmodell
2.5 Kovarianz & Korrelation
3.0 Die Methode: OLS-Schätzung
3.1 Regressionsmodell im Detail
4.0 Die Güteprüfung
5.0 Die Signifikanz
5.5 ANOVA für Gruppenvergleiche
5.6 Heteroskedastizität
6.0 Fazit und Ausblick
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional

from ...config import get_logger, CAMERA_PRESETS
from ...pipeline.get_data import DataResult
from ...pipeline.calculate import RegressionResult
from ...pipeline.plot import PlotCollection

logger = get_logger(__name__)


def render_simple_regression_educational(
    data: DataResult,
    stats_result: RegressionResult,
    plots: PlotCollection,
    show_formulas: bool = True,
    show_true_line: bool = False,
) -> None:
    """
    Render complete simple regression analysis with educational content.
    
    All plots are embedded in educational context explaining:
    - What the plot shows
    - How to interpret it
    - What to look for
    """
    x = data.x
    y = data.y
    n = len(x)
    result = stats_result
    
    # =========================================================================
    # CHAPTER 1.0: EINLEITUNG - DIE ANALYSE VON ZUSAMMENHÄNGEN
    # =========================================================================
    st.markdown(f"""
    <p class="section-header">📖 Kapitel 1: Einleitung - Die Analyse von Zusammenhängen</p>
    """, unsafe_allow_html=True)
    
    st.info(f"""
    **Kontext:** {data.context_title}
    
    {data.context_description}
    
    **Zentrale Fragestellung:** Gibt es einen linearen Zusammenhang zwischen **{data.x_label}** und **{data.y_label}**?
    Und wenn ja: Wie stark ist dieser Zusammenhang?
    """)
    
    st.markdown("""
    ### 🎯 Lernziele dieses Moduls
    
    Nach Abschluss dieses Moduls werden Sie verstehen:
    
    1. **Konzepte**: Mehrdimensionale Verteilungen, Kovarianz, Korrelation
    2. **Methodik**: OLS-Schätzung, Residuenanalyse
    3. **Modellgüte**: R², Standardfehler, Bestimmtheitsmass
    4. **Inferenz**: t-Tests, F-Tests, Signifikanz
    5. **Probleme**: Heteroskedastizität, robuste Standardfehler
    """)
    
    # Key metrics at a glance
    st.markdown("### 📊 Übersicht der Ergebnisse")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("R²", f"{result.r_squared:.4f}", help="Erklärte Varianz")
    col2.metric("β₀", f"{result.intercept:.4f}", help="Y-Achsenabschnitt")
    col3.metric("β₁", f"{result.slope:.4f}", help="Steigung")
    col4.metric("p-Wert", f"{result.p_slope:.4f}", help="Signifikanz")
    col5.metric("n", f"{n}", help="Stichprobengrösse")
    
    # =========================================================================
    # CHAPTER 1.5: MEHRDIMENSIONALE VERTEILUNGEN
    # =========================================================================
    st.markdown("---")
    st.markdown('<p class="section-header">📖 Kapitel 1.5: Mehrdimensionale Verteilungen</p>', unsafe_allow_html=True)
    
    st.markdown("""
    **Das Fundament für Zusammenhänge**
    
    Bevor wir Zusammenhänge analysieren können, müssen wir verstehen wie zwei 
    Zufallsvariablen X und Y **gemeinsam** verteilt sein können.
    """)
    
    # 🎲 Gemeinsame Verteilung f(X,Y)
    with st.expander("🎲 Gemeinsame Verteilung f(X,Y)", expanded=True):
        st.markdown("""
        Die **gemeinsame Dichtefunktion** f(x,y) beschreibt die Wahrscheinlichkeitsverteilung
        zweier Zufallsvariablen X und Y.
        
        **Wichtige Konzepte:**
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Randverteilungen:**
            - f_X(x) = ∫ f(x,y) dy
            - f_Y(y) = ∫ f(x,y) dx
            
            **Bedingte Verteilung:**
            """)
            st.latex(r"f(y|x) = \frac{f(x,y)}{f_X(x)}")
        
        with col2:
            st.markdown("""
            **Für die bivariate Normalverteilung:**
            """)
            st.latex(r"f(x,y) = \frac{1}{2\pi\sigma_X\sigma_Y\sqrt{1-\rho^2}} \exp\left(-\frac{z}{2(1-\rho^2)}\right)")
            st.latex(r"z = \frac{(x-\mu_X)^2}{\sigma_X^2} - \frac{2\rho(x-\mu_X)(y-\mu_Y)}{\sigma_X\sigma_Y} + \frac{(y-\mu_Y)^2}{\sigma_Y^2}")
        
        # Interactive correlation slider
        st.markdown("### 🎛️ Interaktive Bivariate Normalverteilung")
        rho_slider = st.slider("Korrelation ρ", -0.99, 0.99, 0.0, 0.05, key="bivar_rho")
        
        # Generate bivariate normal data
        _render_bivariate_normal_3d(rho_slider)
    
    # 🔗 Stochastische Unabhängigkeit
    with st.expander("🔗 Stochastische Unabhängigkeit"):
        st.markdown("""
        X und Y sind **stochastisch unabhängig** wenn:
        """)
        st.latex(r"f(x,y) = f_X(x) \cdot f_Y(y)")
        st.latex(r"\text{oder äquivalent: } \rho_{XY} = 0 \text{ (für Normalverteilungen)}")
        
        st.warning("""
        ⚠️ **Wichtig:** Unabhängigkeit impliziert keine Korrelation, aber keine Korrelation
        impliziert **nicht** unbedingt Unabhängigkeit (es sei denn bei Normalverteilung)!
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**ρ = 0, Unabhängig** ✅")
            st.image("https://via.placeholder.com/200x150.png?text=Independent", width=200)
        with col2:
            st.markdown("**ρ = 0, NICHT unabhängig** ❌")
            st.markdown("*Beispiel: Y = X²*")
        with col3:
            st.markdown("**ρ ≠ 0, Abhängig** ✅")
    
    # =========================================================================
    # CHAPTER 2.0: DAS FUNDAMENT - REGRESSIONSMODELL
    # =========================================================================
    st.markdown("---")
    st.markdown('<p class="section-header">📖 Kapitel 2.0: Das Fundament - Das einfache lineare Regressionsmodell</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("""
        Das **einfache lineare Regressionsmodell** beschreibt den Zusammenhang 
        zwischen einer unabhängigen Variable X und einer abhängigen Variable Y:
        """)
        
        if show_formulas:
            st.latex(r"Y_i = \beta_0 + \beta_1 X_i + \varepsilon_i")
        
        st.markdown("""
        | Symbol | Bedeutung | Eigenschaften |
        |--------|-----------|---------------|
        | Yᵢ | Abhängige Variable | Beobachtete Werte |
        | Xᵢ | Unabhängige Variable | Erklärende Variable |
        | β₀ | Intercept | Y-Achsenabschnitt |
        | β₁ | Steigung | Effekt von X auf Y |
        | εᵢ | Störterm | E(ε)=0, Var(ε)=σ² |
        """)
    
    with col2:
        st.info(f"""
        ### 💡 Praxisbeispiel: {data.context_title}
        
        **Unabhängige Variable (X):** {data.x_label}  
        **Abhängige Variable (Y):** {data.y_label}
        
        **Erwartung:** {data.context_description}
        """)
    
    # 📊 Rohdaten-Visualisierung
    st.markdown("### 📊 Die Rohdaten visualisieren")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        **Was zeigt dieser Plot?**
        
        Ein Streudiagramm (Scatter Plot) der {n} Beobachtungen. Jeder Punkt repräsentiert 
        eine Messung mit Werten für {data.x_label} (x-Achse) und {data.y_label} (y-Achse).
        
        **Worauf achten?**
        - Gibt es einen **Trend**? (aufsteigend/absteigend)
        - Wie **eng** liegen die Punkte beieinander?
        - Gibt es **Ausreisser**?
        """)
        
        # Create raw scatter plot
        fig_raw = _create_raw_scatter(x, y, data)
        st.plotly_chart(fig_raw, use_container_width=True, key="raw_scatter")
    
    with col2:
        st.markdown("### 📊 Deskriptive Statistik")
        
        desc_df = pd.DataFrame({
            'Statistik': ['Mittelwert', 'Std.Abw.', 'Min', 'Max'],
            data.x_label: [f"{np.mean(x):.2f}", f"{np.std(x, ddof=1):.2f}", 
                          f"{np.min(x):.2f}", f"{np.max(x):.2f}"],
            data.y_label: [f"{np.mean(y):.2f}", f"{np.std(y, ddof=1):.2f}",
                          f"{np.min(y):.2f}", f"{np.max(y):.2f}"],
        })
        st.dataframe(desc_df, hide_index=True, use_container_width=True)
        
        # Correlation preview
        corr = np.corrcoef(x, y)[0, 1]
        st.metric("Korrelation r", f"{corr:.4f}")
    
    # =========================================================================
    # CHAPTER 2.5: KOVARIANZ & KORRELATION
    # =========================================================================
    st.markdown("---")
    st.markdown('<p class="section-header">📖 Kapitel 2.5: Kovarianz & Korrelation - Die Bausteine der Regression</p>', unsafe_allow_html=True)
    
    # 📐 Die Kovarianz
    with st.expander("📐 Die Kovarianz", expanded=True):
        st.markdown("""
        Die **Kovarianz** misst den linearen Zusammenhang zwischen zwei Variablen.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.latex(r"\text{Cov}(X,Y) = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})")
            
            cov_xy = np.cov(x, y, ddof=1)[0, 1]
            st.metric("Cov(X,Y)", f"{cov_xy:.4f}")
            
            st.markdown("""
            **Interpretation:**
            - Cov > 0: Positive Beziehung
            - Cov < 0: Negative Beziehung  
            - Cov = 0: Keine lineare Beziehung
            """)
        
        with col2:
            # 3D Kovarianz-Visualisierung
            _render_covariance_3d(x, y, data)
    
    # 📊 Der Korrelationskoeffizient
    with st.expander("📊 Der Korrelationskoeffizient (Pearson)"):
        st.markdown("""
        Der **Korrelationskoeffizient** ist die standardisierte Kovarianz:
        """)
        
        st.latex(r"r = \frac{\text{Cov}(X,Y)}{s_X \cdot s_Y} = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum(x_i-\bar{x})^2 \sum(y_i-\bar{y})^2}}")
        
        corr = np.corrcoef(x, y)[0, 1]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("Korrelation r", f"{corr:.4f}")
            st.metric("r²", f"{corr**2:.4f}")
            
            if abs(corr) > 0.7:
                st.success("✅ Starke Korrelation")
            elif abs(corr) > 0.4:
                st.info("ℹ️ Mittlere Korrelation")
            else:
                st.warning("⚠️ Schwache Korrelation")
        
        with col2:
            # 6-Panel Korrelations-Beispiele
            _render_correlation_examples()
    
    # 🔬 Signifikanztest für die Korrelation
    with st.expander("🔬 Signifikanztest für die Korrelation"):
        corr = np.corrcoef(x, y)[0, 1]
        t_corr = corr * np.sqrt((n - 2) / (1 - corr**2))
        p_corr = 2 * (1 - stats.t.cdf(abs(t_corr), df=n-2))
        
        st.markdown("""
        **Hypothesen:**
        - H₀: ρ = 0 (keine Korrelation in der Population)
        - H₁: ρ ≠ 0 (es gibt eine Korrelation)
        
        **Teststatistik:**
        """)
        st.latex(r"t = \frac{r\sqrt{n-2}}{\sqrt{1-r^2}}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("t-Wert", f"{t_corr:.3f}")
            st.metric("p-Wert", f"{p_corr:.4f}")
        with col2:
            if p_corr < 0.05:
                st.success(f"✅ Korrelation signifikant bei α=0.05")
            else:
                st.warning(f"⚠️ Korrelation nicht signifikant")
    
    # Bonus: Spearman Rangkorrelation
    with st.expander("📊 Bonus: Spearman Rangkorrelation"):
        spearman_r, spearman_p = stats.spearmanr(x, y)
        
        st.markdown("""
        Die **Spearman Rangkorrelation** ist robust gegen Ausreisser und misst
        monotone (nicht nur lineare) Zusammenhänge.
        """)
        st.latex(r"r_s = 1 - \frac{6\sum d_i^2}{n(n^2-1)}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Spearman ρ", f"{spearman_r:.4f}")
            st.metric("p-Wert", f"{spearman_p:.4f}")
        with col2:
            st.markdown(f"""
            **Vergleich:**
            | Methode | Korrelation |
            |---------|-------------|
            | Pearson | {corr:.4f} |
            | Spearman | {spearman_r:.4f} |
            | Differenz | {abs(corr - spearman_r):.4f} |
            """)
    
    # =========================================================================
    # CHAPTER 3.0: DIE METHODE - OLS-SCHÄTZUNG
    # =========================================================================
    st.markdown("---")
    st.markdown('<p class="section-header">📖 Kapitel 3.0: Die Methode - Schätzung mittels OLS</p>', unsafe_allow_html=True)
    
    st.markdown("""
    **Ordinary Least Squares (OLS)** minimiert die Summe der quadrierten Residuen:
    """)
    st.latex(r"\min_{b_0, b_1} \sum_{i=1}^{n}(y_i - b_0 - b_1 x_i)^2 = \min SSE")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 OLS Visualisierung")
        st.markdown("""
        Die **roten Linien** zeigen die Residuen - die vertikalen Abstände 
        zwischen den Datenpunkten und der Regressionsgerade.
        
        **OLS minimiert die Summe der QUADRATE dieser Abstände.**
        """)
        
        # Main regression plot with residuals
        st.plotly_chart(plots.scatter, use_container_width=True, key="ols_plot")
    
    with col2:
        st.markdown("### 📐 Die OLS-Schätzer")
        
        if show_formulas:
            st.latex(r"b_1 = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sum(x_i-\bar{x})^2} = \frac{\text{Cov}(X,Y)}{\text{Var}(X)}")
            st.latex(r"b_0 = \bar{y} - b_1\bar{x}")
        
        st.markdown(f"""
        **Berechnete Werte:**
        
        | Schätzer | Formel | Wert |
        |----------|--------|------|
        | b₁ | Cov(X,Y)/Var(X) | {result.slope:.4f} |
        | b₀ | ȳ - b₁x̄ | {result.intercept:.4f} |
        """)
        
        # Show the equation
        sign = "+" if result.slope >= 0 else ""
        st.success(f"**ŷ = {result.intercept:.4f} {sign} {result.slope:.4f} · x**")
    
    # =========================================================================
    # CHAPTER 3.1: REGRESSIONSMODELL IM DETAIL
    # =========================================================================
    st.markdown("---")
    st.markdown('<p class="section-header">📖 Kapitel 3.1: Das Regressionsmodell im Detail - Anatomie & Unsicherheit</p>', unsafe_allow_html=True)
    
    with st.expander("🔍 Die Anatomie des Modells", expanded=True):
        st.markdown("""
        Jede Beobachtung lässt sich zerlegen in:
        """)
        st.latex(r"y_i = \underbrace{\hat{y}_i}_{\text{Fitted}} + \underbrace{e_i}_{\text{Residuum}}")
        st.latex(r"y_i = \underbrace{(\bar{y})}_{\text{Mittelwert}} + \underbrace{(\hat{y}_i - \bar{y})}_{\text{Erklärt}} + \underbrace{(y_i - \hat{y}_i)}_{\text{Unerklärt}}")
        
        # Visualisierung der Zerlegung
        _render_decomposition_plot(x, y, result, data)
    
    with st.expander("📏 3D Konfidenz-Trichter"):
        st.markdown("""
        Die Unsicherheit unserer Schätzung wird visualisiert durch:
        - **Konfidenzband**: Unsicherheit der LINIE
        - **Prognoseband**: Unsicherheit einzelner PUNKTE
        """)
        _render_confidence_funnel_3d(x, y, result, data)
    
    with st.expander("📖 Interpretation der Ergebnisse"):
        st.markdown(f"""
        ### 📊 Vollständige Interpretation
        
        **Das Modell:**
        
        {data.y_label} = {result.intercept:.4f} + {result.slope:.4f} × {data.x_label}
        
        **Interpretation des Intercepts (β₀ = {result.intercept:.4f}):**
        
        Wenn {data.x_label} = 0, dann erwarten wir {data.y_label} = {result.intercept:.2f} {data.y_unit}.
        ⚠️ Diese Interpretation ist nur sinnvoll wenn X=0 im relevanten Bereich liegt!
        
        **Interpretation der Steigung (β₁ = {result.slope:.4f}):**
        
        Für jede Einheit Zunahme in {data.x_label} erwarten wir:
        - Eine {"Zunahme" if result.slope > 0 else "Abnahme"} von **{abs(result.slope):.4f}** {data.y_unit} in {data.y_label}
        
        **Praktisches Beispiel:**
        - Bei {data.x_label} = {np.mean(x):.2f}: ŷ = {result.intercept + result.slope * np.mean(x):.2f}
        - Bei {data.x_label} = {np.mean(x) + np.std(x):.2f}: ŷ = {result.intercept + result.slope * (np.mean(x) + np.std(x)):.2f}
        """)
    
    # =========================================================================
    # CHAPTER 4.0: DIE GÜTEPRÜFUNG
    # =========================================================================
    st.markdown("---")
    st.markdown('<p class="section-header">📖 Kapitel 4.0: Die Güteprüfung - Validierung des Regressionsmodells</p>', unsafe_allow_html=True)
    
    # 4.1 Standardfehler der Regression
    with st.expander("📐 4.1 Standardfehler der Regression (sₑ)", expanded=True):
        se_regression = np.sqrt(result.mse)
        
        st.markdown(f"""
        Der **Standardfehler der Regression** (auch: Root Mean Square Error) misst
        die durchschnittliche Abweichung der Beobachtungen von der Regressionsgerade.
        """)
        
        st.latex(r"s_e = \sqrt{\frac{SSE}{n-2}} = \sqrt{\frac{\sum e_i^2}{n-2}} = \sqrt{MSE}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("sₑ (RMSE)", f"{se_regression:.4f}")
            st.metric("MSE", f"{result.mse:.4f}")
            st.metric("SSE", f"{result.sse:.4f}")
        
        with col2:
            st.info(f"""
            **Interpretation:**
            
            Die typische Abweichung vom vorhergesagten Wert 
            beträgt etwa **±{se_regression:.2f}** {data.y_unit}.
            
            Ca. 68% der Beobachtungen liegen innerhalb von ±sₑ.
            Ca. 95% liegen innerhalb von ±2·sₑ = ±{2*se_regression:.2f}.
            """)
    
    # 4.1b Standardfehler der Koeffizienten
    with st.expander("📐 4.1b Standardfehler der Koeffizienten"):
        st.markdown("""
        Die Standardfehler der Koeffizienten messen die **Unsicherheit** unserer Schätzer.
        """)
        
        st.latex(r"SE(b_0) = s_e \sqrt{\frac{1}{n} + \frac{\bar{x}^2}{\sum(x_i-\bar{x})^2}}")
        st.latex(r"SE(b_1) = \frac{s_e}{\sqrt{\sum(x_i-\bar{x})^2}}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("SE(b₀)", f"{result.se_intercept:.4f}")
            st.metric("SE(b₁)", f"{result.se_slope:.4f}")
        with col2:
            st.markdown(f"""
            **95% Konfidenzintervalle:**
            
            | Parameter | Schätzwert | 95% KI |
            |-----------|------------|--------|
            | β₀ | {result.intercept:.4f} | [{result.intercept - 1.96*result.se_intercept:.4f}, {result.intercept + 1.96*result.se_intercept:.4f}] |
            | β₁ | {result.slope:.4f} | [{result.slope - 1.96*result.se_slope:.4f}, {result.slope + 1.96*result.se_slope:.4f}] |
            """)
        
        # Interactive SE visualization
        _render_se_visualization(x, y, result, data)
    
    # 4.2 Bestimmtheitsmass (R²)
    with st.expander("📊 4.2 Bestimmtheitsmass (R²)", expanded=True):
        st.markdown(f"""
        Das **Bestimmtheitsmass R²** gibt an, welcher Anteil der Varianz in Y
        durch das Modell erklärt wird.
        """)
        
        st.latex(r"R^2 = 1 - \frac{SSE}{SST} = \frac{SSR}{SST}")
        
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
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
                height=350
            )
            st.plotly_chart(fig_var, use_container_width=True, key="variance_decomp")
        
        with col2:
            st.markdown("### 📊 Interpretation")
            st.metric("R²", f"{result.r_squared:.4f}")
            st.metric("R² adj.", f"{result.r_squared_adj:.4f}")
            
            st.markdown(f"""
            **{result.r_squared * 100:.1f}%** der Varianz in {data.y_label} 
            wird durch {data.x_label} erklärt.
            
            **{(1-result.r_squared) * 100:.1f}%** bleiben unerklärt.
            """)
            
            if result.r_squared > 0.8:
                st.success("✅ Sehr gute Anpassung")
            elif result.r_squared > 0.5:
                st.info("ℹ️ Akzeptable Anpassung")
            else:
                st.warning("⚠️ Schwache Anpassung")
    
    # =========================================================================
    # CHAPTER 5.0: DIE SIGNIFIKANZ
    # =========================================================================
    st.markdown("---")
    st.markdown('<p class="section-header">📖 Kapitel 5.0: Die Signifikanz - Statistische Inferenz und Hypothesentests</p>', unsafe_allow_html=True)
    
    # 📋 Gauss-Markov Annahmen
    with st.expander("📋 Voraussetzungen: Die Gauss-Markov Annahmen", expanded=True):
        st.markdown("""
        Damit OLS **BLUE** ist (Best Linear Unbiased Estimator), müssen folgende Annahmen erfüllt sein:
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **1. Linearität**
            - E(Y|X) = β₀ + β₁X
            - Der Zusammenhang ist linear in den Parametern
            
            **2. Strikt exogene Regressoren**
            - E(ε|X) = 0
            - Die Fehler sind unkorreliert mit X
            
            **3. Keine perfekte Multikollinearität**
            - Var(X) > 0
            - X hat Variation
            """)
        with col2:
            st.markdown("""
            **4. Homoskedastizität**
            - Var(ε|X) = σ² (konstant)
            - Die Varianz der Fehler ist konstant
            
            **5. Keine Autokorrelation**
            - Cov(εᵢ, εⱼ) = 0 für i ≠ j
            - Fehler sind unabhängig
            
            **6. Normalverteilung** (für Inferenz)
            - ε ~ N(0, σ²)
            """)
        
        # 4-Panel Annahmen-Visualisierung
        _render_assumptions_4panel(x, y, result, data)
    
    # Interactive Annahmen-Verletzung Demo
    with st.expander("🎛️ Interaktiv: Was passiert bei Annahmenverletzung?"):
        _render_assumption_violation_demo()
    
    # 🔬 Der t-Test
    with st.expander("🔬 Der t-Test für die Koeffizienten", expanded=True):
        st.markdown(f"""
        ### t-Test für die Steigung β₁
        
        **H₀:** β₁ = 0 (kein Effekt)  
        **H₁:** β₁ ≠ 0 (es gibt einen Effekt)
        
        **Teststatistik:**
        """)
        
        if show_formulas:
            st.latex(rf"t = \frac{{b_1 - 0}}{{SE(b_1)}} = \frac{{{result.slope:.4f}}}{{{result.se_slope:.4f}}} = {result.t_slope:.3f}")
        
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            _render_t_test_plot(result)
        
        with col2:
            st.markdown("### 📋 Koeffizienten-Tabelle")
            
            coef_df = pd.DataFrame({
                'Parameter': ['β₀ (Intercept)', f'β₁ ({data.x_label})'],
                'Schätzwert': [f"{result.intercept:.4f}", f"{result.slope:.4f}"],
                'Std.Error': [f"{result.se_intercept:.4f}", f"{result.se_slope:.4f}"],
                't-Wert': [f"{result.t_intercept:.3f}", f"{result.t_slope:.3f}"],
                'p-Wert': [f"{result.p_intercept:.4f}", f"{result.p_slope:.4f}"],
                'Signif.': [_get_stars(result.p_intercept), _get_stars(result.p_slope)],
            })
            st.dataframe(coef_df, hide_index=True, use_container_width=True)
            st.caption("Signif.: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1")
    
    # ⚖️ Der F-Test
    with st.expander("⚖️ Der F-Test"):
        f_stat = (result.ssr / 1) / (result.sse / result.df)
        p_f = 1 - stats.f.cdf(f_stat, dfn=1, dfd=result.df)
        
        st.markdown("""
        Der **F-Test** testet die Gesamtsignifikanz des Modells.
        
        **H₀:** β₁ = 0 (Modell erklärt nichts)  
        **H₁:** β₁ ≠ 0 (Modell erklärt etwas)
        """)
        
        st.latex(r"F = \frac{MSR}{MSE} = \frac{SSR/k}{SSE/(n-k-1)}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("F-Statistik", f"{f_stat:.3f}")
            st.metric("p-Wert", f"{p_f:.4f}")
            st.metric("df1, df2", f"1, {result.df}")
        with col2:
            st.info(f"""
            **Hinweis:** Bei einfacher Regression gilt:
            
            F = t² = {result.t_slope**2:.3f}
            
            Der F-Test und t-Test führen zum selben Ergebnis!
            """)
            
            if p_f < 0.05:
                st.success("✅ Modell ist signifikant")
            else:
                st.warning("⚠️ Modell ist nicht signifikant")
    
    # 📊 ANOVA-Tabelle
    with st.expander("📊 Die vollständige ANOVA-Tabelle"):
        msr = result.ssr / 1
        mse = result.sse / result.df
        f_stat = msr / mse
        
        anova_df = pd.DataFrame({
            'Quelle': ['Regression', 'Residuen', 'Total'],
            'SS': [f"{result.ssr:.4f}", f"{result.sse:.4f}", f"{result.sst:.4f}"],
            'df': [1, result.df, n-1],
            'MS': [f"{msr:.4f}", f"{mse:.4f}", ""],
            'F': [f"{f_stat:.4f}", "", ""],
            'p-Wert': [f"{1 - stats.f.cdf(f_stat, 1, result.df):.4f}", "", ""]
        })
        st.dataframe(anova_df, hide_index=True, use_container_width=True)
    
    # 💻 R-Style Output
    with st.expander("💻 Der komplette R-Style Output"):
        _render_r_style_output(result, data, n)
    
    # =========================================================================
    # CHAPTER 5.5: ANOVA FÜR GRUPPENVERGLEICHE
    # =========================================================================
    st.markdown("---")
    st.markdown('<p class="section-header">📖 Kapitel 5.5: ANOVA für Gruppenvergleiche</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Die **Analysis of Variance (ANOVA)** erweitert den t-Test auf mehr als zwei Gruppen.
    
    **Frage:** Unterscheiden sich die Mittelwerte von k Gruppen?
    """)
    
    with st.expander("🔬 Interaktives ANOVA-Beispiel", expanded=True):
        _render_anova_interactive()
    
    with st.expander("📊 3D Verteilungslandschaft"):
        _render_anova_3d_landscape()
    
    # =========================================================================
    # CHAPTER 5.6: HETEROSKEDASTIZITÄT
    # =========================================================================
    st.markdown("---")
    st.markdown('<p class="section-header">📖 Kapitel 5.6: Das grosse Problem - Heteroskedastizität</p>', unsafe_allow_html=True)
    
    st.warning("""
    **Heteroskedastizität** liegt vor, wenn die Varianz der Fehler nicht konstant ist.
    Dies verletzt die Gauss-Markov Annahme 4 und führt zu:
    - Ineffizienten Schätzern
    - Falschen Standardfehlern
    - Invaliden t- und F-Tests
    """)
    
    with st.expander("📊 Trichter-Effekt visualisieren", expanded=True):
        _render_heteroskedasticity_demo()
    
    with st.expander("🔧 Robuste Standardfehler (HC3)"):
        _render_robust_se_comparison(x, y, result, data)
    
    # =========================================================================
    # CHAPTER 6.0: FAZIT UND AUSBLICK
    # =========================================================================
    st.markdown("---")
    st.markdown('<p class="section-header">📖 Kapitel 6.0: Fazit und Ausblick</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.success(f"""
        ### 📝 Zusammenfassung der Analyse
        
        **Regressionsgleichung:**  
        {data.y_label} = {result.intercept:.4f} + {result.slope:.4f} × {data.x_label}
        
        **Interpretation:**
        - R² = {result.r_squared:.4f} → {result.r_squared*100:.1f}% der Varianz erklärt
        - Pro Einheit {data.x_label} ändert sich {data.y_label} um {result.slope:.4f}
        - Die Steigung ist {"✅ signifikant" if result.p_slope < 0.05 else "⚠️ nicht signifikant"} (p = {result.p_slope:.4f})
        
        **Stichprobe:** n = {n} Beobachtungen
        """)
        
        st.markdown("""
        ### ✅ Checkliste für gute Regression
        
        - [ ] Linearität überprüft (Residuen vs. Fitted)
        - [ ] Normalität der Residuen (Q-Q Plot)
        - [ ] Homoskedastizität (keine Trichterform)
        - [ ] Keine Ausreisser/Einflussreiche Punkte
        - [ ] Unabhängigkeit der Beobachtungen
        - [ ] R² interpretiert
        - [ ] Koeffizienten interpretiert
        - [ ] Signifikanz geprüft
        """)
    
    with col2:
        st.markdown("### 📊 Daten")
        with st.expander("Datentabelle anzeigen"):
            df = pd.DataFrame({
                data.x_label: x,
                data.y_label: y,
                'ŷ (Predicted)': result.y_pred,
                'Residuum': result.residuals,
            })
            st.dataframe(df.style.format("{:.4f}"), use_container_width=True)
    
    # 🌊 Bonusgrafik
    with st.expander("🌊 Bonusgrafik: Die bedingte Verteilung f(y|x)"):
        _render_conditional_distribution_3d(x, y, result, data)
    
    logger.info("Simple regression educational tab rendered completely")


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


def _create_raw_scatter(x: np.ndarray, y: np.ndarray, data: DataResult) -> go.Figure:
    """Create raw scatter plot with mean lines."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers',
        marker=dict(size=10, color='#3498db', opacity=0.7),
        name='Datenpunkte'
    ))
    
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    fig.add_hline(y=y_mean, line_dash="dash", line_color="orange", 
                  annotation_text=f"ȳ = {y_mean:.2f}")
    fig.add_vline(x=x_mean, line_dash="dash", line_color="green",
                  annotation_text=f"x̄ = {x_mean:.2f}")
    
    fig.add_trace(go.Scatter(
        x=[x_mean], y=[y_mean], mode='markers',
        marker=dict(size=15, color='red', symbol='x'),
        name=f'Schwerpunkt ({x_mean:.2f}, {y_mean:.2f})'
    ))
    
    fig.update_layout(
        title="Schritt 1: Visualisierung der Rohdaten",
        xaxis_title=data.x_label,
        yaxis_title=data.y_label,
        template="plotly_white",
    )
    return fig


def _render_bivariate_normal_3d(rho: float) -> None:
    """Render 3D bivariate normal distribution."""
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    
    # Bivariate normal PDF
    det = 1 - rho**2
    z_val = X**2 - 2*rho*X*Y + Y**2
    Z = (1 / (2 * np.pi * np.sqrt(det))) * np.exp(-z_val / (2 * det))
    
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')])
    fig.update_layout(
        title=f"Bivariate Normalverteilung (ρ = {rho:.2f})",
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='f(x,y)'),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True, key=f"bivar_3d_{rho}")


def _render_covariance_3d(x: np.ndarray, y: np.ndarray, data: DataResult) -> None:
    """Render 3D covariance visualization."""
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Create deviation products
    dev_x = x - x_mean
    dev_y = y - y_mean
    products = dev_x * dev_y
    
    # Bar colors based on sign
    colors = ['green' if p > 0 else 'red' for p in products]
    
    fig = go.Figure()
    
    for i in range(min(len(x), 20)):  # Limit for clarity
        fig.add_trace(go.Scatter3d(
            x=[x[i], x[i]], y=[y[i], y[i]], z=[0, products[i]],
            mode='lines',
            line=dict(color=colors[i], width=5),
            showlegend=False
        ))
    
    fig.add_trace(go.Scatter3d(
        x=x[:20], y=y[:20], z=products[:20],
        mode='markers',
        marker=dict(size=5, color=products[:20], colorscale='RdYlGn'),
        name='Kovarianz-Beiträge'
    ))
    
    fig.update_layout(
        title="3D Kovarianz-Visualisierung",
        scene=dict(
            xaxis_title=data.x_label,
            yaxis_title=data.y_label,
            zaxis_title='(x-x̄)(y-ȳ)'
        ),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True, key="cov_3d")


def _render_correlation_examples() -> None:
    """Render 6-panel correlation examples."""
    fig = make_subplots(rows=2, cols=3, subplot_titles=[
        'r = -0.95', 'r = -0.50', 'r = 0.00',
        'r = 0.50', 'r = 0.95', 'r = 0.00 (nonlinear)'
    ])
    
    np.random.seed(42)
    n = 50
    
    correlations = [-0.95, -0.50, 0.0, 0.50, 0.95]
    
    for i, rho in enumerate(correlations):
        row = i // 3 + 1
        col = i % 3 + 1
        
        # Generate correlated data
        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        data_gen = np.random.multivariate_normal(mean, cov, n)
        
        fig.add_trace(go.Scatter(
            x=data_gen[:, 0], y=data_gen[:, 1],
            mode='markers', marker=dict(size=5),
            showlegend=False
        ), row=row, col=col)
    
    # Nonlinear example (quadratic)
    x_nl = np.linspace(-2, 2, n)
    y_nl = x_nl**2 + np.random.normal(0, 0.3, n)
    fig.add_trace(go.Scatter(
        x=x_nl, y=y_nl, mode='markers', marker=dict(size=5),
        showlegend=False
    ), row=2, col=3)
    
    fig.update_layout(height=400, title_text="Korrelations-Beispiele")
    st.plotly_chart(fig, use_container_width=True, key="corr_examples")


def _render_decomposition_plot(x: np.ndarray, y: np.ndarray, result: RegressionResult, data: DataResult) -> None:
    """Render decomposition of observations."""
    y_mean = np.mean(y)
    
    # Select a representative point
    idx = len(x) // 2
    
    fig = go.Figure()
    
    # Data point
    fig.add_trace(go.Scatter(
        x=[x[idx]], y=[y[idx]], mode='markers',
        marker=dict(size=15, color='blue'), name=f'yᵢ = {y[idx]:.2f}'
    ))
    
    # Mean line
    fig.add_hline(y=y_mean, line_dash="dash", line_color="gray",
                  annotation_text=f"ȳ = {y_mean:.2f}")
    
    # Fitted value
    fig.add_trace(go.Scatter(
        x=[x[idx]], y=[result.y_pred[idx]], mode='markers',
        marker=dict(size=15, color='green', symbol='diamond'),
        name=f'ŷᵢ = {result.y_pred[idx]:.2f}'
    ))
    
    # Regression line
    x_line = np.linspace(min(x), max(x), 100)
    y_line = result.intercept + result.slope * x_line
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode='lines',
        line=dict(color='red'), name='Regressionsgerade'
    ))
    
    # Decomposition arrows
    fig.add_trace(go.Scatter(
        x=[x[idx], x[idx]], y=[y_mean, result.y_pred[idx]],
        mode='lines', line=dict(color='green', width=3),
        name=f'Erklärt: {result.y_pred[idx] - y_mean:.2f}'
    ))
    
    fig.add_trace(go.Scatter(
        x=[x[idx], x[idx]], y=[result.y_pred[idx], y[idx]],
        mode='lines', line=dict(color='red', width=3),
        name=f'Residuum: {result.residuals[idx]:.2f}'
    ))
    
    fig.update_layout(
        title="Zerlegung einer Beobachtung",
        xaxis_title=data.x_label,
        yaxis_title=data.y_label,
        template="plotly_white",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True, key="decomp_plot")


def _render_confidence_funnel_3d(x: np.ndarray, y: np.ndarray, result: RegressionResult, data: DataResult) -> None:
    """Render 3D confidence funnel."""
    n = len(x)
    x_mean = np.mean(x)
    mse = result.mse
    ss_x = np.sum((x - x_mean)**2)
    
    x_sorted = np.sort(x)
    y_pred = result.intercept + result.slope * x_sorted
    
    # Standard error of prediction
    se_fit = np.sqrt(mse * (1/n + (x_sorted - x_mean)**2 / ss_x))
    se_pred = np.sqrt(mse * (1 + 1/n + (x_sorted - x_mean)**2 / ss_x))
    
    t_crit = stats.t.ppf(0.975, df=result.df)
    
    ci_lower = y_pred - t_crit * se_fit
    ci_upper = y_pred + t_crit * se_fit
    pi_lower = y_pred - t_crit * se_pred
    pi_upper = y_pred + t_crit * se_pred
    
    fig = go.Figure()
    
    # Data points
    fig.add_trace(go.Scatter3d(
        x=x, y=np.zeros(n), z=y,
        mode='markers', marker=dict(size=5, color='blue'),
        name='Datenpunkte'
    ))
    
    # Regression line
    fig.add_trace(go.Scatter3d(
        x=x_sorted, y=np.zeros(len(x_sorted)), z=y_pred,
        mode='lines', line=dict(color='red', width=4),
        name='Regressionsgerade'
    ))
    
    # Confidence band (as surface)
    fig.add_trace(go.Scatter3d(
        x=x_sorted, y=np.ones(len(x_sorted)), z=ci_upper,
        mode='lines', line=dict(color='green', width=2),
        name='95% Konfidenzband'
    ))
    fig.add_trace(go.Scatter3d(
        x=x_sorted, y=np.ones(len(x_sorted)), z=ci_lower,
        mode='lines', line=dict(color='green', width=2),
        showlegend=False
    ))
    
    fig.update_layout(
        title="3D Konfidenz-Trichter",
        scene=dict(
            xaxis_title=data.x_label,
            yaxis_title='',
            zaxis_title=data.y_label
        ),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True, key="conf_funnel_3d")


def _render_se_visualization(x: np.ndarray, y: np.ndarray, result: RegressionResult, data: DataResult) -> None:
    """Render interactive SE visualization."""
    st.markdown("### 📏 Visualisierung der Konfidenzintervalle")
    
    n = len(x)
    x_mean = np.mean(x)
    mse = result.mse
    ss_x = np.sum((x - x_mean)**2)
    
    x_line = np.linspace(min(x), max(x), 100)
    y_pred = result.intercept + result.slope * x_line
    
    se_fit = np.sqrt(mse * (1/n + (x_line - x_mean)**2 / ss_x))
    
    ci_mult = st.slider("Konfidenz-Multiplikator", 1.0, 3.0, 1.96, 0.1, key="se_mult")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers',
        marker=dict(size=8, color='blue'),
        name='Daten'
    ))
    
    fig.add_trace(go.Scatter(
        x=x_line, y=y_pred, mode='lines',
        line=dict(color='red', width=2),
        name='Regression'
    ))
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_line, x_line[::-1]]),
        y=np.concatenate([y_pred + ci_mult * se_fit, (y_pred - ci_mult * se_fit)[::-1]]),
        fill='toself', fillcolor='rgba(0,100,200,0.2)',
        line=dict(color='rgba(0,0,0,0)'),
        name=f'±{ci_mult:.2f} SE'
    ))
    
    fig.update_layout(
        title=f"Konfidenzband (±{ci_mult:.2f} × SE)",
        xaxis_title=data.x_label,
        yaxis_title=data.y_label,
        template="plotly_white",
        height=350
    )
    st.plotly_chart(fig, use_container_width=True, key="se_viz")


def _render_assumptions_4panel(x: np.ndarray, y: np.ndarray, result: RegressionResult, data: DataResult) -> None:
    """Render 4-panel assumption diagnostics."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            '1. Linearität: Residuen vs. Fitted',
            '2. Normalität: Q-Q Plot',
            '3. Homoskedastizität: Scale-Location',
            '4. Einfluss: Residuen vs. Leverage'
        ]
    )
    
    # Panel 1: Residuals vs Fitted
    fig.add_trace(go.Scatter(
        x=result.y_pred, y=result.residuals,
        mode='markers', marker=dict(color='blue', size=6),
        showlegend=False
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # Panel 2: Q-Q Plot
    sorted_res = np.sort(result.residuals)
    n = len(sorted_res)
    theoretical_q = stats.norm.ppf(np.arange(1, n+1) / (n+1))
    
    fig.add_trace(go.Scatter(
        x=theoretical_q, y=sorted_res,
        mode='markers', marker=dict(color='blue', size=6),
        showlegend=False
    ), row=1, col=2)
    
    # Add reference line
    fig.add_trace(go.Scatter(
        x=[-3, 3], y=[-3 * np.std(result.residuals), 3 * np.std(result.residuals)],
        mode='lines', line=dict(color='red', dash='dash'),
        showlegend=False
    ), row=1, col=2)
    
    # Panel 3: Scale-Location
    sqrt_std_res = np.sqrt(np.abs(result.residuals / np.std(result.residuals)))
    fig.add_trace(go.Scatter(
        x=result.y_pred, y=sqrt_std_res,
        mode='markers', marker=dict(color='blue', size=6),
        showlegend=False
    ), row=2, col=1)
    
    # Panel 4: Leverage
    x_mat = np.column_stack([np.ones(len(x)), x])
    hat_matrix = x_mat @ np.linalg.inv(x_mat.T @ x_mat) @ x_mat.T
    leverage = np.diag(hat_matrix)
    
    fig.add_trace(go.Scatter(
        x=leverage, y=result.residuals / np.std(result.residuals),
        mode='markers', marker=dict(color='blue', size=6),
        showlegend=False
    ), row=2, col=2)
    
    fig.update_layout(height=600, title_text="Diagnose-Plots: Gauss-Markov Annahmen")
    st.plotly_chart(fig, use_container_width=True, key="assumption_4panel")


def _render_assumption_violation_demo() -> None:
    """Interactive demo of assumption violations."""
    st.markdown("### Was passiert bei Annahmenverletzung?")
    
    violation_type = st.selectbox(
        "Wähle eine Verletzung:",
        ["Keine (Normal)", "Heteroskedastizität", "Nicht-Linearität", "Ausreisser"],
        key="violation_type"
    )
    
    np.random.seed(42)
    n = 100
    x_demo = np.random.uniform(0, 10, n)
    
    if violation_type == "Keine (Normal)":
        y_demo = 2 + 3 * x_demo + np.random.normal(0, 2, n)
    elif violation_type == "Heteroskedastizität":
        y_demo = 2 + 3 * x_demo + np.random.normal(0, 1, n) * x_demo
    elif violation_type == "Nicht-Linearität":
        y_demo = 2 + 3 * x_demo + 0.5 * x_demo**2 + np.random.normal(0, 2, n)
    else:  # Ausreisser
        y_demo = 2 + 3 * x_demo + np.random.normal(0, 2, n)
        y_demo[0] = 100
        y_demo[1] = -50
    
    # Fit simple regression
    b1_demo = np.cov(x_demo, y_demo, ddof=1)[0, 1] / np.var(x_demo, ddof=1)
    b0_demo = np.mean(y_demo) - b1_demo * np.mean(x_demo)
    y_pred_demo = b0_demo + b1_demo * x_demo
    residuals_demo = y_demo - y_pred_demo
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=x_demo, y=y_demo, mode='markers'))
        x_line = np.linspace(0, 10, 100)
        fig1.add_trace(go.Scatter(x=x_line, y=b0_demo + b1_demo * x_line, 
                                   mode='lines', line=dict(color='red')))
        fig1.update_layout(title="Daten + Regression", height=300)
        st.plotly_chart(fig1, use_container_width=True, key=f"viol_scatter_{violation_type}")
    
    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=y_pred_demo, y=residuals_demo, mode='markers'))
        fig2.add_hline(y=0, line_dash="dash", line_color="red")
        fig2.update_layout(title="Residuen vs. Fitted", height=300)
        st.plotly_chart(fig2, use_container_width=True, key=f"viol_resid_{violation_type}")


def _render_t_test_plot(result: RegressionResult) -> None:
    """Render t-test visualization."""
    x_t = np.linspace(-5, max(5, abs(result.t_slope) + 1), 200)
    y_t = stats.t.pdf(x_t, df=result.df)
    t_crit = stats.t.ppf(0.975, df=result.df)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_t, y=y_t, mode='lines', name='t-Verteilung',
                             line=dict(color='black', width=2)))
    
    # Shade rejection regions
    x_left = x_t[x_t < -t_crit]
    x_right = x_t[x_t > t_crit]
    
    fig.add_trace(go.Scatter(
        x=x_left, y=stats.t.pdf(x_left, df=result.df),
        fill='tozeroy', fillcolor='rgba(255,0,0,0.3)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Ablehnungsbereich'
    ))
    fig.add_trace(go.Scatter(
        x=x_right, y=stats.t.pdf(x_right, df=result.df),
        fill='tozeroy', fillcolor='rgba(255,0,0,0.3)',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False
    ))
    
    # Mark observed t
    fig.add_vline(x=result.t_slope, line_color='blue', line_width=3,
                  annotation_text=f"t = {result.t_slope:.2f}")
    
    fig.update_layout(
        title=f"t-Test (df={result.df}): p = {result.p_slope:.4f}",
        xaxis_title="t-Wert",
        yaxis_title="Dichte",
        template="plotly_white",
        height=350
    )
    st.plotly_chart(fig, use_container_width=True, key="t_test_plot")


def _render_r_style_output(result: RegressionResult, data: DataResult, n: int) -> None:
    """Render R-style regression output."""
    st.code(f"""
Call:
lm(formula = {data.y_label} ~ {data.x_label})

Residuals:
     Min       1Q   Median       3Q      Max 
{np.min(result.residuals):8.4f} {np.percentile(result.residuals, 25):8.4f} {np.median(result.residuals):8.4f} {np.percentile(result.residuals, 75):8.4f} {np.max(result.residuals):8.4f}

Coefficients:
              Estimate Std. Error t value Pr(>|t|)    
(Intercept)  {result.intercept:9.4f}   {result.se_intercept:9.4f}  {result.t_intercept:7.3f}  {result.p_intercept:8.4f} {_get_stars(result.p_intercept)}
{data.x_label:12s} {result.slope:9.4f}   {result.se_slope:9.4f}  {result.t_slope:7.3f}  {result.p_slope:8.4f} {_get_stars(result.p_slope)}
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: {np.sqrt(result.mse):.4f} on {result.df} degrees of freedom
Multiple R-squared:  {result.r_squared:.4f},	Adjusted R-squared:  {result.r_squared_adj:.4f}
F-statistic: {(result.ssr/1)/(result.sse/result.df):.2f} on 1 and {result.df} DF,  p-value: {result.p_slope:.4e}
""", language="text")


def _render_anova_interactive() -> None:
    """Render interactive ANOVA example."""
    st.markdown("### Vergleich von 3 Gruppen")
    
    col1, col2 = st.columns(2)
    
    with col1:
        mean_a = st.slider("Mittelwert Gruppe A", 50.0, 70.0, 60.0, key="anova_mean_a")
        mean_b = st.slider("Mittelwert Gruppe B", 50.0, 70.0, 65.0, key="anova_mean_b")
        mean_c = st.slider("Mittelwert Gruppe C", 50.0, 70.0, 70.0, key="anova_mean_c")
    
    np.random.seed(42)
    n_per_group = 20
    group_a = np.random.normal(mean_a, 5, n_per_group)
    group_b = np.random.normal(mean_b, 5, n_per_group)
    group_c = np.random.normal(mean_c, 5, n_per_group)
    
    f_stat, p_val = stats.f_oneway(group_a, group_b, group_c)
    
    with col2:
        st.metric("F-Statistik", f"{f_stat:.3f}")
        st.metric("p-Wert", f"{p_val:.4f}")
        
        if p_val < 0.05:
            st.success("✅ Mindestens ein Mittelwert ist signifikant verschieden")
        else:
            st.info("ℹ️ Keine signifikanten Unterschiede")
    
    fig = go.Figure()
    fig.add_trace(go.Box(y=group_a, name='Gruppe A', marker_color='blue'))
    fig.add_trace(go.Box(y=group_b, name='Gruppe B', marker_color='green'))
    fig.add_trace(go.Box(y=group_c, name='Gruppe C', marker_color='red'))
    
    fig.update_layout(title="Boxplots der drei Gruppen", template="plotly_white", height=350)
    st.plotly_chart(fig, use_container_width=True, key="anova_box")


def _render_anova_3d_landscape() -> None:
    """Render 3D ANOVA landscape."""
    np.random.seed(42)
    
    # Generate 3 groups
    groups = ['A', 'B', 'C']
    means = [60, 70, 80]
    
    x_all = []
    y_all = []
    z_all = []
    
    for i, (g, m) in enumerate(zip(groups, means)):
        x_vals = np.linspace(m-15, m+15, 100)
        y_density = stats.norm.pdf(x_vals, m, 5)
        
        x_all.extend(x_vals)
        y_all.extend([i] * len(x_vals))
        z_all.extend(y_density)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x_all, y=y_all, z=z_all,
        mode='markers',
        marker=dict(size=2, color=y_all, colorscale='Viridis')
    )])
    
    fig.update_layout(
        title="3D Verteilungslandschaft",
        scene=dict(
            xaxis_title='Wert',
            yaxis_title='Gruppe',
            zaxis_title='Dichte'
        ),
        height=450
    )
    st.plotly_chart(fig, use_container_width=True, key="anova_3d")


def _render_heteroskedasticity_demo() -> None:
    """Render heteroskedasticity demo."""
    np.random.seed(42)
    n = 100
    x_demo = np.random.uniform(1, 10, n)
    
    het_strength = st.slider("Heteroskedastizitäts-Stärke", 0.0, 2.0, 1.0, 0.1, key="het_str")
    
    y_demo = 10 + 2 * x_demo + np.random.normal(0, 1, n) * (1 + het_strength * x_demo)
    
    # Fit regression
    b1 = np.cov(x_demo, y_demo, ddof=1)[0, 1] / np.var(x_demo, ddof=1)
    b0 = np.mean(y_demo) - b1 * np.mean(x_demo)
    y_pred = b0 + b1 * x_demo
    residuals = y_demo - y_pred
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=x_demo, y=y_demo, mode='markers', name='Daten'))
        x_line = np.linspace(1, 10, 100)
        fig1.add_trace(go.Scatter(x=x_line, y=b0 + b1 * x_line,
                                   mode='lines', line=dict(color='red'), name='Regression'))
        fig1.update_layout(title="Daten mit Heteroskedastizität", height=300)
        st.plotly_chart(fig1, use_container_width=True, key="het_scatter")
    
    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers'))
        fig2.add_hline(y=0, line_dash="dash", line_color="red")
        fig2.update_layout(title="Residuen: Trichter-Effekt", height=300)
        st.plotly_chart(fig2, use_container_width=True, key="het_resid")
    
    if het_strength > 0.5:
        st.warning("⚠️ Deutliche Heteroskedastizität erkennbar - Standardfehler sind verzerrt!")


def _render_robust_se_comparison(x: np.ndarray, y: np.ndarray, result: RegressionResult, data: DataResult) -> None:
    """Render comparison of normal vs robust SE."""
    st.markdown("""
    ### Robuste Standardfehler (HC3)
    
    **HC3** (Heteroskedasticity-Consistent) Standardfehler sind robust gegen Heteroskedastizität:
    """)
    st.latex(r"SE_{HC3}(b_1) = \sqrt{\frac{\sum e_i^2 / (1-h_{ii})^2 \cdot (x_i - \bar{x})^2}{(\sum(x_i-\bar{x})^2)^2}}")
    
    # Calculate leverage
    n = len(x)
    x_mat = np.column_stack([np.ones(n), x])
    hat_matrix = x_mat @ np.linalg.inv(x_mat.T @ x_mat) @ x_mat.T
    leverage = np.diag(hat_matrix)
    
    # HC3 standard errors
    hc3_var = np.sum((result.residuals / (1 - leverage))**2 * (x - np.mean(x))**2) / (np.sum((x - np.mean(x))**2)**2)
    se_hc3 = np.sqrt(hc3_var)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Normale SE (OLS)")
        st.metric("SE(b₁)", f"{result.se_slope:.4f}")
        st.metric("t-Wert", f"{result.t_slope:.3f}")
        st.metric("p-Wert", f"{result.p_slope:.4f}")
    
    with col2:
        st.markdown("### Robuste SE (HC3)")
        t_robust = result.slope / se_hc3
        p_robust = 2 * (1 - stats.t.cdf(abs(t_robust), df=result.df))
        
        st.metric("SE(b₁)", f"{se_hc3:.4f}")
        st.metric("t-Wert", f"{t_robust:.3f}")
        st.metric("p-Wert", f"{p_robust:.4f}")
    
    diff_pct = (se_hc3 - result.se_slope) / result.se_slope * 100
    if abs(diff_pct) > 20:
        st.warning(f"⚠️ Die robusten SE weichen um {diff_pct:.1f}% ab - mögliche Heteroskedastizität!")
    else:
        st.success(f"✅ Die SE sind ähnlich (Differenz: {diff_pct:.1f}%) - keine starke Heteroskedastizität")


def _render_conditional_distribution_3d(x: np.ndarray, y: np.ndarray, result: RegressionResult, data: DataResult) -> None:
    """Render 3D conditional distribution f(y|x)."""
    st.markdown("""
    **Die bedingte Verteilung f(y|x)** zeigt die Verteilung von Y für jeden Wert von X.
    
    Bei homoskedastischen Fehlern haben alle bedingten Verteilungen dieselbe Varianz.
    """)
    
    # Generate conditional distributions
    x_vals = np.linspace(np.min(x), np.max(x), 5)
    se = np.sqrt(result.mse)
    
    fig = go.Figure()
    
    for x_val in x_vals:
        y_pred = result.intercept + result.slope * x_val
        y_range = np.linspace(y_pred - 3*se, y_pred + 3*se, 50)
        density = stats.norm.pdf(y_range, y_pred, se)
        
        # Scale density for visibility
        density_scaled = density / np.max(density) * (np.max(x) - np.min(x)) * 0.3
        
        fig.add_trace(go.Scatter3d(
            x=[x_val] * len(y_range),
            y=y_range,
            z=density_scaled,
            mode='lines',
            line=dict(width=3),
            name=f'f(y|x={x_val:.1f})'
        ))
    
    # Add regression line
    x_line = np.linspace(np.min(x), np.max(x), 100)
    y_line = result.intercept + result.slope * x_line
    
    fig.add_trace(go.Scatter3d(
        x=x_line, y=y_line, z=np.zeros_like(x_line),
        mode='lines', line=dict(color='red', width=4),
        name='E(Y|X)'
    ))
    
    fig.update_layout(
        title="3D Bedingte Verteilung f(y|x)",
        scene=dict(
            xaxis_title=data.x_label,
            yaxis_title=data.y_label,
            zaxis_title='Dichte'
        ),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True, key="cond_dist_3d")
