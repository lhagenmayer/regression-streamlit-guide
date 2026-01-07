"""
Simple Regression Educational Tab

This module renders the complete educational content for simple linear regression,
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
    
    # =========================================================
    # CHAPTER 1: EINLEITUNG
    # =========================================================
    st.markdown(f"""
    <p class="section-header">📖 Einfache Lineare Regression: {data.context_title}</p>
    """, unsafe_allow_html=True)
    
    st.info(f"""
    **Kontext:** {data.context_description}
    
    **Fragestellung:** Gibt es einen linearen Zusammenhang zwischen **{data.x_label}** und **{data.y_label}**?
    """)
    
    # Key metrics at a glance
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", f"{result.r_squared:.4f}", help="Erklärte Varianz")
    col2.metric("β₀", f"{result.intercept:.4f}", help="Y-Achsenabschnitt")
    col3.metric("β₁", f"{result.slope:.4f}", help="Steigung")
    col4.metric("n", f"{n}", help="Stichprobengrösse")
    
    # =========================================================
    # CHAPTER 2: DIE ROHDATEN - Plot 1
    # =========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">1️⃣ Die Rohdaten: Erste Visualisierung</p>', unsafe_allow_html=True)
    
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
        fig_raw = go.Figure()
        fig_raw.add_trace(go.Scatter(
            x=x, y=y, mode='markers',
            marker=dict(size=10, color='#3498db', opacity=0.7),
            name='Datenpunkte'
        ))
        
        # Add mean lines
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        fig_raw.add_hline(y=y_mean, line_dash="dash", line_color="orange", 
                         annotation_text=f"ȳ = {y_mean:.2f}")
        fig_raw.add_vline(x=x_mean, line_dash="dash", line_color="green",
                         annotation_text=f"x̄ = {x_mean:.2f}")
        
        # Mark centroid
        fig_raw.add_trace(go.Scatter(
            x=[x_mean], y=[y_mean], mode='markers',
            marker=dict(size=15, color='red', symbol='x'),
            name=f'Schwerpunkt ({x_mean:.2f}, {y_mean:.2f})'
        ))
        
        fig_raw.update_layout(
            title="Schritt 1: Visualisierung der Rohdaten",
            xaxis_title=data.x_label,
            yaxis_title=data.y_label,
            template="plotly_white",
        )
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
        
        # Correlation
        corr = np.corrcoef(x, y)[0, 1]
        st.metric("Korrelation r", f"{corr:.4f}")
        
        if abs(corr) > 0.7:
            st.success("✅ Starker linearer Zusammenhang")
        elif abs(corr) > 0.4:
            st.info("ℹ️ Mittlerer Zusammenhang")
        else:
            st.warning("⚠️ Schwacher Zusammenhang")
    
    # =========================================================
    # CHAPTER 3: OLS REGRESSION - Plot 2
    # =========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">2️⃣ Die Regressionsgerade: OLS-Schätzung</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        **Was zeigt dieser Plot?**
        
        Die **OLS-Regressionsgerade** (Ordinary Least Squares) minimiert die Summe der 
        quadrierten vertikalen Abstände zwischen Datenpunkten und Gerade.
        
        **Die Gleichung:**
        """)
        
        if show_formulas:
            sign = "+" if result.slope >= 0 else ""
            st.latex(rf"\hat{{y}} = {result.intercept:.4f} {sign} {result.slope:.4f} \cdot x")
        
        # Main regression plot
        st.plotly_chart(plots.scatter, use_container_width=True, key="regression_plot")
    
    with col2:
        st.markdown("### 📐 Interpretation")
        
        st.markdown(f"""
        **Intercept β₀ = {result.intercept:.4f}**
        
        Wenn {data.x_label} = 0, erwarten wir 
        {data.y_label} ≈ {result.intercept:.2f} {data.y_unit}
        
        ---
        
        **Steigung β₁ = {result.slope:.4f}**
        
        Pro Einheit {data.x_label} ändert sich 
        {data.y_label} um **{result.slope:.4f}** {data.y_unit}
        """)
        
        # Show true parameters if available
        if show_true_line and data.extra.get("true_slope", 0) != 0:
            st.markdown("---")
            st.markdown("### 🎯 Vergleich mit wahren Werten")
            
            true_b0 = data.extra.get("true_intercept", 0)
            true_b1 = data.extra.get("true_slope", 0)
            
            st.markdown(f"""
            | Parameter | Geschätzt | Wahr | Differenz |
            |-----------|-----------|------|-----------|
            | β₀ | {result.intercept:.4f} | {true_b0:.4f} | {result.intercept - true_b0:.4f} |
            | β₁ | {result.slope:.4f} | {true_b1:.4f} | {result.slope - true_b1:.4f} |
            """)
    
    # =========================================================
    # CHAPTER 4: MODELLGÜTE - Plot 3
    # =========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">3️⃣ Modellgüte: Wie gut passt die Gerade?</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown(f"""
        **R² = {result.r_squared:.4f}** bedeutet:
        
        **{result.r_squared * 100:.1f}%** der Varianz in {data.y_label} wird durch 
        {data.x_label} erklärt. Die restlichen **{(1-result.r_squared) * 100:.1f}%** 
        bleiben unerklärt (Residuen).
        """)
        
        if show_formulas:
            st.latex(r"R^2 = 1 - \frac{SSE}{SST} = \frac{SSR}{SST}")
        
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
        st.markdown("### 📊 Gütekennzahlen")
        
        metrics_df = pd.DataFrame({
            'Kennzahl': ['R²', 'R² adj.', 'RMSE', 'MAE'],
            'Wert': [
                f"{result.r_squared:.4f}",
                f"{result.r_squared_adj:.4f}",
                f"{np.sqrt(result.mse):.4f}",
                f"{np.mean(np.abs(result.residuals)):.4f}"
            ],
            'Interpretation': [
                f"{result.r_squared*100:.1f}% erklärt",
                "Korrigiert für n",
                f"±{np.sqrt(result.mse):.2f} {data.y_unit}",
                f"Ø Fehler: {np.mean(np.abs(result.residuals)):.2f}"
            ]
        })
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)
        
        # Quality assessment
        if result.r_squared > 0.8:
            st.success("✅ Sehr gute Anpassung")
        elif result.r_squared > 0.5:
            st.info("ℹ️ Akzeptable Anpassung")
        else:
            st.warning("⚠️ Schwache Anpassung")
    
    # =========================================================
    # CHAPTER 5: RESIDUENANALYSE - Plot 4 & 5
    # =========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">4️⃣ Residuenanalyse: Modellvalidierung</p>', unsafe_allow_html=True)
    
    st.markdown("""
    **Warum wichtig?** Die Residuen (Abweichungen zwischen beobachteten und vorhergesagten Werten) 
    zeigen, ob unser Modell die Daten gut beschreibt und die OLS-Annahmen erfüllt sind.
    
    **Was wir suchen:**
    - Zufällige Streuung um 0 (keine Muster!)
    - Konstante Varianz (Homoskedastizität)
    - Normalverteilung der Residuen
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Residuen vs. Fitted Values")
        st.markdown("""
        **Interpretation:** Punkte sollten zufällig um die 0-Linie streuen.
        Ein Trichter-Muster deutet auf Heteroskedastizität hin.
        """)
        st.plotly_chart(plots.residuals, use_container_width=True, key="resid_plot")
    
    with col2:
        st.markdown("### Diagnose-Plots")
        st.markdown("""
        **Q-Q Plot:** Punkte auf der Diagonale = Normalverteilung ✅
        **Histogram:** Glockenförmig und zentriert um 0
        """)
        if plots.diagnostics:
            st.plotly_chart(plots.diagnostics, use_container_width=True, key="diag_plots")
    
    # =========================================================
    # CHAPTER 6: SIGNIFIKANZ
    # =========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">5️⃣ Statistische Signifikanz: Hypothesentests</p>', unsafe_allow_html=True)
    
    st.markdown("""
    **Kernfrage:** Ist der gefundene Zusammenhang **statistisch signifikant** oder 
    könnte er rein zufällig entstanden sein?
    """)
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown(f"""
        ### t-Test für die Steigung β₁
        
        **H₀:** β₁ = 0 (kein Effekt)  
        **H₁:** β₁ ≠ 0 (es gibt einen Effekt)
        
        **Teststatistik:**
        """)
        
        if show_formulas:
            st.latex(rf"t = \frac{{b_1 - 0}}{{SE(b_1)}} = \frac{{{result.slope:.4f}}}{{{result.se_slope:.4f}}} = {result.t_slope:.3f}")
        
        # t-distribution plot
        x_t = np.linspace(-5, max(5, abs(result.t_slope) + 1), 200)
        y_t = stats.t.pdf(x_t, df=result.df)
        t_crit = stats.t.ppf(0.975, df=result.df)
        
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=x_t, y=y_t, mode='lines', name='t-Verteilung',
                                   line=dict(color='black', width=2)))
        
        # Shade rejection regions
        x_reject = x_t[np.abs(x_t) > t_crit]
        y_reject = stats.t.pdf(x_reject, df=result.df)
        fig_t.add_trace(go.Scatter(x=x_reject, y=y_reject, fill='tozeroy',
                                   fillcolor='rgba(255,0,0,0.3)', name='Ablehnungsbereich'))
        
        # Mark observed t
        fig_t.add_vline(x=result.t_slope, line_color='blue', line_width=3,
                        annotation_text=f"t = {result.t_slope:.2f}")
        
        fig_t.update_layout(
            title=f"t-Test (df={result.df}): p = {result.p_slope:.4f}",
            xaxis_title="t-Wert",
            yaxis_title="Dichte",
            template="plotly_white",
            height=350
        )
        st.plotly_chart(fig_t, use_container_width=True, key="t_test_plot")
    
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
        
        # Conclusion
        if result.p_slope < 0.05:
            st.success(f"""
            ✅ **Signifikant!** (p = {result.p_slope:.4f} < 0.05)
            
            Wir lehnen H₀ ab. Die Steigung ist signifikant von 0 verschieden.
            {data.x_label} hat einen statistisch nachweisbaren Effekt auf {data.y_label}.
            """)
        else:
            st.warning(f"""
            ⚠️ **Nicht signifikant** (p = {result.p_slope:.4f} ≥ 0.05)
            
            Wir können H₀ nicht ablehnen. Der beobachtete Zusammenhang 
            könnte zufällig sein.
            """)
    
    # =========================================================
    # CHAPTER 7: FAZIT
    # =========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">6️⃣ Zusammenfassung</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.success(f"""
        ### 📝 Ergebnisse der Analyse
        
        **Regressionsgleichung:**  
        {data.y_label} = {result.intercept:.4f} + {result.slope:.4f} × {data.x_label}
        
        **Interpretation:**
        - R² = {result.r_squared:.4f} → {result.r_squared*100:.1f}% der Varianz erklärt
        - Pro Einheit {data.x_label} ändert sich {data.y_label} um {result.slope:.4f}
        - Die Steigung ist {"signifikant" if result.p_slope < 0.05 else "nicht signifikant"} (p = {result.p_slope:.4f})
        
        **Stichprobe:** n = {n} Beobachtungen
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
    
    logger.info("Simple regression educational tab rendered")


def _get_stars(p: float) -> str:
    """Get significance stars."""
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    if p < 0.1: return "."
    return ""
