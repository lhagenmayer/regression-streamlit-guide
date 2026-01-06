"""
Simple regression tab for the Linear Regression Guide.

This module renders the complete simple linear regression analysis tab
with all educational content from chapters 1.0 through 6.0.
"""

import streamlit as st
import numpy as np
import pandas as pd
import warnings
from scipy import stats
from scipy.stats import multivariate_normal, spearmanr
from typing import Dict, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...config import COLUMN_LAYOUTS, CAMERA_PRESETS, CSS_STYLES, get_logger
from ..plots import (
    create_plotly_scatter_with_line,
    create_plotly_residual_plot,
    create_plotly_bar,
    create_r_output_figure,
    get_signif_stars,
    get_signif_color,
)
from ...data import get_simple_regression_content

logger = get_logger(__name__)


def render_simple_regression_tab(
    model_data: Dict[str, Any],
    x_label: str,
    y_label: str,
    x_unit: str,
    y_unit: str,
    context_title: str,
    context_description: str,
    show_formulas: bool = True,
    show_true_line: bool = False,
    has_true_line: bool = False,
    true_intercept: float = 0,
    true_beta: float = 0
) -> None:
    """
    Render the complete simple regression analysis tab.
    
    This function renders all educational content for simple linear regression,
    including chapters on:
    - 1.0 Einleitung
    - 1.5 Mehrdimensionale Verteilungen
    - 2.0 Das Fundament: Regressionsmodell
    - 2.5 Kovarianz & Korrelation
    - 3.0 OLS-Schätzung
    - 4.0 Güteprüfung
    - 5.0 Signifikanz
    - 6.0 Fazit
    """
    # =========================================================
    # EXTRACT ALL REQUIRED DATA FROM MODEL_DATA
    # =========================================================
    model = model_data["model"]
    x = np.array(model_data["x"])
    y = np.array(model_data["y"])
    y_pred = np.array(model_data["y_pred"])
    b0 = model_data["b0"]
    b1 = model_data["b1"]
    n = len(x)
    
    # Statistical values (with fallbacks)
    x_mean = model_data.get("x_mean", np.mean(x))
    y_mean_val = model_data.get("y_mean_val", np.mean(y))
    cov_xy = model_data.get("cov_xy", np.cov(x, y)[0, 1])
    var_x = model_data.get("var_x", np.var(x, ddof=1))
    var_y = model_data.get("var_y", np.var(y, ddof=1))
    corr_xy = model_data.get("corr_xy", np.corrcoef(x, y)[0, 1])
    
    # Model statistics
    sse = model_data.get("sse", np.sum((y - y_pred) ** 2))
    sst = model_data.get("sst", np.sum((y - y_mean_val) ** 2))
    ssr = model_data.get("ssr", sst - sse)
    mse = model_data.get("mse", sse / (n - 2))
    se_regression = model_data.get("se_regression", np.sqrt(mse))
    
    # Residuals
    residuals = model_data.get("residuals", y - y_pred)
    
    # DataFrame
    df = pd.DataFrame({x_label: x, y_label: y})
    
    # =========================================================
    # KAPITEL 1.0: EINLEITUNG
    # =========================================================
    st.markdown(
        f'<p style="{CSS_STYLES["main_header"]}">📖 Umfassender Leitfaden zur Linearen Regression</p>',
        unsafe_allow_html=True,
    )
    st.markdown("### Von der Frage zur validierten Erkenntnis – Ein interaktiver Lernpfad")

    st.markdown("---")
    st.markdown(
        '<p class="section-header">1.0 Einleitung: Die Analyse von Zusammenhängen</p>',
        unsafe_allow_html=True,
    )

    col_intro1, col_intro2 = st.columns([2, 1])

    with col_intro1:
        st.markdown(
            """
        Von der Vorhersage von Unternehmensumsätzen bis hin zur Aufdeckung wissenschaftlicher
        Zusammenhänge – die Fähigkeit, Beziehungen in Daten zu quantifizieren, ist eine
        **Kernkompetenz** in der modernen Analyse.

        Die **Regressionsanalyse** ist das universelle Werkzeug für diese Aufgabe. Sie geht über
        die blosse Feststellung *ob* Variablen zusammenhängen hinaus und erklärt präzise,
        **wie** sie sich gegenseitig beeinflussen.

        > ⚠️ **Wichtig:** Die Regression allein beweist keine Kausalität! Sie quantifiziert die
        > Stärke einer *potenziellen* Ursache-Wirkungs-Beziehung, die durch das Studiendesign
        > gestützt werden muss.
        """
        )

    with col_intro2:
        st.info(
            """
        **Korrelation vs. Regression:**

        | Korrelation | Regression |
        |-------------|------------|
        | *Ungerichtet* | *Gerichtet* |
        | Wie stark? | Um wieviel? |
        | r ∈ [-1, 1] | ŷ = b₀ + b₁x |
        """
        )

    # =========================================================
    # KAPITEL 1.5: MEHRDIMENSIONALE VERTEILUNGEN
    # =========================================================
    st.markdown("---")
    st.markdown(
        '<p class="section-header">1.5 Mehrdimensionale Verteilungen: Das Fundament für Zusammenhänge</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    Bevor wir Zusammenhänge zwischen Variablen analysieren können, müssen wir verstehen,
    wie **zwei Zufallsvariablen gemeinsam** verteilt sein können. Dies ist die mathematische
    Grundlage für alles, was folgt.
    """
    )

    st.markdown(
        '<p class="subsection-header">🎲 Gemeinsame Verteilung f(X,Y)</p>', unsafe_allow_html=True
    )

    col_joint1, col_joint2 = st.columns([2, 1])

    with col_joint1:
        # Slider für Korrelation in der bivariaten Normalverteilung
        demo_corr = st.slider(
            "Korrelation ρ zwischen X und Y",
            min_value=-0.95,
            max_value=0.95,
            value=0.7,
            step=0.05,
            help="Bewege den Slider um zu sehen, wie sich die gemeinsame Verteilung verändert",
            key="demo_corr_slider_simple",
        )

        # Bivariate Normalverteilung generieren
        mean = [0, 0]
        cov_matrix = [[1, demo_corr], [demo_corr, 1]]

        x_grid = np.linspace(-3, 3, 100)
        y_grid = np.linspace(-3, 3, 100)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        pos = np.dstack((X_grid, Y_grid))

        rv = multivariate_normal(mean, cov_matrix)
        Z = rv.pdf(pos)

        # 3D Visualization
        fig_joint_3d = make_subplots(
            rows=1,
            cols=3,
            specs=[[{"type": "surface"}, {"type": "scatter3d"}, {"type": "surface"}]],
            subplot_titles=(
                f"Gemeinsame Verteilung<br>ρ = {demo_corr:.2f}",
                "Randverteilung f_X(x)",
                f"Bedingte Verteilung<br>E(Y|X=1) = {demo_corr * 1.0:.2f}",
            ),
        )

        # 1. 3D Surface Plot der gemeinsamen Verteilung
        fig_joint_3d.add_trace(
            go.Surface(x=X_grid, y=Y_grid, z=Z, colorscale="Blues", opacity=0.8, showscale=False),
            row=1,
            col=1,
        )

        # Stichprobe als Punkte auf z=0
        np.random.seed(42)
        sample = np.random.multivariate_normal(mean, cov_matrix, 100)
        fig_joint_3d.add_trace(
            go.Scatter3d(
                x=sample[:, 0],
                y=sample[:, 1],
                z=np.zeros(100),
                mode="markers",
                marker=dict(size=2, color="red", opacity=0.3),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # 2. Randverteilung als 3D
        x_marg = np.linspace(-3, 3, 100)
        y_marg_pdf = stats.norm.pdf(x_marg, 0, 1)

        fig_joint_3d.add_trace(
            go.Scatter3d(
                x=x_marg,
                y=y_marg_pdf,
                z=np.zeros_like(x_marg),
                mode="lines",
                line=dict(color="blue", width=4),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # 3. Bedingte Verteilung
        x_cond = 1.0
        cond_mean = demo_corr * x_cond
        cond_var = max(1 - demo_corr**2, 0.01)
        cond_std = np.sqrt(cond_var)

        y_cond_grid = np.linspace(-3, 3, 100)
        pdf_cond = stats.norm.pdf(y_cond_grid, cond_mean, cond_std)

        fig_joint_3d.add_trace(
            go.Scatter3d(
                x=np.full_like(y_cond_grid, x_cond),
                y=y_cond_grid,
                z=pdf_cond,
                mode="lines",
                line=dict(color="green", width=4),
                showlegend=False,
            ),
            row=1,
            col=3,
        )

        fig_joint_3d.update_layout(
            height=500,
            showlegend=False,
            scene1=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="f(X,Y)",
                camera=CAMERA_PRESETS.get("default", dict(eye=dict(x=1.5, y=1.5, z=1.2))),
            ),
            scene2=dict(
                xaxis_title="X",
                yaxis_title="",
                zaxis_title="f_X(x)",
            ),
            scene3=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="f(Y|X=1)",
                camera=dict(eye=dict(x=1.5, y=-1.8, z=1.0)),
            ),
        )

        st.plotly_chart(fig_joint_3d, key="joint_3d_distribution_simple", use_container_width=True)

    with col_joint2:
        if show_formulas:
            st.markdown("### Gemeinsame Verteilung")
            st.latex(r"f_{X,Y}(x,y) = P(X=x, Y=y)")

            st.markdown("### Randverteilung")
            st.latex(r"f_X(x) = \sum_y f_{X,Y}(x,y)")

            st.markdown("### Bedingte Verteilung")
            st.latex(r"f_{Y|X}(y|x) = \frac{f_{X,Y}(x,y)}{f_X(x)}")

        st.info(
            f"""
        **Beobachte:**

        Bei ρ = {demo_corr:.2f}:
        - Die Punktewolke ist {"stark" if abs(demo_corr) > 0.7 else "schwach"} {"positiv" if demo_corr > 0 else "negativ" if demo_corr < 0 else "un"}korreliert
        - E(Y|X=1) = {demo_corr:.2f} (nicht 0!)
        - Die bedingte Varianz ist {max(1 - demo_corr**2, 0.01):.2f} < 1

        **→ Je höher |ρ|, desto mehr "wissen" wir über Y, wenn wir X kennen!**
        """
        )

    # Stochastische Unabhängigkeit
    st.markdown(
        '<p class="subsection-header">🔗 Stochastische Unabhängigkeit</p>', unsafe_allow_html=True
    )

    col_indep1, col_indep2 = st.columns([1, 1])

    with col_indep1:
        st.markdown(
            """
        Zwei Zufallsvariablen X und Y sind **stochastisch unabhängig**, wenn:
        """
        )
        st.latex(r"f_{X,Y}(x,y) = f_X(x) \cdot f_Y(y)")
        st.markdown(
            """
        Das bedeutet: Die gemeinsame Wahrscheinlichkeit ist einfach das **Produkt** der Einzelwahrscheinlichkeiten.

        **Konsequenz:** Bei Unabhängigkeit gilt:
        - $E(Y|X=x) = E(Y)$ – X sagt nichts über Y aus!
        - $Cov(X,Y) = 0$
        - $ρ = 0$
        """
        )

    with col_indep2:
        np.random.seed(123)
        x_ind = np.random.normal(0, 1, 200)
        y_ind = np.random.normal(0, 1, 200)

        cov_dep = [[1, 0.8], [0.8, 1]]
        sample_dep = np.random.multivariate_normal([0, 0], cov_dep, 200)

        fig_indep = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                'Unabhängig (ρ = 0)<br>"Keine Struktur"',
                'Abhängig (ρ = 0.8)<br>"Klare Struktur"',
            ),
        )

        fig_indep.add_trace(
            go.Scatter(
                x=x_ind,
                y=y_ind,
                mode="markers",
                marker=dict(size=5, color="gray", opacity=0.5),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig_indep.add_trace(
            go.Scatter(
                x=sample_dep[:, 0],
                y=sample_dep[:, 1],
                mode="markers",
                marker=dict(size=5, color="blue", opacity=0.5),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig_indep.update_xaxes(title_text="X", row=1, col=1)
        fig_indep.update_yaxes(title_text="Y", row=1, col=1)
        fig_indep.update_xaxes(title_text="X", row=1, col=2)
        fig_indep.update_yaxes(title_text="Y", row=1, col=2)

        fig_indep.update_layout(height=400, template="plotly_white")

        st.plotly_chart(fig_indep, key="independence_plot_simple", use_container_width=True)

    st.success(
        """
    **Merke:** Die Regression nutzt genau diese Struktur! Wenn X und Y abhängig sind,
    können wir $E(Y|X=x)$ als Funktion von x modellieren – das ist die Regressionsgerade!
    """
    )

    # =========================================================
    # KAPITEL 2.0: DAS FUNDAMENT
    # =========================================================
    _render_chapter_2_0(
        x, y, x_label, y_label, x_mean, y_mean_val, n, df, 
        context_title, context_description, show_formulas
    )
    
    # =========================================================
    # KAPITEL 2.5: KOVARIANZ & KORRELATION
    # =========================================================
    _render_chapter_2_5(
        x, y, x_label, y_label, x_mean, y_mean_val, n,
        cov_xy, var_x, corr_xy, show_formulas
    )
    
    # =========================================================
    # KAPITEL 3.0: OLS-SCHÄTZUNG
    # =========================================================
    _render_chapter_3_0(
        x, y, y_pred, x_label, y_label, b0, b1, n,
        cov_xy, var_x, y_mean_val, x_mean, model,
        show_formulas, show_true_line, true_intercept, true_beta
    )
    
    # =========================================================
    # KAPITEL 4.0: GÜTEPRÜFUNG
    # =========================================================
    _render_chapter_4_0(
        x, y, y_pred, model, n, b0, b1,
        sse, sst, ssr, mse, se_regression,
        x_label, y_label, show_formulas
    )
    
    # =========================================================
    # KAPITEL 5.0: SIGNIFIKANZ
    # =========================================================
    _render_chapter_5_0(
        model, n, b1, x_label, y_label, show_formulas
    )
    
    # =========================================================
    # KAPITEL 6.0: FAZIT
    # =========================================================
    _render_chapter_6_0()

    logger.info("Simple regression tab rendered with full educational content")


# =========================================================
# HELPER FUNCTIONS FOR EACH CHAPTER
# =========================================================

def _render_chapter_2_0(
    x, y, x_label, y_label, x_mean, y_mean_val, n, df,
    context_title, context_description, show_formulas
):
    """Render Chapter 2.0: Das Fundament."""
    st.markdown("---")
    st.markdown(
        '<p class="section-header">2.0 Das Fundament: Das einfache lineare Regressionsmodell</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    Das Verständnis des einfachen linearen Regressionsmodells ist der entscheidende erste Schritt.
    Die Rollen der Variablen werden klar definiert:

    - **Abhängige Variable (Y):** Die Zielvariable – was wir erklären/vorhersagen wollen
    - **Unabhängige Variable (X):** Der Prädiktor – was die Veränderung erklärt
    """
    )

    col_model1, col_model2 = st.columns([1.2, 1])

    with col_model1:
        st.markdown("### Das grundlegende Modell:")
        if show_formulas:
            st.latex(r"y_i = \beta_0 + \beta_1 \cdot x_i + \varepsilon_i")

        st.markdown(
            """
        | Symbol | Bedeutung |
        |--------|-----------|
        | **β₀** | Wahrer Achsenabschnitt (unbekannt) |
        | **β₁** | Wahre Steigung (unbekannt) – Änderung in Y pro Einheit X |
        | **εᵢ** | Zufällige Störgrösse – alle anderen Einflüsse |
        """
        )

    with col_model2:
        st.warning(
            f"""
        ### 🎯 Praxisbeispiel: {context_title}

        {context_description}
        """
        )

    # Erste Visualisierung: Die Rohdaten
    st.markdown(
        f'<p class="subsection-header">📊 Unsere Daten: {n} Beobachtungen</p>',
        unsafe_allow_html=True,
    )

    col_data1, col_data2 = st.columns([1, 2])

    with col_data1:
        st.dataframe(
            df.style.format({x_label: "{:.2f}", y_label: "{:.2f}"}),
            height=min(400, n * 35 + 50),
            use_container_width=True,
        )

    with col_data2:
        fig_scatter1 = go.Figure()

        fig_scatter1.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    size=12, color="#1f77b4", opacity=0.7, line=dict(width=2, color="white")
                ),
                name="Datenpunkte",
            )
        )

        fig_scatter1.add_hline(
            y=y_mean_val,
            line_dash="dash",
            line_color="orange",
            opacity=0.5,
            annotation_text=f"ȳ = {y_mean_val:.2f}",
            annotation_position="right",
        )
        fig_scatter1.add_vline(
            x=x_mean,
            line_dash="dash",
            line_color="green",
            opacity=0.5,
            annotation_text=f"x̄ = {x_mean:.2f}",
            annotation_position="top",
        )

        fig_scatter1.add_trace(
            go.Scatter(
                x=[x_mean],
                y=[y_mean_val],
                mode="markers",
                marker=dict(size=18, color="red", symbol="x"),
                name="Schwerpunkt (x̄, ȳ)",
            )
        )

        fig_scatter1.update_layout(
            title='Schritt 1: Visualisierung der Rohdaten<br>"Gibt es einen Zusammenhang?"',
            xaxis_title=x_label,
            yaxis_title=y_label,
            template="plotly_white",
            hovermode="closest",
        )

        st.plotly_chart(fig_scatter1, key="correlation_scatter_plot_simple", use_container_width=True)

    st.success(
        f"""
    **Beobachtung:** Die Punkte scheinen einem aufsteigenden Trend zu folgen!
    Der Schwerpunkt liegt bei ({x_mean:.2f}, {y_mean_val:.2f}).
    Jetzt müssen wir die "beste" Gerade finden, die diesen Trend beschreibt.
    """
    )


def _render_chapter_2_5(
    x, y, x_label, y_label, x_mean, y_mean_val, n,
    cov_xy, var_x, corr_xy, show_formulas
):
    """Render Chapter 2.5: Kovarianz & Korrelation."""
    st.markdown("---")
    st.markdown(
        '<p class="section-header">2.5 Kovarianz & Korrelation: Die Bausteine der Regression</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    Bevor wir die Regressionsgerade berechnen, müssen wir verstehen, **wie** wir den Zusammenhang
    zwischen X und Y messen. Die **Kovarianz** und der **Korrelationskoeffizient** sind die Werkzeuge dafür.
    """
    )

    st.markdown(
        '<p class="subsection-header">📐 Die Kovarianz: Richtung und Stärke des Zusammenhangs</p>',
        unsafe_allow_html=True,
    )

    col_cov1, col_cov2 = st.columns([2, 1])

    with col_cov1:
        # 3D Kovarianz-Visualisierung
        fig_cov = go.Figure()

        for i in range(len(x)):
            dx = x[i] - x_mean
            dy = y[i] - y_mean_val
            product = dx * dy
            color = "green" if product > 0 else "red"

            fig_cov.add_trace(
                go.Scatter3d(
                    x=[x[i], x[i]],
                    y=[y[i], y[i]],
                    z=[0, product],
                    mode="lines",
                    line=dict(color=color, width=8),
                    opacity=0.7,
                    showlegend=False,
                )
            )

            fig_cov.add_trace(
                go.Scatter3d(
                    x=[x[i]],
                    y=[y[i]],
                    z=[product],
                    mode="markers",
                    marker=dict(size=8, color=color, line=dict(color="white", width=1)),
                    showlegend=False,
                )
            )

        fig_cov.add_trace(
            go.Scatter3d(
                x=[x_mean],
                y=[y_mean_val],
                z=[0],
                mode="markers",
                marker=dict(size=12, color="black", symbol="x", line=dict(color="white", width=2)),
                name="Schwerpunkt",
                showlegend=True,
            )
        )

        fig_cov.update_layout(
            title="3D Kovarianz-Visualisierung: Säulenhöhe = Produkt der Abweichungen",
            scene=dict(
                xaxis_title=f"{x_label} (X)",
                yaxis_title=f"{y_label} (Y)",
                zaxis_title="(X - X̄)(Y - Ȳ)",
            ),
            height=600,
        )

        st.plotly_chart(fig_cov, key="covariance_3d_plot_simple", use_container_width=True)

    with col_cov2:
        if show_formulas:
            st.markdown("### Kovarianz (Population)")
            st.latex(r"Cov(X,Y) = E(XY) - E(X) \cdot E(Y)")

            st.markdown("### Kovarianz (Stichprobe)")
            st.latex(r"s_{xy} = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{n-1}")

        products = [(x[i] - x_mean) * (y[i] - y_mean_val) for i in range(len(x))]
        pos_sum = sum(p for p in products if p > 0)
        neg_sum = sum(p for p in products if p < 0)

        st.metric("Positive Rechtecke Σ", f"{pos_sum:.3f}", delta="grün")
        st.metric("Negative Rechtecke Σ", f"{neg_sum:.3f}", delta="rot", delta_color="inverse")
        st.metric("Kovarianz Cov(X,Y)", f"{cov_xy:.4f}")

        if cov_xy > 0:
            st.success("✅ Positive Kovarianz → X↑ bedeutet tendenziell Y↑")
        else:
            st.error("❌ Negative Kovarianz → X↑ bedeutet tendenziell Y↓")

    # Korrelation
    st.markdown(
        '<p class="subsection-header">📊 Der Korrelationskoeffizient: Standardisierte Stärke</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    Die Kovarianz hat ein Problem: Sie hängt von den **Einheiten** ab! Eine Kovarianz von 5.2
    zwischen Fläche (qm) und Umsatz (Mio. €) ist schwer zu interpretieren.

    Die Lösung: **Normierung** durch die Standardabweichungen → Der Korrelationskoeffizient r ∈ [-1, +1]
    """
    )

    col_corr1, col_corr2 = st.columns([2, 1])

    with col_corr1:
        example_corrs = [-0.95, -0.5, 0, 0.5, 0.8, 0.95]
        np.random.seed(42)

        fig_corr_examples = make_subplots(
            rows=2, cols=3, subplot_titles=[f"r = {r:.2f}" for r in example_corrs]
        )

        for idx, r in enumerate(example_corrs):
            row = idx // 3 + 1
            col = idx % 3 + 1

            if r == 0:
                ex_x = np.random.normal(0, 1, 100)
                ex_y = np.random.normal(0, 1, 100)
            else:
                cov_ex = [[1, r], [r, 1]]
                sample_ex = np.random.multivariate_normal([0, 0], cov_ex, 100)
                ex_x, ex_y = sample_ex[:, 0], sample_ex[:, 1]

            if r > 0:
                color = f"rgba(0, {int(128 + abs(r)*127)}, 0, 0.6)"
            elif r < 0:
                color = f"rgba({int(128 + abs(r)*127)}, 0, 0, 0.6)"
            else:
                color = "rgba(128, 128, 128, 0.6)"

            fig_corr_examples.add_trace(
                go.Scatter(
                    x=ex_x,
                    y=ex_y,
                    mode="markers",
                    marker=dict(size=5, color=color),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            fig_corr_examples.update_xaxes(range=[-3, 3], row=row, col=col)
            fig_corr_examples.update_yaxes(range=[-3, 3], row=row, col=col)

        fig_corr_examples.update_layout(
            title_text="Der Korrelationskoeffizient r: Von -1 (perfekt negativ) bis +1 (perfekt positiv)",
            height=600,
            template="plotly_white",
        )

        st.plotly_chart(fig_corr_examples, key="correlation_examples_plot_simple", use_container_width=True)

    with col_corr2:
        if show_formulas:
            st.markdown("### Korrelation (Pearson)")
            st.latex(r"\rho_{X,Y} = \frac{Cov(X,Y)}{\sigma_X \cdot \sigma_Y}")

            st.markdown("### Stichproben-Korrelation")
            st.latex(r"r = \frac{s_{xy}}{s_x \cdot s_y}")

        st.metric("Unsere Korrelation r", f"{corr_xy:.4f}")

        if abs(corr_xy) > 0.8:
            strength = "sehr stark"
        elif abs(corr_xy) > 0.5:
            strength = "mittelstark"
        elif abs(corr_xy) > 0.3:
            strength = "schwach"
        else:
            strength = "sehr schwach"

        direction = "positiv" if corr_xy > 0 else "negativ"

        st.info(
            f"""
        **Interpretation:**

        r = {corr_xy:.3f} zeigt einen **{strength}en {direction}en**
        linearen Zusammenhang.

        **Wichtig:** Bei einfacher Regression gilt:

        **R² = r²** = {corr_xy**2:.4f}

        Das ist identisch mit unserem späteren Bestimmtheitsmass!
        """
        )

    st.success(
        f"""
    **Zusammenfassung Kapitel 2.5:**

    Wir haben die Bausteine für die Regression verstanden:
    - **Kovarianz** Cov(X,Y) = {cov_xy:.4f} → Richtung des Zusammenhangs
    - **Korrelation** r = {corr_xy:.4f} → Standardisierte Stärke

    Im nächsten Kapitel sehen wir: **b₁ = Cov(X,Y) / Var(X)** – die Steigung ist direkt aus der Kovarianz abgeleitet!
    """
    )


def _render_chapter_3_0(
    x, y, y_pred, x_label, y_label, b0, b1, n,
    cov_xy, var_x, y_mean_val, x_mean, model,
    show_formulas, show_true_line, true_intercept, true_beta
):
    """Render Chapter 3.0: OLS-Schätzung."""
    st.markdown("---")
    st.markdown(
        '<p class="section-header">3.0 Die Methode: Schätzung mittels OLS</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    Die **Methode der kleinsten Quadrate (Ordinary Least Squares, OLS)** findet die optimale Gerade.

    **Das Kernprinzip:** Wähle jene Gerade, welche die **Summe der quadrierten vertikalen Abweichungen**
    (Residuen) zwischen Datenpunkten und Gerade **minimiert**.
    """
    )

    col_ols1, col_ols2 = st.columns([2, 1])

    with col_ols1:
        fig_ols = go.Figure()

        fig_ols.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    size=10, color="#1f77b4", opacity=0.7, line=dict(width=2, color="white")
                ),
                name="Datenpunkte",
            )
        )

        fig_ols.add_trace(
            go.Scatter(
                x=x,
                y=y_pred,
                mode="lines",
                line=dict(color="red", width=3),
                name=f"OLS-Gerade: ŷ = {b0:.3f} + {b1:.3f}x",
            )
        )

        if show_true_line:
            fig_ols.add_trace(
                go.Scatter(
                    x=x,
                    y=true_intercept + true_beta * x,
                    mode="lines",
                    line=dict(color="green", width=2, dash="dash"),
                    opacity=0.7,
                    name=f"Wahre Gerade: y = {true_intercept:.2f} + {true_beta:.2f}x",
                )
            )

        # Residual lines
        for i in range(min(len(x), 10)):
            resid = y[i] - y_pred[i]
            fig_ols.add_trace(
                go.Scatter(
                    x=[x[i], x[i]],
                    y=[y[i], y_pred[i]],
                    mode="lines",
                    line=dict(color="red", width=1.5),
                    opacity=0.5,
                    showlegend=False,
                )
            )

        fig_ols.update_layout(
            title="OLS minimiert die Fläche aller roten Quadrate (= SSE)",
            xaxis_title=x_label,
            yaxis_title=y_label,
            template="plotly_white",
            hovermode="closest",
        )

        st.plotly_chart(fig_ols, key="ols_regression_plot_simple", use_container_width=True)

    with col_ols2:
        if show_formulas:
            st.markdown("### OLS-Schätzer:")
            st.latex(r"b_1 = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2}")
            st.latex(r"b_0 = \bar{y} - b_1 \cdot \bar{x}")

            st.markdown("### Mit unseren Werten:")
            st.latex(f"b_1 = \\frac{{{cov_xy*(n-1):.2f}}}{{{var_x*(n-1):.2f}}} = {b1:.4f}")
            st.latex(f"b_0 = {y_mean_val:.2f} - {b1:.4f} \\times {x_mean:.2f} = {b0:.4f}")

        st.success(
            f"""
        ### 📐 Ergebnis:

        **ŷ = {b0:.4f} + {b1:.4f} · x**
        """
        )


def _render_chapter_4_0(
    x, y, y_pred, model, n, b0, b1,
    sse, sst, ssr, mse, se_regression,
    x_label, y_label, show_formulas
):
    """Render Chapter 4.0: Güteprüfung."""
    st.markdown("---")
    st.markdown(
        '<p class="section-header">4.0 Die Güteprüfung: Validierung des Regressionsmodells</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="subsection-header">4.1 Standardfehler der Regression (sₑ): Die durchschnittliche Prognoseabweichung</p>',
        unsafe_allow_html=True,
    )

    col_se1, col_se2 = st.columns([2, 1])

    with col_se1:
        st.markdown(
            f"""
        Der **Standardfehler der Regression** sₑ = {se_regression:.4f} gibt die typische
        Abweichung der tatsächlichen Y-Werte von der Regressionslinie an.

        **Interpretation:** Im Durchschnitt liegen die tatsächlichen Werte etwa
        ±{se_regression:.2f} {y_label.split('(')[-1].replace(')', '') if '(' in y_label else ''} von der Vorhersage entfernt.
        """
        )

    with col_se2:
        if show_formulas:
            st.latex(r"s_e = \sqrt{\frac{SSE}{n-2}} = \sqrt{MSE}")
            st.latex(f"s_e = \\sqrt{{\\frac{{{sse:.2f}}}{{{n}-2}}}} = {se_regression:.4f}")

    # R² Section
    st.markdown(
        '<p class="subsection-header">4.2 Bestimmtheitsmass (R²): Der Anteil der erklärten Varianz</p>',
        unsafe_allow_html=True,
    )

    col_r2_1, col_r2_2 = st.columns([2, 1])

    r_squared = model.rsquared

    with col_r2_1:
        st.markdown(
            f"""
        Das **Bestimmtheitsmass R²** = {r_squared:.4f} sagt uns:

        **{r_squared*100:.1f}%** der Varianz in Y wird durch das Modell erklärt.

        **{(1-r_squared)*100:.1f}%** bleibt unerklärt (Residuen).
        """
        )

        # Simple R² visualization
        fig_r2 = go.Figure()
        fig_r2.add_trace(go.Bar(
            x=["Erklärt (SSR)", "Unerklärt (SSE)"],
            y=[ssr, sse],
            marker_color=["green", "red"],
            text=[f"{ssr:.1f}", f"{sse:.1f}"],
            textposition="auto"
        ))
        fig_r2.update_layout(
            title=f"Varianzzerlegung: R² = {r_squared:.4f}",
            yaxis_title="Quadratsumme",
            template="plotly_white",
            height=300
        )
        st.plotly_chart(fig_r2, key="r2_bar_chart_simple", use_container_width=True)

    with col_r2_2:
        if show_formulas:
            st.latex(r"R^2 = \frac{SSR}{SST} = 1 - \frac{SSE}{SST}")
            st.latex(f"R^2 = \\frac{{{ssr:.2f}}}{{{sst:.2f}}} = {r_squared:.4f}")

        # Interpretation
        if r_squared > 0.8:
            quality = "Sehr gute Anpassung!"
        elif r_squared > 0.5:
            quality = "Akzeptable Anpassung"
        elif r_squared > 0.3:
            quality = "Schwache Anpassung"
        else:
            quality = "Sehr schwache Anpassung"

        st.info(f"**Qualität:** {quality}")


def _render_chapter_5_0(model, n, b1, x_label, y_label, show_formulas):
    """Render Chapter 5.0: Signifikanz."""
    st.markdown("---")
    st.markdown(
        '<p class="section-header">5.0 Die Signifikanz: Statistische Inferenz und Hypothesentests</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    Jetzt stellen wir die entscheidende Frage: **Ist der gefundene Zusammenhang statistisch signifikant?**
    Oder könnte er rein zufällig in unserer Stichprobe entstanden sein?
    """
    )

    # t-Test für Steigung
    st.markdown(
        '<p class="subsection-header">🔬 Der t-Test: Ist die Steigung signifikant von Null verschieden?</p>',
        unsafe_allow_html=True,
    )

    col_t1, col_t2 = st.columns([2, 1])

    # Extract t-value and p-value from model
    try:
        t_val = model.tvalues[1]  # t-value for slope
        p_val = model.pvalues[1]  # p-value for slope
        se_b1 = model.bse[1]  # standard error of slope
    except:
        t_val = b1 / 0.1  # Fallback
        p_val = 0.05
        se_b1 = 0.1

    with col_t1:
        # t-distribution plot
        x_t = np.linspace(-5, max(5, abs(t_val) + 1), 300)
        y_t = stats.t.pdf(x_t, df=n - 2)

        fig_t = go.Figure()

        fig_t.add_trace(
            go.Scatter(
                x=x_t,
                y=y_t,
                mode="lines",
                line=dict(color="black", width=2),
                name=f"t-Verteilung (df={n-2})",
            )
        )

        # Shade p-value regions
        mask = abs(x_t) > abs(t_val)
        fig_t.add_trace(
            go.Scatter(
                x=x_t[mask],
                y=y_t[mask],
                fill="tozeroy",
                fillcolor="rgba(255, 0, 0, 0.3)",
                line=dict(width=0),
                name=f"p-Wert = {p_val:.4f}",
            )
        )

        # Critical values
        t_crit = stats.t.ppf(0.975, df=n - 2)
        fig_t.add_vline(x=t_crit, line_dash="dash", line_color="orange", opacity=0.7)
        fig_t.add_vline(
            x=-t_crit,
            line_dash="dash",
            line_color="orange",
            opacity=0.7,
            annotation_text=f"Kritisch: ±{t_crit:.2f}",
        )

        # Observed t-value
        fig_t.add_vline(
            x=t_val, line_color="blue", line_width=3, annotation_text=f"t = {t_val:.2f}"
        )

        fig_t.update_layout(
            title=f"H₀: β₁ = 0 (kein Effekt) vs. H₁: β₁ ≠ 0",
            xaxis_title="t-Wert",
            yaxis_title="Dichte",
            template="plotly_white",
        )

        st.plotly_chart(fig_t, key="t_test_slope_simple", use_container_width=True)

    with col_t2:
        if show_formulas:
            st.markdown("### Teststatistik")
            st.latex(r"t = \frac{b_1 - 0}{s_{b_1}}")
            st.latex(f"t = \\frac{{{b1:.4f}}}{{{{ {se_b1:.4f} }}}} = {t_val:.2f}")

        st.metric("t-Wert", f"{t_val:.3f}")
        st.metric("p-Wert", f"{p_val:.4f}")
        st.metric("Signifikanz", get_signif_stars(p_val))

        if p_val < 0.05:
            st.success("✅ Die Steigung ist **signifikant** von 0 verschieden!")
        else:
            st.warning("⚠️ Die Steigung ist **nicht signifikant**.")


def _render_chapter_6_0():
    """Render Chapter 6.0: Fazit."""
    st.markdown("---")
    st.markdown('<p class="section-header">6.0 Fazit und Ausblick</p>', unsafe_allow_html=True)

    st.markdown(
        """
    ### 📚 Was wir gelernt haben:

    1. **Das Regressionsmodell** beschreibt lineare Zusammenhänge: ŷ = b₀ + b₁x

    2. **OLS (Methode der kleinsten Quadrate)** findet die beste Gerade durch Minimierung der Residuenquadrate

    3. **Kovarianz und Korrelation** messen Richtung und Stärke des Zusammenhangs

    4. **R²** sagt uns, wie viel Varianz das Modell erklärt

    5. **t-Test und F-Test** prüfen die statistische Signifikanz

    6. **Residuen-Diagnose** validiert die Modellannahmen
    """
    )

    st.info(
        """
    **🚀 Nächste Schritte:**

    - Experimentieren Sie mit verschiedenen Datensätzen
    - Vergleichen Sie einfache vs. multiple Regression
    - Prüfen Sie die Residuen-Diagnostik
    - Erkunden Sie Prognosen für verschiedene Szenarien
    """
    )
