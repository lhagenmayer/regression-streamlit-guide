"""
Streamlit Content Renderer - Interprets ContentStructure for Streamlit.

This renderer takes framework-agnostic ContentStructure and renders it
using Streamlit's API (st.markdown, st.metric, st.plotly_chart, etc.)
"""

import streamlit as st
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional, Callable

from ...content.structure import (
    ContentElement, Chapter, Section, EducationalContent,
    Markdown, Metric, MetricRow, Formula, Plot, Table,
    Columns, Expander, InfoBox, WarningBox, SuccessBox,
    CodeBlock, Divider, ElementType
)


class StreamlitContentRenderer:
    """
    Renders EducationalContent using Streamlit API.
    
    The same ContentStructure can be rendered by different renderers
    (StreamlitContentRenderer, HTMLContentRenderer, etc.)
    """
    
    def __init__(
        self, 
        plots: Dict[str, Any] = None,
        data: Dict[str, Any] = None,
        stats: Dict[str, Any] = None
    ):
        """
        Initialize renderer with plot figures and data.
        
        Args:
            plots: Dictionary of plot_key -> Plotly figure
            data: Dictionary with x, y arrays and metadata
            stats: Dictionary with regression statistics
        """
        self.plots = plots or {}
        self.data = data or {}
        self.stats = stats or {}
        
        # Map of interactive plot generators
        self._interactive_plots: Dict[str, Callable] = {
            "bivariate_normal_3d": self._render_bivariate_normal_3d,
            "covariance_3d": self._render_covariance_3d,
            "correlation_examples": self._render_correlation_examples,
            "raw_scatter": self._render_raw_scatter,
            "ols_regression": self._render_ols_regression,
            "decomposition": self._render_decomposition,
            "confidence_funnel_3d": self._render_confidence_funnel_3d,
            "se_visualization": self._render_se_visualization,
            "variance_decomposition": self._render_variance_decomposition,
            "assumptions_4panel": self._render_assumptions_4panel,
            "assumption_violation_demo": self._render_assumption_violation_demo,
            "t_test_plot": self._render_t_test_plot,
            "anova_interactive": self._render_anova_interactive,
            "anova_3d_landscape": self._render_anova_3d_landscape,
            "heteroskedasticity_demo": self._render_heteroskedasticity_demo,
            "conditional_distribution_3d": self._render_conditional_distribution_3d,
            "data_table": self._render_data_table,
        }
    
    def render(self, content: EducationalContent) -> None:
        """Render complete educational content."""
        # Title
        st.title(content.title)
        st.markdown(f"*{content.subtitle}*")
        
        # Render each chapter
        for chapter in content.chapters:
            self._render_chapter(chapter)
    
    def _render_chapter(self, chapter: Chapter) -> None:
        """Render a chapter with its sections."""
        st.markdown("---")
        st.markdown(f'<p class="section-header">{chapter.icon} Kapitel {chapter.number}: {chapter.title}</p>', 
                    unsafe_allow_html=True)
        
        for section in chapter.sections:
            self._render_element(section)
    
    def _render_element(self, element: ContentElement) -> None:
        """Render a single content element."""
        if isinstance(element, Markdown):
            st.markdown(element.text)
        
        elif isinstance(element, Metric):
            st.metric(element.label, element.value, help=element.help_text or None)
        
        elif isinstance(element, MetricRow):
            cols = st.columns(len(element.metrics))
            for col, metric in zip(cols, element.metrics):
                with col:
                    st.metric(metric.label, metric.value, help=metric.help_text or None)
        
        elif isinstance(element, Formula):
            if element.inline:
                st.markdown(f"${element.latex}$")
            else:
                st.latex(element.latex)
        
        elif isinstance(element, Plot):
            self._render_plot(element)
        
        elif isinstance(element, Table):
            self._render_table(element)
        
        elif isinstance(element, Columns):
            cols = st.columns(element.widths)
            for col, content in zip(cols, element.columns):
                with col:
                    for item in content:
                        self._render_element(item)
        
        elif isinstance(element, Expander):
            with st.expander(element.title, expanded=element.expanded):
                for item in element.content:
                    self._render_element(item)
        
        elif isinstance(element, InfoBox):
            st.info(element.content)
        
        elif isinstance(element, WarningBox):
            st.warning(element.content)
        
        elif isinstance(element, SuccessBox):
            st.success(element.content)
        
        elif isinstance(element, CodeBlock):
            st.code(element.code, language=element.language)
        
        elif isinstance(element, Divider):
            st.markdown("---")
        
        elif isinstance(element, Section):
            st.markdown(f"### {element.icon} {element.title}")
            for item in element.content:
                self._render_element(item)
    
    def _render_plot(self, plot: Plot) -> None:
        """Render a plot element."""
        key = plot.plot_key
        
        # Check if we have a pre-generated plot
        if key in self.plots:
            fig = self.plots[key]
            st.plotly_chart(fig, use_container_width=True, key=f"plot_{key}")
        
        # Check if we have an interactive plot generator
        elif key in self._interactive_plots:
            self._interactive_plots[key]()
        
        else:
            st.warning(f"Plot '{key}' nicht verfügbar")
    
    def _render_table(self, table: Table) -> None:
        """Render a table element."""
        import pandas as pd
        
        df = pd.DataFrame(table.rows, columns=table.headers)
        st.dataframe(df, hide_index=True, use_container_width=True)
        
        if table.caption:
            st.caption(table.caption)
    
    # =========================================================================
    # INTERACTIVE PLOT GENERATORS
    # =========================================================================
    
    def _render_bivariate_normal_3d(self) -> None:
        """Render interactive bivariate normal distribution."""
        rho = st.slider("Korrelation ρ", -0.99, 0.99, 0.0, 0.05, key="bivar_rho")
        
        x = np.linspace(-3, 3, 50)
        y = np.linspace(-3, 3, 50)
        X, Y = np.meshgrid(x, y)
        
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
    
    def _render_covariance_3d(self) -> None:
        """Render 3D covariance visualization."""
        x = self.data.get('x', np.array([]))
        y = self.data.get('y', np.array([]))
        
        if len(x) == 0:
            st.warning("Keine Daten für Kovarianz-Plot")
            return
        
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        dev_x = x - x_mean
        dev_y = y - y_mean
        products = dev_x * dev_y
        colors = ['green' if p > 0 else 'red' for p in products]
        
        fig = go.Figure()
        
        for i in range(min(len(x), 20)):
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
                xaxis_title=self.data.get('x_label', 'X'),
                yaxis_title=self.data.get('y_label', 'Y'),
                zaxis_title='(x-x̄)(y-ȳ)'
            ),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True, key="cov_3d")
    
    def _render_correlation_examples(self) -> None:
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
            mean = [0, 0]
            cov = [[1, rho], [rho, 1]]
            data_gen = np.random.multivariate_normal(mean, cov, n)
            
            fig.add_trace(go.Scatter(
                x=data_gen[:, 0], y=data_gen[:, 1],
                mode='markers', marker=dict(size=5),
                showlegend=False
            ), row=row, col=col)
        
        # Nonlinear example
        x_nl = np.linspace(-2, 2, n)
        y_nl = x_nl**2 + np.random.normal(0, 0.3, n)
        fig.add_trace(go.Scatter(
            x=x_nl, y=y_nl, mode='markers', marker=dict(size=5),
            showlegend=False
        ), row=2, col=3)
        
        fig.update_layout(height=400, title_text="Korrelations-Beispiele")
        st.plotly_chart(fig, use_container_width=True, key="corr_examples")
    
    def _render_raw_scatter(self) -> None:
        """Render raw data scatter plot."""
        x = self.data.get('x', np.array([]))
        y = self.data.get('y', np.array([]))
        
        if len(x) == 0:
            st.warning("Keine Daten für Scatter-Plot")
            return
        
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
            xaxis_title=self.data.get('x_label', 'X'),
            yaxis_title=self.data.get('y_label', 'Y'),
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True, key="raw_scatter")
    
    def _render_ols_regression(self) -> None:
        """Render OLS regression with residuals."""
        x = self.data.get('x', np.array([]))
        y = self.data.get('y', np.array([]))
        
        if len(x) == 0:
            st.warning("Keine Daten für Regression-Plot")
            return
        
        slope = self.stats.get('slope', 0)
        intercept = self.stats.get('intercept', 0)
        y_pred = intercept + slope * x
        
        fig = go.Figure()
        
        # Data points
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='markers',
            marker=dict(size=10, color='#3498db'),
            name='Datenpunkte'
        ))
        
        # Regression line
        x_line = np.linspace(min(x), max(x), 100)
        y_line = intercept + slope * x_line
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line, mode='lines',
            line=dict(color='red', width=2),
            name='Regressionsgerade'
        ))
        
        # Residuals
        for i in range(len(x)):
            fig.add_trace(go.Scatter(
                x=[x[i], x[i]], y=[y[i], y_pred[i]],
                mode='lines',
                line=dict(color='rgba(255,0,0,0.3)', width=1),
                showlegend=False
            ))
        
        fig.update_layout(
            title="OLS Regression",
            xaxis_title=self.data.get('x_label', 'X'),
            yaxis_title=self.data.get('y_label', 'Y'),
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True, key="ols_reg")
    
    def _render_decomposition(self) -> None:
        """Render decomposition of observation."""
        x = self.data.get('x', np.array([]))
        y = self.data.get('y', np.array([]))
        
        if len(x) == 0:
            return
        
        slope = self.stats.get('slope', 0)
        intercept = self.stats.get('intercept', 0)
        y_pred = intercept + slope * x
        y_mean = np.mean(y)
        
        idx = len(x) // 2
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=[x[idx]], y=[y[idx]], mode='markers',
            marker=dict(size=15, color='blue'), name=f'yᵢ = {y[idx]:.2f}'
        ))
        
        fig.add_hline(y=y_mean, line_dash="dash", line_color="gray",
                      annotation_text=f"ȳ = {y_mean:.2f}")
        
        fig.add_trace(go.Scatter(
            x=[x[idx]], y=[y_pred[idx]], mode='markers',
            marker=dict(size=15, color='green', symbol='diamond'),
            name=f'ŷᵢ = {y_pred[idx]:.2f}'
        ))
        
        x_line = np.linspace(min(x), max(x), 100)
        y_line = intercept + slope * x_line
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line, mode='lines',
            line=dict(color='red'), name='Regressionsgerade'
        ))
        
        fig.add_trace(go.Scatter(
            x=[x[idx], x[idx]], y=[y_mean, y_pred[idx]],
            mode='lines', line=dict(color='green', width=3),
            name=f'Erklärt: {y_pred[idx] - y_mean:.2f}'
        ))
        
        fig.add_trace(go.Scatter(
            x=[x[idx], x[idx]], y=[y_pred[idx], y[idx]],
            mode='lines', line=dict(color='red', width=3),
            name=f'Residuum: {y[idx] - y_pred[idx]:.2f}'
        ))
        
        fig.update_layout(
            title="Zerlegung einer Beobachtung",
            xaxis_title=self.data.get('x_label', 'X'),
            yaxis_title=self.data.get('y_label', 'Y'),
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True, key="decomp_plot")
    
    def _render_confidence_funnel_3d(self) -> None:
        """Render 3D confidence funnel."""
        x = self.data.get('x', np.array([]))
        y = self.data.get('y', np.array([]))
        
        if len(x) == 0:
            return
        
        n = len(x)
        x_mean = np.mean(x)
        slope = self.stats.get('slope', 0)
        intercept = self.stats.get('intercept', 0)
        mse = self.stats.get('mse', 1)
        df = self.stats.get('df', n-2)
        ss_x = np.sum((x - x_mean)**2)
        
        x_sorted = np.sort(x)
        y_pred = intercept + slope * x_sorted
        
        se_fit = np.sqrt(mse * (1/n + (x_sorted - x_mean)**2 / ss_x))
        t_crit = stats.t.ppf(0.975, df=df)
        
        ci_lower = y_pred - t_crit * se_fit
        ci_upper = y_pred + t_crit * se_fit
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter3d(
            x=x, y=np.zeros(n), z=y,
            mode='markers', marker=dict(size=5, color='blue'),
            name='Datenpunkte'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=x_sorted, y=np.zeros(len(x_sorted)), z=y_pred,
            mode='lines', line=dict(color='red', width=4),
            name='Regressionsgerade'
        ))
        
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
                xaxis_title=self.data.get('x_label', 'X'),
                yaxis_title='',
                zaxis_title=self.data.get('y_label', 'Y')
            ),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True, key="conf_funnel_3d")
    
    def _render_se_visualization(self) -> None:
        """Render SE confidence band visualization."""
        x = self.data.get('x', np.array([]))
        y = self.data.get('y', np.array([]))
        
        if len(x) == 0:
            return
        
        n = len(x)
        x_mean = np.mean(x)
        slope = self.stats.get('slope', 0)
        intercept = self.stats.get('intercept', 0)
        mse = self.stats.get('mse', 1)
        ss_x = np.sum((x - x_mean)**2)
        
        x_line = np.linspace(min(x), max(x), 100)
        y_pred = intercept + slope * x_line
        se_fit = np.sqrt(mse * (1/n + (x_line - x_mean)**2 / ss_x))
        
        ci_mult = st.slider("Konfidenz-Multiplikator", 1.0, 3.0, 1.96, 0.1, key="se_mult")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
                                  marker=dict(size=8, color='blue'), name='Daten'))
        
        fig.add_trace(go.Scatter(x=x_line, y=y_pred, mode='lines',
                                  line=dict(color='red', width=2), name='Regression'))
        
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_line, x_line[::-1]]),
            y=np.concatenate([y_pred + ci_mult * se_fit, (y_pred - ci_mult * se_fit)[::-1]]),
            fill='toself', fillcolor='rgba(0,100,200,0.2)',
            line=dict(color='rgba(0,0,0,0)'),
            name=f'±{ci_mult:.2f} SE'
        ))
        
        fig.update_layout(
            title=f"Konfidenzband (±{ci_mult:.2f} × SE)",
            xaxis_title=self.data.get('x_label', 'X'),
            yaxis_title=self.data.get('y_label', 'Y'),
            template="plotly_white",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True, key="se_viz")
    
    def _render_variance_decomposition(self) -> None:
        """Render variance decomposition bar chart."""
        sst = self.stats.get('sst', 0)
        ssr = self.stats.get('ssr', 0)
        sse = self.stats.get('sse', 0)
        r2 = self.stats.get('r_squared', 0)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["SST (Total)", "SSR (Erklärt)", "SSE (Unerklärt)"],
            y=[sst, ssr, sse],
            marker_color=["gray", "#2ecc71", "#e74c3c"],
            text=[f"{sst:.1f}", f"{ssr:.1f}", f"{sse:.1f}"],
            textposition="auto"
        ))
        fig.update_layout(
            title=f"Varianzzerlegung: R² = {r2:.4f}",
            yaxis_title="Quadratsumme",
            template="plotly_white",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True, key="variance_decomp")
    
    def _render_assumptions_4panel(self) -> None:
        """Render 4-panel assumption diagnostics."""
        x = self.data.get('x', np.array([]))
        y = self.data.get('y', np.array([]))
        
        if len(x) == 0:
            return
        
        slope = self.stats.get('slope', 0)
        intercept = self.stats.get('intercept', 0)
        y_pred = intercept + slope * x
        residuals = y - y_pred
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                '1. Linearität: Residuen vs. Fitted',
                '2. Normalität: Q-Q Plot',
                '3. Homoskedastizität: Scale-Location',
                '4. Einfluss: Residuen vs. Leverage'
            ]
        )
        
        # Panel 1
        fig.add_trace(go.Scatter(
            x=y_pred, y=residuals,
            mode='markers', marker=dict(color='blue', size=6),
            showlegend=False
        ), row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # Panel 2
        sorted_res = np.sort(residuals)
        n = len(sorted_res)
        theoretical_q = stats.norm.ppf(np.arange(1, n+1) / (n+1))
        
        fig.add_trace(go.Scatter(
            x=theoretical_q, y=sorted_res,
            mode='markers', marker=dict(color='blue', size=6),
            showlegend=False
        ), row=1, col=2)
        
        fig.add_trace(go.Scatter(
            x=[-3, 3], y=[-3 * np.std(residuals), 3 * np.std(residuals)],
            mode='lines', line=dict(color='red', dash='dash'),
            showlegend=False
        ), row=1, col=2)
        
        # Panel 3
        sqrt_std_res = np.sqrt(np.abs(residuals / np.std(residuals)))
        fig.add_trace(go.Scatter(
            x=y_pred, y=sqrt_std_res,
            mode='markers', marker=dict(color='blue', size=6),
            showlegend=False
        ), row=2, col=1)
        
        # Panel 4
        x_mat = np.column_stack([np.ones(len(x)), x])
        hat_matrix = x_mat @ np.linalg.inv(x_mat.T @ x_mat) @ x_mat.T
        leverage = np.diag(hat_matrix)
        
        fig.add_trace(go.Scatter(
            x=leverage, y=residuals / np.std(residuals),
            mode='markers', marker=dict(color='blue', size=6),
            showlegend=False
        ), row=2, col=2)
        
        fig.update_layout(height=600, title_text="Diagnose-Plots: Gauss-Markov Annahmen")
        st.plotly_chart(fig, use_container_width=True, key="assumption_4panel")
    
    def _render_assumption_violation_demo(self) -> None:
        """Interactive demo of assumption violations."""
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
        else:
            y_demo = 2 + 3 * x_demo + np.random.normal(0, 2, n)
            y_demo[0] = 100
            y_demo[1] = -50
        
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
    
    def _render_t_test_plot(self) -> None:
        """Render t-test visualization."""
        t_slope = self.stats.get('t_slope', 0)
        p_slope = self.stats.get('p_slope', 0)
        df = self.stats.get('df', 10)
        
        x_t = np.linspace(-5, max(5, abs(t_slope) + 1), 200)
        y_t = stats.t.pdf(x_t, df=df)
        t_crit = stats.t.ppf(0.975, df=df)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_t, y=y_t, mode='lines', name='t-Verteilung',
                                 line=dict(color='black', width=2)))
        
        x_left = x_t[x_t < -t_crit]
        x_right = x_t[x_t > t_crit]
        
        fig.add_trace(go.Scatter(
            x=x_left, y=stats.t.pdf(x_left, df=df),
            fill='tozeroy', fillcolor='rgba(255,0,0,0.3)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Ablehnungsbereich'
        ))
        fig.add_trace(go.Scatter(
            x=x_right, y=stats.t.pdf(x_right, df=df),
            fill='tozeroy', fillcolor='rgba(255,0,0,0.3)',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False
        ))
        
        fig.add_vline(x=t_slope, line_color='blue', line_width=3,
                      annotation_text=f"t = {t_slope:.2f}")
        
        fig.update_layout(
            title=f"t-Test (df={df}): p = {p_slope:.4f}",
            xaxis_title="t-Wert",
            yaxis_title="Dichte",
            template="plotly_white",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True, key="t_test_plot")
    
    def _render_anova_interactive(self) -> None:
        """Render interactive ANOVA example."""
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
    
    def _render_anova_3d_landscape(self) -> None:
        """Render 3D ANOVA landscape."""
        np.random.seed(42)
        groups = ['A', 'B', 'C']
        means = [60, 70, 80]
        
        x_all, y_all, z_all = [], [], []
        
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
            scene=dict(xaxis_title='Wert', yaxis_title='Gruppe', zaxis_title='Dichte'),
            height=450
        )
        st.plotly_chart(fig, use_container_width=True, key="anova_3d")
    
    def _render_heteroskedasticity_demo(self) -> None:
        """Render heteroskedasticity demo."""
        np.random.seed(42)
        n = 100
        x_demo = np.random.uniform(1, 10, n)
        
        het_strength = st.slider("Heteroskedastizitäts-Stärke", 0.0, 2.0, 1.0, 0.1, key="het_str")
        
        y_demo = 10 + 2 * x_demo + np.random.normal(0, 1, n) * (1 + het_strength * x_demo)
        
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
    
    def _render_conditional_distribution_3d(self) -> None:
        """Render 3D conditional distribution."""
        x = self.data.get('x', np.array([]))
        y = self.data.get('y', np.array([]))
        
        if len(x) == 0:
            return
        
        slope = self.stats.get('slope', 0)
        intercept = self.stats.get('intercept', 0)
        se = np.sqrt(self.stats.get('mse', 1))
        
        x_vals = np.linspace(np.min(x), np.max(x), 5)
        
        fig = go.Figure()
        
        for x_val in x_vals:
            y_pred = intercept + slope * x_val
            y_range = np.linspace(y_pred - 3*se, y_pred + 3*se, 50)
            density = stats.norm.pdf(y_range, y_pred, se)
            density_scaled = density / np.max(density) * (np.max(x) - np.min(x)) * 0.3
            
            fig.add_trace(go.Scatter3d(
                x=[x_val] * len(y_range),
                y=y_range,
                z=density_scaled,
                mode='lines',
                line=dict(width=3),
                name=f'f(y|x={x_val:.1f})'
            ))
        
        x_line = np.linspace(np.min(x), np.max(x), 100)
        y_line = intercept + slope * x_line
        
        fig.add_trace(go.Scatter3d(
            x=x_line, y=y_line, z=np.zeros_like(x_line),
            mode='lines', line=dict(color='red', width=4),
            name='E(Y|X)'
        ))
        
        fig.update_layout(
            title="3D Bedingte Verteilung f(y|x)",
            scene=dict(
                xaxis_title=self.data.get('x_label', 'X'),
                yaxis_title=self.data.get('y_label', 'Y'),
                zaxis_title='Dichte'
            ),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True, key="cond_dist_3d")
    
    def _render_data_table(self) -> None:
        """Render data table."""
        import pandas as pd
        
        x = self.data.get('x', np.array([]))
        y = self.data.get('y', np.array([]))
        
        if len(x) == 0:
            return
        
        slope = self.stats.get('slope', 0)
        intercept = self.stats.get('intercept', 0)
        y_pred = intercept + slope * x
        residuals = y - y_pred
        
        df = pd.DataFrame({
            self.data.get('x_label', 'X'): x,
            self.data.get('y_label', 'Y'): y,
            'ŷ (Predicted)': y_pred,
            'Residuum': residuals,
        })
        st.dataframe(df.style.format("{:.4f}"), use_container_width=True)
