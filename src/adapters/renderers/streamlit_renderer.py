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
        """Render 6-panel correlation examples (converted to 3D)."""
        fig = make_subplots(
            rows=2, cols=3,
            specs=[
                [{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}],
                [{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]
            ],
            subplot_titles=[
                'r = -0.95', 'r = -0.50', 'r = 0.00',
                'r = 0.50', 'r = 0.95', 'r = 0.00 (nonlinear)'
            ]
        )
        
        np.random.seed(42)
        n = 50
        correlations = [-0.95, -0.50, 0.0, 0.50, 0.95]
        
        for i, rho in enumerate(correlations):
            row = i // 3 + 1
            col = i % 3 + 1
            mean = [0, 0]
            cov = [[1, rho], [rho, 1]]
            data_gen = np.random.multivariate_normal(mean, cov, n)
            
            fig.add_trace(go.Scatter3d(
                x=data_gen[:, 0], y=data_gen[:, 1], z=np.zeros(n),
                mode='markers', marker=dict(size=3),
                showlegend=False
            ), row=row, col=col)
        
        # Nonlinear example
        x_nl = np.linspace(-2, 2, n)
        y_nl = x_nl**2 + np.random.normal(0, 0.3, n)
        fig.add_trace(go.Scatter3d(
            x=x_nl, y=y_nl, z=np.zeros(n), mode='markers', marker=dict(size=3),
            showlegend=False
        ), row=2, col=3)
        
        fig.update_layout(height=600, title_text="Korrelations-Beispiele (3D View)")
        st.plotly_chart(fig, use_container_width=True, key="corr_examples")
    
    def _render_raw_scatter(self) -> None:
        """Render raw data scatter plot in 3D."""
        x = self.data.get('x', np.array([]))
        y = self.data.get('y', np.array([]))
        
        if len(x) == 0:
            st.warning("Keine Daten für Scatter-Plot")
            return
        
        z_zeros = np.zeros(len(x))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z_zeros, mode='markers',
            marker=dict(size=5, color='#3498db', opacity=0.7),
            name='Datenpunkte'
        ))
        
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        # Mean lines in 3D
        fig.add_trace(go.Scatter3d(
            x=[min(x), max(x)], y=[y_mean, y_mean], z=[0, 0],
            mode='lines', line=dict(color='orange', dash='dash'),
            name=f"ȳ = {y_mean:.2f}"
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[x_mean, x_mean], y=[min(y), max(y)], z=[0, 0],
            mode='lines', line=dict(color='green', dash='dash'),
            name=f"x̄ = {x_mean:.2f}"
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[x_mean], y=[y_mean], z=[0], mode='markers',
            marker=dict(size=8, color='red', symbol='cross'),
            name=f'Schwerpunkt ({x_mean:.2f}, {y_mean:.2f})'
        ))
        
        fig.update_layout(
            title="Schritt 1: Visualisierung der Rohdaten (3D)",
            scene=dict(
                xaxis_title=self.data.get('x_label', 'X'),
                yaxis_title=self.data.get('y_label', 'Y'),
                zaxis_title="Z",
            ),
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True, key="raw_scatter")
    
    def _render_ols_regression(self) -> None:
        """Render OLS regression with residuals in 3D."""
        x = self.data.get('x', np.array([]))
        y = self.data.get('y', np.array([]))
        
        if len(x) == 0:
            st.warning("Keine Daten für Regression-Plot")
            return
        
        slope = self.stats.get('slope', 0)
        intercept = self.stats.get('intercept', 0)
        y_pred = intercept + slope * x
        z_zeros = np.zeros(len(x))
        
        fig = go.Figure()
        
        # Data points
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z_zeros, mode='markers',
            marker=dict(size=5, color='#3498db'),
            name='Datenpunkte'
        ))
        
        # Regression line
        x_line = np.linspace(min(x), max(x), 100)
        y_line = intercept + slope * x_line
        z_line = np.zeros(len(x_line))
        
        fig.add_trace(go.Scatter3d(
            x=x_line, y=y_line, z=z_line, mode='lines',
            line=dict(color='red', width=5),
            name='Regressionsgerade'
        ))
        
        # Residuals (vertical lines in 3D, though they lie on z=0 plane)
        for i in range(len(x)):
            fig.add_trace(go.Scatter3d(
                x=[x[i], x[i]], y=[y[i], y_pred[i]], z=[0, 0],
                mode='lines',
                line=dict(color='rgba(255,0,0,0.3)', width=2),
                showlegend=False
            ))
        
        fig.update_layout(
            title="OLS Regression (3D)",
            scene=dict(
                xaxis_title=self.data.get('x_label', 'X'),
                yaxis_title=self.data.get('y_label', 'Y'),
                zaxis_title="Z"
            ),
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True, key="ols_reg")
    
    def _render_decomposition(self) -> None:
        """Render decomposition of observation in 3D."""
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
        
        fig.add_trace(go.Scatter3d(
            x=[x[idx]], y=[y[idx]], z=[0], mode='markers',
            marker=dict(size=8, color='blue'), name=f'yᵢ = {y[idx]:.2f}'
        ))
        
        # Mean line
        fig.add_trace(go.Scatter3d(
            x=[min(x), max(x)], y=[y_mean, y_mean], z=[0, 0],
            mode='lines', line=dict(color='gray', dash='dash'),
            name=f"ȳ = {y_mean:.2f}"
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[x[idx]], y=[y_pred[idx]], z=[0], mode='markers',
            marker=dict(size=8, color='green', symbol='diamond'),
            name=f'ŷᵢ = {y_pred[idx]:.2f}'
        ))
        
        x_line = np.linspace(min(x), max(x), 100)
        y_line = intercept + slope * x_line
        fig.add_trace(go.Scatter3d(
            x=x_line, y=y_line, z=np.zeros(len(x_line)), mode='lines',
            line=dict(color='red', width=4), name='Regressionsgerade'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[x[idx], x[idx]], y=[y_mean, y_pred[idx]], z=[0, 0],
            mode='lines', line=dict(color='green', width=5),
            name=f'Erklärt: {y_pred[idx] - y_mean:.2f}'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[x[idx], x[idx]], y=[y_pred[idx], y[idx]], z=[0, 0],
            mode='lines', line=dict(color='red', width=5),
            name=f'Residuum: {y[idx] - y_pred[idx]:.2f}'
        ))
        
        fig.update_layout(
            title="Zerlegung einer Beobachtung (3D)",
            scene=dict(
                xaxis_title=self.data.get('x_label', 'X'),
                yaxis_title=self.data.get('y_label', 'Y'),
                zaxis_title='Z'
            ),
            template="plotly_white",
            height=500
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
        """Render SE confidence band visualization in 3D."""
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
        
        z_zeros = np.zeros(len(x))
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z_zeros, mode='markers',
            marker=dict(size=5, color='blue'), name='Daten'
        ))
        
        z_line = np.zeros(len(x_line))
        fig.add_trace(go.Scatter3d(
            x=x_line, y=y_pred, z=z_line, mode='lines',
            line=dict(color='red', width=4), name='Regression'
        ))
        
        # Upper and lower bounds as lines in 3D
        fig.add_trace(go.Scatter3d(
            x=x_line,
            y=y_pred + ci_mult * se_fit,
            z=z_line,
            mode='lines',
            line=dict(color='green', width=2),
            name=f'+{ci_mult:.2f} SE'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=x_line,
            y=y_pred - ci_mult * se_fit,
            z=z_line,
            mode='lines',
            line=dict(color='green', width=2),
            name=f'-{ci_mult:.2f} SE'
        ))
        
        fig.update_layout(
            title=f"Konfidenzband (±{ci_mult:.2f} × SE) - 3D View",
            scene=dict(
                xaxis_title=self.data.get('x_label', 'X'),
                yaxis_title=self.data.get('y_label', 'Y'),
                zaxis_title="Z",
                camera=dict(eye=dict(x=1.5, y=-1.5, z=0.5)),
            ),
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True, key="se_viz")
    
    def _render_variance_decomposition(self) -> None:
        """Render variance decomposition bar chart in 3D."""
        sst = self.stats.get('sst', 0)
        ssr = self.stats.get('ssr', 0)
        sse = self.stats.get('sse', 0)
        r2 = self.stats.get('r_squared', 0)
        
        fig = go.Figure()
        
        # Simulate 3D bars using lines
        names = ["SST (Total)", "SSR (Erklärt)", "SSE (Unerklärt)"]
        values = [sst, ssr, sse]
        colors = ["gray", "#2ecc71", "#e74c3c"]
        
        for i, (name, val, col) in enumerate(zip(names, values, colors)):
            fig.add_trace(go.Scatter3d(
                x=[i, i], y=[0, 0], z=[0, val],
                mode='lines',
                line=dict(color=col, width=15),
                name=name
            ))
            # Top marker
            fig.add_trace(go.Scatter3d(
                x=[i], y=[0], z=[val],
                mode='markers+text',
                marker=dict(size=5, color=col),
                text=[f"{val:.1f}"],
                textposition="top center",
                showlegend=False
            ))
            
        fig.update_layout(
            title=f"Varianzzerlegung: R² = {r2:.4f} (3D Bars)",
            scene=dict(
                xaxis=dict(title="", ticktext=names, tickvals=[0, 1, 2]),
                yaxis_title="",
                zaxis_title="Quadratsumme",
                camera=dict(eye=dict(x=1.5, y=-1.5, z=0.5)),
            ),
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True, key="variance_decomp")
    
    def _render_assumptions_4panel(self) -> None:
        """Render 4-panel assumption diagnostics in 3D."""
        x = self.data.get('x', np.array([]))
        y = self.data.get('y', np.array([]))
        
        if len(x) == 0:
            return
        
        slope = self.stats.get('slope', 0)
        intercept = self.stats.get('intercept', 0)
        y_pred = intercept + slope * x
        residuals = y - y_pred
        z_zeros = np.zeros(len(residuals))
        
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{'type': 'scene'}, {'type': 'scene'}],
                [{'type': 'scene'}, {'type': 'scene'}]
            ],
            subplot_titles=[
                '1. Linearität (3D)',
                '2. Normalität (3D)',
                '3. Homoskedastizität (3D)',
                '4. Einfluss (3D)'
            ]
        )
        
        # Panel 1: Resid vs Fitted
        fig.add_trace(go.Scatter3d(
            x=y_pred, y=residuals, z=z_zeros,
            mode='markers', marker=dict(color='blue', size=4),
            showlegend=False
        ), row=1, col=1)
        
        # Panel 2: Q-Q
        sorted_res = np.sort(residuals)
        n = len(sorted_res)
        theoretical_q = stats.norm.ppf(np.arange(1, n+1) / (n+1))
        
        fig.add_trace(go.Scatter3d(
            x=theoretical_q, y=sorted_res, z=z_zeros,
            mode='markers', marker=dict(color='blue', size=4),
            showlegend=False
        ), row=1, col=2)
        
        # Panel 3: Scale-Location
        sqrt_std_res = np.sqrt(np.abs(residuals / np.std(residuals)))
        fig.add_trace(go.Scatter3d(
            x=y_pred, y=sqrt_std_res, z=z_zeros,
            mode='markers', marker=dict(color='blue', size=4),
            showlegend=False
        ), row=2, col=1)
        
        # Panel 4: Leverage
        x_mat = np.column_stack([np.ones(len(x)), x])
        hat_matrix = x_mat @ np.linalg.inv(x_mat.T @ x_mat) @ x_mat.T
        leverage = np.diag(hat_matrix)
        
        fig.add_trace(go.Scatter3d(
            x=leverage, y=residuals / np.std(residuals), z=z_zeros,
            mode='markers', marker=dict(color='blue', size=4),
            showlegend=False
        ), row=2, col=2)
        
        fig.update_layout(height=800, title_text="Diagnose-Plots (3D View)")
        st.plotly_chart(fig, use_container_width=True, key="assumption_4panel")
    
    def _render_assumption_violation_demo(self) -> None:
        """Interactive demo of assumption violations in 3D."""
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
        z_zeros = np.zeros(n)
        
        with col1:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter3d(x=x_demo, y=y_demo, z=z_zeros, mode='markers', marker=dict(size=4)))
            x_line = np.linspace(0, 10, 100)
            z_line = np.zeros(100)
            fig1.add_trace(go.Scatter3d(x=x_line, y=b0_demo + b1_demo * x_line, z=z_line,
                                       mode='lines', line=dict(color='red', width=4)))
            fig1.update_layout(title="Daten + Regression (3D)", scene=dict(zaxis_title='Z'), height=400)
            st.plotly_chart(fig1, use_container_width=True, key=f"viol_scatter_{violation_type}")
        
        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter3d(x=y_pred_demo, y=residuals_demo, z=z_zeros, mode='markers', marker=dict(size=4)))
            
            # Zero line
            x_range = np.linspace(min(y_pred_demo), max(y_pred_demo), 100)
            fig2.add_trace(go.Scatter3d(x=x_range, y=np.zeros(100), z=np.zeros(100), mode='lines', line=dict(color='red', dash='dash')))
            
            fig2.update_layout(title="Residuen vs. Fitted (3D)", scene=dict(zaxis_title='Z'), height=400)
            st.plotly_chart(fig2, use_container_width=True, key=f"viol_resid_{violation_type}")
    
    def _render_t_test_plot(self) -> None:
        """Render t-test visualization in 3D."""
        t_slope = self.stats.get('t_slope', 0)
        p_slope = self.stats.get('p_slope', 0)
        df = self.stats.get('df', 10)
        
        x_t = np.linspace(-5, max(5, abs(t_slope) + 1), 200)
        y_t = stats.t.pdf(x_t, df=df)
        
        fig = go.Figure()
        
        # t-distribution curve in 3D (Z=0)
        fig.add_trace(go.Scatter3d(
            x=x_t, y=y_t, z=np.zeros(len(x_t)),
            mode='lines', name='t-Verteilung',
            line=dict(color='black', width=4)
        ))
        
        # Vertical line for t-statistic
        fig.add_trace(go.Scatter3d(
            x=[t_slope, t_slope],
            y=[0, stats.t.pdf(t_slope, df=df)],
            z=[0, 0],
            mode='lines',
            line=dict(color='blue', width=6),
            name=f"t = {t_slope:.2f}"
        ))
        
        fig.update_layout(
            title=f"t-Test (df={df}): p = {p_slope:.4f} (3D View)",
            scene=dict(
                xaxis_title="t-Wert",
                yaxis_title="Dichte",
                zaxis_title="Z",
                camera=dict(eye=dict(x=1.5, y=-1.5, z=0.5)),
            ),
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True, key="t_test_plot")
    
    def _render_anova_interactive(self) -> None:
        """Render interactive ANOVA example in 3D (Scatter instead of Boxplot)."""
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
        
        # Plot points in 3D
        # X = Group (0, 1, 2), Y = Value, Z = Jitter
        jitter_a = np.random.uniform(-0.2, 0.2, n_per_group)
        jitter_b = np.random.uniform(-0.2, 0.2, n_per_group)
        jitter_c = np.random.uniform(-0.2, 0.2, n_per_group)
        
        fig.add_trace(go.Scatter3d(
            x=[0] * n_per_group, y=group_a, z=jitter_a,
            mode='markers', name='Gruppe A', marker=dict(color='blue', size=4)
        ))
        fig.add_trace(go.Scatter3d(
            x=[1] * n_per_group, y=group_b, z=jitter_b,
            mode='markers', name='Gruppe B', marker=dict(color='green', size=4)
        ))
        fig.add_trace(go.Scatter3d(
            x=[2] * n_per_group, y=group_c, z=jitter_c,
            mode='markers', name='Gruppe C', marker=dict(color='red', size=4)
        ))
        
        fig.update_layout(
            title="Gruppen-Vergleich (3D Scatter)",
            scene=dict(
                xaxis=dict(title="Gruppe", ticktext=["A", "B", "C"], tickvals=[0, 1, 2]),
                yaxis_title="Wert",
                zaxis_title="Jitter (zur besseren Sicht)",
            ),
            template="plotly_white",
            height=500
        )
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
        """Render heteroskedasticity demo in 3D."""
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
        z_zeros = np.zeros(n)
        
        with col1:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter3d(x=x_demo, y=y_demo, z=z_zeros, mode='markers', name='Daten', marker=dict(size=4)))
            x_line = np.linspace(1, 10, 100)
            z_line = np.zeros(100)
            fig1.add_trace(go.Scatter3d(x=x_line, y=b0 + b1 * x_line, z=z_line,
                                       mode='lines', line=dict(color='red', width=4), name='Regression'))
            fig1.update_layout(title="Daten mit Heteroskedastizität (3D)", scene=dict(zaxis_title='Z'), height=400)
            st.plotly_chart(fig1, use_container_width=True, key="het_scatter")
        
        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter3d(x=y_pred, y=residuals, z=z_zeros, mode='markers', marker=dict(size=4)))
            
            x_range = np.linspace(min(y_pred), max(y_pred), 100)
            fig2.add_trace(go.Scatter3d(x=x_range, y=np.zeros(100), z=np.zeros(100), mode='lines', line=dict(color='red', dash='dash')))
            
            fig2.update_layout(title="Residuen: Trichter-Effekt (3D)", scene=dict(zaxis_title='Z'), height=400)
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
