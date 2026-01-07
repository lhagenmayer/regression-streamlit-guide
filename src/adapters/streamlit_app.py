"""
Streamlit Adapter - Renders regression analysis in Streamlit.
"""

import streamlit as st
from typing import Any, Dict

from .base import BaseRenderer, RenderContext
from ..pipeline import RegressionPipeline
from ..pipeline.plot import PlotCollection
from ..data import get_multiple_regression_formulas, get_multiple_regression_descriptions
from ..config import get_logger

logger = get_logger(__name__)


class StreamlitRenderer(BaseRenderer):
    """
    Streamlit implementation of the regression renderer.
    
    Uses Streamlit's native components for interactive rendering.
    """
    
    def __init__(self):
        self.pipeline = RegressionPipeline()
    
    def render(self, context: RenderContext) -> None:
        """Render based on analysis type."""
        if context.analysis_type == "simple":
            self.render_simple_regression(context)
        else:
            self.render_multiple_regression(context)
    
    def render_simple_regression(self, context: RenderContext) -> None:
        """Render simple regression with educational content."""
        from ..ui.tabs.simple_regression_educational import render_simple_regression_educational
        
        # Deserialize plots back to Plotly figures
        plots = self._deserialize_plots(context.plots_json)
        
        render_simple_regression_educational(
            data=context.data,
            stats_result=context.stats,
            plots=plots,
            show_formulas=context.show_formulas,
            show_true_line=context.show_true_line,
        )
    
    def render_multiple_regression(self, context: RenderContext) -> None:
        """Render multiple regression with educational content."""
        from ..ui.tabs.multiple_regression_educational import render_multiple_regression_educational
        
        plots = self._deserialize_plots(context.plots_json)
        
        render_multiple_regression_educational(
            data=context.data,
            stats_result=context.stats,
            plots=plots,
            content=context.content,
            formulas=context.formulas,
            show_formulas=context.show_formulas,
        )
    
    def _deserialize_plots(self, plots_json: Dict[str, str]) -> PlotCollection:
        """Deserialize JSON plots back to Plotly figures."""
        import plotly.io as pio
        
        scatter = None
        residuals = None
        diagnostics = None
        extra = {}
        
        for name, json_str in plots_json.items():
            fig = pio.from_json(json_str)
            if name == "scatter":
                scatter = fig
            elif name == "residuals":
                residuals = fig
            elif name == "diagnostics":
                diagnostics = fig
            else:
                extra[name] = fig
        
        return PlotCollection(
            scatter=scatter,
            residuals=residuals,
            diagnostics=diagnostics,
            extra=extra
        )
    
    def run(self, host: str = "0.0.0.0", port: int = 8501, debug: bool = False) -> None:
        """
        Run the Streamlit app.
        
        Note: Streamlit handles its own server, this method sets up the UI.
        """
        self._setup_page()
        self._render_sidebar()
        self._render_main_content()
    
    def _setup_page(self) -> None:
        """Configure Streamlit page."""
        st.set_page_config(
            page_title="📊 Linear Regression Guide",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        
        # Custom CSS
        st.markdown("""
        <style>
            .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; }
            .section-header { font-size: 1.6rem; font-weight: bold; color: #2c3e50; border-bottom: 2px solid #1f77b4; padding-bottom: 0.5rem; }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_sidebar(self) -> None:
        """Render sidebar controls."""
        st.sidebar.markdown("# ⚙️ Einstellungen")
        
        # Analysis type
        st.sidebar.markdown("### 📊 Analyse-Typ")
        analysis_type = st.sidebar.radio(
            "Wähle Analyse",
            ["Einfache Regression", "Multiple Regression"],
            key="analysis_type"
        )
        
        # Store in session state
        st.session_state["current_analysis"] = "simple" if "Einfach" in analysis_type else "multiple"
        
        # Dataset selection
        st.sidebar.markdown("### 📚 Datensatz")
        
        if st.session_state["current_analysis"] == "simple":
            datasets = {
                "electronics": "🏪 Elektronikmarkt",
                "advertising": "📢 Werbestudie",
            }
            dataset = st.sidebar.selectbox(
                "Datensatz",
                list(datasets.keys()),
                format_func=lambda x: datasets[x],
                key="simple_dataset"
            )
        else:
            datasets = {
                "cities": "🌆 Städte-Studie",
                "houses": "🏠 Immobilien",
            }
            dataset = st.sidebar.selectbox(
                "Datensatz",
                list(datasets.keys()),
                format_func=lambda x: datasets[x],
                key="multiple_dataset"
            )
        
        st.session_state["current_dataset"] = dataset
        
        # Parameters
        st.sidebar.markdown("### 🔧 Parameter")
        st.session_state["n_samples"] = st.sidebar.slider("Stichprobengrösse", 30, 200, 100)
        st.session_state["seed"] = st.sidebar.number_input("Seed", 1, 9999, 42)
        st.session_state["show_formulas"] = st.sidebar.checkbox("Formeln anzeigen", True)
    
    def _render_main_content(self) -> None:
        """Render main content area."""
        st.markdown('<p class="main-header">📊 Leitfaden zur Linearen Regression</p>', unsafe_allow_html=True)
        
        analysis_type = st.session_state.get("current_analysis", "simple")
        dataset = st.session_state.get("current_dataset", "electronics")
        n = st.session_state.get("n_samples", 100)
        seed = st.session_state.get("seed", 42)
        show_formulas = st.session_state.get("show_formulas", True)
        
        # Run pipeline
        if analysis_type == "simple":
            result = self.pipeline.run_simple(dataset=dataset, n=n, seed=seed)
            plots = self.pipeline.plotter.simple_regression_plots(result.data, result.stats)
            
            context = RenderContext(
                analysis_type="simple",
                data=result.data,
                stats=result.stats,
                plots_json=self.serialize_plots(plots),
                show_formulas=show_formulas,
                dataset_name=dataset,
            )
            self.render_simple_regression(context)
        else:
            result = self.pipeline.run_multiple(dataset=dataset, n=n, seed=seed)
            plots = self.pipeline.plotter.multiple_regression_plots(result.data, result.stats)
            
            content = get_multiple_regression_descriptions(dataset)
            formulas = get_multiple_regression_formulas(dataset)
            
            context = RenderContext(
                analysis_type="multiple",
                data=result.data,
                stats=result.stats,
                plots_json=self.serialize_plots(plots),
                show_formulas=show_formulas,
                content=content,
                formulas=formulas,
                dataset_name=dataset,
            )
            self.render_multiple_regression(context)


def create_streamlit_app() -> StreamlitRenderer:
    """Factory function to create Streamlit app."""
    return StreamlitRenderer()


# Direct execution entry point
if __name__ == "__main__":
    app = create_streamlit_app()
    app.run()
