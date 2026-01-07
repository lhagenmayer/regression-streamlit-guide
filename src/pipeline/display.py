"""
Step 4: DISPLAY

This module connects the Pipeline to the UI tabs.
It delegates rendering to the specialized tab modules in /ui/tabs/
which contain the full educational content.

The display step is responsible for:
- Passing pipeline results to the correct tab renderer
- Ensuring all plots have educational context
- Dynamic content adaptation based on dataset
"""

from typing import Dict, Any, Union
import streamlit as st

from ..config import get_logger
from .get_data import DataResult, MultipleRegressionDataResult
from .calculate import RegressionResult, MultipleRegressionResult
from .plot import PlotCollection

logger = get_logger(__name__)


class UIRenderer:
    """
    Step 4: DISPLAY
    
    Renders pipeline results using the educational tab modules.
    Ensures all plots are embedded with meaningful educational content.
    
    The actual rendering is delegated to:
    - src/ui/tabs/simple_regression.py
    - src/ui/tabs/multiple_regression.py
    
    These modules contain the full educational content (chapters, formulas, etc.)
    """
    
    def __init__(self):
        logger.info("UIRenderer initialized")
    
    def simple_regression(
        self,
        data: DataResult,
        result: RegressionResult,
        plots: PlotCollection,
        show_formulas: bool = True,
        show_true_line: bool = False,
    ) -> None:
        """
        Display simple regression with full educational content.
        
        Delegates to the specialized tab renderer which contains
        all chapters and educational material.
        """
        # Build model_data dict expected by the tab renderer
        model_data = self._build_simple_model_data(data, result)
        
        # Use the specialized educational tab renderer
        from ..ui.tabs.simple_regression import render_simple_regression_tab
        
        render_simple_regression_tab(
            model_data=model_data,
            x_label=data.x_label,
            y_label=data.y_label,
            x_unit=data.x_unit,
            y_unit=data.y_unit,
            context_title=data.context_title,
            context_description=data.context_description,
            show_formulas=show_formulas,
            show_true_line=show_true_line,
            has_true_line=data.extra.get("true_slope", 0) != 0,
            true_intercept=data.extra.get("true_intercept", 0),
            true_beta=data.extra.get("true_slope", 0),
        )
    
    def multiple_regression(
        self,
        data: MultipleRegressionDataResult,
        result: MultipleRegressionResult,
        plots: PlotCollection,
        dataset_choice: str = "cities",
        show_formulas: bool = True,
    ) -> None:
        """
        Display multiple regression with full educational content.
        
        Delegates to the specialized tab renderer which contains
        all chapters and educational material.
        """
        # Build model_data dict expected by the tab renderer
        model_data = self._build_multiple_model_data(data, result)
        
        # Map internal dataset name to display name for content.py
        dataset_display_name = self._map_dataset_name(dataset_choice)
        
        # Use the specialized educational tab renderer
        from ..ui.tabs.multiple_regression import render_multiple_regression_tab
        
        render_multiple_regression_tab(
            model_data=model_data,
            dataset_choice=dataset_display_name,
            show_formulas=show_formulas,
        )
    
    def simple_regression_compact(
        self,
        data: DataResult,
        result: RegressionResult,
        plots: PlotCollection,
        show_formulas: bool = True,
    ) -> None:
        """
        Compact display for simple regression (without full chapters).
        
        Use this when you need a quick display without the full
        educational content. All plots still have context.
        """
        import pandas as pd
        
        # Header with context
        st.markdown(f"# 📈 {data.context_title}")
        st.info(data.context_description)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R²", f"{result.r_squared:.4f}")
        col2.metric("β₀", f"{result.intercept:.4f}")
        col3.metric("β₁", f"{result.slope:.4f}")
        col4.metric("n", f"{result.n}")
        
        # Main plot with educational context
        st.markdown("---")
        st.markdown("## 📊 Regressionsanalyse")
        st.markdown(f"""
        **Interpretation:** Die Regressionsgerade zeigt den geschätzten Zusammenhang
        zwischen **{data.x_label}** und **{data.y_label}**.
        
        - Pro Einheit {data.x_label} ändert sich {data.y_label} um **{result.slope:.4f}** {data.y_unit}
        - Bei {data.x_label} = 0 wäre der erwartete Wert **{result.intercept:.4f}** {data.y_unit}
        """)
        st.plotly_chart(plots.scatter, use_container_width=True, key="scatter_compact")
        
        # Equation
        if show_formulas:
            sign = "+" if result.slope >= 0 else ""
            st.latex(rf"\hat{{y}} = {result.intercept:.4f} {sign} {result.slope:.4f} \cdot x")
        
        # Residual analysis with context
        st.markdown("---")
        st.markdown("## 🔍 Residuenanalyse")
        st.markdown("""
        **Warum wichtig?** Die Residuen zeigen, wie gut unser Modell die Daten beschreibt.
        Zufällige Streuung um 0 = gutes Modell. Muster = Modellverletzung!
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plots.residuals, use_container_width=True, key="resid_compact")
        with col2:
            if plots.diagnostics:
                st.plotly_chart(plots.diagnostics, use_container_width=True, key="diag_compact")
        
        # Coefficient table
        self._display_coefficient_table_simple(result, data)
    
    def multiple_regression_compact(
        self,
        data: MultipleRegressionDataResult,
        result: MultipleRegressionResult,
        plots: PlotCollection,
        show_formulas: bool = True,
    ) -> None:
        """
        Compact display for multiple regression.
        """
        import pandas as pd
        
        # Header
        st.markdown("# 📊 Multiple Regression")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R²", f"{result.r_squared:.4f}")
        col2.metric("R² adj", f"{result.r_squared_adj:.4f}")
        col3.metric("F", f"{result.f_statistic:.2f}")
        col4.metric("n", f"{result.n}")
        
        # 3D Plot with context
        st.markdown("---")
        st.markdown("## 📊 Regressionsebene")
        st.markdown(f"""
        **Interpretation:** Die Ebene zeigt den geschätzten Zusammenhang zwischen
        **{data.x1_label}**, **{data.x2_label}** und **{data.y_label}**.
        
        - Pro Einheit {data.x1_label}: {data.y_label} ändert sich um **{result.coefficients[0]:.3f}** (ceteris paribus)
        - Pro Einheit {data.x2_label}: {data.y_label} ändert sich um **{result.coefficients[1]:.3f}** (ceteris paribus)
        """)
        st.plotly_chart(plots.scatter, use_container_width=True, key="scatter3d_compact")
        
        # Equation
        if show_formulas:
            b0 = result.intercept
            b1, b2 = result.coefficients
            sign1 = "+" if b1 >= 0 else ""
            sign2 = "+" if b2 >= 0 else ""
            st.latex(rf"\hat{{y}} = {b0:.3f} {sign1} {b1:.3f} \cdot x_1 {sign2} {b2:.3f} \cdot x_2")
        
        # Residual analysis
        st.markdown("---")
        st.markdown("## 🔍 Residuenanalyse")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plots.residuals, use_container_width=True, key="resid_mult_compact")
        with col2:
            if plots.diagnostics:
                st.plotly_chart(plots.diagnostics, use_container_width=True, key="diag_mult_compact")
        
        # Coefficient table
        self._display_coefficient_table_multiple(result, data)
    
    # =========================================================
    # PRIVATE HELPERS
    # =========================================================
    
    def _build_simple_model_data(
        self, data: DataResult, result: RegressionResult
    ) -> Dict[str, Any]:
        """Build model_data dict for simple regression tab renderer."""
        import numpy as np
        
        # Create a mock model object with required attributes
        class MockModel:
            def __init__(self, result):
                self.rsquared = result.r_squared
                self.rsquared_adj = result.r_squared_adj
                self.params = [result.intercept, result.slope]
                self.bse = [result.se_intercept, result.se_slope]
                self.tvalues = [result.t_intercept, result.t_slope]
                self.pvalues = [result.p_intercept, result.p_slope]
                self.resid = result.residuals
                self.fittedvalues = result.y_pred
                self.ssr = result.sse  # Note: naming convention differs
                self.centered_tss = result.sst
                self.ess = result.ssr
                self.mse_resid = result.mse
                self.df_resid = result.df
        
        return {
            "model": MockModel(result),
            "x": data.x,
            "y": data.y,
            "y_pred": result.y_pred,
            "b0": result.intercept,
            "b1": result.slope,
            "residuals": result.residuals,
            "x_mean": result.extra.get("x_mean", np.mean(data.x)),
            "y_mean_val": result.extra.get("y_mean", np.mean(data.y)),
            "cov_xy": result.extra.get("cov_xy", np.cov(data.x, data.y)[0, 1]),
            "var_x": np.var(data.x, ddof=1),
            "var_y": np.var(data.y, ddof=1),
            "corr_xy": result.extra.get("correlation", np.corrcoef(data.x, data.y)[0, 1]),
            "sse": result.sse,
            "sst": result.sst,
            "ssr": result.ssr,
            "mse": result.mse,
            "se_regression": result.extra.get("se_regression", np.sqrt(result.mse)),
        }
    
    def _build_multiple_model_data(
        self, data: MultipleRegressionDataResult, result: MultipleRegressionResult
    ) -> Dict[str, Any]:
        """Build model_data dict for multiple regression tab renderer."""
        return {
            "x2_preis": data.x1,
            "x3_werbung": data.x2,
            "y_mult": data.y,
            "x1_name": data.x1_label,
            "x2_name": data.x2_label,
            "y_name": data.y_label,
            "model_mult": None,  # Not needed for display
            "y_pred_mult": result.y_pred,
            "mult_coeffs": {
                "params": [result.intercept] + result.coefficients,
                "bse": result.se_coefficients,
                "tvalues": result.t_values,
                "pvalues": result.p_values,
            },
            "mult_summary": {
                "rsquared": result.r_squared,
                "rsquared_adj": result.r_squared_adj,
                "fvalue": result.f_statistic,
                "f_pvalue": result.f_pvalue,
            },
            "mult_diagnostics": {
                "resid": result.residuals,
                "sse": result.sse,
            },
        }
    
    def _map_dataset_name(self, dataset: str) -> str:
        """Map internal dataset name to display name for content.py."""
        mapping = {
            "cities": "🏙️ Städte-Umsatzstudie (75 Städte)",
            "houses": "🏠 Häuserpreise mit Pool (1000 Häuser)",
            "electronics": "🏪 Elektronikmarkt (simuliert)",
        }
        return mapping.get(dataset, dataset)
    
    def _display_coefficient_table_simple(
        self, result: RegressionResult, data: DataResult
    ) -> None:
        """Display coefficient table with significance."""
        import pandas as pd
        
        st.markdown("### 📋 Koeffizienten")
        
        df = pd.DataFrame({
            "Parameter": ["β₀ (Intercept)", f"β₁ ({data.x_label})"],
            "Schätzwert": [result.intercept, result.slope],
            "Std. Error": [result.se_intercept, result.se_slope],
            "t-Wert": [result.t_intercept, result.t_slope],
            "p-Wert": [result.p_intercept, result.p_slope],
            "Signif.": [_get_signif_stars(result.p_intercept), _get_signif_stars(result.p_slope)],
        })
        
        st.dataframe(
            df.style.format({
                "Schätzwert": "{:.4f}",
                "Std. Error": "{:.4f}",
                "t-Wert": "{:.3f}",
                "p-Wert": "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Signif.: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1")
    
    def _display_coefficient_table_multiple(
        self, result: MultipleRegressionResult, data: MultipleRegressionDataResult
    ) -> None:
        """Display coefficient table for multiple regression."""
        import pandas as pd
        
        st.markdown("### 📋 Koeffizienten")
        
        labels = ["β₀ (Intercept)", f"β₁ ({data.x1_label})", f"β₂ ({data.x2_label})"]
        coefs = [result.intercept] + result.coefficients
        
        df = pd.DataFrame({
            "Parameter": labels,
            "Schätzwert": coefs,
            "Std. Error": result.se_coefficients,
            "t-Wert": result.t_values,
            "p-Wert": result.p_values,
            "Signif.": [_get_signif_stars(p) for p in result.p_values],
        })
        
        st.dataframe(
            df.style.format({
                "Schätzwert": "{:.4f}",
                "Std. Error": "{:.4f}",
                "t-Wert": "{:.3f}",
                "p-Wert": "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Signif.: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1")


def _get_signif_stars(p: float) -> str:
    """Get significance stars like R."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return ""
