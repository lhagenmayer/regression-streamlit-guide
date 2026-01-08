"""
Streamlit Application - 100% Platform Agnostic.

Uses the same API layer as external frontends (Next.js, Vite, etc.)
This ensures consistency across all frontends.

Architecture:
    Streamlit App → API Layer → Core Pipeline
    
    Same as:
    Next.js App → HTTP → API Layer → Core Pipeline
"""

import streamlit as st
import numpy as np
from typing import Dict, Any, Optional

from ...config import get_logger
from ...api import RegressionAPI, ContentAPI, AIInterpretationAPI

logger = get_logger(__name__)


def run_streamlit_app():
    """Main Streamlit application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="Regression Analysis",
        page_icon="📈",
        layout="centered",
        initial_sidebar_state="auto"
    )
    
    # Custom CSS - Removed for native look
    # from .styles import inject_custom_css, render_hero
    # inject_custom_css()
    
    # Initialize APIs
    regression_api = RegressionAPI()
    content_api = ContentAPI()
    ai_api = AIInterpretationAPI()
    
    # Sidebar
    with st.sidebar:
        st.title("RegAnalysis")
        st.caption("Interactive Learning Platform")
        
        st.markdown("### 📊 Analyse")
        analysis_type = st.radio(
            "Analysis Type",
            ["Simple Regression", "Multiple Regression"],
            key="analysis_type",
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Dataset selection
        datasets_response = regression_api.get_datasets()
        st.subheader("Dataset")
        
        if analysis_type == "Simple Regression":
            dataset_options = {d["name"]: d["id"] for d in datasets_response["data"]["simple"]}
            dataset_name = st.selectbox(
                "Select Dataset:",
                list(dataset_options.keys()),
                key="dataset_simple"
            )
            dataset_id = dataset_options[dataset_name]
            n_points = st.slider("Samples", 20, 200, 50, key="n_simple")
        else:
            dataset_options = {d["name"]: d["id"] for d in datasets_response["data"]["multiple"]}
            dataset_name = st.selectbox(
                "Select Dataset:",
                list(dataset_options.keys()),
                key="dataset_multiple"
            )
            dataset_id = dataset_options[dataset_name]
            n_points = st.slider("Samples", 30, 200, 75, key="n_multiple")
        
        st.subheader("Parameters")
        noise = st.slider("Noise Level", 0.1, 2.0, 0.4, 0.1, key="noise")
        seed = st.number_input("Random Seed", 1, 9999, 42, key="seed")
        
        st.divider()
        
        # API Status in Sidebar
        status = ai_api.get_status()
        if status["status"]["configured"]:
            st.success("✅ AI Connected")
        else:
            st.warning("⚠️ AI Fallback")
            
    # Main Content
    
    # Hero / Title
    if "hero_shown" not in st.session_state:
        st.session_state.hero_shown = True
        
    title_suffix = "Simple" if analysis_type == "Simple Regression" else "Multiple"
    
    st.title(f"{title_suffix} Regression")
    st.caption("Explore relationships, analyze residuals, and master statistical modeling.")
    
    if analysis_type == "Simple Regression":
        render_simple_regression(content_api, ai_api, dataset_id, n_points, noise, seed)
    else:
        render_multiple_regression(content_api, ai_api, dataset_id, n_points, noise, seed)


def render_simple_regression(
    content_api: ContentAPI,
    ai_api: AIInterpretationAPI,
    dataset: str,
    n_points: int,
    noise: float,
    seed: int
):
    """Render simple regression analysis using API."""
    
    # Call API (same as external frontend would)
    with st.spinner("📊 Lade Daten via API..."):
        response = content_api.get_simple_content(
            dataset=dataset,
            n=n_points,
            noise=noise,
            seed=seed
        )
    
    if not response["success"]:
        st.error(f"API Fehler: {response.get('error', 'Unknown error')}")
        return
    
    # Extract data from API response
    content = response["content"]
    plots = response["plots"]
    stats = response["stats"]
    data = response["data"]
    
    # Build stats dict for renderer
    stats_dict = _flatten_stats(stats, data)
    
    # Render content using StreamlitContentRenderer
    from ..renderers import StreamlitContentRenderer
    
    renderer = StreamlitContentRenderer(
        plots={},  # Interactive plots generated on-demand
        data={
            "x": np.array(data["x"]),
            "y": np.array(data["y"]),
            "x_label": data["x_label"],
            "y_label": data["y_label"],
        },
        stats=stats_dict
    )
    
    # Build content structure from API response
    from ...content import SimpleRegressionContent
    content_builder = SimpleRegressionContent(stats_dict, {})
    content_obj = content_builder.build()
    
    # Render
    renderer.render(content_obj)
    
    # AI Interpretation
    _render_ai_interpretation(ai_api, stats_dict)


def render_multiple_regression(
    content_api: ContentAPI,
    ai_api: AIInterpretationAPI,
    dataset: str,
    n_points: int,
    noise: float,
    seed: int
):
    """Render multiple regression analysis using API."""
    
    # Call API
    with st.spinner("📊 Lade Daten via API..."):
        response = content_api.get_multiple_content(
            dataset=dataset,
            n=n_points,
            noise=noise,
            seed=seed
        )
    
    if not response["success"]:
        st.error(f"API Fehler: {response.get('error', 'Unknown error')}")
        return
    
    # Extract data
    content = response["content"]
    plots = response["plots"]
    stats = response["stats"]
    data = response["data"]
    
    # Build stats dict
    stats_dict = _flatten_multiple_stats(stats, data)
    
    # Render
    from ..renderers import StreamlitContentRenderer
    from ...content import MultipleRegressionContent
    
    renderer = StreamlitContentRenderer(
        plots={},
        data={
            "x1": np.array(data["x1"]),
            "x2": np.array(data["x2"]),
            "y": np.array(data["y"]),
            "x1_label": data["x1_label"],
            "x2_label": data["x2_label"],
            "y_label": data["y_label"],
        },
        stats=stats_dict
    )
    
    content_builder = MultipleRegressionContent(stats_dict, {})
    content_obj = content_builder.build()
    
    renderer.render(content_obj)
    
    # AI Interpretation
    _render_ai_interpretation(ai_api, stats_dict)


def _render_ai_interpretation(ai_api: AIInterpretationAPI, stats_dict: Dict[str, Any]):
    """Render AI interpretation section using API."""
    
    st.divider()
    
    # Use native container/info instead of HTML
    st.subheader("🤖 AI-Interpretation des R-Outputs")
    st.info("Lass dir alle statistischen Werte gesamtheitlich von Perplexity AI erklären.")
    
    # Status from API
    status = ai_api.get_status()
    if status["status"]["configured"]:
        st.caption("✅ Perplexity API verbunden")
    else:
        st.warning("⚠️ Kein API-Key - Fallback-Interpretation wird verwendet")
        st.caption("Setze `PERPLEXITY_API_KEY` als Umgebungsvariable")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        Klicke auf **"R-Output interpretieren"** für eine vollständige Erklärung:
        - Zusammenfassung des Modells
        - Interpretation der Koeffizienten
        - Bewertung der Modellgüte
        - Signifikanz-Erklärung
        - Praktische Bedeutung
        """)
    
    with col2:
        interpret_clicked = st.button(
            "🔍 Interpretieren",
            type="primary",
            use_container_width=True,
            key="ai_interpret_btn"
        )
    
    # Show R-Output
    with st.expander("📄 R-Output anzeigen"):
        r_output_response = ai_api.get_r_output(stats_dict)
        if r_output_response["success"]:
            st.code(r_output_response["r_output"], language="r")
    
    # Session state
    if "ai_interpretation_result" not in st.session_state:
        st.session_state.ai_interpretation_result = None
    
    if interpret_clicked:
        with st.spinner("🤖 AI analysiert via API..."):
            # Call API (same as HTTP request from external frontend)
            response = ai_api.interpret(stats=stats_dict, use_cache=True)
            st.session_state.ai_interpretation_result = response
    
    # Display interpretation
    if st.session_state.ai_interpretation_result:
        response = st.session_state.ai_interpretation_result
        
        st.markdown("### 📊 Interpretation")
        
        # Main content
        interpretation = response.get("interpretation", {})
        st.markdown(interpretation.get("content", "Keine Interpretation verfügbar."))
        
        # Metadata
        if response.get("success"):
            meta_cols = st.columns(4)
            meta_cols[0].caption(f"📡 {interpretation.get('model', 'N/A')}")
            meta_cols[1].caption(f"⏱️ {interpretation.get('latency_ms', 0):.0f}ms")
            
            usage = response.get("usage", {})
            if usage:
                meta_cols[2].caption(f"📝 {usage.get('total_tokens', 'N/A')} Tokens")
            
            meta_cols[3].caption(f"💾 {'Cached' if interpretation.get('cached') else 'Live'}")
        
        # Citations
        citations = response.get("citations", [])
        if citations:
            with st.expander("📚 Quellen"):
                for i, citation in enumerate(citations, 1):
                    st.markdown(f"{i}. [{citation}]({citation})")


def _flatten_stats(stats: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten API stats response for content builder."""
    coefficients = stats.get("coefficients", {})
    model_fit = stats.get("model_fit", {})
    t_tests = stats.get("t_tests", {})
    sum_of_squares = stats.get("sum_of_squares", {})
    sample = stats.get("sample", {})
    extra = stats.get("extra", {})
    
    x_arr = np.array(data.get("x", []))
    y_arr = np.array(data.get("y", []))
    
    return {
        # Context
        "context_title": data.get("context", {}).get("title", "Regressionsanalyse"),
        "context_description": data.get("context", {}).get("description", ""),
        "x_label": data.get("x_label", "X"),
        "y_label": data.get("y_label", "Y"),
        "y_unit": data.get("y_unit", ""),
        
        # Sample
        "n": sample.get("n", len(x_arr)),
        
        # Descriptive (computed from data)
        "x_mean": float(np.mean(x_arr)) if len(x_arr) > 0 else 0,
        "x_std": float(np.std(x_arr, ddof=1)) if len(x_arr) > 1 else 0,
        "x_min": float(np.min(x_arr)) if len(x_arr) > 0 else 0,
        "x_max": float(np.max(x_arr)) if len(x_arr) > 0 else 0,
        "y_mean": float(np.mean(y_arr)) if len(y_arr) > 0 else 0,
        "y_std": float(np.std(y_arr, ddof=1)) if len(y_arr) > 1 else 0,
        "y_min": float(np.min(y_arr)) if len(y_arr) > 0 else 0,
        "y_max": float(np.max(y_arr)) if len(y_arr) > 0 else 0,
        
        # Correlation
        "correlation": extra.get("correlation", 0),
        "covariance": float(np.cov(x_arr, y_arr, ddof=1)[0, 1]) if len(x_arr) > 1 else 0,
        
        # Coefficients
        "intercept": coefficients.get("intercept", 0),
        "slope": coefficients.get("slope", 0),
        "se_intercept": stats.get("standard_errors", {}).get("intercept", 0),
        "se_slope": stats.get("standard_errors", {}).get("slope", 0),
        
        # t-tests
        "t_intercept": t_tests.get("intercept", {}).get("t_value", 0),
        "t_slope": t_tests.get("slope", {}).get("t_value", 0),
        "p_intercept": t_tests.get("intercept", {}).get("p_value", 1),
        "p_slope": t_tests.get("slope", {}).get("p_value", 1),
        
        # Model fit
        "r_squared": model_fit.get("r_squared", 0),
        "r_squared_adj": model_fit.get("r_squared_adj", 0),
        
        # Sum of squares
        "sse": sum_of_squares.get("sse", 0),
        "sst": sum_of_squares.get("sst", 0),
        "ssr": sum_of_squares.get("ssr", 0),
        "mse": sum_of_squares.get("mse", 0),
        "df": sample.get("df", 0),
        
        # F-test (computed)
        "f_statistic": (sum_of_squares.get("ssr", 0) / 1) / sum_of_squares.get("mse", 1) if sum_of_squares.get("mse", 0) > 0 else 0,
        "p_f": t_tests.get("slope", {}).get("p_value", 1),  # Same as p_slope for simple regression
        
        # Residuals
        "residuals": stats.get("residuals", []),
        "y_pred": stats.get("predictions", []),
    }


def _flatten_multiple_stats(stats: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten API stats response for multiple regression content builder."""
    coefficients = stats.get("coefficients", {})
    model_fit = stats.get("model_fit", {})
    t_tests = stats.get("t_tests", {})
    sample = stats.get("sample", {})
    
    x1_arr = np.array(data.get("x1", []))
    x2_arr = np.array(data.get("x2", []))
    y_arr = np.array(data.get("y", []))
    
    # VIF calculation
    if len(x1_arr) > 1 and len(x2_arr) > 1:
        corr_x1_x2 = float(np.corrcoef(x1_arr, x2_arr)[0, 1])
        r2_x = corr_x1_x2 ** 2
        vif = 1 / (1 - r2_x) if r2_x < 1 else float('inf')
    else:
        corr_x1_x2 = 0
        vif = 1
    
    slopes = coefficients.get("slopes", [0, 0])
    se_coeffs = stats.get("standard_errors", [0, 0, 0])
    t_values = t_tests.get("t_values", [0, 0, 0])
    p_values = t_tests.get("p_values", [1, 1, 1])
    
    return {
        # Context
        "context_title": "Multiple Regression",
        "context_description": "Analyse mit mehreren Prädiktoren",
        "x1_label": data.get("x1_label", "X₁"),
        "x2_label": data.get("x2_label", "X₂"),
        "y_label": data.get("y_label", "Y"),
        
        # Sample
        "n": sample.get("n", len(y_arr)),
        "k": sample.get("k", 2),
        
        # Coefficients
        "intercept": coefficients.get("intercept", 0),
        "beta1": slopes[0] if len(slopes) > 0 else 0,
        "beta2": slopes[1] if len(slopes) > 1 else 0,
        "se_intercept": se_coeffs[0] if len(se_coeffs) > 0 else 0,
        "se_beta1": se_coeffs[1] if len(se_coeffs) > 1 else 0,
        "se_beta2": se_coeffs[2] if len(se_coeffs) > 2 else 0,
        "t_intercept": t_values[0] if len(t_values) > 0 else 0,
        "t_beta1": t_values[1] if len(t_values) > 1 else 0,
        "t_beta2": t_values[2] if len(t_values) > 2 else 0,
        "p_intercept": p_values[0] if len(p_values) > 0 else 1,
        "p_beta1": p_values[1] if len(p_values) > 1 else 1,
        "p_beta2": p_values[2] if len(p_values) > 2 else 1,
        
        # Model fit
        "r_squared": model_fit.get("r_squared", 0),
        "r_squared_adj": model_fit.get("r_squared_adj", 0),
        "f_statistic": model_fit.get("f_statistic", 0),
        "p_f": model_fit.get("f_p_value", 1),
        "df": sample.get("n", 0) - 3,
        
        # Multicollinearity
        "corr_x1_x2": corr_x1_x2,
        "vif_x1": vif,
        "vif_x2": vif,
        
        # Durbin-Watson
        "durbin_watson": 2.0,
        
        # Residuals
        "residuals": stats.get("residuals", []),
    }


if __name__ == "__main__":
    run_streamlit_app()
