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
from ...core.domain.value_objects import SplitConfig # Added import for SplitConfig

logger = get_logger(__name__)


def run_streamlit_app():
    """Main Streamlit application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="Regression & Classification",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    from .styles import inject_custom_css, render_hero
    inject_custom_css()
    
    # Initialize APIs
    regression_api = RegressionAPI()
    content_api = ContentAPI()
    ai_api = AIInterpretationAPI()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="margin-bottom: 2rem;">
            <div class="api-badge">API-POWERED</div>
            <h1 style="font-size: 1.5rem; margin: 0;">RegAnalysis</h1>
            <p style="color: #94a3b8; font-size: 0.9rem;">Interactive Learning Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        analysis_type = st.radio(
            "Analysis Type",
            ["Simple Regression", "Multiple Regression", "Binary Classification"],
            key="analysis_type",
            label_visibility="collapsed"
        )
        
        # Data Split Config moved below dataset selection

        st.markdown("---")
        
        # Dataset selection
        datasets_response = regression_api.get_datasets()
        st.markdown("### 📊 Dataset")
        
        # Default params
        method = "logistic"
        k_neighbors = 3
        
        if analysis_type == "Simple Regression":
            dataset_options = {d["name"]: d["id"] for d in datasets_response["data"]["simple"]}
            dataset_name = st.selectbox("Select Dataset:", list(dataset_options.keys()), key="dataset_simple")
            dataset_id = dataset_options[dataset_name]
            n_points = st.slider("Samples", 20, 200, 50, key="n_simple")
            
        elif analysis_type == "Multiple Regression":
            dataset_options = {d["name"]: d["id"] for d in datasets_response["data"]["multiple"]}
            dataset_name = st.selectbox("Select Dataset:", list(dataset_options.keys()), key="dataset_multiple")
            dataset_id = dataset_options[dataset_name]
            n_points = st.slider("Samples", 30, 200, 75, key="n_multiple")
            
        else: # Binary Classification
            # Hardcoded options for now or fetch via API if endpoint exists?
            # We reuse simple/multiple datasets + special ones
            # For "native" classification support we should probably have an endpoint get_classification_datasets
            # But generators.py handles conversion.
            
            # Mix of dedicated classification + convertibles
            cls_options = {
                "🍎 Fruits (2D)": "fruits",
                "🔢 Digits (64D)": "digits",
                "📱 Electronics (Simple->Binary)": "binary_electronics",
                "🏠 Housing (Simple->Binary)": "binary_housing",
                "🏥 WHO Health (External)": "who_health",
                "🏦 World Bank (External)": "world_bank",
            }
            dataset_name = st.selectbox("Select Dataset:", list(cls_options.keys()), key="dataset_cls")
            dataset_id = cls_options[dataset_name]
            n_points = st.slider("Samples", 50, 500, 100, step=10, key="n_cls")
            
            st.markdown("### 🧠 Model")
            method_display = st.selectbox("Method", ["Logistic Regression", "K-Nearest Neighbors"], key="method_select")
            method = "logistic" if "Logistic" in method_display else "knn"
            
            if method == "knn":
                k_neighbors = st.slider("Neighbors (k)", 1, 25, 3, key="k_knn")
                
            # Data Split Configuration (Only for Classification)
            with st.expander("Data Split & Stratification", expanded=True):
                st.markdown("Configure Training/Test split.")
                
                col1, col2 = st.columns(2)
                with col1:
                    train_size = st.slider(
                        "Training Size", 
                        min_value=0.1, 
                        max_value=0.9, 
                        value=0.8, 
                        step=0.05,
                        key="train_size_slider",
                        help="Proportion of data used for training."
                    )
                with col2:
                    stratify = st.checkbox(
                        "Stratify Split", 
                        value=False,
                        key="stratify_checkbox",
                        help="Maintain class proportions."
                    )
                    
                # Live Preview
                try:
                    # Look up dataset_id from current selection
                    # n_points is defined above
                    # noise/seed defined below, need defaults or move
                    # To avoid circular dep, we assume defaults for preview if vars not ready?
                    # But n_points is ready. dataset_id is ready.
                    # noise/seed are below. Let's assume defaults for PREVIEW or move them up.
                    # Moving noise/seed up is better UI practice anyway (Global params).
                    # But let's just use defaults for preview 
                    
                    preview_noise_val = 0.2
                    preview_seed_val = 42
                    
                    api = ContentAPI()
                    preview = api.get_split_preview(
                        dataset=dataset_id, 
                        train_size=train_size,
                        stratify=stratify, 
                        seed=preview_seed_val,
                        n=n_points,
                        noise=preview_noise_val
                    )
                    
                    if preview["success"]:
                        stats = preview["stats"]
                        # Compact display
                        # st.caption(f"Train: {stats['train_count']} | Test: {stats['test_count']}")
                        
                        import pandas as pd
                        import plotly.express as px
                        
                        dist_data = []
                        for k, v in stats["train_distribution"].items():
                            dist_data.append({"Class": str(k), "Count": v, "Set": "Train"})
                        for k, v in stats["test_distribution"].items():
                            dist_data.append({"Class": str(k), "Count": v, "Set": "Test"})
                            
                        df_dist = pd.DataFrame(dist_data)
                        fig_dist = px.bar(
                            df_dist, 
                            x="Set", 
                            y="Count", 
                            color="Class", 
                            barmode="group",
                            height=150,
                            title=None
                        )
                        fig_dist.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
                        st.plotly_chart(fig_dist, use_container_width=True)
                except Exception as e:
                    pass
        
        st.markdown("### ⚙️ Parameters")
        noise = st.slider("Noise Level", 0.0, 2.0, 0.2 if analysis_type == "Binary Classification" else 0.4, 0.1, key="noise")
        seed = st.number_input("Random Seed", 1, 9999, 42, key="seed")
        
        st.markdown("---")
        
        # API Status
        status = ai_api.get_status()
        if status["status"]["configured"]:
            st.success("✅ AI Connected")
        else:
            st.warning("⚠️ AI Fallback")
            
    # Main Content
    if "hero_shown" not in st.session_state:
        st.session_state.hero_shown = True
        
    # Tabs for Content vs Data
    tab_analysis, tab_data = st.tabs(["📊 Analysis", "🗃️ Data Explorer"])
    
    with tab_analysis:
        if analysis_type == "Simple Regression":
            render_hero("Simple Regression", "Explore relationships, analyze residuals, and master statistical modeling.")
            render_simple_regression(content_api, ai_api, dataset_id, n_points, noise, seed)
        elif analysis_type == "Multiple Regression":
            render_hero("Multiple Regression", "Multivariate analysis with 3D visualizations.")
            render_multiple_regression(content_api, ai_api, dataset_id, n_points, noise, seed)
        else:
            render_hero("Machine Learning", "From Logistic Regression to KNN classification.")
            render_classification(content_api, ai_api, dataset_id, n_points, noise, seed, method, k_neighbors, train_size, stratify)

    with tab_data:
        st.markdown(f"### 🗃️ Raw Data: {dataset_name}")
        st.markdown(f"**ID:** `{dataset_id}` | **Samples:** {n_points}")
        
        try:
             raw_resp = content_api.get_dataset_raw(dataset_id)
             if raw_resp.get("success"):
                 data = raw_resp["data"]["data"]
                 columns = raw_resp["data"]["columns"]
                 
                 import pandas as pd
                 df = pd.DataFrame(data)
                 # Reorder columns to put Target last if generic dict didn't preserve
                 if "Target" in columns and "Target" in df.columns:
                      cols = [c for c in df.columns if c != "Target"] + ["Target"]
                      df = df[cols]
                 
                 st.dataframe(df, use_container_width=True)
                 
                 csv = df.to_csv(index=False).encode('utf-8')
                 st.download_button(
                     "📥 Download CSV",
                     csv,
                     f"{dataset_id}.csv",
                     "text/csv",
                     key='download-csv'
                 )
             else:
                 st.error(f"Could not load data: {raw_resp.get('error')}")
        except Exception as e:
            st.error(f"Data Explorer Error: {e}")


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
    
    st.subheader("🤖 AI-Interpretation des R-Outputs")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); 
                padding: 1.5rem; border-radius: 1rem; color: white; margin: 1rem 0;">
        <p style="margin: 0; opacity: 0.9;">
            Lass dir alle statistischen Werte gesamtheitlich von Perplexity AI erklären.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status from API
    status = ai_api.get_status()
    if status["status"]["configured"]:
        st.success("✅ Perplexity API verbunden")
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


    renderer.render(content_obj)
    
    # AI Interpretation
    _render_ai_interpretation(ai_api, stats_dict)


def render_classification(
    content_api: ContentAPI,
    ai_api: AIInterpretationAPI,
    dataset: str,
    n_points: int,
    noise: float,
    seed: int,
    method: str,
    k_neighbors: int,
    train_size: float,
    stratify: bool
):
    """Render classification analysis (Machine Learning)."""
    
    with st.spinner(f"🧠 Trainiere {method.upper()} Modell..."):
        response = content_api.get_classification_content(
            dataset=dataset,
            n=n_points,
            noise=noise,
            seed=seed,
            method=method,
            k=k_neighbors,
            train_size=train_size,
            stratify=stratify
        )
        
    if not response["success"]:
        st.error(f"ML Fehler: {response.get('error')}")
        return
        
    # Extract
    content_dict = response["content"]
    plots_dict = response["plots"]
    stats_dict = response["stats"]
    data_dict = response["data"]
    results_dict = response.get("results", {})
    test_metrics = results_dict.get("test_metrics")
    
    # Display Metrics (Train vs Test)
    st.markdown("### 📉 Model Performance")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    metrics = stats_dict # Train metrics
    
    with m_col1:
        st.metric("Accuracy (Train)", f"{metrics.get('accuracy',0):.2%}")
        if test_metrics:
            st.metric("Accuracy (Test)", f"{test_metrics.get('accuracy',0):.2%}", 
                     delta=f"{test_metrics.get('accuracy',0) - metrics.get('accuracy',0):.2%}")
            
    with m_col2:
        st.metric("Precision (Train)", f"{metrics.get('precision',0):.2f}")
        if test_metrics:
            st.metric("Precision (Test)", f"{test_metrics.get('precision',0):.2f}")

    with m_col3:
        st.metric("Recall (Train)", f"{metrics.get('recall',0):.2f}")
        if test_metrics:
            st.metric("Recall (Test)", f"{test_metrics.get('recall',0):.2f}")
            
    with m_col4:
        st.metric("F1 Score (Train)", f"{metrics.get('f1',0):.2f}") # Note: 'f1' key from API check
        if test_metrics:
            st.metric("F1 Score (Test)", f"{test_metrics.get('f1',0):.2f}")
            
    st.markdown("---")
    
    # Reconstruct Content Object
    from ...infrastructure.content.structure import EducationalContent
    try:
        content_obj = EducationalContent.from_dict(content_dict)
    except Exception as e:
        st.error(f"Fehler beim Laden des Inhalts: {e}")
        return
    
    # Reconstruct Plots (Flatten for renderer: scatter, residuals, diagnostics, + extras)
    import plotly.graph_objects as go
    renderer_plots = {}
    
    if plots_dict:
        # Standard keys from PlotCollection
        for key in ["scatter", "residuals", "diagnostics"]:
           if plots_dict.get(key):
               renderer_plots[key] = go.Figure(plots_dict[key])
        
        # Extra keys
        if plots_dict.get("extra"):
            for k, v in plots_dict["extra"].items():
                if v:
                    renderer_plots[k] = go.Figure(v)
                
    # Initialize Renderer
    from ..renderers import StreamlitContentRenderer
    
    # Prepare data dict for interactive plots
    renderer_data = {
        "x": np.array(data_dict.get("X", [])),
        "y": np.array(data_dict.get("y", [])),
        "target_names": data_dict.get("target_names", []),
        "feature_names": stats_dict.get("feature_names", [])
    }
    
    renderer = StreamlitContentRenderer(
        plots=renderer_plots,
        data=renderer_data,
        stats=stats_dict
    )
    
    renderer.render(content_obj)
    
    renderer.render(content_obj)
    
    # AI Interpretation
    _render_ai_interpretation(ai_api, stats_dict)


if __name__ == "__main__":
    run_streamlit_app()
