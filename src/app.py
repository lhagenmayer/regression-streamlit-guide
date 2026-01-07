"""
🎓 Leitfaden zur Linearen Regression
====================================

Ein didaktisches Tool mit klarer 4-Stufen-Pipeline:
    1. GET      → Daten holen
    2. CALCULATE → Statistiken berechnen
    3. PLOT     → Visualisierungen erstellen
    4. DISPLAY  → Im UI anzeigen (via /tabs)

Die DISPLAY-Logik ist in den spezialisierten Tab-Modulen implementiert,
die den vollständigen edukativen Content enthalten.

Start: streamlit run src/app.py
"""

import warnings
import streamlit as st

warnings.filterwarnings('ignore')

# Path setup
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for path_dir in [current_dir, parent_dir]:
    if path_dir not in sys.path:
        sys.path.insert(0, path_dir)

from .pipeline import RegressionPipeline
from .config import get_logger, UI_DEFAULTS
from .data import get_multiple_regression_formulas, get_multiple_regression_descriptions

logger = get_logger(__name__)


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="📖 Leitfaden Lineare Regression",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for educational styling
st.markdown("""
<style>
    .main-header { 
        font-size: 2.5rem; 
        font-weight: bold; 
        color: #1f77b4; 
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header { 
        font-size: 1.6rem; 
        font-weight: bold; 
        color: #2c3e50; 
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    .subsection-header {
        font-size: 1.3rem;
        font-weight: bold;
        color: #34495e;
        margin-top: 1rem;
    }
    .concept-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
    .formula-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        padding: 15px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR - Parameter Configuration
# =============================================================================
st.sidebar.markdown("# ⚙️ Einstellungen")

# Dataset type selection
st.sidebar.markdown("### 📚 Datensatz")

# Simple regression datasets
simple_datasets = {
    "electronics": "🏪 Elektronikmarkt (simuliert)",
    "advertising": "📢 Werbestudie (75 Städte)",
    "temperature": "🍦 Eisverkauf & Temperatur",
}

# Multiple regression datasets
multiple_datasets = {
    "cities": "🏙️ Städte-Umsatzstudie",
    "houses": "🏠 Häuserpreise mit Pool",
}

dataset_simple = st.sidebar.selectbox(
    "Einfache Regression",
    list(simple_datasets.keys()),
    format_func=lambda x: simple_datasets[x],
)

dataset_multiple = st.sidebar.selectbox(
    "Multiple Regression",
    list(multiple_datasets.keys()),
    format_func=lambda x: multiple_datasets[x],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Parameter")

# Sample size
n_simple = st.sidebar.slider("n (Einfache Reg.)", 10, 100, 50)
n_multiple = st.sidebar.slider("n (Multiple Reg.)", 20, 200, 75)

# Noise
noise_simple = st.sidebar.slider("Rauschen (σ) - Einfach", 0.1, 2.0, 0.4, 0.1)
noise_multiple = st.sidebar.slider("Rauschen (σ) - Multipel", 1.0, 10.0, 3.5, 0.5)

# Seed
seed = st.sidebar.number_input("Random Seed", 1, 9999, 42)

st.sidebar.markdown("---")

# True parameters (for simulated data)
with st.sidebar.expander("🎯 Wahre Parameter (Simulation)"):
    true_intercept = st.slider("β₀ (wahrer Intercept)", -2.0, 3.0, 0.6, 0.1)
    true_slope = st.slider("β₁ (wahre Steigung)", 0.1, 2.0, 0.52, 0.01)
    show_true_line = st.checkbox("Wahre Linie anzeigen", True)

# Display options
with st.sidebar.expander("🔧 Anzeigeoptionen"):
    show_formulas = st.checkbox("Formeln anzeigen", True)
    compact_mode = st.checkbox("Kompaktmodus", False)

st.sidebar.markdown("---")
st.sidebar.success("✅ Pipeline bereit")


# =============================================================================
# INITIALIZE PIPELINE
# =============================================================================
pipeline = RegressionPipeline()


# =============================================================================
# MAIN CONTENT - Tabs
# =============================================================================

# Header
st.markdown('<p class="main-header">📖 Leitfaden zur Linearen Regression</p>', unsafe_allow_html=True)
st.markdown("### Von der Frage zur validierten Erkenntnis")

# Pipeline info
with st.expander("ℹ️ Über die Pipeline-Architektur", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    col1.info("**1. GET**\n\nDaten generieren/laden")
    col2.info("**2. CALCULATE**\n\nOLS, R², t-Tests")
    col3.info("**3. PLOT**\n\nVisualisierungen")
    col4.info("**4. DISPLAY**\n\nEdukative Einbettung")
    
    st.markdown("""
    Diese Anwendung folgt einer klaren 4-Stufen-Pipeline. 
    Jeder Plot ist in edukativen Content eingebettet und hat eine klare Bedeutung.
    Der Content passt sich dynamisch an den gewählten Datensatz an.
    """)

st.markdown("---")

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "📈 Einfache Regression", 
    "📊 Multiple Regression", 
    "📚 Datensatz-Info"
])


# =============================================================================
# TAB 1: EINFACHE REGRESSION
# =============================================================================
with tab1:
    try:
        # STEP 1 & 2 & 3: GET → CALCULATE → PLOT via Pipeline
        result_simple = pipeline.run_simple(
            dataset=dataset_simple,
            n=n_simple,
            noise=noise_simple,
            seed=seed,
            true_intercept=true_intercept,
            true_slope=true_slope,
            show_true_line=show_true_line,
        )
        
        # STEP 4: DISPLAY with educational content
        if compact_mode:
            # Compact display (still with context)
            pipeline.renderer.simple_regression_compact(
                data=result_simple.data,
                result=result_simple.stats,
                plots=result_simple.plots,
                show_formulas=show_formulas,
            )
        else:
            # Full educational display via tab renderer
            # Build the model_data structure expected by the tab
            from .ui.tabs.simple_regression_educational import render_simple_regression_educational
            
            render_simple_regression_educational(
                data=result_simple.data,
                stats=result_simple.stats,
                plots=result_simple.plots,
                show_formulas=show_formulas,
                show_true_line=show_true_line,
            )
            
    except Exception as e:
        logger.error(f"Simple regression error: {e}")
        st.error(f"❌ Fehler: {str(e)}")


# =============================================================================
# TAB 2: MULTIPLE REGRESSION
# =============================================================================
with tab2:
    try:
        # STEP 1 & 2 & 3: GET → CALCULATE → PLOT via Pipeline
        result_multiple = pipeline.run_multiple(
            dataset=dataset_multiple,
            n=n_multiple,
            noise=noise_multiple,
            seed=seed,
        )
        
        # Get dynamic content based on dataset
        dataset_display = {
            "cities": "🏙️ Städte-Umsatzstudie (75 Städte)",
            "houses": "🏠 Häuserpreise mit Pool (1000 Häuser)",
        }.get(dataset_multiple, dataset_multiple)
        
        descriptions = get_multiple_regression_descriptions(dataset_display)
        formulas = get_multiple_regression_formulas(dataset_display)
        
        # STEP 4: DISPLAY with educational content
        if compact_mode:
            pipeline.renderer.multiple_regression_compact(
                data=result_multiple.data,
                result=result_multiple.stats,
                plots=result_multiple.plots,
                show_formulas=show_formulas,
            )
        else:
            from .ui.tabs.multiple_regression_educational import render_multiple_regression_educational
            
            render_multiple_regression_educational(
                data=result_multiple.data,
                stats=result_multiple.stats,
                plots=result_multiple.plots,
                content=descriptions,
                formulas=formulas,
                show_formulas=show_formulas,
            )
            
    except Exception as e:
        logger.error(f"Multiple regression error: {e}")
        st.error(f"❌ Fehler: {str(e)}")


# =============================================================================
# TAB 3: DATENSATZ-INFO
# =============================================================================
with tab3:
    st.markdown("## 📚 Verfügbare Datensätze")
    
    st.markdown("### Einfache Regression")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🏪 Elektronikmarkt")
        st.markdown("""
        **Simulierte Daten** für Einsteiger
        
        - **X:** Verkaufsfläche (100 qm)
        - **Y:** Umsatz (Mio. €)
        - **n:** Anpassbar (10-100)
        - **Vorteil:** Wahre Parameter bekannt!
        """)
    
    with col2:
        st.markdown("#### 📢 Werbestudie")
        st.markdown("""
        **75 Städte** - Werbeeffektivität
        
        - **X:** Werbeausgaben ($)
        - **Y:** Umsatz ($)
        - **n:** 75 (fix)
        - **Kontext:** Handelskette
        """)
    
    with col3:
        st.markdown("#### 🍦 Eisverkauf")
        st.markdown("""
        **Temperatur & Konsum**
        
        - **X:** Temperatur (°C)
        - **Y:** Eisverkauf (Einheiten)
        - **n:** Anpassbar
        - **Klassisches Beispiel**
        """)
    
    st.markdown("---")
    st.markdown("### Multiple Regression")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏙️ Städte-Umsatzstudie")
        st.markdown("""
        **2 Prädiktoren** - Preis & Werbung
        
        - **X₁:** Produktpreis (CHF)
        - **X₂:** Werbeausgaben (1000 CHF)
        - **Y:** Umsatz (1000 CHF)
        - **Zeigt:** Ceteris-paribus-Interpretation
        """)
    
    with col2:
        st.markdown("#### 🏠 Häuserpreise")
        st.markdown("""
        **Dummy-Variable** enthalten!
        
        - **X₁:** Wohnfläche (sqft/10)
        - **X₂:** Pool (0/1) ← Dummy!
        - **Y:** Preis ($1000)
        - **Zeigt:** Kategoriale Variablen
        """)
    
    st.markdown("---")
    st.info("""
    **💡 Tipp:** Wählen Sie den Elektronikmarkt-Datensatz, um die Konzepte zu lernen.
    Die wahren Parameter sind bekannt, sodass Sie sehen können, wie gut die OLS-Schätzung funktioniert!
    """)


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    📖 Leitfaden zur Linearen Regression | 
    Pipeline: GET → CALCULATE → PLOT → DISPLAY |
    Alle Plots mit edukativem Kontext
</div>
""", unsafe_allow_html=True)

logger.info("Application rendered successfully")
