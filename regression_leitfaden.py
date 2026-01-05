"""
🎓 Umfassender Leitfaden zur Linearen Regression
=================================================
Ein didaktisches Tool zum Verstehen der einfachen linearen Regression.
Alle Konzepte auf einer Seite mit logischem roten Faden.

Starten mit: streamlit run regression_leitfaden.py
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import warnings
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- Hilfsfunktion für typ-sichere Pandas-Zugriffe ---
def safe_scalar(val):
    """Konvertiert Series/ndarray zu Skalar, falls nötig."""
    if isinstance(val, (pd.Series, np.ndarray)):
        return float(val.iloc[0] if hasattr(val, 'iloc') else val[0])
    return float(val)

# ---------------------------------------------------------
# ZENTRALE KONFIGURATION: Farben & Schriftgroessen
# ---------------------------------------------------------
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#2c3e50",
    "correct": "#2ecc71",
    "violated": "#e74c3c",
    "warning": "#f39c12",
    "neutral": "#95a5a6",
    "residual_pos": "#27ae60",
    "residual_neg": "#c0392b",
    "regression_line": "#e74c3c",
    "data_points": "#3498db",
    "confidence_band": "#85c1e9",
}

FONT_SIZES = {
    "axis_label_3d": 14,
    "axis_label_2d": 12,
    "title_3d": 14,
    "title_2d": 13,
    "tick_3d": 11,
    "legend": 10,
}

# ---------------------------------------------------------
# DATENSATZ-GENERIERUNG (zentralisiert)
# ---------------------------------------------------------
def generate_dataset(name, seed=42):
    """
    Generiert einen Datensatz basierend auf dem Namen.
    Gibt x, y, labels und Metadaten zurueck.
    """
    np.random.seed(seed)
    
    if name == "elektronikmarkt":
        # Default-Werte, werden durch Slider ueberschrieben
        return None  # Handled separately due to sliders
    
    elif name == "staedte":
        n = 75
        x2_preis = np.random.normal(5.69, 0.52, n)
        x2_preis = np.clip(x2_preis, 4.83, 6.49)
        x3_werbung = np.random.normal(1.84, 0.83, n)
        x3_werbung = np.clip(x3_werbung, 0.50, 3.10)
        y_base = 100 - 5 * x2_preis + 8 * x3_werbung
        noise = np.random.normal(0, 3.5, n)
        y = y_base + noise
        y = np.clip(y, 62.4, 91.2)
        y = (y - np.mean(y)) / np.std(y) * 6.49 + 77.37
        return {
            "x_preis": x2_preis,
            "x_werbung": x3_werbung,
            "y": y,
            "n": n,
            "x1_name": "Preis (CHF)",
            "x2_name": "Werbung (CHF1000)",
            "y_name": "Umsatz (1000 CHF)",
        }
    
    elif name == "haeuser":
        n = 1000
        x_wohnflaeche = np.random.normal(25.21, 2.92, n)
        x_wohnflaeche = np.clip(x_wohnflaeche, 20.03, 30.00)
        x_pool = np.random.binomial(1, 0.204, n).astype(float)
        y_base = 50 + 7.5 * x_wohnflaeche + 35 * x_pool
        noise = np.random.normal(0, 20, n)
        y = y_base + noise
        y = np.clip(y, 134.32, 345.20)
        y = (y - np.mean(y)) / np.std(y) * 42.19 + 247.66
        return {
            "x_wohnflaeche": x_wohnflaeche,
            "x_pool": x_pool,
            "y": y,
            "n": n,
            "x1_name": "Wohnflaeche (sqft/10)",
            "x2_name": "Pool (0/1)",
            "y_name": "Preis (USD)",
        }
    
    return None

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="📖 Leitfaden Lineare Regression",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .subsection-header {
        font-size: 1.4rem;
        font-weight: bold;
        color: #34495e;
        margin-top: 1.5rem;
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
        margin: 10px 0;
    }
    .interpretation-box {
        background-color: #d4edda;
        border: 1px solid #28a745;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def get_signif_stars(p):
    """Signifikanz-Codes wie in R"""
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    if p < 0.1:   return '.'
    return ' '

def get_signif_color(p):
    """Farbe basierend auf Signifikanz"""
    if p < 0.001: return '#006400'
    if p < 0.01:  return '#228B22'
    if p < 0.05:  return '#32CD32'
    if p < 0.1:   return '#FFA500'
    return '#DC143C'

# ---------------------------------------------------------
# PLOTLY HELPER FUNCTIONS FOR COMMON PLOT TYPES
# ---------------------------------------------------------
def create_plotly_scatter(x, y, x_label='X', y_label='Y', title='', 
                         marker_color='blue', marker_size=8, show_legend=True):
    """Create a basic plotly scatter plot"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers',
        marker=dict(size=marker_size, color=marker_color, opacity=0.7,
                   line=dict(width=1, color='white')),
        name='Data Points' if show_legend else None,
        showlegend=show_legend
    ))
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='plotly_white',
        hovermode='closest'
    )
    return fig

def create_plotly_scatter_with_line(x, y, y_pred, x_label='X', y_label='Y', title=''):
    """Create scatter plot with regression line"""
    fig = go.Figure()
    
    # Data points
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers',
        marker=dict(size=10, color='#1f77b4', opacity=0.7,
                   line=dict(width=2, color='white')),
        name='Datenpunkte'
    ))
    
    # Regression line
    fig.add_trace(go.Scatter(
        x=x, y=y_pred, mode='lines',
        line=dict(color='#e74c3c', width=3),
        name='Regressionslinie'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='plotly_white',
        hovermode='closest',
        showlegend=True
    )
    return fig

def create_plotly_3d_scatter(x1, x2, y, x1_label='X1', x2_label='X2', y_label='Y', 
                            title='', marker_color='red'):
    """Create 3D scatter plot"""
    fig = go.Figure(data=[go.Scatter3d(
        x=x1, y=x2, z=y,
        mode='markers',
        marker=dict(
            size=5,
            color=marker_color if isinstance(marker_color, str) else y,
            colorscale='Viridis' if not isinstance(marker_color, str) else None,
            opacity=0.7,
            colorbar=dict(title=y_label) if not isinstance(marker_color, str) else None
        ),
        name='Data Points'
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x1_label,
            yaxis_title=x2_label,
            zaxis_title=y_label
        ),
        template='plotly_white'
    )
    return fig

def create_plotly_3d_surface(X1_mesh, X2_mesh, Y_mesh, x1, x2, y,
                             x1_label='X1', x2_label='X2', y_label='Y', title=''):
    """Create 3D surface plot with data points"""
    fig = go.Figure()
    
    # Surface
    fig.add_trace(go.Surface(
        x=X1_mesh, y=X2_mesh, z=Y_mesh,
        colorscale='Viridis',
        opacity=0.7,
        name='Regression Plane',
        showscale=False
    ))
    
    # Data points
    fig.add_trace(go.Scatter3d(
        x=x1, y=x2, z=y,
        mode='markers',
        marker=dict(size=4, color='red', opacity=0.8),
        name='Data Points'
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x1_label,
            yaxis_title=x2_label,
            zaxis_title=y_label,
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2))
        ),
        template='plotly_white'
    )
    return fig

def create_plotly_residual_plot(y_pred, residuals, title='Residual Plot'):
    """Create residual plot"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals,
        mode='markers',
        marker=dict(size=8, color='blue', opacity=0.6),
        name='Residuals'
    ))
    
    fig.add_hline(y=0, line_dash='dash', line_color='red', line_width=2)
    
    fig.update_layout(
        title=title,
        xaxis_title='Fitted Values',
        yaxis_title='Residuals',
        template='plotly_white',
        hovermode='closest'
    )
    return fig

def create_plotly_bar(categories, values, title='', x_label='', y_label='',
                     colors=None):
    """Create bar chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors if colors else 'blue',
        opacity=0.7,
        text=[f'{v:.2f}' for v in values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='plotly_white'
    )
    return fig

def create_plotly_distribution(x_vals, y_vals, title='', x_label='', fill_area=None):
    """Create distribution plot"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode='lines',
        line=dict(color='black', width=2),
        name='Distribution'
    ))
    
    if fill_area is not None:
        mask = fill_area(x_vals)
        fig.add_trace(go.Scatter(
            x=x_vals[mask], y=y_vals[mask],
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.3)',
            line=dict(width=0),
            showlegend=False
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title='Density',
        template='plotly_white',
        hovermode='x'
    )
    return fig

# ---------------------------------------------------------
# R-OUTPUT DISPLAY (Simplified text-based display)
# ---------------------------------------------------------
def create_r_output_display(model, feature_name="X"):
    """
    Creates a structured display of R-style output using Streamlit components
    instead of matplotlib figure. This provides better interactivity.
    """
    # Extract all values
    resid = model.resid
    q = np.percentile(resid, [0, 25, 50, 75, 100])
    params = model.params
    bse = model.bse
    tvals = model.tvalues
    pvals = model.pvalues
    rse = np.sqrt(model.mse_resid)
    df_resid = int(model.df_resid)
    df_model = int(model.df_model)
    
    # Create formatted text output
    output_text = f"""
Python Replikation des R-Outputs: summary(lm_model)
===================================================

Residuals:
    Min      1Q  Median      3Q     Max
{q[0]:7.4f} {q[1]:7.4f} {q[2]:7.4f} {q[3]:7.4f} {q[4]:7.4f}

Coefficients:
             Estimate Std.Err  t val  Pr(>|t|)    
(Intercept)  {params[0]:9.4f} {bse[0]:8.4f} {tvals[0]:7.2f} {pvals[0]:10.4g} {get_signif_stars(pvals[0])}
{feature_name:<13}{params[1]:9.4f} {bse[1]:8.4f} {tvals[1]:7.2f} {pvals[1]:10.4g} {get_signif_stars(pvals[1])}
---
Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: {rse:.4f} on {df_resid} degrees of freedom
Multiple R-squared:  {model.rsquared:.4f},    Adjusted R-squared:  {model.rsquared_adj:.4f}
F-statistic: {model.fvalue:.1f} on {df_model} and {df_resid} DF,  p-value: {model.f_pvalue:.4g}
"""
    return output_text

# ---------------------------------------------------------
# SIDEBAR - INTERAKTIVE PARAMETER
# ---------------------------------------------------------
st.sidebar.markdown("# 🎛️ Parameter")

# === REGRESSIONS-MODUL AUSWAHL ===
st.sidebar.markdown("---")
st.sidebar.markdown("## 🎯 Modul-Auswahl")

regression_type = st.sidebar.radio(
    "Regressionsart:",
    ["📈 Einfache Regression", "📊 Multiple Regression"],
    index=0,
    help="Wählen Sie zwischen einfacher (1 Prädiktor) und multipler (mehrere Prädiktoren) Regression"
)

# Gemeinsame Navigation (harmonisiert)
nav_options_simple = [
    "1.0 Einleitung",
    "1.5 Mehrdimensionale Verteilungen",
    "2.0 Das Fundament",
    "2.5 Kovarianz & Korrelation",
    "3.0 OLS-Methode",
    "4.0 Güteprüfung",
    "5.0 Signifikanz & Tests",
    "5.5 ANOVA Gruppenvergleich",
    "5.6 Heteroskedastizität",
    "6.0 Fazit"
]
nav_options_mult = [
    "M1. Von der Linie zur Ebene",
    "M2. Das Grundmodell",
    "M3. OLS & Gauss-Markov",
    "M4. Modellvalidierung",
    "M5. Anwendungsbeispiel",
    "M6. Dummy-Variablen",
    "M7. Multikollinearität",
    "M8. Residuen-Diagnostik",
    "M9. Zusammenfassung"
]

st.sidebar.markdown("---")
with st.sidebar.expander("📍 Inhaltsverzeichnis", expanded=True):
    if regression_type == "📈 Einfache Regression":
        # Bei einfacher Regression: Alle Kapitel werden auf einmal angezeigt
        st.markdown("**Alle Kapitel werden geladen:**")
        for chapter in nav_options_simple:
            st.markdown(f"• {chapter}")
    else:
        # Bei multipler Regression: Alle Kapitel werden auf einmal angezeigt
        st.markdown("**Alle Kapitel werden geladen:**")
        for chapter in nav_options_mult:
            st.markdown(f"• {chapter}")

# Gemeinsamer Datensatz-Block
st.sidebar.markdown("---")
with st.sidebar.expander("📊 Datensatz", expanded=True):
    if regression_type == "📈 Einfache Regression":
        dataset_choice = st.selectbox(
            "Datensatz wählen:",
            ["🏪 Elektronikmarkt (simuliert)", "🏙️ Städte-Umsatzstudie (75 Städte)", "🏠 Häuserpreise mit Pool (1000 Häuser)"],
            index=0,
            help="Wählen Sie zwischen einem simulierten Datensatz, Städtedaten oder Häuserpreisen mit Dummy-Variable (Pool)."
        )
    else:
        dataset_choice_mult = st.selectbox(
            "Datensatz wählen:",
            ["🏙️ Städte-Umsatzstudie (75 Städte)", "🏠 Häuserpreise mit Pool (1000 Häuser)", "🏪 Elektronikmarkt (erweitert)"],
            index=0,
            help="Wählen Sie einen Datensatz für multiple Regression (2+ Prädiktoren).",
            key="mult_dataset"
        )

# Einheitlicher Daten-Parameter-Block mit Sliders
if regression_type == "📊 Multiple Regression":
    st.sidebar.markdown("---")
    with st.sidebar.expander("🎛️ Daten-Parameter", expanded=True):
        if dataset_choice_mult == "🏙️ Städte-Umsatzstudie (75 Städte)":
            st.markdown("**Stichproben-Eigenschaften:**")
            n_mult = st.slider("Anzahl Städte (n)", min_value=20, max_value=150, value=75, step=5,
                             help="Grösse der Stichprobe", key="n_mult_staedte")
            
            st.markdown("**Zufallskomponente:**")
            noise_mult_level = st.slider("Rauschen (σ)", min_value=1.0, max_value=8.0, value=3.5, step=0.5,
                                       help="Standardabweichung der Störgrösse", key="noise_mult_staedte")
            seed_mult = st.number_input("Random Seed", min_value=1, max_value=999, value=42,
                                      help="Zufallsseed für Reproduzierbarkeit", key="seed_mult_staedte")
        
        elif dataset_choice_mult == "🏠 Häuserpreise mit Pool (1000 Häuser)":
            st.markdown("**Stichproben-Eigenschaften:**")
            n_mult = st.slider("Anzahl Häuser (n)", min_value=100, max_value=2000, value=1000, step=100,
                             help="Grösse der Stichprobe", key="n_mult_haeuser")
            
            st.markdown("**Zufallskomponente:**")
            noise_mult_level = st.slider("Rauschen (σ)", min_value=5.0, max_value=40.0, value=20.0, step=5.0,
                                       help="Standardabweichung der Störgrösse", key="noise_mult_haeuser")
            seed_mult = st.number_input("Random Seed", min_value=1, max_value=999, value=42,
                                      help="Zufallsseed für Reproduzierbarkeit", key="seed_mult_haeuser")
        
        else:  # Elektronikmarkt
            st.markdown("**Stichproben-Eigenschaften:**")
            n_mult = st.slider("Anzahl Beobachtungen (n)", min_value=20, max_value=100, value=50, step=5,
                             help="Grösse der Stichprobe", key="n_mult_elektro")
            
            st.markdown("**Zufallskomponente:**")
            noise_mult_level = st.slider("Rauschen (σ)", min_value=0.1, max_value=1.0, value=0.35, step=0.05,
                                       help="Standardabweichung der Störgrösse", key="noise_mult_elektro")
            seed_mult = st.number_input("Random Seed", min_value=1, max_value=999, value=42,
                                      help="Zufallsseed für Reproduzierbarkeit", key="seed_mult_elektro")

# === MULTIPLE REGRESSION DATA PREPARATION (gemeinsam strukturierte Sidebar) ===
if regression_type == "📊 Multiple Regression":
    with st.spinner("Lade Datensatz..."):
        np.random.seed(int(seed_mult))

        if dataset_choice_mult == "🏙️ Städte-Umsatzstudie (75 Städte)":
            x2_preis = np.random.normal(5.69, 0.52, n_mult)
            x2_preis = np.clip(x2_preis, 4.83, 6.49)
            x3_werbung = np.random.normal(1.84, 0.83, n_mult)
            x3_werbung = np.clip(x3_werbung, 0.50, 3.10)
            y_base_mult = 100 - 5 * x2_preis + 8 * x3_werbung
            noise_mult = np.random.normal(0, noise_mult_level, n_mult)
            y_mult = y_base_mult + noise_mult
            y_mult = np.clip(y_mult, 62.4, 91.2)
            y_mult = (y_mult - np.mean(y_mult)) / np.std(y_mult) * 6.49 + 77.37

            X_mult = sm.add_constant(np.column_stack([x2_preis, x3_werbung]))
            model_mult = sm.OLS(y_mult, X_mult).fit()
            y_pred_mult = model_mult.predict(X_mult)

            x1_name, x2_name, y_name = "Preis (CHF)", "Werbung (CHF1000)", "Umsatz (1000 CHF)"

        elif dataset_choice_mult == "🏠 Häuserpreise mit Pool (1000 Häuser)":
            x2_wohnflaeche = np.random.normal(25.21, 2.92, n_mult)
            x2_wohnflaeche = np.clip(x2_wohnflaeche, 20.03, 30.00)

            x3_pool = np.random.binomial(1, 0.204, n_mult).astype(float)

            y_base_mult = 50 + 7.5 * x2_wohnflaeche + 35 * x3_pool
            noise_mult = np.random.normal(0, noise_mult_level, n_mult)
            y_mult = y_base_mult + noise_mult
            y_mult = np.clip(y_mult, 134.32, 345.20)
            y_mult = (y_mult - np.mean(y_mult)) / np.std(y_mult) * 42.19 + 247.66

            x2_preis = x2_wohnflaeche
            x3_werbung = x3_pool

            X_mult = sm.add_constant(np.column_stack([x2_preis, x3_werbung]))
            model_mult = sm.OLS(y_mult, X_mult).fit()
            y_pred_mult = model_mult.predict(X_mult)

            x1_name, x2_name, y_name = "Wohnfläche (sqft/10)", "Pool (0/1)", "Preis (USD)"

        else:  # Elektronikmarkt (erweitert)
            x2_flaeche = np.random.uniform(2, 12, n_mult)
            x3_marketing = np.random.uniform(0.5, 5.0, n_mult)

            y_base_mult = 0.6 + 0.48 * x2_flaeche + 0.15 * x3_marketing
            noise_mult = np.random.normal(0, noise_mult_level, n_mult)
            y_mult = y_base_mult + noise_mult

            x2_preis = x2_flaeche
            x3_werbung = x3_marketing

            X_mult = sm.add_constant(np.column_stack([x2_preis, x3_werbung]))
            model_mult = sm.OLS(y_mult, X_mult).fit()
            y_pred_mult = model_mult.predict(X_mult)

            x1_name, x2_name, y_name = "Verkaufsfläche (100qm)", "Marketing (10k€)", "Umsatz (Mio. €)"

    st.sidebar.markdown("---")
    with st.sidebar.expander("🔧 Anzeigeoptionen", expanded=False):
        show_formulas = st.checkbox("Formeln anzeigen", value=True,
                                    help="Zeige mathematische Formeln in der Anleitung")
        show_true_line = False

# === GEMEINSAME PARAMETER-SEKTION === (nur bei einfacher Regression)
if regression_type == "📈 Einfache Regression":
    has_true_line = False
    st.sidebar.markdown("---")
    with st.sidebar.expander("🎛️ Daten-Parameter", expanded=True):
        if dataset_choice == "🏪 Elektronikmarkt (simuliert)":
            # X-Variable als Dropdown (nur eine Option verfügbar)
            x_variable_options = ["Verkaufsfläche (100qm)"]
            x_variable = st.selectbox(
                "X-Variable (Prädiktor):",
                x_variable_options,
                index=0,
                help="Beim simulierten Datensatz ist nur die Verkaufsfläche als Prädiktor verfügbar."
            )
            
            st.markdown("**Stichproben-Eigenschaften:**")
            n = st.slider("Anzahl Beobachtungen (n)", min_value=8, max_value=50, value=12, step=1,
                         help="Grösse der Stichprobe (mehr Beobachtungen = präzisere Schätzungen)")
            
            st.markdown("**Wahre Parameter (bekannt bei Simulation):**")
            true_intercept = st.slider("Wahrer β₀ (Intercept)", min_value=-1.0, max_value=3.0, value=0.6, step=0.1,
                                      help="Y-Achsenabschnitt: Wert von Y wenn X=0")
            true_beta = st.slider("Wahre Steigung β₁", min_value=0.1, max_value=1.5, value=0.52, step=0.01,
                                 help="Steigung: Änderung in Y pro Einheit X")
            
            st.markdown("**Zufallskomponente:**")
            noise_level = st.slider("Rauschen (σ)", min_value=0.1, max_value=1.5, value=0.4, step=0.05,
                                   help="Standardabweichung der Störgrösse (mehr Rauschen = schlechteres R²)")
            seed = st.number_input("Random Seed", min_value=1, max_value=999, value=42,
                                  help="Zufallsseed für Reproduzierbarkeit")
            
            # Simulierte Daten generieren
            with st.spinner("Generiere Daten..."):
                np.random.seed(int(seed))
                x = np.linspace(2, 12, n)  # Verkaufsfläche in 100qm (200-1200qm)
                noise = np.random.normal(0, noise_level, n)
                y = true_intercept + true_beta * x + noise  # Umsatz in Mio. €
            
            # Variablen-Namen für konsistente Anzeige
            x_label = "Verkaufsfläche (100qm)"
            y_label = "Umsatz (Mio. €)"
            x_unit = "100 qm"
            y_unit = "Mio. €"
            context_title = "Elektronikfachmärkte"
            context_description = """
            Das Management möchte untersuchen:
            - **X** = Verkaufsfläche (in 100 qm)
            - **Y** = Umsatz (in Mio. €)
            
            **Fragen:**
            1. Wie stark steigt der Umsatz pro 100 qm mehr Fläche?
            2. Welchen Umsatz erwarten wir für eine 1200 qm Filiale?
            """
            has_true_line = True
        
        elif dataset_choice == "🏙️ Städte-Umsatzstudie (75 Städte)":
            # X-Variable als Dropdown (zwei Optionen verfügbar)
            x_variable_options = ["Werbung (CHF1000)", "Preis (CHF)"]
            x_variable = st.selectbox(
                "X-Variable (Prädiktor):",
                x_variable_options,
                index=0,
                help="Einfache Regression: Nur EIN Prädiktor → grösserer Fehlerterm (didaktisch wertvoll!)"
            )
        
        elif dataset_choice == "🏠 Häuserpreise mit Pool (1000 Häuser)":
            # X-Variable als Dropdown (zwei Optionen verfügbar)
            x_variable_options = ["Wohnfläche (sqft/10)", "Pool (0/1)"]
            x_variable = st.selectbox(
                "X-Variable (Prädiktor):",
                x_variable_options,
                index=0,
                help="Einfache Regression: Nur EIN Prädiktor. Pool ist eine Dummy-Variable (0 = kein Pool, 1 = Pool)."
            )
        else:
            x_variable = None
    
    st.sidebar.markdown("**Stichproben-Info:**")
    
    if dataset_choice == "🏠 Häuserpreise mit Pool (1000 Häuser)":
        st.sidebar.info("n = 1000 Häuser (fixiert)")
        n = 1000
    elif dataset_choice == "🏙️ Städte-Umsatzstudie (75 Städte)":
        st.sidebar.info("n = 75 Städte (fixiert)")
        n = 75
    
    # Datensatz-spezifische Generierung
    if dataset_choice == "🏠 Häuserpreise mit Pool (1000 Häuser)":
        # Häuserpreise-Datensatz generieren (basierend auf gegebenen Statistiken)
        np.random.seed(42)
        
        # Wohnfläche in sqft/10 (20.03 bis 30.00, Mittelwert 25.21, SD 2.92)
        x_wohnflaeche = np.random.normal(25.21, 2.92, n)
        x_wohnflaeche = np.clip(x_wohnflaeche, 20.03, 30.00)
        
        # Pool Dummy-Variable (20.4% haben Pool)
        x_pool = np.random.binomial(1, 0.204, n)
        
        # Preis als Funktion von Wohnfläche und Pool
        # Basierend auf: Preis Mittelwert 247.66, SD 42.19, Min 134.32, Max 345.20
        y_base = 50 + 7.5 * x_wohnflaeche + 35 * x_pool
        noise = np.random.normal(0, 20, n)
        y = y_base + noise
        y = np.clip(y, 134.32, 345.20)
        
        # Skaliere auf gewünschte Statistiken
        y = (y - np.mean(y)) / np.std(y) * 42.19 + 247.66
        
        if x_variable == "Wohnfläche (sqft/10)":
            x = x_wohnflaeche
            x_label = "Wohnfläche (sqft/10)"
            y_label = "Preis (USD)"
            x_unit = "sqft/10"
            y_unit = "USD"
            context_description = """
            Eine Studie von **1000 Hausverkäufen** in einer Universitätsstadt:
            - **X** = Wohnfläche (in sqft/10, d.h. 20.03 = 200.3 sqft)
            - **Y** = Hauspreis (in USD)
            
            **Erwartung:** Grössere Wohnfläche → höherer Preis?
            
            ⚠️ **Didaktisch:** Nur EIN Prädiktor → grosser Fehlerterm 
            (Pool-Ausstattung fehlt als Erklärungsvariable!)
            """
        else:  # Pool
            x = x_pool.astype(float)
            x_label = "Pool (0/1)"
            y_label = "Preis (USD)"
            x_unit = "0/1"
            y_unit = "USD"
            context_description = """
            Eine Studie von **1000 Hausverkäufen** in einer Universitätsstadt:
            - **X** = Pool-Vorhandensein (0 = kein Pool, 1 = Pool vorhanden)
            - **Y** = Hauspreis (in USD)
            
            **Erwartung:** Pool → höherer Preis? (Dummy-Variable!)
            
            ⚠️ **Didaktisch:** Dies zeigt den Effekt einer **kategorischen Variable** (Pool ja/nein).
            Nur 20.4% der Häuser haben einen Pool.
            
            💡 **Interpretation der Steigung β₁:**
            β₁ = durchschnittlicher Preisunterschied zwischen Häusern MIT Pool vs. OHNE Pool
            """
        
        context_title = "Häuserpreise-Studie"
        has_true_line = False
        true_intercept = 0
        true_beta = 0
        seed = 42
        
    elif dataset_choice == "🏙️ Städte-Umsatzstudie (75 Städte)":
        np.random.seed(42)
        
        # Generiere korrelierte Daten basierend auf den deskriptiven Statistiken
        x2_preis = np.random.normal(5.69, 0.52, n)
        x2_preis = np.clip(x2_preis, 4.83, 6.49)
        
        x3_werbung = np.random.normal(1.84, 0.83, n)
        x3_werbung = np.clip(x3_werbung, 0.50, 3.10)
        
        # y = f(preis, werbung) + noise
        y_base = 100 - 5 * x2_preis + 8 * x3_werbung
        noise = np.random.normal(0, 3.5, n)
        y = y_base + noise
        y = np.clip(y, 62.4, 91.2)
        
        # Skaliere auf gewünschte Statistiken
        y = (y - np.mean(y)) / np.std(y) * 6.49 + 77.37
        
        if x_variable == "Preis (CHF)":
            x = x2_preis
            x_label = "Preis (CHF)"
            y_label = "Umsatz (1'000 CHF)"
            x_unit = "CHF"
            y_unit = "1'000 CHF"
            context_description = """
            Eine Handelskette untersucht in **75 Städten**:
            - **X** = Produktpreis (in CHF)
            - **Y** = Umsatz (in 1'000 CHF)
            
            **Erwartung:** Höherer Preis → niedrigerer Umsatz?
            
            ⚠️ **Didaktisch:** Nur EIN Prädiktor → grosser Fehlerterm 
            (Werbung fehlt als Erklärungsvariable!)
            """
        else:  # Werbung
            x = x3_werbung
            x_label = "Werbung (CHF1000)"
            y_label = "Umsatz (1'000 CHF)"
            x_unit = "CHF1000"
            y_unit = "1'000 CHF"
            context_description = """
            Eine Handelskette untersucht in **75 Städten**:
            - **X** = Werbeausgaben (in 1'000 CHF)
            - **Y** = Umsatz (in 1'000 CHF)
            
            **Erwartung:** Mehr Werbung → höherer Umsatz?
            
            ⚠️ **Didaktisch:** Nur EIN Prädiktor → grosser Fehlerterm 
            (Preis fehlt als Erklärungsvariable!)
            """
        
        context_title = "Städte-Umsatzstudie"
        has_true_line = False
        true_intercept = 0  # Nicht bekannt bei echten Daten
        true_beta = 0
        seed = 42  # Fester Seed für konsistente ANOVA-Daten

    st.sidebar.markdown("---")
    with st.sidebar.expander("🔧 Anzeigeoptionen", expanded=False):
        show_formulas = st.checkbox("Formeln anzeigen", value=True,
                                    help="Zeige mathematische Formeln in der Anleitung")
        show_true_line = st.checkbox("Wahre Linie zeigen", value=has_true_line,
                                     help="Zeige die wahre Regressionslinie (nur bei Simulation)") if has_true_line else False

# ---------------------------------------------------------
# MODELL & KENNZAHLEN BERECHNEN (nur einfache Regression)
# ---------------------------------------------------------
if regression_type == "📈 Einfache Regression":
    df = pd.DataFrame({
        x_label: x,
        y_label: y
    })

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    y_pred = model.predict(X)
    y_mean = np.mean(y)

    b0, b1 = model.params[0], model.params[1]
    sse = np.sum((y - y_pred)**2)
    sst = np.sum((y - y_mean)**2)
    ssr = sst - sse
    mse = sse / (n - 2)
    msr = ssr / 1
    se_regression = np.sqrt(mse)
    sb1, sb0 = model.bse[1], model.bse[0]
    t_val = model.tvalues[1]
    f_val = model.fvalue
    df_resid = int(model.df_resid)
    x_mean, y_mean_val = np.mean(x), np.mean(y)
    cov_xy = np.sum((x - x_mean) * (y - y_mean_val)) / (n - 1)
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    corr_xy = cov_xy / (np.sqrt(var_x) * np.sqrt(var_y))
else:
    # Platzhalter, damit Variablen nicht versehentlich genutzt werden
    df = None
    model = None
    y_pred = None
    y_mean = None
    b0 = b1 = sse = sst = ssr = mse = msr = se_regression = sb1 = sb0 = t_val = f_val = df_resid = x_mean = y_mean_val = cov_xy = var_x = var_y = corr_xy = None

# =========================================================
# HAUPTINHALT - Bedingte Anzeige basierend auf Modulauswahl
# =========================================================

# =========================================================
# MULTIPLE REGRESSION MODULE
# =========================================================
if regression_type == "📊 Multiple Regression":
    st.markdown('<p class="main-header">📊 Leitfaden zur Multiplen Regression</p>', unsafe_allow_html=True)
    st.markdown("### Von der einfachen zur multiplen Regression – Mehrere Prädiktoren gleichzeitig")
    
    # =========================================================
    # M1: VON DER LINIE ZUR EBENE
    # =========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">M1. Von der Linie zur Ebene: Der konzeptionelle Sprung</p>', unsafe_allow_html=True)
    
    col_m1_1, col_m1_2 = st.columns([1.5, 1])
    
    with col_m1_1:
        st.markdown("""
        Bei der **einfachen linearen Regression** haben wir gesehen, wie eine Gerade den Zusammenhang 
        zwischen **einer** unabhängigen Variable X und der abhängigen Variable Y beschreibt.
        
        In der Praxis hängt aber eine Zielvariable oft von **mehreren Faktoren** ab:
        - Umsatz ← Preis, Werbung, Standort, Saison, ...
        - Gehalt ← Ausbildung, Erfahrung, Branche, ...
        - Aktienkurs ← Zinsen, Inflation, Gewinn, ...
        
        Die **multiple Regression** erweitert die einfache Regression, um diese Komplexität zu modellieren.
        """)
        
        st.info("""
        **🔑 Der zentrale Unterschied:**
        
        | Aspekt | Einfache Regression | Multiple Regression |
        |--------|---------------------|---------------------|
        | **Prädiktoren** | 1 Variable (X) | K Variablen (X₁, X₂, ..., Xₖ) |
        | **Geometrie** | Gerade in 2D | Ebene/Hyperebene in (K+1)D |
        | **Gleichung** | ŷ = b₀ + b₁x | ŷ = b₀ + b₁x₁ + b₂x₂ + ... + bₖxₖ |
        | **Interpretation** | "Pro Einheit X" | "Bei Konstanthaltung der anderen" |
        """)
    
    with col_m1_2:
        # 3D Visualisierung: Ebene statt Linie
        # Erstelle Mesh für die Ebene
        x1_range = np.linspace(x2_preis.min(), x2_preis.max(), 20)
        x2_range = np.linspace(x3_werbung.min(), x3_werbung.max(), 20)
        X1_mesh, X2_mesh = np.meshgrid(x1_range, x2_range)
        
        # Berechne Ebene
        Y_mesh = model_mult.params[0] + model_mult.params[1]*X1_mesh + model_mult.params[2]*X2_mesh
        
        # Create plotly 3D surface plot
        fig_3d_plane = create_plotly_3d_surface(
            X1_mesh, X2_mesh, Y_mesh,
            x2_preis, x3_werbung, y_mult,
            x1_label=x1_name,
            x2_label=x2_name,
            y_label=y_name,
            title='Multiple Regression: Ebene statt Gerade'
        )
        
        st.plotly_chart(fig_3d_plane, use_container_width=True)
        
    # =========================================================
    # M2: DAS GRUNDMODELL
    # =========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">M2. Das Grundmodell der Multiplen Regression</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Das multiple Regressionsmodell erweitert die einfache lineare Regression um **K unabhängige Variablen**.
    """)
    
    if show_formulas:
        st.markdown("### 📐 Das allgemeine Modell")
        st.latex(r"y_i = \beta_0 + \beta_1 \cdot x_{1i} + \beta_2 \cdot x_{2i} + \cdots + \beta_K \cdot x_{Ki} + \varepsilon_i")
        
        st.markdown(f"### 📊 Unser Beispiel: {dataset_choice_mult}")
        if dataset_choice_mult == "🏙️ Städte-Umsatzstudie (75 Städte)":
            st.latex(r"\text{Umsatz}_i = \beta_0 + \beta_1 \cdot \text{Preis}_i + \beta_2 \cdot \text{Werbung}_i + \varepsilon_i")
        elif dataset_choice_mult == "🏠 Häuserpreise mit Pool (1000 Häuser)":
            st.latex(r"\text{Preis}_i = \beta_0 + \beta_1 \cdot \text{Wohnfläche}_i + \beta_2 \cdot \text{Pool}_i + \varepsilon_i")
        else:
            st.latex(r"\text{Umsatz}_i = \beta_0 + \beta_1 \cdot \text{Fläche}_i + \beta_2 \cdot \text{Marketing}_i + \varepsilon_i")
    
    col_m2_1, col_m2_2 = st.columns([1, 1])
    
    with col_m2_1:
        st.markdown("### 📋 Modellkomponenten")
        st.markdown("""
        | Symbol | Bedeutung | Beispiel |
        |--------|-----------|----------|
        | **yᵢ** | Zielvariable (abhängig) | Umsatz in Stadt i |
        | **xₖᵢ** | k-ter Prädiktor (unabhängig) | Preis, Werbung in Stadt i |
        | **β₀** | Achsenabschnitt (Intercept) | Basis-Umsatz ohne Einflüsse |
        | **βₖ** | Partieller Regressionskoeffizient | Effekt von xₖ **ceteris paribus** |
        | **εᵢ** | Störgrösse | Alle anderen Einflüsse |
        """)
        
        st.success(f"""
        **🎯 Unser geschätztes Modell:**
        
        Umsatz = {model_mult.params[0]:.2f} 
                 {model_mult.params[1]:+.2f} · Preis 
                 {model_mult.params[2]:+.2f} · Werbung
        """)
    
    with col_m2_2:
        st.markdown("### 🔬 Partielle Koeffizienten")
        st.markdown(f"""
        **β₁ (Preis) = {model_mult.params[1]:.3f}**
        
        → Pro CHF Preiserhöhung sinkt der Umsatz um {abs(model_mult.params[1]):.2f} Tausend CHF,
        **wenn Werbung konstant gehalten wird**.
        
        **β₂ (Werbung) = {model_mult.params[2]:.3f}**
        
        → Pro 1000 CHF mehr Werbung steigt der Umsatz um {model_mult.params[2]:.2f} Tausend CHF,
        **wenn Preis konstant gehalten wird**.
        """)
        
        st.warning("""
        **⚠️ Wichtig: Ceteris Paribus**
        
        Die Interpretation "bei Konstanthaltung der anderen Variablen" ist zentral!
        
        Anders als bei der einfachen Regression misst βₖ den **isolierten Effekt**
        einer Variable.
        """)
    
    # Daten anzeigen
    st.markdown("### 📊 Die Daten")
    df_mult = pd.DataFrame({
        x1_name: x2_preis,
        x2_name: x3_werbung,
        y_name: y_mult
    })
    st.dataframe(df_mult.head(15).style.format({
        'Preis (CHF)': '{:.2f}',
        'Werbung (CHF1000)': '{:.2f}',
        'Umsatz (1000 CHF)': '{:.2f}'
    }), width='stretch')

    # =========================================================
    # M3: OLS & GAUSS-MARKOV
    # =========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">M3. OLS-Schätzer und Gauss-Markov Theorem</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Wie bei der einfachen Regression bestimmen wir die Koeffizienten durch **Minimierung der Fehlerquadratsumme**.
    """)
    
    if show_formulas:
        st.markdown("### 📐 OLS-Zielfunktion")
        st.latex(r"\min \sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n} (y_i - b_0 - b_1 \cdot x_{1i} - b_2 \cdot x_{2i} - \cdots - b_K \cdot x_{Ki})^2")
        
        st.markdown("### 📊 Matrixform (elegant!)")
        st.latex(r"\mathbf{b} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}")
        st.markdown("""
        Wo:
        - **y** ist der Vektor der abhängigen Variable (n×1)
        - **X** ist die Design-Matrix der Prädiktoren (n×(K+1))
        - **b** ist der Vektor der geschätzten Koeffizienten ((K+1)×1)
        """)
    
    col_m3_1, col_m3_2 = st.columns([1.2, 1])
    
    with col_m3_1:
        st.markdown("### 🏆 Gauss-Markov Theorem")
        st.markdown("""
        Wenn die folgenden **Annahmen** erfüllt sind:
        
        1. **Linearität**: E(ε|X) = 0
        2. **Homoskedastizität**: Var(ε|X) = σ²
        3. **Keine Autokorrelation**: Cov(εᵢ, εⱼ) = 0
        4. **Keine perfekte Multikollinearität**: X hat vollen Rang
        
        Dann ist der OLS-Schätzer **BLUE**:
        - **B**est: Kleinste Varianz unter allen linearen Schätzern
        - **L**inear: Lineare Funktion der Daten
        - **U**nbiased: Erwartungstreu, E(b) = β
        - **E**stimator: Schätzer für die wahren Parameter
        """)
        
        # Residuen-Plot
        fig_resid = create_plotly_residual_plot(y_pred_mult, model_mult.resid, title="Residual Plot")
        st.plotly_chart(fig_resid, use_container_width=True)
            
    with col_m3_2:
        st.markdown("### 📊 Unsere Schätzungen")
        params_df = pd.DataFrame({
            'Koeffizient': ['β₀ (Intercept)', f'β₁ ({x1_name.split("(")[0].strip()})', f'β₂ ({x2_name.split("(")[0].strip()})'],
            'Schätzwert': [f'{model_mult.params[0]:.4f}', f'{model_mult.params[1]:.4f}', f'{model_mult.params[2]:.4f}'],
            'Std. Error': [f'{model_mult.bse[0]:.4f}', f'{model_mult.bse[1]:.4f}', f'{model_mult.bse[2]:.4f}']
        })
        st.dataframe(params_df, width='stretch', hide_index=True)
        
        st.success(f"""
        **✅ Modellgüte:**
        
        - R² = {model_mult.rsquared:.4f} ({model_mult.rsquared*100:.1f}%)
        - Adjustiertes R² = {model_mult.rsquared_adj:.4f}
        - F-Statistik = {model_mult.fvalue:.2f}
        - p-Wert (F-Test) = {model_mult.f_pvalue:.4g}
        """)
    
    # 3D Residual Visualization Toggle
    show_3d_resid_m3 = st.checkbox("🎲 3D-Residuen visualisieren (Abstände zur Ebene)", value=False)
    
    if show_3d_resid_m3:
        st.markdown("### 🎲 3D-Visualisierung: Residuen als Abstände zur Regressions-Ebene")
        
        # Create 3D residual plot with plotly
        x1_range = np.linspace(x2_preis.min(), x2_preis.max(), 20)
        x2_range = np.linspace(x3_werbung.min(), x3_werbung.max(), 20)
        X1_mesh, X2_mesh = np.meshgrid(x1_range, x2_range)
        Y_mesh = model_mult.params[0] + model_mult.params[1]*X1_mesh + model_mult.params[2]*X2_mesh
        
        fig_3d_resid = go.Figure()
        
        # Add regression surface
        fig_3d_resid.add_trace(go.Surface(
            x=X1_mesh, y=X2_mesh, z=Y_mesh,
            colorscale='Viridis',
            opacity=0.7,
            name='Regression Plane',
            showscale=False
        ))
        
        # Add data points
        fig_3d_resid.add_trace(go.Scatter3d(
            x=x2_preis, y=x3_werbung, z=y_mult,
            mode='markers',
            marker=dict(size=5, color='red', opacity=0.8),
            name='Datenpunkte'
        ))
        
        # Add residual lines
        for i in range(len(x2_preis)):
            fig_3d_resid.add_trace(go.Scatter3d(
                x=[x2_preis[i], x2_preis[i]],
                y=[x3_werbung[i], x3_werbung[i]],
                z=[y_pred_mult[i], y_mult[i]],
                mode='lines',
                line=dict(color='black', width=2),
                opacity=0.3,
                showlegend=False
            ))
        
        fig_3d_resid.update_layout(
            title='OLS: Minimierung der Residuen-Quadratsumme<br>(Vertikale Abstände zur Ebene)',
            scene=dict(
                xaxis_title=x1_name,
                yaxis_title=x2_name,
                zaxis_title=y_name,
                camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2))
            ),
            template='plotly_white',
            height=600
        )
        
        st.plotly_chart(fig_3d_resid, use_container_width=True)
                
        st.info("""
        **💡 3D-Interpretation:**
        
        - **Schwarze Linien** = Residuen (Abstände der roten Punkte zur Ebene)
        - **OLS minimiert** die Summe der quadrierten Längen dieser Linien
        - Je kleiner die Linien, desto besser passt die Ebene zu den Daten
        - **BLUE-Eigenschaft**: Diese Methode liefert die beste lineare unverzerrte Schätzung!
        """)

    # =========================================================
    # M4: MODELLVALIDIERUNG
    # =========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">M4. Modellvalidierung: R² und Adjustiertes R²</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Wie gut ist unser Modell? Wir brauchen Kennzahlen, die die **Erklärungskraft** messen.
    """)
    
    # Berechne Kennzahlen
    sst_mult = np.sum((y_mult - np.mean(y_mult))**2)
    sse_mult = np.sum(model_mult.resid**2)
    ssr_mult = sst_mult - sse_mult
    
    col_m4_1, col_m4_2 = st.columns([1.5, 1])
    
    with col_m4_1:
        if show_formulas:
            st.markdown("### 📐 Bestimmtheitsmass R²")
            st.latex(r"R^2 = 1 - \frac{SSE}{SST} = \frac{SSR}{SST}")
            st.latex(r"R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}")
        
        st.markdown(f"""
        **Interpretation:**
        
        R² = {model_mult.rsquared:.4f} bedeutet: **{model_mult.rsquared*100:.1f}%** der Varianz in Y
        wird durch die Prädiktoren X₁, X₂ erklärt.
        
        **⚠️ Problem:** R² steigt **immer**, wenn wir neue Variablen hinzufügen, 
        selbst wenn sie irrelevant sind!
        """)
        
        # Varianzzerlegung
        fig_var_mult = create_plotly_bar(
            categories=['SST\n(Total)', 'SSR\n(Erklärt)', 'SSE\n(Unerklärt)'],
            values=[sst_mult, ssr_mult, sse_mult],
            colors=['gray', 'green', 'red'],
            title=f"Varianzzerlegung: R² = {model_mult.rsquared:.4f}"
        )
        st.plotly_chart(fig_var_mult, use_container_width=True)
            
    with col_m4_2:
        if show_formulas:
            st.markdown("### 📐 Adjustiertes R²")
            st.latex(r"R^2_{adj} = 1 - (1-R^2) \cdot \frac{n-1}{n-K-1}")
        
        st.markdown(f"""
        **Adjustiertes R² = {model_mult.rsquared_adj:.4f}**
        
        **Vorteile:**
        - Bestraft unnötige Komplexität (mehr K → Strafe)
        - Erlaubt fairen Vergleich von Modellen
        - Kann sogar sinken beim Hinzufügen schwacher Prädiktoren!
        
        **Interpretation:**
        
        Unser Modell mit K=2 Prädiktoren hat:
        - R² = {model_mult.rsquared:.4f}
        - R²_adj = {model_mult.rsquared_adj:.4f}
        
        Die Differenz von {(model_mult.rsquared - model_mult.rsquared_adj):.4f} ist klein
        → Die Prädiktoren sind **substanziell relevant**.
        """)
        
        # Vergleich
        st.info(f"""
        **📊 Vergleich:**
        
        | Mass | Wert | Deutung |
        |-----|------|---------|
        | R² | {model_mult.rsquared:.4f} | Roh-Erklärungskraft |
        | R²_adj | {model_mult.rsquared_adj:.4f} | Korrigiert für Komplexität |
        | Differenz | {(model_mult.rsquared - model_mult.rsquared_adj):.4f} | Sehr klein → gut! |
        """)
    
    # 3D Variance Decomposition Toggle
    show_3d_var_m4 = st.checkbox("🎲 3D-Varianzzerlegung visualisieren", value=False)
    
    if show_3d_var_m4:
        st.markdown("### 🎲 3D-Visualisierung: Varianzzerlegung im Prädiktorraum")
        
        # Create side-by-side 3D plots using plotly subplots
        from plotly.subplots import make_subplots
        
        fig_3d_var = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=(f'SSR (Erklärt): {ssr_mult:.1f}<br>Varianz durch Modell',
                          f'SSE (Unerklärt): {sse_mult:.1f}<br>Nicht erfasste Varianz')
        )
        
        # Left: Explained variance (SSR)
        fig_3d_var.add_trace(
            go.Scatter3d(
                x=x2_preis, y=x3_werbung, z=y_pred_mult,
                mode='markers',
                marker=dict(
                    size=5,
                    color=y_pred_mult,
                    colorscale='Greens',
                    opacity=0.7,
                    showscale=True,
                    colorbar=dict(x=0.45, len=0.5)
                ),
                name='Predicted'
            ),
            row=1, col=1
        )
        
        # Right: Unexplained variance (SSE)
        residual_sizes = 3 + np.abs(model_mult.resid) * 5  # Scale for visibility
        fig_3d_var.add_trace(
            go.Scatter3d(
                x=x2_preis, y=x3_werbung, z=model_mult.resid,
                mode='markers',
                marker=dict(
                    size=residual_sizes,
                    color=model_mult.resid,
                    colorscale='Reds',
                    opacity=0.7,
                    showscale=True,
                    colorbar=dict(x=1.05, len=0.5)
                ),
                name='Residuals'
            ),
            row=1, col=2
        )
        
        # Add zero plane for residuals
        x_range = [x2_preis.min(), x2_preis.max()]
        y_range = [x3_werbung.min(), x3_werbung.max()]
        xx, yy = np.meshgrid(x_range, y_range)
        zz = np.zeros_like(xx)
        
        fig_3d_var.add_trace(
            go.Surface(x=xx, y=yy, z=zz, opacity=0.2,
                      colorscale=[[0, 'gray'], [1, 'gray']],
                      showscale=False),
            row=1, col=2
        )
        
        fig_3d_var.update_layout(
            height=600,
            template='plotly_white',
            scene=dict(
                xaxis_title=x1_name,
                yaxis_title=x2_name,
                zaxis_title=y_name,
                camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2))
            ),
            scene2=dict(
                xaxis_title=x1_name,
                yaxis_title=x2_name,
                zaxis_title='Residuen',
                camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2))
            )
        )
        
        st.plotly_chart(fig_3d_var, use_container_width=True)
                
        st.info(f"""
        **💡 3D-Interpretation:**
        
        - **Links (grün)**: Vorhergesagte Werte ŷ → zeigt die **Systematik** die unser Modell erfasst
        - **Rechts (rot)**: Residuen e → zeigt die **Abweichungen** die unser Modell nicht erklärt
        - **R² = {model_mult.rsquared:.4f}** bedeutet: {model_mult.rsquared*100:.1f}% der Varianz ist "grün" (erklärt)
        - Grössere rote Punkte = grössere Residuen = schlechtere Vorhersage an dieser Stelle
        """)

    # =========================================================
    # M5: ANWENDUNGSBEISPIEL
    # =========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">M5. Anwendungsbeispiel und Interpretation</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Wie nutzen wir unser Modell in der Praxis? Schauen wir uns konkrete Szenarien an.
    """)
    
    col_m5_1, col_m5_2 = st.columns([1, 1])
    
    with col_m5_1:
        st.markdown("### 🔮 Prognose")
        st.markdown(f"Wir wollen {y_name.split('(')[0].strip()} für einen neuen Datenpunkt vorhersagen.")
        
        # Interaktive Eingabe - dynamic ranges based on dataset
        if dataset_choice_mult == "🏙️ Städte-Umsatzstudie (75 Städte)":
            slider1_val = st.slider(x1_name, min_value=4.5, max_value=7.0, value=5.5, step=0.1)
            slider2_val = st.slider(x2_name, min_value=0.5, max_value=3.5, value=2.0, step=0.1)
        elif dataset_choice_mult == "🏠 Häuserpreise mit Pool (1000 Häuser)":
            slider1_val = st.slider(x1_name, min_value=20.0, max_value=30.0, value=25.0, step=0.5)
            slider2_val = st.slider(x2_name, min_value=0.0, max_value=1.0, value=0.0, step=1.0)
        else:  # Elektronikmarkt
            slider1_val = st.slider(x1_name, min_value=2.0, max_value=12.0, value=7.0, step=0.5)
            slider2_val = st.slider(x2_name, min_value=0.5, max_value=5.0, value=2.5, step=0.5)
        
        # Prognose berechnen
        new_X = np.array([1, slider1_val, slider2_val])
        pred_value = model_mult.predict(new_X)[0]
        
        # Konfidenzintervall
        pred_frame = pd.DataFrame({'const': [1], 'x1': [slider1_val], 'x2': [slider2_val]})
        pred_obj = model_mult.get_prediction(pred_frame)
        pred_summary = pred_obj.summary_frame(alpha=0.05)
        
        st.success(f"""
        **Prognose für:**
        - {x1_name} = {slider1_val:.2f}
        - {x2_name} = {slider2_val:.2f}
        
        **Erwarteter {y_name}:**
        
        {pred_value:.2f}
        
        **95% Konfidenzintervall:**
        [{pred_summary['mean_ci_lower'].values[0]:.2f}, {pred_summary['mean_ci_upper'].values[0]:.2f}]
        """)
        
        if show_formulas:
            st.latex(r"\hat{y} = b_0 + b_1 \cdot x_1 + b_2 \cdot x_2")
            st.latex(f"\\hat{{y}} = {model_mult.params[0]:.2f} + {model_mult.params[1]:.2f} \\cdot {slider1_val:.2f} + {model_mult.params[2]:.2f} \\cdot {slider2_val:.2f}")
            st.latex(f"\\hat{{y}} = {pred_value:.2f}")
    
    with col_m5_2:
        st.markdown("### 📊 Sensitivitätsanalyse")
        st.markdown(f"Wie verändert sich {y_name.split('(')[0].strip()} bei Änderung der Variablen?")
        
        # Sensitivität: Variable 1
        if dataset_choice_mult == "🏙️ Städte-Umsatzstudie (75 Städte)":
            var1_range = np.linspace(4.5, 7.0, 50)
            var2_range = np.linspace(0.5, 3.5, 50)
        elif dataset_choice_mult == "🏠 Häuserpreise mit Pool (1000 Häuser)":
            var1_range = np.linspace(20.0, 30.0, 50)
            var2_range = np.array([0.0, 1.0])  # Dummy variable
        else:  # Elektronikmarkt
            var1_range = np.linspace(2.0, 12.0, 50)
            var2_range = np.linspace(0.5, 5.0, 50)
        
        response_var1 = model_mult.params[0] + model_mult.params[1]*var1_range + model_mult.params[2]*slider2_val
        
        # Create sensitivity plot with plotly
        fig_sens = go.Figure()
        
        # Variable 1 sensitivity line
        fig_sens.add_trace(go.Scatter(
            x=var1_range, y=response_var1,
            mode='lines',
            line=dict(color='blue', width=3),
            name='Predicted Response'
        ))
        
        # Current value point
        fig_sens.add_trace(go.Scatter(
            x=[slider1_val], y=[pred_value],
            mode='markers',
            marker=dict(size=15, color='red'),
            name='Aktuell'
        ))
        
        fig_sens.update_layout(
            title=f'Sensitivität {x1_name.split("(")[0].strip()}<br>({x2_name.split("(")[0].strip()}={slider2_val:.1f} konstant)',
            xaxis_title=x1_name,
            yaxis_title=y_name,
            template='plotly_white',
            hovermode='x'
        )
        
        st.plotly_chart(fig_sens, use_container_width=True)
        
    # =========================================================
    # M6: DUMMY-VARIABLEN
    # =========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">M6. Dummy-Variablen: Kategoriale Prädiktoren</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Nicht alle Prädiktoren sind numerisch! Was ist mit **kategorialen Variablen** wie Region, 
    Geschlecht, oder Produkttyp?
    
    **Lösung: Dummy-Variablen** (0/1-Kodierung)
    """)
    
    # Erstelle Dummy-Daten
    np.random.seed(42)
    regions = np.random.choice(['Nord', 'Süd', 'Ost'], size=n_mult)
    df_dummy = pd.DataFrame({
        'Preis': x2_preis,
        'Werbung': x3_werbung,
        'Region': regions,
        'Umsatz': y_mult
    })
    
    # Dummy-Kodierung
    df_dummy_encoded = pd.get_dummies(df_dummy, columns=['Region'], drop_first=True)
    
    col_m6_1, col_m6_2 = st.columns([1, 1])
    
    with col_m6_1:
        st.markdown("### 📋 Konzept")
        st.markdown("""
        Für eine kategoriale Variable mit **m Ausprägungen** erstellen wir **m-1 Dummy-Variablen**.
        
        **Beispiel: Region (3 Ausprägungen)**
        - Nord, Süd, Ost
        - Wir brauchen **2 Dummies**: Region_Ost, Region_Süd
        - **Referenzkategorie**: Nord (beide Dummies = 0)
        """)
        
        st.dataframe(df_dummy[['Region']].head(10), width='stretch')
        st.markdown("**→ wird zu →**")
        st.dataframe(df_dummy_encoded[['Region_Ost', 'Region_Süd']].head(10), width='stretch')
        
        st.warning("""
        **⚠️ Dummy-Variable Trap:**
        
        Niemals **alle** m Dummies verwenden! Das führt zu perfekter Multikollinearität.
        
        Grund: Region_Nord = 1 - Region_Ost - Region_Süd
        """)
    
    with col_m6_2:
        if show_formulas:
            st.markdown("### 📐 Modell mit Dummies")
            st.latex(r"\text{Umsatz}_i = \beta_0 + \beta_1 \cdot \text{Preis}_i + \beta_2 \cdot \text{Werbung}_i + \beta_3 \cdot \text{Ost}_i + \beta_4 \cdot \text{Süd}_i + \varepsilon_i")
        
        st.markdown("### 📊 Interpretation")
        st.markdown("""
        **β₀:** Basis-Umsatz in der **Referenzregion** (Nord)
        
        **β₃ (Ost-Dummy):** Zusätzlicher Umsatz in **Ost** verglichen mit Nord
        (ceteris paribus)
        
        **β₄ (Süd-Dummy):** Zusätzlicher Umsatz in **Süd** verglichen mit Nord
        (ceteris paribus)
        
        **Prognose für Ost:**
        ŷ = β₀ + β₁·Preis + β₂·Werbung + β₃·1 + β₄·0
        
        **Prognose für Nord:**
        ŷ = β₀ + β₁·Preis + β₂·Werbung + β₃·0 + β₄·0
        """)
        
        # Modell mit Dummies fitten
        X_dummy = df_dummy_encoded[['Preis', 'Werbung', 'Region_Ost', 'Region_Süd']]
        X_dummy_const = sm.add_constant(X_dummy)
        model_dummy = sm.OLS(df_dummy_encoded['Umsatz'], X_dummy_const).fit()
        
        st.success(f"""
        **Unser Modell:**
        
        β₀ = {model_dummy.params[0]:.2f} (Nord-Basis)
        β₁ = {model_dummy.params[1]:.2f} (Preis)
        β₂ = {model_dummy.params[2]:.2f} (Werbung)
        β₃ = {model_dummy.params[3]:.2f} (Ost-Effekt)
        β₄ = {model_dummy.params[4]:.2f} (Süd-Effekt)
        """)

    # =========================================================
    # M7: MULTIKOLLINEARITÄT
    # =========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">M7. Multikollinearität: Wenn Prädiktoren korreliert sind</p>', unsafe_allow_html=True)
    
    st.markdown("""
    **Multikollinearität** liegt vor, wenn unabhängige Variablen **stark miteinander korrelieren**.
    
    Das ist ein **Problem**, weil es schwer wird, die individuellen Effekte zu trennen!
    """)
    
    # Add 3D toggle
    show_3d_m7 = st.checkbox("🎲 3D-Ansicht aktivieren (Multikollinearität)", value=False, help="Zeigt die Beziehung zwischen beiden Prädiktoren und der Zielvariable in 3D - Multikollinearität wird durch die Ausrichtung der Punktwolke sichtbar")
    
    col_m7_1, col_m7_2 = st.columns([1.2, 1])
    
    with col_m7_1:
        st.markdown("### 🔍 Diagnose")
        
        if show_3d_m7:
            # 3D Scatter: Zeigt wie Prädiktoren zusammen die Zielvariable beeinflussen
            x1_range = np.linspace(x2_preis.min(), x2_preis.max(), 20)
            x2_range = np.linspace(x3_werbung.min(), x3_werbung.max(), 20)
            X1_mesh, X2_mesh = np.meshgrid(x1_range, x2_range)
            Y_mesh = model_mult.params[0] + model_mult.params[1]*X1_mesh + model_mult.params[2]*X2_mesh
            
            fig_3d_m7 = go.Figure()
            
            # Add regression surface
            fig_3d_m7.add_trace(go.Surface(
                x=X1_mesh, y=X2_mesh, z=Y_mesh,
                colorscale='RdBu',
                opacity=0.6,
                showscale=False,
                name='Regression Plane'
            ))
            
            # Add scatter plot of data points
            fig_3d_m7.add_trace(go.Scatter3d(
                x=x2_preis, y=x3_werbung, z=y_mult,
                mode='markers',
                marker=dict(
                    size=5,
                    color=y_mult,
                    colorscale='Viridis',
                    opacity=0.7,
                    showscale=True,
                    colorbar=dict(title=y_name)
                ),
                name='Data Points'
            ))
            
            fig_3d_m7.update_layout(
                title='3D: Multikollinearität Visualisierung<br>(Korrelation zwischen Prädiktoren sichtbar in Punktverteilung)',
                scene=dict(
                    xaxis_title=x1_name,
                    yaxis_title=x2_name,
                    zaxis_title=y_name,
                    camera=dict(eye=dict(x=1.5, y=-1.5, z=1.3))
                ),
                template='plotly_white',
                height=600
            )
            
            st.plotly_chart(fig_3d_m7, use_container_width=True)
                        
            st.info("""
            **💡 3D-Interpretation:**
            
            - Wenn Prädiktoren **unkorreliert** sind: Punkte bilden eine "Wolke" über die gesamte x-y-Ebene
            - Wenn Prädiktoren **korreliert** sind: Punkte liegen entlang einer Diagonale/Linie in der x-y-Ebene
            - **Problem:** Korrelierte Prädiktoren → schwer zu trennen, welcher Prädiktor welchen Effekt hat!
            """)
        else:
            # Original 2D Correlation Matrix
            # Korrelationsmatrix
            corr_matrix = np.corrcoef(x2_preis, x3_werbung)
            
            # Use dynamic names
            var1_short = x1_name.split('(')[0].strip()
            var2_short = x2_name.split('(')[0].strip()
            
            # Create plotly heatmap
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=[var1_short, var2_short],
                y=[var1_short, var2_short],
                colorscale='RdBu_r',
                zmid=0,
                zmin=-1,
                zmax=1,
                text=[[f'{corr_matrix[i, j]:.3f}' for j in range(2)] for i in range(2)],
                texttemplate='%{text}',
                textfont={"size": 16, "color": "black"},
                showscale=True,
                colorbar=dict(title="Korrelation")
            ))
            
            fig_corr.update_layout(
                title='Korrelationsmatrix der Prädiktoren',
                template='plotly_white',
                xaxis=dict(side='bottom'),
                height=500
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
                        
            st.info(f"""
            **Korrelation(Preis, Werbung) = {corr_matrix[0,1]:.3f}**
            
            Interpretation:
            - |r| < 0.3: Schwach → kein Problem
            - 0.3 < |r| < 0.7: Moderat → akzeptabel
            - |r| > 0.7: Stark → **Multikollinearität!**
            
            **Unser Fall:** {
                "Schwach - kein Problem ✅" if abs(corr_matrix[0,1]) < 0.3 else
                "Moderat - akzeptabel ⚠️" if abs(corr_matrix[0,1]) < 0.7 else
                "Stark - Multikollinearität! ❌"
            }
            """)
    
    with col_m7_2:
        st.markdown("### 📊 VIF (Variance Inflation Factor)")
        
        if show_formulas:
            st.latex(r"VIF_k = \frac{1}{1 - R_k^2}")
            st.markdown("""
            Wo R²ₖ das R² ist, wenn wir xₖ durch alle anderen Prädiktoren vorhersagen.
            """)
        
        # Berechne VIF
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        X_for_vif = np.column_stack([x2_preis, x3_werbung])
        vif_data = pd.DataFrame({
            'Variable': [x1_name.split('(')[0].strip(), x2_name.split('(')[0].strip()],
            'VIF': [variance_inflation_factor(X_for_vif, i) for i in range(X_for_vif.shape[1])]
        })
        
        st.dataframe(vif_data, width='stretch', hide_index=True)
        
        st.markdown("""
        **Interpretation:**
        - VIF < 5: Keine Multikollinearität ✅
        - 5 < VIF < 10: Moderate Multikollinearität ⚠️
        - VIF > 10: Starke Multikollinearität ❌
        """)
        
        st.warning("""
        **⚠️ Konsequenzen von Multikollinearität:**
        
        1. **Standardfehler steigen** → unsichere Schätzungen
        2. **Koeffizienten instabil** → ändern sich stark bei kleinen Datenänderungen
        3. **t-Tests unzuverlässig** → falsche Schlüsse über Signifikanz
        4. **Interpretation schwierig** → "ceteris paribus" fraglich
        
        **Lösungen:**
        - Variable(n) entfernen
        - Mehr Daten sammeln
        - Ridge/Lasso Regression
        - Hauptkomponentenanalyse
        """)

    # =========================================================
    # M8: RESIDUEN-DIAGNOSTIK
    # =========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">M8. Residuen-Diagnostik: Modellprüfung</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Bevor wir unserem Modell vertrauen, müssen wir die **Gauss-Markov Annahmen** prüfen!
    """)
    
    # Diagnostik-Plots using plotly subplots
    from plotly.subplots import make_subplots
    from scipy.stats import probplot
    from statsmodels.stats.outliers_influence import OLSInfluence
    
    fig_diag = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Residuals vs Fitted<br>(Linearität & Homoskedastizität)',
                       'Normal Q-Q<br>(Normalität)',
                       'Scale-Location<br>(Homoskedastizität)',
                       'Residuals vs Leverage<br>(Einflussreiche Punkte)')
    )
    
    # 1. Residuen vs. Fitted
    fig_diag.add_trace(
        go.Scatter(x=y_pred_mult, y=model_mult.resid,
                  mode='markers',
                  marker=dict(size=6, opacity=0.6),
                  showlegend=False),
        row=1, col=1
    )
    fig_diag.add_hline(y=0, line_dash='dash', line_color='red', line_width=2,
                      row=1, col=1)
    
    # 2. Q-Q Plot
    qq = probplot(model_mult.resid, dist="norm")
    fig_diag.add_trace(
        go.Scatter(x=qq[0][0], y=qq[0][1],
                  mode='markers',
                  marker=dict(size=6, opacity=0.6),
                  showlegend=False),
        row=1, col=2
    )
    # Add reference line
    fig_diag.add_trace(
        go.Scatter(x=qq[0][0], y=qq[1][1] + qq[1][0]*qq[0][0],
                  mode='lines',
                  line=dict(color='red', dash='dash'),
                  showlegend=False),
        row=1, col=2
    )
    
    # 3. Scale-Location
    standardized_resid = model_mult.resid / np.std(model_mult.resid)
    fig_diag.add_trace(
        go.Scatter(x=y_pred_mult, y=np.sqrt(np.abs(standardized_resid)),
                  mode='markers',
                  marker=dict(size=6, opacity=0.6),
                  showlegend=False),
        row=2, col=1
    )
    
    # 4. Residuals vs Leverage
    influence = OLSInfluence(model_mult)
    leverage = influence.hat_matrix_diag
    fig_diag.add_trace(
        go.Scatter(x=leverage, y=standardized_resid,
                  mode='markers',
                  marker=dict(size=6, opacity=0.6),
                  showlegend=False),
        row=2, col=2
    )
    fig_diag.add_hline(y=0, line_dash='dash', line_color='red', line_width=2,
                      row=2, col=2)
    
    # Update axes labels
    fig_diag.update_xaxes(title_text="Fitted values", row=1, col=1)
    fig_diag.update_yaxes(title_text="Residuals", row=1, col=1)
    
    fig_diag.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
    fig_diag.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
    
    fig_diag.update_xaxes(title_text="Fitted values", row=2, col=1)
    fig_diag.update_yaxes(title_text="√|Standardized residuals|", row=2, col=1)
    
    fig_diag.update_xaxes(title_text="Leverage", row=2, col=2)
    fig_diag.update_yaxes(title_text="Standardized residuals", row=2, col=2)
    
    fig_diag.update_layout(height=800, template='plotly_white', showlegend=False)
    
    st.plotly_chart(fig_diag, use_container_width=True)
        
    col_m8_1, col_m8_2 = st.columns([1, 1])
    
    with col_m8_1:
        st.markdown("### ✅ Was wir suchen")
        st.markdown("""
        **Plot 1 (Residuals vs Fitted):**
        - Zufällige Streuung um 0
        - Keine Muster (Kurven, Trichter)
        
        **Plot 2 (Q-Q):**
        - Punkte auf der Diagonale
        - Zeigt Normalverteilung der Residuen
        
        **Plot 3 (Scale-Location):**
        - Horizontales Band
        - Konstante Varianz (Homoskedastizität)
        
        **Plot 4 (Residuals vs Leverage):**
        - Keine Punkte ausserhalb Cook's Distance
        - Zeigt einflussreiche Beobachtungen
        """)
    
    with col_m8_2:
        st.markdown("### 📊 Statistische Tests")
        
        # Jarque-Bera Test (Normalität)
        from scipy.stats import jarque_bera
        jb_stat, jb_pval = jarque_bera(model_mult.resid)
        
        # Breusch-Pagan Test (Heteroskedastizität)
        from statsmodels.stats.diagnostic import het_breuschpagan
        bp_stat, bp_pval, _, _ = het_breuschpagan(model_mult.resid, X_mult)
        
        st.info(f"""
        **Jarque-Bera Test (Normalität):**
        - Statistik: {jb_stat:.3f}
        - p-Wert: {jb_pval:.4f}
        - {'✅ H₀ nicht verwerfen → Normalität OK' if jb_pval > 0.05 else '❌ H₀ verwerfen → Nicht normalverteilt'}
        
        **Breusch-Pagan Test (Homoskedastizität):**
        - Statistik: {bp_stat:.3f}
        - p-Wert: {bp_pval:.4f}
        - {'✅ H₀ nicht verwerfen → Homoskedastizität OK' if bp_pval > 0.05 else '❌ H₀ verwerfen → Heteroskedastizität'}
        """)
    
    # 3D Residual Diagnostics Toggle
    show_3d_resid_m8 = st.checkbox("🎲 3D-Residuen über Prädiktorraum", value=False)
    
    if show_3d_resid_m8:
        st.markdown("### 🎲 3D-Visualisierung: Residuen im Prädiktorraum")
        
        # Create 3D residual plot with plotly
        x1_range = np.linspace(x2_preis.min(), x2_preis.max(), 10)
        x2_range = np.linspace(x3_werbung.min(), x3_werbung.max(), 10)
        X1_mesh, X2_mesh = np.meshgrid(x1_range, x2_range)
        Z_zero = np.zeros_like(X1_mesh)
        
        fig_3d_resid_m8 = go.Figure()
        
        # Add zero plane
        fig_3d_resid_m8.add_trace(go.Surface(
            x=X1_mesh, y=X2_mesh, z=Z_zero,
            colorscale=[[0, 'gray'], [1, 'gray']],
            opacity=0.3,
            showscale=False,
            name='Zero Plane'
        ))
        
        # Add scatter plot with residuals colored
        fig_3d_resid_m8.add_trace(go.Scatter3d(
            x=x2_preis, y=x3_werbung, z=model_mult.resid,
            mode='markers',
            marker=dict(
                size=7,
                color=model_mult.resid,
                colorscale='RdBu_r',
                opacity=0.8,
                showscale=True,
                colorbar=dict(title='Residuengrösse'),
                line=dict(width=1, color='black')
            ),
            name='Residuals'
        ))
        
        fig_3d_resid_m8.update_layout(
            title='Residuen über Prädiktorraum<br>(Muster → Modellverletzungen)',
            scene=dict(
                xaxis_title=x1_name,
                yaxis_title=x2_name,
                zaxis_title='Residuen',
                camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2))
            ),
            template='plotly_white',
            height=600
        )
        
        st.plotly_chart(fig_3d_resid_m8, use_container_width=True)
                
        st.info("""
        **💡 3D-Interpretation:**
        
        - **Graue Ebene** = Null-Linie (perfekte Vorhersage)
        - **Rote Punkte** = Positive Residuen (Modell unterschätzt)
        - **Blaue Punkte** = Negative Residuen (Modell überschätzt)
        - **Zufällige Verteilung** = Gutes Modell ✅
        - **Systematische Muster** = Modellprobleme (Nonlinearität, fehlende Variablen) ❌
        - **Cluster in Bereichen** = Heteroskedastizität ⚠️
        """)

    # =========================================================
    # M9: ZUSAMMENFASSUNG
    # =========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">M9. Zusammenfassung: Multiple Regression</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Sie haben die **multiple Regression** von Grund auf verstanden! 🎉
    
    Fassen wir die wichtigsten Konzepte zusammen:
    """)
    
    # Vollständiger R-Output
    st.markdown("### 💻 Vollständiger Modell-Output")
    st.code(model_mult.summary().as_text(), language=None)
    
    col_m9_1, col_m9_2 = st.columns([1, 1])
    
    with col_m9_1:
        st.markdown("### 📋 Kernkonzepte")
        concepts_table = pd.DataFrame({
            'Konzept': [
                'Grundmodell',
                'OLS-Schätzer',
                'R²',
                'Adjustiertes R²',
                't-Test',
                'F-Test',
                'Partielle Koeffizienten',
                'Dummy-Variablen',
                'Multikollinearität',
                'VIF'
            ],
            'Status': ['✅'] * 10
        })
        st.dataframe(concepts_table, width='stretch', hide_index=True)
    
    with col_m9_2:
        st.markdown("### 📊 Unser Modell")
        st.success(f"""
        **Modellgleichung:**
        
        {y_name.split('(')[0].strip()} = {model_mult.params[0]:.2f} 
                 {model_mult.params[1]:+.2f} · {x1_name.split('(')[0].strip()} 
                 {model_mult.params[2]:+.2f} · {x2_name.split('(')[0].strip()}
        
        **Modellgüte:**
        - R² = {model_mult.rsquared:.4f}
        - R²_adj = {model_mult.rsquared_adj:.4f}
        - F = {model_mult.fvalue:.2f} (p < 0.001)
        
        **Interpretation:**
        - Pro Einheit {x1_name.split('(')[0].strip()}: {model_mult.params[1]:.2f} Einheiten {y_name.split('(')[0].strip()}
        - Pro Einheit {x2_name.split('(')[0].strip()}: {model_mult.params[2]:+.2f} Einheiten {y_name.split('(')[0].strip()}
        
        Beide Effekte sind **statistisch signifikant** (p < 0.05)!
        """)
    
    st.markdown("### 🎯 Wichtigste Erkenntnisse")
    st.markdown("""
    1. **Multiple Regression** erlaubt uns, den Einfluss **mehrerer Variablen gleichzeitig** zu untersuchen
    
    2. **Partielle Koeffizienten** messen den Effekt **ceteris paribus** (bei Konstanthaltung der anderen)
    
    3. **Adjustiertes R²** ist besser als R² für Modellvergleiche (bestraft Komplexität)
    
    4. **Multikollinearität** ist ein Problem - prüfen mit Korrelationen und VIF
    
    5. **Residuen-Diagnostik** ist essentiell - Annahmen müssen erfüllt sein!
    
    6. **Dummy-Variablen** ermöglichen kategoriale Prädiktoren
    
    7. **F-Test** prüft Gesamtsignifikanz, **t-Tests** prüfen einzelne Koeffizienten
    """)
    
    st.info("""
    **🚀 Nächste Schritte:**
    
    - Experimentieren Sie mit den Parametern
    - Vergleichen Sie einfache vs. multiple Regression
    - Prüfen Sie die Residuen-Diagnostik
    - Erkunden Sie Prognosen für verschiedene Szenarien
    """)

# Nur bei Einfacher Regression: Zeige die bestehenden Kapitel
elif regression_type == "📈 Einfache Regression":
    # =========================================================
    # KAPITEL 1: EINLEITUNG
    # =========================================================
    st.markdown('<p class="main-header">📖 Umfassender Leitfaden zur Linearen Regression</p>', unsafe_allow_html=True)
    st.markdown("### Von der Frage zur validierten Erkenntnis – Ein interaktiver Lernpfad")

    st.markdown("---")
    st.markdown('<p class="section-header">1.0 Einleitung: Die Analyse von Zusammenhängen</p>', unsafe_allow_html=True)

    col_intro1, col_intro2 = st.columns([2, 1])

    with col_intro1:
        st.markdown("""
        Von der Vorhersage von Unternehmensumsätzen bis hin zur Aufdeckung wissenschaftlicher 
        Zusammenhänge – die Fähigkeit, Beziehungen in Daten zu quantifizieren, ist eine 
        **Kernkompetenz** in der modernen Analyse.
    
        Die **Regressionsanalyse** ist das universelle Werkzeug für diese Aufgabe. Sie geht über 
        die blosse Feststellung *ob* Variablen zusammenhängen hinaus und erklärt präzise, 
        **wie** sie sich gegenseitig beeinflussen.
    
        > ⚠️ **Wichtig:** Die Regression allein beweist keine Kausalität! Sie quantifiziert die 
        > Stärke einer *potenziellen* Ursache-Wirkungs-Beziehung, die durch das Studiendesign 
        > gestützt werden muss.
        """)

    with col_intro2:
        st.info("""
        **Korrelation vs. Regression:**
    
        | Korrelation | Regression |
        |-------------|------------|
        | *Ungerichtet* | *Gerichtet* |
        | Wie stark? | Um wieviel? |
        | r ∈ [-1, 1] | ŷ = b₀ + b₁x |
        """)

    # =========================================================
    # KAPITEL 1.5: MEHRDIMENSIONALE VERTEILUNGEN (NEU)
    # =========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">1.5 Mehrdimensionale Verteilungen: Das Fundament für Zusammenhänge</p>', unsafe_allow_html=True)

    st.markdown("""
    Bevor wir Zusammenhänge zwischen Variablen analysieren können, müssen wir verstehen, 
    wie **zwei Zufallsvariablen gemeinsam** verteilt sein können. Dies ist die mathematische 
    Grundlage für alles, was folgt.
    """)

    # --- Interaktive Parameter für Mehrdimensionale Verteilungen ---
    st.markdown('<p class="subsection-header">🎲 Gemeinsame Verteilung f(X,Y)</p>', unsafe_allow_html=True)

    col_joint1, col_joint2 = st.columns([2, 1])

    with col_joint1:
        # Slider für Korrelation in der bivariaten Normalverteilung
        demo_corr = st.slider("Korrelation ρ zwischen X und Y", min_value=-0.95, max_value=0.95, value=0.7, step=0.05,
                              help="Bewege den Slider um zu sehen, wie sich die gemeinsame Verteilung verändert",
                              key="demo_corr_slider")
    
        # Toggle für 3D-Ansicht
        show_3d_joint = st.toggle("🎲 3D-Ansicht aktivieren", value=False, key="toggle_3d_joint")
    
        # Bivariate Normalverteilung generieren
        from scipy.stats import multivariate_normal
    
        mean = [0, 0]
        cov_matrix = [[1, demo_corr], [demo_corr, 1]]
    
        x_grid = np.linspace(-3, 3, 100)
        y_grid = np.linspace(-3, 3, 100)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        pos = np.dstack((X_grid, Y_grid))
    
        rv = multivariate_normal(mean, cov_matrix)
        Z = rv.pdf(pos)
    
        if show_3d_joint:
            # === 3D VERSION ===
            fig_joint_3d = make_subplots(
                rows=1, cols=3,
                specs=[[{'type': 'surface'}, {'type': 'scatter3d'}, {'type': 'surface'}]],
                subplot_titles=(
                    f'Gemeinsame Verteilung<br>ρ = {demo_corr:.2f}',
                    'Randverteilung f_X(x)',
                    f'Bedingte Verteilung<br>E(Y|X=1) = {demo_corr * 1.0:.2f}'
                )
            )
        
            # 1. 3D Surface Plot der gemeinsamen Verteilung
            fig_joint_3d.add_trace(
                go.Surface(x=X_grid, y=Y_grid, z=Z, colorscale='Blues', opacity=0.8, showscale=False),
                row=1, col=1
            )
            
            # Stichprobe als Punkte auf z=0
            np.random.seed(42)
            sample = np.random.multivariate_normal(mean, cov_matrix, 100)
            fig_joint_3d.add_trace(
                go.Scatter3d(x=sample[:, 0], y=sample[:, 1], z=np.zeros(100),
                           mode='markers', marker=dict(size=2, color='red', opacity=0.3),
                           showlegend=False),
                row=1, col=1
            )
        
            # 2. Randverteilung als 3D
            x_marg = np.linspace(-3, 3, 100)
            y_marg_pdf = stats.norm.pdf(x_marg, 0, 1)
            
            # Main curve at z=0
            fig_joint_3d.add_trace(
                go.Scatter3d(x=x_marg, y=y_marg_pdf, z=np.zeros_like(x_marg),
                           mode='lines', line=dict(color='blue', width=4),
                           showlegend=False),
                row=1, col=2
            )
            
            # Vertical lines
            for i in range(0, len(x_marg), 5):
                fig_joint_3d.add_trace(
                    go.Scatter3d(x=[x_marg[i], x_marg[i]], y=[0, 0], z=[0, y_marg_pdf[i]],
                               mode='lines', line=dict(color='blue', width=1),
                               opacity=0.3, showlegend=False),
                    row=1, col=2
                )
        
            # 3. Bedingte Verteilung als 3D-Schnitt
            x_cond = 1.0
            cond_mean = demo_corr * x_cond
            cond_var = max(1 - demo_corr**2, 0.01)
            cond_std = np.sqrt(cond_var)
        
            y_cond_grid = np.linspace(-3, 3, 100)
            pdf_cond = stats.norm.pdf(y_cond_grid, cond_mean, cond_std)
        
            # Bedingte Verteilung Kurve
            fig_joint_3d.add_trace(
                go.Scatter3d(x=np.full_like(y_cond_grid, x_cond), y=y_cond_grid, z=pdf_cond,
                           mode='lines', line=dict(color='green', width=4),
                           showlegend=False),
                row=1, col=3
            )
            
            # Vertical lines
            for i in range(0, len(y_cond_grid), 5):
                fig_joint_3d.add_trace(
                    go.Scatter3d(x=[x_cond, x_cond], y=[y_cond_grid[i], y_cond_grid[i]], z=[0, pdf_cond[i]],
                               mode='lines', line=dict(color='green', width=1),
                               opacity=0.3, showlegend=False),
                    row=1, col=3
                )
            
            # Fläche füllen
            fig_joint_3d.add_trace(
                go.Surface(x=X_grid, y=Y_grid, z=Z, colorscale='Blues', opacity=0.3, showscale=False),
                row=1, col=3
            )
        
            # Update layout
            fig_joint_3d.update_layout(
                height=500,
                showlegend=False,
                scene1=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='f(X,Y)',
                           camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2))),
                scene2=dict(xaxis_title='X', yaxis_title='', zaxis_title='f_X(x)',
                           camera=dict(eye=dict(x=1.5, y=-1.8, z=1.0))),
                scene3=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='f(Y|X=1)',
                           camera=dict(eye=dict(x=1.5, y=-1.8, z=1.0)))
            )
        
            st.plotly_chart(fig_joint_3d, use_container_width=True)
                    
        else:
            # === 2D VERSION (Original) ===
            fig_joint = make_subplots(
                rows=1, cols=3,
                subplot_titles=(
                    f'Gemeinsame Verteilung f(X,Y)<br>ρ = {demo_corr:.2f}',
                    'Randverteilung f_X(x)<br>(Marginale von X)',
                    f'Bedingte Verteilung f(Y|X=1)<br>σ² = {(1 - demo_corr**2):.2f}'
                )
            )
        
            # 1. Contour Plot der gemeinsamen Verteilung
            fig_joint.add_trace(
                go.Contour(x=x_grid, y=y_grid, z=Z, colorscale='Blues', showscale=False),
                row=1, col=1
            )
            
            # Grid lines
            fig_joint.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
            fig_joint.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
        
            # Stichprobe einzeichnen
            np.random.seed(42)
            sample = np.random.multivariate_normal(mean, cov_matrix, 100)
            fig_joint.add_trace(
                go.Scatter(x=sample[:, 0], y=sample[:, 1], mode='markers',
                          marker=dict(size=4, color='red', opacity=0.3),
                          name='Stichprobe', showlegend=False),
                row=1, col=1
            )
        
            # 2. Randverteilung f_X (oben projiziert)
            x_pdf = stats.norm.pdf(x_grid, 0, 1)
            fig_joint.add_trace(
                go.Scatter(x=x_grid, y=x_pdf, fill='tozeroy', fillcolor='rgba(0,0,255,0.3)',
                          line=dict(color='blue', width=2), showlegend=False),
                row=1, col=2
            )
            fig_joint.add_vline(x=0, line_dash="dash", line_color="orange", annotation_text="E(X) = 0",
                              row=1, col=2)
        
            # 3. Bedingte Verteilung f(Y|X=1)
            x_cond = 1.0
            cond_mean = demo_corr * x_cond
            cond_var = 1 - demo_corr**2
            cond_std = np.sqrt(max(cond_var, 0.01))
        
            y_cond_grid = np.linspace(-3, 3, 100)
            pdf_cond = stats.norm.pdf(y_cond_grid, cond_mean, cond_std)
        
            fig_joint.add_trace(
                go.Scatter(x=y_cond_grid, y=pdf_cond, fill='tozeroy', fillcolor='rgba(0,255,0,0.3)',
                          line=dict(color='green', width=2), showlegend=False),
                row=1, col=3
            )
            fig_joint.add_vline(x=cond_mean, line_dash="dash", line_color="red", line_width=2,
                              annotation_text=f'E(Y|X={x_cond}) = {cond_mean:.2f}', row=1, col=3)
        
            # Update layout
            fig_joint.update_xaxes(title_text="X", row=1, col=1, showgrid=True, gridcolor='lightgray')
            fig_joint.update_yaxes(title_text="Y", row=1, col=1, showgrid=True, gridcolor='lightgray')
            fig_joint.update_xaxes(title_text="X", row=1, col=2, showgrid=True, gridcolor='lightgray')
            fig_joint.update_yaxes(title_text="f_X(x)", row=1, col=2, showgrid=True, gridcolor='lightgray')
            fig_joint.update_xaxes(title_text="Y", row=1, col=3, showgrid=True, gridcolor='lightgray')
            fig_joint.update_yaxes(title_text="f(Y|X=1)", row=1, col=3, showgrid=True, gridcolor='lightgray')
            
            fig_joint.update_layout(height=400, showlegend=False)
        
            st.plotly_chart(fig_joint, use_container_width=True)
            
    with col_joint2:
        if show_formulas:
            st.markdown("### Gemeinsame Verteilung")
            st.latex(r"f_{X,Y}(x,y) = P(X=x, Y=y)")
        
            st.markdown("### Randverteilung")
            st.latex(r"f_X(x) = \sum_y f_{X,Y}(x,y)")
        
            st.markdown("### Bedingte Verteilung")
            st.latex(r"f_{Y|X}(y|x) = \frac{f_{X,Y}(x,y)}{f_X(x)}")
    
        st.info(f"""
        **Beobachte:**
    
        Bei ρ = {demo_corr:.2f}:
        - Die Punktewolke ist {"stark" if abs(demo_corr) > 0.7 else "schwach"} {"positiv" if demo_corr > 0 else "negativ" if demo_corr < 0 else "un"}korreliert
        - E(Y|X=1) = {demo_corr:.2f} (nicht 0!)
        - Die bedingte Varianz ist {max(1 - demo_corr**2, 0.01):.2f} < 1
    
        **→ Je höher |ρ|, desto mehr "wissen" wir über Y, wenn wir X kennen!**
        """)

    # Stochastische Unabhängigkeit
    st.markdown('<p class="subsection-header">🔗 Stochastische Unabhängigkeit</p>', unsafe_allow_html=True)

    col_indep1, col_indep2 = st.columns([1, 1])

    with col_indep1:
        st.markdown("""
        Zwei Zufallsvariablen X und Y sind **stochastisch unabhängig**, wenn:
        """)
        st.latex(r"f_{X,Y}(x,y) = f_X(x) \cdot f_Y(y)")
        st.markdown("""
        Das bedeutet: Die gemeinsame Wahrscheinlichkeit ist einfach das **Produkt** der Einzelwahrscheinlichkeiten.
    
        **Konsequenz:** Bei Unabhängigkeit gilt:
        - $E(Y|X=x) = E(Y)$ – X sagt nichts über Y aus!
        - $Cov(X,Y) = 0$
        - $ρ = 0$
        """)

    with col_indep2:
        # Create 2-panel independence visualization with plotly
        from plotly.subplots import make_subplots
        
        np.random.seed(123)
        # Unabhängig (ρ=0)
        x_ind = np.random.normal(0, 1, 200)
        y_ind = np.random.normal(0, 1, 200)
        
        # Abhängig (ρ=0.8)
        cov_dep = [[1, 0.8], [0.8, 1]]
        sample_dep = np.random.multivariate_normal([0, 0], cov_dep, 200)
        
        fig_indep = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Unabhängig (ρ = 0)<br>"Keine Struktur"',
                           'Abhängig (ρ = 0.8)<br>"Klare Struktur"')
        )
        
        # Left panel: Independent
        fig_indep.add_trace(
            go.Scatter(x=x_ind, y=y_ind, mode='markers',
                      marker=dict(size=5, color='gray', opacity=0.5),
                      showlegend=False),
            row=1, col=1
        )
        
        # Right panel: Dependent
        fig_indep.add_trace(
            go.Scatter(x=sample_dep[:, 0], y=sample_dep[:, 1], mode='markers',
                      marker=dict(size=5, color='blue', opacity=0.5),
                      showlegend=False),
            row=1, col=2
        )
        
        fig_indep.update_xaxes(title_text="X", row=1, col=1)
        fig_indep.update_yaxes(title_text="Y", row=1, col=1)
        fig_indep.update_xaxes(title_text="X", row=1, col=2)
        fig_indep.update_yaxes(title_text="Y", row=1, col=2)
        
        fig_indep.update_layout(height=400, template='plotly_white')
    
        st.plotly_chart(fig_indep, use_container_width=True)
        
    st.success("""
    **Merke:** Die Regression nutzt genau diese Struktur! Wenn X und Y abhängig sind, 
    können wir $E(Y|X=x)$ als Funktion von x modellieren – das ist die Regressionsgerade!
    """)

    # =========================================================
    # KAPITEL 2: DAS FUNDAMENT
    # =========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">2.0 Das Fundament: Das einfache lineare Regressionsmodell</p>', unsafe_allow_html=True)

    st.markdown("""
    Das Verständnis des einfachen linearen Regressionsmodells ist der entscheidende erste Schritt. 
    Die Rollen der Variablen werden klar definiert:

    - **Abhängige Variable (Y):** Die Zielvariable – was wir erklären/vorhersagen wollen
    - **Unabhängige Variable (X):** Der Prädiktor – was die Veränderung erklärt
    """)

    col_model1, col_model2 = st.columns([1.2, 1])

    with col_model1:
        st.markdown("### Das grundlegende Modell:")
        if show_formulas:
            st.latex(r"y_i = \beta_0 + \beta_1 \cdot x_i + \varepsilon_i")
    
        st.markdown("""
        | Symbol | Bedeutung |
        |--------|-----------|
        | **β₀** | Wahrer Achsenabschnitt (unbekannt) |
        | **β₁** | Wahre Steigung (unbekannt) – Änderung in Y pro Einheit X |
        | **εᵢ** | Zufällige Störgrösse – alle anderen Einflüsse |
        """)

    with col_model2:
        st.warning(f"""
        ### 🎯 Praxisbeispiel: {context_title}
    
        {context_description}
        """)

    # Erste Visualisierung: Die Rohdaten
    st.markdown(f'<p class="subsection-header">📊 Unsere Daten: {n} Beobachtungen</p>', unsafe_allow_html=True)

    col_data1, col_data2 = st.columns([1, 2])

    with col_data1:
        st.dataframe(df.style.format({x_label: '{:.2f}', y_label: '{:.2f}'}), 
                     height=min(400, n * 35 + 50), width='stretch')

    with col_data2:
        # Create scatter plot with plotly
        fig_scatter1 = go.Figure()
        
        fig_scatter1.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            marker=dict(size=12, color='#1f77b4', opacity=0.7,
                       line=dict(width=2, color='white')),
            name='Datenpunkte'
        ))
        
        # Mean lines
        fig_scatter1.add_hline(y=y_mean_val, line_dash='dash', line_color='orange',
                              opacity=0.5, annotation_text=f'ȳ = {y_mean_val:.2f}',
                              annotation_position="right")
        fig_scatter1.add_vline(x=x_mean, line_dash='dash', line_color='green',
                              opacity=0.5, annotation_text=f'x̄ = {x_mean:.2f}',
                              annotation_position="top")
        
        # Center point
        fig_scatter1.add_trace(go.Scatter(
            x=[x_mean], y=[y_mean_val],
            mode='markers',
            marker=dict(size=18, color='red', symbol='x'),
            name='Schwerpunkt (x̄, ȳ)'
        ))
        
        fig_scatter1.update_layout(
            title='Schritt 1: Visualisierung der Rohdaten<br>"Gibt es einen Zusammenhang?"',
            xaxis_title=x_label,
            yaxis_title=y_label,
            template='plotly_white',
            hovermode='closest'
        )
    
        st.plotly_chart(fig_scatter1, use_container_width=True)
        
    st.success(f"""
    **Beobachtung:** Die Punkte scheinen einem aufsteigenden Trend zu folgen! 
    Der Schwerpunkt liegt bei ({x_mean:.2f}, {y_mean_val:.2f}).
    Jetzt müssen wir die "beste" Gerade finden, die diesen Trend beschreibt.
    """)

    # =========================================================
    # KAPITEL 2.5: KOVARIANZ & KORRELATION (NEU)
    # =========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">2.5 Kovarianz & Korrelation: Die Bausteine der Regression</p>', unsafe_allow_html=True)

    st.markdown("""
    Bevor wir die Regressionsgerade berechnen, müssen wir verstehen, **wie** wir den Zusammenhang 
    zwischen X und Y messen. Die **Kovarianz** und der **Korrelationskoeffizient** sind die Werkzeuge dafür.
    """)

    # --- KOVARIANZ ---
    st.markdown('<p class="subsection-header">📐 Die Kovarianz: Richtung und Stärke des Zusammenhangs</p>', unsafe_allow_html=True)

    col_cov1, col_cov2 = st.columns([2, 1])

    with col_cov1:
        show_3d_cov = st.toggle("📊 3D-Ansicht aktivieren (Kovarianz)", value=False, key="toggle_3d_cov")
    
        if show_3d_cov:
            # 3D Visualisierung: Vertikale Säulen
            fig_cov = go.Figure()
        
            x_mean_val = x.mean()
            y_mean_val_local = y.mean()
        
            # Daten-Säulen als 3D bars (using mesh3d for bars)
            for i in range(len(x)):
                dx = x[i] - x_mean_val
                dy = y[i] - y_mean_val_local
                product = dx * dy
                color = 'green' if product > 0 else 'red'
            
                # Create bar as mesh3d (simplified box)
                bar_width = 0.3
                if product > 0:
                    z_base = 0
                    z_height = product
                else:
                    z_base = product
                    z_height = abs(product)
                
                # Simple vertical line representation
                fig_cov.add_trace(go.Scatter3d(
                    x=[x[i], x[i]], y=[y[i], y[i]], z=[0, product],
                    mode='lines',
                    line=dict(color=color, width=8),
                    opacity=0.7,
                    showlegend=False
                ))
            
                # Datenpunkt oben
                fig_cov.add_trace(go.Scatter3d(
                    x=[x[i]], y=[y[i]], z=[product],
                    mode='markers',
                    marker=dict(size=8, color=color, line=dict(color='white', width=1)),
                    showlegend=False
                ))
        
            # Schwerpunkt
            fig_cov.add_trace(go.Scatter3d(
                x=[x_mean_val], y=[y_mean_val_local], z=[0],
                mode='markers',
                marker=dict(size=12, color='black', symbol='x', line=dict(color='white', width=2)),
                name='Schwerpunkt',
                showlegend=True
            ))
        
            fig_cov.update_layout(
                title='3D Kovarianz-Visualisierung: Säulenhöhe = Produkt der Abweichungen',
                scene=dict(
                    xaxis_title=f'{x_label} (X)',
                    yaxis_title=f'{y_label} (Y)',
                    zaxis_title='(X - X̄)(Y - Ȳ)',
                    camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2))
                ),
                height=600
            )
        
            st.plotly_chart(fig_cov, use_container_width=True)
        else:
            # 2D Original: Quadranten mit Rechtecken
            st.markdown("""
            Die Kovarianz misst, ob X und Y **gemeinsam** von ihren Mittelwerten abweichen:
            - Wenn X über dem Mittelwert ist UND Y auch → **positiver Beitrag**
            - Wenn X über dem Mittelwert ist ABER Y darunter → **negativer Beitrag**
            """)
        
            fig_cov = go.Figure()
        
            # Quadranten einfärben
            x_mean_val = x.mean()
            y_mean_val_local = y.mean()
            
            x_min, x_max = min(x) - 0.5, max(x) + 0.5
            y_min, y_max = min(y) - 0.5, max(y) + 0.5
        
            # Quadrant I (top right, positive, green)
            fig_cov.add_shape(type="rect", x0=x_mean_val, y0=y_mean_val_local, x1=x_max, y1=y_max,
                            fillcolor="green", opacity=0.1, line_width=0, layer="below")
            
            # Quadrant II (top left, negative, red)
            fig_cov.add_shape(type="rect", x0=x_min, y0=y_mean_val_local, x1=x_mean_val, y1=y_max,
                            fillcolor="red", opacity=0.1, line_width=0, layer="below")
            
            # Quadrant III (bottom left, positive, green)
            fig_cov.add_shape(type="rect", x0=x_min, y0=y_min, x1=x_mean_val, y1=y_mean_val_local,
                            fillcolor="green", opacity=0.1, line_width=0, layer="below")
            
            # Quadrant IV (bottom right, negative, red)
            fig_cov.add_shape(type="rect", x0=x_mean_val, y0=y_min, x1=x_max, y1=y_mean_val_local,
                            fillcolor="red", opacity=0.1, line_width=0, layer="below")
        
            # Mean lines
            fig_cov.add_hline(y=y_mean_val_local, line_dash="dash", line_color="orange", line_width=2,
                            annotation_text=f'ȳ = {y_mean_val_local:.2f}', annotation_position="right")
            fig_cov.add_vline(x=x_mean_val, line_dash="dash", line_color="green", line_width=2,
                            annotation_text=f'x̄ = {x_mean_val:.2f}', annotation_position="top")
        
            # Datenpunkte mit Rechtecken für (xi - x̄)(yi - ȳ)
            for i in range(len(x)):
                dx = x[i] - x_mean_val
                dy = y[i] - y_mean_val_local
                product = dx * dy
                color = 'green' if product > 0 else 'red'
            
                # Rechteck für das Produkt
                fig_cov.add_shape(type="rect", x0=x_mean_val, y0=y_mean_val_local,
                                x1=x[i], y1=y[i],
                                fillcolor=color, opacity=0.15, line=dict(color=color, width=1),
                                layer="below")
            
            # Datenpunkte on top
            for i in range(len(x)):
                dx = x[i] - x_mean_val
                dy = y[i] - y_mean_val_local
                product = dx * dy
                color = 'green' if product > 0 else 'red'
                fig_cov.add_trace(go.Scatter(
                    x=[x[i]], y=[y[i]], mode='markers',
                    marker=dict(size=10, color=color, line=dict(color='white', width=2)),
                    showlegend=False
                ))
        
            # Schwerpunkt
            fig_cov.add_trace(go.Scatter(
                x=[x_mean_val], y=[y_mean_val_local], mode='markers',
                marker=dict(size=15, color='black', symbol='x', line=dict(color='white', width=2)),
                name='Schwerpunkt'
            ))
        
            fig_cov.update_layout(
                title='Kovarianz-Visualisierung: Grüne Rechtecke addieren, rote subtrahieren',
                xaxis_title=f'{x_label} (X)',
                yaxis_title=f'{y_label} (Y)',
                showlegend=True,
                xaxis=dict(range=[x_min, x_max], showgrid=True, gridcolor='lightgray'),
                yaxis=dict(range=[y_min, y_max], showgrid=True, gridcolor='lightgray'),
                height=600
            )
        
            st.plotly_chart(fig_cov, use_container_width=True)
            
    with col_cov2:
        if show_formulas:
            st.markdown("### Kovarianz (Population)")
            st.latex(r"Cov(X,Y) = E(XY) - E(X) \cdot E(Y)")
        
            st.markdown("### Kovarianz (Stichprobe)")
            st.latex(r"s_{xy} = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{n-1}")
    
        # Berechnung zeigen
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

    # --- KORRELATION ---
    st.markdown('<p class="subsection-header">📊 Der Korrelationskoeffizient: Standardisierte Stärke</p>', unsafe_allow_html=True)

    st.markdown("""
    Die Kovarianz hat ein Problem: Sie hängt von den **Einheiten** ab! Eine Kovarianz von 5.2 
    zwischen Fläche (qm) und Umsatz (Mio. €) ist schwer zu interpretieren.

    Die Lösung: **Normierung** durch die Standardabweichungen → Der Korrelationskoeffizient r ∈ [-1, +1]
    """)

    col_corr1, col_corr2 = st.columns([2, 1])

    with col_corr1:
        # Verschiedene Korrelationen zeigen mit plotly subplots
        from plotly.subplots import make_subplots
        
        example_corrs = [-0.95, -0.5, 0, 0.5, 0.8, 0.95]
        np.random.seed(42)
        
        fig_corr_examples = make_subplots(
            rows=2, cols=3,
            subplot_titles=[f'r = {r:.2f}' for r in example_corrs]
        )
    
        for idx, r in enumerate(example_corrs):
            row = idx // 3 + 1
            col = idx % 3 + 1
        
            # Daten generieren
            if r == 0:
                ex_x = np.random.normal(0, 1, 100)
                ex_y = np.random.normal(0, 1, 100)
            else:
                cov_ex = [[1, r], [r, 1]]
                sample_ex = np.random.multivariate_normal([0, 0], cov_ex, 100)
                ex_x, ex_y = sample_ex[:, 0], sample_ex[:, 1]
        
            # Farbe basierend auf r
            if r > 0:
                color = f'rgba(0, {int(128 + abs(r)*127)}, 0, 0.6)'
            elif r < 0:
                color = f'rgba({int(128 + abs(r)*127)}, 0, 0, 0.6)'
            else:
                color = 'rgba(128, 128, 128, 0.6)'
        
            fig_corr_examples.add_trace(
                go.Scatter(x=ex_x, y=ex_y, mode='markers',
                          marker=dict(size=5, color=color),
                          showlegend=False),
                row=row, col=col
            )
            
            # Regressionslinie wenn r ≠ 0
            if r != 0:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        z = np.polyfit(ex_x, ex_y, 1)
                        p = np.poly1d(z)
                        fig_corr_examples.add_trace(
                            go.Scatter(x=[-3, 3], y=[p(-3), p(3)],
                                     mode='lines',
                                     line=dict(color='black', dash='dash', width=1),
                                     showlegend=False),
                            row=row, col=col
                        )
                except (np.linalg.LinAlgError, ValueError):
                    pass
                
            fig_corr_examples.update_xaxes(range=[-3, 3], row=row, col=col)
            fig_corr_examples.update_yaxes(range=[-3, 3], row=row, col=col)
    
        fig_corr_examples.update_layout(
            title_text='Der Korrelationskoeffizient r: Von -1 (perfekt negativ) bis +1 (perfekt positiv)',
            height=600,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_corr_examples, use_container_width=True)
        
    with col_corr2:
        if show_formulas:
            st.markdown("### Korrelation (Pearson)")
            st.latex(r"\rho_{X,Y} = \frac{Cov(X,Y)}{\sigma_X \cdot \sigma_Y}")
        
            st.markdown("### Stichproben-Korrelation")
            st.latex(r"r = \frac{s_{xy}}{s_x \cdot s_y}")
    
        st.metric("Unsere Korrelation r", f"{corr_xy:.4f}")
    
        # Interpretation
        if abs(corr_xy) > 0.8:
            strength = "sehr stark"
        elif abs(corr_xy) > 0.5:
            strength = "mittelstark"
        elif abs(corr_xy) > 0.3:
            strength = "schwach"
        else:
            strength = "sehr schwach"
    
        direction = "positiv" if corr_xy > 0 else "negativ"
    
        st.info(f"""
        **Interpretation:**
    
        r = {corr_xy:.3f} zeigt einen **{strength}en {direction}en** 
        linearen Zusammenhang.
    
        **Wichtig:** Bei einfacher Regression gilt:
    
        **R² = r²** = {corr_xy**2:.4f}
    
        Das ist identisch mit unserem späteren Bestimmtheitsmass!
        """)

    # --- t-Test für Korrelation ---
    st.markdown('<p class="subsection-header">🔬 Signifikanztest für die Korrelation</p>', unsafe_allow_html=True)

    col_ttest_corr1, col_ttest_corr2 = st.columns([2, 1])

    with col_ttest_corr1:
        # t-Statistik für Korrelation
        t_corr = abs(corr_xy) * np.sqrt((n - 2) / max(1 - corr_xy**2, 0.001))
        p_corr = 2 * (1 - stats.t.cdf(t_corr, df=n-2))
    
        # Create t-distribution plot with plotly
        x_t = np.linspace(-5, max(5, t_corr + 1), 300)
        y_t = stats.t.pdf(x_t, df=n-2)
        
        fig_t_corr = go.Figure()
        
        # Main distribution curve
        fig_t_corr.add_trace(go.Scatter(
            x=x_t, y=y_t,
            mode='lines',
            line=dict(color='black', width=2),
            name=f't-Verteilung (df={n-2})'
        ))
        
        # Shaded p-value regions
        mask = abs(x_t) > t_corr
        fig_t_corr.add_trace(go.Scatter(
            x=x_t[mask], y=y_t[mask],
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.3)',
            line=dict(width=0),
            name=f'p-Wert = {p_corr:.4f}',
            showlegend=True
        ))
        
        # Critical values
        t_crit = stats.t.ppf(0.975, df=n-2)
        fig_t_corr.add_vline(x=t_crit, line_dash='dash', line_color='orange', opacity=0.7)
        fig_t_corr.add_vline(x=-t_crit, line_dash='dash', line_color='orange', opacity=0.7,
                            annotation_text=f'Kritisch: ±{t_crit:.2f}')
        
        # Observed t-values
        fig_t_corr.add_vline(x=t_corr, line_color='blue', line_width=3,
                            annotation_text=f't = {t_corr:.2f}')
        fig_t_corr.add_vline(x=-t_corr, line_color='blue', line_width=2, opacity=0.5)
        
        fig_t_corr.update_layout(
            title=f'H₀: ρ = 0 (kein Zusammenhang) vs. H₁: ρ ≠ 0',
            xaxis_title='t-Wert',
            yaxis_title='Dichte',
            template='plotly_white',
            hovermode='x'
        )
    
        st.plotly_chart(fig_t_corr, use_container_width=True)
        
    with col_ttest_corr2:
        if show_formulas:
            st.markdown("### Teststatistik")
            st.latex(r"t = |r| \cdot \sqrt{\frac{n-2}{1-r^2}}")
            st.latex(f"t = |{corr_xy:.4f}| \\cdot \\sqrt{{\\frac{{{n}-2}}{{1-{corr_xy:.4f}^2}}}} = {t_corr:.2f}")
    
        st.metric("t-Wert", f"{t_corr:.3f}")
        st.metric("p-Wert", f"{p_corr:.4f}")
        st.metric("Signifikanz", get_signif_stars(p_corr))
    
        if p_corr < 0.05:
            st.success("✅ Der Zusammenhang ist **signifikant**!")
        else:
            st.warning("⚠️ Der Zusammenhang ist **nicht signifikant**.")

    # --- Spearman Rangkorrelation ---
    with st.expander("📊 Bonus: Spearman Rangkorrelation (für nicht-lineare Zusammenhänge)"):
        col_sp1, col_sp2 = st.columns([2, 1])
    
        with col_sp1:
            # Ränge berechnen
            from scipy.stats import spearmanr
            rho_spearman, p_spearman = spearmanr(x, y)
        
            # Create 2-panel plot with plotly subplots
            from plotly.subplots import make_subplots
            
            fig_spear = make_subplots(
                rows=1, cols=2,
                subplot_titles=(f'Original-Daten<br>Pearson r = {corr_xy:.3f}',
                               f'Rang-Daten<br>Spearman ρ = {rho_spearman:.3f}')
            )
            
            # Original data
            fig_spear.add_trace(
                go.Scatter(x=x, y=y, mode='markers',
                          marker=dict(size=8, color='blue', opacity=0.7),
                          showlegend=False),
                row=1, col=1
            )
            
            # Rank data
            rank_x = stats.rankdata(x)
            rank_y = stats.rankdata(y)
            fig_spear.add_trace(
                go.Scatter(x=rank_x, y=rank_y, mode='markers',
                          marker=dict(size=8, color='green', opacity=0.7),
                          showlegend=False),
                row=1, col=2
            )
            
            fig_spear.update_xaxes(title_text="X", row=1, col=1)
            fig_spear.update_yaxes(title_text="Y", row=1, col=1)
            fig_spear.update_xaxes(title_text="Rang(X)", row=1, col=2)
            fig_spear.update_yaxes(title_text="Rang(Y)", row=1, col=2)
            
            fig_spear.update_layout(height=400, template='plotly_white')
        
            st.plotly_chart(fig_spear, use_container_width=True)
                
        with col_sp2:
            st.latex(r"r_s = 1 - \frac{6 \sum d_i^2}{n(n^2-1)}")
            st.markdown("wobei $d_i$ = Differenz der Ränge")
        
            st.metric("Spearman ρ", f"{rho_spearman:.4f}")
            st.metric("p-Wert", f"{p_spearman:.4f}")
        
            st.info("""
            **Wann Spearman?**
            - Ordinale Daten
            - Nicht-lineare monotone Zusammenhänge
            - Ausreisser-robust
            """)

    st.success(f"""
    **Zusammenfassung Kapitel 2.5:**

    Wir haben die Bausteine für die Regression verstanden:
    - **Kovarianz** Cov(X,Y) = {cov_xy:.4f} → Richtung des Zusammenhangs
    - **Korrelation** r = {corr_xy:.4f} → Standardisierte Stärke

    Im nächsten Kapitel sehen wir: **b₁ = Cov(X,Y) / Var(X)** – die Steigung ist direkt aus der Kovarianz abgeleitet!
    """)

    # =========================================================
    # KAPITEL 3: DIE METHODE (OLS)
    # =========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">3.0 Die Methode: Schätzung mittels OLS</p>', unsafe_allow_html=True)

    st.markdown("""
    Die **Methode der kleinsten Quadrate (Ordinary Least Squares, OLS)** findet die optimale Gerade.

    **Das Kernprinzip:** Wähle jene Gerade, welche die **Summe der quadrierten vertikalen Abweichungen** 
    (Residuen) zwischen Datenpunkten und Gerade **minimiert**.
    """)

    # OLS Visualisierung
    col_ols1, col_ols2 = st.columns([2, 1])

    with col_ols1:
        # Create OLS plot with plotly
        fig_ols = go.Figure()
        
        # Data points
        fig_ols.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            marker=dict(size=10, color='#1f77b4', opacity=0.7,
                       line=dict(width=2, color='white')),
            name='Datenpunkte'
        ))
        
        # OLS regression line
        fig_ols.add_trace(go.Scatter(
            x=x, y=y_pred,
            mode='lines',
            line=dict(color='red', width=3),
            name=f'OLS-Gerade: ŷ = {b0:.3f} + {b1:.3f}x'
        ))
        
        # True line if shown
        if show_true_line:
            fig_ols.add_trace(go.Scatter(
                x=x, y=true_intercept + true_beta * x,
                mode='lines',
                line=dict(color='green', width=2, dash='dash'),
                opacity=0.7,
                name=f'Wahre Gerade: y = {true_intercept:.2f} + {true_beta:.2f}x'
            ))
        
        # Residual lines (for first 10 points)
        for i in range(min(len(x), 10)):
            resid = y[i] - y_pred[i]
            fig_ols.add_trace(go.Scatter(
                x=[x[i], x[i]],
                y=[y[i], y_pred[i]],
                mode='lines',
                line=dict(color='red', width=1.5),
                opacity=0.5,
                showlegend=False
            ))
            
            # Add rectangles for squared residuals (as shapes)
            if abs(resid) > 0.05:
                size = min(abs(resid), 1.5)
                fig_ols.add_shape(
                    type='rect',
                    x0=x[i], x1=x[i] + size,
                    y0=min(y[i], y_pred[i]), y1=min(y[i], y_pred[i]) + abs(resid),
                    fillcolor='red',
                    opacity=0.2,
                    line=dict(color='red', width=1)
                )
        
        fig_ols.update_layout(
            title='OLS minimiert die Fläche aller roten Quadrate (= SSE)',
            xaxis_title=x_label,
            yaxis_title=y_label,
            template='plotly_white',
            hovermode='closest'
        )
    
        st.plotly_chart(fig_ols, use_container_width=True)
        
    with col_ols2:
        if show_formulas:
            st.markdown("### OLS-Schätzer:")
            st.latex(r"b_1 = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2}")
            st.latex(r"b_0 = \bar{y} - b_1 \cdot \bar{x}")
        
            st.markdown("### Mit unseren Werten:")
            st.latex(f"b_1 = \\frac{{{cov_xy*(n-1):.2f}}}{{{var_x*(n-1):.2f}}} = {b1:.4f}")
            st.latex(f"b_0 = {y_mean_val:.2f} - {b1:.4f} \\times {x_mean:.2f} = {b0:.4f}")
    
        st.success(f"""
        ### 📐 Ergebnis:
    
        **ŷ = {b0:.4f} + {b1:.4f} · x**
        """)

    # =========================================================
    # KAPITEL 3.1: DAS REGRESSIONSMODELL IM DETAIL
    # =========================================================
    st.markdown('<p class="subsection-header">3.1 Das Regressionsmodell im Detail: Anatomie & Unsicherheit</p>', unsafe_allow_html=True)

    st.markdown("""
    Dieses Dashboard zeigt die **komplette Anatomie** des Regressionsmodells:
    - **Steigungsdreieck**: Visualisiert b₁ = Δy/Δx
    - **Fehlerterm εᵢ**: Die Abweichung zwischen Beobachtung und Modell
    - **Konfidenzintervall**: Die Unsicherheit der Vorhersage
    """)

    show_3d_detail = st.toggle("🔍 3D-Ansicht aktivieren (Anatomie)", value=False, key="toggle_3d_detail")

    # Berechne Konfidenzintervall mit der stabilen get_prediction() API
    predictions = model.get_prediction(X)
    pred_frame = predictions.summary_frame(alpha=0.05)
    iv_l = pred_frame['obs_ci_lower'].values
    iv_u = pred_frame['obs_ci_upper'].values
    residuals = model.resid

    if show_3d_detail:
        # 3D Visualisierung: Anatomie im 3D Raum
        fig_detail = go.Figure()
    
        # Fehlerterme als vertikale Linien
        for i in range(len(x)):
            fig_detail.add_trace(go.Scatter3d(
                x=[x[i], x[i]], y=[y_pred[i], y_pred[i]], z=[y_pred[i], y[i]],
                mode='lines',
                line=dict(color='red', width=2),
                opacity=0.3,
                showlegend=False
            ))
    
        # Regressionslinie in 3D
        fig_detail.add_trace(go.Scatter3d(
            x=x, y=y_pred, z=y_pred,
            mode='lines',
            line=dict(color='blue', width=4),
            name='Regressionsgerade'
        ))
    
        # Datenpunkte
        fig_detail.add_trace(go.Scatter3d(
            x=x, y=y_pred, z=y,
            mode='markers',
            marker=dict(size=6, color='#1f77b4', opacity=0.7, line=dict(color='white', width=1)),
            name='Beobachtungen (y)'
        ))
    
        # Konfidenzintervall als Linien
        x_sorted_idx = np.argsort(x)
        x_sorted = x[x_sorted_idx]
        y_pred_sorted = y_pred[x_sorted_idx]
        iv_u_sorted = iv_u[x_sorted_idx]
        iv_l_sorted = iv_l[x_sorted_idx]
    
        fig_detail.add_trace(go.Scatter3d(
            x=x_sorted, y=y_pred_sorted, z=iv_u_sorted,
            mode='lines',
            line=dict(color='blue', width=2, dash='dash'),
            opacity=0.5,
            name='95% KI (upper)',
            showlegend=True
        ))
        fig_detail.add_trace(go.Scatter3d(
            x=x_sorted, y=y_pred_sorted, z=iv_l_sorted,
            mode='lines',
            line=dict(color='blue', width=2, dash='dash'),
            opacity=0.5,
            name='95% KI (lower)',
            showlegend=False
        ))
    
        # Schwerpunkt
        fig_detail.add_trace(go.Scatter3d(
            x=[x_mean], y=[y_mean_val], z=[y_mean_val],
            mode='markers',
            marker=dict(size=10, color='orange', symbol='diamond', line=dict(color='black', width=2)),
            name='Schwerpunkt'
        ))
    
        fig_detail.update_layout(
            title='3D Anatomie: Datenpunkte, Regressionslinie & Fehlerterme',
            scene=dict(
                xaxis_title=f'{x_label} (X)',
                yaxis_title='ŷ (Vorhersage)',
                zaxis_title=f'{y_label} (Y)',
                camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2))
            ),
            height=600,
            showlegend=True
        )
    
        st.plotly_chart(fig_detail, use_container_width=True)
    else:
        # 2D Original: Alle Annotationen
        fig_detail = go.Figure()

        # 1. Konfidenzintervall-Band (iv_l, iv_u) - draw first so it's in background
        x_sorted_idx = np.argsort(x)
        x_sorted = x[x_sorted_idx]
        iv_l_sorted = iv_l[x_sorted_idx]
        iv_u_sorted = iv_u[x_sorted_idx]
        
        fig_detail.add_trace(go.Scatter(
            x=np.concatenate([x_sorted, x_sorted[::-1]]),
            y=np.concatenate([iv_u_sorted, iv_l_sorted[::-1]]),
            fill='toself',
            fillcolor='rgba(0,0,255,0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Konfidenzintervall',
            showlegend=True
        ))

        # 2. Regressionsgerade
        fig_detail.add_trace(go.Scatter(
            x=x, y=y_pred,
            mode='lines',
            line=dict(color='blue', width=3),
            name=f'Modell: ŷ = {b0:.2f} + {b1:.2f}x'
        ))

        # 3. Datenpunkte
        fig_detail.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            marker=dict(size=10, color='#1f77b4', opacity=0.7, line=dict(color='white', width=2)),
            name='Beobachtungen yᵢ'
        ))
    
        # 4. Steigungsdreieck (Δx = 2, Position im mittleren Bereich)
        x_start = x_mean
        x_end = x_mean + 2
        y_start = b0 + b1 * x_start
        y_end = b0 + b1 * x_end
    
        # Horizontale Linie (Δx)
        fig_detail.add_trace(go.Scatter(
            x=[x_start, x_end], y=[y_start, y_start],
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            showlegend=False
        ))
        # Vertikale Linie (Δy)
        fig_detail.add_trace(go.Scatter(
            x=[x_end, x_end], y=[y_start, y_end],
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            showlegend=False
        ))
        
        # Annotations for triangle
        fig_detail.add_annotation(
            x=(x_start + x_end)/2, y=y_start - 0.4,
            text='Δx = 2',
            showarrow=False,
            font=dict(size=12)
        )
        fig_detail.add_annotation(
            x=x_end + 0.3, y=(y_start + y_end)/2,
            text=f'Δy = {b1*2:.2f}',
            showarrow=False,
            font=dict(size=12)
        )
        fig_detail.add_annotation(
            x=x_end + 0.3, y=y_start + 0.3,
            text=f'b₁ = Δy/Δx = {b1:.2f}',
            showarrow=False,
            font=dict(size=11),
            bgcolor='wheat',
            bordercolor='black',
            borderwidth=1
        )
    
        # 5. Epsilon-Annotation (ein markantes Residuum hervorheben)
        # Finde einen Punkt mit mittlerem Residuum für gute Sichtbarkeit
        resid_abs = np.abs(residuals)
        idx_eps = np.argmax(resid_abs)  # Grösstes Residuum
        if idx_eps > len(x) - 3:
            idx_eps = len(x) // 2  # Fallback zur Mitte
    
        # Vertikale Linie für εᵢ
        fig_detail.add_trace(go.Scatter(
            x=[x[idx_eps], x[idx_eps]], y=[y[idx_eps], y_pred[idx_eps]],
            mode='lines',
            line=dict(color='red', width=2.5),
            showlegend=False
        ))
        
        # Annotation mit Pfeil
        mid_y = (y[idx_eps] + y_pred[idx_eps]) / 2
        fig_detail.add_annotation(
            x=x[idx_eps] + 1, y=mid_y + 0.5,
            text=f'εᵢ = {residuals[idx_eps]:.2f}',
            ax=x[idx_eps], ay=mid_y,
            axref='x', ayref='y',
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor='red',
            font=dict(size=12, color='red'),
            bgcolor='mistyrose',
            bordercolor='red',
            borderwidth=1
        )
    
        # 6. Schwerpunkt markieren
        fig_detail.add_trace(go.Scatter(
            x=[x_mean], y=[y_mean_val],
            mode='markers',
            marker=dict(size=15, color='orange', symbol='star', line=dict(color='black', width=2)),
            name=f'Schwerpunkt (x̄, ȳ) = ({x_mean:.1f}, {y_mean_val:.1f})'
        ))
    
        fig_detail.update_layout(
            title='Anatomie des Regressionsmodells: Steigung, Fehlerterm & Konfidenzintervall',
            xaxis_title=x_label,
            yaxis_title=y_label,
            template='plotly_white',
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray'),
            height=600
        )
    
        st.plotly_chart(fig_detail, use_container_width=True)
        
    # Erklärungstext
    col_exp1, col_exp2, col_exp3 = st.columns(3)

    with col_exp1:
        st.markdown("### 📐 Steigungsdreieck")
        st.markdown(rf"""
        Das gestrichelte Dreieck zeigt visuell:
    
        $b_1 = \frac{{\Delta y}}{{\Delta x}} = \frac{{{b1*2:.2f}}}{{2}} = {b1:.2f}$
    
        **Interpretation:** Pro Einheit mehr x 
        steigt y um {b1:.2f} Einheiten.
        """)

    with col_exp2:
        st.markdown("### ⭐ Schwerpunkt")
        st.markdown(f"""
        Der orangene Stern markiert (x̄, ȳ).
    
        **Wichtig:** Die Regressionsgerade geht
        **immer** durch den Schwerpunkt!
    
        Dies folgt aus: $b_0 = \\bar{{y}} - b_1 \\bar{{x}}$
        """)

    with col_exp3:
        st.markdown("### 🔴 Fehlerterm εᵢ")
        st.markdown(f"""
        Die rote Linie zeigt das **Residuum**:
    
        $\\epsilon_i = y_i - \\hat{{y}}_i$
    
        Das Modell erklärt nicht alles – 
        dieser Rest ist der "Fehler".
        """)

    st.info("""
    **Das Konfidenzintervall (hellblauer Bereich):**

    Der 95%-Bereich zeigt die Unsicherheit unserer Vorhersage. Je weiter wir vom Schwerpunkt 
    entfernt sind, desto **breiter** wird das Intervall (mehr Unsicherheit bei Extrapolation).
    """)

    # Die vollständige OLS-Herleitung
    with st.expander("🧮 Mathematische Herleitung der OLS-Schätzer", expanded=False):
        st.markdown("### Schritt 1: Das Optimierungsproblem")
        st.latex(r"\min_{b_0, b_1} \sum_{i=1}^{n} (y_i - b_0 - b_1 \cdot x_i)^2 = \min_{b_0, b_1} SSE")
        st.caption("Wir suchen b₀ und b₁, die die Summe der quadrierten Abweichungen minimieren")
    
        st.markdown("### Schritt 2: Bedingungen erster Ordnung (FOC)")
        st.latex(r"\frac{\partial SSE}{\partial b_0} = -2 \sum_{i=1}^{n}(y_i - b_0 - b_1 x_i) \stackrel{!}{=} 0")
        st.latex(r"\frac{\partial SSE}{\partial b_1} = -2 \sum_{i=1}^{n}(y_i - b_0 - b_1 x_i) \cdot x_i \stackrel{!}{=} 0")
    
        st.markdown("### Schritt 3: Aus FOC für b₀ (Normalgleichung 1)")
        st.latex(r"\sum y_i = n \cdot b_0 + b_1 \sum x_i")
        st.latex(r"\Rightarrow b_0 = \bar{y} - b_1 \bar{x}")
        st.info("📍 **Wichtige Erkenntnis:** Die Regressionsgerade geht immer durch den Schwerpunkt (x̄, ȳ)!")
    
        st.markdown("### Schritt 4: Einsetzen in FOC für b₁")
        st.latex(r"\sum(y_i - \bar{y} + b_1 \bar{x} - b_1 x_i) \cdot x_i = 0")
        st.latex(r"\sum(y_i - \bar{y}) x_i = b_1 \sum(x_i - \bar{x}) x_i")
    
        st.markdown("### Schritt 5: Lösung für b₁")
        st.latex(r"b_1 = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2} = \frac{Cov(x,y)}{Var(x)}")
    
        st.success("""
        **Zusammenfassung:** Die OLS-Schätzer ergeben sich aus der Ableitung des Minimierungsproblems!
    
        Dies zeigt auch: Je grösser die Varianz von x (der Nenner), desto **präziser** kann b₁ geschätzt werden.
        """)

    # Interpretation der Koeffizienten
    st.markdown('<p class="subsection-header">📖 Interpretation der Ergebnisse</p>', unsafe_allow_html=True)

    col_int1, col_int2, col_int3 = st.columns(3)

    with col_int1:
        st.metric("Steigung b₁", f"{b1:.4f}")
        st.markdown(f"""
        **Interpretation:** 
    
        Im Durchschnitt verändert sich Y um **{b1:.2f} {y_unit}** 
        pro Einheit mehr X ({x_unit}).
        """)

    with col_int2:
        st.metric("Achsenabschnitt b₀", f"{b0:.4f}")
        st.markdown(f"""
        **Interpretation:** 
    
        Theoretisch der Y-Wert bei x=0. Praktisch ist b₀ oft **nicht direkt interpretierbar**, 
        weil x=0 ausserhalb des beobachteten Datenbereichs liegt (hier: X von {x.min():.1f} bis {x.max():.1f}).
    
        **Warum existiert b₀ trotzdem?**
        
        Die Regressionsgerade geht immer durch den Schwerpunkt (x̄={x_mean:.2f}, ȳ={y_mean_val:.2f}).
        Da die Steigung b₁ festliegt, ergibt sich b₀ automatisch aus: b₀ = ȳ - b₁·x̄
        
        Merke: b₀ sichert, dass die Gerade korrekt positioniert ist.
        """)

    with col_int3:
        # Prognose für einen Beispielwert
        x_new = np.percentile(x, 75)  # 75%-Perzentil von X
        y_new = b0 + b1 * x_new
        st.metric(f"Prognose (X={x_new:.1f})", f"{y_new:.2f} {y_unit}")
        st.markdown(f"""
        **Für X = {x_new:.1f}:**
    
        ŷ = {b0:.2f} + {b1:.2f} × {x_new:.1f} = **{y_new:.2f}**
        """)

    # =========================================================
    # KAPITEL 4: DIE GÜTEPRÜFUNG
    # =========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">4.0 Die Güteprüfung: Validierung des Regressionsmodells</p>', unsafe_allow_html=True)

    st.markdown("""
    Wir haben ein Modell – aber **wie gut** passt es wirklich? Die folgenden Gütemasse quantifizieren die Anpassung.
    """)

    # 4.1 Standardfehler der Regression
    st.markdown('<p class="subsection-header">4.1 Standardfehler der Regression (sₑ): Die durchschnittliche Prognoseabweichung</p>', unsafe_allow_html=True)

    col_se1, col_se2 = st.columns([2, 1])

    with col_se1:
        # Create standard error plot with plotly
        fig_se = go.Figure()
        
        # Data points
        fig_se.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            marker=dict(size=8, color='#1f77b4', opacity=0.6,
                       line=dict(width=1, color='white')),
            name='Data'
        ))
        
        # Regression line
        fig_se.add_trace(go.Scatter(
            x=x, y=y_pred,
            mode='lines',
            line=dict(color='red', width=3),
            name='Regressionsgerade'
        ))
        
        # ±2 se band
        fig_se.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([y_pred + 2*se_regression, (y_pred - 2*se_regression)[::-1]]),
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.1)',
            line=dict(width=0),
            name=f'±2·sₑ',
            showlegend=True
        ))
        
        # ±1 se band
        fig_se.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([y_pred + se_regression, (y_pred - se_regression)[::-1]]),
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(width=0),
            name=f'±1·sₑ = ±{se_regression:.3f}',
            showlegend=True
        ))
        
        fig_se.update_layout(
            title=f'Der Standardfehler sₑ = {se_regression:.4f} zeigt die typische Streuung um die Linie',
            xaxis_title=x_label,
            yaxis_title=y_label,
            template='plotly_white',
            hovermode='closest'
        )
    
        st.plotly_chart(fig_se, use_container_width=True)
        
    with col_se2:
        if show_formulas:
            st.latex(r"s_e = \sqrt{\frac{SSE}{n-2}} = \sqrt{\frac{\sum(y_i - \hat{y}_i)^2}{n-2}}")
            st.latex(f"s_e = \\sqrt{{\\frac{{{sse:.4f}}}{{{n}-2}}}} = {se_regression:.4f}")
    
        st.info(f"""
        **Interpretation:**
    
        Im Durchschnitt weichen die tatsächlichen Y-Werte um ca. **{se_regression:.3f} {y_unit}** 
        von den vorhergesagten Werten ab.
    
        Ein **kleinerer** Wert = bessere Anpassung.
        """)
        
        with st.expander("Warum n-2? (Degrees of Freedom)", expanded=False):
            st.markdown(f"""
            **Freiheitsgrade (df) = n - 2 = {n} - 2 = {n-2}**
            
            Wir teilen SSE durch (n-2), nicht durch n. Der Grund:
            
            1. **Wir haben 2 Parameter geschaetzt** (b₀ und b₁), 
               die aus den Daten berechnet wurden.
            
            2. **Jeder geschaetzte Parameter "verbraucht" einen Freiheitsgrad.**
               Die Residuen sind nicht mehr voellig frei – sie muessen sich 
               zu Null addieren und um die geschaetzte Linie streuen.
            
            3. **Konsequenz bei kleinem n:**
               Bei n=10 haben wir nur df=8. Die Schaetzung von sₑ wird unsicherer,
               t-Werte werden extremer bewertet, Konfidenzintervalle breiter.
            
            **Faustregel:** Bei n < 30 wirkt sich die Korrektur spuerbar aus.
            """)

    # --- Unsicherheit der Koeffizienten s_b₀ und s_b₁ ---
    st.markdown('<p class="subsection-header">4.1b Standardfehler der Koeffizienten: Die Unsicherheit von b₀ und b₁</p>', unsafe_allow_html=True)

    st.markdown("""
    Der Standardfehler **sₑ** beschreibt die Streuung der Punkte um die Linie. Aber wie **sicher** sind 
    wir uns über die Steigung (b₁) und den Achsenabschnitt (b₀) selbst? Das zeigen uns **s_b₀** und **s_b₁**.
    """)

    col_sb1, col_sb2 = st.columns([2, 1])

    with col_sb1:
        # Create 2-panel standard error comparison with plotly
        from plotly.subplots import make_subplots
        
        fig_sb = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'sₑ = {se_regression:.4f}<br>(Streuung der PUNKTE um die Linie)',
                           f's_b₁ = {sb1:.4f}<br>(Unsicherheit der STEIGUNG)')
        )
        
        # Left panel: se (residual standard error)
        fig_sb.add_trace(
            go.Scatter(x=x, y=y, mode='markers',
                      marker=dict(size=6, color='gray', opacity=0.5),
                      showlegend=False),
            row=1, col=1
        )
        
        fig_sb.add_trace(
            go.Scatter(x=x, y=y_pred, mode='lines',
                      line=dict(color='blue', width=3),
                      name='Unsere Schätzung'),
            row=1, col=1
        )
        
        # ±2 se band
        fig_sb.add_trace(
            go.Scatter(x=np.concatenate([x, x[::-1]]),
                      y=np.concatenate([y_pred + 2*se_regression, (y_pred - 2*se_regression)[::-1]]),
                      fill='toself', fillcolor='rgba(0, 0, 255, 0.1)',
                      line=dict(width=0), name=f'±2·sₑ', showlegend=True),
            row=1, col=1
        )
        
        # ±1 se band
        fig_sb.add_trace(
            go.Scatter(x=np.concatenate([x, x[::-1]]),
                      y=np.concatenate([y_pred + se_regression, (y_pred - se_regression)[::-1]]),
                      fill='toself', fillcolor='rgba(0, 0, 255, 0.2)',
                      line=dict(width=0), name=f'±1·sₑ = ±{se_regression:.3f}', showlegend=True),
            row=1, col=1
        )
        
        # Right panel: sb1 (standard error of slope)
        fig_sb.add_trace(
            go.Scatter(x=x, y=y, mode='markers',
                      marker=dict(size=4, color='gray', opacity=0.4),
                      showlegend=False),
            row=1, col=2
        )
        
        # Simulate multiple regression lines
        np.random.seed(456)
        x_sim = np.linspace(min(x), max(x), 100)
        for i in range(80):
            sim_slope = np.random.normal(b1, sb1)
            sim_intercept = np.random.normal(b0, sb0)
            fig_sb.add_trace(
                go.Scatter(x=x_sim, y=sim_intercept + sim_slope * x_sim,
                          mode='lines', line=dict(color='green', width=0.5),
                          opacity=0.05, showlegend=False),
                row=1, col=2
            )
        
        fig_sb.add_trace(
            go.Scatter(x=x, y=y_pred, mode='lines',
                      line=dict(color='black', width=3),
                      name='Unsere Schätzung'),
            row=1, col=2
        )
        
        fig_sb.update_xaxes(title_text=x_label, row=1, col=1)
        fig_sb.update_yaxes(title_text=y_label, row=1, col=1)
        fig_sb.update_xaxes(title_text=x_label, row=1, col=2)
        
        fig_sb.update_layout(height=400, template='plotly_white', showlegend=True)
    
        st.plotly_chart(fig_sb, use_container_width=True)
        
    with col_sb2:
        if show_formulas:
            st.markdown("### s_b₁ (Std. Error Steigung)")
            st.latex(r"s_{b_1} = \frac{s_e}{\sqrt{(n-1) \cdot s_x^2}} = \frac{s_e}{\sqrt{SS_x}}")
            st.latex(f"s_{{b_1}} = {sb1:.4f}")
        
            st.markdown("### s_b₀ (Std. Error Achsenabschnitt)")
            st.latex(f"s_{{b_0}} = {sb0:.4f}")
    
        st.warning(f"""
        **Wichtiger Unterschied:**
        - **sₑ = {se_regression:.4f}** → Wie stark streuen die **Punkte** um die Linie?
        - **s_b₁ = {sb1:.4f}** → Wie genau ist unsere geschätzte **Steigung**?
        - **s_b₀ = {sb0:.4f}** → Wie genau ist unser **Achsenabschnitt**?
    
        Die grünen Linien rechts zeigen: Mit anderen Stichproben hätten wir 
        etwas andere Steigungen bekommen!
        """)

    # 4.2 Bestimmtheitsmass R²
    st.markdown('<p class="subsection-header">4.2 Bestimmtheitsmass (R²): Der Anteil der erklärten Varianz</p>', unsafe_allow_html=True)

    col_r2_1, col_r2_2 = st.columns([2, 1])

    with col_r2_1:
        show_3d_var = st.toggle("📈 3D-Ansicht aktivieren (Varianz)", value=False, key="toggle_3d_var")
    
        if show_3d_var:
            # 3D Visualisierung: Würfel für SST, SSR, SSE using plotly
            from plotly.subplots import make_subplots
            
            fig_var = make_subplots(
                rows=1, cols=3,
                specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                subplot_titles=(
                    f'SST = {sst:.2f}<br>(Gesamte Variation)',
                    f'SSR = {ssr:.2f}<br>(Durch Modell erklärt)',
                    f'SSE = {sse:.2f}<br>(Unerklärt/Residuen)'
                )
            )
        
            # Normalisierung für Visualisierung
            sst_norm = sst
            ssr_norm = ssr / sst_norm
            sse_norm = sse / sst_norm
        
            # Helper function to create cube mesh
            def create_cube(height, color, row, col):
                # Cube vertices
                x = [0, 1, 1, 0, 0, 1, 1, 0]
                y = [0, 0, 1, 1, 0, 0, 1, 1]
                z = [0, 0, 0, 0, height, height, height, height]
                
                # Cube faces (triangles)
                i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
                j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
                k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
                
                fig_var.add_trace(
                    go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k,
                            color=color, opacity=0.3, flatshading=True,
                            showscale=False, showlegend=False),
                    row=row, col=col
                )
            
            # 1. SST (full cube)
            create_cube(1.0, 'orange', 1, 1)
            
            # 2. SSR (cube with ssr_norm height)
            create_cube(ssr_norm, 'green', 1, 2)
            
            # 3. SSE (cube with sse_norm height)
            create_cube(sse_norm, 'red', 1, 3)
        
            # Update layout
            fig_var.update_layout(
                title_text=f'3D Varianzzerlegung: SST = SSR + SSE → R² = {model.rsquared:.1%}',
                title_font_size=14,
                height=500,
                showlegend=False
            )
            
            # Update all 3D scenes
            for i in range(1, 4):
                fig_var.update_scenes(
                    {f'scene{i}': dict(
                        xaxis=dict(title='X', range=[0, 1]),
                        yaxis=dict(title='Y', range=[0, 1]),
                        zaxis=dict(title='Varianz', range=[0, 1]),
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
                    )}
                )
        
            st.plotly_chart(fig_var, use_container_width=True)
        else:
            # 2D Original: 3 Subplots nebeneinander
            fig_var = make_subplots(
                rows=1, cols=3,
                subplot_titles=(
                    f'SST = {sst:.2f}<br>(Gesamte Variation)',
                    f'SSR = {ssr:.2f}<br>(Durch Modell erklärt)',
                    f'SSE = {sse:.2f}<br>(Unerklärt/Residuen)'
                )
            )
        
            # SST
            fig_var.add_trace(
                go.Scatter(x=x, y=y, mode='markers',
                          marker=dict(size=8, color='gray', opacity=0.6),
                          showlegend=False),
                row=1, col=1
            )
            fig_var.add_hline(y=y_mean_val, line_color='orange', line_width=3,
                            annotation_text=f'ȳ = {y_mean_val:.2f}',
                            annotation_position="right", row=1, col=1)
            for i in range(len(x)):
                fig_var.add_trace(
                    go.Scatter(x=[x[i], x[i]], y=[y[i], y_mean_val],
                              mode='lines', line=dict(color='orange', width=2),
                              opacity=0.5, showlegend=False),
                    row=1, col=1
                )
        
            # SSR
            fig_var.add_trace(
                go.Scatter(x=x, y=y, mode='markers',
                          marker=dict(size=8, color='gray', opacity=0.3),
                          showlegend=False),
                row=1, col=2
            )
            fig_var.add_hline(y=y_mean_val, line_color='orange', line_width=2,
                            line_dash='dash', opacity=0.5, row=1, col=2)
            fig_var.add_trace(
                go.Scatter(x=x, y=y_pred, mode='lines',
                          line=dict(color='blue', width=3),
                          showlegend=False),
                row=1, col=2
            )
            for i in range(len(x)):
                fig_var.add_trace(
                    go.Scatter(x=[x[i], x[i]], y=[y_pred[i], y_mean_val],
                              mode='lines', line=dict(color='green', width=2),
                              opacity=0.6, showlegend=False),
                    row=1, col=2
                )
        
            # SSE
            fig_var.add_trace(
                go.Scatter(x=x, y=y, mode='markers',
                          marker=dict(size=8, color='gray', opacity=0.6),
                          showlegend=False),
                row=1, col=3
            )
            fig_var.add_trace(
                go.Scatter(x=x, y=y_pred, mode='lines',
                          line=dict(color='blue', width=3),
                          showlegend=False),
                row=1, col=3
            )
            for i in range(len(x)):
                fig_var.add_trace(
                    go.Scatter(x=[x[i], x[i]], y=[y[i], y_pred[i]],
                              mode='lines', line=dict(color='red', width=2),
                              opacity=0.6, showlegend=False),
                    row=1, col=3
                )
        
            # Update layout
            fig_var.update_xaxes(title_text="X", showgrid=True, gridcolor='lightgray')
            fig_var.update_yaxes(title_text="Y", row=1, col=1, showgrid=True, gridcolor='lightgray')
            fig_var.update_yaxes(showgrid=True, gridcolor='lightgray', row=1, col=2)
            fig_var.update_yaxes(showgrid=True, gridcolor='lightgray', row=1, col=3)
            
            fig_var.update_layout(
                title_text='Die Zerlegung der Varianz: SST = SSR + SSE',
                title_font_size=14,
                height=400,
                showlegend=False
            )
        
            st.plotly_chart(fig_var, use_container_width=True)
            
    with col_r2_2:
        if show_formulas:
            st.latex(r"R^2 = \frac{SSR}{SST} = 1 - \frac{SSE}{SST}")
            st.latex(f"R^2 = \\frac{{{ssr:.2f}}}{{{sst:.2f}}} = {model.rsquared:.4f}")
    
        # R² als Balken
        fig_r2bar = create_plotly_bar(
            categories=['SST\n(Total)', 'SSR\n(Erklärt)', 'SSE\n(Unerklärt)'],
            values=[sst, ssr, sse],
            colors=['gray', 'green', 'red'],
            title=f"R² = {model.rsquared:.1%}"
        )
        st.plotly_chart(fig_r2bar, use_container_width=True)
        
        st.success(f"""
        **{model.rsquared:.1%}** der Varianz in Y 
        wird durch X erklärt!
        """)

    # =========================================================
    # KAPITEL 5: DIE SIGNIFIKANZ
    # =========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">5.0 Die Signifikanz: Statistische Inferenz und Hypothesentests</p>', unsafe_allow_html=True)

    st.markdown("""
    Unser R² von {:.1%} sieht beeindruckend aus. Aber basiert es nur auf einer **Zufallsstichprobe**?
    Können wir sicher sein, dass dieser Zusammenhang auch in der **Grundgesamtheit** gilt?

    Dafür brauchen wir **Hypothesentests**!
    """.format(model.rsquared))

    # Annahmen
    st.markdown('<p class="subsection-header">📋 Voraussetzungen für valide Inferenz: Die Gauss-Markov Annahmen</p>', unsafe_allow_html=True)

    st.markdown("""
    Die OLS-Schätzung liefert **unverzerrte (erwartungstreue) Schätzer** für β₀ und β₁. 
    Für die Durchführung von Hypothesentests werden aber weitere Annahmen benötigt:
    """)

    # Visualisierung aller 4 Annahmen: Korrekt vs. Verletzt
    fig_assumptions = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            '✅ (1) E(εᵢ|xᵢ) = 0<br>Fehler symmetrisch um Null',
            '❌ E(εᵢ|xᵢ) ≠ 0<br>Nicht-linearer Zusammenhang ignoriert',
            '✅ (2) Var(εᵢ) = σ² (konstant)<br>Homoskedastizität',
            '❌ Var(εᵢ|xᵢ) = f(xᵢ)<br>Heteroskedastizität (Trichter)',
            '✅ (3) Cov(εᵢ, εⱼ) = 0<br>Keine Autokorrelation',
            '❌ Cov(εᵢ, εⱼ) ≠ 0<br>Autokorrelation (Muster in Residuen)',
            '✅ (4) ε ~ N(0, σ²)<br>Normalverteilte Residuen',
            '❌ ε nicht normalverteilt<br>Schiefe Verteilung'
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )
    
    np.random.seed(123)
    n_demo = 100
    x_demo = np.linspace(1, 10, n_demo)

    # === ANNAHME 1: E(εᵢ|xᵢ) = 0 ===
    # Korrekt
    y_correct_1 = 2 + 0.5 * x_demo + np.random.normal(0, 1, n_demo)
    fig_assumptions.add_trace(
        go.Scatter(x=x_demo, y=y_correct_1, mode='markers',
                  marker=dict(size=4, color='green', opacity=0.5),
                  showlegend=False),
        row=1, col=1
    )
    fig_assumptions.add_trace(
        go.Scatter(x=x_demo, y=2 + 0.5 * x_demo, mode='lines',
                  line=dict(color='blue', width=2), showlegend=False),
        row=1, col=1
    )
    fig_assumptions.add_hline(y=2 + 0.5 * 5.5, line_dash="dash", line_color="gray",
                             opacity=0.5, row=1, col=1)

    # Verletzt
    y_violated_1 = 2 + 0.5 * x_demo + 0.1 * (x_demo - 5)**2 + np.random.normal(0, 0.5, n_demo)
    fig_assumptions.add_trace(
        go.Scatter(x=x_demo, y=y_violated_1, mode='markers',
                  marker=dict(size=4, color='red', opacity=0.5),
                  showlegend=False),
        row=1, col=2
    )
    fig_assumptions.add_trace(
        go.Scatter(x=x_demo, y=2 + 0.5 * x_demo, mode='lines',
                  line=dict(color='blue', width=2),
                  name='Lineares Modell', showlegend=False),
        row=1, col=2
    )
    fig_assumptions.add_trace(
        go.Scatter(x=x_demo, y=2 + 0.5 * x_demo + 0.1 * (x_demo - 5)**2,
                  mode='lines', line=dict(color='green', width=2, dash='dash'),
                  name='Wahrer Zusammenhang', showlegend=False),
        row=1, col=2
    )

    # === ANNAHME 2: Var(εᵢ) = σ² (Homoskedastizität) ===
    # Korrekt
    y_correct_2 = 2 + 0.5 * x_demo + np.random.normal(0, 1, n_demo)
    y_line_2 = 2 + 0.5 * x_demo
    fig_assumptions.add_trace(
        go.Scatter(x=np.concatenate([x_demo, x_demo[::-1]]),
                  y=np.concatenate([y_line_2 + 2, (y_line_2 - 2)[::-1]]),
                  fill='toself', fillcolor='rgba(0,0,255,0.2)',
                  line=dict(width=0), showlegend=False),
        row=2, col=1
    )
    fig_assumptions.add_trace(
        go.Scatter(x=x_demo, y=y_correct_2, mode='markers',
                  marker=dict(size=4, color='green', opacity=0.5),
                  showlegend=False),
        row=2, col=1
    )
    fig_assumptions.add_trace(
        go.Scatter(x=x_demo, y=y_line_2, mode='lines',
                  line=dict(color='blue', width=2), showlegend=False),
        row=2, col=1
    )

    # Verletzt
    hetero_noise = np.random.normal(0, 0.3 * x_demo, n_demo)
    y_violated_2 = 2 + 0.5 * x_demo + hetero_noise
    fig_assumptions.add_trace(
        go.Scatter(x=np.concatenate([x_demo, x_demo[::-1]]),
                  y=np.concatenate([y_line_2 + 0.6 * x_demo, (y_line_2 - 0.6 * x_demo)[::-1]]),
                  fill='toself', fillcolor='rgba(255,0,0,0.2)',
                  line=dict(width=0), showlegend=False),
        row=2, col=2
    )
    fig_assumptions.add_trace(
        go.Scatter(x=x_demo, y=y_violated_2, mode='markers',
                  marker=dict(size=4, color='red', opacity=0.5),
                  showlegend=False),
        row=2, col=2
    )
    fig_assumptions.add_trace(
        go.Scatter(x=x_demo, y=y_line_2, mode='lines',
                  line=dict(color='blue', width=2), showlegend=False),
        row=2, col=2
    )

    # === ANNAHME 3: Cov(εᵢ, εⱼ) = 0 (Keine Autokorrelation) ===
    # Korrekt
    y_correct_3 = 2 + 0.5 * x_demo + np.random.normal(0, 1, n_demo)
    resid_correct_3 = y_correct_3 - (2 + 0.5 * x_demo)
    fig_assumptions.add_trace(
        go.Scatter(x=list(range(n_demo-1)), y=resid_correct_3[:-1],
                  mode='markers',
                  marker=dict(size=4, color=resid_correct_3[1:],
                            colorscale='RdBu', showscale=False),
                  opacity=0.7, showlegend=False),
        row=3, col=1
    )
    fig_assumptions.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

    # Verletzt
    auto_error = np.zeros(n_demo)
    auto_error[0] = np.random.normal(0, 1)
    for i in range(1, n_demo):
        auto_error[i] = 0.8 * auto_error[i-1] + np.random.normal(0, 0.5)
    fig_assumptions.add_trace(
        go.Scatter(x=list(range(n_demo)), y=auto_error,
                  mode='lines+markers',
                  line=dict(color='red', width=1.5),
                  marker=dict(size=3, color='red'),
                  opacity=0.7, showlegend=False),
        row=3, col=2
    )
    fig_assumptions.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=2)

    # === ANNAHME 4: ε ~ N(0, σ²) (Normalverteilung) ===
    # Korrekt
    normal_resid = np.random.normal(0, 1, n_demo)
    fig_assumptions.add_trace(
        go.Histogram(x=normal_resid, histnorm='probability density',
                    marker_color='green', opacity=0.7, nbinsx=20,
                    showlegend=False),
        row=4, col=1
    )
    x_norm = np.linspace(-4, 4, 100)
    fig_assumptions.add_trace(
        go.Scatter(x=x_norm, y=stats.norm.pdf(x_norm, 0, 1),
                  mode='lines', line=dict(color='blue', width=2),
                  showlegend=False),
        row=4, col=1
    )

    # Verletzt
    skewed_resid = np.random.exponential(1, n_demo) - 1
    fig_assumptions.add_trace(
        go.Histogram(x=skewed_resid, histnorm='probability density',
                    marker_color='red', opacity=0.7, nbinsx=20,
                    showlegend=False),
        row=4, col=2
    )
    fig_assumptions.add_trace(
        go.Scatter(x=x_norm, y=stats.norm.pdf(x_norm, 0, 1),
                  mode='lines', line=dict(color='blue', width=2, dash='dash'),
                  name='Normalverteilung', showlegend=False),
        row=4, col=2
    )

    # Update axes labels
    fig_assumptions.update_xaxes(title_text="X", row=1, col=1)
    fig_assumptions.update_yaxes(title_text="Y", row=1, col=1)
    fig_assumptions.update_xaxes(title_text="X", row=1, col=2)
    fig_assumptions.update_xaxes(title_text="X", row=2, col=1)
    fig_assumptions.update_yaxes(title_text="Y", row=2, col=1)
    fig_assumptions.update_xaxes(title_text="X", row=2, col=2)
    fig_assumptions.update_xaxes(title_text="Beobachtung i", row=3, col=1)
    fig_assumptions.update_yaxes(title_text="Residuum eᵢ", row=3, col=1)
    fig_assumptions.update_xaxes(title_text="Zeit/Beobachtung", row=3, col=2)
    fig_assumptions.update_yaxes(title_text="Residuum", row=3, col=2)
    fig_assumptions.update_xaxes(title_text="Residuum", row=4, col=1)
    fig_assumptions.update_yaxes(title_text="Dichte", row=4, col=1)
    fig_assumptions.update_xaxes(title_text="Residuum", row=4, col=2)

    fig_assumptions.update_layout(
        title_text='Die 4 Gauss-Markov Annahmen: Korrekt (links) vs. Verletzt (rechts)',
        title_font_size=16,
        height=1400,
        showlegend=False
    )

    st.plotly_chart(fig_assumptions, use_container_width=True)
    
    # Erklärungstext zu den Konsequenzen
    col_gm1, col_gm2 = st.columns(2)

    with col_gm1:
        st.success("""
        **Wenn alle Annahmen erfüllt sind:**
    
        Nach dem **Satz von Gauss-Markov** ist der OLS-Schätzer dann **BLUE**:
        - **B**est = kleinste Varianz
        - **L**inear = lineare Funktion der Daten
        - **U**nbiased = erwartungstreu
        - **E**stimator = Schätzer
    
        → Es gibt keinen anderen linearen Schätzer mit kleinerer Varianz!
        """)

    with col_gm2:
        st.error("""
        **Konsequenzen bei Verletzungen:**
    
        | Verletzung | Problem |
        |------------|---------|
        | **(1) E(ε|x) ≠ 0** | Schätzer **verzerrt** (biased) |
        | **(2) Heteroskedastizität** | Standardfehler **falsch** → t/F-Tests ungültig |
        | **(3) Autokorrelation** | Standardfehler **falsch** → Tests ungültig |
        | **(4) Nicht-Normalität** | Bei kleinem n: Tests **ungültig** |
        """)

    # Unsere Daten prüfen
    with st.expander("🔍 Diagnose: Erfüllen unsere Daten die Annahmen?"):
        col_diag1, col_diag2 = st.columns(2)
    
        with col_diag1:
            # Create residual plot with plotly
            fig_diag1 = go.Figure()
            
            fig_diag1.add_trace(go.Scatter(
                x=y_pred, y=model.resid,
                mode='markers',
                marker=dict(size=7, color='blue', opacity=0.6),
                showlegend=False
            ))
            
            fig_diag1.add_hline(y=0, line_dash='dash', line_color='red', line_width=2)
            
            fig_diag1.update_layout(
                title='Residuenplot: Prüfung (1) & (2)',
                xaxis_title='Vorhergesagte Werte (ŷ)',
                yaxis_title='Residuen (e)',
                template='plotly_white',
                hovermode='closest'
            )
            
            st.plotly_chart(fig_diag1, use_container_width=True)
                    
            st.markdown("""
            **Interpretation:**
            - Punkte sollten **zufällig** um 0 streuen
            - Kein Muster/Trichter → Homoskedastizität ✓
            - Kein Bogen → Linearität ✓
            """)
    
        with col_diag2:
            # Q-Q Plot (Normalität) with plotly
            from scipy.stats import probplot
            qq = probplot(model.resid, dist="norm")
            
            fig_diag2 = go.Figure()
            
            fig_diag2.add_trace(go.Scatter(
                x=qq[0][0], y=qq[0][1],
                mode='markers',
                marker=dict(size=6, color='blue', opacity=0.6),
                name='Data'
            ))
            
            # Add reference line
            fig_diag2.add_trace(go.Scatter(
                x=qq[0][0], y=qq[1][1] + qq[1][0]*qq[0][0],
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name='Reference Line'
            ))
            
            fig_diag2.update_layout(
                title='Q-Q Plot: Prüfung (4) Normalität',
                xaxis_title='Theoretical Quantiles',
                yaxis_title='Sample Quantiles',
                template='plotly_white',
                showlegend=False
            )
            
            st.plotly_chart(fig_diag2, use_container_width=True)
                    
            st.markdown("""
            **Interpretation:**
            - Punkte sollten auf der **Diagonale** liegen
            - Abweichungen → Nicht-Normalität
            - Bei n > 30: Weniger kritisch (CLT)
            """)

    # t-Test
    st.markdown('<p class="subsection-header">🔬 Der t-Test: Ist die Steigung signifikant von Null verschieden?</p>', unsafe_allow_html=True)

    col_t1, col_t2 = st.columns([2, 1])

    with col_t1:
        # Create t-distribution plot with plotly
        x_t = np.linspace(-5, max(5, abs(t_val) + 2), 300)
        y_t = stats.t.pdf(x_t, df=df_resid)
        
        fig_t = go.Figure()
        
        # Main distribution curve
        fig_t.add_trace(go.Scatter(
            x=x_t, y=y_t,
            mode='lines',
            line=dict(color='black', width=3),
            name=f't-Verteilung (df={df_resid})'
        ))
        
        # Shaded p-value regions
        mask = abs(x_t) > abs(t_val)
        fig_t.add_trace(go.Scatter(
            x=x_t[mask], y=y_t[mask],
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.3)',
            line=dict(width=0),
            name=f'p-Wert = {model.pvalues[1]:.4g}',
            showlegend=True
        ))
        
        # Critical values
        t_crit = stats.t.ppf(0.975, df=df_resid)
        fig_t.add_vline(x=t_crit, line_dash='dash', line_color='orange',
                       line_width=2, opacity=0.7)
        fig_t.add_vline(x=-t_crit, line_dash='dash', line_color='orange',
                       line_width=2, opacity=0.7,
                       annotation_text=f'Kritische Werte: ±{t_crit:.2f}')
        
        # Observed t-value
        fig_t.add_vline(x=t_val, line_color='blue', line_width=4,
                       annotation_text=f'Unser t-Wert = {t_val:.2f}')
        
        fig_t.update_layout(
            title=f'H₀: β₁ = 0 vs. H₁: β₁ ≠ 0<br>t = b₁/s_b₁ = {b1:.4f}/{sb1:.4f} = {t_val:.2f}',
            xaxis_title='t-Wert',
            yaxis_title='Dichte',
            template='plotly_white',
            hovermode='x'
        )
    
        st.plotly_chart(fig_t, use_container_width=True)
        
    with col_t2:
        if show_formulas:
            st.latex(r"t = \frac{b_1 - 0}{s_{b_1}}")
            st.latex(f"t = \\frac{{{b1:.4f}}}{{{sb1:.4f}}} = {t_val:.2f}")
    
        stars = get_signif_stars(model.pvalues[1])
        p_val = model.pvalues[1]
    
        st.metric("t-Wert", f"{t_val:.3f}")
        st.metric("p-Wert", f"{p_val:.4g}")
        st.metric("Signifikanz", stars)
    
        if p_val < 0.001:
            st.success("✅ **Höchst signifikant!** H₀ wird verworfen.")
        elif p_val < 0.05:
            st.success("✅ **Signifikant!** H₀ wird verworfen.")
        else:
            st.error("❌ **Nicht signifikant.** H₀ kann nicht verworfen werden.")

    # F-Test
    st.markdown('<p class="subsection-header">⚖️ Der F-Test: Erklärt das Modell signifikant Varianz?</p>', unsafe_allow_html=True)

    col_f1, col_f2 = st.columns([2, 1])

    with col_f1:
        # Create F-distribution plot with plotly
        x_f = np.linspace(0, max(10, f_val + 5), 300)
        y_f = stats.f.pdf(x_f, dfn=1, dfd=df_resid)
        
        fig_f = go.Figure()
        
        # Main distribution curve
        fig_f.add_trace(go.Scatter(
            x=x_f, y=y_f,
            mode='lines',
            line=dict(color='black', width=3),
            name=f'F-Verteilung (df₁=1, df₂={df_resid})'
        ))
        
        # Shaded p-value region
        mask = x_f > f_val
        fig_f.add_trace(go.Scatter(
            x=x_f[mask], y=y_f[mask],
            fill='tozeroy',
            fillcolor='rgba(128, 0, 128, 0.3)',
            line=dict(width=0),
            name=f'p-Wert = {model.f_pvalue:.4g}',
            showlegend=True
        ))
        
        # Critical value
        f_crit = stats.f.ppf(0.95, dfn=1, dfd=df_resid)
        fig_f.add_vline(x=f_crit, line_dash='dash', line_color='orange',
                       line_width=2, opacity=0.7,
                       annotation_text=f'Kritisch: {f_crit:.2f}')
        
        # Observed F-value
        fig_f.add_vline(x=f_val, line_color='purple', line_width=4,
                       annotation_text=f'Unser F-Wert = {f_val:.2f}')
        
        fig_f.update_layout(
            title=f'H₀: R² = 0 (Modell erklärt nichts)<br>F = MSR/MSE = {msr:.2f}/{mse:.2f} = {f_val:.2f}',
            xaxis_title='F-Wert',
            yaxis_title='Dichte',
            xaxis_range=[0, max(15, f_val + 5)],
            template='plotly_white',
            hovermode='x'
        )
    
        st.plotly_chart(fig_f, use_container_width=True)
        
    with col_f2:
        if show_formulas:
            st.latex(r"F = \frac{MSR}{MSE} = \frac{SSR/df_1}{SSE/df_2}")
            st.latex(f"F = \\frac{{{ssr:.2f}/1}}{{{sse:.2f}/{n-2}}} = {f_val:.2f}")
    
        st.metric("F-Wert", f"{f_val:.2f}")
        st.metric("p-Wert", f"{model.f_pvalue:.4g}")
    
        st.info(f"""
        **Bei einfacher Regression gilt:**
    
        t² = F
    
        {t_val:.2f}² = {t_val**2:.2f} ≈ {f_val:.2f} ✓
    
        Beide Tests führen zum **gleichen Schluss**!
        """)

    # ANOVA-Tabelle
    st.markdown('<p class="subsection-header">📊 Die vollständige ANOVA-Tabelle</p>', unsafe_allow_html=True)

    anova_df = pd.DataFrame({
        'Quelle': ['Regression (SSR)', 'Residuen (SSE)', 'Total (SST)'],
        'Quadratsumme': [f'{ssr:.4f}', f'{sse:.4f}', f'{sst:.4f}'],
        'df': [1, n-2, n-1],
        'Mittlere Quadratsumme': [f'{msr:.4f}', f'{mse:.4f}', '—'],
        'F-Wert': [f'{f_val:.2f}', '—', '—'],
        'p-Wert': [f'{model.f_pvalue:.4g} {get_signif_stars(model.f_pvalue)}', '—', '—']
    })

    st.dataframe(anova_df, width='stretch', hide_index=True)

    # R-Style Output mit Annotationen
    st.markdown('<p class="subsection-header">💻 Der komplette R-Style Output mit Annotationen</p>', unsafe_allow_html=True)

    st.markdown("""
    Dies ist das Herzstück unserer Analyse: Die **vollständige Zusammenfassung** des Regressionsmodells 
    im R-Stil – aber mit **farbigen Annotationen**, die jedes Element erklären!
    """)

    # Die annotierte R-Output Figur
    fig_r_output = create_r_output_figure(model, feature_name=x_label, figsize=(18, 13))
    st.plotly_chart(fig_r_output, use_container_width=True)
    
    # Zusätzlich noch den textuellen Output
    with st.expander("📜 Reiner Text-Output (zum Kopieren)"):
        st.code(model.summary().as_text(), language=None)

    # =========================================================
    # KAPITEL 5.5: ANOVA FÜR GRUPPENVERGLEICHE (NEU)
    # =========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">5.5 ANOVA für Gruppenvergleiche: Mehr als zwei Gruppen</p>', unsafe_allow_html=True)

    st.markdown("""
    Der F-Test, den wir gerade gesehen haben, ist ein **Spezialfall der ANOVA** (Analysis of Variance). 
    Die ANOVA erweitert den Vergleich auf **mehr als zwei Gruppen**.

    **Praxisbeispiel:** Unser Elektronikmarkt hat Filialen in **3 Regionen** (Nord, Mitte, Süd). 
    Unterscheiden sich die durchschnittlichen Umsätze zwischen den Regionen signifikant?
    """)

    # --- Interaktive ANOVA-Parameter ---
    st.sidebar.markdown("---")
    with st.sidebar.expander("🧪 ANOVA-Beispiel", expanded=False):
        anova_effect = st.slider("Effektstärke Regionen", 0.0, 2.0, 0.8, 0.1,
                                help="Wie stark unterscheiden sich die Regionsmittelwerte?")
        anova_noise_level = st.slider("Streuung innerhalb Gruppen", 0.5, 2.0, 1.0, 0.1, key="anova_noise",
                                      help="Varianz innerhalb jeder Gruppe")

    # Daten für 3 Regionen generieren
    np.random.seed(int(seed) + 100)
    n_per_group = max(n // 3, 4)

    # Regionsmittelwerte
    mu_nord = y_mean_val - anova_effect
    mu_mitte = y_mean_val
    mu_sued = y_mean_val + anova_effect

    # Daten generieren
    region_nord = np.random.normal(mu_nord, anova_noise_level, n_per_group)
    region_mitte = np.random.normal(mu_mitte, anova_noise_level, n_per_group)
    region_sued = np.random.normal(mu_sued, anova_noise_level, n_per_group)

    # DataFrame erstellen
    df_anova = pd.DataFrame({
        'Umsatz': np.concatenate([region_nord, region_mitte, region_sued]),
        'Region': ['Nord'] * n_per_group + ['Mitte'] * n_per_group + ['Süd'] * n_per_group
    })

    # ANOVA berechnen
    from statsmodels.formula.api import ols as ols_formula

    model_anova = ols_formula('Umsatz ~ C(Region)', data=df_anova).fit()
    anova_table = sm.stats.anova_lm(model_anova, typ=2)

    # Kennzahlen extrahieren (typ-sicher)
    grand_mean_anova = df_anova['Umsatz'].mean()
    group_means = df_anova.groupby('Region')['Umsatz'].mean()
    sstr_anova = safe_scalar(anova_table.loc['C(Region)', 'sum_sq'])
    sse_anova = safe_scalar(anova_table.loc['Residual', 'sum_sq'])
    sst_anova = sstr_anova + sse_anova
    f_anova = safe_scalar(anova_table.loc['C(Region)', 'F'])
    p_anova = safe_scalar(anova_table.loc['C(Region)', 'PR(>F)'])

    col_anova1, col_anova2 = st.columns([2, 1])

    with col_anova1:
        show_3d_anova = st.toggle("🏔️ 3D-Ansicht aktivieren (ANOVA Landscape)", value=False, key="toggle_3d_anova")
    
        if show_3d_anova:
            # 3D Landscape Visualisierung: Berglandschaft für jede Gruppe
            from scipy.stats import norm
        
            fig_anova_viz = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'scatter3d'}, {'type': 'bar'}]],
                subplot_titles=('3D Landscape: Gruppen als Verteilungshuegel',
                              f'SST = SSTR + SSE = {sst_anova:.2f}')
            )
        
            # 1. 3D Surface für Gruppen-Verteilungen
            regions = ['Nord', 'Mitte', 'Süd']
            colors_list = ['#3498db', '#2ecc71', '#e74c3c']
        
            # X-Achse: Umsatz-Werte
            x_vals = np.linspace(
                min(df_anova['Umsatz']) - 1,
                max(df_anova['Umsatz']) + 1,
                100
            )
        
            for i, (region, color) in enumerate(zip(regions, colors_list)):
                data = df_anova[df_anova['Region'] == region]['Umsatz']
                mu = data.mean()
                sigma = data.std()
            
                # Normalverteilung für jede Gruppe
                y_vals = norm.pdf(x_vals, mu, sigma)
            
                # 3D Plot: x-Achse = Umsatz, y-Achse = Region (i), z-Achse = Dichte
                fig_anova_viz.add_trace(
                    go.Scatter3d(x=x_vals, y=np.full_like(x_vals, i), z=y_vals,
                               mode='lines', line=dict(color=color, width=4),
                               name=f'{region}'),
                    row=1, col=1
                )
            
                # Bars under curve (simplified as vertical lines)
                for j in range(0, len(x_vals), 10):
                    fig_anova_viz.add_trace(
                        go.Scatter3d(x=[x_vals[j], x_vals[j]], y=[i, i], z=[0, y_vals[j]],
                                   mode='lines', line=dict(color=color, width=2),
                                   opacity=0.3, showlegend=False),
                        row=1, col=1
                    )
        
            # Gesamtmittelwert als Linie
            y_overall = norm.pdf(x_vals, grand_mean_anova, df_anova['Umsatz'].std())
            fig_anova_viz.add_trace(
                go.Scatter3d(x=x_vals, y=np.full_like(x_vals, -0.5), z=y_overall,
                           mode='lines', line=dict(color='black', width=3, dash='dash'),
                           name='Gesamtverteilung'),
                row=1, col=1
            )
        
            # 2. Varianzzerlegung
            fig_anova_viz.add_trace(
                go.Bar(y=['Varianzzerlegung'], x=[sstr_anova],
                      orientation='h', marker_color='green', opacity=0.7,
                      name=f'SSTR (Zwischen) = {sstr_anova:.2f}',
                      text=f'{sstr_anova/sst_anova*100:.1f}%',
                      textposition='inside', textfont=dict(color='white', size=12)),
                row=1, col=2
            )
            fig_anova_viz.add_trace(
                go.Bar(y=['Varianzzerlegung'], x=[sse_anova],
                      orientation='h', marker_color='red', opacity=0.7,
                      name=f'SSE (Innerhalb) = {sse_anova:.2f}',
                      text=f'{sse_anova/sst_anova*100:.1f}%',
                      textposition='inside', textfont=dict(color='white', size=12)),
                row=1, col=2
            )
        
            # Update layout
            fig_anova_viz.update_layout(
                height=500,
                barmode='stack',
                showlegend=True,
                scene=dict(
                    xaxis_title=y_label,
                    yaxis_title='Gruppen',
                    zaxis_title='Dichte',
                    yaxis=dict(tickvals=[0, 1, 2], ticktext=regions),
                    camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2))
                )
            )
            fig_anova_viz.update_xaxes(title_text='Quadratsumme', row=1, col=2)
        
            st.plotly_chart(fig_anova_viz, use_container_width=True)
        else:
            # 2D Original: Boxplot + Varianzzerlegung
            fig_anova_viz = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Gruppenvergleich: Streuung innerhalb (SSE) vs. zwischen (SSTR)',
                              f'SST = SSTR + SSE = {sst_anova:.2f}')
            )
        
            # 1. Boxplot mit Punkten
            regions = ['Nord', 'Mitte', 'Süd']
            colors_list = ['#3498db', '#2ecc71', '#e74c3c']
        
            for i, (region, color) in enumerate(zip(regions, colors_list)):
                data = df_anova[df_anova['Region'] == region]['Umsatz']
            
                # Jittered Scatter
                np.random.seed(42 + i)
                jitter = np.random.normal(0, 0.08, len(data))
                fig_anova_viz.add_trace(
                    go.Scatter(x=np.full(len(data), i) + jitter, y=data,
                              mode='markers',
                              marker=dict(size=10, color=color, opacity=0.6,
                                        line=dict(color='white', width=1)),
                              name=region, showlegend=False),
                    row=1, col=1
                )
            
                # Gruppenmittelwert horizontal line
                group_mean = data.mean()
                fig_anova_viz.add_shape(
                    type='line',
                    x0=i-0.3, x1=i+0.3, y0=group_mean, y1=group_mean,
                    line=dict(color=color, width=4),
                    row=1, col=1
                )
            
                # Linien zu Gruppenmittelwert (SSE) - nur ein paar zeigen
                for j in range(min(5, len(data))):
                    fig_anova_viz.add_trace(
                        go.Scatter(x=[i + jitter[j], i + jitter[j]],
                                  y=[data.iloc[j], group_mean],
                                  mode='lines', line=dict(color=color, width=1),
                                  opacity=0.2, showlegend=False),
                        row=1, col=1
                    )
        
            # Gesamtmittelwert
            fig_anova_viz.add_hline(y=grand_mean_anova, line_dash="dash", line_color="black",
                                   line_width=2, annotation_text=f'Gesamtmittel: {grand_mean_anova:.2f}',
                                   annotation_position="right", row=1, col=1)
        
            # 2. Varianzzerlegung als gestapelter Balken
            fig_anova_viz.add_trace(
                go.Bar(y=['Varianzzerlegung'], x=[sstr_anova],
                      orientation='h', marker_color='green', opacity=0.7,
                      name=f'SSTR (Zwischen) = {sstr_anova:.2f}',
                      text=f'{sstr_anova/sst_anova*100:.1f}%',
                      textposition='inside', textfont=dict(color='white', size=12)),
                row=1, col=2
            )
            fig_anova_viz.add_trace(
                go.Bar(y=['Varianzzerlegung'], x=[sse_anova],
                      orientation='h', marker_color='red', opacity=0.7,
                      name=f'SSE (Innerhalb) = {sse_anova:.2f}',
                      text=f'{sse_anova/sst_anova*100:.1f}%',
                      textposition='inside', textfont=dict(color='white', size=12)),
                row=1, col=2
            )
        
            # Update layout
            fig_anova_viz.update_xaxes(title_text='Region', tickvals=[0, 1, 2],
                                      ticktext=regions, row=1, col=1)
            fig_anova_viz.update_yaxes(title_text=y_label, showgrid=True,
                                      gridcolor='lightgray', row=1, col=1)
            fig_anova_viz.update_xaxes(title_text='Quadratsumme', row=1, col=2)
            
            fig_anova_viz.update_layout(
                height=500,
                barmode='stack',
                showlegend=True
            )
        
            st.plotly_chart(fig_anova_viz, use_container_width=True)
            
    with col_anova2:
        if show_formulas:
            st.markdown("### ANOVA-Formeln")
            st.latex(r"SSTR = \sum_{j=1}^{k} n_j (\bar{X}_j - \bar{\bar{X}})^2")
            st.latex(r"SSE = \sum_{j=1}^{k} \sum_{i=1}^{n_j} (X_{ij} - \bar{X}_j)^2")
            st.latex(r"F = \frac{MSTR}{MSE} = \frac{SSTR/(k-1)}{SSE/(n-k)}")
    
        st.metric("F-Wert", f"{f_anova:.2f}")
        st.metric("p-Wert", f"{p_anova:.4f}")
        st.metric("Signifikanz", get_signif_stars(p_anova))
    
        if p_anova < 0.05:
            st.success("✅ Die Regionen unterscheiden sich **signifikant**!")
        else:
            st.warning("⚠️ Kein signifikanter Unterschied zwischen Regionen.")

    # ANOVA-Tabelle
    st.markdown('<p class="subsection-header">📋 Die ANOVA-Tabelle (Gruppenvergleich)</p>', unsafe_allow_html=True)

    k = 3  # Anzahl Gruppen
    n_total = len(df_anova)
    df_between = k - 1
    df_within = n_total - k
    mstr_anova = sstr_anova / df_between
    mse_anova_val = sse_anova / df_within

    anova_display = pd.DataFrame({
        'Quelle': ['Zwischen Gruppen (SSTR)', 'Innerhalb Gruppen (SSE)', 'Total (SST)'],
        'Quadratsumme': [f'{sstr_anova:.3f}', f'{sse_anova:.3f}', f'{sst_anova:.3f}'],
        'df': [df_between, df_within, n_total - 1],
        'Mittlere QS': [f'{mstr_anova:.3f}', f'{mse_anova_val:.3f}', '—'],
        'F-Wert': [f'{f_anova:.3f}', '—', '—'],
        'p-Wert': [f'{p_anova:.4f} {get_signif_stars(p_anova)}', '—', '—']
    })

    st.dataframe(anova_display, width='stretch', hide_index=True)

    # Verbindung zur Regression
    st.info(f"""
    **🔗 Verbindung zur Regression:**

    Die ANOVA ist eigentlich eine **kategoriale Regression**! 
    Wenn wir die Region als Dummy-Variable kodieren, erhalten wir das gleiche Ergebnis.

    In der einfachen Regression (1 kontinuierliche Variable) gilt:
    - **F = t²** (bei einer Gruppe df₁ = 1)
    - Unser F-Test prüft: "Erklärt X signifikant Varianz in Y?"
    - ANOVA prüft: "Erklärt die Gruppenzugehörigkeit signifikant Varianz in Y?"

    **→ Beides sind Spezialfälle des allgemeinen linearen Modells!**
    """)

    # =========================================================
    # KAPITEL 5.6: HOMO- vs. HETEROSKEDASTIZITÄT (Das grosse Problem)
    # =========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">⚠️ Das grosse Problem: Heteroskedastizität</p>', unsafe_allow_html=True)

    st.markdown("""
    **Heteroskedastizität** ist einer der häufigsten Gründe, warum die schönen "Sterne" (★★★) 
    im R-Output **falsch** sein können!

    Das Problem: Die Varianz der Fehler ist **nicht konstant**. Die Daten "streuen" bei hohen 
    Werten stärker als bei niedrigen (oder umgekehrt) – der klassische **"Trichter-Effekt"**.
    """)

    col_hetero1, col_hetero2 = st.columns([1.5, 1])

    with col_hetero1:
        # Trichter-Vergleich
        np.random.seed(42)
        x_demo = np.linspace(1, 10, 100)
        X_demo = sm.add_constant(x_demo)
    
        # Homoskedastizität
        noise_homo = np.random.normal(0, 2.0, 100)
        y_homo = 2 + 1.5 * x_demo + noise_homo
        model_homo = sm.OLS(y_homo, X_demo).fit()
    
        # Heteroskedastizität
        noise_hetero = np.random.normal(0, 0.8 * x_demo, 100)
        y_hetero = 2 + 1.5 * x_demo + noise_hetero
        model_hetero = sm.OLS(y_hetero, X_demo).fit()
    
        fig_trichter = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '✅ Homoskedastizität (Ideal)<br>Gleichmässiger Schlauch',
                '⚠️ Heteroskedastizität (Problem)<br>Trichter-Effekt!',
                'Residual-Plot<br>✅ Wolke ohne Muster',
                'Residual-Plot<br>⚠️ Typische Trichterform!'
            ),
            vertical_spacing=0.12
        )
    
        # Homo Scatter
        y_pred_homo = model_homo.predict(X_demo)
        fig_trichter.add_trace(
            go.Scatter(x=np.concatenate([x_demo, x_demo[::-1]]),
                      y=np.concatenate([y_pred_homo + 4, (y_pred_homo - 4)[::-1]]),
                      fill='toself', fillcolor='rgba(0,255,0,0.15)',
                      line=dict(width=0), showlegend=False),
            row=1, col=1
        )
        fig_trichter.add_trace(
            go.Scatter(x=x_demo, y=y_homo, mode='markers',
                      marker=dict(size=4, color='green', opacity=0.6),
                      showlegend=False),
            row=1, col=1
        )
        fig_trichter.add_trace(
            go.Scatter(x=x_demo, y=y_pred_homo, mode='lines',
                      line=dict(color='black', width=2),
                      showlegend=False),
            row=1, col=1
        )
    
        # Homo Residual
        fig_trichter.add_trace(
            go.Scatter(x=y_pred_homo, y=model_homo.resid, mode='markers',
                      marker=dict(size=4, color='green', opacity=0.6),
                      showlegend=False),
            row=2, col=1
        )
        fig_trichter.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
    
        # Hetero Scatter
        y_pred_hetero = model_hetero.predict(X_demo)
        fig_trichter.add_trace(
            go.Scatter(x=np.concatenate([x_demo, x_demo[::-1]]),
                      y=np.concatenate([y_pred_hetero + (0.8*x_demo)*2,
                                      (y_pred_hetero - (0.8*x_demo)*2)[::-1]]),
                      fill='toself', fillcolor='rgba(255,0,0,0.15)',
                      line=dict(width=0), showlegend=False),
            row=1, col=2
        )
        fig_trichter.add_trace(
            go.Scatter(x=x_demo, y=y_hetero, mode='markers',
                      marker=dict(size=4, color='red', opacity=0.6),
                      showlegend=False),
            row=1, col=2
        )
        fig_trichter.add_trace(
            go.Scatter(x=x_demo, y=y_pred_hetero, mode='lines',
                      line=dict(color='black', width=2),
                      showlegend=False),
            row=1, col=2
        )
    
        # Hetero Residual
        fig_trichter.add_trace(
            go.Scatter(x=y_pred_hetero, y=model_hetero.resid, mode='markers',
                      marker=dict(size=4, color='red', opacity=0.6),
                      showlegend=False),
            row=2, col=2
        )
        fig_trichter.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=2)
        
        # Annotations for hetero residual plot
        fig_trichter.add_annotation(
            x=4, y=1, text='Kleine Fehler',
            showarrow=True, arrowhead=2, arrowsize=1, arrowcolor='red',
            ax=6, ay=8, font=dict(size=10, color='red'),
            row=2, col=2
        )
        fig_trichter.add_annotation(
            x=14, y=8, text='Grosse Fehler',
            showarrow=True, arrowhead=2, arrowsize=1, arrowcolor='red',
            ax=12, ay=12, font=dict(size=10, color='red'),
            row=2, col=2
        )
    
        # Update axes
        fig_trichter.update_yaxes(title_text="Y", row=1, col=1, showgrid=True, gridcolor='lightgray')
        fig_trichter.update_yaxes(title_text="Residuen", row=2, col=1, showgrid=True, gridcolor='lightgray')
        fig_trichter.update_xaxes(title_text="Fitted Values", row=2, col=1, showgrid=True, gridcolor='lightgray')
        fig_trichter.update_yaxes(row=1, col=2, showgrid=True, gridcolor='lightgray')
        fig_trichter.update_yaxes(title_text="Residuen", row=2, col=2, showgrid=True, gridcolor='lightgray')
        fig_trichter.update_xaxes(title_text="Fitted Values", row=2, col=2, showgrid=True, gridcolor='lightgray')
    
        fig_trichter.update_layout(
            title_text='🔍 Diagnose: Der Blick auf die Residuen',
            title_font_size=16,
            height=800,
            showlegend=False
        )
    
        st.plotly_chart(fig_trichter, use_container_width=True)
        
    with col_hetero2:
        st.error("""
        ### ⚠️ Warum ist das schlimm?
    
        Bei Heteroskedastizität:
        - Standardfehler **zu klein** berechnet
        - t-Werte **zu gross**
        - p-Werte **zu klein**
    
        **→ Falsche Sterne! ★★★**
    
        Du glaubst, etwas ist signifikant, 
        aber die Unsicherheit ist viel grösser!
        """)
    
        st.success("""
        ### ✅ Die Lösung
    
        **Robuste Standardfehler (HC3):**
    
        ```python
        model.get_robustcov_results(
            cov_type='HC3'
        ).summary()
        ```
    
        Die korrigierten Standardfehler sind 
        grösser → ehrlichere p-Werte!
        """)

    # Live-Vergleich Normal vs. Robust
    st.markdown('<p class="subsection-header">📊 Live-Vergleich: Normal vs. Robuste Standardfehler</p>', unsafe_allow_html=True)

    try:
        model_robust = model.get_robustcov_results(cov_type='HC3')
    
        col_norm, col_robust = st.columns(2)
    
        with col_norm:
            st.markdown("### 🔍 Normale OLS")
            st.metric("SE(b₁)", f"{model.bse[1]:.6f}")
            st.metric("t-Wert", f"{model.tvalues[1]:.3f}")
            st.metric("p-Wert", f"{model.pvalues[1]:.6f}")
            st.metric("Signifikanz", get_signif_stars(model.pvalues[1]))
    
        with col_robust:
            se_diff = ((model_robust.bse[1] - model.bse[1]) / model.bse[1]) * 100
            st.markdown("### 🛡️ Robuste HC3")
            st.metric("SE(b₁)", f"{model_robust.bse[1]:.6f}", delta=f"{se_diff:+.1f}%")
            st.metric("t-Wert", f"{model_robust.tvalues[1]:.3f}")
            st.metric("p-Wert", f"{model_robust.pvalues[1]:.6f}")
            st.metric("Signifikanz", get_signif_stars(model_robust.pvalues[1]))
    
        if get_signif_stars(model.pvalues[1]) != get_signif_stars(model_robust.pvalues[1]):
            st.warning("🚨 **Signifikanz-Unterschied!** Die normalen Sterne waren zu optimistisch!")
        else:
            st.success("✅ Beide Methoden zeigen gleiche Signifikanz.")
    except Exception as e:
        st.error(f"Robuste SE konnten nicht berechnet werden: {e}")

    # =========================================================
    # KAPITEL 6: FAZIT
    # =========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">6.0 Fazit und Ausblick</p>', unsafe_allow_html=True)

    col_fazit1, col_fazit2 = st.columns([2, 1])

    with col_fazit1:
        st.markdown(f"""
        Wir begannen mit einer einfachen **Frage**: Wie hängen {x_label} und {y_label} zusammen?
    
        Durch einen rigorosen Prozess aus:
        1. **Modellformulierung** (Y = β₀ + β₁X + ε)
        2. **Parameterschätzung** mittels OLS (b₁ = {b1:.4f})
        3. **Güteprüfung** (R² = {model.rsquared:.1%}, sₑ = {se_regression:.3f})
        4. **Statistische Inferenz** (t = {t_val:.2f}, p = {model.pvalues[1]:.4g})
    
        haben wir eine **quantifizierbare, vertrauenswürdige Antwort** mit einem bekannten Grad 
        an Sicherheit entwickelt.
    
        **Diese Reise von der Frage zur validierten Erkenntnis ist die Essenz der angewandten Statistik.**
        """)
    
        st.info("""
        **Korrelation vs. Regression – Der Zusammenhang:**
    
        Bei einfacher linearer Regression gilt: **R² = r²**
    
        Diese elegante Beziehung ist einzigartig für die einfache Regression und 
        gilt **nicht** mehr in der multiplen Regression!
        """)

    with col_fazit2:
        # Zusammenfassende Kennzahlen
        st.markdown("### 📋 Zusammenfassung")
    
        st.metric("Steigung b₁", f"{b1:.4f}", help=f"Veränderung in Y pro Einheit X")
        st.metric("R²", f"{model.rsquared:.2%}", help="Erklärte Varianz")
        st.metric("p-Wert", f"{model.pvalues[1]:.4g}", help="Signifikanz der Steigung")
        x_example = np.percentile(x, 75)
        st.metric(f"Prognose (X={x_example:.1f})", f"{b0 + b1*x_example:.2f} {y_unit}")

    # 3D Visualisierung der bedingten Verteilung
    st.markdown("---")
    st.markdown('<p class="subsection-header">🌊 Bonusgrafik: Die bedingte Verteilung f(y|x)</p>', unsafe_allow_html=True)

    col_3d1, col_3d2 = st.columns([2, 1])

    with col_3d1:
        fig_3d = go.Figure()
    
        x_line = np.linspace(float(x.min()), float(x.max()), 100)
        y_line = b0 + b1 * x_line
        
        # Regression line at z=0
        fig_3d.add_trace(go.Scatter3d(
            x=x_line, y=y_line, z=np.zeros_like(x_line),
            mode='lines',
            line=dict(color='blue', width=4),
            name='E(y|x)'
        ))
    
        x_points = np.linspace(float(x.min()) + 0.5, float(x.max()) - 0.5, 5)
        # Create a color palette similar to plasma
        plasma_colors = ['#0d0887', '#7e03a8', '#cc4778', '#f89540', '#f0f921']
    
        for i, x_point in enumerate(x_points):
            y_exp = b0 + b1 * x_point
            sigma = se_regression * 1.5
            y_range = np.linspace(y_exp - 3*sigma, y_exp + 3*sigma, 80)
            density = stats.norm.pdf(y_range, y_exp, sigma)
            density = density / density.max() * 1.5
        
            # Distribution curve
            fig_3d.add_trace(go.Scatter3d(
                x=np.full_like(y_range, x_point), y=y_range, z=density,
                mode='lines',
                line=dict(color=plasma_colors[i], width=3),
                showlegend=False
            ))
            
            # Point on regression line
            fig_3d.add_trace(go.Scatter3d(
                x=[x_point], y=[y_exp], z=[0],
                mode='markers',
                marker=dict(size=6, color=plasma_colors[i]),
                showlegend=False
            ))
    
        # Data points at z=0
        fig_3d.add_trace(go.Scatter3d(
            x=x, y=y, z=np.zeros(len(x)),
            mode='markers',
            marker=dict(size=5, color='green', opacity=0.6, symbol='diamond'),
            name='Daten'
        ))
    
        fig_3d.update_layout(
            title='Bedingte Verteilung: Fuer jeden X-Wert gibt es eine<br>Verteilung moeglicher Y-Werte',
            scene=dict(
                xaxis_title=f'X ({x_label})',
                yaxis_title=f'Y ({y_label})',
                zaxis_title='f(y|x)',
                camera=dict(eye=dict(x=1.5, y=-1.8, z=1.2))
            ),
            height=600,
            showlegend=True
        )
    
        st.plotly_chart(fig_3d, use_container_width=True)
        
    with col_3d2:
        st.latex(r"Y_i | X_i = x \sim N(\beta_0 + \beta_1 x, \sigma^2)")
    
        st.markdown(f"""
        **Das zeigt diese Grafik:**
    
        - Für jeden X-Wert gibt es eine **Normalverteilung** möglicher Y-Werte
        - Der **Mittelwert** dieser Verteilung liegt auf der Regressionsgeraden
        - Die **Breite** entspricht σ ≈ sₑ = {se_regression:.3f}
    
        Bei **Homoskedastizität**: Alle Glocken gleich breit
        Bei **Heteroskedastizität**: Glocken werden breiter!
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px; padding: 20px;'>
    📖 Umfassender Leitfaden zur Linearen Regression | 
    Von der Frage zur validierten Erkenntnis |
    Erstellt mit Streamlit & statsmodels
</div>
""", unsafe_allow_html=True)
