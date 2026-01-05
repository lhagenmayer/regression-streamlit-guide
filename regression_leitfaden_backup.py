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
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import warnings
import streamlit as st
import plotly.graph_objects as go

# --- Hilfsfunktion für typ-sichere Pandas-Zugriffe ---
def safe_scalar(val):
    """Konvertiert Series/ndarray zu Skalar, falls nötig."""
    if isinstance(val, (pd.Series, np.ndarray)):
        return float(val.iloc[0] if hasattr(val, 'iloc') else val[0])
    return float(val)

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


def create_r_output_display(model, feature_name="X"):
    """Erzeugt einen textbasierten R-Style Output als einheitliche Quelle."""
    resid = model.resid
    q = np.percentile(resid, [0, 25, 50, 75, 100])
    params = model.params
    bse = model.bse
    tvals = model.tvalues
    pvals = model.pvalues
    rse = np.sqrt(model.mse_resid)
    df_resid = int(model.df_resid)
    df_model = int(model.df_model)

    output_text = f"""Python Replikation des R-Outputs:
summary(lm_model)
===================================================
Residuals:
    Min      1Q  Median      3Q     Max
{q[0]:7.4f} {q[1]:7.4f} {q[2]:7.4f} {q[3]:7.4f} {q[4]:7.4f}

Coefficients:
             Estimate Std.Err  t val  Pr(>|t|)    
(Intercept) {params[0]:9.4f} {bse[0]:8.4f} {tvals[0]:7.2f} {pvals[0]:10.4g} {get_signif_stars(pvals[0])}
{feature_name:<13}{params[1]:9.4f} {bse[1]:8.4f} {tvals[1]:7.2f} {pvals[1]:10.4g} {get_signif_stars(pvals[1])}
---
Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: {rse:.4f} on {df_resid} degrees of freedom
Multiple R-squared:  {model.rsquared:.4f},    Adjusted R-squared: {model.rsquared_adj:.4f}
F-statistic: {model.fvalue:.1f} on {df_model} and {df_resid} DF,  p-value: {model.f_pvalue:.4g}
"""
    return output_text


def create_r_output_figure(model, feature_name="X", figsize=(16, 12)):
    """Verpackt den textbasierten R-Style Output in eine Plotly-Figur."""
    output_text = create_r_output_display(model, feature_name=feature_name)

    fig = go.Figure()
    fig.add_annotation(
        x=0,
        y=1,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        align="left",
        text=f"<pre style='font-size:14px; font-family:monospace;'>{output_text}</pre>",
        showarrow=False,
    )

    fig.update_layout(
        template="plotly_white",
        width=figsize[0] * 50,
        height=figsize[1] * 50,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig

# ---------------------------------------------------------
# SIDEBAR - INTERAKTIVE PARAMETER
# ---------------------------------------------------------
st.sidebar.markdown("# 🎛️ Parameter")

# === NAVIGATION ===
st.sidebar.markdown("---")
st.sidebar.markdown("## 📍 Navigation")
nav_options = [
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
selected_chapter = st.sidebar.radio("Kapitel auswählen:", nav_options, index=0, label_visibility="collapsed")

st.sidebar.markdown("---")

# === DATENSATZ-AUSWAHL ===
st.sidebar.markdown("## 📊 Datensatz")
dataset_choice = st.sidebar.selectbox(
    "Datensatz wählen:",
    ["🏪 Elektronikmarkt (simuliert)", "🏙️ Städte-Umsatzstudie (75 Städte)"],
    index=0
)

# === GEMEINSAME PARAMETER-SEKTION ===
st.sidebar.markdown("---")
st.sidebar.markdown("## 🎛️ Daten-Parameter")

if dataset_choice == "🏪 Elektronikmarkt (simuliert)":
    # X-Variable als Dropdown (nur eine Option verfügbar)
    x_variable_options = ["Verkaufsfläche (100qm)"]
    x_variable = st.sidebar.selectbox(
        "X-Variable (Prädiktor):",
        x_variable_options,
        index=0,
        help="Beim simulierten Datensatz ist nur die Verkaufsfläche als Prädiktor verfügbar."
    )
    
    st.sidebar.markdown("**Stichproben-Eigenschaften:**")
    n = st.sidebar.slider("Anzahl Beobachtungen (n)", min_value=8, max_value=50, value=12, step=1)
    
    st.sidebar.markdown("**Wahre Parameter (bekannt bei Simulation):**")
    true_intercept = st.sidebar.slider("Wahrer β₀ (Intercept)", min_value=-1.0, max_value=3.0, value=0.6, step=0.1)
    true_beta = st.sidebar.slider("Wahre Steigung β₁", min_value=0.1, max_value=1.5, value=0.52, step=0.01)
    
    st.sidebar.markdown("**Zufallskomponente:**")
    noise_level = st.sidebar.slider("Rauschen (σ)", min_value=0.1, max_value=1.5, value=0.4, step=0.05)
    seed = st.sidebar.number_input("Random Seed", min_value=1, max_value=999, value=42)
    
    # Simulierte Daten generieren
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
    
else:  # Städte-Umsatzstudie
    # X-Variable als Dropdown (zwei Optionen verfügbar)
    x_variable_options = ["Werbung (CHF1000)", "Preis (CHF)"]
    x_variable = st.sidebar.selectbox(
        "X-Variable (Prädiktor):",
        x_variable_options,
        index=0,
        help="Einfache Regression: Nur EIN Prädiktor → größerer Fehlerterm (didaktisch wertvoll!)"
    )
    
    st.sidebar.markdown("**Stichproben-Info:**")
    st.sidebar.info("n = 75 Städte (fixiert)")
    n = 75
    
    # Städte-Datensatz generieren (basierend auf Screenshot-Statistiken)
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
        
        ⚠️ **Didaktisch:** Nur EIN Prädiktor → großer Fehlerterm 
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
        
        ⚠️ **Didaktisch:** Nur EIN Prädiktor → großer Fehlerterm 
        (Preis fehlt als Erklärungsvariable!)
        """
    
    context_title = "Städte-Umsatzstudie"
    has_true_line = False
    true_intercept = 0  # Nicht bekannt bei echten Daten
    true_beta = 0
    seed = 42  # Fester Seed für konsistente ANOVA-Daten

st.sidebar.markdown("---")
st.sidebar.markdown("## 🔧 Anzeigeoptionen")
show_formulas = st.sidebar.checkbox("Formeln anzeigen", value=True)
show_true_line = st.sidebar.checkbox("Wahre Linie zeigen", value=has_true_line) if has_true_line else False

# ---------------------------------------------------------
# MODELL & KENNZAHLEN BERECHNEN (Einheitlich für alle Datensätze)
# ---------------------------------------------------------
# Als DataFrame für bessere Darstellung
df = pd.DataFrame({
    x_label: x,
    y_label: y
})

# Modell fitten
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
y_pred = model.predict(X)
y_mean = np.mean(y)

# Alle Kennzahlen berechnen
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
    
    > 🚨 **Klassischer Fehler (Omitted Variable Bias):** Dass wir Y aus X vorhersagen können,
    > beweist NICHT, dass X die Ursache für Y ist! Möglicherweise gibt es eine dritte Variable Z,
    > die sowohl X als auch Y beeinflusst (Scheinkorrelation).
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
        fig_joint_3d = plt.figure(figsize=(16, 6))
        
        # 1. 3D Surface Plot der gemeinsamen Verteilung
        ax1 = fig_joint_3d.add_subplot(131, projection='3d')
        surf = ax1.plot_surface(X_grid, Y_grid, Z, cmap='Blues', alpha=0.8, 
                               linewidth=0, antialiased=True)
        ax1.set_xlabel('X', fontsize=10)
        ax1.set_ylabel('Y', fontsize=10)
        ax1.set_zlabel('f(X,Y)', fontsize=10)
        ax1.set_title(f'Gemeinsame Verteilung\nρ = {demo_corr:.2f}', fontsize=12, fontweight='bold')
        ax1.view_init(elev=25, azim=-45)
        
        # Stichprobe als Punkte auf z=0
        np.random.seed(42)
        sample = np.random.multivariate_normal(mean, cov_matrix, 100)
        ax1.scatter(sample[:, 0], sample[:, 1], np.zeros(100), alpha=0.3, s=10, c='red')
        
        # 2. Randverteilung als 3D
        ax2 = fig_joint_3d.add_subplot(132, projection='3d')
        x_marg = np.linspace(-3, 3, 100)
        y_marg_pdf = stats.norm.pdf(x_marg, 0, 1)
        ax2.plot(x_marg, y_marg_pdf, zs=0, zdir='z', color='blue', linewidth=2)
        ax2.plot(x_marg, np.zeros_like(x_marg), y_marg_pdf, color='blue', alpha=0.5, linewidth=2)
        for i in range(0, len(x_marg), 5):
            ax2.plot([x_marg[i], x_marg[i]], [0, 0], [0, y_marg_pdf[i]], 'b-', alpha=0.3)
        ax2.set_xlabel('X', fontsize=10)
        ax2.set_ylabel('', fontsize=10)
        ax2.set_zlabel('f_X(x)', fontsize=10)
        ax2.set_title('Randverteilung f_X(x)', fontsize=12, fontweight='bold')
        ax2.view_init(elev=20, azim=-60)
        
        # 3. Bedingte Verteilung als 3D-Schnitt
        ax3 = fig_joint_3d.add_subplot(133, projection='3d')
        x_cond = 1.0
        cond_mean = demo_corr * x_cond
        cond_var = max(1 - demo_corr**2, 0.01)
        cond_std = np.sqrt(cond_var)
        
        y_cond_grid = np.linspace(-3, 3, 100)
        pdf_cond = stats.norm.pdf(y_cond_grid, cond_mean, cond_std)
        
        # Schnittebene bei X=1
        ax3.plot(np.full_like(y_cond_grid, x_cond), y_cond_grid, pdf_cond, 'g-', linewidth=2)
        for i in range(0, len(y_cond_grid), 5):
            ax3.plot([x_cond, x_cond], [y_cond_grid[i], y_cond_grid[i]], [0, pdf_cond[i]], 'g-', alpha=0.3)
        
        # Fläche füllen
        ax3.plot_surface(X_grid, Y_grid, Z, cmap='Blues', alpha=0.3)
        ax3.set_xlabel('X', fontsize=10)
        ax3.set_ylabel('Y', fontsize=10)
        ax3.set_zlabel('f(Y|X=1)', fontsize=10)
        ax3.set_title(f'Bedingte Verteilung\nE(Y|X=1) = {cond_mean:.2f}', fontsize=12, fontweight='bold')
        ax3.view_init(elev=20, azim=-60)
        
        plt.tight_layout()
        st.pyplot(fig_joint_3d)
        plt.close()
        
    else:
        # === 2D VERSION (Original) ===
        fig_joint, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # 1. Contour Plot der gemeinsamen Verteilung
        contour = axes[0].contourf(X_grid, Y_grid, Z, levels=20, cmap='Blues')
        axes[0].set_xlabel('X', fontsize=12)
        axes[0].set_ylabel('Y', fontsize=12)
        axes[0].set_title(f'Gemeinsame Verteilung f(X,Y)\nρ = {demo_corr:.2f}', fontsize=13, fontweight='bold')
        axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[0].axvline(0, color='gray', linestyle='--', alpha=0.5)
        
        # Stichprobe einzeichnen
        np.random.seed(42)
        sample = np.random.multivariate_normal(mean, cov_matrix, 100)
        axes[0].scatter(sample[:, 0], sample[:, 1], alpha=0.3, s=20, c='red', label='Stichprobe')
        
        # 2. Randverteilung f_X (oben projiziert)
        axes[1].fill_between(x_grid, stats.norm.pdf(x_grid, 0, 1), alpha=0.5, color='blue')
        axes[1].plot(x_grid, stats.norm.pdf(x_grid, 0, 1), 'b-', linewidth=2)
        axes[1].set_xlabel('X', fontsize=12)
        axes[1].set_ylabel('f_X(x)', fontsize=12)
        axes[1].set_title('Randverteilung f_X(x)\n(Marginale von X)', fontsize=13, fontweight='bold')
        axes[1].axvline(0, color='orange', linestyle='--', label=f'E(X) = 0')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Bedingte Verteilung f(Y|X=1)
        x_cond = 1.0
        cond_mean = demo_corr * x_cond
        cond_var = 1 - demo_corr**2
        cond_std = np.sqrt(max(cond_var, 0.01))
        
        y_cond_grid = np.linspace(-3, 3, 100)
        pdf_cond = stats.norm.pdf(y_cond_grid, cond_mean, cond_std)
        
        axes[2].fill_between(y_cond_grid, pdf_cond, alpha=0.5, color='green')
        axes[2].plot(y_cond_grid, pdf_cond, 'g-', linewidth=2)
        axes[2].axvline(cond_mean, color='red', linestyle='--', linewidth=2, label=f'E(Y|X={x_cond}) = {cond_mean:.2f}')
        axes[2].set_xlabel('Y', fontsize=12)
        axes[2].set_ylabel('f(Y|X=1)', fontsize=12)
        axes[2].set_title(f'Bedingte Verteilung f(Y|X=1)\nσ² = {cond_var:.2f}', fontsize=13, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig_joint)
        plt.close()

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
    # Visualisierung: Unabhängig vs. Abhängig
    fig_indep = plt.figure(figsize=(14, 6))
    
    np.random.seed(123)
    # Unabhängig (ρ=0) - 3D
    ax1 = fig_indep.add_subplot(121, projection='3d')
    x_ind = np.random.normal(0, 1, 200)
    y_ind = np.random.normal(0, 1, 200)
    z_ind = np.random.normal(0, 0.2, 200)  # Small z variation for 3D
    ax1.scatter(x_ind, y_ind, z_ind, alpha=0.5, c='gray', s=20)
    ax1.set_title('Unabhängig (ρ = 0)\n"Keine Struktur"', fontweight='bold', color='gray')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Dichte')
    ax1.view_init(elev=20, azim=-60)
    
    # Abhängig (ρ=0.8) - 3D
    ax2 = fig_indep.add_subplot(122, projection='3d')
    cov_dep = [[1, 0.8], [0.8, 1]]
    sample_dep = np.random.multivariate_normal([0, 0], cov_dep, 200)
    z_dep = np.random.normal(0, 0.2, 200)  # Small z variation
    ax2.scatter(sample_dep[:, 0], sample_dep[:, 1], z_dep, alpha=0.5, c='blue', s=20)
    ax2.set_title('Abhängig (ρ = 0.8)\n"Klare Struktur"', fontweight='bold', color='blue')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Dichte')
    ax2.view_init(elev=20, azim=-60)
    
    plt.tight_layout()
    st.pyplot(fig_indep)
    plt.close()

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
    | **εᵢ** | Zufällige Störgröße – alle anderen Einflüsse |
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
                 height=min(400, n * 35 + 50), use_container_width=True)

with col_data2:
    fig_scatter1 = plt.figure(figsize=(12, 8))
    ax1 = fig_scatter1.add_subplot(111, projection='3d')
    
    # 3D scatter with z-axis for density/frequency
    z_vals = np.random.normal(0, 0.3, len(x))  # Small z variation for 3D visualization
    ax1.scatter(x, y, z_vals, s=80, c='#1f77b4', alpha=0.7, edgecolor='white', linewidth=1.5, label='Datenpunkte')
    ax1.set_xlabel(x_label, fontsize=12)
    ax1.set_ylabel(y_label, fontsize=12)
    ax1.set_zlabel('Variation', fontsize=12)
    ax1.set_title('Schritt 1: Visualisierung der Rohdaten (3D)\n"Gibt es einen Zusammenhang?"', fontsize=14, fontweight='bold')
    
    # Mittelwertebenen
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    zlim = ax1.get_zlim()
    
    # Ebene für y-Mittelwert
    xx, zz = np.meshgrid(xlim, zlim)
    yy = np.full_like(xx, y_mean_val)
    ax1.plot_surface(xx, yy, zz, alpha=0.2, color='orange')
    
    # Schwerpunkt
    ax1.scatter([x_mean], [y_mean_val], [0], s=200, c='red', marker='X', label='Schwerpunkt (x̄, ȳ)')
    ax1.legend(loc='upper left')
    ax1.view_init(elev=20, azim=-60)
    
    plt.tight_layout()
    st.pyplot(fig_scatter1)
    plt.close()

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
        fig_cov = plt.figure(figsize=(14, 8))
        ax_cov = fig_cov.add_subplot(111, projection='3d')
        
        x_mean_val = x.mean()
        y_mean_val_local = y.mean()
        
        # Daten-Säulen
        for i in range(len(x)):
            dx = x[i] - x_mean_val
            dy = y[i] - y_mean_val_local
            product = dx * dy
            color = 'green' if product > 0 else 'red'
            
            # Vertikale Säule (von 0 bis |product|)
            if product > 0:
                ax_cov.bar3d(x[i], y[i], 0, 0.3, 0.3, product, 
                            color=color, alpha=0.7, edgecolor='darkgreen')
            else:
                ax_cov.bar3d(x[i], y[i], product, 0.3, 0.3, abs(product), 
                            color=color, alpha=0.7, edgecolor='darkred')
            
            # Datenpunkt oben
            ax_cov.scatter([x[i]], [y[i]], [product], s=100, c=color, edgecolor='white', linewidth=2)
        
        # Schwerpunkt
        ax_cov.scatter([x_mean_val], [y_mean_val_local], [0], s=300, c='black', marker='X', edgecolor='white', linewidth=2)
        
        ax_cov.set_xlabel(f'{x_label} (X)', fontsize=11)
        ax_cov.set_ylabel(f'{y_label} (Y)', fontsize=11)
        ax_cov.set_zlabel('(X - X̄)(Y - Ȳ)', fontsize=11)
        ax_cov.set_title('3D Kovarianz-Visualisierung: Säulenhöhe = Produkt der Abweichungen', 
                        fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig_cov)
        plt.close()
    else:
        # 3D Version: Bars instead of rectangles
        st.markdown("""
        Die Kovarianz misst, ob X und Y **gemeinsam** von ihren Mittelwerten abweichen:
        - Wenn X über dem Mittelwert ist UND Y auch → **positiver Beitrag** (grüne Säule nach oben)
        - Wenn X über dem Mittelwert ist ABER Y darunter → **negativer Beitrag** (rote Säule nach unten)
        """)
        
        fig_cov = plt.figure(figsize=(14, 9))
        ax_cov = fig_cov.add_subplot(111, projection='3d')
        
        x_mean_val = x.mean()
        y_mean_val_local = y.mean()
        
        # Plot data points and bars for covariance products
        for i in range(len(x)):
            dx = x[i] - x_mean_val
            dy = y[i] - y_mean_val_local
            product = dx * dy
            color = 'green' if product > 0 else 'red'
            
            # Bar from z=0 to product height
            if product >= 0:
                ax_cov.bar3d(x[i], y[i], 0, 0.3, 0.3, product, 
                            color=color, alpha=0.6, edgecolor='darkgreen', linewidth=1)
            else:
                ax_cov.bar3d(x[i], y[i], product, 0.3, 0.3, abs(product), 
                            color=color, alpha=0.6, edgecolor='darkred', linewidth=1)
            
            # Data point marker
            ax_cov.scatter(x[i], y[i], product, s=100, c=color, edgecolor='white', linewidth=2, zorder=10)
        
        # Mean planes
        xlim = (min(x) - 0.5, max(x) + 0.5)
        ylim = (min(y) - 0.5, max(y) + 0.5)
        
        # Zero plane (z=0)
        xx, yy = np.meshgrid([xlim[0], xlim[1]], [ylim[0], ylim[1]])
        zz = np.zeros_like(xx)
        ax_cov.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
        
        # Schwerpunkt
        ax_cov.scatter([x_mean_val], [y_mean_val_local], [0], s=300, c='black', marker='X', edgecolor='white', linewidth=2)
        ax_cov.set_xlabel(f'{x_label} (X)', fontsize=11)
        ax_cov.set_ylabel(f'{y_label} (Y)', fontsize=11)
        ax_cov.set_zlabel('(X - X̄)(Y - Ȳ)', fontsize=11)
        ax_cov.set_title('3D Kovarianz: Grüne Säulen addieren, rote subtrahieren', 
                        fontsize=12, fontweight='bold')
        ax_cov.view_init(elev=25, azim=-60)
        plt.tight_layout()
        st.pyplot(fig_cov)
        plt.close()

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
    # Verschiedene Korrelationen zeigen - 3D Version
    fig_corr_examples = plt.figure(figsize=(16, 10))
    
    example_corrs = [-0.95, -0.5, 0, 0.5, 0.8, 0.95]
    np.random.seed(42)
    
    for idx, r in enumerate(example_corrs):
        ax = fig_corr_examples.add_subplot(2, 3, idx+1, projection='3d')
        
        # Daten generieren
        if r == 0:
            ex_x = np.random.normal(0, 1, 100)
            ex_y = np.random.normal(0, 1, 100)
        else:
            cov_ex = [[1, r], [r, 1]]
            sample_ex = np.random.multivariate_normal([0, 0], cov_ex, 100)
            ex_x, ex_y = sample_ex[:, 0], sample_ex[:, 1]
        
        # Z-Werte (Höhe) basieren auf Dichte
        ex_z = np.random.normal(0, 0.3, 100)
        
        # Farbe basierend auf r
        if r > 0:
            color = plt.cm.Greens(0.3 + abs(r) * 0.7)
        elif r < 0:
            color = plt.cm.Reds(0.3 + abs(r) * 0.7)
        else:
            color = 'gray'
        
        ax.scatter(ex_x, ex_y, ex_z, alpha=0.6, c=[color], s=20)
        ax.set_title(f'r = {r:.2f}', fontsize=12, fontweight='bold', 
                    color='green' if r > 0 else 'red' if r < 0 else 'gray')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-1, 1)
        ax.set_xlabel('X', fontsize=9)
        ax.set_ylabel('Y', fontsize=9)
        ax.set_zlabel('Z', fontsize=9)
        ax.view_init(elev=20, azim=-60)
        
        # Regressionsebene wenn r ≠ 0
        if r != 0:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    # Simple plane fit
                    x_plane = np.array([-3, 3])
                    y_plane = np.array([-3, 3])
                    X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
                    Z_plane = r * X_plane * 0.1  # Simplified correlation surface
                    ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.2, color='gray')
            except (np.linalg.LinAlgError, ValueError):
                pass
    
    plt.suptitle('Der Korrelationskoeffizient r: Von -1 (perfekt negativ) bis +1 (perfekt positiv) in 3D', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_corr_examples)
    plt.close()

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
    
    Das ist identisch mit unserem späteren Bestimmtheitsmaß!
    """)

# --- t-Test für Korrelation ---
st.markdown('<p class="subsection-header">🔬 Signifikanztest für die Korrelation</p>', unsafe_allow_html=True)

col_ttest_corr1, col_ttest_corr2 = st.columns([2, 1])

with col_ttest_corr1:
    # t-Statistik für Korrelation
    t_corr = abs(corr_xy) * np.sqrt((n - 2) / max(1 - corr_xy**2, 0.001))
    p_corr = 2 * (1 - stats.t.cdf(t_corr, df=n-2))
    
    fig_t_corr = plt.figure(figsize=(12, 7))
    ax_t_corr = fig_t_corr.add_subplot(111, projection='3d')
    
    x_t = np.linspace(-5, max(5, t_corr + 1), 300)
    y_t = stats.t.pdf(x_t, df=n-2)
    z_t = np.zeros_like(x_t)  # Base at z=0
    
    # Plot as 3D line and surface
    ax_t_corr.plot(x_t, y_t, z_t, 'k-', linewidth=2, label=f't-Verteilung (df={n-2})')
    
    # Fill critical regions as vertical bars
    crit_idx = abs(x_t) > t_corr
    for i in range(len(x_t)-1):
        if crit_idx[i]:
            ax_t_corr.plot([x_t[i], x_t[i]], [0, y_t[i]], [0, 0], 'r-', alpha=0.3, linewidth=2)
    
    t_crit = stats.t.ppf(0.975, df=n-2)
    # Critical value planes
    ylim = (0, max(y_t)*1.1)
    ax_t_corr.plot([t_crit, t_crit], ylim, [0, 0], 'orange', linestyle='--', alpha=0.7, linewidth=2)
    ax_t_corr.plot([-t_crit, -t_crit], ylim, [0, 0], 'orange', linestyle='--', alpha=0.7, linewidth=2, label=f'Kritisch: ±{t_crit:.2f}')
    
    # Observed t-value
    ax_t_corr.plot([t_corr, t_corr], [0, max(y_t)], [0, 0], 'b-', linewidth=3, label=f't = {t_corr:.2f}')
    ax_t_corr.plot([-t_corr, -t_corr], [0, max(y_t)], [0, 0], 'b-', linewidth=2, alpha=0.5)
    
    ax_t_corr.set_xlabel('t-Wert', fontsize=12)
    ax_t_corr.set_ylabel('Dichte', fontsize=12)
    ax_t_corr.set_zlabel('Höhe', fontsize=12)
    ax_t_corr.set_title(f'H₀: ρ = 0 (kein Zusammenhang) vs. H₁: ρ ≠ 0 (3D)', fontsize=13, fontweight='bold')
    ax_t_corr.legend()
    ax_t_corr.view_init(elev=25, azim=-60)
    
    plt.tight_layout()
    st.pyplot(fig_t_corr)
    plt.close()

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
        
        # Visualisierung: Original vs. Ränge in 3D
        fig_spear = plt.figure(figsize=(16, 7))
        
        ax_orig = fig_spear.add_subplot(121, projection='3d')
        z_orig = np.random.normal(0, 0.2, len(x))
        ax_orig.scatter(x, y, z_orig, s=60, c='blue', alpha=0.7)
        ax_orig.set_title(f'Original-Daten (3D)\nPearson r = {corr_xy:.3f}', fontweight='bold')
        ax_orig.set_xlabel('X')
        ax_orig.set_ylabel('Y')
        ax_orig.set_zlabel('Variation')
        ax_orig.view_init(elev=20, azim=-60)
        
        # Ränge in 3D
        rank_x = stats.rankdata(x)
        rank_y = stats.rankdata(y)
        rank_diff = abs(rank_x - rank_y)
        
        ax_rank = fig_spear.add_subplot(122, projection='3d')
        ax_rank.scatter(rank_x, rank_y, rank_diff, s=60, c=rank_diff, cmap='Greens', alpha=0.7)
        ax_rank.set_title(f'Rang-Daten (3D)\nSpearman ρ = {rho_spearman:.3f}', fontweight='bold', color='green')
        ax_rank.set_xlabel('Rang(X)')
        ax_rank.set_ylabel('Rang(Y)')
        ax_rank.set_zlabel('|Rang-Diff|')
        ax_rank.view_init(elev=20, azim=-60)
        
        plt.tight_layout()
        st.pyplot(fig_spear)
        plt.close()
    
    with col_sp2:
        st.latex(r"r_s = 1 - \frac{6 \sum d_i^2}{n(n^2-1)}")
        st.markdown("wobei $d_i$ = Differenz der Ränge")
        
        st.metric("Spearman ρ", f"{rho_spearman:.4f}")
        st.metric("p-Wert", f"{p_spearman:.4f}")
        
        st.info("""
        **Wann Spearman?**
        - Ordinale Daten
        - Nicht-lineare monotone Zusammenhänge
        - Ausreißer-robust
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
    fig_ols = plt.figure(figsize=(14, 9))
    ax_ols = fig_ols.add_subplot(111, projection='3d')
    
    # Data points in 3D
    ax_ols.scatter(x, y, np.zeros_like(x), s=100, c='#1f77b4', alpha=0.7, edgecolor='white', linewidth=2, label='Datenpunkte')
    
    # OLS regression line
    ax_ols.plot(x, y_pred, np.zeros_like(x), 'r-', linewidth=3, label=f'OLS-Gerade: ŷ = {b0:.3f} + {b1:.3f}x')
    
    if show_true_line:
        true_line = true_intercept + true_beta * x
        ax_ols.plot(x, true_line, np.zeros_like(x), 'g--', linewidth=2, alpha=0.7, 
                   label=f'Wahre Gerade: y = {true_intercept:.2f} + {true_beta:.2f}x')
    
    # Residuen als 3D-Säulen (Quadrate werden zu Würfeln/Balken)
    for i in range(min(len(x), 10)):
        resid = y[i] - y_pred[i]
        # Vertikale Verbindung
        ax_ols.plot([x[i], x[i]], [y[i], y_pred[i]], [0, 0], 'r-', alpha=0.5, linewidth=1.5)
        # 3D Bar für Quadrat (Höhe = resid²)
        if abs(resid) > 0.05:
            size = min(abs(resid), 1.5)
            height = resid**2 * 2  # Scale for visibility
            ax_ols.bar3d(x[i]-size/2, min(y[i], y_pred[i]), 0, size, abs(resid), height,
                        color='red', alpha=0.3, edgecolor='darkred', linewidth=1)
    
    ax_ols.set_xlabel(x_label, fontsize=12)
    ax_ols.set_ylabel(y_label, fontsize=12)
    ax_ols.set_zlabel('SSE Beitrag', fontsize=12)
    ax_ols.set_title('OLS minimiert die Summe aller roten Volumen (3D: SSE)', fontsize=14, fontweight='bold')
    ax_ols.legend(loc='upper left')
    ax_ols.view_init(elev=20, azim=-60)
    
    plt.tight_layout()
    st.pyplot(fig_ols)
    plt.close()

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
    
    # Interpretation des Intercepts hinzufügen
    st.warning(f"""
    ### ⚠️ Interpretation von b₀ = {b0:.4f}
    
    **Vorsicht bei der Interpretation des Achsenabschnitts!**
    
    b₀ wäre der erwartete Y-Wert bei x = 0.
    
    **Problem:** In unseren Daten liegt x zwischen {x.min():.1f} und {x.max():.1f}.
    Der Wert x = 0 liegt **außerhalb des beobachteten Bereichs**!
    
    → Der Intercept ist oft nur ein **technischer Parameter** ohne sinnvolle inhaltliche Interpretation.
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
    fig_detail = plt.figure(figsize=(14, 8))
    ax_detail = fig_detail.add_subplot(111, projection='3d')
    
    # Fehlerterme als vertikale Linien
    for i in range(len(x)):
        ax_detail.plot([x[i], x[i]], [y_pred[i], y_pred[i]], [y_pred[i], y[i]], 
                      color='red', alpha=0.3, linewidth=1)
    
    # Regressionslinie in 3D
    ax_detail.plot(x, y_pred, y_pred, 'b-', linewidth=4, label='Regressionsgerade')
    
    # Datenpunkte
    ax_detail.scatter(x, y_pred, y, s=100, c='#1f77b4', alpha=0.7, 
                     edgecolor='white', linewidth=2, label='Beobachtungen (y)')
    
    # Konfidenzintervall als Band
    x_sorted_idx = np.argsort(x)
    x_sorted = x[x_sorted_idx]
    y_pred_sorted = y_pred[x_sorted_idx]
    iv_u_sorted = np.sort(iv_u)
    iv_l_sorted = np.sort(iv_l)
    
    ax_detail.plot(x_sorted, y_pred_sorted, iv_u_sorted, 'b--', linewidth=2, alpha=0.5, label='95% KI')
    ax_detail.plot(x_sorted, y_pred_sorted, iv_l_sorted, 'b--', linewidth=2, alpha=0.5)
    
    # Schwerpunkt
    ax_detail.scatter([x_mean], [y_mean_val], [y_mean_val], s=300, c='orange', 
                     marker='*', edgecolor='black', linewidth=2, label='Schwerpunkt')
    
    ax_detail.set_xlabel(f'{x_label} (X)', fontsize=11)
    ax_detail.set_ylabel('ŷ (Vorhersage)', fontsize=11)
    ax_detail.set_zlabel(f'{y_label} (Y)', fontsize=11)
    ax_detail.set_title('3D Anatomie: Datenpunkte, Regressionslinie & Fehlerterme', 
                       fontsize=12, fontweight='bold')
    ax_detail.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig_detail)
    plt.close()
else:
    # 3D Version: Anatomy with prediction surface
    fig_detail = plt.figure(figsize=(16, 10))
    ax_detail = fig_detail.add_subplot(111, projection='3d')

    # 1. Datenpunkte in 3D
    ax_detail.scatter(x, y, y, s=100, c='#1f77b4', alpha=0.7, edgecolor='white', linewidth=2, label='Beobachtungen yᵢ')

    # 2. Regressionsgerade in 3D
    ax_detail.plot(x, y_pred, y_pred, 'b-', linewidth=3, label=f'Modell: ŷ = {b0:.2f} + {b1:.2f}x')

    # 3. Prognoseintervall als 3D-Band
    ax_detail.plot(x, iv_l, iv_l, 'b--', linewidth=1.5, alpha=0.5, label='95% PI')
    ax_detail.plot(x, iv_u, iv_u, 'b--', linewidth=1.5, alpha=0.5)
    
    # 4. Residuen als vertikale Linien
    for i in range(0, len(x), max(1, len(x)//15)):  # Show subset for clarity
        ax_detail.plot([x[i], x[i]], [y[i], y_pred[i]], [y[i], y_pred[i]], 'r-', alpha=0.4, linewidth=1.5)
    
    # 5. Epsilon-Annotation (größtes Residuum)
    resid_abs = np.abs(residuals)
    idx_eps = np.argmax(resid_abs)
    if idx_eps > len(x) - 3:
        idx_eps = len(x) // 2
    
    ax_detail.plot([x[idx_eps], x[idx_eps]], [y[idx_eps], y_pred[idx_eps]], [y[idx_eps], y_pred[idx_eps]], 
                  'r-', linewidth=3, label=f'εᵢ = {residuals[idx_eps]:.2f}')
    
    # 6. Schwerpunkt
    ax_detail.scatter([x_mean], [y_mean_val], [y_mean_val], s=300, c='orange', marker='*', edgecolor='black', 
                     linewidth=2, label=f'Schwerpunkt ({x_mean:.1f}, {y_mean_val:.1f})')
    
    ax_detail.set_xlabel(x_label, fontsize=12)
    ax_detail.set_ylabel('ŷ (Vorhersage)', fontsize=12)
    ax_detail.set_zlabel(y_label + ' (Beobachtet)', fontsize=12)
    ax_detail.set_title('3D Anatomie: Steigung, Fehlerterm & Prognoseintervall', 
                       fontsize=14, fontweight='bold')
    ax_detail.legend(loc='upper left', fontsize=10)
    ax_detail.view_init(elev=20, azim=-60)
    
    plt.tight_layout()
    st.pyplot(fig_detail)
    plt.close()

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
**⚠️ Wichtige Unterscheidung: Konfidenz- vs. Prognoseintervall**

| **Konfidenzintervall (schmal)** | **Prognoseintervall (breit)** |
|--------------------------------|-------------------------------|
| Für den **Mittelwert** E(Y\|X=x) | Für eine **neue Beobachtung** yₙₑᵤ |
| Nur Unsicherheit der Linie | Unsicherheit der Linie **+** Streuung ε |
| Wird schmaler bei großem n | Bleibt breit (wegen σ²) |

👆 Der hellblaue Bereich oben ist das **Prognoseintervall** – deshalb ist es breiter!
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
    
    Dies zeigt auch: Je größer die Varianz von x (der Nenner), desto **präziser** kann b₁ geschätzt werden.
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
    
    Der Wert von Y bei x=0.
    
    ⚠️ **Vorsicht:** Extrapolation außerhalb 
    der Daten ist oft nicht sinnvoll!
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
Wir haben ein Modell – aber **wie gut** passt es wirklich? Die folgenden Gütemaße quantifizieren die Anpassung.
""")

# 4.1 Standardfehler der Regression
st.markdown('<p class="subsection-header">4.1 Standardfehler der Regression (sₑ): Die durchschnittliche Prognoseabweichung</p>', unsafe_allow_html=True)

col_se1, col_se2 = st.columns([2, 1])

with col_se1:
    fig_se = plt.figure(figsize=(14, 8))
    ax_se = fig_se.add_subplot(111, projection='3d')
    
    # Data points
    ax_se.scatter(x, y, np.zeros_like(x), s=80, c='#1f77b4', alpha=0.6, edgecolor='white')
    
    # Regression line
    ax_se.plot(x, y_pred, np.zeros_like(x), 'r-', linewidth=2.5, label='Regressionsgerade')
    
    # Confidence bands as 3D surfaces
    z_offset_1 = np.ones_like(x) * 0.5
    z_offset_2 = np.ones_like(x) * 1.0
    
    ax_se.plot(x, y_pred - se_regression, z_offset_1, 'r--', linewidth=1.5, alpha=0.6, label=f'±1·sₑ = ±{se_regression:.3f}')
    ax_se.plot(x, y_pred + se_regression, z_offset_1, 'r--', linewidth=1.5, alpha=0.6)
    ax_se.plot(x, y_pred - 2*se_regression, z_offset_2, 'r--', linewidth=1.5, alpha=0.4, label=f'±2·sₑ')
    ax_se.plot(x, y_pred + 2*se_regression, z_offset_2, 'r--', linewidth=1.5, alpha=0.4)
    
    # Fill between as vertical ribbons
    for i in range(len(x)-1):
        # Inner band
        xs = [x[i], x[i+1], x[i+1], x[i]]
        ys = [y_pred[i]-se_regression, y_pred[i+1]-se_regression, y_pred[i+1]+se_regression, y_pred[i]+se_regression]
        zs = [z_offset_1[i], z_offset_1[i+1], z_offset_1[i+1], z_offset_1[i]]
        if i % 3 == 0:  # Sample to reduce clutter
            ax_se.plot_surface(np.array([xs[:2], xs[2:]]), np.array([ys[:2], ys[2:]]), 
                             np.array([zs[:2], zs[2:]]), alpha=0.1, color='red')
    
    ax_se.set_xlabel(x_label, fontsize=12)
    ax_se.set_ylabel(y_label, fontsize=12)
    ax_se.set_zlabel('Konfidenz-Niveau', fontsize=12)
    ax_se.set_title(f'Der Standardfehler sₑ = {se_regression:.4f} zeigt die typische Streuung um die Linie (3D)',
                   fontsize=13, fontweight='bold')
    ax_se.legend(loc='upper left')
    ax_se.view_init(elev=20, azim=-60)
    
    plt.tight_layout()
    st.pyplot(fig_se)
    plt.close() 
                   fontsize=13, fontweight='bold')
    ax_se.legend(loc='upper left')
    ax_se.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig_se)
    plt.close()

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
    
    st.warning(f"""
    ### 🎯 Warum n-2 (Freiheitsgrade)?
    
    **Intuition:** Wir haben **2 "Informationen verbraucht"**, um die Linie festzulegen:
    - 1 für b₀ (Achsenabschnitt)
    - 1 für b₁ (Steigung)
    
    Diese fehlen uns nun für die Schätzung der Varianz σ².
    
    **Allgemein:** df = n - (Anzahl geschätzter Parameter)
    
    Bei einfacher Regression: df = {n} - 2 = **{n-2}**
    """)

# --- Unsicherheit der Koeffizienten s_b₀ und s_b₁ ---
st.markdown('<p class="subsection-header">4.1b Standardfehler der Koeffizienten: Die Unsicherheit von b₀ und b₁</p>', unsafe_allow_html=True)

st.markdown("""
Der Standardfehler **sₑ** beschreibt die Streuung der Punkte um die Linie. Aber wie **sicher** sind 
wir uns über die Steigung (b₁) und den Achsenabschnitt (b₀) selbst? Das zeigen uns **s_b₀** und **s_b₁**.
""")

col_sb1, col_sb2 = st.columns([2, 1])

with col_sb1:
    fig_sb = plt.figure(figsize=(16, 7))
    
    # Links: sₑ (Streuung um die Linie) - 3D
    ax_sb1 = fig_sb.add_subplot(121, projection='3d')
    ax_sb1.scatter(x, y, np.zeros_like(x), color='gray', alpha=0.5, s=50)
    ax_sb1.plot(x, y_pred, np.zeros_like(x), 'b-', linewidth=2.5, label='Unsere Schätzung')
    
    # Confidence bands as 3D ribbons
    z1 = np.ones_like(x) * 0.3
    z2 = np.ones_like(x) * 0.6
    ax_sb1.plot(x, y_pred - se_regression, z1, 'b--', linewidth=1.5, alpha=0.5, label=f'±1·sₑ = ±{se_regression:.3f}')
    ax_sb1.plot(x, y_pred + se_regression, z1, 'b--', linewidth=1.5, alpha=0.5)
    ax_sb1.plot(x, y_pred - 2*se_regression, z2, 'b--', linewidth=1.5, alpha=0.3, label=f'±2·sₑ')
    ax_sb1.plot(x, y_pred + 2*se_regression, z2, 'b--', linewidth=1.5, alpha=0.3)
    
    ax_sb1.set_title(f'sₑ = {se_regression:.4f}\n(Streuung der PUNKTE um die Linie)', fontsize=11, fontweight='bold')
    ax_sb1.legend(loc='upper left', fontsize=9)
    ax_sb1.set_xlabel(x_label)
    ax_sb1.set_ylabel(y_label)
    ax_sb1.set_zlabel('Unsicherheit')
    ax_sb1.view_init(elev=20, azim=-60)
    
    # Rechts: s_b₁ (Unsicherheit der Steigung) - 3D mit simulierten Flächen
    ax_sb2 = fig_sb.add_subplot(122, projection='3d')
    np.random.seed(456)
    x_sim = np.linspace(min(x), max(x), 100)
    
    # Draw simulated regression lines in 3D space
    for i in range(60):
        sim_slope = np.random.normal(b1, sb1)
        sim_intercept = np.random.normal(b0, sb0)
        y_sim = sim_intercept + sim_slope * x_sim
        z_sim = np.ones_like(x_sim) * (i / 200.0)  # Stack in z-direction
        ax_sb2.plot(x_sim, y_sim, z_sim, color='green', alpha=0.08, linewidth=0.8)
    
    ax_sb2.plot(x, y_pred, np.zeros_like(x), 'k-', linewidth=2.5, label='Unsere Schätzung')
    ax_sb2.scatter(x, y, np.zeros_like(x), color='gray', alpha=0.4, s=30)
    ax_sb2.set_title(f's_b₁ = {sb1:.4f}\n(Unsicherheit der STEIGUNG)', fontsize=11, fontweight='bold', color='darkgreen')
    ax_sb2.legend(loc='upper left', fontsize=9)
    ax_sb2.set_xlabel(x_label)
    ax_sb2.set_ylabel(y_label)
    ax_sb2.set_zlabel('Simulations-Ebene')
    ax_sb2.view_init(elev=20, azim=-60)
    
    plt.tight_layout()
    st.pyplot(fig_sb)
    plt.close()

with col_sb2:
    if show_formulas:
        st.markdown("### s_b₁ (Std. Error Steigung)")
        st.latex(r"s_{b_1} = \frac{s_e}{\sqrt{(n-1) \cdot s_x^2}} = \frac{s_e}{\sqrt{SS_x}}")
        st.latex(f"s_{{b_1}} = {sb1:.4f}")
        
        st.markdown("### s_b₀ (Std. Error Achsenabschnitt)")
        st.latex(r"s_{b_0} = s_e \sqrt{\frac{1}{n} + \frac{\bar{x}^2}{\sum(x_i - \bar{x})^2}}")
        st.latex(f"s_{{b_0}} = {sb0:.4f}")
        
        st.caption("""
        **Intuition für s_b₀:** Die Formel enthält x̄². 
        Je weiter der Mittelwert von Null entfernt ist, 
        desto unsicherer wird der Achsenabschnitt!
        """)
    
    st.warning(f"""
    **Wichtiger Unterschied:**
    - **sₑ = {se_regression:.4f}** → Wie stark streuen die **Punkte** um die Linie?
    - **s_b₁ = {sb1:.4f}** → Wie genau ist unsere geschätzte **Steigung**?
    - **s_b₀ = {sb0:.4f}** → Wie genau ist unser **Achsenabschnitt**?
    
    Die grünen Linien rechts zeigen: Mit anderen Stichproben hätten wir 
    etwas andere Steigungen bekommen!
    """)

# 4.2 Bestimmtheitsmaß R²
st.markdown('<p class="subsection-header">4.2 Bestimmtheitsmaß (R²): Der Anteil der erklärten Varianz</p>', unsafe_allow_html=True)

col_r2_1, col_r2_2 = st.columns([2, 1])

with col_r2_1:
    show_3d_var = st.toggle("📈 3D-Ansicht aktivieren (Varianz)", value=False, key="toggle_3d_var")
    
    if show_3d_var:
        # 3D Visualisierung: Würfel für SST, SSR, SSE
        fig_var = plt.figure(figsize=(15, 5))
        
        # Normalisierung für Visualisierung
        sst_norm = sst
        ssr_norm = ssr / sst_norm
        sse_norm = sse / sst_norm
        
        # 1. SST (Gesamtwürfel)
        ax1 = fig_var.add_subplot(131, projection='3d')
        # Würfel für SST
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ])
        faces = [
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[7], vertices[6], vertices[2], vertices[3]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]]
        ]
        ax1.add_collection3d(Poly3DCollection(faces, alpha=0.25, facecolor='orange', edgecolor='darkorange', linewidth=2))
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_zlim(0, 1)
        ax1.set_title(f'SST = {sst:.2f}\n(Gesamte Variation)', fontweight='bold', color='orange')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Varianz')
        
        # 2. SSR (Würfel mit ssr_norm Höhe)
        ax2 = fig_var.add_subplot(132, projection='3d')
        vertices_ssr = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, ssr_norm], [1, 0, ssr_norm], [1, 1, ssr_norm], [0, 1, ssr_norm]
        ])
        faces_ssr = [
            [vertices_ssr[0], vertices_ssr[1], vertices_ssr[5], vertices_ssr[4]],
            [vertices_ssr[7], vertices_ssr[6], vertices_ssr[2], vertices_ssr[3]],
            [vertices_ssr[0], vertices_ssr[3], vertices_ssr[7], vertices_ssr[4]],
            [vertices_ssr[1], vertices_ssr[2], vertices_ssr[6], vertices_ssr[5]],
            [vertices_ssr[0], vertices_ssr[1], vertices_ssr[2], vertices_ssr[3]],
            [vertices_ssr[4], vertices_ssr[5], vertices_ssr[6], vertices_ssr[7]]
        ]
        ax2.add_collection3d(Poly3DCollection(faces_ssr, alpha=0.25, facecolor='green', edgecolor='darkgreen', linewidth=2))
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_zlim(0, 1)
        ax2.set_title(f'SSR = {ssr:.2f}\n(Durch Modell erklärt)', fontweight='bold', color='green')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Varianz')
        
        # 3. SSE (Würfel mit sse_norm Höhe)
        ax3 = fig_var.add_subplot(133, projection='3d')
        vertices_sse = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, sse_norm], [1, 0, sse_norm], [1, 1, sse_norm], [0, 1, sse_norm]
        ])
        faces_sse = [
            [vertices_sse[0], vertices_sse[1], vertices_sse[5], vertices_sse[4]],
            [vertices_sse[7], vertices_sse[6], vertices_sse[2], vertices_sse[3]],
            [vertices_sse[0], vertices_sse[3], vertices_sse[7], vertices_sse[4]],
            [vertices_sse[1], vertices_sse[2], vertices_sse[6], vertices_sse[5]],
            [vertices_sse[0], vertices_sse[1], vertices_sse[2], vertices_sse[3]],
            [vertices_sse[4], vertices_sse[5], vertices_sse[6], vertices_sse[7]]
        ]
        ax3.add_collection3d(Poly3DCollection(faces_sse, alpha=0.25, facecolor='red', edgecolor='darkred', linewidth=2))
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.set_zlim(0, 1)
        ax3.set_title(f'SSE = {sse:.2f}\n(Unerklärt/Residuen)', fontweight='bold', color='red')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Varianz')
        
        plt.suptitle(f'3D Varianzzerlegung: SST = SSR + SSE → R² = {model.rsquared:.1%}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_var)
        plt.close()
    else:
        # 3D Version: 3D bars for SST, SSR, SSE
        fig_var = plt.figure(figsize=(16, 6))
        
        # SST - 3D bars from points to mean
        ax1 = fig_var.add_subplot(131, projection='3d')
        for i in range(len(x)):
            height = abs(y[i] - y_mean_val)
            ax1.bar3d(x[i], y_mean_val, 0, 0.3, 0, height, color='orange', alpha=0.6, edgecolor='darkorange')
        ax1.scatter(x, y, np.zeros_like(x), color='gray', alpha=0.6, s=60)
        # Mean plane
        xlim = (min(x)-0.5, max(x)+0.5)
        xx, yy = np.meshgrid([xlim[0], xlim[1]], [y_mean_val-0.5, y_mean_val+0.5])
        zz = np.zeros_like(xx)
        ax1.plot_surface(xx, yy, zz, alpha=0.2, color='orange')
        ax1.set_title(f'SST = {sst:.2f}\n(Gesamte Variation)', fontsize=12, fontweight='bold', color='orange')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Abweichung')
        ax1.view_init(elev=20, azim=-60)
        
        # SSR - 3D bars from pred to mean
        ax2 = fig_var.add_subplot(132, projection='3d')
        for i in range(len(x)):
            height = abs(y_pred[i] - y_mean_val)
            ax2.bar3d(x[i], min(y_pred[i], y_mean_val), 0, 0.3, 0, height, color='green', alpha=0.6, edgecolor='darkgreen')
        ax2.scatter(x, y, np.zeros_like(x), color='gray', alpha=0.3, s=60)
        ax2.plot(x, y_pred, np.zeros_like(x), color='blue', linewidth=3)
        ax2.set_title(f'SSR = {ssr:.2f}\n(Durch Modell erklärt)', fontsize=12, fontweight='bold', color='green')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Erklärte Var')
        ax2.view_init(elev=20, azim=-60)
        
        # SSE - 3D bars from points to pred
        ax3 = fig_var.add_subplot(133, projection='3d')
        for i in range(len(x)):
            height = abs(y[i] - y_pred[i])
            ax3.bar3d(x[i], min(y[i], y_pred[i]), 0, 0.3, 0, height, color='red', alpha=0.6, edgecolor='darkred')
        ax3.scatter(x, y, np.zeros_like(x), color='gray', alpha=0.6, s=60)
        ax3.plot(x, y_pred, np.zeros_like(x), color='blue', linewidth=3)
        ax3.set_title(f'SSE = {sse:.2f}\n(Unerklärt/Residuen)', fontsize=12, fontweight='bold', color='red')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Residuen')
        ax3.view_init(elev=20, azim=-60)
        
        plt.suptitle(f'3D Varianzzerlegung: SST = SSR + SSE → R² = {model.rsquared:.1%}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_var)
        plt.close()

with col_r2_2:
    if show_formulas:
        st.latex(r"R^2 = \frac{SSR}{SST} = 1 - \frac{SSE}{SST}")
        st.latex(f"R^2 = \\frac{{{ssr:.2f}}}{{{sst:.2f}}} = {model.rsquared:.4f}")
    
    # R² als 3D Balken
    fig_r2bar = plt.figure(figsize=(8, 6))
    ax_r2 = fig_r2bar.add_subplot(111, projection='3d')
    
    x_pos = np.array([0, 1, 2])
    y_pos = np.zeros(3)
    z_pos = np.zeros(3)
    dx = np.ones(3) * 0.5
    dy = np.ones(3) * 0.5
    dz = np.array([sst, ssr, sse])
    colors = ['gray', 'green', 'red']
    
    for i in range(3):
        ax_r2.bar3d(x_pos[i], y_pos[i], z_pos[i], dx[i], dy[i], dz[i], 
                   color=colors[i], alpha=0.7, edgecolor='black', linewidth=2)
    
    ax_r2.set_xticks(x_pos + dx/2)
    ax_r2.set_xticklabels(['SST\n(Total)', 'SSR\n(Erklärt)', 'SSE\n(Unerklärt)'])
    ax_r2.set_ylabel('')
    ax_r2.set_zlabel('Quadratsumme')
    ax_r2.set_title(f'R² = {model.rsquared:.1%} (3D)', fontweight='bold', fontsize=13)
    ax_r2.view_init(elev=20, azim=-60)
    
    plt.tight_layout()
    st.pyplot(fig_r2bar)
    plt.close()
    
    st.success(f"""
    **{model.rsquared:.1%}** der Varianz in Y 
    wird durch X erklärt!
    """)
    
    # Adjusted R² Erklärung hinzufügen
    st.info(f"""
    ### 📊 R² vs. Adjusted R²
    
    | Maß | Wert | Bedeutung |
    |-----|------|-----------|
    | **R²** | {model.rsquared:.4f} | Anteil erklärter Varianz |
    | **Adj. R²** | {model.rsquared_adj:.4f} | Mit "Strafterm" für Komplexität |
    
    **Warum Adj. R² < R²?**
    
    Das adjustierte R² **bestraft** zusätzliche Prädiktoren:
    
    $R^2_{{adj}} = 1 - \\frac{{SSE/(n-k-1)}}{{SST/(n-1)}}$
    
    Bei **einfacher Regression** (k=1) ist der Unterschied gering.
    Bei **multipler Regression** wichtig: Verhindert "Overfitting"!
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

# Die 5 Schritte des Hypothesentests
with st.expander("📚 Die 5 Schritte des Hypothesentests (Klausur-Schema)", expanded=False):
    st.markdown("""
    ### Das klassische Vorgehen vs. der Computer-Output
    
    In der Vorlesung lernen Sie das **5-Schritte-Schema**. Der Computer verkürzt diesen Prozess.
    Hier die Zuordnung:
    
    | Schritt | Manuelles Vorgehen | Im Computer-Output |
    |---------|-------------------|-------------------|
    | **1. Hypothesen** | H₀: β₁ = 0 vs. H₁: β₁ ≠ 0 | *Implizit für jeden Koeffizienten* |
    | **2. Signifikanzniveau** | α = 0.05 festlegen | *Standard, aber anpassbar* |
    | **3. Teststatistik** | t = b₁ / s_b₁ berechnen | **t val** Spalte im Output |
    | **4. Kritischer Wert** | t_krit aus Tabelle (df=n-2) | *Nicht direkt gezeigt* |
    | **5. Entscheidung** | \\|t\\| > t_krit → H₀ ablehnen | **p-Wert** und **Sterne ★★★** |
    
    ---
    
    ### 💡 Der p-Wert ersetzt Schritt 4 und 5!
    
    Statt den kritischen Wert nachzuschlagen, nutzen wir:
    
    - **p < 0.05** → H₀ ablehnen (signifikant)
    - **p ≥ 0.05** → H₀ nicht ablehnen (nicht signifikant)
    
    Die **Sterne** (★★★) zeigen das Signifikanzniveau direkt an:
    - ★★★ p < 0.001 (hochsignifikant)
    - ★★ p < 0.01
    - ★ p < 0.05
    - . p < 0.1 (marginal signifikant)
    """)

# Annahmen
st.markdown('<p class="subsection-header">📋 Voraussetzungen für valide Inferenz: Die Gauss-Markov Annahmen</p>', unsafe_allow_html=True)

st.markdown("""
Die OLS-Schätzung liefert **unverzerrte (erwartungstreue) Schätzer** für β₀ und β₁. 
Für die Durchführung von Hypothesentests werden aber weitere Annahmen benötigt:
""")

# Visualisierung aller 4 Annahmen: Korrekt vs. Verletzt (3D Version)
fig_assumptions = plt.figure(figsize=(16, 18))
np.random.seed(123)
n_demo = 100
x_demo = np.linspace(1, 10, n_demo)

# === ANNAHME 1: E(εᵢ|xᵢ) = 0 ===
# Korrekt: Fehler zufällig um 0
ax1 = fig_assumptions.add_subplot(4, 2, 1, projection='3d')
y_correct_1 = 2 + 0.5 * x_demo + np.random.normal(0, 1, n_demo)
z_jitter_1 = np.random.normal(0, 0.2, n_demo)
ax1.scatter(x_demo, y_correct_1, z_jitter_1, alpha=0.6, c='green', s=30)
ax1.plot(x_demo, 2 + 0.5 * x_demo, np.zeros_like(x_demo), 'b-', linewidth=2)
ax1.set_title('✅ (1) E(εᵢ|xᵢ) = 0\nFehler symmetrisch um Null', fontweight='bold', color='green')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Varianz')
ax1.view_init(elev=20, azim=-60)

# Verletzt: Systematischer Bias (z.B. quadratischer Term fehlt)
ax2 = fig_assumptions.add_subplot(4, 2, 2, projection='3d')
y_violated_1 = 2 + 0.5 * x_demo + 0.1 * (x_demo - 5)**2 + np.random.normal(0, 0.5, n_demo)
z_jitter_2 = np.random.normal(0, 0.2, n_demo)
ax2.scatter(x_demo, y_violated_1, z_jitter_2, alpha=0.6, c='red', s=30, label='Daten')
ax2.plot(x_demo, 2 + 0.5 * x_demo, np.zeros_like(x_demo), 'b-', linewidth=2, label='Lineares Modell')
ax2.plot(x_demo, 2 + 0.5 * x_demo + 0.1 * (x_demo - 5)**2, np.zeros_like(x_demo), 'g--', linewidth=2, label='Wahrer Zusammenhang')
ax2.set_title('❌ E(εᵢ|xᵢ) ≠ 0\nNicht-linearer Zusammenhang ignoriert', fontweight='bold', color='red')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Varianz')
ax2.legend(fontsize=7)
ax2.view_init(elev=20, azim=-60)

# === ANNAHME 2: Var(εᵢ) = σ² (Homoskedastizität) ===
# Korrekt: Konstante Varianz
ax3 = fig_assumptions.add_subplot(4, 2, 3, projection='3d')
y_correct_2 = 2 + 0.5 * x_demo + np.random.normal(0, 1, n_demo)
z_jitter_3 = np.random.normal(0, 0.2, n_demo)
ax3.scatter(x_demo, y_correct_2, z_jitter_3, alpha=0.6, c='green', s=30)
ax3.plot(x_demo, 2 + 0.5 * x_demo, np.zeros_like(x_demo), 'b-', linewidth=2)
ax3.set_title('✅ (2) Var(εᵢ) = σ² (konstant)\nHomoskedastizität', fontweight='bold', color='green')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Varianz')
ax3.view_init(elev=20, azim=-60)

# Verletzt: Varianz steigt mit X
ax4 = fig_assumptions.add_subplot(4, 2, 4, projection='3d')
hetero_noise = np.random.normal(0, 0.3 * x_demo, n_demo)
y_violated_2 = 2 + 0.5 * x_demo + hetero_noise
z_hetero = 0.1 * x_demo  # Increasing variance
ax4.scatter(x_demo, y_violated_2, z_hetero, alpha=0.6, c='red', s=30)
ax4.plot(x_demo, 2 + 0.5 * x_demo, np.zeros_like(x_demo), 'b-', linewidth=2)
ax4.set_title('❌ Var(εᵢ|xᵢ) = f(xᵢ)\nHeteroskedastizität (Trichter)', fontweight='bold', color='red')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Varianz (wächst!)')
ax4.view_init(elev=20, azim=-60)

# === ANNAHME 3: Cov(εᵢ, εⱼ) = 0 (Keine Autokorrelation) ===
# Korrekt: Unabhängige Fehler
ax5 = fig_assumptions.add_subplot(4, 2, 5, projection='3d')
y_correct_3 = 2 + 0.5 * x_demo + np.random.normal(0, 1, n_demo)
resid_correct_3 = y_correct_3 - (2 + 0.5 * x_demo)
z_color = resid_correct_3[1:]  # Next residual as height
x_idx = np.arange(n_demo-1)
ax5.scatter(x_idx, resid_correct_3[:-1], z_color, c=z_color, cmap='coolwarm', s=30, alpha=0.7)
# Zero plane
xx_zero, yy_zero = np.meshgrid([0, n_demo-1], [0, 0])
zz_zero = np.zeros_like(xx_zero)
ax5.plot_surface(xx_zero, yy_zero, zz_zero, alpha=0.3, color='gray')
ax5.set_title('✅ (3) Cov(εᵢ, εⱼ) = 0\nKeine Autokorrelation', fontweight='bold', color='green')
ax5.set_xlabel('Beobachtung i')
ax5.set_ylabel('Residuum eᵢ')
ax5.set_zlabel('eᵢ₊₁')
ax5.view_init(elev=20, azim=-60)

# Verletzt: Autokorrelierte Fehler (z.B. Zeitreihe)
ax6 = fig_assumptions.add_subplot(4, 2, 6, projection='3d')
auto_error = np.zeros(n_demo)
auto_error[0] = np.random.normal(0, 1)
for i in range(1, n_demo):
    auto_error[i] = 0.8 * auto_error[i-1] + np.random.normal(0, 0.5)
y_violated_3 = 2 + 0.5 * x_demo + auto_error
x_time = np.arange(n_demo)
z_auto = np.abs(auto_error) * 0.5  # Height shows autocorrelation strength
ax6.plot(x_time, auto_error, z_auto, 'r-', alpha=0.7, linewidth=1.5)
ax6.scatter(x_time, auto_error, z_auto, c='red', s=20, alpha=0.6)
xx_zero2, yy_zero2 = np.meshgrid([0, n_demo], [0, 0])
zz_zero2 = np.zeros_like(xx_zero2)
ax6.plot_surface(xx_zero2, yy_zero2, zz_zero2, alpha=0.3, color='gray')
ax6.set_title('❌ Cov(εᵢ, εⱼ) ≠ 0\nAutokorrelation (Muster in Residuen)', fontweight='bold', color='red')
ax6.set_xlabel('Zeit/Beobachtung')
ax6.set_ylabel('Residuum')
ax6.set_zlabel('Korrelation')
ax6.view_init(elev=20, azim=-60)

# === ANNAHME 4: ε ~ N(0, σ²) (Normalverteilung) ===
# Korrekt: Normalverteilte Residuen
ax7 = fig_assumptions.add_subplot(4, 2, 7, projection='3d')
normal_resid = np.random.normal(0, 1, n_demo)
hist_vals, bin_edges = np.histogram(normal_resid, bins=20, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
widths = bin_edges[1:] - bin_edges[:-1]
for i, (center, height, width) in enumerate(zip(bin_centers, hist_vals, widths)):
    ax7.bar3d(center, 0, 0, width*0.8, 0.1, height, color='green', alpha=0.7, edgecolor='black')
x_norm = np.linspace(-4, 4, 100)
ax7.plot(x_norm, np.zeros_like(x_norm), stats.norm.pdf(x_norm, 0, 1), 'b-', linewidth=2)
ax7.set_title('✅ (4) ε ~ N(0, σ²)\nNormalverteilte Residuen', fontweight='bold', color='green')
ax7.set_xlabel('Residuum')
ax7.set_ylabel('')
ax7.set_zlabel('Dichte')
ax7.view_init(elev=20, azim=-60)

# Verletzt: Schiefe/Nicht-normale Verteilung
ax8 = fig_assumptions.add_subplot(4, 2, 8, projection='3d')
skewed_resid = np.random.exponential(1, n_demo) - 1  # Rechtsschiefe Verteilung
hist_vals_s, bin_edges_s = np.histogram(skewed_resid, bins=20, density=True)
bin_centers_s = (bin_edges_s[:-1] + bin_edges_s[1:]) / 2
widths_s = bin_edges_s[1:] - bin_edges_s[:-1]
for i, (center, height, width) in enumerate(zip(bin_centers_s, hist_vals_s, widths_s)):
    ax8.bar3d(center, 0, 0, width*0.8, 0.1, height, color='red', alpha=0.7, edgecolor='black')
ax8.plot(x_norm, np.zeros_like(x_norm), stats.norm.pdf(x_norm, 0, 1), 'b--', linewidth=2, label='Normalverteilung')
ax8.set_title('❌ ε nicht normalverteilt\nSchiefe Verteilung', fontweight='bold', color='red')
ax8.set_xlabel('Residuum')
ax8.set_ylabel('')
ax8.set_zlabel('Dichte')
ax8.legend(fontsize=7)
ax8.view_init(elev=20, azim=-60)

plt.suptitle('Die 4 Gauss-Markov Annahmen: Korrekt (links) vs. Verletzt (rechts) - 3D', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
st.pyplot(fig_assumptions)
plt.close()

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
        # Residuenplot (Homoskedastizität & E(ε)=0) in 3D
        fig_diag1 = plt.figure(figsize=(10, 7))
        ax_diag1 = fig_diag1.add_subplot(111, projection='3d')
        
        z_diag = np.abs(model.resid)  # Use absolute residual as z
        ax_diag1.scatter(y_pred, model.resid, z_diag, s=60, c='blue', alpha=0.6)
        
        # Zero plane
        xlim = (min(y_pred)-1, max(y_pred)+1)
        ylim = (-max(np.abs(model.resid))-1, max(np.abs(model.resid))+1)
        xx, yy = np.meshgrid([xlim[0], xlim[1]], [0, 0])
        zz = np.zeros_like(xx)
        ax_diag1.plot_surface(xx, yy, zz, alpha=0.3, color='red')
        
        ax_diag1.set_xlabel('Vorhergesagte Werte (ŷ)')
        ax_diag1.set_ylabel('Residuen (e)')
        ax_diag1.set_zlabel('|Residuen|')
        ax_diag1.set_title('Residuenplot (3D): Prüfung (1) & (2)', fontweight='bold')
        ax_diag1.view_init(elev=20, azim=-60)
        
        plt.tight_layout()
        st.pyplot(fig_diag1)
        plt.close()
        
        st.markdown("""
        **Interpretation:**
        - Punkte sollten **zufällig** um 0 streuen
        - Kein Muster/Trichter → Homoskedastizität ✓
        - Kein Bogen → Linearität ✓
        """)
    
    with col_diag2:
        # Q-Q Plot (Normalität) in 3D
        from scipy.stats import probplot
        fig_diag2 = plt.figure(figsize=(10, 7))
        ax_diag2 = fig_diag2.add_subplot(111, projection='3d')
        
        # Get Q-Q data
        qq = probplot(model.resid, dist="norm")
        theoretical = qq[0][0]
        sample = qq[0][1]
        z_qq = np.abs(sample - theoretical)  # Deviation from diagonal
        
        ax_diag2.scatter(theoretical, sample, z_qq, s=50, c='blue', alpha=0.6)
        
        # Diagonal reference plane
        diag_range = [min(theoretical), max(theoretical)]
        xx, yy = np.meshgrid(diag_range, diag_range)
        zz = np.zeros_like(xx)
        ax_diag2.plot_surface(xx, yy, zz, alpha=0.3, color='red')
        
        ax_diag2.set_xlabel('Theoretische Quantile')
        ax_diag2.set_ylabel('Stichproben-Quantile')
        ax_diag2.set_zlabel('Abweichung')
        ax_diag2.set_title('Q-Q Plot (3D): Prüfung (4) Normalität', fontweight='bold')
        ax_diag2.view_init(elev=20, azim=-60)
        
        plt.tight_layout()
        st.pyplot(fig_diag2)
        plt.close()
        
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
    fig_t = plt.figure(figsize=(14, 8))
    ax_t = fig_t.add_subplot(111, projection='3d')
    
    x_t = np.linspace(-5, max(5, abs(t_val) + 2), 300)
    y_t = stats.t.pdf(x_t, df=df_resid)
    z_t = np.zeros_like(x_t)
    
    # t-distribution curve in 3D
    ax_t.plot(x_t, y_t, z_t, 'k-', linewidth=2.5, label=f't-Verteilung (df={df_resid})')
    
    # Critical region as vertical bars
    crit_idx = abs(x_t) > abs(t_val)
    for i in range(0, len(x_t), 5):
        if crit_idx[i]:
            ax_t.plot([x_t[i], x_t[i]], [0, y_t[i]], [0, 0], 'r-', alpha=0.2, linewidth=2)
    
    t_crit = stats.t.ppf(0.975, df=df_resid)
    # Critical value planes
    ylim = (0, max(y_t)*1.1)
    ax_t.plot([t_crit, t_crit], ylim, [0, 0], color='orange', linestyle='--', linewidth=2, alpha=0.7)
    ax_t.plot([-t_crit, -t_crit], ylim, [0, 0], color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Kritische Werte (α=0.05): ±{t_crit:.2f}')
    ax_t.plot([t_val, t_val], [0, max(y_t)], [0, 0], color='blue', linewidth=4, label=f'Unser t-Wert = {t_val:.2f}')
    
    ax_t.set_xlabel('t-Wert', fontsize=12)
    ax_t.set_ylabel('Dichte', fontsize=12)
    ax_t.set_zlabel('Höhe', fontsize=12)
    ax_t.set_title(f'H₀: β₁ = 0 vs. H₁: β₁ ≠ 0 (3D)\nt = b₁/s_b₁ = {b1:.4f}/{sb1:.4f} = {t_val:.2f}', 
                  fontsize=13, fontweight='bold')
    ax_t.legend(loc='upper right')
    ax_t.view_init(elev=20, azim=-60)
    
    plt.tight_layout()
    st.pyplot(fig_t)
    plt.close()

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
    fig_f = plt.figure(figsize=(14, 8))
    ax_f = fig_f.add_subplot(111, projection='3d')
    
    x_f = np.linspace(0, max(10, f_val + 5), 300)
    y_f = stats.f.pdf(x_f, dfn=1, dfd=df_resid)
    z_f = np.zeros_like(x_f)
    
    # F-distribution curve in 3D
    ax_f.plot(x_f, y_f, z_f, 'k-', linewidth=2.5, label=f'F-Verteilung (df₁=1, df₂={df_resid})')
    
    # Critical region as vertical bars
    crit_idx = x_f > f_val
    for i in range(0, len(x_f), 5):
        if crit_idx[i]:
            ax_f.plot([x_f[i], x_f[i]], [0, y_f[i]], [0, 0], color='purple', alpha=0.2, linewidth=2)
    
    f_crit = stats.f.ppf(0.95, dfn=1, dfd=df_resid)
    ylim = (0, max(y_f)*1.1)
    ax_f.plot([f_crit, f_crit], ylim, [0, 0], color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Kritisch (α=0.05): {f_crit:.2f}')
    ax_f.plot([f_val, f_val], [0, max(y_f)], [0, 0], color='purple', linewidth=4, label=f'Unser F-Wert = {f_val:.2f}')
    
    ax_f.set_xlabel('F-Wert', fontsize=12)
    ax_f.set_ylabel('Dichte', fontsize=12)
    ax_f.set_zlabel('Höhe', fontsize=12)
    ax_f.set_title(f'H₀: R² = 0 (Modell erklärt nichts) (3D)\nF = MSR/MSE = {msr:.2f}/{mse:.2f} = {f_val:.2f}', 
                  fontsize=13, fontweight='bold')
    ax_f.legend(loc='upper right')
    ax_f.set_xlim(0, max(15, f_val + 5))
    ax_f.view_init(elev=20, azim=-60)
    
    plt.tight_layout()
    st.pyplot(fig_f)
    plt.close()

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
    'Quadratsumme': [ssr, sse, sst],
    'df': [1, n-2, n-1],
    'Mittlere Quadratsumme': [msr, mse, '-'],
    'F-Wert': [f'{f_val:.2f}', '-', '-'],
    'p-Wert': [f'{model.f_pvalue:.4g} {get_signif_stars(model.f_pvalue)}', '-', '-']
})

st.dataframe(anova_df, use_container_width=True, hide_index=True)

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
st.sidebar.markdown("### ANOVA-Beispiel")
anova_effect = st.sidebar.slider("Effektstärke Regionen", 0.0, 2.0, 0.8, 0.1,
                                  help="Wie stark unterscheiden sich die Regionsmittelwerte?")
anova_noise_level = st.sidebar.slider("Streuung innerhalb Gruppen", 0.5, 2.0, 1.0, 0.1, key="anova_noise")

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
        
        fig_anova_viz = plt.figure(figsize=(14, 6))
        
        # 1. 3D Surface für Gruppen-Verteilungen
        ax_3d = fig_anova_viz.add_subplot(121, projection='3d')
        
        regions = ['Nord', 'Mitte', 'Süd']
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        # X-Achse: Umsatz-Werte
        x_vals = np.linspace(
            min(df_anova['Umsatz']) - 1,
            max(df_anova['Umsatz']) + 1,
            100
        )
        
        for i, (region, color) in enumerate(zip(regions, colors)):
            data = df_anova[df_anova['Region'] == region]['Umsatz']
            mu = data.mean()
            sigma = data.std()
            
            # Normalverteilung für jede Gruppe
            y_vals = norm.pdf(x_vals, mu, sigma)
            
            # 3D Plot: x-Achse = Umsatz, y-Achse = Region (i), z-Achse = Dichte
            ax_3d.plot(x_vals, np.full_like(x_vals, i), y_vals, 
                      color=color, linewidth=3, label=f'{region}')
            
            # Fläche unter Kurve
            ax_3d.bar(x_vals[::5], np.full_like(x_vals[::5], i), y_vals[::5], 
                     zdir='z', zs=0, width=0.3, alpha=0.3, color=color)
        
        # Gesamtmittelwert als Linie
        y_overall = norm.pdf(x_vals, grand_mean_anova, df_anova['Umsatz'].std())
        ax_3d.plot(x_vals, np.full_like(x_vals, -0.5), y_overall, 
                  color='black', linewidth=2, linestyle='--', label='Gesamtverteilung')
        
        ax_3d.set_xlabel(y_label, fontsize=11)
        ax_3d.set_ylabel('Gruppen', fontsize=11)
        ax_3d.set_zlabel('Dichte', fontsize=11)
        ax_3d.set_yticks(range(3))
        ax_3d.set_yticklabels(regions)
        ax_3d.set_title('3D Landscape: Gruppen als Verteilungshügel', fontsize=12, fontweight='bold')
        ax_3d.legend(loc='upper left', fontsize=9)
        
        # 2. Varianzzerlegung
        ax_var = fig_anova_viz.add_subplot(122)
        
        ax_var.barh(0, sstr_anova, color='green', alpha=0.7, label=f'SSTR (Zwischen) = {sstr_anova:.2f}')
        ax_var.barh(0, sse_anova, left=sstr_anova, color='red', alpha=0.7, label=f'SSE (Innerhalb) = {sse_anova:.2f}')
        
        ax_var.set_xlim(0, sst_anova * 1.1)
        ax_var.set_yticks([0])
        ax_var.set_yticklabels(['Varianz-\nzerlegung'])
        ax_var.set_xlabel('Quadratsumme', fontsize=12)
        ax_var.set_title(f'SST = SSTR + SSE = {sst_anova:.2f}', fontsize=12, fontweight='bold')
        ax_var.legend(loc='upper right')
        
        # Prozentwerte
        pct_sstr = sstr_anova / sst_anova * 100
        pct_sse = sse_anova / sst_anova * 100
        ax_var.annotate(f'{pct_sstr:.1f}%', xy=(sstr_anova/2, 0), fontsize=12, 
                       ha='center', va='center', color='white', fontweight='bold')
        ax_var.annotate(f'{pct_sse:.1f}%', xy=(sstr_anova + sse_anova/2, 0), fontsize=12, 
                       ha='center', va='center', color='white', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig_anova_viz)
        plt.close()
    else:
        # 2D Original: Boxplot + Varianzzerlegung
        fig_anova_viz, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Boxplot mit Punkten
        regions = ['Nord', 'Mitte', 'Süd']
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        for i, (region, color) in enumerate(zip(regions, colors)):
            data = df_anova[df_anova['Region'] == region]['Umsatz']
            
            # Jittered Scatter
            jitter = np.random.normal(0, 0.08, len(data))
            axes[0].scatter(np.full(len(data), i) + jitter, data, 
                           c=color, alpha=0.6, s=60, edgecolor='white')
            
            # Gruppenmittelwert
            axes[0].hlines(data.mean(), i - 0.3, i + 0.3, colors=color, linewidths=4, 
                          label=f'{region}: μ = {data.mean():.2f}')
            
            # Linien zu Gruppenmittelwert (SSE)
            for j, val in enumerate(data):
                axes[0].plot([i + jitter[j], i + jitter[j]], [val, data.mean()], 
                            color=color, alpha=0.2, linewidth=1)
        
        # Gesamtmittelwert
        axes[0].axhline(grand_mean_anova, color='black', linestyle='--', linewidth=2, 
                       label=f'Gesamtmittel: {grand_mean_anova:.2f}')
        
        axes[0].set_xticks(range(3))
        axes[0].set_xticklabels(regions)
        axes[0].set_ylabel(y_label, fontsize=12)
        axes[0].set_title('Gruppenvergleich: Streuung innerhalb (SSE) vs. zwischen (SSTR)', 
                         fontsize=12, fontweight='bold')
        axes[0].legend(loc='upper left', fontsize=9)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 2. Varianzzerlegung als gestapelter Balken
        axes[1].barh(0, sstr_anova, color='green', alpha=0.7, label=f'SSTR (Zwischen) = {sstr_anova:.2f}')
        axes[1].barh(0, sse_anova, left=sstr_anova, color='red', alpha=0.7, label=f'SSE (Innerhalb) = {sse_anova:.2f}')
        
        axes[1].set_xlim(0, sst_anova * 1.1)
        axes[1].set_yticks([0])
        axes[1].set_yticklabels(['Varianz-\nzerlegung'])
        axes[1].set_xlabel('Quadratsumme', fontsize=12)
        axes[1].set_title(f'SST = SSTR + SSE = {sst_anova:.2f}', fontsize=12, fontweight='bold')
        axes[1].legend(loc='upper right')
        
        # Prozentwerte annotieren
        pct_sstr = sstr_anova / sst_anova * 100
        pct_sse = sse_anova / sst_anova * 100
        axes[1].annotate(f'{pct_sstr:.1f}%', xy=(sstr_anova/2, 0), fontsize=12, 
                        ha='center', va='center', color='white', fontweight='bold')
        axes[1].annotate(f'{pct_sse:.1f}%', xy=(sstr_anova + sse_anova/2, 0), fontsize=12, 
                        ha='center', va='center', color='white', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig_anova_viz)
        plt.close()

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

st.dataframe(anova_display, use_container_width=True, hide_index=True)

# Post-Hoc Tests hinzufügen
if p_anova < 0.05:
    st.markdown('<p class="subsection-header">🔍 Post-Hoc Analyse: Welche Gruppen unterscheiden sich?</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Die ANOVA sagt uns nur: **"Es gibt einen Unterschied."** 
    Aber **zwischen welchen Gruppen genau?** Dafür brauchen wir Post-Hoc Tests.
    """)
    
    # Paarweise t-Tests mit Bonferroni-Korrektur
    from scipy.stats import ttest_ind
    
    regions = ['Nord', 'Mitte', 'Süd']
    alpha_bonferroni = 0.05 / 3  # Korrektur für 3 Vergleiche
    
    col_ph1, col_ph2 = st.columns([2, 1])
    
    with col_ph1:
        posthoc_results = []
        for i in range(len(regions)):
            for j in range(i+1, len(regions)):
                group1 = df_anova[df_anova['Region'] == regions[i]]['Umsatz']
                group2 = df_anova[df_anova['Region'] == regions[j]]['Umsatz']
                t_stat, p_val = ttest_ind(group1, group2)
                diff = group1.mean() - group2.mean()
                sig = "✅ Signifikant" if p_val < alpha_bonferroni else "❌ Nicht signifikant"
                posthoc_results.append({
                    'Vergleich': f'{regions[i]} vs. {regions[j]}',
                    'Differenz': f'{diff:.3f}',
                    't-Wert': f'{t_stat:.3f}',
                    'p-Wert': f'{p_val:.4f}',
                    'Signifikant (α_korr=0.017)': sig
                })
        
        posthoc_df = pd.DataFrame(posthoc_results)
        st.dataframe(posthoc_df, use_container_width=True, hide_index=True)
    
    with col_ph2:
        st.warning(f"""
        ### ⚠️ Bonferroni-Korrektur
        
        Bei **mehreren Tests** steigt die Fehlerrate!
        
        **Problem:** 3 Tests × α=0.05 → ~15% Fehlerchance
        
        **Lösung:** α_korrigiert = 0.05 / 3 = **{alpha_bonferroni:.4f}**
        
        Nur p < {alpha_bonferroni:.4f} gilt als signifikant.
        """)

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
# KAPITEL 5.6: HOMO- vs. HETEROSKEDASTIZITÄT (Das große Problem)
# =========================================================
st.markdown("---")
st.markdown('<p class="section-header">⚠️ Das große Problem: Heteroskedastizität</p>', unsafe_allow_html=True)

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
    
    fig_trichter = plt.figure(figsize=(16, 10))
    
    # Homo Scatter - 3D
    ax1 = fig_trichter.add_subplot(2, 2, 1, projection='3d')
    z_homo = np.random.normal(0, 0.3, n_demo)
    ax1.scatter(x_demo, y_homo, z_homo, color='green', alpha=0.6, s=30)
    ax1.plot(x_demo, model_homo.predict(X_demo), np.zeros_like(x_demo), 'k-', lw=2)
    ax1.set_title("✅ Homoskedastizität (Ideal)\nGleichmäßiger Schlauch", fontweight='bold', color='green')
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Varianz")
    ax1.view_init(elev=20, azim=-60)
    
    # Homo Residual - 3D
    ax2 = fig_trichter.add_subplot(2, 2, 3, projection='3d')
    z_resid_homo = np.abs(model_homo.resid)
    ax2.scatter(model_homo.predict(X_demo), model_homo.resid, z_resid_homo, color='green', alpha=0.6, s=30)
    # Zero plane
    xlim_h = (min(model_homo.predict(X_demo)), max(model_homo.predict(X_demo)))
    xx_h, yy_h = np.meshgrid([xlim_h[0], xlim_h[1]], [0, 0])
    zz_h = np.zeros_like(xx_h)
    ax2.plot_surface(xx_h, yy_h, zz_h, alpha=0.3, color='black')
    ax2.set_title("Residual-Plot (3D)\n✅ Wolke ohne Muster", fontweight='bold', color='green')
    ax2.set_ylabel("Residuen")
    ax2.set_xlabel("Fitted Values")
    ax2.set_zlabel("|Residuen|")
    ax2.view_init(elev=20, azim=-60)
    
    # Hetero Scatter - 3D
    ax3 = fig_trichter.add_subplot(2, 2, 2, projection='3d')
    z_hetero = 0.1 * x_demo  # Variance increases with x
    ax3.scatter(x_demo, y_hetero, z_hetero, color='red', alpha=0.6, s=30)
    ax3.plot(x_demo, model_hetero.predict(X_demo), np.zeros_like(x_demo), 'k-', lw=2)
    ax3.set_title("⚠️ Heteroskedastizität (Problem)\nTrichter-Effekt!", fontweight='bold', color='red')
    ax3.set_zlabel("Varianz (wächst!)")
    ax3.view_init(elev=20, azim=-60)
    
    # Hetero Residual - 3D
    ax4 = fig_trichter.add_subplot(2, 2, 4, projection='3d')
    z_resid_hetero = np.abs(model_hetero.resid)
    ax4.scatter(model_hetero.predict(X_demo), model_hetero.resid, z_resid_hetero, color='red', alpha=0.6, s=30)
    xlim_het = (min(model_hetero.predict(X_demo)), max(model_hetero.predict(X_demo)))
    xx_het, yy_het = np.meshgrid([xlim_het[0], xlim_het[1]], [0, 0])
    zz_het = np.zeros_like(xx_het)
    ax4.plot_surface(xx_het, yy_het, zz_het, alpha=0.3, color='black')
    ax4.set_title("Residual-Plot (3D)\n⚠️ Trichterform sichtbar!", fontweight='bold', color='red')
    ax4.set_ylabel("Residuen")
    ax4.set_xlabel("Fitted Values")
    ax4.set_zlabel("|Residuen|")
    ax4.view_init(elev=20, azim=-60)
    
    plt.suptitle("🔍 Diagnose (3D): Der Blick auf die Residuen", fontsize=16, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_trichter)
    plt.close()

with col_hetero2:
    st.error("""
    ### ⚠️ Warum ist das schlimm?
    
    Bei Heteroskedastizität:
    - Standardfehler **zu klein** berechnet
    - t-Werte **zu groß**
    - p-Werte **zu klein**
    
    **→ Falsche Sterne! ★★★**
    
    Du glaubst, etwas ist signifikant, 
    aber die Unsicherheit ist viel größer!
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
    größer → ehrlichere p-Werte!
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
    fig_3d = plt.figure(figsize=(12, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    x_line = np.linspace(float(x.min()), float(x.max()), 100)
    y_line = b0 + b1 * x_line
    ax_3d.plot(x_line, y_line, np.zeros_like(x_line), 'b-', linewidth=3, label='E(y|x)')
    
    x_points = np.linspace(float(x.min()) + 0.5, float(x.max()) - 0.5, 5)
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, 5))
    
    for i, x_pt in enumerate(x_points):
        y_exp = b0 + b1 * x_pt
        sigma = se_regression * 1.5
        y_range = np.linspace(y_exp - 3*sigma, y_exp + 3*sigma, 80)
        z_pdf = stats.norm.pdf(y_range, y_exp, sigma)
        z_pdf = z_pdf / z_pdf.max() * 1.5
        
        ax_3d.plot(np.full_like(y_range, x_pt), y_range, z_pdf, color=colors[i], linewidth=2)
        ax_3d.scatter([x_pt], [y_exp], [0], color=colors[i], s=50)
    
    ax_3d.scatter(x, y, np.zeros(len(x)), c='green', s=40, alpha=0.6, marker='^', label='Daten')
    
    ax_3d.set_xlabel(f'X ({x_label})', fontsize=11)
    ax_3d.set_ylabel(f'Y ({y_label})', fontsize=11)
    ax_3d.set_zlabel('f(y|x)', fontsize=11)
    ax_3d.set_title('Bedingte Verteilung: Für jeden X-Wert gibt es eine\nVerteilung möglicher Y-Werte', 
                   fontsize=13, fontweight='bold')
    ax_3d.legend()
    ax_3d.view_init(elev=25, azim=-60)
    
    plt.tight_layout()
    st.pyplot(fig_3d)
    plt.close()

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
