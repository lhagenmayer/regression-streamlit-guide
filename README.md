# 📊 Linear Regression Guide

Ein interaktives, didaktisches Tool für lineare Regressionsanalyse.

**Frontend-Agnostisch:** Läuft sowohl mit **Streamlit** als auch mit **Flask** - automatische Framework-Erkennung!

## 🎯 Architektur

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND-AGNOSTIC                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────┐   ┌───────────┐   ┌──────┐   ┌─────────┐    │
│   │   GET   │ → │ CALCULATE │ → │ PLOT │ → │ DISPLAY │    │
│   └─────────┘   └───────────┘   └──────┘   └─────────┘    │
│        │              │             │            │         │
│   DataFetcher   Statistics     PlotBuilder   Adapters     │
│                 Calculator                                 │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│              FRAMEWORK ADAPTERS                             │
│  ┌──────────────────┐    ┌──────────────────┐              │
│  │    Streamlit     │    │      Flask       │              │
│  │   (Interactive)  │    │   (Traditional)  │              │
│  └──────────────────┘    └──────────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Schnellstart

### Option 1: Streamlit (Interaktiv)
```bash
pip install -r requirements.txt
streamlit run run.py
```

### Option 2: Flask (Traditionell)
```bash
pip install -r requirements.txt
python run.py
# oder: flask --app src.adapters.flask_app:create_flask_app run
```

### Option 3: WSGI Server (Production)
```bash
gunicorn "run:create_app()"
# oder: waitress-serve --port=5000 run:create_app
```

## 📁 Projektstruktur

```
src/
├── pipeline/                 # Core Pipeline (Framework-Agnostic)
│   ├── get_data.py          # Step 1: GET - Daten generieren
│   ├── calculate.py         # Step 2: CALCULATE - Statistiken
│   ├── plot.py              # Step 3: PLOT - Visualisierungen
│   ├── display.py           # Step 4: DISPLAY - Data Preparation
│   └── regression_pipeline.py  # Pipeline Orchestrator
│
├── adapters/                 # Framework Adapters
│   ├── detector.py          # Auto-Detection (Streamlit/Flask)
│   ├── base.py              # Abstract Renderer Interface
│   ├── streamlit_app.py     # Streamlit Implementation
│   ├── flask_app.py         # Flask Implementation
│   └── templates/           # Flask HTML Templates
│
├── ui/tabs/                  # Educational Content
│   ├── simple_regression_educational.py
│   └── multiple_regression_educational.py
│
├── data/content.py          # Dynamic Content
└── config/                  # Configuration & Logging

run.py                       # Unified Entry Point
```

## 🔄 Auto-Detection

Das Framework wird automatisch erkannt:

| Aufruf | Erkanntes Framework |
|--------|---------------------|
| `streamlit run run.py` | Streamlit |
| `python run.py` | Flask |
| `REGRESSION_FRAMEWORK=flask python run.py` | Flask (explizit) |
| `gunicorn "run:create_app()"` | Flask (WSGI) |

## 💻 API Usage

```python
from src.pipeline import RegressionPipeline

# Pipeline initialisieren
pipeline = RegressionPipeline()

# Einfache Regression
result = pipeline.run_simple(
    dataset="electronics",
    n=100,
    seed=42
)

print(f"R² = {result.stats.r_squared:.4f}")
print(f"β₁ = {result.stats.slope:.4f}")

# Multiple Regression
result = pipeline.run_multiple(
    dataset="cities",
    n=100,
    seed=42
)

print(f"R² = {result.stats.r_squared:.4f}")
print(f"F = {result.stats.f_statistic:.2f}")
```

## 🎓 Features

### Einfache Regression
- OLS-Schätzung mit transparenten Formeln
- R², adjustiertes R², Standardfehler
- t-Tests, p-Werte, Konfidenzintervalle
- Residuenanalyse & Diagnostik-Plots
- Interaktive Visualisierungen

### Multiple Regression
- Mehrere Prädiktoren
- 3D Regressionsebene
- VIF & Multikollinearität
- F-Test für Gesamtsignifikanz
- Ceteris Paribus Interpretation

## 🧪 Tests

```bash
# Alle Tests
pytest tests/ -v

# Nur Pipeline Tests
pytest tests/unit/test_pipeline.py -v
```

## 📦 Dependencies

```
numpy>=1.24.0      # Numerische Berechnungen
pandas>=2.0.0      # Datenstrukturen
scipy>=1.11.0      # Statistische Funktionen
plotly>=5.18.0     # Interaktive Plots

# Web Frameworks (mindestens eines)
streamlit>=1.28.0  # Interaktive Web App
flask>=3.0.0       # Traditionelle Web App
```

## 🏗️ Eigenen Adapter erstellen

```python
from src.adapters.base import BaseRenderer, RenderContext

class MyCustomRenderer(BaseRenderer):
    def render(self, context: RenderContext):
        # Eigene Rendering-Logik
        pass
    
    def render_simple_regression(self, context: RenderContext):
        # Simple Regression rendern
        pass
    
    def render_multiple_regression(self, context: RenderContext):
        # Multiple Regression rendern
        pass
    
    def run(self, host="0.0.0.0", port=8000, debug=False):
        # Server starten
        pass
```

## 📄 Lizenz

MIT License
