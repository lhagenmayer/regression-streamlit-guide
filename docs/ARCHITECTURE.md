# 🏗️ Architektur-Dokumentation

**Regression Analysis Platform - 100% Plattform-Agnostisch**

Diese Dokumentation beschreibt die Architektur der Anwendung aus Top-Down und Bottom-Up Perspektive.

---

## 📋 Inhaltsverzeichnis

1. [Architektur-Übersicht](#architektur-übersicht)
2. [Layer-Struktur (Top-Down)](#layer-struktur-top-down)
3. [Datenfluss](#datenfluss)
4. [Module im Detail](#module-im-detail)
5. [Design-Prinzipien](#design-prinzipien)
6. [Abhängigkeits-Regeln](#abhängigkeits-regeln)
7. [Erweiterbarkeit](#erweiterbarkeit)

---

## 🏛️ Architektur-Übersicht

Die Anwendung folgt einer **Schichtenarchitektur** mit strikter Trennung zwischen:

- **Presentation Layer** (Adapters) - Framework-spezifischer Code
- **API Layer** - REST-Schnittstelle für alle Frontends
- **Business Logic** (Content) - Edukativer Content als Datenstruktur
- **Core Layer** (Pipeline) - Statistische Berechnungen
- **External Integration** (AI) - Perplexity AI

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ENTRY POINTS                                       │
│                            run.py                                            │
│         --api (REST) │ --flask (HTML) │ --streamlit (Interactive)           │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        ↓                          ↓                          ↓
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────────────┐
│   src/api/        │   │  src/adapters/    │   │  src/adapters/streamlit/  │
│   (Pure JSON)     │   │  flask_app.py     │   │  (Interactive Python)     │
│   No frameworks   │   │  (HTML/Jinja2)    │   │                           │
└─────────┬─────────┘   └─────────┬─────────┘   └─────────────┬─────────────┘
          │                       │                           │
          └───────────────────────┼───────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CONTENT LAYER - src/content/                          │
│                                                                              │
│   ContentBuilder → EducationalContent (Pure Data, JSON-serializable)        │
│   SimpleRegressionContent | MultipleRegressionContent                        │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE LAYER - src/pipeline/                        │
│                                                                              │
│   ┌─────────────┐    ┌──────────────────┐    ┌─────────────┐                │
│   │ DataFetcher │ →  │ StatsCalculator  │ →  │ PlotBuilder │                │
│   │ (get_data)  │    │   (calculate)    │    │   (plot)    │                │
│   └─────────────┘    └──────────────────┘    └─────────────┘                │
│                                                                              │
│   Pure NumPy/SciPy, keine Framework-Abhängigkeiten                          │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AI LAYER - src/ai/                                  │
│                                                                              │
│   PerplexityClient (External API Integration)                                │
│   100% Framework-agnostisch, nur Environment Variables                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Layer-Struktur (Top-Down)

### Layer 1: Entry Points

| Datei | Zweck | Abhängigkeiten |
|-------|-------|----------------|
| `run.py` | Unified Entry Point | Auto-Detection |

**Verantwortlichkeiten:**
- Erkennung des gewünschten Frameworks (`--api`, `--flask`, `--streamlit`)
- Delegation an entsprechenden Adapter
- WSGI-Support für Production

### Layer 2: API Layer (`src/api/`)

| Datei | Zweck | LOC |
|-------|-------|-----|
| `endpoints.py` | Business Logic | ~600 |
| `serializers.py` | JSON Conversion | ~500 |
| `server.py` | HTTP Server | ~300 |

**Verantwortlichkeiten:**
- REST-Endpunkte für alle Operationen
- JSON-Serialisierung aller Datenstrukturen
- CORS-Support
- OpenAPI/Swagger-Dokumentation

**Erlaubt:** Import von `pipeline`, `content`, `ai`
**Verboten:** Import von `adapters`, Framework-spezifischer Code

### Layer 3: Adapters (`src/adapters/`)

| Datei | Framework | Zweck |
|-------|-----------|-------|
| `flask_app.py` | Flask | HTML/Jinja2 Rendering |
| `streamlit/app.py` | Streamlit | Interactive UI |
| `renderers/` | Beide | Content → UI Conversion |
| `ai_components.py` | Beide | AI UI Components |

**Verantwortlichkeiten:**
- Framework-spezifische UI-Logik
- Template-Rendering
- User Interactions

**Erlaubt:** Import von allen anderen Modulen + Framework-Libraries
**Verboten:** Geschäftslogik, Berechnungen

### Layer 4: Content (`src/content/`)

| Datei | Zweck |
|-------|-------|
| `structure.py` | Content-Datenklassen |
| `builder.py` | Abstract Builder |
| `simple_regression.py` | 11 Kapitel Simple Reg. |
| `multiple_regression.py` | 9 Kapitel Multiple Reg. |

**Verantwortlichkeiten:**
- Definition des edukativen Contents als DATEN
- Keine UI-Logik, nur Strukturen
- Alle Klassen haben `to_dict()` für JSON

**Erlaubt:** Import von `pipeline` für Statistik-Zugriff
**Verboten:** Framework-Imports, UI-Code

### Layer 5: Pipeline (`src/pipeline/`)

| Datei | Step | Zweck |
|-------|------|-------|
| `get_data.py` | GET | Datengenerierung |
| `calculate.py` | CALCULATE | OLS, R², t-Tests |
| `plot.py` | PLOT | Plotly Figures |
| `regression_pipeline.py` | Orchestration | 4-Step Pipeline |

**Verantwortlichkeiten:**
- Statistische Berechnungen
- Transparente, verifizierbare Formeln
- Plotly-Visualisierungen

**Erlaubt:** NumPy, SciPy, Plotly
**Verboten:** Framework-Imports, UI-Code

### Layer 6: AI (`src/ai/`)

| Datei | Zweck |
|-------|-------|
| `perplexity_client.py` | Perplexity API Client |

**Verantwortlichkeiten:**
- Externe API-Integration
- Response-Caching
- Fallback-Interpretationen

**Erlaubt:** `requests`, `os` (für Environment)
**Verboten:** Framework-Imports

### Layer 7: Config (`src/config/`)

| Datei | Zweck |
|-------|-------|
| `config.py` | Globale Konfiguration |
| `logger.py` | Logging-Setup |

---

## 🔄 Datenfluss

### Simple Regression Request

```
HTTP Request
     │
     ↓
┌────────────────────────────────────────────────────────────────┐
│ 1. API Layer (src/api/endpoints.py)                            │
│    ContentAPI.get_simple_content(dataset="electronics", n=50)  │
└────────────────────────────────┬───────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ 2. Pipeline (src/pipeline/regression_pipeline.py)              │
│    RegressionPipeline.run_simple()                             │
│    → DataFetcher.get_simple() → DataResult                     │
│    → StatisticsCalculator.simple_regression() → RegressionResult│
│    → PlotBuilder.simple_regression_plots() → PlotCollection    │
└────────────────────────────────┬───────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ 3. Serialization (src/api/serializers.py)                      │
│    StatsSerializer.to_flat_dict() → Dict                       │
└────────────────────────────────┬───────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ 4. Content Build (src/content/simple_regression.py)            │
│    SimpleRegressionContent(stats_dict, plot_keys)              │
│    → EducationalContent (11 Chapters)                          │
└────────────────────────────────┬───────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ 5. Final Serialization                                         │
│    ContentSerializer.serialize() → JSON                        │
│    PlotSerializer.serialize_collection() → Plotly JSON         │
└────────────────────────────────┬───────────────────────────────┘
                                 ↓
                          HTTP Response
                          {
                            "success": true,
                            "content": {...},
                            "plots": {...},
                            "stats": {...}
                          }
```

---

## 📦 Module im Detail

### Pipeline-Datentypen

```python
@dataclass
class DataResult:
    x: np.ndarray
    y: np.ndarray
    x_label: str
    y_label: str
    context_title: str
    context_description: str

@dataclass
class RegressionResult:
    intercept: float
    slope: float
    r_squared: float
    r_squared_adj: float
    se_slope: float
    t_slope: float
    p_slope: float
    # ... weitere Statistiken

@dataclass
class PlotCollection:
    scatter: go.Figure
    residuals: go.Figure
    diagnostics: go.Figure
    extra: Dict[str, go.Figure]
```

### Content-Struktur

```python
@dataclass
class EducationalContent:
    title: str
    subtitle: str
    chapters: List[Chapter]

@dataclass
class Chapter:
    number: str
    title: str
    icon: str
    sections: List[ContentElement]

# ContentElement Types:
# - Markdown(text)
# - Formula(latex, inline)
# - Plot(plot_key, height)
# - Metric(label, value, help_text)
# - MetricRow(metrics)
# - Table(headers, rows)
# - Expander(title, content)
# - InfoBox/WarningBox/SuccessBox(content)
```

---

## 🎯 Design-Prinzipien

### 1. Platform-Agnostik

**Jeder** Output ist JSON-serialisierbar:
- NumPy Arrays → Python Lists
- Plotly Figures → JSON
- Dataclasses → Dictionaries

### 2. Layer-Isolation

Jeder Layer kennt nur die Layer UNTER sich:

```
API Layer
    ↓ (kann importieren)
Content Layer
    ↓ (kann importieren)
Pipeline Layer
    ↓ (kann importieren)
AI Layer
```

### 3. Dependency Injection

APIs werden lazy geladen, um zirkuläre Importe zu vermeiden:

```python
class RegressionAPI:
    def __init__(self):
        self._pipeline = None  # Lazy
    
    @property
    def pipeline(self):
        if self._pipeline is None:
            from ..pipeline import RegressionPipeline
            self._pipeline = RegressionPipeline()
        return self._pipeline
```

### 4. Single Responsibility

Jedes Modul hat eine klare, einzelne Verantwortlichkeit:
- `get_data.py` - NUR Datengenerierung
- `calculate.py` - NUR Statistik
- `plot.py` - NUR Visualisierungen

---

## 🚦 Abhängigkeits-Regeln

### ✅ ERLAUBT

```python
# API kann Pipeline importieren
from ..pipeline import RegressionPipeline

# Adapters können alles importieren
from ..api import RegressionAPI
from ..content import SimpleRegressionContent
import streamlit as st

# Content kann Pipeline importieren
from ..pipeline.calculate import RegressionResult
```

### ❌ VERBOTEN

```python
# Pipeline darf NICHT Adapters/API importieren
from ..api import ...  # NEIN!
from ..adapters import ...  # NEIN!

# Content darf NICHT Framework importieren
import streamlit  # NEIN!
from flask import ...  # NEIN!

# AI darf NICHT Framework importieren
import streamlit  # NEIN!
```

---

## 🔧 Erweiterbarkeit

### Neues Frontend hinzufügen (z.B. Vue.js)

1. **Keine Backend-Änderungen nötig!**
2. Vue-App konsumiert `/api/content/simple` Endpunkt
3. Rendert `content.chapters` mit Vue-Komponenten
4. Zeigt Plots mit `plotly.js` an

### Neuen Dataset-Typ hinzufügen

1. `src/pipeline/get_data.py` erweitern
2. Neue Methode in `DataFetcher`
3. Automatisch in API verfügbar

### Neuen Content-Typ hinzufügen

1. `src/content/structure.py` - Neue Dataclass
2. `src/content/builder.py` - Helper-Methode
3. `src/adapters/renderers/` - Render-Logik

---

## 📊 Metriken

| Layer | Dateien | LOC | Abhängigkeiten |
|-------|---------|-----|----------------|
| Entry | 1 | ~230 | Auto-Detection |
| API | 4 | ~1320 | Flask/FastAPI (optional) |
| Adapters | 9 | ~2150 | Streamlit, Flask |
| Content | 5 | ~1600 | NumPy |
| Pipeline | 6 | ~1170 | NumPy, SciPy, Plotly |
| AI | 2 | ~450 | requests |
| Config | 3 | ~320 | - |

**Gesamte Codebasis: ~7240 LOC**

---

## 🧪 Testing

```bash
# Unit Tests
pytest tests/unit/ -v

# Integration Tests
pytest tests/integration/ -v

# API Test
curl http://localhost:8000/api/health
curl -X POST http://localhost:8000/api/regression/simple \
  -H "Content-Type: application/json" \
  -d '{"dataset": "electronics", "n": 50}'
```

---

## 📚 Weiterführende Dokumentation

- **[API.md](API.md)** - Vollständige REST API Dokumentation
- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** - Frontend-Integration
- **[openapi.yaml](openapi.yaml)** - OpenAPI Specification
