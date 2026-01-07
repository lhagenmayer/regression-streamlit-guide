# 🏗️ Architektur

## Übersicht

Diese Anwendung implementiert **Option B: Content als Datenstruktur** für eine vollständig frontend-agnostische Architektur.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              run.py                                      │
│                         (Framework Detection)                            │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                ┌────────────────┴────────────────┐
                ↓                                 ↓
    ┌───────────────────────┐         ┌───────────────────────┐
    │   Streamlit Frontend  │         │    Flask Frontend     │
    │   adapters/streamlit/ │         │  adapters/flask_app   │
    └───────────────────────┘         └───────────────────────┘
                │                                 │
                └────────────────┬────────────────┘
                                 ↓
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         Pipeline                                     │
    │                                                                      │
    │   GET → CALCULATE → PLOT → DISPLAY                                  │
    │                                                                      │
    └────────────────────────────────┬────────────────────────────────────┘
                                     ↓
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    ContentBuilder (content/)                         │
    │                                                                      │
    │   SimpleRegressionContent        MultipleRegressionContent          │
    │   └── build() → EducationalContent                                  │
    │                                                                      │
    │   Content ist DATEN, nicht UI-Code!                                 │
    └────────────────────────────────┬────────────────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ↓                                 ↓
    ┌─────────────────────────────┐   ┌─────────────────────────────┐
    │  StreamlitContentRenderer   │   │    HTMLContentRenderer      │
    │                             │   │                             │
    │  Interpretiert Content als: │   │  Interpretiert Content als: │
    │  - st.markdown()            │   │  - HTML/Jinja2              │
    │  - st.plotly_chart()        │   │  - Plotly.js                │
    │  - st.expander()            │   │  - Bootstrap Accordion      │
    └─────────────────────────────┘   └─────────────────────────────┘
```

---

## Module

### 1. Content (`src/content/`)

Framework-agnostische Definition des Educational Contents.

#### `structure.py`
Definiert alle Content-Elemente als Dataclasses:

```python
@dataclass
class Chapter:
    number: str
    title: str
    icon: str
    sections: List[ContentElement]

@dataclass
class Markdown(ContentElement):
    text: str

@dataclass
class Formula(ContentElement):
    latex: str
    inline: bool = False

@dataclass
class Plot(ContentElement):
    plot_key: str
    title: str = ""
    height: int = 400
```

#### `builder.py`
Base class für Content Builder mit Helper-Methoden:

```python
class ContentBuilder(ABC):
    def __init__(self, stats: Dict, plots: Dict):
        self.stats = stats
        self.plots = plots
    
    @abstractmethod
    def build(self) -> EducationalContent:
        pass
    
    def fmt(self, value: float, decimals: int = 4) -> str:
        """Format numeric value."""
        
    def interpret_r2(self, r2: float) -> str:
        """Interpret R² value."""
```

#### `simple_regression.py` / `multiple_regression.py`
Konkrete Content Builder:

```python
class SimpleRegressionContent(ContentBuilder):
    def build(self) -> EducationalContent:
        return EducationalContent(
            title="📊 Einfache Lineare Regression",
            chapters=[
                self._chapter_1_introduction(),
                self._chapter_2_model(),
                # ... 11 Kapitel
            ]
        )
```

---

### 2. Pipeline (`src/pipeline/`)

4-Schritt Datenverarbeitung:

```
┌─────────┐    ┌───────────┐    ┌──────┐    ┌─────────┐
│   GET   │ → │ CALCULATE │ → │ PLOT │ → │ DISPLAY │
└─────────┘    └───────────┘    └──────┘    └─────────┘
```

#### `get_data.py`
Datengenerierung und -laden:

```python
class DataFetcher:
    def get_simple(self, dataset, n, noise, seed) -> DataResult
    def get_multiple(self, dataset, n, noise, seed) -> MultipleRegressionDataResult
```

#### `calculate.py`
Statistische Berechnungen:

```python
class StatisticsCalculator:
    def simple_regression(self, x, y) -> RegressionResult
    def multiple_regression(self, x1, x2, y) -> MultipleRegressionResult
```

#### `plot.py`
Plot-Generierung:

```python
class PlotBuilder:
    def simple_regression_plots(self, data, result) -> PlotCollection
    def multiple_regression_plots(self, data, result) -> PlotCollection
```

#### `regression_pipeline.py`
Unified Pipeline:

```python
class RegressionPipeline:
    def run_simple(self, dataset, n, noise, seed) -> PipelineResult
    def run_multiple(self, dataset, n, noise, seed) -> PipelineResult
```

---

### 3. Adapters (`src/adapters/`)

Framework-spezifische Implementierungen.

#### `detector.py`
Framework Auto-Detection:

```python
class Framework(Enum):
    STREAMLIT = "streamlit"
    FLASK = "flask"
    UNKNOWN = "unknown"

class FrameworkDetector:
    @staticmethod
    def detect() -> Framework
```

#### `renderers/streamlit_renderer.py`
Streamlit-spezifisches Rendering:

```python
class StreamlitContentRenderer:
    def render(self, content: EducationalContent) -> None:
        for chapter in content.chapters:
            st.markdown(f"## {chapter.title}")
            for section in chapter.sections:
                self._render_element(section)
    
    def _render_element(self, element: ContentElement):
        if isinstance(element, Markdown):
            st.markdown(element.text)
        elif isinstance(element, Formula):
            st.latex(element.latex)
        elif isinstance(element, Plot):
            st.plotly_chart(self.plots[element.plot_key])
        # ...
```

#### `renderers/html_renderer.py`
HTML/Flask-spezifisches Rendering:

```python
class HTMLContentRenderer:
    def render(self, content: EducationalContent) -> str:
        html_parts = []
        for chapter in content.chapters:
            html_parts.append(self._render_chapter(chapter))
        return '\n'.join(html_parts)
    
    def _render_element(self, element: ContentElement) -> str:
        if isinstance(element, Markdown):
            return f'<div class="markdown">{element.text}</div>'
        elif isinstance(element, Formula):
            return f'<div class="math">\\[{element.latex}\\]</div>'
        # ...
```

---

## Datenfluss

### 1. Request kommt rein

```
User → /simple?dataset=Bildung&n=50
```

### 2. Pipeline wird ausgeführt

```python
pipeline = RegressionPipeline()
result = pipeline.run_simple(dataset="electronics", n=50)
# result.data, result.stats, result.plots
```

### 3. Stats Dictionary wird erstellt

```python
stats_dict = {
    'context_title': 'Bildung und Einkommen',
    'x_label': 'Bildungsjahre',
    'y_label': 'Einkommen (CHF)',
    'slope': 5000.0,
    'intercept': 20000.0,
    'r_squared': 0.72,
    # ... alle Statistiken
}
```

### 4. Content wird generiert

```python
builder = SimpleRegressionContent(stats_dict, {})
content = builder.build()
# content.chapters[0].sections[0] → InfoBox mit "Bildung und Einkommen"
```

### 5. Content wird gerendert

**Streamlit:**
```python
renderer = StreamlitContentRenderer(stats=stats_dict)
renderer.render(content)  # → st.* Aufrufe
```

**Flask:**
```python
renderer = HTMLContentRenderer(stats=stats_dict)
html = renderer.render(content)  # → HTML String
return render_template('educational_content.html', content_html=html)
```

---

## Content-Elemente Mapping

| ContentElement | Streamlit | Flask/HTML |
|----------------|-----------|------------|
| `Markdown` | `st.markdown()` | `<div class="markdown">` |
| `Formula` | `st.latex()` | MathJax `\\[...\\]` |
| `Plot` | `st.plotly_chart()` | `Plotly.newPlot()` |
| `Table` | `st.dataframe()` | `<table class="table">` |
| `Metric` | `st.metric()` | `<div class="metric-card">` |
| `MetricRow` | `st.columns()` | `<div class="metrics-grid">` |
| `Expander` | `st.expander()` | Bootstrap Accordion |
| `Columns` | `st.columns()` | Bootstrap Grid Row |
| `InfoBox` | `st.info()` | `<div class="alert alert-info">` |
| `WarningBox` | `st.warning()` | `<div class="alert alert-warning">` |
| `SuccessBox` | `st.success()` | `<div class="alert alert-success">` |
| `CodeBlock` | `st.code()` | `<pre><code>` |

---

## Dynamischer Content

Der Content ist vollständig dynamisch basierend auf dem `stats` Dictionary:

### Beispiel: Interpretation

```python
def _build_interpretation_section(self, s: Dict) -> Markdown:
    return Markdown(f"""
**Das Modell:**
{s['y_label']} = {s['intercept']:.4f} + {s['slope']:.4f} × {s['x_label']}

**Interpretation:**
Pro Einheit {s['x_label']} ändert sich {s['y_label']} um {s['slope']:.4f}.
""")
```

### Beispiel: R-Output

```python
def _build_r_output(self, s: Dict) -> str:
    return f"""
Call:
lm(formula = {s['y_label']} ~ {s['x_label']})

Coefficients:
(Intercept)  {s['intercept']:9.4f}
{s['x_label']}  {s['slope']:9.4f}

Multiple R-squared:  {s['r_squared']:.4f}
"""
```

---

## Erweiterung

### Neues Frontend

1. Neuen Renderer in `adapters/renderers/` erstellen
2. `_render_element()` für alle ContentElement-Typen implementieren
3. In Adapter einbinden

### Neuer Content

1. Neuen ContentBuilder in `content/` erstellen
2. `build()` Methode implementieren
3. Kapitel-Methoden definieren
4. Alle Renderer zeigen es automatisch an

### Neues Content-Element

1. Neue Dataclass in `structure.py` definieren
2. In allen Renderern `_render_element()` erweitern
