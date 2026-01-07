# Pipeline Architecture

## Überblick

Die Anwendung folgt einer klaren **4-Stufen-Pipeline**:

```
┌─────────┐    ┌───────────┐    ┌──────┐    ┌─────────┐
│   GET   │ → │ CALCULATE │ → │ PLOT │ → │ DISPLAY │
└─────────┘    └───────────┘    └──────┘    └─────────┘
```

## Die 4 Stufen

### 1. GET (`src/pipeline/get_data.py`)

**Aufgabe:** Daten generieren oder laden

**Klassen:**
- `DataFetcher` - Hauptklasse für Datenbeschaffung
- `DataResult` - Ergebnis für einfache Regression
- `MultipleRegressionDataResult` - Ergebnis für multiple Regression

**Beispiel:**
```python
from src.pipeline import DataFetcher

fetcher = DataFetcher()
data = fetcher.get_simple("electronics", n=50, seed=42)
# data.x, data.y, data.x_label, data.context_title, ...
```

### 2. CALCULATE (`src/pipeline/calculate.py`)

**Aufgabe:** OLS-Regression und Statistiken berechnen

**Klassen:**
- `StatisticsCalculator` - Berechnet alle Statistiken
- `RegressionResult` - Ergebnis für einfache Regression
- `MultipleRegressionResult` - Ergebnis für multiple Regression

**Berechnete Werte:**
- Koeffizienten (β₀, β₁, ...)
- R², R² adj.
- Standard Errors
- t-Werte, p-Werte
- Residuen

**Beispiel:**
```python
from src.pipeline import StatisticsCalculator

calc = StatisticsCalculator()
result = calc.simple_regression(data.x, data.y)
# result.slope, result.r_squared, result.p_slope, ...
```

### 3. PLOT (`src/pipeline/plot.py`)

**Aufgabe:** Plotly-Visualisierungen erstellen

**Klassen:**
- `PlotBuilder` - Erstellt alle Plots
- `PlotCollection` - Sammlung von Plots

**Plots:**
- Scatter mit Regressionslinie
- Residuenplot
- Diagnose-Plots (Q-Q, Histogram, etc.)
- 3D-Oberflächen (Multiple Regression)

**Beispiel:**
```python
from src.pipeline import PlotBuilder

plotter = PlotBuilder()
plots = plotter.simple_regression_plots(data, result)
# plots.scatter, plots.residuals, plots.diagnostics
```

### 4. DISPLAY (`src/pipeline/display.py` + `src/ui/tabs/`)

**Aufgabe:** Plots mit edukativem Content im UI anzeigen

**Klassen:**
- `UIRenderer` - Verbindet Pipeline mit Tabs
- Educational Tab Modules in `src/ui/tabs/`

**Prinzip:** Kein Plot ohne Kontext!

Jeder Plot ist eingebettet in:
- Erklärung was der Plot zeigt
- Interpretation der Ergebnisse
- Hinweise worauf zu achten ist

**Beispiel:**
```python
from src.pipeline import RegressionPipeline

pipeline = RegressionPipeline()
result = pipeline.run_simple(dataset="electronics", n=50)
pipeline.display(result, show_formulas=True)
```

## Datei-Struktur

```
src/pipeline/
├── __init__.py              # Exports
├── get_data.py              # Step 1: GET
├── calculate.py             # Step 2: CALCULATE
├── plot.py                  # Step 3: PLOT
├── display.py               # Step 4: DISPLAY (delegiert zu tabs)
└── regression_pipeline.py   # Orchestriert alle Steps

src/ui/tabs/
├── simple_regression_educational.py   # Vollständiger edukativer Content
├── multiple_regression_educational.py # Vollständiger edukativer Content
└── datasets.py                        # Datensatz-Übersicht
```

## Verwendung

### Komplette Pipeline

```python
from src.pipeline import RegressionPipeline

# Pipeline initialisieren
pipeline = RegressionPipeline()

# Einfache Regression
result = pipeline.run_simple(
    dataset="electronics",
    n=50,
    noise=0.4,
    seed=42
)

# In Streamlit anzeigen
pipeline.display(result, show_formulas=True)
```

### Einzelne Schritte

```python
# Nur Daten
data = pipeline.get_data("simple", dataset="electronics", n=50)

# Nur Berechnung
stats = pipeline.calculate(data, "simple")

# Nur Plots
plots = pipeline.plot(data, stats, "simple")
```

## Dynamischer Content

Der Content passt sich automatisch an den Datensatz an:

```python
# content.py liefert datensatz-spezifische Texte
from src.data import get_multiple_regression_descriptions

content = get_multiple_regression_descriptions("🏙️ Städte-Umsatzstudie")
# content["main"], content["variables"], ...
```

## Tests

```bash
# Alle Pipeline-Tests
pytest tests/unit/test_pipeline.py -v

# Alle Tests
pytest tests/ -v
```

**Testabdeckung:**
- ✅ DataFetcher (5 Tests)
- ✅ StatisticsCalculator (6 Tests)
- ✅ PlotBuilder (2 Tests)
- ✅ RegressionPipeline (4 Tests)
- ✅ Konsistenz (2 Tests)

## Design-Prinzipien

1. **Separation of Concerns**: Jede Stufe hat eine klare Aufgabe
2. **Immutable Results**: DataResult, RegressionResult sind Dataclasses
3. **Lazy Loading**: UI-Komponenten werden nur bei Bedarf geladen
4. **Educational First**: Jeder Plot hat Kontext und Interpretation
5. **Dynamic Content**: Texte passen sich an Datensatz an
