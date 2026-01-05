## Linear Regression Guide

Interaktiver Leitfaden zur linearen Regression mit Streamlit. Gedacht für alle, die Regression visuell und nachvollziehbar lernen wollen.

### Funktionsumfang

* Interaktive Plots (plotly, statsmodels)
* Zwei Datensätze: simuliert und echte Stadtdaten
* Kapitelweise Navigation von Grundlagen bis ANOVA
* R-ähnliche Ausgabeformate mit Erklärungen
* Performance-optimiert mit Caching und Session State Management

### Schnellstart

1) Python 3.9+ und optional ein virtuelles Environment
2) Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt
   ```
3) App starten:
   ```bash
   streamlit run app.py
   ```
   Browser öffnet automatisch oder manuell `http://localhost:8501` aufrufen.

### Testing

Umfassende Test-Suite mit:
- Unit-Tests für Datengenerie und Plotting-Funktionen
- Integration-Tests mit Streamlit AppTest Framework
- Performance-Regressionstests
- GitHub Actions CI/CD

```bash
# Tests ausführen
pip install -r requirements-dev.txt
pytest tests/

# Mit Coverage-Report
pytest --cov --cov-report=html
```

Siehe [TESTING.md](TESTING.md) für Details zur Test-Infrastruktur.

### Dateien

| Datei | Beschreibung |
|-------|--------------|
| app.py | Haupt-App mit Streamlit UI und Tab-Navigation |
| data.py | Datengenerie functions und data handling |
| plots.py | Plotting functions (plotly visualizations) |
| config.py | Configuration constants |
| requirements.txt | Laufzeitabhängigkeiten |
| requirements-dev.txt | Entwicklungs- und Test-Abhängigkeiten |
| tests/ | Comprehensive test suite |
| TESTING.md | Testing documentation |
| PERFORMANCE_OPTIMIZATIONS.md | Performance optimization details |
| README.md | Projektüberblick |

### Nutzung

In der Sidebar Kapitel auswählen, Parameter anpassen, Visualisierungen beobachten. Für Reproduzierbarkeit: Seed und Stichprobengröße sind einstellbar.

### Performance Optimizations

Die App nutzt umfassende Performance-Optimierungen:
- `@st.cache_data` für Datengenerie (50x-100x Geschwindigkeitsverbesserung)
- Session State für Model-Caching
- Smart Recalculation (nur bei Parameter-Änderungen)
- Loading-Indikatoren für bessere UX
- Lazy Tab Loading

### Lizenz

MIT. Siehe LICENSE.
