## Linear Regression Guide

Interaktiver Leitfaden zur linearen Regression mit Streamlit. Gedacht für alle, die Regression visuell und nachvollziehbar lernen wollen.

[![Tests](https://github.com/lhagenmayer/linear-regression-guide/workflows/Run%20Tests/badge.svg)](https://github.com/lhagenmayer/linear-regression-guide/actions/workflows/tests.yml)
[![Lint](https://github.com/lhagenmayer/linear-regression-guide/workflows/Lint/badge.svg)](https://github.com/lhagenmayer/linear-regression-guide/actions/workflows/lint.yml)
[![Deployment Validation](https://github.com/lhagenmayer/linear-regression-guide/workflows/Deployment%20Validation/badge.svg)](https://github.com/lhagenmayer/linear-regression-guide/actions/workflows/deployment-validation.yml)

<!-- Deployment badge - uncomment and update URL after deployment -->
<!-- [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app) -->

### 🚀 Live Demo

The app is ready for deployment to Streamlit Cloud. Once deployed, the live demo will be available here.

**To deploy your own instance:**
1. Fork this repository
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account and select this repository
4. Set main file path to `app.py`
5. Deploy!

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions or [QUICKSTART_DEPLOYMENT.md](QUICKSTART_DEPLOYMENT.md) for a 5-minute quick start guide.

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

### Development

#### Code Quality Standards

Das Projekt verwendet moderne Code-Quality-Tools für konsistente Formatierung und Qualität:

- **Black**: Automatische Code-Formatierung (Zeilenlänge: 100)
- **Flake8**: Style-Guide-Enforcement und Error-Checking
- **MyPy**: Statische Type-Checking (optional)
- **Pre-commit Hooks**: Automatische Checks vor jedem Commit

#### Setup für Entwicklung

```bash
# Entwicklungs-Abhängigkeiten installieren
pip install -r requirements-dev.txt

# Pre-commit Hooks einrichten
pre-commit install

# Code formatieren
black *.py tests/*.py

# Linting überprüfen
flake8 *.py tests/*.py

# Type-Checking (optional)
mypy app.py config.py data.py plots.py
```

#### Pre-commit Hooks

Pre-commit Hooks führen automatisch Code-Formatierung und Checks durch:

```bash
# Manuell ausführen
pre-commit run --all-files

# Hooks werden automatisch bei 'git commit' ausgeführt
git commit -m "Your message"
```

Die Hooks prüfen:
- Trailing Whitespace
- End-of-File Fixer
- YAML/JSON/TOML Syntax
- Code-Formatierung (Black)
- Linting (Flake8)

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
| content.py | Content and text for the app |
| requirements.txt | Laufzeitabhängigkeiten |
| requirements-dev.txt | Entwicklungs- und Test-Abhängigkeiten |
| tests/ | Comprehensive test suite |
| .streamlit/config.toml | Streamlit Cloud configuration |
| .github/workflows/ | GitHub Actions CI/CD workflows |
| validate_deployment.py | Deployment validation script |
| pyproject.toml | Black und Pytest Konfiguration |
| .flake8 | Flake8 Konfiguration |
| mypy.ini | MyPy Konfiguration |
| .pre-commit-config.yaml | Pre-commit Hooks Konfiguration |
| DEPLOYMENT.md | Comprehensive Streamlit Cloud deployment guide |
| QUICKSTART_DEPLOYMENT.md | 5-minute deployment quick start |
| TESTING.md | Testing documentation |
| DEVELOPMENT.md | Development guide |
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
