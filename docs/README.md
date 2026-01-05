# Linear Regression Guide

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)

Eine interaktive Web-App zum Erlernen linearer Regression. Gebaut mit Streamlit, plotly und statsmodels - für alle, die Regression verstehen wollen, ohne sich durch Formeln zu kämpfen.

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
>>>>>>> origin/copilot/setup-streamlit-cloud-deployment

**Warum diese App?**
Regression ist ein wichtiges statistisches Werkzeug, aber die Theorie kann überwältigend sein. Diese App macht Regression greifbar: Spiele mit Daten herum, sieh live, wie Modelle funktionieren, und verstehe die Konzepte visuell. Perfekt für Studierende, Datenanalysten und alle, die Regression anwenden wollen.

## Was kann die App?

**Interaktive Visualisierungen:**
- Scatterplots mit Regressionslinien
- 3D-Oberflächen für multiple Regression
- Residuenplots und Diagnostik
- Live-Updates bei Parameteränderungen

**Verschiedene Datensätze:**
- Simulierte Daten (Elektronikmarkt, Häuser, Städte)
- Echte Schweizer Daten (Kantone, Wetterstationen)
- Vollständig offline - keine API-Abhängigkeiten

**Lernpfad:**
- Grundlagen der linearen Regression
- Multiple Regression mit mehreren Prädiktoren
- Modellinterpretation und Diagnostik
- Statistische Tests und Hypothesen

**Einfach zu bedienen:**
- Navigation mit Tabs
- Anpassbare Parameter
- Klare Erklärungen
- Reagiert schnell

## Los geht's

**Voraussetzungen:**
- Python 3.9 oder neuer
- Ein virtuelles Environment (empfohlen)

**Installation:**
```bash
# Repository klonen
git clone <repository-url>
cd linear-regression-guide

# Abhängigkeiten installieren
pip install -r requirements.txt

# App starten
streamlit run app.py
```

Die App öffnet sich automatisch im Browser. Wenn nicht, gehe zu `http://localhost:8501`.

**Erste Schritte:**
1. Wähle ein Kapitel in der Sidebar
2. Spiele mit den Parametern herum
3. Beobachte, wie sich die Plots ändern
4. Lies die Erklärungen zu den statistischen Konzepten

## Entwicklung

Falls du den Code ändern möchtest:

```bash
# Zusätzliche Tools installieren
pip install -r requirements-dev.txt

# Automatische Code-Prüfung einrichten
pre-commit install

# Code formatieren
black *.py tests/*.py

# Tests laufen lassen
pytest tests/
```

## Tests

Es gibt Tests, um sicherzustellen, dass alles funktioniert.

<<<<<<< HEAD
```bash
# Tests laufen lassen
pytest tests/
```
=======
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
>>>>>>> origin/copilot/setup-streamlit-cloud-deployment

Mehr Details in [TESTING.md](TESTING.md).

## Dateien

- `app.py` - Haupt-App
- `data.py` - Datenfunktionen
- `plots.py` - Diagramme
- `content.py` - Texte und Formeln
- `config.py` - Einstellungen
- `tests/` - Tests
- `requirements.txt` - Abhängigkeiten

## Wie benutzt man die App?

1. **Kapitel wählen:** In der Sidebar ein Thema auswählen
2. **Parameter anpassen:** Spiele mit Stichprobengröße, Rauschen und Seeds
3. **Visualisierungen beobachten:** Siehe live, wie sich Modelle ändern
4. **Erklärungen lesen:** Verstehe die statistischen Konzepte

**Tipp:** Verwende verschiedene Seeds, um zu sehen, wie zufällige Variationen die Ergebnisse beeinflussen.

## Technisches

Die App nutzt:
- Streamlit für die Web-Oberfläche
- Plotly für Diagramme
- Statsmodels für statistische Berechnungen
- Caching für bessere Performance

## Änderungen

Falls du etwas ändern möchtest, schau dir [DEVELOPMENT.md](DEVELOPMENT.md) an.

## Lizenz

MIT License - siehe [LICENSE](LICENSE) für Details. Frei verwendbar für Bildung und Forschung.
