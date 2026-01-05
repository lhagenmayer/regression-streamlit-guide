# Linear Regression Guide

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![CI](https://github.com/lhagenmayer/linear-regression-guide/actions/workflows/tests.yml/badge.svg)](https://github.com/lhagenmayer/linear-regression-guide/actions)
[![Quality](https://github.com/lhagenmayer/linear-regression-guide/actions/workflows/lint.yml/badge.svg)](https://github.com/lhagenmayer/linear-regression-guide/actions)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Eine interaktive Web-App zum Erlernen linearer Regression. Gebaut mit Streamlit, plotly und statsmodels - für alle, die Regression verstehen wollen, ohne sich durch Formeln zu kämpfen.

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

**Benutzerfreundlich:**
- Einfache Navigation mit Tabs
- Anpassbare Parameter (Stichprobengröße, Rauschen, Seeds)
- Klare Erklärungen statt komplizierter Formeln
- Performance-optimiert für flüssige Bedienung

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

## Für Entwickler

Das Projekt legt Wert auf Code-Qualität. Wir verwenden moderne Tools, um den Code konsistent und wartbar zu halten.

### Setup

```bash
# Zusätzliche Tools für Entwicklung installieren
pip install -r requirements-dev.txt

# Git Hooks einrichten (automatische Code-Qualität)
pre-commit install
```

### Code-Qualität

**Automatische Formatierung:**
```bash
# Code automatisch formatieren
black *.py tests/*.py
```

**Qualitätsprüfung:**
```bash
# Style und Fehler prüfen
flake8 *.py tests/*.py

# Type-Checking (optional, aber empfohlen)
mypy app.py config.py data.py plots.py
```

**Git Hooks:**
Die pre-commit Hooks laufen automatisch bei jedem Commit und prüfen:
- Entfernung von Leerzeichen am Zeilenende
- Korrekte Dateienden
- Code-Formatierung mit Black
- Style-Regeln mit Flake8

Manuell ausführen:
```bash
pre-commit run --all-files
```

## Tests

Das Projekt hat eine umfassende Test-Suite, um sicherzustellen, dass alles funktioniert.

**Verfügbare Tests:**
- Unit-Tests für Datenfunktionen und Plots
- Integration-Tests für die Streamlit-App
- Performance-Tests mit Caching-Validierung
- Automatische Tests bei jedem Push (GitHub Actions)

```bash
# Alle Tests laufen lassen
pytest tests/

# Mit Coverage-Report (zeigt, wie viel Code getestet ist)
pytest --cov --cov-report=html

# Nur schnelle Tests (ohne Performance-Tests)
pytest tests/ -m "not slow"
```

Für detaillierte Informationen zu den Tests siehe [TESTING.md](TESTING.md).

## Projekt-Struktur

| Datei/Ordner | Was ist da drin? |
|-------------|------------------|
| `app.py` | Die Haupt-Streamlit-App mit Tabs und Navigation |
| `data.py` | Funktionen zur Datengenerierung und -verarbeitung |
| `plots.py` | Alle Plotly-Visualisierungen und Charts |
| `config.py` | Konfiguration, Konstanten und Einstellungen |
| `content.py` | Texte, Formeln und Beschreibungen für die UI |
| `requirements.txt` | Python-Pakete für den Betrieb |
| `requirements-dev.txt` | Zusätzliche Tools für Entwicklung |
| `tests/` | Automatisierte Tests für alles |
| `pyproject.toml` | Konfiguration für Black und Tests |
| `.flake8` | Style-Regeln für Python-Code |
| `mypy.ini` | Type-Checking Konfiguration |
| `.pre-commit-config.yaml` | Automatische Code-Qualitäts-Checks |
| `DEVELOPMENT.md` | Detaillierte Anleitung für Entwickler |
| `TESTING.md` | Alles über Tests und Qualitätssicherung |

## Wie benutzt man die App?

1. **Kapitel wählen:** In der Sidebar ein Thema auswählen
2. **Parameter anpassen:** Spiele mit Stichprobengröße, Rauschen und Seeds
3. **Visualisierungen beobachten:** Siehe live, wie sich Modelle ändern
4. **Erklärungen lesen:** Verstehe die statistischen Konzepte

**Tipp:** Verwende verschiedene Seeds, um zu sehen, wie zufällige Variationen die Ergebnisse beeinflussen.

## Technische Details

**Performance:**
- Smart Caching für schnelle Reaktionen
- Session State für nahtlose Interaktionen
- Lazy Loading für effiziente Ressourcennutzung
- Optimiert für Desktop und Mobile

**Qualität:**
- Automatische Code-Formatierung und Tests
- Professionelle Entwicklungsumgebung
- Umfassende Dokumentation

## Mitmachen

Beiträge sind willkommen! Das Projekt legt Wert auf Qualität und Benutzerfreundlichkeit.

**Bevor du beiträgst:**
1. Installiere die Entwicklungs-Tools: `pip install -r requirements-dev.txt`
2. Richte pre-commit Hooks ein: `pre-commit install`
3. Führe Tests aus: `pytest tests/`
4. Stelle sicher, dass alles formatiert ist: `black *.py tests/*.py`

**Code-Qualität:**
- Verwende beschreibende Variablennamen
- Schreibe Tests für neue Funktionen
- Halte die Dokumentation aktuell
- Folge dem bestehenden Stil

Für detaillierte Anleitungen siehe [DEVELOPMENT.md](DEVELOPMENT.md).

## Lizenz

MIT License - siehe [LICENSE](LICENSE) für Details. Frei verwendbar für Bildung und Forschung.
