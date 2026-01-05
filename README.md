# Linear Regression Guide

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)

Eine interaktive Web-App zum Erlernen linearer Regression. Gebaut mit Streamlit, plotly und statsmodels - für alle, die Regression verstehen wollen, ohne sich durch Formeln zu kämpfen.

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
streamlit run run.py
```

Die App öffnet sich automatisch im Browser.

## Projekt-Struktur

```
linear-regression-guide/
├── src/                    # Haupt-Code
│   ├── app.py             # Streamlit-App
│   ├── data.py            # Datenfunktionen
│   ├── plots.py           # Visualisierungen
│   ├── config.py          # Konfiguration
│   └── ...
├── docs/                  # Dokumentation
├── config/                # Konfigurationsdateien
├── tests/                 # Automatisierte Tests
├── scripts/               # Hilfsskripte
└── run.py                 # App-Startpunkt
```

## Weitere Informationen

- **[Vollständige Dokumentation](docs/README.md)** - Detaillierte Anleitung
- **[Entwicklung](docs/DEVELOPMENT.md)** - Für Mitwirkende
- **[Tests](docs/TESTING.md)** - Qualitätssicherung

## Lizenz

MIT License - siehe [LICENSE](LICENSE) für Details.