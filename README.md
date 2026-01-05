## Regression Streamlit Guide

Interaktiver Leitfaden zur einfachen und multiplen linearen Regression mit Streamlit. Gedacht für alle, die Regression visuell und nachvollziehbar lernen wollen.

### Funktionsumfang

* Interaktive Plots (plotly, statsmodels)
* Zwei Datensätze: simuliert und echte Stadtdaten
* Kapitelweise Navigation von Grundlagen bis ANOVA
* R-ähnliche Ausgabeformate mit Erklärungen

### Schnellstart

1) Python 3.10+ und optional ein virtuelles Environment
2) Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt
   ```
3) App starten:
   ```bash
   streamlit run app.py
   ```
   Browser öffnet automatisch oder manuell `http://localhost:8501` aufrufen.

### Dateien

| Datei | Beschreibung |
|-------|--------------|
| app.py | Haupt-App mit allen Kapiteln |
| requirements.txt | Laufzeitabhängigkeiten |
| README.md | Projektüberblick |

### Nutzung

In der Sidebar Kapitel auswählen, Parameter anpassen, Visualisierungen beobachten. Für Reproduzierbarkeit: Seed und Stichprobengröße sind einstellbar.

### Lizenz

MIT. Siehe LICENSE.
