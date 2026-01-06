# 🎓 Linear Regression Guide

Ein umfassendes didaktisches Tool zum Verstehen der linearen Regression mit interaktiven Visualisierungen und schrittweisen Erklärungen.

## 🚀 Schnellstart

### Voraussetzungen
- Python 3.8 oder höher
- Streamlit

### Installation

1. **Repository klonen:**
   ```bash
   git clone <repository-url>
   cd linear-regression-guide
   ```

2. **Virtuelle Umgebung erstellen und aktivieren:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # venv\Scripts\activate   # Windows
   ```

3. **Abhängigkeiten installieren:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Anwendung starten:**
   ```bash
   streamlit run run.py
   ```

   **⚠️ Wichtig:** Verwende `streamlit run run.py`, nicht `python run.py`

## 📋 Verwendung

Die Anwendung öffnet sich automatisch in Ihrem Webbrowser. Die Anwendung bietet:

- **Einfache lineare Regression**: Schritt-für-Schritt Erklärung
- **Multiple lineare Regression**: Mit mehreren Prädiktoren
- **Datensatz-Explorer**: Eingebaute Beispieldatensätze
- **Interaktive Visualisierungen**: Plotly-basierte Charts
- **Statistische Analysen**: Vollständige Regressionsdiagnostik

## 🏗️ Architektur

Die Anwendung folgt Clean Architecture Prinzipien:

```
📁 src/
├── 🏛️ core/                    # Business Logic Layer
│   ├── domain/                # Domain Entities & Business Rules
│   └── application/           # Use Cases & Application Services
├── 🏗️ infrastructure/          # External Concerns (DB, APIs, etc.)
├── 🎨 ui/                     # Presentation Layer
└── 📊 data/                   # Data Access & Generation
```

### Wichtige Hinweise zur Ausführung

⚠️ **Diese Anwendung ist speziell für Streamlit designed und kann nicht direkt mit `python app.py` ausgeführt werden.**

**Korrekte Ausführung:**
```bash
streamlit run run.py
```

**Warum nicht direkte Ausführung?**
- Die Anwendung verwendet relative Imports, die nur im Streamlit-Kontext funktionieren
- Streamlit richtet automatisch die Python-Pfad-Struktur ein
- Direkte Ausführung führt zu Import-Fehlern: `ImportError: attempted relative import with no known parent package`

## 🔧 Entwicklung

### Architektur-Validierung
Überprüfen Sie die Einhaltung der Clean Architecture Standards:

```bash
python scripts/check_modular_separation.py
```

### Tests ausführen
```bash
python -m pytest tests/
```

### Code-Qualität
```bash
# Linting
flake8 src/ --config=config/.flake8

# Type checking
mypy src/ --config-file config/mypy.ini
```

## 📦 Abhängigkeiten

### Kernabhängigkeiten
- **streamlit**: Web-Framework für interaktive Data-Apps
- **numpy**: Numerische Berechnungen
- **pandas**: Datenmanipulation
- **plotly**: Interaktive Visualisierungen

### Wissenschaftliche Bibliotheken
- **statsmodels**: Statistische Modelle und Tests
- **scipy**: Wissenschaftliche Berechnungen

### Externe APIs
- **requests**: HTTP-Anfragen für externe Daten
- **openai**: Perplexity API Integration

## 🚨 Problembehandlung

### Import-Fehler
```
ImportError: attempted relative import with no known parent package
```

**Lösung:** Verwenden Sie immer `streamlit run run.py`, nicht `python run.py`.

### Port-Konflikte
```bash
streamlit run run.py --server.port 8502
```

### Virtuelle Umgebung Probleme
```bash
# Umgebung neu erstellen
rm -rf venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

## 🤝 Mitwirken

1. Fork das Repository
2. Erstellen Sie einen Feature-Branch
3. Führen Sie Tests aus: `python scripts/check_modular_separation.py`
4. Commit Ihre Änderungen
5. Erstellen Sie einen Pull Request

## 📄 Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert - siehe [LICENSE](LICENSE) Datei für Details.

## 🙏 Danksagungen

- Streamlit Community für das fantastische Framework
- Wissenschaftliche Python Community für die exzellenten Bibliotheken
- Alle Mitwirkenden, die dieses Bildungs-Tool verbessert haben