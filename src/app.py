"""
🎓 Leitfaden zur Linearen Regression
====================================

HINWEIS: Diese Datei ist ein Legacy-Entry-Point.

Die empfohlene Ausführung ist:
    streamlit run run.py

Oder direkt:
    python run.py --streamlit

Diese Datei delegiert an die neue Architektur:
    src/adapters/streamlit/app.py

Die neue Architektur bietet:
- 100% Platform-Agnostik (API Layer)
- Konsistente Schnittstelle für alle Frontends
- Saubere Layer-Trennung
"""

import warnings
warnings.filterwarnings('ignore')

# Delegate to the new streamlit adapter
from .adapters.streamlit.app import run_streamlit_app

# This allows: streamlit run src/app.py
if __name__ == "__main__":
    run_streamlit_app()
else:
    # When imported by Streamlit
    run_streamlit_app()
