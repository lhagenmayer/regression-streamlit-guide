"""
Conclusion and Summary section for Simple Linear Regression.

This module summarizes key learnings and provides practical guidance
for applying simple linear regression.
"""

import streamlit as st
from ...content import get_content
from ...logger import get_logger

logger = get_logger(__name__)


def render_conclusion() -> None:
    """Render the conclusion and summary section."""
    logger.debug("Rendering conclusion section")

    st.subheader("🎓 Zusammenfassung & Ausblick")

    st.markdown("""
    ### ✅ Was haben wir gelernt?

    **Einfache Lineare Regression** ist ein mächtiges Werkzeug zur Untersuchung
    linearer Zusammenhänge zwischen zwei kontinuierlichen Variablen.

    **Schlüsselpunkte**:
    - 📏 **OLS-Schätzung** minimiert quadrierte Residuen
    - 📊 **R²** misst Anteil erklärter Varianz
    - 🧪 **Hypothesentests** prüfen Signifikanz der Koeffizienten
    - 🔧 **Diagnostik** validiert Modellannahmen
    """)

    # Key takeaways
    with st.expander("🔑 Wichtige Erkenntnisse", expanded=True):
        st.markdown("""
        1. **Interpretation**: $\\beta_1$ zeigt Änderung in y bei Erhöhung von x um 1 Einheit
        2. **p-Wert < 0.05**: Statistisch signifikanter Zusammenhang
        3. **R² > 0.7**: Gutes Modell (kontextabhängig)
        4. **Residuen**: Sollten zufällig um 0 streuen
        """)

    # Practical applications
    st.markdown("""
    ### 🚀 Praktische Anwendungen

    **Einfache Lineare Regression** findet Anwendung in:
    - 📈 **Wirtschaft**: Umsatzprognose basierend auf Werbeausgaben
    - 🏥 **Medizin**: Dosis-Wirkungs-Beziehungen
    - 🌡️ **Umwelt**: Temperaturtrends analysieren
    - 📊 **Qualitätskontrolle**: Prozessoptimierung
    """)

    # Next steps
    st.info("""
    💡 **Nächster Schritt**: Multiple Regression für mehrere Prädiktoren!
    Diese erweitert das Konzept auf $y = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + \\dots + \\epsilon$
    """)