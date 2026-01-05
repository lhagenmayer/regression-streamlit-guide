"""
Data generation and handling functions for regression analysis.
"""

import numpy as np
import pandas as pd


def safe_scalar(val):
    """Konvertiert Series/ndarray zu Skalar, falls nötig."""
    if isinstance(val, (pd.Series, np.ndarray)):
        return float(val.iloc[0] if hasattr(val, 'iloc') else val[0])
    return float(val)


def generate_dataset(name, seed=42):
    """
    Generiert einen Datensatz basierend auf dem Namen.
    Gibt x, y, labels und Metadaten zurueck.
    """
    np.random.seed(seed)
    
    if name == "elektronikmarkt":
        # Default-Werte, werden durch Slider ueberschrieben
        return None  # Handled separately due to sliders
    
    elif name == "staedte":
        n = 75
        x2_preis = np.random.normal(5.69, 0.52, n)
        x2_preis = np.clip(x2_preis, 4.83, 6.49)
        x3_werbung = np.random.normal(1.84, 0.83, n)
        x3_werbung = np.clip(x3_werbung, 0.50, 3.10)
        y_base = 100 - 5 * x2_preis + 8 * x3_werbung
        noise = np.random.normal(0, 3.5, n)
        y = y_base + noise
        y = np.clip(y, 62.4, 91.2)
        y = (y - np.mean(y)) / np.std(y) * 6.49 + 77.37
        return {
            "x_preis": x2_preis,
            "x_werbung": x3_werbung,
            "y": y,
            "n": n,
            "x1_name": "Preis (CHF)",
            "x2_name": "Werbung (CHF1000)",
            "y_name": "Umsatz (1000 CHF)",
        }
    
    elif name == "haeuser":
        n = 1000
        x_wohnflaeche = np.random.normal(25.21, 2.92, n)
        x_wohnflaeche = np.clip(x_wohnflaeche, 20.03, 30.00)
        x_pool = np.random.binomial(1, 0.204, n).astype(float)
        y_base = 50 + 7.5 * x_wohnflaeche + 35 * x_pool
        noise = np.random.normal(0, 20, n)
        y = y_base + noise
        y = np.clip(y, 134.32, 345.20)
        y = (y - np.mean(y)) / np.std(y) * 42.19 + 247.66
        return {
            "x_wohnflaeche": x_wohnflaeche,
            "x_pool": x_pool,
            "y": y,
            "n": n,
            "x1_name": "Wohnflaeche (sqft/10)",
            "x2_name": "Pool (0/1)",
            "y_name": "Preis (USD)",
        }
    
    return None


def get_signif_stars(p):
    """Signifikanz-Codes wie in R"""
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    if p < 0.1:   return '.'
    return ' '


def get_signif_color(p):
    """Farbe basierend auf Signifikanz"""
    if p < 0.001: return '#006400'
    if p < 0.01:  return '#228B22'
    if p < 0.05:  return '#32CD32'
    if p < 0.1:   return '#FFA500'
    return '#DC143C'


# ---------------------------------------------------------
# ZENTRALE KONFIGURATION: Farben & Schriftgroessen
# ---------------------------------------------------------
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#2c3e50",
    "correct": "#2ecc71",
    "violated": "#e74c3c",
    "warning": "#f39c12",
    "neutral": "#95a5a6",
    "residual_pos": "#27ae60",
    "residual_neg": "#c0392b",
    "regression_line": "#e74c3c",
    "data_points": "#3498db",
    "confidence_band": "#85c1e9",
}

FONT_SIZES = {
    "axis_label_3d": 14,
    "axis_label_2d": 12,
    "title_3d": 14,
    "title_2d": 13,
    "tick_3d": 11,
    "legend": 10,
}
