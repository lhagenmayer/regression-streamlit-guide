"""
Data generation and handling for the Linear Regression Guide.

This module contains all data generation functions and data manipulation utilities.
"""

from typing import Dict, Optional, Union, Any
import numpy as np
import pandas as pd
import streamlit as st


def safe_scalar(val: Union[pd.Series, np.ndarray, float, int]) -> float:
    """Konvertiert Series/ndarray zu Skalar, falls nötig."""
    if isinstance(val, (pd.Series, np.ndarray)):
        return float(val.iloc[0] if hasattr(val, 'iloc') else val[0])
    return float(val)


@st.cache_data
def generate_dataset(name: str, seed: int = 42) -> Optional[Dict[str, Any]]:
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


@st.cache_data
def generate_multiple_regression_data(
    dataset_choice_mult: str, 
    n_mult: int, 
    noise_mult_level: float, 
    seed_mult: int
) -> Dict[str, Union[np.ndarray, str]]:
    """
    Generate data for multiple regression based on dataset choice.
    
    Args:
        dataset_choice_mult: Name of the dataset
        n_mult: Number of observations
        noise_mult_level: Noise standard deviation
        seed_mult: Random seed
        
    Returns:
        Dictionary with x2_preis, x3_werbung, y_mult, x1_name, x2_name, y_name
    """
    np.random.seed(int(seed_mult))

    if dataset_choice_mult == "🏙️ Städte-Umsatzstudie (75 Städte)":
        x2_preis = np.random.normal(5.69, 0.52, n_mult)
        x2_preis = np.clip(x2_preis, 4.83, 6.49)
        x3_werbung = np.random.normal(1.84, 0.83, n_mult)
        x3_werbung = np.clip(x3_werbung, 0.50, 3.10)
        y_base_mult = 100 - 5 * x2_preis + 8 * x3_werbung
        noise_mult = np.random.normal(0, noise_mult_level, n_mult)
        y_mult = y_base_mult + noise_mult
        y_mult = np.clip(y_mult, 62.4, 91.2)
        y_mult = (y_mult - np.mean(y_mult)) / np.std(y_mult) * 6.49 + 77.37

        x1_name, x2_name, y_name = "Preis (CHF)", "Werbung (CHF1000)", "Umsatz (1000 CHF)"

    elif dataset_choice_mult == "🏠 Häuserpreise mit Pool (1000 Häuser)":
        x2_wohnflaeche = np.random.normal(25.21, 2.92, n_mult)
        x2_wohnflaeche = np.clip(x2_wohnflaeche, 20.03, 30.00)

        x3_pool = np.random.binomial(1, 0.204, n_mult).astype(float)

        y_base_mult = 50 + 7.5 * x2_wohnflaeche + 35 * x3_pool
        noise_mult = np.random.normal(0, noise_mult_level, n_mult)
        y_mult = y_base_mult + noise_mult
        y_mult = np.clip(y_mult, 134.32, 345.20)
        y_mult = (y_mult - np.mean(y_mult)) / np.std(y_mult) * 42.19 + 247.66

        x2_preis = x2_wohnflaeche
        x3_werbung = x3_pool

        x1_name, x2_name, y_name = "Wohnfläche (sqft/10)", "Pool (0/1)", "Preis (USD)"

    else:  # Elektronikmarkt (erweitert)
        x2_flaeche = np.random.uniform(2, 12, n_mult)
        x3_marketing = np.random.uniform(0.5, 5.0, n_mult)

        y_base_mult = 0.6 + 0.48 * x2_flaeche + 0.15 * x3_marketing
        noise_mult = np.random.normal(0, noise_mult_level, n_mult)
        y_mult = y_base_mult + noise_mult

        x2_preis = x2_flaeche
        x3_werbung = x3_marketing

        x1_name, x2_name, y_name = "Verkaufsfläche (100qm)", "Marketing (10k€)", "Umsatz (Mio. €)"

    return {
        'x2_preis': x2_preis,
        'x3_werbung': x3_werbung,
        'y_mult': y_mult,
        'x1_name': x1_name,
        'x2_name': x2_name,
        'y_name': y_name
    }


@st.cache_data
def generate_simple_regression_data(
    dataset_choice: str, 
    x_variable: str, 
    n: int, 
    seed: int = 42
) -> Dict[str, Union[np.ndarray, str]]:
    """
    Generate data for simple regression based on dataset choice.
    
    Args:
        dataset_choice: Name of the dataset
        x_variable: Selected X variable
        n: Number of observations
        seed: Random seed
        
    Returns:
        Dictionary with x, y, x_label, y_label, x_unit, y_unit, context_title, context_description
    """
    np.random.seed(seed)
    
    if dataset_choice == "🏙️ Städte-Umsatzstudie (75 Städte)":
        # Generiere korrelierte Daten basierend auf den deskriptiven Statistiken
        x2_preis = np.random.normal(5.69, 0.52, n)
        x2_preis = np.clip(x2_preis, 4.83, 6.49)
        
        x3_werbung = np.random.normal(1.84, 0.83, n)
        x3_werbung = np.clip(x3_werbung, 0.50, 3.10)
        
        # y = f(preis, werbung) + noise
        y_base = 100 - 5 * x2_preis + 8 * x3_werbung
        noise = np.random.normal(0, 3.5, n)
        y = y_base + noise
        y = np.clip(y, 62.4, 91.2)
        
        # Skaliere auf gewünschte Statistiken
        y = (y - np.mean(y)) / np.std(y) * 6.49 + 77.37
        
        if x_variable == "Preis (CHF)":
            x = x2_preis
            x_label = "Preis (CHF)"
            y_label = "Umsatz (1'000 CHF)"
            x_unit = "CHF"
            y_unit = "1'000 CHF"
            context_description = """
            Eine Handelskette untersucht in **75 Städten**:
            - **X** = Produktpreis (in CHF)
            - **Y** = Umsatz (in 1'000 CHF)
            
            **Erwartung:** Höherer Preis → niedrigerer Umsatz?
            
            ⚠️ **Didaktisch:** Nur EIN Prädiktor → grosser Fehlerterm 
            (Werbung fehlt als Erklärungsvariable!)
            """
        else:  # Werbung
            x = x3_werbung
            x_label = "Werbung (CHF1000)"
            y_label = "Umsatz (1'000 CHF)"
            x_unit = "CHF1000"
            y_unit = "1'000 CHF"
            context_description = """
            Eine Handelskette untersucht in **75 Städten**:
            - **X** = Werbeausgaben (in 1'000 CHF)
            - **Y** = Umsatz (in 1'000 CHF)
            
            **Erwartung:** Mehr Werbung → höherer Umsatz?
            
            ⚠️ **Didaktisch:** Nur EIN Prädiktor → grosser Fehlerterm 
            (Preis fehlt als Erklärungsvariable!)
            """
        
        context_title = "Städte-Umsatzstudie"
        
    elif dataset_choice == "🏠 Häuserpreise mit Pool (1000 Häuser)":
        # Häuserpreise-Datensatz generieren (basierend auf gegebenen Statistiken)
        
        # Wohnfläche in sqft/10 (20.03 bis 30.00, Mittelwert 25.21, SD 2.92)
        x_wohnflaeche = np.random.normal(25.21, 2.92, n)
        x_wohnflaeche = np.clip(x_wohnflaeche, 20.03, 30.00)
        
        # Pool Dummy-Variable (20.4% haben Pool)
        x_pool = np.random.binomial(1, 0.204, n)
        
        # Preis als Funktion von Wohnfläche und Pool
        # Basierend auf: Preis Mittelwert 247.66, SD 42.19, Min 134.32, Max 345.20
        y_base = 50 + 7.5 * x_wohnflaeche + 35 * x_pool
        noise = np.random.normal(0, 20, n)
        y = y_base + noise
        y = np.clip(y, 134.32, 345.20)
        
        # Skaliere auf gewünschte Statistiken
        y = (y - np.mean(y)) / np.std(y) * 42.19 + 247.66
        
        if x_variable == "Wohnfläche (sqft/10)":
            x = x_wohnflaeche
            x_label = "Wohnfläche (sqft/10)"
            y_label = "Preis (USD)"
            x_unit = "sqft/10"
            y_unit = "USD"
            context_description = """
            Eine Studie von **1000 Hausverkäufen** in einer Universitätsstadt:
            - **X** = Wohnfläche (in sqft/10, d.h. 20.03 = 200.3 sqft)
            - **Y** = Hauspreis (in USD)
            
            **Erwartung:** Grössere Wohnfläche → höherer Preis?
            
            ⚠️ **Didaktisch:** Nur EIN Prädiktor → grosser Fehlerterm 
            (Pool-Ausstattung fehlt als Erklärungsvariable!)
            """
        else:  # Pool
            x = x_pool.astype(float)
            x_label = "Pool (0/1)"
            y_label = "Preis (USD)"
            x_unit = "0/1"
            y_unit = "USD"
            context_description = """
            Eine Studie von **1000 Hausverkäufen** in einer Universitätsstadt:
            - **X** = Pool-Vorhandensein (0 = kein Pool, 1 = Pool vorhanden)
            - **Y** = Hauspreis (in USD)
            
            **Erwartung:** Pool → höherer Preis? (Dummy-Variable!)
            
            ⚠️ **Didaktisch:** Dies zeigt den Effekt einer **kategorischen Variable** (Pool ja/nein).
            Nur 20.4% der Häuser haben einen Pool.
            
            💡 **Interpretation der Steigung β₁:**
            β₁ = durchschnittlicher Preisunterschied zwischen Häusern MIT Pool vs. OHNE Pool
            """
        
        context_title = "Häuserpreise-Studie"
    
    else:
        # Should not reach here as elektronikmarkt is handled separately
        raise ValueError(f"Unknown dataset: {dataset_choice}")
    
    return {
        'x': x,
        'y': y,
        'x_label': x_label,
        'y_label': y_label,
        'x_unit': x_unit,
        'y_unit': y_unit,
        'context_title': context_title,
        'context_description': context_description
    }
