"""
Dataset-specific content for the Linear Regression Guide.

This module contains all text snippets, LaTeX formulas, descriptions,
and context information that vary depending on the selected dataset.
"""

from typing import Dict, Any


# ============================================================================
# MULTIPLE REGRESSION CONTENT
# ============================================================================

def get_multiple_regression_formulas(dataset_choice_mult: str) -> Dict[str, str]:
    """
    Get LaTeX formulas for multiple regression based on dataset.

    Args:
        dataset_choice_mult: The selected dataset

    Returns:
        Dictionary with 'general' and 'specific' LaTeX formulas
    """
    formulas = {
        "general": r"y_i = \beta_0 + \beta_1 \cdot x_{1i} + \beta_2 \cdot x_{2i} + \cdots + \beta_K \cdot x_{Ki} + \varepsilon_i"
    }

    if dataset_choice_mult == "🏙️ Städte-Umsatzstudie (75 Städte)":
        formulas["specific"] = r"\text{Umsatz}_i = \beta_0 + \beta_1 \cdot \text{Preis}_i + \beta_2 \cdot \text{Werbung}_i + \varepsilon_i"
        formulas["context"] = "Handelskette in 75 Städten"
    elif dataset_choice_mult == "🏠 Häuserpreise mit Pool (1000 Häuser)":
        formulas["specific"] = r"\text{Preis}_i = \beta_0 + \beta_1 \cdot \text{Wohnfläche}_i + \beta_2 \cdot \text{Pool}_i + \varepsilon_i"
        formulas["context"] = "Hausverkäufe in Universitätsstadt"
    elif dataset_choice_mult == "🇨🇭 Schweizer Kantone (sozioökonomisch)":
        formulas["specific"] = r"\text{GDP}_i = \beta_0 + \beta_1 \cdot \text{Population Density}_i + \beta_2 \cdot \text{Foreign \%}_i + \beta_3 \cdot \text{Unemployment}_i + \varepsilon_i"
        formulas["context"] = "Schweizer Kantone Sozioökonomie"
    elif dataset_choice_mult == "🌤️ Schweizer Wetterstationen":
        formulas["specific"] = r"\text{Temperature}_i = \beta_0 + \beta_1 \cdot \text{Altitude}_i + \beta_2 \cdot \text{Sunshine}_i + \beta_3 \cdot \text{Humidity}_i + \varepsilon_i"
        formulas["context"] = "Schweizer Klimastationen"
    else:  # Elektronikmarkt
        formulas["specific"] = r"\text{Umsatz}_i = \beta_0 + \beta_1 \cdot \text{Fläche}_i + \beta_2 \cdot \text{Marketing}_i + \varepsilon_i"
        formulas["context"] = "Elektronikmarkt-Kette"

    return formulas


def get_multiple_regression_descriptions(dataset_choice_mult: str) -> Dict[str, str]:
    """
    Get descriptions and context for multiple regression based on dataset.
    """
    descriptions = {}

    if dataset_choice_mult == "🏙️ Städte-Umsatzstudie (75 Städte)":
        descriptions["main"] = "Eine Handelskette untersucht in **75 Städten** den Zusammenhang zwischen Produktpreis, Werbeausgaben und Umsatz."
        descriptions["variables"] = {
            "x1": "Produktpreis (in CHF)",
            "x2": "Werbeausgaben (in 1'000 CHF)",
            "y": "Umsatz (in 1'000 CHF)"
        }
    elif dataset_choice_mult == "🏠 Häuserpreise mit Pool (1000 Häuser)":
        descriptions["main"] = "Eine Studie von **1000 Hausverkäufen** in einer Universitätsstadt untersucht den Einfluss von Wohnfläche und Pool auf den Hauspreis."
        descriptions["variables"] = {
            "x1": "Wohnfläche (sqft/10)",
            "x2": "Pool vorhanden (0/1)",
            "y": "Hauspreis (USD)"
        }
    elif dataset_choice_mult == "🇨🇭 Schweizer Kantone (sozioökonomisch)":
        descriptions["main"] = "**26 Schweizer Kantone** - Analyse des Zusammenhangs zwischen Bevölkerungsdichte, Ausländeranteil, Arbeitslosigkeit und Wirtschaftskraft."
        descriptions["variables"] = {
            "x1": "Bevölkerungsdichte (pro km²)",
            "x2": "Ausländeranteil (%)",
            "x3": "Arbeitslosenquote (%)",
            "y": "BIP pro Kopf (CHF)"
        }
    elif dataset_choice_mult == "🌤️ Schweizer Wetterstationen":
        descriptions["main"] = "**7 Schweizer Wetterstationen** von 273m bis 3576m Höhe - Untersuchung der Zusammenhänge zwischen geografischen Faktoren und Temperatur."
        descriptions["variables"] = {
            "x1": "Höhe über Meer (m)",
            "x2": "Sonnenstunden pro Jahr",
            "x3": "Luftfeuchtigkeit (%)",
            "y": "Durchschnittstemperatur (°C)"
        }
    else:  # Elektronikmarkt
        descriptions["main"] = "Eine Elektronikmarkt-Kette analysiert **50 Filialen** - Zusammenhang zwischen Verkaufsfläche, Marketingbudget und Umsatz."
        descriptions["variables"] = {
            "x1": "Verkaufsfläche (100 qm)",
            "x2": "Marketingbudget (1'000 €)",
            "y": "Umsatz (Mio. €)"
        }

    return descriptions


# ============================================================================
# SIMPLE REGRESSION CONTENT
# ============================================================================

def get_simple_regression_content(dataset_choice: str, x_variable: str) -> Dict[str, Any]:
    """
    Get all content for simple regression based on dataset and x_variable.

    Returns:
        Dictionary with labels, descriptions, formulas, etc.
    """
    content = {
        "x_label": "X",
        "y_label": "Y",
        "x_unit": "",
        "y_unit": "",
        "context_title": "Regression Analysis",
        "context_description": "Statistical analysis of relationship between variables.",
        "formula_latex": r"y = \beta_0 + \beta_1 \cdot x + \varepsilon"
    }

    # Elektronikmarkt
    if dataset_choice == "🏪 Elektronikmarkt (simuliert)":
        content.update({
            "y_label": "Umsatz (Mio. €)",
            "y_unit": "Mio. €",
            "context_title": "Elektronikmarkt-Analyse",
            "context_description": """
            Eine Elektronikmarkt-Kette analysiert den Zusammenhang zwischen Verkaufsfläche und Umsatz.
            Die Daten zeigen, wie sich eine Vergrößerung der Verkaufsfläche auf den Umsatz auswirkt.
            """
        })

    # Städte-Umsatzstudie
    elif dataset_choice == "🏙️ Städte-Umsatzstudie (75 Städte)":
        if x_variable == "Preis (CHF)":
            content.update({
                "x_label": "Preis (CHF)",
                "y_label": "Umsatz (1'000 CHF)",
                "x_unit": "CHF",
                "y_unit": "1'000 CHF",
                "context_title": "Preisstrategie-Analyse",
                "context_description": """
                Eine Handelskette untersucht in **75 Städten**:
                - **X** = Produktpreis (in CHF)
                - **Y** = Umsatz (in 1'000 CHF)

                **Erwartung:** Höherer Preis → niedrigerer Umsatz?
                """
            })
        else:  # Werbung
            content.update({
                "x_label": "Werbeausgaben (CHF1000)",
                "y_label": "Umsatz (1'000 CHF)",
                "x_unit": "1'000 CHF",
                "y_unit": "1'000 CHF",
                "context_title": "Werbeeffektivität",
                "context_description": """
                Eine Handelskette untersucht in **75 Städten**:
                - **X** = Werbeausgaben (in 1'000 CHF)
                - **Y** = Umsatz (in 1'000 CHF)

                **Erwartung:** Mehr Werbung → höherer Umsatz?
                """
            })

    # Häuserpreise
    elif dataset_choice == "🏠 Häuserpreise mit Pool (1000 Häuser)":
        if x_variable == "Wohnfläche (sqft/10)":
            content.update({
                "x_label": "Wohnfläche (sqft/10)",
                "y_label": "Preis (USD)",
                "x_unit": "sqft/10",
                "y_unit": "USD",
                "context_title": "Wohnflächen-Analyse",
                "context_description": """
                Eine Studie von **1000 Hausverkäufen** in einer Universitätsstadt:
                - **X** = Wohnfläche (in sqft/10, d.h. 20.03 = 200.3 sqft)
                - **Y** = Hauspreis (in USD)

                **Erwartung:** Grössere Wohnfläche → höherer Preis?

                ⚠️ **Didaktisch:** Nur EIN Prädiktor → grosser Fehlerterm
                (Pool-Ausstattung fehlt als Erklärungsvariable!)
                """
            })
        else:  # Pool
            content.update({
                "x_label": "Pool (0/1)",
                "y_label": "Preis (USD)",
                "x_unit": "0/1",
                "y_unit": "USD",
                "context_title": "Pool-Effekt-Analyse",
                "context_description": """
                Eine Studie von **1000 Hausverkäufen** in einer Universitätsstadt:
                - **X** = Pool-Vorhandensein (0 = kein Pool, 1 = Pool vorhanden)
                - **Y** = Hauspreis (in USD)

                **Erwartung:** Pool → höherer Preis? (Dummy-Variable!)

                ⚠️ **Didaktisch:** Dies zeigt den Effekt einer **kategorischen Variable** (Pool ja/nein).
                Nur 20.4% der Häuser haben einen Pool.

                💡 **Interpretation der Steigung β₁:**
                β₁ = durchschnittlicher Preisunterschied zwischen Häusern MIT Pool vs. OHNE Pool
                """
            })

    # Schweizer Kantone
    elif dataset_choice == "🇨🇭 Schweizer Kantone (sozioökonomisch)":
        if x_variable == "Population Density":
            content.update({
                "x_label": "Population Density (per km²)",
                "y_label": "GDP per Capita (CHF)",
                "x_unit": "per km²",
                "y_unit": "CHF",
                "context_title": "Schweizer Kantone: Bevölkerungsdichte",
                "context_description": """
                Analyse der **26 Schweizer Kantone**:
                - **X** = Bevölkerungsdichte (Einwohner pro km²)
                - **Y** = BIP pro Kopf (in CHF)

                **Erwartung:** Höhere Bevölkerungsdichte → höheres BIP?
                """
            })
        elif x_variable == "Foreign Population %":
            content.update({
                "x_label": "Foreign Population (%)",
                "y_label": "GDP per Capita (CHF)",
                "x_unit": "%",
                "y_unit": "CHF",
                "context_title": "Schweizer Kantone: Ausländeranteil",
                "context_description": """
                Analyse der **26 Schweizer Kantone**:
                - **X** = Ausländeranteil (%)
                - **Y** = BIP pro Kopf (in CHF)

                **Erwartung:** Mehr Ausländer → höheres BIP? (Urbanisierungseffekt)
                """
            })
        else:  # Unemployment
            content.update({
                "x_label": "Unemployment Rate (%)",
                "y_label": "GDP per Capita (CHF)",
                "x_unit": "%",
                "y_unit": "CHF",
                "context_title": "Schweizer Kantone: Arbeitslosigkeit",
                "context_description": """
                Analyse der **26 Schweizer Kantone**:
                - **X** = Arbeitslosenquote (%)
                - **Y** = BIP pro Kopf (in CHF)

                **Erwartung:** Höhere Arbeitslosigkeit → niedrigeres BIP?
                """
            })

    # Schweizer Wetterstationen
    elif dataset_choice == "🌤️ Schweizer Wetterstationen":
        if x_variable == "Altitude":
            content.update({
                "x_label": "Altitude (m)",
                "y_label": "Average Temperature (°C)",
                "x_unit": "m",
                "y_unit": "°C",
                "context_title": "Schweizer Wetterstationen: Höhenprofil",
                "context_description": """
                **7 Schweizer Wetterstationen** von 273m bis 3576m Höhe:
                - **X** = Höhe über Meer (in m)
                - **Y** = Durchschnittstemperatur (°C)

                **Erwartung:** Höhere Lage → niedrigere Temperatur? (-0.6°C pro 100m)
                """
            })
        elif x_variable == "Sunshine Hours":
            content.update({
                "x_label": "Sunshine Hours per Year",
                "y_label": "Average Temperature (°C)",
                "x_unit": "hours",
                "y_unit": "°C",
                "context_title": "Schweizer Wetterstationen: Sonnenstrahlung",
                "context_description": """
                **7 Schweizer Wetterstationen**:
                - **X** = Sonnenstunden pro Jahr
                - **Y** = Durchschnittstemperatur (°C)

                **Erwartung:** Mehr Sonne → höhere Temperatur?
                """
            })
        else:  # Humidity
            content.update({
                "x_label": "Humidity (%)",
                "y_label": "Average Temperature (°C)",
                "x_unit": "%",
                "y_unit": "°C",
                "context_title": "Schweizer Wetterstationen: Luftfeuchtigkeit",
                "context_description": """
                **7 Schweizer Wetterstationen**:
                - **X** = Luftfeuchtigkeit (%)
                - **Y** = Durchschnittstemperatur (°C)

                **Erwartung:** Höhere Feuchtigkeit → niedrigere Temperatur?
                """
            })

    # Globale APIs
    elif dataset_choice == "🏦 World Bank (Länder-Entwicklung)":
        content.update({
            "x_label": "GDP per Capita (USD)",
            "y_label": "Life Expectancy (years)",
            "x_unit": "USD",
            "y_unit": "years",
            "context_title": "World Bank: Preston Curve",
            "context_description": """
            Cross-country analysis of GDP per capita vs. life expectancy (Preston Curve) from World Bank data.
            Shows the relationship between economic development and health outcomes.
            """
        })

    elif dataset_choice == "💰 FRED (US Wirtschaft)":
        content.update({
            "x_label": "Unemployment Rate (%)",
            "y_label": "GDP (Billions USD)",
            "x_unit": "%",
            "y_unit": "Billions USD",
            "context_title": "FRED: Phillips Curve",
            "context_description": """
            US economic time series analysis of unemployment rate vs. GDP (Phillips Curve) from Federal Reserve data.
            Examines the relationship between employment and economic output.
            """
        })

    elif dataset_choice == "🏥 WHO (Globale Gesundheit)":
        content.update({
            "x_label": "GDP per Capita (USD)",
            "y_label": "Life Expectancy (years)",
            "x_unit": "USD",
            "y_unit": "years",
            "context_title": "WHO: Global Health",
            "context_description": """
            World Health Organization data analyzing GDP per capita vs. life expectancy across countries.
            Demonstrates global health disparities and economic development relationships.
            """
        })

    return content


def get_dataset_info(dataset_choice: str) -> Dict[str, Any]:
    """
    Get general information about a dataset.
    """
    info = {
        "name": dataset_choice,
        "type": "simulated",
        "source": "Generated",
        "description": "Dataset for regression analysis"
    }

    if "Schweizer" in dataset_choice or "🇨🇭" in dataset_choice or "🌤️" in dataset_choice:
        info.update({
            "type": "real",
            "source": "Switzerland",
            "description": "Authentic Swiss data for educational purposes"
        })
    elif any(api in dataset_choice for api in ["🏦", "💰", "🏥", "🇪🇺"]):
        info.update({
            "type": "api",
            "source": dataset_choice.split()[1] if len(dataset_choice.split()) > 1 else "International",
            "description": "Real data from international organizations"
        })

    return info