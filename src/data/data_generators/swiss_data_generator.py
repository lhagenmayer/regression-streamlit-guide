"""
Swiss data generators for regression analysis.

This module provides functions for generating Swiss canton and weather data
for educational regression analysis examples.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
import streamlit as st

from ...config import get_logger

logger = get_logger(__name__)


def get_swiss_canton_data() -> Dict[str, Dict[str, Any]]:
    """
    Get comprehensive Swiss canton data for regression analysis.
    Includes demographic, economic, and geographic variables.

    Returns:
        Dictionary with canton data suitable for multiple regression
    """
    # Mock data based on real Swiss statistics - in production, fetch from BFS API
    cantons_data = {
        "ZH": {  # Zürich
            "population": 1520968,
            "area_km2": 1729,
            "gdp_per_capita": 95600,
            "unemployment_rate": 3.2,
            "median_income": 8500,
            "foreign_population_pct": 28.5,
            "population_density": 879,
            "life_expectancy": 84.2,
        },
        "BE": {  # Bern
            "population": 1034977,
            "area_km2": 5959,
            "gdp_per_capita": 67800,
            "unemployment_rate": 2.8,
            "median_income": 7200,
            "foreign_population_pct": 18.2,
            "population_density": 174,
            "life_expectancy": 83.8,
        },
        "GE": {  # Geneva
            "population": 499480,
            "area_km2": 282,
            "gdp_per_capita": 89200,
            "unemployment_rate": 4.1,
            "median_income": 7800,
            "foreign_population_pct": 45.8,
            "population_density": 1770,
            "life_expectancy": 84.5,
        },
        "VD": {  # Vaud
            "population": 799145,
            "area_km2": 3212,
            "gdp_per_capita": 71500,
            "unemployment_rate": 3.5,
            "median_income": 6900,
            "foreign_population_pct": 35.2,
            "population_density": 249,
            "life_expectancy": 84.1,
        },
        "TI": {  # Ticino
            "population": 350986,
            "area_km2": 2812,
            "gdp_per_capita": 61200,
            "unemployment_rate": 2.9,
            "median_income": 6500,
            "foreign_population_pct": 22.1,
            "population_density": 125,
            "life_expectancy": 83.9,
        },
    }
    return cantons_data


def generate_swiss_canton_regression_data() -> Dict[str, Any]:
    """
    Generate Swiss canton data for multiple regression analysis.

    Perfect for demonstrating:
    - GDP per capita as dependent variable
    - Population density, foreign population %, unemployment as predictors

    Returns:
        Dictionary with regression-ready data
    """
    cantons = get_swiss_canton_data()
    canton_names = list(cantons.keys())
    canton_data = list(cantons.values())

    # Extract variables for regression
    population_density = [c["population_density"] for c in canton_data]
    foreign_pct = [c["foreign_population_pct"] for c in canton_data]
    unemployment = [c["unemployment_rate"] for c in canton_data]
    gdp_per_capita = [c["gdp_per_capita"] for c in canton_data]

    return {
        "x_population_density": np.array(population_density),
        "x_foreign_pct": np.array(foreign_pct),
        "x_unemployment": np.array(unemployment),
        "y_gdp_per_capita": np.array(gdp_per_capita),
        "canton_names": canton_names,
        "n": len(canton_names),
        "x1_name": "Population Density (per km²)",
        "x2_name": "Foreign Population (%)",
        "x3_name": "Unemployment Rate (%)",
        "y_name": "GDP per Capita (CHF)",
        "data_source": "Swiss Federal Statistical Office (simplified)",
        "description": "Swiss canton data for multiple regression: GDP ~ Population Density + Foreign Population % + Unemployment",
    }


@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_swiss_weather_data() -> pd.DataFrame:
    """
    Fetch Swiss weather data from MeteoSwiss API (when available in 2026).

    Currently returns mock data based on real Swiss climate patterns.
    Perfect for regression analysis of temperature vs. altitude, precipitation patterns, etc.

    Returns:
        DataFrame with weather station data
    """
    # Mock data representing Swiss weather stations
    # In production, this would call: https://opendata.meteoswiss.ch/

    weather_data = pd.DataFrame(
        {
            "station": [
                "Zurich",
                "Bern",
                "Geneva",
                "Basel",
                "Lugano",
                "St. Moritz",
                "Jungfraujoch",
            ],
            "altitude_m": [556, 540, 375, 316, 273, 1822, 3576],
            "avg_temperature_c": [9.4, 8.7, 9.7, 9.3, 11.2, 3.8, -7.1],
            "precipitation_mm": [1042, 1025, 791, 842, 1528, 688, 468],
            "sunshine_hours": [1569, 1598, 2159, 1702, 2164, 1935, 2058],
            "humidity_pct": [74, 76, 72, 75, 73, 68, 55],
        }
    )

    return weather_data


def generate_swiss_weather_regression_data() -> Dict[str, Any]:
    """
    Generate Swiss weather data for regression analysis.

    Perfect for demonstrating:
    - Temperature as dependent variable
    - Altitude as main predictor
    - Additional predictors: sunshine, humidity

    Returns:
        Dictionary with regression-ready weather data
    """
    weather_df = fetch_swiss_weather_data()

    return {
        "x_altitude": weather_df["altitude_m"].values,
        "x_sunshine": weather_df["sunshine_hours"].values,
        "x_humidity": weather_df["humidity_pct"].values,
        "y_temperature": weather_df["avg_temperature_c"].values,
        "station_names": weather_df["station"].tolist(),
        "n": len(weather_df),
        "x1_name": "Altitude (m)",
        "x2_name": "Sunshine Hours",
        "x3_name": "Humidity (%)",
        "y_name": "Average Temperature (°C)",
        "data_source": "MeteoSwiss (simplified)",
        "description": "Swiss weather stations: Temperature ~ Altitude + Sunshine + Humidity",
    }


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_available_swiss_datasets() -> Dict[str, Dict[str, Any]]:
    """
    Return information about all available Swiss datasets for regression.

    Returns:
        Dictionary with dataset metadata
    """
    return {
        "swiss_cantons": {
            "name": "Swiss Cantons Socioeconomic Data",
            "description": "Demographic and economic data for Swiss cantons",
            "variables": ["population_density", "foreign_pct", "unemployment", "gdp_per_capita"],
            "n_observations": 26,
            "source": "BFS (simplified)",
            "ideal_for": "Multiple regression, socioeconomic analysis",
            "api_available": False,  # Would be True when BFS API is fully implemented
            "python_package": "pxwebpy",
            "api_docs": "https://www.pxweb.bfs.admin.ch/api/v1/",
        },
        "swiss_weather": {
            "name": "Swiss Weather Stations",
            "description": "Climate data from Swiss weather stations at different altitudes",
            "variables": ["altitude", "sunshine_hours", "humidity", "temperature"],
            "n_observations": 7,
            "source": "MeteoSwiss (simplified)",
            "ideal_for": "Simple and multiple regression, environmental analysis",
            "api_available": False,  # Would be True in 2026
            "api_docs": "https://opendata.meteoswiss.ch/",
        },
        "cross_border_commuters": {
            "name": "Cross-border Commuters",
            "description": "Foreign workers commuting into Switzerland",
            "variables": ["workers_by_canton", "workers_by_country", "economic_sector"],
            "n_observations": 20000,  # Approximate
            "source": "BFS PX-Web API",
            "api_endpoint": "https://www.pxweb.bfs.admin.ch/pxweb/en/px-x-0302010000_105/",
            "ideal_for": "Time series regression, labor market analysis",
            "api_available": True,
            "python_package": "pxwebpy",
        },
        "dwelling_structure": {
            "name": "Swiss Housing Market",
            "description": "Housing data by canton: rooms, rent, surface area",
            "variables": ["rooms", "monthly_rent", "surface_area", "occupancy_status"],
            "n_observations": 50000,  # Approximate
            "source": "BFS PX-Web API",
            "api_endpoint": "https://www.pxweb.bfs.admin.ch/pxweb/en/px-x-0903020000_111/",
            "ideal_for": "Housing market regression, socioeconomic analysis",
            "api_available": True,
            "python_package": "pxwebpy",
        },
    }