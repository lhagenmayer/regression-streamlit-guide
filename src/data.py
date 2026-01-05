"""
Data generation and handling for the Linear Regression Guide.

This module contains all data generation functions and data manipulation utilities,
including synthetic data generation and integration with Swiss open government data APIs.
"""

from typing import Dict, Optional, Union, Any, List, Tuple
import time
import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.api as sm
from config import CITIES_DATASET, HOUSES_DATASET
from logger import get_logger, log_function_call, log_error_with_context

# Initialize logger for this module
logger = get_logger(__name__)

# ============================================================================
# SWISS OPEN GOVERNMENT DATA INTEGRATION
# ============================================================================


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_bfs_data(table_id: str, variables: Dict[str, list] = None) -> pd.DataFrame:
    """
    Fetch data from Swiss Federal Statistical Office (BFS) PX-Web API.

    Args:
        table_id: BFS table identifier (e.g., 'px-x-1504000000_173')
        variables: Dictionary of variable filters

    Returns:
        DataFrame with BFS data

    Example:
        # Teachers in public schools by canton
        data = fetch_bfs_data('px-x-1504000000_173', {
            'Canton': ['Zurich', 'Bern', 'Geneva'],
            'Year': ['2020', '2021']
        })
    """
    try:
        logger.info(f"Fetching BFS data: table_id={table_id}")
        log_function_call(logger, "fetch_bfs_data", table_id=table_id, variables=variables)

        # For this demo, we'll use mock data since the actual API requires specific table structures
        # In production, you would use: pxwebpy library or direct API calls
        # BFS PX-Web API endpoint would be: "https://www.pxweb.bfs.admin.ch/api/v1/en"

        st.info(f"🔄 BFS API Integration: Would fetch table {table_id}")
        logger.info(f"BFS API mock response returned for table {table_id}")
        return pd.DataFrame()  # Return empty for now

    except Exception as e:
        logger.error(f"BFS API Error for table {table_id}: {e}", exc_info=True)
        log_error_with_context(logger, e, "fetch_bfs_data", table_id=table_id)
        st.error(f"❌ BFS API Error: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_world_bank_data(
    indicators: List[str], countries: List[str] = None, years: List[int] = None
) -> pd.DataFrame:
    """
    Fetch economic indicators from World Bank API using wbgapi.

    Args:
        indicators: List of World Bank indicator codes (e.g., ['NY.GDP.PCAP.KD', 'SP.POP.TOTL'])
        countries: List of country codes (default: major economies)
        years: List of years (default: last 10 years)

    Returns:
        DataFrame with World Bank data

    Example:
        # GDP per capita and population for major economies
        data = fetch_world_bank_data(['NY.GDP.PCAP.KD', 'SP.POP.TOTL'],
                                   ['USA', 'CHN', 'DEU', 'JPN', 'GBR'],
                                   list(range(2010, 2021)))
    """
    try:
        # Default parameters if not provided
        if countries is None:
            countries = ["USA", "CHN", "DEU", "JPN", "GBR", "FRA", "ITA", "CAN", "AUS", "ESP"]

        if years is None:
            years = list(range(2010, 2021))

        # Mock implementation - in production, you would use:
        # import wbgapi as wb
        # df = wb.data.DataFrame(indicators, countries, years)

        st.info(
            f"🔄 World Bank API: Would fetch {len(indicators)} indicators for {len(countries)} countries"
        )

        # Create mock data for demonstration
        mock_data = []
        for country in countries:
            for year in years:
                for indicator in indicators:
                    value = (
                        np.random.normal(1000, 200)
                        if "GDP" in indicator
                        else np.random.normal(50, 10)
                    )
                    mock_data.append(
                        {
                            "country": country,
                            "year": year,
                            "indicator": indicator,
                            "value": max(0, value),  # Ensure non-negative
                        }
                    )

        df = pd.DataFrame(mock_data)
        df = df.pivot(index=["country", "year"], columns="indicator", values="value").reset_index()
        return df

    except Exception as e:
        st.error(f"❌ World Bank API Error: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_fred_data(
    series_ids: List[str], start_date: str = None, end_date: str = None
) -> pd.DataFrame:
    """
    Fetch economic time series from Federal Reserve Economic Data (FRED) API.

    Args:
        series_ids: List of FRED series IDs (e.g., ['GDP', 'UNRATE', 'FEDFUNDS'])
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with FRED time series data

    Example:
        # US GDP, unemployment rate, and federal funds rate
        data = fetch_fred_data(['GDP', 'UNRATE', 'FEDFUNDS'],
                             '2010-01-01', '2023-12-31')
    """
    try:
        # Default dates if not provided
        if start_date is None:
            start_date = "2010-01-01"
        if end_date is None:
            end_date = "2023-12-31"

        # Mock implementation - in production, you would use:
        # from fredapi import Fred
        # fred = Fred(api_key='YOUR_API_KEY')
        # df = pd.DataFrame()
        # for series_id in series_ids:
        #     series_data = fred.get_series(series_id, start_date, end_date)
        #     df[series_id] = series_data

        st.info(
            f"🔄 FRED API: Would fetch {len(series_ids)} series from {start_date} to {end_date}"
        )

        # Create mock time series data
        date_range = pd.date_range(start=start_date, end=end_date, freq="QS")  # Quarterly
        mock_data = {"date": date_range}

        for series_id in series_ids:
            if "GDP" in series_id:
                values = np.cumsum(np.random.normal(100, 20, len(date_range))) + 20000
            elif "UNRATE" in series_id:
                values = np.random.normal(5, 2, len(date_range))
                values = np.clip(values, 0, 15)
            elif "FEDFUNDS" in series_id:
                values = np.random.normal(2.5, 1.5, len(date_range))
                values = np.clip(values, 0, 10)
            else:
                values = np.random.normal(100, 20, len(date_range))

            mock_data[series_id] = values

        df = pd.DataFrame(mock_data)
        return df

    except Exception as e:
        st.error(f"❌ FRED API Error: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_who_health_data(
    indicators: List[str], countries: List[str] = None, years: List[int] = None
) -> pd.DataFrame:
    """
    Fetch health indicators from World Health Organization (WHO) API.

    Args:
        indicators: List of WHO indicator codes
        countries: List of country codes
        years: List of years

    Returns:
        DataFrame with WHO health data

    Example:
        # Life expectancy and mortality rates
        data = fetch_who_health_data(['WHOSIS_000001', 'WHOSIS_000002'],
                                   ['USA', 'CHN', 'DEU'],
                                   list(range(2010, 2021)))
    """
    try:
        # Default parameters
        if countries is None:
            countries = ["USA", "CHN", "DEU", "JPN", "GBR", "FRA", "ITA", "CAN"]

        if years is None:
            years = list(range(2010, 2021))

        if not indicators:
            indicators = ["WHOSIS_000001"]  # Life expectancy

        # Mock implementation - in production, you would use:
        # from apidatawho import WHO
        # who = WHO()
        # df = who.get_data(indicators, countries, years)

        st.info(
            f"🔄 WHO API: Would fetch {len(indicators)} indicators for {len(countries)} countries"
        )

        # Create mock health data
        mock_data = []
        for country in countries:
            for year in years:
                for indicator in indicators:
                    if "WHOSIS_000001" in indicator:  # Life expectancy
                        value = np.random.normal(75, 5)
                        value = np.clip(value, 50, 90)
                    elif "WHOSIS_000002" in indicator:  # Mortality
                        value = np.random.normal(8, 2)
                        value = np.clip(value, 2, 20)
                    else:
                        value = np.random.normal(100, 20)

                    mock_data.append(
                        {"country": country, "year": year, "indicator": indicator, "value": value}
                    )

        df = pd.DataFrame(mock_data)
        df = df.pivot(index=["country", "year"], columns="indicator", values="value").reset_index()
        return df

    except Exception as e:
        st.error(f"❌ WHO API Error: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_eurostat_data(
    dataset_codes: List[str], countries: List[str] = None, years: List[int] = None
) -> pd.DataFrame:
    """
    Fetch socioeconomic data from Eurostat API.

    Args:
        dataset_codes: List of Eurostat dataset codes
        countries: List of country codes
        years: List of years

    Returns:
        DataFrame with Eurostat data

    Example:
        # GDP and employment data for EU countries
        data = fetch_eurostat_data(['nama_10_gdp', 'lfsi_emp_a'],
                                 ['DE', 'FR', 'IT', 'ES', 'NL'],
                                 list(range(2010, 2021)))
    """
    try:
        # Default parameters
        if countries is None:
            countries = ["DE", "FR", "IT", "ES", "NL", "BE", "AT", "SE", "DK", "FI"]

        if years is None:
            years = list(range(2010, 2021))

        if not dataset_codes:
            dataset_codes = ["nama_10_gdp"]  # GDP

        # Mock implementation - in production, you would use:
        # import eurostat
        # df = eurostat.get_data_df(dataset_codes[0], filter_pars={'geo': countries, 'time': years})

        st.info(
            f"🔄 Eurostat API: Would fetch {len(dataset_codes)} datasets for {len(countries)} countries"
        )

        # Create mock socioeconomic data
        mock_data = []
        for country in countries:
            for year in years:
                for dataset in dataset_codes:
                    if "nama_10_gdp" in dataset:  # GDP
                        value = np.random.normal(2000000, 500000)
                    elif "lfsi_emp_a" in dataset:  # Employment
                        value = np.random.normal(70, 5)
                        value = np.clip(value, 50, 90)
                    elif "educ" in dataset:  # Education
                        value = np.random.normal(30, 5)
                        value = np.clip(value, 20, 50)
                    else:
                        value = np.random.normal(100, 20)

                    mock_data.append(
                        {
                            "country": country,
                            "year": year,
                            "dataset": dataset,
                            "value": max(0, value),
                        }
                    )

        df = pd.DataFrame(mock_data)
        df = df.pivot(index=["country", "year"], columns="dataset", values="value").reset_index()
        return df

    except Exception as e:
        st.error(f"❌ Eurostat API Error: {e}")
        return pd.DataFrame()


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


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_global_regression_datasets() -> Dict[str, Dict[str, Any]]:
    """
    Return information about excellent global datasets from free APIs for regression analysis.

    Returns:
        Dictionary with dataset metadata for global APIs
    """
    return {
        "world_bank_indicators": {
            "name": "World Bank Development Indicators",
            "description": "Global economic, social, and environmental indicators for countries worldwide",
            "variables": [
                "gdp_per_capita",
                "population",
                "life_expectancy",
                "unemployment",
                "inflation",
                "trade_balance",
            ],
            "n_observations": "200+ countries × 50+ years",
            "source": "World Bank API",
            "api_endpoint": "https://api.worldbank.org/v2/",
            "python_package": "wbgapi",
            "ideal_for": "Cross-country regression, development economics, time series analysis",
            "api_available": True,
            "api_docs": "https://datahelpdesk.worldbank.org/knowledgebase/topics/125589-developer-information",
            "example_query": "GDP per capita vs. life expectancy across countries",
            "data_frequency": "Annual",
            "geographic_coverage": "Global (200+ countries)",
        },
        "fred_economic_data": {
            "name": "Federal Reserve Economic Data (FRED)",
            "description": "Comprehensive US economic time series from Federal Reserve Bank of St. Louis",
            "variables": [
                "gdp",
                "unemployment_rate",
                "inflation",
                "interest_rates",
                "housing_prices",
                "consumer_spending",
            ],
            "n_observations": "800,000+ time series",
            "source": "Federal Reserve Bank of St. Louis",
            "api_endpoint": "https://fred.stlouisfed.org/docs/api/fred/",
            "python_package": "fredapi",
            "ideal_for": "US macroeconomic analysis, time series regression, business cycles",
            "api_available": True,
            "requires_api_key": True,
            "api_docs": "https://fred.stlouisfed.org/docs/api/fred/",
            "example_query": "Unemployment rate vs. GDP growth (Phillips curve)",
            "data_frequency": "Daily/Monthly/Quarterly",
            "geographic_coverage": "United States",
        },
        "who_health_indicators": {
            "name": "World Health Organization Indicators",
            "description": "Global health statistics and indicators from WHO",
            "variables": [
                "life_expectancy",
                "mortality_rates",
                "disease_incidence",
                "health_expenditure",
                "immunization_rates",
            ],
            "n_observations": "200+ countries × 20+ years",
            "source": "World Health Organization",
            "api_endpoint": "https://ghoapi.azureedge.net/api/",
            "python_package": "apidatawho",
            "ideal_for": "Health economics, epidemiology, global health disparities",
            "api_available": True,
            "api_docs": "https://www.who.int/data/gho/info/gho-odata-api",
            "example_query": "GDP per capita vs. life expectancy (Preston curve)",
            "data_frequency": "Annual",
            "geographic_coverage": "Global",
        },
        "eurostat_european_data": {
            "name": "Eurostat European Statistics",
            "description": "Comprehensive socioeconomic statistics for European countries",
            "variables": [
                "gdp",
                "employment",
                "education",
                "poverty_rates",
                "migration",
                "energy_consumption",
            ],
            "n_observations": "30+ countries × 20+ years",
            "source": "European Commission (Eurostat)",
            "api_endpoint": "https://ec.europa.eu/eurostat/web/json-and-unicode-web-services",
            "python_package": "eurostat",
            "ideal_for": "European integration studies, comparative economics, policy analysis",
            "api_available": True,
            "api_docs": "https://ec.europa.eu/eurostat/web/json-and-unicode-web-services",
            "example_query": "Education spending vs. economic growth across EU countries",
            "data_frequency": "Annual/Quarterly",
            "geographic_coverage": "European Union + EFTA countries",
        },
        "open_weather_historical": {
            "name": "OpenWeather Historical Weather",
            "description": "Historical weather data for cities worldwide",
            "variables": ["temperature", "humidity", "pressure", "wind_speed", "precipitation"],
            "n_observations": "200,000+ cities × historical data",
            "source": "OpenWeather",
            "api_endpoint": "https://openweathermap.org/api/one-call-3",
            "python_package": "pyowm",
            "ideal_for": "Environmental regression, climate impact studies, urban planning",
            "api_available": True,
            "requires_api_key": True,
            "api_docs": "https://openweathermap.org/api/one-call-3",
            "example_query": "Temperature vs. economic productivity by city",
            "data_frequency": "Hourly/Daily",
            "geographic_coverage": "Global cities",
        },
        "nasa_power_earth_science": {
            "name": "NASA POWER Earth Science Data",
            "description": "Global meteorological and solar irradiance data from NASA satellites",
            "variables": [
                "temperature",
                "humidity",
                "solar_radiation",
                "wind_speed",
                "precipitation",
            ],
            "n_observations": "Global grid × 40+ years",
            "source": "NASA Prediction Of Worldwide Energy Resources",
            "api_endpoint": "https://power.larc.nasa.gov/api/temporal/",
            "python_package": "nasapy",
            "ideal_for": "Climate science, renewable energy analysis, agricultural studies",
            "api_available": True,
            "api_docs": "https://power.larc.nasa.gov/docs/services/api/temporal/",
            "example_query": "Solar irradiance vs. GDP by country (renewable energy potential)",
            "data_frequency": "Daily/Monthly",
            "geographic_coverage": "Global (1° × 1° grid)",
        },
    }


def safe_scalar(val: Union[pd.Series, np.ndarray, float, int]) -> float:
    """Konvertiert Series/ndarray zu Skalar, falls nötig."""
    if isinstance(val, (pd.Series, np.ndarray)):
        return float(val.iloc[0] if hasattr(val, "iloc") else val[0])
    return float(val)




@st.cache_data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def generate_multiple_regression_data(
    dataset_choice_mult: str, n_mult: int, noise_mult_level: float, seed_mult: int
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
    # Validate inputs
    if not isinstance(n_mult, int) or n_mult <= 0:
        raise ValueError(f"Sample size n_mult must be a positive integer, got {n_mult}")

    if not isinstance(seed_mult, int):
        raise ValueError(f"Seed seed_mult must be an integer, got {seed_mult}")

    if not isinstance(noise_mult_level, (int, float)) or noise_mult_level < 0:
        raise ValueError(f"Noise level noise_mult_level must be a non-negative number, got {noise_mult_level}")

    logger.info(
        f"Generating multiple regression data: dataset={dataset_choice_mult}, n={n_mult}, noise={noise_mult_level}, seed={seed_mult}"
    )
    start_time = time.time()

    np.random.seed(int(seed_mult))

    if dataset_choice_mult == "🏙️ Städte-Umsatzstudie (75 Städte)":
        x2_preis = np.random.normal(
            CITIES_DATASET["price_mean"], CITIES_DATASET["price_std"], n_mult
        )
        x2_preis = np.clip(x2_preis, CITIES_DATASET["price_min"], CITIES_DATASET["price_max"])
        x3_werbung = np.random.normal(
            CITIES_DATASET["advertising_mean"], CITIES_DATASET["advertising_std"], n_mult
        )
        x3_werbung = np.clip(
            x3_werbung, CITIES_DATASET["advertising_min"], CITIES_DATASET["advertising_max"]
        )
        y_base_mult = 100 - 5 * x2_preis + 8 * x3_werbung
        noise_mult = np.random.normal(0, noise_mult_level, n_mult)
        y_mult = y_base_mult + noise_mult
        y_mult = np.clip(y_mult, CITIES_DATASET["y_min"], CITIES_DATASET["y_max"])
        y_mult = (y_mult - np.mean(y_mult)) / np.std(y_mult) * CITIES_DATASET[
            "y_std_target"
        ] + CITIES_DATASET["y_mean_target"]

        x1_name, x2_name, y_name = "Preis (CHF)", "Werbung (CHF1000)", "Umsatz (1000 CHF)"

    elif dataset_choice_mult == "🇨🇭 Schweizer Kantone (sozioökonomisch)":
        # Use the canton data generation function
        canton_data = generate_swiss_canton_regression_data()
        x_population_density = canton_data["x_population_density"][:n_mult]  # Limit to requested n
        x_foreign_pct = canton_data["x_foreign_pct"][:n_mult]
        x_unemployment = canton_data["x_unemployment"][:n_mult]
        y_gdp_per_capita = canton_data["y_gdp_per_capita"][:n_mult]

        # Return in the expected format for multiple regression
        return {
            "x_population_density": x_population_density,
            "x_foreign_pct": x_foreign_pct,
            "x_unemployment": x_unemployment,
            "y_gdp_per_capita": y_gdp_per_capita,
            "x1_name": canton_data["x1_name"],
            "x2_name": canton_data["x2_name"],
            "x3_name": canton_data["x3_name"],
            "y_name": canton_data["y_name"],
            "data_source": canton_data["data_source"],
            "description": canton_data["description"],
        }

    elif dataset_choice_mult == "🌤️ Schweizer Wetterstationen":
        # Use the weather data generation function
        weather_data = generate_swiss_weather_regression_data()
        x_altitude = weather_data["x_altitude"][:n_mult]  # Limit to requested n
        x_sunshine = weather_data["x_sunshine"][:n_mult]
        x_humidity = weather_data["x_humidity"][:n_mult]
        y_temperature = weather_data["y_temperature"][:n_mult]

        # Return in the expected format for multiple regression
        return {
            "x_altitude": x_altitude,
            "x_sunshine": x_sunshine,
            "x_humidity": x_humidity,
            "y_temperature": y_temperature,
            "x1_name": weather_data["x1_name"],
            "x2_name": weather_data["x2_name"],
            "x3_name": weather_data["x3_name"],
            "y_name": weather_data["y_name"],
            "data_source": weather_data["data_source"],
            "description": weather_data["description"],
        }

    elif dataset_choice_mult == "🏦 World Bank (Länder-Entwicklung)":
        # For multiple regression, use canton data as fallback since World Bank data
        # would need different processing for multiple predictors
        return generate_swiss_canton_regression_data()

    elif dataset_choice_mult == "💰 FRED (US Wirtschaft)":
        # For multiple regression, use canton data as fallback
        return generate_swiss_canton_regression_data()

    elif dataset_choice_mult == "🏥 WHO (Globale Gesundheit)":
        # For multiple regression, use canton data as fallback
        return generate_swiss_canton_regression_data()

    elif dataset_choice_mult == "🏠 Häuserpreise mit Pool (1000 Häuser)":
        x2_wohnflaeche = np.random.normal(
            HOUSES_DATASET["area_mean"], HOUSES_DATASET["area_std"], n_mult
        )
        x2_wohnflaeche = np.clip(
            x2_wohnflaeche, HOUSES_DATASET["area_min"], HOUSES_DATASET["area_max"]
        )

        x3_pool = np.random.binomial(1, HOUSES_DATASET["pool_probability"], n_mult).astype(float)

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

    duration = time.time() - start_time
    logger.info(
        f"Generated multiple regression data in {duration:.3f}s: dataset={dataset_choice_mult}, shape=({n_mult},)"
    )

    return {
        "x2_preis": x2_preis,
        "x3_werbung": x3_werbung,
        "y_mult": y_mult,
        "x1_name": x1_name,
        "x2_name": x2_name,
        "y_name": y_name,
    }


@st.cache_data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def generate_simple_regression_data(
    dataset_choice: str, x_variable: str, n: int, seed: int = 42
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
    # Validate inputs
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"Sample size n must be a positive integer, got {n}")

    if not isinstance(seed, int):
        raise ValueError(f"Seed must be an integer, got {seed}")

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

    elif dataset_choice == "🇨🇭 Schweizer Kantone (sozioökonomisch)":
        # Für Simple Regression: Wähle eine Variable basierend auf x_variable
        canton_data = generate_swiss_canton_regression_data()

        if x_variable == "Population Density":
            x = canton_data["x_population_density"]
            x_label = "Population Density (per km²)"
            y_label = "GDP per Capita (CHF)"
            context_description = """
            Analyse der **26 Schweizer Kantone**:
            - **X** = Bevölkerungsdichte (Einwohner pro km²)
            - **Y** = BIP pro Kopf (in CHF)

            **Erwartung:** Höhere Bevölkerungsdichte → höheres BIP?
            """
        elif x_variable == "Foreign Population %":
            x = canton_data["x_foreign_pct"]
            x_label = "Foreign Population (%)"
            y_label = "GDP per Capita (CHF)"
            context_description = """
            Analyse der **26 Schweizer Kantone**:
            - **X** = Ausländeranteil (%)
            - **Y** = BIP pro Kopf (in CHF)

            **Erwartung:** Mehr Ausländer → höheres BIP? (Urbanisierungseffekt)
            """
        else:  # Default: Unemployment
            x = canton_data["x_unemployment"]
            x_label = "Unemployment Rate (%)"
            y_label = "GDP per Capita (CHF)"
            context_description = """
            Analyse der **26 Schweizer Kantone**:
            - **X** = Arbeitslosenquote (%)
            - **Y** = BIP pro Kopf (in CHF)

            **Erwartung:** Höhere Arbeitslosigkeit → niedrigeres BIP?
            """

        y = canton_data["y_gdp_per_capita"]
        context_title = "Schweizer Kantone: Sozioökonomische Analyse"
        x_unit = "varies"
        y_unit = "CHF"

        return {
            "x": x,
            "y": y,
            "x_label": x_label,
            "y_label": y_label,
            "x_unit": x_unit,
            "y_unit": y_unit,
            "context_title": context_title,
            "context_description": context_description,
        }

    elif dataset_choice == "🌤️ Schweizer Wetterstationen":
        # Für Simple Regression: Wähle eine Variable basierend auf x_variable
        weather_data = generate_swiss_weather_regression_data()

        if x_variable == "Altitude":
            x = weather_data["x_altitude"]
            x_label = "Altitude (m)"
            y_label = "Average Temperature (°C)"
            context_description = """
            **7 Schweizer Wetterstationen** von 273m bis 3576m Höhe:
            - **X** = Höhe über Meer (in m)
            - **Y** = Durchschnittstemperatur (°C)

            **Erwartung:** Höhere Lage → niedrigere Temperatur? (-0.6°C pro 100m)
            """
        elif x_variable == "Sunshine Hours":
            x = weather_data["x_sunshine"]
            x_label = "Sunshine Hours per Year"
            y_label = "Average Temperature (°C)"
            context_description = """
            **7 Schweizer Wetterstationen**:
            - **X** = Sonnenstunden pro Jahr
            - **Y** = Durchschnittstemperatur (°C)

            **Erwartung:** Mehr Sonne → höhere Temperatur?
            """
        else:  # Default: Humidity
            x = weather_data["x_humidity"]
            x_label = "Humidity (%)"
            y_label = "Average Temperature (°C)"
            context_description = """
            **7 Schweizer Wetterstationen**:
            - **X** = Luftfeuchtigkeit (%)
            - **Y** = Durchschnittstemperatur (°C)

            **Erwartung:** Höhere Feuchtigkeit → niedrigere Temperatur?
            """

        y = weather_data["y_temperature"]
        context_title = "Schweizer Wetterstationen: Klimaanalyse"
        x_unit = "varies"
        y_unit = "°C"

        return {
            "x": x,
            "y": y,
            "x_label": x_label,
            "y_label": y_label,
            "x_unit": x_unit,
            "y_unit": y_unit,
            "context_title": context_title,
            "context_description": context_description,
        }

    else:
        # Should not reach here as elektronikmarkt is handled separately
        raise ValueError(f"Unknown dataset: {dataset_choice}")

    return {
        "x": x,
        "y": y,
        "x_label": x_label,
        "y_label": y_label,
        "x_unit": x_unit,
        "y_unit": y_unit,
        "context_title": context_title,
        "context_description": context_description,
    }


@st.cache_data(ttl=300)  # Cache for 5 minutes
def create_dummy_encoded_dataset(
    base_data: Dict[str, np.ndarray],
    categorical_column: str,
    categories: List[str],
    n_samples: int,
    seed: int = 42
) -> Dict[str, Union[pd.DataFrame, np.ndarray]]:
    """
    Create a dataset with dummy variable encoding for categorical data.

    Args:
        base_data: Dictionary with base data arrays
        categorical_column: Name for the categorical column
        categories: List of category names
        n_samples: Number of samples to generate
        seed: Random seed

    Returns:
        Dictionary with original DataFrame and dummy-encoded DataFrame
    """
    np.random.seed(seed)

    # Create categorical data
    categorical_values = np.random.choice(categories, size=n_samples)

    # Create DataFrame with all data
    df_data = {categorical_column: categorical_values}

    # Add numerical data from base_data
    for key, value in base_data.items():
        if isinstance(value, np.ndarray) and len(value) >= n_samples:
            df_data[key] = value[:n_samples]

    df = pd.DataFrame(df_data)

    # Create dummy encoding
    df_encoded = pd.get_dummies(df, columns=[categorical_column], drop_first=True)

    return {
        "original_df": df,
        "encoded_df": df_encoded,
        "categorical_column": categorical_column,
        "categories": categories
    }


@st.cache_data(ttl=300)  # Cache for 5 minutes
def generate_electronics_market_data(
    n: int, true_intercept: float, true_beta: float, noise_level: float, seed: int = 42
) -> Dict[str, Union[np.ndarray, str]]:
    """
    Generate electronics market data for simple regression with interactive parameters.

    Args:
        n: Number of observations
        true_intercept: True intercept (β₀)
        true_beta: True slope (β₁)
        noise_level: Standard deviation of noise
        seed: Random seed

    Returns:
        Dictionary with x, y arrays and metadata
    """
    # Validate inputs
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"Sample size n must be a positive integer, got {n}")

    if not isinstance(seed, int):
        raise ValueError(f"Seed must be an integer, got {seed}")

    if not isinstance(true_intercept, (int, float)):
        raise ValueError(f"True intercept must be a number, got {true_intercept}")

    if not isinstance(true_beta, (int, float)):
        raise ValueError(f"True beta must be a number, got {true_beta}")

    if not isinstance(noise_level, (int, float)) or noise_level < 0:
        raise ValueError(f"Noise level must be a non-negative number, got {noise_level}")

    logger.info(
        f"Generating electronics market data: n={n}, intercept={true_intercept}, beta={true_beta}, noise={noise_level}, seed={seed}"
    )

    np.random.seed(seed)
    x = np.linspace(2, 12, n)  # Verkaufsfläche in 100qm (200-1200qm)
    noise = np.random.normal(0, noise_level, n)
    y = true_intercept + true_beta * x + noise  # Umsatz in Mio. €

    return {
        "x": x,
        "y": y,
        "x_label": "Verkaufsfläche (100qm)",
        "y_label": "Umsatz (Mio. €)",
        "x_unit": "100 qm",
        "y_unit": "Mio. €",
        "context_title": "Elektronikfachmärkte",
        "context_description": "Eine Elektronikmarkt-Kette analysiert den Zusammenhang zwischen Verkaufsfläche und Umsatz. Die Daten zeigen, wie sich eine Vergrößerung der Verkaufsfläche auf den Umsatz auswirkt.",
        "n": n,
        "true_intercept": true_intercept,
        "true_beta": true_beta,
        "noise_level": noise_level,
    }


