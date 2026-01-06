"""
World Health Organization (WHO) API client.

This module provides functions to fetch health indicators from
the World Health Organization (WHO) API.
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd
import streamlit as st

from ...config import get_logger

logger = get_logger(__name__)


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