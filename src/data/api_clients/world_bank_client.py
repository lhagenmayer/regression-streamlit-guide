"""
World Bank API client.

This module provides functions to fetch economic indicators from
the World Bank API using wbgapi.
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd
import streamlit as st

from ...config import get_logger

logger = get_logger(__name__)


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