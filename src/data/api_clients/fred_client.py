"""
Federal Reserve Economic Data (FRED) API client.

This module provides functions to fetch economic time series from
the Federal Reserve Economic Data (FRED) API.
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd
import streamlit as st

from ...config import get_logger

logger = get_logger(__name__)


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