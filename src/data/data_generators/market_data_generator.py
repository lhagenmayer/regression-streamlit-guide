"""
Market data generator for the Linear Regression Guide.

This module provides functions for generating realistic market datasets
for regression analysis demonstrations.
"""

from typing import Dict, Union, Any, List, Tuple
import numpy as np
import pandas as pd

from ...config import get_logger

logger = get_logger(__name__)


def generate_electronics_market_data(
    n_obs: int = 500,
    time_period_years: int = 3,
    seasonal_amplitude: float = 0.3,
    trend_coefficient: float = 0.02,
    noise_level: float = 0.1,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generate realistic electronics market data with seasonal patterns and trends.

    Args:
        n_obs: Number of observations (monthly data points)
        time_period_years: Length of time series in years
        seasonal_amplitude: Strength of seasonal effects
        trend_coefficient: Linear trend coefficient
        noise_level: Standard deviation of noise
        seed: Random seed

    Returns:
        Dictionary with time series market data
    """
    np.random.seed(seed)

    # Generate time variable (months)
    time_months = np.arange(n_obs)

    # Convert to years for trend calculation
    time_years = time_months / 12.0

    # Base sales level
    base_sales = 1000

    # Linear trend
    trend = trend_coefficient * time_years

    # Seasonal component (monthly pattern)
    seasonal = seasonal_amplitude * np.sin(2 * np.pi * time_months / 12)

    # Marketing spend (predictor)
    marketing_spend = 200 + 50 * np.sin(2 * np.pi * time_months / 12) + np.random.normal(0, 20, n_obs)
    marketing_spend = np.maximum(marketing_spend, 0)  # non-negative

    # Competitor activity (negative effect)
    competitor_activity = 0.5 + 0.3 * np.cos(2 * np.pi * time_months / 12) + np.random.normal(0, 0.1, n_obs)
    competitor_activity = np.clip(competitor_activity, 0, 1)

    # Economic indicator (GDP proxy)
    economic_indicator = 1.0 + 0.1 * np.sin(2 * np.pi * time_months / 12) + np.random.normal(0, 0.05, n_obs)

    # Generate sales with realistic relationship
    sales = (
        base_sales +
        trend * base_sales +  # Trend effect
        seasonal * base_sales +  # Seasonal effect
        2.5 * marketing_spend +  # Marketing effect
        -300 * competitor_activity +  # Competitor effect
        200 * economic_indicator  # Economic effect
    )

    # Add noise
    sales += np.random.normal(0, noise_level * base_sales, n_obs)
    sales = np.maximum(sales, 0)  # non-negative sales

    # Create date index
    start_date = pd.Timestamp('2020-01-01')
    dates = pd.date_range(start_date, periods=n_obs, freq='M')

    return {
        "dates": dates,
        "time_months": time_months,
        "sales": sales,
        "marketing_spend": marketing_spend,
        "competitor_activity": competitor_activity,
        "economic_indicator": economic_indicator,
        "trend": trend,
        "seasonal": seasonal,
        "base_sales": base_sales,
        "trend_coefficient": trend_coefficient,
        "seasonal_amplitude": seasonal_amplitude
    }


def generate_real_estate_market_data(
    n_obs: int = 300,
    city_factors: Dict[str, float] = None,
    time_trend: float = 0.01,
    seasonal_effect: bool = True,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generate realistic real estate market data.

    Args:
        n_obs: Number of observations
        city_factors: Dictionary mapping cities to price multipliers
        time_trend: Monthly price appreciation rate
        seasonal_effect: Whether to include seasonal patterns
        seed: Random seed

    Returns:
        Dictionary with real estate market data
    """
    if city_factors is None:
        city_factors = {
            "Downtown": 1.5,
            "Suburb": 1.0,
            "Rural": 0.7
        }

    np.random.seed(seed)

    # Generate property characteristics
    size_sqm = np.random.normal(150, 40, n_obs)
    size_sqm = np.clip(size_sqm, 50, 500)

    bedrooms = np.random.choice([1, 2, 3, 4, 5], size=n_obs, p=[0.1, 0.2, 0.4, 0.2, 0.1])
    bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3], size=n_obs, p=[0.2, 0.3, 0.3, 0.15, 0.05])

    age_years = np.random.exponential(15, n_obs)  # Older homes are less common
    age_years = np.clip(age_years, 0, 100)

    # Generate location data
    cities = list(city_factors.keys())
    city_data = np.random.choice(cities, size=n_obs)

    # Time variable (months since start)
    time_months = np.random.uniform(0, 60, n_obs)  # 5 years of data

    # Base price calculation
    base_price = 50000  # Base price in currency units

    # Size effect
    price = base_price + 300 * size_sqm

    # Bedroom effect
    price += 15000 * bedrooms

    # Bathroom effect
    price += 10000 * bathrooms

    # Age effect (depreciation)
    price -= 800 * age_years

    # City effect
    for i, city in enumerate(city_data):
        price[i] *= city_factors[city]

    # Time trend (appreciation)
    price *= (1 + time_trend) ** time_months

    # Seasonal effect
    if seasonal_effect:
        seasonal_factor = 1 + 0.05 * np.sin(2 * np.pi * time_months / 12)
        price *= seasonal_factor

    # Add noise
    price += np.random.normal(0, 0.1 * price, n_obs)
    price = np.maximum(price, 10000)  # Minimum price

    return {
        "price": price,
        "size_sqm": size_sqm,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age_years,
        "city": city_data,
        "time_months": time_months,
        "cities": cities,
        "city_factors": city_factors,
        "time_trend": time_trend,
        "seasonal_effect": seasonal_effect
    }


def generate_stock_market_data(
    n_obs: int = 1000,
    n_stocks: int = 5,
    market_volatility: float = 0.02,
    correlation_structure: str = "moderate",
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generate correlated stock market data for portfolio analysis.

    Args:
        n_obs: Number of observations (trading days)
        n_stocks: Number of stocks in portfolio
        market_volatility: Overall market volatility
        correlation_structure: "low", "moderate", or "high" correlation
        seed: Random seed

    Returns:
        Dictionary with stock price data
    """
    np.random.seed(seed)

    # Define correlation structure
    if correlation_structure == "low":
        base_correlation = 0.2
    elif correlation_structure == "moderate":
        base_correlation = 0.5
    elif correlation_structure == "high":
        base_correlation = 0.8
    else:
        base_correlation = 0.5

    # Create correlation matrix
    corr_matrix = np.full((n_stocks, n_stocks), base_correlation)
    np.fill_diagonal(corr_matrix, 1.0)

    # Generate correlated returns
    mean_returns = np.random.normal(0.0005, 0.0002, n_stocks)  # Daily returns
    cholesky = np.linalg.cholesky(corr_matrix)

    # Generate random shocks
    shocks = np.random.normal(0, market_volatility, (n_obs, n_stocks))
    correlated_shocks = shocks @ cholesky.T

    # Calculate returns
    returns = np.tile(mean_returns, (n_obs, 1)) + correlated_shocks

    # Calculate prices (starting from $100)
    prices = np.zeros((n_obs, n_stocks))
    prices[0] = 100.0

    for t in range(1, n_obs):
        prices[t] = prices[t-1] * (1 + returns[t])

    # Create stock names
    stock_names = [f"Stock_{i+1}" for i in range(n_stocks)]

    return {
        "prices": prices,
        "returns": returns,
        "stock_names": stock_names,
        "correlation_matrix": corr_matrix,
        "mean_returns": mean_returns,
        "market_volatility": market_volatility,
        "correlation_structure": correlation_structure,
        "n_obs": n_obs,
        "n_stocks": n_stocks
    }