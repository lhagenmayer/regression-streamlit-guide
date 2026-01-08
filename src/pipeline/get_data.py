"""
Step 1: GET DATA

This module handles all data fetching and generation.
It provides a unified interface to get data from various sources.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np

from ..config import get_logger

logger = get_logger(__name__)


@dataclass
class DataResult:
    """Result from data fetching operation."""
    x: np.ndarray
    y: np.ndarray
    x_label: str
    y_label: str
    x_unit: str = ""
    y_unit: str = ""
    context_title: str = ""
    context_description: str = ""
    extra: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra is None:
            self.extra = {}
    
    @property
    def n(self) -> int:
        """Number of observations."""
        return len(self.x)


@dataclass 
class MultipleRegressionDataResult:
    """Result from multiple regression data fetching."""
    x1: np.ndarray
    x2: np.ndarray
    y: np.ndarray
    x1_label: str
    x2_label: str
    y_label: str
    extra: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra is None:
            self.extra = {}
    
    @property
    def n(self) -> int:
        """Number of observations."""
        return len(self.y)


class DataFetcher:
    """
    Step 1: GET DATA
    
    Fetches or generates data for regression analysis.
    Provides a simple, unified interface for all data sources.
    
    Example:
        fetcher = DataFetcher()
        data = fetcher.get_simple("electronics", n=50, seed=42)
    """
    
    def __init__(self):
        self._generators = {}
        logger.info("DataFetcher initialized")
    
    def get_simple(
        self,
        dataset: str,
        n: int = 50,
        noise: float = 0.4,
        seed: int = 42,
        true_intercept: float = 0.6,
        true_slope: float = 0.52,
    ) -> DataResult:
        """
        Get data for simple regression.
        
        Args:
            dataset: Dataset name ("electronics", "advertising", "temperature")
            n: Number of observations
            noise: Noise level (standard deviation)
            seed: Random seed for reproducibility
            true_intercept: True β₀ (for simulated data)
            true_slope: True β₁ (for simulated data)
        
        Returns:
            DataResult with x, y arrays and metadata
        """
        logger.info(f"Fetching simple regression data: {dataset}, n={n}")
        np.random.seed(seed)
        
        if dataset == "electronics":
            return self._generate_electronics(n, noise, true_intercept, true_slope)
        elif dataset == "advertising":
            return self._generate_advertising(n, noise)
        elif dataset == "temperature":
            return self._generate_temperature(n, noise)
        else:
            # Default synthetic data
            return self._generate_synthetic(n, noise, true_intercept, true_slope)
    
    def get_multiple(
        self,
        dataset: str,
        n: int = 75,
        noise: float = 3.5,
        seed: int = 42,
    ) -> MultipleRegressionDataResult:
        """
        Get data for multiple regression.
        
        Args:
            dataset: Dataset name ("cities", "houses")
            n: Number of observations
            noise: Noise level
            seed: Random seed
        
        Returns:
            MultipleRegressionDataResult with x1, x2, y arrays
        """
        logger.info(f"Fetching multiple regression data: {dataset}, n={n}")
        np.random.seed(seed)
        
        if dataset == "cities":
            return self._generate_cities(n, noise)
        elif dataset == "houses":
            return self._generate_houses(n, noise)
        else:
            return self._generate_cities(n, noise)
    
    # =========================================================
    # PRIVATE: Data Generators
    # =========================================================
    
    def _generate_electronics(
        self, n: int, noise: float, intercept: float, slope: float
    ) -> DataResult:
        """Generate electronics market data (Verkaufsfläche vs Umsatz)."""
        x = np.random.uniform(2, 10, n)  # Verkaufsfläche (100 qm)
        y = intercept + slope * x + np.random.normal(0, noise, n)
        
        return DataResult(
            x=x, y=y,
            x_label="Verkaufsfläche (100 qm)",
            y_label="Umsatz (Mio. €)",
            x_unit="100 qm",
            y_unit="Mio. €",
            context_title="🏪 Elektronikmarkt",
            context_description="Analyse des Zusammenhangs zwischen Verkaufsfläche und Umsatz",
            extra={"true_intercept": intercept, "true_slope": slope}
        )
    
    def _generate_advertising(self, n: int, noise: float) -> DataResult:
        """Generate advertising study data."""
        x = np.random.uniform(1000, 10000, n)  # Werbeausgaben
        y = 50000 + 5.0 * x + np.random.normal(0, noise * 5000, n)
        
        return DataResult(
            x=x, y=y,
            x_label="Werbeausgaben ($)",
            y_label="Umsatz ($)",
            x_unit="$",
            y_unit="$",
            context_title="📢 Werbestudie",
            context_description="Zusammenhang zwischen Werbeausgaben und Umsatz"
        )
    
    def _generate_temperature(self, n: int, noise: float) -> DataResult:
        """Generate temperature vs ice cream sales data."""
        x = np.random.uniform(15, 35, n)  # Temperatur
        y = 20 + 3.0 * x + np.random.normal(0, noise * 10, n)
        
        return DataResult(
            x=x, y=y,
            x_label="Temperatur (°C)",
            y_label="Eisverkauf (Einheiten)",
            x_unit="°C",
            y_unit="Einheiten",
            context_title="🍦 Eisverkauf",
            context_description="Zusammenhang zwischen Temperatur und Eisverkauf"
        )
    
    def _generate_synthetic(
        self, n: int, noise: float, intercept: float, slope: float
    ) -> DataResult:
        """Generate generic synthetic data."""
        x = np.random.uniform(0, 100, n)
        y = intercept + slope * x + np.random.normal(0, noise, n)
        
        return DataResult(
            x=x, y=y,
            x_label="X",
            y_label="Y",
            context_title="Synthetische Daten",
            context_description="Generierte Daten für Demonstrationszwecke"
        )
    
    def _generate_cities(self, n: int, noise: float) -> MultipleRegressionDataResult:
        """Generate cities sales study data (Preis, Werbung → Umsatz)."""
        # Preis (CHF) ~ N(5.69, 0.52)
        x1 = np.random.normal(5.69, 0.52, n)
        x1 = np.clip(x1, 4.5, 7.0)
        
        # Werbung (1000 CHF) ~ N(1.84, 0.83)  
        x2 = np.random.normal(1.84, 0.83, n)
        x2 = np.clip(x2, 0.5, 3.5)
        
        # Umsatz = 120 - 8*Preis + 4*Werbung + noise
        y = 120 - 8 * x1 + 4 * x2 + np.random.normal(0, noise, n)
        
        return MultipleRegressionDataResult(
            x1=x1, x2=x2, y=y,
            x1_label="Preis (CHF)",
            x2_label="Werbung (1000 CHF)",
            y_label="Umsatz (1000 CHF)",
            extra={"true_b0": 120, "true_b1": -8, "true_b2": 4}
        )
    
    def _generate_houses(self, n: int, noise: float) -> MultipleRegressionDataResult:
        """Generate house prices data (Area, Pool → Price)."""
        # Wohnfläche (sqft/10) ~ N(25, 3)
        x1 = np.random.normal(25.21, 2.92, n)
        x1 = np.clip(x1, 18, 35)
        
        # Pool (Dummy: 0 oder 1, ~20% haben Pool)
        x2 = (np.random.random(n) < 0.204).astype(float)
        
        # Preis = 50 + 8*Fläche + 30*Pool + noise
        y = 50 + 8 * x1 + 30 * x2 + np.random.normal(0, noise, n)
        
        return MultipleRegressionDataResult(
            x1=x1, x2=x2, y=y,
            x1_label="Wohnfläche (sqft/10)",
            x2_label="Pool (0/1)",
            y_label="Preis ($1000)",
            extra={"true_b0": 50, "true_b1": 8, "true_b2": 30}
        )
