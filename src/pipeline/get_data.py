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
        x_variable: str = "x1",
    ) -> DataResult:
        """
        Get data for simple regression.
        
        Args:
            dataset: Dataset name ("electronics", "advertising", "temperature", "cities", "houses")
            n: Number of observations
            noise: Noise level (standard deviation)
            seed: Random seed for reproducibility
            true_intercept: True β₀ (for simulated data)
            true_slope: True β₁ (for simulated data)
            x_variable: Which X variable to use ("x1" or "x2") for multi-variable datasets
        
        Returns:
            DataResult with x, y arrays and metadata
        """
        logger.info(f"Fetching simple regression data: {dataset}, n={n}, x_variable={x_variable}")
        np.random.seed(seed)
        
        if dataset == "electronics":
            return self._generate_electronics(n, noise, true_intercept, true_slope)
        elif dataset == "advertising":
            return self._generate_advertising(n, noise)
        elif dataset == "temperature":
            return self._generate_temperature(n, noise)
        elif dataset == "cities":
            return self._generate_cities_simple(n, noise, x_variable)
        elif dataset == "houses":
            return self._generate_houses_simple(n, noise, x_variable)
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
            dataset: Dataset name ("cities", "houses", "electronics", "advertising", "temperature")
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
        elif dataset == "electronics":
            return self._generate_electronics_multiple(n, noise)
        elif dataset == "advertising":
            return self._generate_advertising_multiple(n, noise)
        elif dataset == "temperature":
            return self._generate_temperature_multiple(n, noise)
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
    
    # =========================================================
    # Simple Regression versions of Multiple Regression datasets
    # =========================================================
    
    def _generate_cities_simple(
        self, n: int, noise: float, x_variable: str = "x1"
    ) -> DataResult:
        """
        Generate cities data for SIMPLE regression (one variable only).
        
        Educational purpose: Shows larger error term when only using one predictor,
        then students can compare with multiple regression to see improvement.
        """
        # Generate same underlying data as multiple regression
        x1 = np.random.normal(5.69, 0.52, n)  # Preis
        x1 = np.clip(x1, 4.5, 7.0)
        
        x2 = np.random.normal(1.84, 0.83, n)  # Werbung
        x2 = np.clip(x2, 0.5, 3.5)
        
        # True model: y = 120 - 8*Preis + 4*Werbung + error
        y = 120 - 8 * x1 + 4 * x2 + np.random.normal(0, noise, n)
        
        if x_variable == "x2":
            # Nur Werbung → Umsatz (Preis-Effekt wird ignoriert → größerer Fehler!)
            return DataResult(
                x=x2, y=y,
                x_label="Werbeausgaben (1000 CHF)",
                y_label="Umsatz (1000 CHF)",
                x_unit="1000 CHF",
                y_unit="1000 CHF",
                context_title="🏙️ Städte-Studie: Nur Werbung",
                context_description="""
                **Einfache Regression:** Nur Werbeausgaben → Umsatz
                
                ⚠️ **Didaktisch wichtig:** Der Preis-Effekt wird NICHT berücksichtigt!
                → Größerer Fehlerterm als bei multipler Regression.
                
                💡 Wechsle zu **Multipler Regression**, um zu sehen wie sich R² verbessert!
                """,
                extra={"true_b1": 4, "omitted_variable": "Preis", "omitted_effect": -8}
            )
        else:
            # Nur Preis → Umsatz (Werbe-Effekt wird ignoriert → größerer Fehler!)
            return DataResult(
                x=x1, y=y,
                x_label="Preis (CHF)",
                y_label="Umsatz (1000 CHF)",
                x_unit="CHF",
                y_unit="1000 CHF",
                context_title="🏙️ Städte-Studie: Nur Preis",
                context_description="""
                **Einfache Regression:** Nur Preis → Umsatz
                
                ⚠️ **Didaktisch wichtig:** Der Werbe-Effekt wird NICHT berücksichtigt!
                → Größerer Fehlerterm als bei multipler Regression.
                
                💡 Wechsle zu **Multipler Regression**, um zu sehen wie sich R² verbessert!
                """,
                extra={"true_b1": -8, "omitted_variable": "Werbung", "omitted_effect": 4}
            )
    
    def _generate_houses_simple(
        self, n: int, noise: float, x_variable: str = "x1"
    ) -> DataResult:
        """
        Generate houses data for SIMPLE regression (one variable only).
        
        Educational purpose: Shows larger error term when only using one predictor.
        """
        # Generate same underlying data as multiple regression
        x1 = np.random.normal(25.21, 2.92, n)  # Wohnfläche
        x1 = np.clip(x1, 18, 35)
        
        x2 = (np.random.random(n) < 0.204).astype(float)  # Pool
        
        # True model: y = 50 + 8*Fläche + 30*Pool + error
        y = 50 + 8 * x1 + 30 * x2 + np.random.normal(0, noise, n)
        
        if x_variable == "x2":
            # Nur Pool → Preis (Flächen-Effekt wird ignoriert → größerer Fehler!)
            return DataResult(
                x=x2, y=y,
                x_label="Pool (0/1)",
                y_label="Preis ($1000)",
                x_unit="0/1",
                y_unit="$1000",
                context_title="🏠 Häuserpreise: Nur Pool",
                context_description="""
                **Einfache Regression:** Nur Pool → Hauspreis
                
                ⚠️ **Didaktisch wichtig:** Die Wohnfläche wird NICHT berücksichtigt!
                → Größerer Fehlerterm als bei multipler Regression.
                
                💡 Dies ist eine **Dummy-Variable** (0/1). β₁ zeigt den Preisunterschied
                zwischen Häusern MIT vs. OHNE Pool.
                
                ➡️ Wechsle zu **Multipler Regression** für bessere Vorhersage!
                """,
                extra={"true_b1": 30, "omitted_variable": "Wohnfläche", "omitted_effect": 8}
            )
        else:
            # Nur Wohnfläche → Preis (Pool-Effekt wird ignoriert → größerer Fehler!)
            return DataResult(
                x=x1, y=y,
                x_label="Wohnfläche (sqft/10)",
                y_label="Preis ($1000)",
                x_unit="sqft/10",
                y_unit="$1000",
                context_title="🏠 Häuserpreise: Nur Fläche",
                context_description="""
                **Einfache Regression:** Nur Wohnfläche → Hauspreis
                
                ⚠️ **Didaktisch wichtig:** Der Pool-Effekt wird NICHT berücksichtigt!
                → Größerer Fehlerterm als bei multipler Regression.
                
                💡 Wechsle zu **Multipler Regression**, um zu sehen wie sich R² verbessert!
                """,
                extra={"true_b1": 8, "omitted_variable": "Pool", "omitted_effect": 30}
            )
    
    # =========================================================
    # Multiple Regression versions of Simple Regression datasets
    # =========================================================
    
    def _generate_electronics_multiple(self, n: int, noise: float) -> MultipleRegressionDataResult:
        """
        Generate electronics data for MULTIPLE regression.
        
        Adds Marketing budget as second predictor.
        Educational: Shows how R² improves with additional relevant variable.
        """
        # Verkaufsfläche (100 qm)
        x1 = np.random.uniform(2, 10, n)
        
        # Marketingbudget (1000 €) - korreliert leicht positiv mit Fläche
        x2 = 0.3 * x1 + np.random.normal(2.5, 0.8, n)
        x2 = np.clip(x2, 0.5, 5.0)
        
        # Umsatz = 0.6 + 0.52*Fläche + 0.35*Marketing + noise
        y = 0.6 + 0.52 * x1 + 0.35 * x2 + np.random.normal(0, noise, n)
        
        return MultipleRegressionDataResult(
            x1=x1, x2=x2, y=y,
            x1_label="Verkaufsfläche (100 qm)",
            x2_label="Marketingbudget (1000 €)",
            y_label="Umsatz (Mio. €)",
            extra={
                "true_b0": 0.6, 
                "true_b1": 0.52, 
                "true_b2": 0.35,
                "context": "Elektronikmarkt-Kette mit zwei Prädiktoren"
            }
        )
    
    def _generate_advertising_multiple(self, n: int, noise: float) -> MultipleRegressionDataResult:
        """
        Generate advertising data for MULTIPLE regression.
        
        Adds product quality rating as second predictor.
        """
        # Werbeausgaben ($)
        x1 = np.random.uniform(1000, 10000, n)
        
        # Produktqualitäts-Rating (1-10)
        x2 = np.random.uniform(4, 9, n)
        
        # Umsatz = 20000 + 5*Werbung + 8000*Qualität + noise
        y = 20000 + 5.0 * x1 + 8000 * x2 + np.random.normal(0, noise * 5000, n)
        
        return MultipleRegressionDataResult(
            x1=x1, x2=x2, y=y,
            x1_label="Werbeausgaben ($)",
            x2_label="Produktqualität (1-10)",
            y_label="Umsatz ($)",
            extra={
                "true_b0": 20000, 
                "true_b1": 5.0, 
                "true_b2": 8000,
                "context": "Werbestudie mit Qualitätsfaktor"
            }
        )
    
    def _generate_temperature_multiple(self, n: int, noise: float) -> MultipleRegressionDataResult:
        """
        Generate temperature/ice cream data for MULTIPLE regression.
        
        Adds weekend indicator as second predictor.
        """
        # Temperatur (°C)
        x1 = np.random.uniform(15, 35, n)
        
        # Wochenende (0/1) - Dummy-Variable
        x2 = (np.random.random(n) < 0.286).astype(float)  # ~2/7 sind Wochenende
        
        # Eisverkauf = 20 + 3*Temperatur + 25*Wochenende + noise
        y = 20 + 3.0 * x1 + 25 * x2 + np.random.normal(0, noise * 10, n)
        
        return MultipleRegressionDataResult(
            x1=x1, x2=x2, y=y,
            x1_label="Temperatur (°C)",
            x2_label="Wochenende (0/1)",
            y_label="Eisverkauf (Einheiten)",
            extra={
                "true_b0": 20, 
                "true_b1": 3.0, 
                "true_b2": 25,
                "context": "Eisverkauf mit Wochenend-Effekt"
            }
        )
