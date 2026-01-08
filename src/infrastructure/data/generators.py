"""
Step 1: GET DATA

This module handles all data fetching and generation.
It provides a unified interface to get data from various sources.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import requests
from typing import Dict, Any, Optional, List, Union

from ...config import get_logger

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
        
        result = None
        if dataset == "electronics":
            result = self._generate_electronics(n, noise, true_intercept, true_slope)
        elif dataset == "advertising":
            result = self._generate_advertising(n, noise)
        elif dataset == "temperature":
            result = self._generate_temperature(n, noise)
        elif dataset == "cantons":
             res = self._generate_cantons(n, noise)
             result = DataResult(x=res.x1, y=res.y, x_label=res.x1_label, y_label=res.y_label, context_title="Swiss Cantons", context_description=res.extra.get("context", ""))
        elif dataset == "weather":
             res = self._generate_weather(n, noise)
             # Map Altitude (x1) -> Temperature (y)
             result = DataResult(x=res.x1, y=res.y, x_label=res.x1_label, y_label=res.y_label, context_title="Swiss Weather", context_description="Altitude -> Temperature")
        elif dataset == "world_bank":
             res = self._generate_world_bank(n, noise)
             result = DataResult(x=res.x1, y=res.y, x_label=res.x1_label, y_label=res.y_label, context_title="World Bank", context_description="GDP -> Life Exp")
        elif dataset == "fred_economic":
             res = self._generate_fred(n, noise)
             result = DataResult(x=res.x1, y=res.y, x_label=res.x1_label, y_label=res.y_label, context_title="FRED", context_description="Unemployment -> GDP")
        elif dataset == "who_health":
             res = self._generate_who(n, noise)
             result = DataResult(x=res.x1, y=res.y, x_label=res.x1_label, y_label=res.y_label, context_title="WHO", context_description="Health Spend -> Life Exp")
        elif dataset == "eurostat":
             res = self._generate_eurostat(n, noise)
             result = DataResult(x=res.x1, y=res.y, x_label=res.x1_label, y_label=res.y_label, context_title="Eurostat", context_description="Emp -> GDP")
        elif dataset == "nasa_weather":
             res = self._generate_nasa(n, noise)
             result = DataResult(x=res.x1, y=res.y, x_label=res.x1_label, y_label=res.y_label, context_title="NASA", context_description="Temp -> Crop Yield")
        else:
            # Default synthetic data
            result = self._generate_synthetic(n, noise, true_intercept, true_slope)
            
        if result:
            result.extra["dataset"] = dataset
            
        return result
    
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
        
        result = None
        if dataset == "cities":
            result = self._generate_cities(n, noise)
        elif dataset == "houses":
            result = self._generate_houses(n, noise)
        elif dataset == "cantons":
             result = self._generate_cantons(n, noise)
        elif dataset == "weather":
             result = self._generate_weather(n, noise)
        elif dataset == "world_bank":
            result = self._generate_world_bank(n, noise)
        elif dataset == "fred_economic":
            result = self._generate_fred(n, noise)
        elif dataset == "who_health":
            result = self._generate_who(n, noise)
        elif dataset == "eurostat":
             result = self._generate_eurostat(n, noise)
        elif dataset == "nasa_weather":
             result = self._generate_nasa(n, noise)
        else:
            result = self._generate_cities(n, noise)
            
        if result:
            result.extra["dataset"] = dataset
            
        return result
    
    # =========================================================
    # PRIVATE: External Data Fetchers (Mocked for Stability)
    # =========================================================

    def _fetch_world_bank(self, indicators: List[str], countries: List[str] = None, years: List[int] = None) -> pd.DataFrame:
        """Fetch/Mock World Bank data."""
        try:
            if countries is None:
                countries = ['USA', 'CHN', 'DEU', 'JPN', 'GBR', 'FRA', 'ITA', 'CAN', 'AUS', 'ESP']
            if years is None:
                years = list(range(2010, 2021))

            logger.info(f"World Bank API: Fetching {len(indicators)} indicators for {len(countries)} countries")

            # Mock data
            mock_data = []
            for country in countries:
                for year in years:
                    for indicator in indicators:
                        value = np.random.normal(1000, 200) if 'GDP' in indicator else np.random.normal(50, 10)
                        mock_data.append({
                            'country': country, 'year': year, 'indicator': indicator,
                            'value': max(0, value)
                        })
            
            df = pd.DataFrame(mock_data)
            return df.pivot(index=['country', 'year'], columns='indicator', values='value').reset_index()
        except Exception as e:
            logger.error(f"World Bank API Error: {e}")
            return pd.DataFrame()

    def _fetch_fred(self, series_ids: List[str], start_date: str = '2010-01-01', end_date: str = '2023-12-31') -> pd.DataFrame:
        """Fetch/Mock FRED data."""
        try:
            logger.info(f"FRED API: Fetching {len(series_ids)} series")
            date_range = pd.date_range(start=start_date, end=end_date, freq='QS')
            mock_data = {'date': date_range}

            for series_id in series_ids:
                if 'GDP' in series_id:
                    values = np.cumsum(np.random.normal(100, 20, len(date_range))) + 20000
                elif 'UNRATE' in series_id:
                    values = np.clip(np.random.normal(5, 2, len(date_range)), 0, 15)
                else:
                    values = np.random.normal(100, 20, len(date_range))
                mock_data[series_id] = values

            return pd.DataFrame(mock_data)
        except Exception as e:
            logger.error(f"FRED API Error: {e}")
            return pd.DataFrame()

    def _fetch_who(self, indicators: List[str], countries: List[str] = None, years: List[int] = None) -> pd.DataFrame:
        """Fetch/Mock WHO data."""
        try:
            if countries is None:
                 countries = ['USA', 'CHN', 'DEU', 'JPN', 'GBR', 'FRA', 'ITA', 'CAN']
            if years is None:
                years = list(range(2010, 2021))
                
            logger.info(f"WHO API: Fetching {len(indicators)} indicators")
            
            mock_data = []
            for country in countries:
                for year in years:
                    for indicator in indicators:
                        if 'WHOSIS_000001' in indicator: value = np.clip(np.random.normal(75, 5), 50, 90)
                        else: value = np.random.normal(100, 20)
                        mock_data.append({'country': country, 'year': year, 'indicator': indicator, 'value': value})
            
            df = pd.DataFrame(mock_data)
            return df.pivot(index=['country', 'year'], columns='indicator', values='value').reset_index()
        except Exception as e:
            logger.error(f"WHO API Error: {e}")
            return pd.DataFrame()

    def _fetch_eurostat(self, dataset_codes: List[str], countries: List[str] = None, years: List[int] = None) -> pd.DataFrame:
        """Fetch/Mock Eurostat data."""
        try:
            if countries is None: countries = ['DE', 'FR', 'IT', 'ES', 'NL', 'BE', 'AT', 'SE', 'DK', 'FI']
            if years is None: years = list(range(2010, 2021))
            
            logger.info(f"Eurostat API: Fetching {len(dataset_codes)} datasets")
            mock_data = []
            for country in countries:
                for year in years:
                    for ds in dataset_codes:
                        if 'gdp' in ds: val = np.random.normal(2000000, 500000)
                        elif 'emp' in ds: val = np.clip(np.random.normal(70, 5), 50, 90)
                        else: val = np.random.normal(30, 5)
                        mock_data.append({'country': country, 'year': year, 'dataset': ds, 'value': max(0, val)})
            
            df = pd.DataFrame(mock_data)
            return df.pivot(index=['country', 'year'], columns='dataset', values='value').reset_index()
        except Exception as e:
            logger.error(f"Eurostat Error: {e}")
            return pd.DataFrame()

    def _fetch_nasa(self, variables: List[str], locations: int = 50) -> pd.DataFrame:
        """Fetch/Mock NASA POWER data."""
        try:
            logger.info(f"NASA POWER API: Fetching data for {locations} locations")
            # Lat/Lon grid
            lats = np.random.uniform(-50, 50, locations)
            # Solar Radiation (kWh/m^2/day)
            solar = np.abs(lats) * -0.1 + 8 + np.random.normal(0, 1, locations)
            # Temperature (C)
            temp = 30 - np.abs(lats) * 0.5 + np.random.normal(0, 3, locations)
            
            return pd.DataFrame({'lat': lats, 'solar': solar, 'temp': temp})
        except Exception as e:
            logger.error(f"NASA API Error: {e}")
            return pd.DataFrame()

    def _generate_eurostat(self, n: int, noise: float) -> MultipleRegressionDataResult:
        """Generate Eurostat data."""
        es = self._fetch_eurostat(['nama_10_gdp', 'lfsi_emp_a'])
        if es.empty: return self._generate_cities(n, noise)
        
        # Predict GDP based on Employment
        emp = es['lfsi_emp_a'].values
        gdp = es['nama_10_gdp'].values
        # 2nd predictor: Education Index (mock)
        edu = np.clip(0.00001 * gdp + np.random.normal(0, 5, len(gdp)), 10, 60)
        
        return MultipleRegressionDataResult(
            x1=emp, x2=edu, y=gdp,
            x1_label="Employment Rate (%)",
            x2_label="Tertiary Education (%)",
            y_label="GDP (Million €)",
            extra={"context": "Eurostat Socioeconomic Data"}
        )

    def _generate_nasa(self, n: int, noise: float) -> MultipleRegressionDataResult:
        """Generate NASA data."""
        nasa = self._fetch_nasa([], locations=n)
        if nasa.empty: return self._generate_cities(n, noise)
        
        temp = nasa['temp'].values
        solar = nasa['solar'].values
        # Predict Crop Yield (for example) using Temp and Solar
        yield_val = 20 + 2 * temp + 5 * solar + np.random.normal(0, noise, len(temp))
        
        return MultipleRegressionDataResult(
            x1=temp, x2=solar, y=yield_val,
            x1_label="Temperature (°C)",
            x2_label="Solar Radiation (kWh/m²/day)",
            y_label="Crop Yield Index",
            extra={"context": "NASA POWER Agro-Climatology"}
        )

    def _generate_world_bank(self, n: int, noise: float) -> MultipleRegressionDataResult:
        """Generate World Bank data (GDP -> Life Expectancy)."""
        wb_data = self._fetch_world_bank(['NY.GDP.PCAP.KD', 'SP.DYN.LE00.IN'])
        
        if wb_data.empty: return self._generate_cities(n, noise) # Fallback

        gdp = wb_data['NY.GDP.PCAP.KD'].fillna(wb_data['NY.GDP.PCAP.KD'].mean()).values
        life = wb_data['SP.DYN.LE00.IN'].fillna(wb_data['SP.DYN.LE00.IN'].mean()).values
        
        # We need 2 predictors for MultipleRegressionDataResult. 
        # API expects x1, x2. Let's create a dummy or split GDP if needed.
        # Or better: "Education" as 2nd predictor (mocked)
        education = np.clip(0.0005 * gdp + np.random.normal(0, 2, len(gdp)), 5, 20)
        
        return MultipleRegressionDataResult(
            x1=gdp, x2=education, y=life,
            x1_label="GDP per Capita (USD)",
            x2_label="Education Years",
            y_label="Life Expectancy (years)",
            extra={"context": "World Bank Development Indicators"}
        )

    def _generate_fred(self, n: int, noise: float) -> MultipleRegressionDataResult:
        """Generate FRED data (Unemployment -> GDP)."""
        fred = self._fetch_fred(['GDP', 'UNRATE'])
        
        if fred.empty: return self._generate_cities(n, noise)
        
        unrate = fred['UNRATE'].values
        gdp = fred['GDP'].values
        # 2nd predictor: Interest Rate (Inverse to GDP usually)
        interest = np.clip(5 - 0.0001 * gdp + np.random.normal(0, 1, len(gdp)), 0, 10)
        
        return MultipleRegressionDataResult(
            x1=unrate, x2=interest, y=gdp,
            x1_label="Unemployment Rate (%)",
            x2_label="Interest Rate (%)",
            y_label="GDP (Billions USD)",
            extra={"context": "FRED US Economic Data"}
        )

    def _generate_who(self, n: int, noise: float) -> MultipleRegressionDataResult:
        """Generate WHO data."""
        who = self._fetch_who(['WHOSIS_000001'])
        if who.empty: return self._generate_cities(n, noise)
        
        life_exp = who['WHOSIS_000001'].fillna(75).values
        # Inverse problem: We usually predict Life Exp. 
        # Let's mock Predictors: Healthcare Spend, Sanitation
        spend = (life_exp - 50) * 100 + np.random.normal(0, 500, len(life_exp))
        sanitation = np.clip((life_exp - 40) * 2 + np.random.normal(0, 5, len(life_exp)), 0, 100)
        
        return MultipleRegressionDataResult(
            x1=spend, x2=sanitation, y=life_exp,
            x1_label="Health Expenditure ($)",
            x2_label="Sanitation Access (%)",
            y_label="Life Expectancy (years)",
            extra={"context": "WHO Global Health Indicators"}
        )
    
    # =========================================================
    # PRIVATE: Data Generators
    # =========================================================
    
    def _generate_cantons(self, n: int, noise: float) -> MultipleRegressionDataResult:
        """
        Generate Swiss Cantons data (Density, Foreigners, Unemployment -> GDP).
        Real-ish parameters based on Swiss socio-economic data.
        """
        # Feature 1: Population Density (log-normal distributed)
        # Most cantons < 500, some (Zurich, Geneva, Basel) very high
        x1 = np.random.lognormal(5.5, 0.8, n)
        x1 = np.clip(x1, 50, 5000)
        
        # Feature 2: Foreign Population % (15% to 50%)
        x2 = np.random.normal(25, 8, n)
        x2 = np.clip(x2, 10, 50)
        
        # Feature 3: Unemployment Rate (1.5% to 5.0%)
        # Correlated with Foreign % (slightly)
        x3 = 1.5 + 0.05 * x2 + np.random.normal(0, 0.5, n)
        x3 = np.clip(x3, 0.5, 6.0)
        
        # GDP per Capita (CHF)
        # Base 60k + Density bonus + Foreign bonus (skilled expat effect) - Unempl malus
        # Note: In Switzerland, high foreign % often correlates with high GDP (Zurich/Geneva/Basel)
        y = 55000 + 5 * x1 + 800 * x2 - 2000 * x3 + np.random.normal(0, noise * 2000, n)
        
        # For simple 2D regression compatibility, we return first two X
        # But maybe we should return a dedicated result for >2 variables?
        # For now, we map x3 to 'extra' or handle it if the system supports >2
        # current MultipleRegressionDataResult only supports x1, x2. 
        # Wait, the content says x1, x2, x3. We need to handle this.
        # But result struct is x1, x2.
        # I will pass x3 in 'extra' and rely on the pipeline to ignoring it or 
        # update the struct. 
        # Looking at DataResult, it seems strictly 2 predictors?
        # MultipleRegressionDataResult: x1, x2.
        # If I want 3, I need to upgrade the data/pipeline structure or 
        # simplify the dataset to 2 variables for this version.
        # "GDP = Density + Foreign %" is decent.
        
        return MultipleRegressionDataResult(
            x1=x1, x2=x2, y=y,
            x1_label="Bevölkerungsdichte (Einwohner/km²)",
            x2_label="Ausländeranteil (%)",
            y_label="BIP pro Kopf (CHF)",
            extra={
                "true_b0": 55000, "true_b1": 5, "true_b2": 800,
                "context": "Schweizer Kantone (Sozioökonomisch)"
            }
        )

    def _generate_weather(self, n: int, noise: float) -> MultipleRegressionDataResult:
        """
        Generate Swiss Weather stations data (Altitude, Sunshine -> Temperature).
        """
        # x1: Altitude (meters) - 300 to 3500
        x1 = np.random.uniform(300, 2500, n)
        
        # x2: Sunshine Hours (per year) - 1200 to 2500
        # Higher altitude often more sun (above fog)
        x2 = 1500 + 0.1 * x1 + np.random.normal(0, 200, n)
        x2 = np.clip(x2, 1000, 3000)
        
        # Temperature decreases with altitude (-0.65°C per 100m)
        # Increases with sunshine
        y = 15 - 0.0065 * x1 + 0.002 * x2 + np.random.normal(0, noise, n)
        
        return MultipleRegressionDataResult(
            x1=x1, x2=x2, y=y,
            x1_label="Höhe über Meer (m)",
            x2_label="Sonnenstunden (h/Jahr)",
            y_label="Jahresmitteltemperatur (°C)",
            extra={
                "true_b0": 15, "true_b1": -0.0065, "true_b2": 0.002,
                "context": "Schweizer Wetterstationen"
            }
        )
    
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
