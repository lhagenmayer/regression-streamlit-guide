"""
Infrastructure: Data Provider Implementation.
Implements IDataProvider using the existing DataFetcher (which uses numpy/pandas).
"""
from typing import Dict, Any, List
from ...core.domain.interfaces import IDataProvider
from ...core.domain.value_objects import DatasetMetadata
from .generators import DataFetcher


class DataProviderImpl(IDataProvider):
    """Concrete implementation of IDataProvider using existing DataFetcher."""
    
    def __init__(self):
        self._fetcher = DataFetcher()
    
    def get_dataset(self, dataset_id: str, n: int, **kwargs) -> Dict[str, Any]:
        """
        Fetch data and convert to dictionary suitable for Use Case.
        Converts numpy arrays to lists for domain layer compatibility.
        """
        noise = kwargs.get("noise", 0.4)
        seed = kwargs.get("seed", 42)
        true_intercept = kwargs.get("true_intercept", 0.6)
        true_slope = kwargs.get("true_slope", 0.52)
        regression_type = kwargs.get("regression_type", "simple")
        
        if regression_type == "multiple":
            result = self._fetcher.get_multiple(dataset_id, n=n, noise=noise, seed=seed)
            return {
                "x1": result.x1.tolist(),
                "x2": result.x2.tolist(),
                "y": result.y.tolist(),
                "x1_label": result.x1_label,
                "x2_label": result.x2_label,
                "y_label": result.y_label,
                "context_title": result.extra.get("context", "Multiple Regression"),
                "context_description": f"Dataset: {dataset_id}",
                "extra": result.extra,
                "metadata": DatasetMetadata(
                    id=dataset_id,
                    name=dataset_id,
                    description="",
                    source="generated",
                    variables=[result.x1_label, result.x2_label, result.y_label],
                    n_observations=result.n
                )
            }
        else:
            result = self._fetcher.get_simple(
                dataset_id, n=n, noise=noise, seed=seed,
                true_intercept=true_intercept, true_slope=true_slope
            )
            return {
                "x": result.x.tolist(),
                "y": result.y.tolist(),
                "x_label": result.x_label,
                "y_label": result.y_label,
                "context_title": result.context_title,
                "context_description": result.context_description,
                "extra": result.extra,
                "metadata": DatasetMetadata(
                    id=dataset_id,
                    name=dataset_id,
                    description=result.context_description,
                    source="generated",
                    variables=[result.x_label, result.y_label],
                    n_observations=result.n
                )
            }
    
    def list_datasets(self) -> List[DatasetMetadata]:
        """List available datasets."""
        # Hardcoded for now, could be dynamic
        return [
            DatasetMetadata("electronics", "Elektronikmarkt", "Verkaufsfläche → Umsatz", "simulated", ["Verkaufsfläche", "Umsatz"], 50),
            DatasetMetadata("cities", "Städtestudie", "Preis & Werbung → Umsatz", "simulated", ["Preis", "Werbung", "Umsatz"], 75),
            DatasetMetadata("houses", "Hauspreise", "Fläche & Pool → Preis", "simulated", ["Fläche", "Pool", "Preis"], 1000),
            DatasetMetadata("cantons", "Schweizer Kantone", "Sozioökonomische Daten", "real", ["Dichte", "Ausländer", "BIP"], 26),
            DatasetMetadata("weather", "Schweizer Wetter", "Klimadaten", "real", ["Höhe", "Sonne", "Temp"], 7),
            DatasetMetadata("world_bank", "World Bank", "Entwicklungsindikatoren", "api", ["GDP", "Education", "LifeExp"], 100),
            DatasetMetadata("fred_economic", "FRED", "US-Wirtschaft", "api", ["Unemployment", "Interest", "GDP"], 50),
            DatasetMetadata("who_health", "WHO", "Globale Gesundheit", "api", ["Spending", "Sanitation", "LifeExp"], 100),
            DatasetMetadata("eurostat", "Eurostat", "EU-Daten", "api", ["Employment", "Education", "GDP"], 100),
            DatasetMetadata("nasa_weather", "NASA POWER", "Klimadaten", "api", ["Temp", "Solar", "Yield"], 50),
        ]
