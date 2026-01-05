"""
Unit tests for data generation functions in data.py

Tests cover:
- safe_scalar function for type conversion
- generate_dataset for all dataset types
- generate_multiple_regression_data with all variations
- generate_simple_regression_data with all variations
- Edge cases and error handling
"""

import pytest
import numpy as np
import pandas as pd
from data import (
    safe_scalar,
    generate_dataset,
    generate_multiple_regression_data,
    generate_simple_regression_data,
    generate_swiss_canton_regression_data,
    generate_swiss_weather_regression_data,
    fetch_world_bank_data,
    fetch_fred_data,
    fetch_who_health_data,
    get_available_swiss_datasets,
    get_global_regression_datasets
)


class TestSafeScalar:
    """Test the safe_scalar utility function."""
    
    @pytest.mark.unit
    def test_safe_scalar_with_float(self):
        """Test safe_scalar with float input."""
        result = safe_scalar(3.14)
        assert isinstance(result, float)
        assert result == 3.14
    
    @pytest.mark.unit
    def test_safe_scalar_with_int(self):
        """Test safe_scalar with int input."""
        result = safe_scalar(42)
        assert isinstance(result, float)
        assert result == 42.0
    
    @pytest.mark.unit
    def test_safe_scalar_with_numpy_array(self):
        """Test safe_scalar with numpy array."""
        arr = np.array([2.5, 3.5, 4.5])
        result = safe_scalar(arr)
        assert isinstance(result, float)
        assert result == 2.5
    
    @pytest.mark.unit
    def test_safe_scalar_with_pandas_series(self):
        """Test safe_scalar with pandas Series."""
        series = pd.Series([1.5, 2.5, 3.5])
        result = safe_scalar(series)
        assert isinstance(result, float)
        assert result == 1.5
    
    @pytest.mark.unit
    def test_safe_scalar_with_single_element_array(self):
        """Test safe_scalar with single element array."""
        arr = np.array([7.7])
        result = safe_scalar(arr)
        assert isinstance(result, float)
        assert result == 7.7


class TestGenerateDataset:
    """Test the generate_dataset function."""
    
    @pytest.mark.unit
    def test_generate_dataset_elektronikmarkt(self):
        """Test generate_dataset returns None for elektronikmarkt (handled by sliders)."""
        result = generate_dataset("elektronikmarkt", seed=42)
        assert result is None
    
    @pytest.mark.unit
    def test_generate_dataset_staedte(self):
        """Test generate_dataset for Städte-Umsatzstudie."""
        result = generate_dataset("staedte", seed=42)
        
        assert result is not None
        assert "x_preis" in result
        assert "x_werbung" in result
        assert "y" in result
        assert "n" in result
        assert result["n"] == 75
        
        # Check array shapes
        assert len(result["x_preis"]) == 75
        assert len(result["x_werbung"]) == 75
        assert len(result["y"]) == 75
        
        # Check variable names
        assert result["x1_name"] == "Preis (CHF)"
        assert result["x2_name"] == "Werbung (CHF1000)"
        assert result["y_name"] == "Umsatz (1000 CHF)"
    
    @pytest.mark.unit
    def test_generate_dataset_haeuser(self):
        """Test generate_dataset for Häuserpreise."""
        result = generate_dataset("haeuser", seed=42)
        
        assert result is not None
        assert "x_wohnflaeche" in result
        assert "x_pool" in result
        assert "y" in result
        assert "n" in result
        assert result["n"] == 1000
        
        # Check array shapes
        assert len(result["x_wohnflaeche"]) == 1000
        assert len(result["x_pool"]) == 1000
        assert len(result["y"]) == 1000
        
        # Check pool is binary
        assert set(result["x_pool"]) == {0.0, 1.0} or set(result["x_pool"]) == {0.0} or set(result["x_pool"]) == {1.0}
        
        # Check variable names
        assert result["x1_name"] == "Wohnflaeche (sqft/10)"
        assert result["x2_name"] == "Pool (0/1)"
        assert result["y_name"] == "Preis (USD)"
    
    @pytest.mark.unit
    def test_generate_dataset_unknown(self):
        """Test generate_dataset with unknown dataset name."""
        result = generate_dataset("unknown", seed=42)
        assert result is None
    
    @pytest.mark.unit
    def test_generate_dataset_reproducibility(self):
        """Test that same seed produces same results."""
        result1 = generate_dataset("staedte", seed=123)
        result2 = generate_dataset("staedte", seed=123)
        
        np.testing.assert_array_equal(result1["x_preis"], result2["x_preis"])
        np.testing.assert_array_equal(result1["x_werbung"], result2["x_werbung"])
        np.testing.assert_array_equal(result1["y"], result2["y"])


class TestGenerateMultipleRegressionData:
    """Test the generate_multiple_regression_data function."""
    
    @pytest.mark.unit
    def test_staedte_dataset(self):
        """Test multiple regression data generation for Städte."""
        result = generate_multiple_regression_data(
            "🏙️ Städte-Umsatzstudie (75 Städte)",
            n_mult=75,
            noise_mult_level=3.5,
            seed_mult=42
        )
        
        assert "x2_preis" in result
        assert "x3_werbung" in result
        assert "y_mult" in result
        
        # Check shapes
        assert len(result["x2_preis"]) == 75
        assert len(result["x3_werbung"]) == 75
        assert len(result["y_mult"]) == 75
        
        # Check variable names
        assert result["x1_name"] == "Preis (CHF)"
        assert result["x2_name"] == "Werbung (CHF1000)"
        assert result["y_name"] == "Umsatz (1000 CHF)"
        
        # Check data ranges (should be clipped)
        assert np.all(result["x2_preis"] >= 4.83)
        assert np.all(result["x2_preis"] <= 6.49)
        assert np.all(result["x3_werbung"] >= 0.50)
        assert np.all(result["x3_werbung"] <= 3.10)
    
    @pytest.mark.unit
    def test_haeuser_dataset(self):
        """Test multiple regression data generation for Häuser."""
        result = generate_multiple_regression_data(
            "🏠 Häuserpreise mit Pool (1000 Häuser)",
            n_mult=1000,
            noise_mult_level=20.0,
            seed_mult=42
        )
        
        assert "x2_preis" in result  # Actually wohnflaeche
        assert "x3_werbung" in result  # Actually pool
        assert "y_mult" in result
        
        # Check shapes
        assert len(result["x2_preis"]) == 1000
        assert len(result["x3_werbung"]) == 1000
        assert len(result["y_mult"]) == 1000
        
        # Check variable names
        assert result["x1_name"] == "Wohnfläche (sqft/10)"
        assert result["x2_name"] == "Pool (0/1)"
        assert result["y_name"] == "Preis (USD)"
        
        # Check pool is binary
        assert set(result["x3_werbung"]) <= {0.0, 1.0}
    
    @pytest.mark.unit
    def test_elektronikmarkt_dataset(self):
        """Test multiple regression data generation for Elektronikmarkt."""
        result = generate_multiple_regression_data(
            "🏪 Elektronikmarkt (erweitert)",
            n_mult=50,
            noise_mult_level=0.35,
            seed_mult=42
        )
        
        assert "x2_preis" in result  # Actually flaeche
        assert "x3_werbung" in result  # Actually marketing
        assert "y_mult" in result
        
        # Check shapes
        assert len(result["x2_preis"]) == 50
        assert len(result["x3_werbung"]) == 50
        assert len(result["y_mult"]) == 50
        
        # Check variable names
        assert result["x1_name"] == "Verkaufsfläche (100qm)"
        assert result["x2_name"] == "Marketing (10k€)"
        assert result["y_name"] == "Umsatz (Mio. €)"
    
    @pytest.mark.unit
    def test_variable_sample_size(self):
        """Test that function works with different sample sizes."""
        for n in [20, 50, 100, 150]:
            result = generate_multiple_regression_data(
                "🏙️ Städte-Umsatzstudie (75 Städte)",
                n_mult=n,
                noise_mult_level=3.5,
                seed_mult=42
            )
            assert len(result["x2_preis"]) == n
            assert len(result["x3_werbung"]) == n
            assert len(result["y_mult"]) == n
    
    @pytest.mark.unit
    def test_variable_noise_level(self):
        """Test that different noise levels produce different variability."""
        result_low = generate_multiple_regression_data(
            "🏙️ Städte-Umsatzstudie (75 Städte)",
            n_mult=75,
            noise_mult_level=1.0,
            seed_mult=42
        )
        
        result_high = generate_multiple_regression_data(
            "🏙️ Städte-Umsatzstudie (75 Städte)",
            n_mult=75,
            noise_mult_level=8.0,
            seed_mult=42
        )
        
        # Higher noise should generally produce higher variance
        # (though not guaranteed for every seed)
        assert "y_mult" in result_low
        assert "y_mult" in result_high
    
    @pytest.mark.unit
    def test_reproducibility(self):
        """Test that same seed produces same results."""
        result1 = generate_multiple_regression_data(
            "🏙️ Städte-Umsatzstudie (75 Städte)",
            n_mult=75,
            noise_mult_level=3.5,
            seed_mult=999
        )
        
        result2 = generate_multiple_regression_data(
            "🏙️ Städte-Umsatzstudie (75 Städte)",
            n_mult=75,
            noise_mult_level=3.5,
            seed_mult=999
        )
        
        np.testing.assert_array_equal(result1["x2_preis"], result2["x2_preis"])
        np.testing.assert_array_equal(result1["x3_werbung"], result2["x3_werbung"])
        np.testing.assert_array_equal(result1["y_mult"], result2["y_mult"])


class TestGenerateSimpleRegressionData:
    """Test the generate_simple_regression_data function."""
    
    @pytest.mark.unit
    def test_staedte_preis(self):
        """Test simple regression with Städte dataset and Preis variable."""
        result = generate_simple_regression_data(
            "🏙️ Städte-Umsatzstudie (75 Städte)",
            "Preis (CHF)",
            n=75,
            seed=42
        )
        
        assert "x" in result
        assert "y" in result
        assert len(result["x"]) == 75
        assert len(result["y"]) == 75
        
        assert result["x_label"] == "Preis (CHF)"
        assert result["y_label"] == "Umsatz (1'000 CHF)"
        assert result["x_unit"] == "CHF"
        assert result["y_unit"] == "1'000 CHF"
        assert result["context_title"] == "Städte-Umsatzstudie"
    
    @pytest.mark.unit
    def test_staedte_werbung(self):
        """Test simple regression with Städte dataset and Werbung variable."""
        result = generate_simple_regression_data(
            "🏙️ Städte-Umsatzstudie (75 Städte)",
            "Werbung (CHF1000)",
            n=75,
            seed=42
        )
        
        assert "x" in result
        assert "y" in result
        assert len(result["x"]) == 75
        assert len(result["y"]) == 75
        
        assert result["x_label"] == "Werbung (CHF1000)"
        assert result["y_label"] == "Umsatz (1'000 CHF)"
    
    @pytest.mark.unit
    def test_haeuser_wohnflaeche(self):
        """Test simple regression with Häuser dataset and Wohnfläche."""
        result = generate_simple_regression_data(
            "🏠 Häuserpreise mit Pool (1000 Häuser)",
            "Wohnfläche (sqft/10)",
            n=1000,
            seed=42
        )
        
        assert "x" in result
        assert "y" in result
        assert len(result["x"]) == 1000
        assert len(result["y"]) == 1000
        
        assert result["x_label"] == "Wohnfläche (sqft/10)"
        assert result["y_label"] == "Preis (USD)"
    
    @pytest.mark.unit
    def test_haeuser_pool(self):
        """Test simple regression with Häuser dataset and Pool dummy variable."""
        result = generate_simple_regression_data(
            "🏠 Häuserpreise mit Pool (1000 Häuser)",
            "Pool (0/1)",
            n=1000,
            seed=42
        )
        
        assert "x" in result
        assert "y" in result
        assert len(result["x"]) == 1000
        assert len(result["y"]) == 1000
        
        assert result["x_label"] == "Pool (0/1)"
        assert result["y_label"] == "Preis (USD)"
        
        # Check pool is binary
        assert set(result["x"]) <= {0.0, 1.0}
    
    @pytest.mark.unit
    def test_reproducibility(self):
        """Test that same seed produces same results."""
        result1 = generate_simple_regression_data(
            "🏙️ Städte-Umsatzstudie (75 Städte)",
            "Preis (CHF)",
            n=75,
            seed=123
        )
        
        result2 = generate_simple_regression_data(
            "🏙️ Städte-Umsatzstudie (75 Städte)",
            "Preis (CHF)",
            n=75,
            seed=123
        )
        
        np.testing.assert_array_equal(result1["x"], result2["x"])
        np.testing.assert_array_equal(result1["y"], result2["y"])
    
    @pytest.mark.unit
    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different data."""
        result1 = generate_simple_regression_data(
            "🏙️ Städte-Umsatzstudie (75 Städte)",
            "Preis (CHF)",
            n=75,
            seed=42
        )
        
        result2 = generate_simple_regression_data(
            "🏙️ Städte-Umsatzstudie (75 Städte)",
            "Preis (CHF)",
            n=75,
            seed=999
        )
        
        # Arrays should be different with different seeds
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(result1["x"], result2["x"])


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.mark.unit
    def test_small_sample_size(self):
        """Test with minimum sample size."""
        result = generate_multiple_regression_data(
            "🏙️ Städte-Umsatzstudie (75 Städte)",
            n_mult=20,
            noise_mult_level=3.5,
            seed_mult=42
        )
        assert len(result["x2_preis"]) == 20
    
    @pytest.mark.unit
    def test_large_sample_size(self):
        """Test with large sample size."""
        result = generate_multiple_regression_data(
            "🏠 Häuserpreise mit Pool (1000 Häuser)",
            n_mult=2000,
            noise_mult_level=20.0,
            seed_mult=42
        )
        assert len(result["x2_preis"]) == 2000
    
    @pytest.mark.unit
    def test_zero_noise(self):
        """Test with zero noise level."""
        result = generate_multiple_regression_data(
            "🏪 Elektronikmarkt (erweitert)",
            n_mult=50,
            noise_mult_level=0.0,
            seed_mult=42
        )
        # Should still work, just perfect fit
        assert len(result["y_mult"]) == 50
    
    @pytest.mark.unit
    def test_high_noise(self):
        """Test with very high noise level."""
        result = generate_multiple_regression_data(
            "🏙️ Städte-Umsatzstudie (75 Städte)",
            n_mult=75,
            noise_mult_level=100.0,
            seed_mult=42
        )
        # Should still work, just noisy data
        assert len(result["y_mult"]) == 75


class TestSwissDatasets:
    """Test Swiss dataset generation functions."""

    @pytest.mark.unit
    def test_swiss_canton_regression_data(self):
        """Test Swiss canton socioeconomic data generation."""
        result = generate_swiss_canton_regression_data()

        # Check required keys
        required_keys = ['x_population_density', 'x_foreign_pct', 'x_unemployment',
                        'y_gdp_per_capita', 'canton_names', 'n', 'x1_name', 'x2_name', 'x3_name', 'y_name']
        for key in required_keys:
            assert key in result

        # Check data types and shapes
        assert isinstance(result['x_population_density'], np.ndarray)
        assert isinstance(result['y_gdp_per_capita'], np.ndarray)
        assert len(result['x_population_density']) == len(result['y_gdp_per_capita'])
        assert result['n'] == len(result['canton_names'])
        assert result['n'] == 5  # Sample of 5 cantons

    @pytest.mark.unit
    def test_swiss_weather_regression_data(self):
        """Test Swiss weather station data generation."""
        result = generate_swiss_weather_regression_data()

        # Check required keys
        required_keys = ['x_altitude', 'x_sunshine', 'x_humidity',
                        'y_temperature', 'station_names', 'n', 'x1_name', 'x2_name', 'x3_name', 'y_name']
        for key in required_keys:
            assert key in result

        # Check data types and shapes
        assert isinstance(result['x_altitude'], np.ndarray)
        assert isinstance(result['y_temperature'], np.ndarray)
        assert len(result['x_altitude']) == len(result['y_temperature'])
        assert result['n'] == len(result['station_names'])
        assert result['n'] == 7  # 7 weather stations

    @pytest.mark.unit
    def test_simple_regression_swiss_datasets(self):
        """Test simple regression with Swiss datasets."""
        # Test cantons
        result_canton = generate_simple_regression_data(
            '🇨🇭 Schweizer Kantone (sozioökonomisch)',
            'Population Density',
            5, 42
        )
        assert 'x' in result_canton
        assert 'y' in result_canton
        assert 'x_label' in result_canton
        assert 'y_label' in result_canton
        assert len(result_canton['x']) == 5

        # Test weather
        result_weather = generate_simple_regression_data(
            '🌤️ Schweizer Wetterstationen',
            'Altitude',
            7, 42
        )
        assert 'x' in result_weather
        assert 'y' in result_weather
        assert 'x_label' in result_weather
        assert 'y_label' in result_weather
        assert len(result_weather['x']) == 7


class TestGlobalAPIs:
    """Test global API data fetching functions."""

    @pytest.mark.integration
    def test_world_bank_data_fetching(self):
        """Test World Bank data fetching (mock implementation)."""
        result = fetch_world_bank_data(
            indicators=['NY.GDP.PCAP.KD', 'SP.POP.TOTL'],
            countries=['USA', 'CHN', 'DEU'],
            years=[2015, 2018, 2020]
        )

        # Should return DataFrame (even if empty due to mock)
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.integration
    def test_fred_data_fetching(self):
        """Test FRED economic data fetching (mock implementation)."""
        result = fetch_fred_data(
            series_ids=['GDP', 'UNRATE'],
            start_date='2010-01-01',
            end_date='2020-01-01'
        )

        # Should return DataFrame (even if empty due to mock)
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.integration
    def test_who_health_data_fetching(self):
        """Test WHO health data fetching (mock implementation)."""
        result = fetch_who_health_data(
            indicators=['WHOSIS_000001'],
            countries=['USA', 'CHN', 'DEU'],
            years=[2015, 2018, 2020]
        )

        # Should return DataFrame (even if empty due to mock)
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.unit
    def test_dataset_metadata_functions(self):
        """Test dataset metadata functions."""
        swiss_datasets = get_available_swiss_datasets()
        global_datasets = get_global_regression_datasets()

        # Check structure
        assert isinstance(swiss_datasets, dict)
        assert isinstance(global_datasets, dict)

        # Check that datasets have required metadata
        for name, info in swiss_datasets.items():
            assert 'name' in info
            assert 'description' in info
            assert 'variables' in info
            assert 'source' in info

        for name, info in global_datasets.items():
            assert 'name' in info
            assert 'description' in info
            assert 'variables' in info
            assert 'source' in info
            assert 'python_package' in info
            assert 'api_available' in info
