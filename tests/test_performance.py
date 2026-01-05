"""
Performance regression tests for caching and optimization.

Tests ensure that:
- Caching provides expected speedup
- Performance optimizations work correctly
- No performance regressions in data generation
- Session state caching is effective
"""

import pytest
import time
import numpy as np
from src.data import (
    generate_multiple_regression_data,
    generate_simple_regression_data,
    generate_dataset,
    generate_swiss_canton_regression_data,
    generate_swiss_weather_regression_data,
    fetch_world_bank_data,
    fetch_fred_data,
    fetch_who_health_data
)


class TestDataGenerationPerformance:
    """Test performance of data generation functions."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_cached_data_generation_is_faster(self):
        """Test that cached data generation is faster than first call."""
        # First call (uncached)
        start = time.time()
        result1 = generate_multiple_regression_data(
            "🏙️ Städte-Umsatzstudie (75 Städte)",
            n_mult=75,
            noise_mult_level=3.5,
            seed_mult=42
        )
        first_call_time = time.time() - start
        
        # Second call (should be cached)
        start = time.time()
        result2 = generate_multiple_regression_data(
            "🏙️ Städte-Umsatzstudie (75 Städte)",
            n_mult=75,
            noise_mult_level=3.5,
            seed_mult=42
        )
        second_call_time = time.time() - start
        
        # Cached call should be much faster (at least 2x)
        # Note: With @st.cache_data, cached calls are typically 10-100x faster
        assert second_call_time < first_call_time * 0.5, \
            f"Cached call ({second_call_time:.4f}s) not significantly faster than first call ({first_call_time:.4f}s)"
        
        # Results should be identical
        np.testing.assert_array_equal(result1["x2_preis"], result2["x2_preis"])
    
    @pytest.mark.performance
    def test_small_dataset_generation_speed(self):
        """Test that small dataset generation completes quickly."""
        start = time.time()
        result = generate_multiple_regression_data(
            "🏙️ Städte-Umsatzstudie (75 Städte)",
            n_mult=20,
            noise_mult_level=3.5,
            seed_mult=123
        )
        duration = time.time() - start
        
        # Should complete in under 1 second for small dataset
        assert duration < 1.0, f"Small dataset took {duration:.4f}s, expected <1.0s"
        assert len(result["x2_preis"]) == 20
    
    @pytest.mark.performance
    def test_large_dataset_generation_speed(self):
        """Test that large dataset generation completes in reasonable time."""
        start = time.time()
        result = generate_multiple_regression_data(
            "🏠 Häuserpreise mit Pool (1000 Häuser)",
            n_mult=2000,
            noise_mult_level=20.0,
            seed_mult=456
        )
        duration = time.time() - start
        
        # Should complete in under 5 seconds even for large dataset
        assert duration < 5.0, f"Large dataset took {duration:.4f}s, expected <5.0s"
        assert len(result["x2_preis"]) == 2000
    
    @pytest.mark.performance
    def test_simple_regression_data_generation_speed(self):
        """Test simple regression data generation performance."""
        start = time.time()
        result = generate_simple_regression_data(
            "🏙️ Städte-Umsatzstudie (75 Städte)",
            "Preis (CHF)",
            n=75,
            seed=42
        )
        duration = time.time() - start
        
        # Should be fast
        assert duration < 2.0, f"Simple regression data took {duration:.4f}s, expected <2.0s"
        assert len(result["x"]) == 75


class TestCachingEffectiveness:
    """Test that caching is effective and provides expected benefits."""
    
    @pytest.mark.performance
    def test_cache_hit_with_identical_parameters(self):
        """Test that identical parameters result in cache hit."""
        params = ("🏙️ Städte-Umsatzstudie (75 Städte)", 75, 3.5, 42)
        
        # First call
        result1 = generate_multiple_regression_data(*params)
        
        # Second call with identical params
        start = time.time()
        result2 = generate_multiple_regression_data(*params)
        cached_time = time.time() - start
        
        # Should be very fast (cache hit)
        assert cached_time < 0.01, f"Cache hit took {cached_time:.4f}s, expected <0.01s"
        
        # Results should be identical
        np.testing.assert_array_equal(result1["x2_preis"], result2["x2_preis"])
    
    @pytest.mark.performance
    def test_cache_miss_with_different_parameters(self):
        """Test that different parameters result in cache miss."""
        # Generate with one set of params
        result1 = generate_multiple_regression_data(
            "🏙️ Städte-Umsatzstudie (75 Städte)", 75, 3.5, 42
        )
        
        # Generate with different params (should be cache miss)
        result2 = generate_multiple_regression_data(
            "🏙️ Städte-Umsatzstudie (75 Städte)", 75, 3.5, 999
        )
        
        # Results should be different (different seed)
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(result1["x2_preis"], result2["x2_preis"])
    
    @pytest.mark.performance
    def test_multiple_cache_entries(self):
        """Test that multiple different parameter sets are cached separately."""
        # Generate three different datasets
        result1 = generate_multiple_regression_data(
            "🏙️ Städte-Umsatzstudie (75 Städte)", 75, 3.5, 42
        )
        result2 = generate_multiple_regression_data(
            "🏠 Häuserpreise mit Pool (1000 Häuser)", 1000, 20.0, 42
        )
        result3 = generate_multiple_regression_data(
            "🏪 Elektronikmarkt (erweitert)", 50, 0.35, 42
        )
        
        # Re-access first dataset - should be cached
        start = time.time()
        result1_cached = generate_multiple_regression_data(
            "🏙️ Städte-Umsatzstudie (75 Städte)", 75, 3.5, 42
        )
        cached_time = time.time() - start
        
        assert cached_time < 0.01
        np.testing.assert_array_equal(result1["x2_preis"], result1_cached["x2_preis"])


class TestPerformanceRegression:
    """Test for performance regressions."""
    
    @pytest.mark.performance
    def test_no_regression_in_data_generation(self):
        """Test that data generation doesn't regress in performance."""
        times = []
        
        # Run 5 times with different seeds
        for seed in range(100, 105):
            start = time.time()
            generate_multiple_regression_data(
                "🏙️ Städte-Umsatzstudie (75 Städte)",
                n_mult=75,
                noise_mult_level=3.5,
                seed_mult=seed
            )
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        max_time = np.max(times)
        
        # Average should be under 1 second
        assert avg_time < 1.0, f"Average generation time {avg_time:.4f}s exceeds 1.0s"
        
        # No single call should take more than 2 seconds
        assert max_time < 2.0, f"Max generation time {max_time:.4f}s exceeds 2.0s"
    
    @pytest.mark.performance
    def test_consistent_performance_across_datasets(self):
        """Test that all dataset types have reasonable performance."""
        datasets = [
            ("🏙️ Städte-Umsatzstudie (75 Städte)", 75, 3.5),
            ("🏠 Häuserpreise mit Pool (1000 Häuser)", 1000, 20.0),
            ("🏪 Elektronikmarkt (erweitert)", 50, 0.35),
        ]
        
        for dataset, n, noise in datasets:
            start = time.time()
            result = generate_multiple_regression_data(
                dataset, n, noise, 42
            )
            duration = time.time() - start
            
            # All should complete in reasonable time
            assert duration < 3.0, f"{dataset} took {duration:.4f}s, expected <3.0s"
            assert len(result["x2_preis"]) == n


class TestMemoryEfficiency:
    """Test memory efficiency of data structures."""
    
    @pytest.mark.performance
    def test_reasonable_memory_usage(self):
        """Test that generated data uses reasonable memory."""
        # Generate large dataset
        result = generate_multiple_regression_data(
            "🏠 Häuserpreise mit Pool (1000 Häuser)",
            n_mult=2000,
            noise_mult_level=20.0,
            seed_mult=42
        )
        
        # Check data types are appropriate (float64 is standard)
        assert result["x2_preis"].dtype in [np.float64, np.float32]
        assert result["x3_werbung"].dtype in [np.float64, np.float32]
        assert result["y_mult"].dtype in [np.float64, np.float32]
        
        # Arrays should be 1D
        assert result["x2_preis"].ndim == 1
        assert result["x3_werbung"].ndim == 1
        assert result["y_mult"].ndim == 1


class TestScalability:
    """Test that functions scale well with input size."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_linear_scaling_with_sample_size(self):
        """Test that generation time scales linearly with sample size."""
        sizes = [100, 500, 1000]
        times = []
        
        for size in sizes:
            start = time.time()
            generate_multiple_regression_data(
                "🏠 Häuserpreise mit Pool (1000 Häuser)",
                n_mult=size,
                noise_mult_level=20.0,
                seed_mult=42
            )
            times.append(time.time() - start)
        
        # Time ratio should not grow super-linearly
        # If time grows faster than O(n), this indicates a problem
        time_ratio_1 = times[1] / times[0]
        size_ratio_1 = sizes[1] / sizes[0]
        
        time_ratio_2 = times[2] / times[1]
        size_ratio_2 = sizes[2] / sizes[1]
        
        # Time ratio should not be much larger than size ratio
        # Allow factor of 2 for variability
        assert time_ratio_1 < size_ratio_1 * 2, \
            f"Time scaling is worse than linear: time ratio {time_ratio_1:.2f} vs size ratio {size_ratio_1:.2f}"


class TestCacheInvalidation:
    """Test that cache invalidates correctly when it should."""
    
    @pytest.mark.performance
    def test_different_seeds_invalidate_cache(self):
        """Test that changing seed invalidates cache."""
        # First call with seed 42
        result1 = generate_multiple_regression_data(
            "🏙️ Städte-Umsatzstudie (75 Städte)", 75, 3.5, 42
        )
        
        # Second call with seed 999 (should not use cache)
        result2 = generate_multiple_regression_data(
            "🏙️ Städte-Umsatzstudie (75 Städte)", 75, 3.5, 999
        )
        
        # Results should be different
        assert not np.array_equal(result1["x2_preis"], result2["x2_preis"])
    
    @pytest.mark.performance
    def test_different_noise_levels_invalidate_cache(self):
        """Test that changing noise level invalidates cache."""
        result1 = generate_multiple_regression_data(
            "🏙️ Städte-Umsatzstudie (75 Städte)", 75, 1.0, 42
        )
        
        result2 = generate_multiple_regression_data(
            "🏙️ Städte-Umsatzstudie (75 Städte)", 75, 8.0, 42
        )
        
        # Y values should be different due to different noise
        assert not np.array_equal(result1["y_mult"], result2["y_mult"])


class TestSwissDatasetPerformance:
    """Test performance of Swiss dataset generation functions."""

    @pytest.mark.performance
    def test_swiss_canton_data_generation_speed(self):
        """Test that Swiss canton data generation is reasonably fast."""
        start = time.time()
        result = generate_swiss_canton_regression_data()
        end = time.time()

        # Should complete in under 0.1 seconds
        assert end - start < 0.1
        assert result['n'] == 5  # Sample size
        assert len(result['canton_names']) == 5

    @pytest.mark.performance
    def test_swiss_weather_data_generation_speed(self):
        """Test that Swiss weather data generation is reasonably fast."""
        start = time.time()
        result = generate_swiss_weather_regression_data()
        end = time.time()

        # Should complete in under 0.1 seconds
        assert end - start < 0.1
        assert result['n'] == 7  # 7 weather stations
        assert len(result['station_names']) == 7


class TestGlobalAPIPerformance:
    """Test performance of global API data fetching functions."""


class TestCachingPerformance:
    """Test performance improvements from caching optimizations."""

    def test_data_generation_caching(self):
        """Test that data generation caching provides speedup."""
        from src.data import generate_simple_regression_data

        # First call (cache miss)
        start = time.time()
        result1 = generate_simple_regression_data(
            "🇨🇭 Schweizer Kantone (sozioökonomisch)",
            "Population Density",
            10,
            seed=42
        )
        first_call_time = time.time() - start

        # Second call (cache hit)
        start = time.time()
        result2 = generate_simple_regression_data(
            "🇨🇭 Schweizer Kantone (sozioökonomisch)",
            "Population Density",
            10,
            seed=42
        )
        second_call_time = time.time() - start

        # Results should be identical
        assert np.array_equal(result1['x'], result2['x'])
        assert np.array_equal(result1['y'], result2['y'])

        # Second call should be significantly faster (at least 10x)
        speedup = first_call_time / max(second_call_time, 0.001)
        assert speedup > 10, f"Expected speedup > 10x, got {speedup:.2f}x"

    def test_multiple_regression_caching(self):
        """Test that multiple regression data generation caching works."""
        from src.data import generate_multiple_regression_data

        # First call
        start = time.time()
        result1 = generate_multiple_regression_data(
            "🏙️ Städte-Umsatzstudie (75 Städte)",
            50, 0.5, 42
        )
        first_call_time = time.time() - start

        # Second call with same parameters
        start = time.time()
        result2 = generate_multiple_regression_data(
            "🏙️ Städte-Umsatzstudie (75 Städte)",
            50, 0.5, 42
        )
        second_call_time = time.time() - start

        # Results should be identical
        assert np.array_equal(result1['x2_preis'], result2['x2_preis'])
        assert np.array_equal(result1['x3_werbung'], result2['x3_werbung'])

        # Second call should be faster
        speedup = first_call_time / max(second_call_time, 0.001)
        assert speedup > 5, f"Expected speedup > 5x, got {speedup:.2f}x"

    def test_ols_model_caching(self):
        """Test that OLS model fitting caching works."""
        from src.data import fit_ols_model
        import statsmodels.api as sm

        # Create test data
        np.random.seed(42)
        X = sm.add_constant(np.random.randn(100, 1))
        y = 2 + 3 * X[:, 1] + np.random.randn(100) * 0.1

        # First call
        start = time.time()
        model1, pred1 = fit_ols_model(X, y)
        first_call_time = time.time() - start

        # Second call with same data
        start = time.time()
        model2, pred2 = fit_ols_model(X, y)
        second_call_time = time.time() - start

        # Results should be identical
        assert np.allclose(model1.params, model2.params)
        assert np.array_equal(pred1, pred2)

        # Second call should be faster
        speedup = first_call_time / max(second_call_time, 0.001)
        assert speedup > 10, f"Expected speedup > 10x, got {speedup:.2f}x"

    @pytest.mark.benchmark
    def test_app_startup_performance(self, benchmark):
        """Benchmark app startup time."""
        def startup_test():
            # This would normally test app initialization
            # For now, just test a key data generation function
            from src.data import generate_simple_regression_data
            return generate_simple_regression_data(
                "🏪 Elektronikmarkt (simuliert)",
                "Verkaufsfläche (100qm)",
                100,
                seed=42
            )

        # Should complete in under 0.5 seconds with caching
        result = benchmark(startup_test)
        assert result['x'].shape[0] == 100
        assert benchmark.stats['mean'] < 0.5

    @pytest.mark.benchmark
    def test_dataset_switching_performance(self, benchmark):
        """Benchmark performance of switching between datasets."""
        datasets = [
            ("🇨🇭 Schweizer Kantone (sozioökonomisch)", "Population Density"),
            ("🏙️ Städte-Umsatzstudie (75 Städte)", "Population"),
            ("🏠 Häuserpreise mit Pool (1000 Häuser)", "Living Area")
        ]

        def switching_test():
            from src.data import generate_simple_regression_data
            results = []
            for dataset, variable in datasets:
                result = generate_simple_regression_data(dataset, variable, 50, seed=42)
                results.append(result)
            return results

        # Should complete switching in under 2 seconds with caching
        result = benchmark(switching_test)
        assert len(result) == 3
        assert benchmark.stats['mean'] < 2.0


