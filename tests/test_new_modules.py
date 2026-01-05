"""
Tests for the newly created modular components.

This test file verifies that the extracted modules work correctly
and maintain the same functionality as the original app.py.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import streamlit as st
import numpy as np
import pandas as pd


class TestSessionState:
    """Tests for session_state.py module."""
    
    def test_initialize_session_state(self):
        """Test that session state initialization creates all required keys."""
        from src.session_state import initialize_session_state
        
        with patch.object(st, 'session_state', {}):
            initialize_session_state()
            
            # Check all required keys are created
            assert 'active_tab' in st.session_state
            assert 'last_mult_params' in st.session_state
            assert 'mult_model_cache' in st.session_state
            assert 'last_simple_params' in st.session_state
            assert 'simple_model_cache' in st.session_state
            assert 'simple_data_temp' in st.session_state
            assert 'current_model' in st.session_state
            assert 'current_feature_names' in st.session_state
    
    def test_check_params_changed_new_params(self):
        """Test parameter change detection with new parameters."""
        from src.session_state import check_params_changed
        
        with patch.object(st, 'session_state', {'last_mult_params': (1, 2, 3)}):
            # Different params should return True
            assert check_params_changed((4, 5, 6), 'last_mult_params') is True
    
    def test_check_params_changed_same_params(self):
        """Test parameter change detection with same parameters."""
        from src.session_state import check_params_changed
        
        with patch.object(st, 'session_state', {'last_mult_params': (1, 2, 3)}):
            # Same params should return False
            assert check_params_changed((1, 2, 3), 'last_mult_params') is False
    
    def test_update_cached_params(self):
        """Test updating cached parameters."""
        from src.session_state import update_cached_params
        
        session_state = {}
        with patch.object(st, 'session_state', session_state):
            params = (1, 2, 3, 4)
            update_cached_params(params, 'test_key')
            assert session_state['test_key'] == params
    
    def test_cache_model_data(self):
        """Test caching model data."""
        from src.session_state import cache_model_data
        
        session_state = {}
        with patch.object(st, 'session_state', session_state):
            data = {'model': 'test_model', 'params': [1, 2, 3]}
            cache_model_data(data, 'test_cache')
            assert session_state['test_cache'] == data
    
    def test_get_cached_model_data(self):
        """Test retrieving cached model data."""
        from src.session_state import get_cached_model_data
        
        cached_data = {'model': 'test_model'}
        with patch.object(st, 'session_state', {'test_cache': cached_data}):
            result = get_cached_model_data('test_cache')
            assert result == cached_data
    
    def test_update_current_model(self):
        """Test updating current model for R output."""
        from src.session_state import update_current_model
        
        session_state = {}
        with patch.object(st, 'session_state', session_state):
            model = Mock()
            features = ['x1', 'x2', 'x3']
            update_current_model(model, features)
            assert session_state['current_model'] == model
            assert session_state['current_feature_names'] == features
    
    def test_clear_cache(self):
        """Test clearing cache."""
        from src.session_state import clear_cache
        
        session_state = {
            'last_mult_params': (1, 2, 3),
            'mult_model_cache': {'data': 'test'},
            'other_key': 'should_remain'
        }
        with patch.object(st, 'session_state', session_state):
            clear_cache(['last_mult_params', 'mult_model_cache'])
            assert session_state['last_mult_params'] is None
            assert session_state['mult_model_cache'] is None
            assert session_state['other_key'] == 'should_remain'


class TestUIConfig:
    """Tests for ui_config.py module."""
    
    @patch('src.ui_config.st.set_page_config')
    def test_setup_page_config(self, mock_set_page_config):
        """Test page configuration setup."""
        from src.ui_config import setup_page_config
        
        setup_page_config()
        
        mock_set_page_config.assert_called_once()
        call_kwargs = mock_set_page_config.call_args[1]
        assert 'page_title' in call_kwargs
        assert 'page_icon' in call_kwargs
        assert call_kwargs['layout'] == 'wide'
    
    @patch('src.ui_config.st.markdown')
    def test_inject_custom_css(self, mock_markdown):
        """Test CSS injection."""
        from src.ui_config import inject_custom_css
        
        inject_custom_css()
        
        mock_markdown.assert_called_once()
        call_args = mock_markdown.call_args[0][0]
        assert '<style>' in call_args
        assert '.main-header' in call_args
        assert '.section-header' in call_args
    
    @patch('src.ui_config.st.markdown')
    def test_render_footer(self, mock_markdown):
        """Test footer rendering."""
        from src.ui_config import render_footer
        
        render_footer()
        
        assert mock_markdown.call_count >= 1
        # Check that footer content is rendered
        footer_call = [call for call in mock_markdown.call_args_list 
                      if 'Leitfaden zur Linearen Regression' in str(call)]
        assert len(footer_call) > 0


class TestSidebar:
    """Tests for sidebar.py module."""
    
    def test_dataset_selection_dataclass(self):
        """Test DatasetSelection dataclass."""
        from src.sidebar import DatasetSelection
        
        selection = DatasetSelection(
            simple_dataset="Test Simple",
            multiple_dataset="Test Multiple"
        )
        assert selection.simple_dataset == "Test Simple"
        assert selection.multiple_dataset == "Test Multiple"
    
    def test_simple_regression_params_dataclass(self):
        """Test SimpleRegressionParams dataclass."""
        from src.sidebar import SimpleRegressionParams
        
        params = SimpleRegressionParams(
            n=50,
            true_intercept=0.5,
            true_beta=0.8,
            noise_level=0.3,
            seed=42,
            x_variable="test_var"
        )
        assert params.n == 50
        assert params.true_intercept == 0.5
        assert params.true_beta == 0.8
        assert params.noise_level == 0.3
        assert params.seed == 42
        assert params.x_variable == "test_var"
    
    def test_multiple_regression_params_dataclass(self):
        """Test MultipleRegressionParams dataclass."""
        from src.sidebar import MultipleRegressionParams
        
        params = MultipleRegressionParams(
            n=75,
            noise_level=0.5,
            seed=123
        )
        assert params.n == 75
        assert params.noise_level == 0.5
        assert params.seed == 123
    
    def test_display_options_dataclass(self):
        """Test DisplayOptions dataclass."""
        from src.sidebar import DisplayOptions
        
        options = DisplayOptions(
            show_formulas=True,
            show_true_line=False
        )
        assert options.show_formulas is True
        assert options.show_true_line is False


class TestDataPreparation:
    """Tests for data_preparation.py module."""
    
    @patch('src.data_preparation.st.spinner')
    @patch('src.data_preparation.generate_multiple_regression_data')
    @patch('src.data_preparation.fit_ols_model')
    @patch('src.data_preparation.create_design_matrix')
    def test_prepare_multiple_regression_data_cache_miss(
        self, mock_create_matrix, mock_fit_ols, mock_gen_data, mock_spinner
    ):
        """Test multiple regression data preparation with cache miss."""
        from src.data_preparation import prepare_multiple_regression_data
        
        # Setup mocks
        mock_gen_data.return_value = {
            'x2_preis': np.array([1, 2, 3]),
            'x3_werbung': np.array([4, 5, 6]),
            'y_mult': np.array([7, 8, 9]),
            'x1_name': 'X1',
            'x2_name': 'X2',
            'y_name': 'Y'
        }
        mock_model = Mock()
        mock_fit_ols.return_value = (mock_model, np.array([7.1, 8.1, 9.1]))
        mock_create_matrix.return_value = np.array([[1, 1, 4], [1, 2, 5], [1, 3, 6]])
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock()
        
        # Mock session state
        session_state = {
            'last_mult_params': None,
            'mult_model_cache': None
        }
        
        with patch.object(st, 'session_state', session_state):
            result = prepare_multiple_regression_data(
                dataset_choice="Test Dataset",
                n=75,
                noise_level=0.5,
                seed=42
            )
        
        # Verify result structure
        assert 'x2_preis' in result
        assert 'x3_werbung' in result
        assert 'y_mult' in result
        assert 'model_mult' in result
        assert 'y_pred_mult' in result
    
    @patch('src.data_preparation.generate_electronics_market_data')
    def test_prepare_simple_regression_data_electronics(self, mock_gen_data):
        """Test simple regression data preparation for electronics market."""
        from src.data_preparation import prepare_simple_regression_data
        
        # Setup mock
        mock_gen_data.return_value = {
            'x': np.array([1, 2, 3]),
            'y': np.array([4, 5, 6]),
            'x_label': 'Area',
            'y_label': 'Sales',
            'x_unit': 'sqm',
            'y_unit': 'EUR',
            'context_title': 'Electronics',
            'context_description': 'Test'
        }
        
        # Mock session state
        session_state = {
            'last_simple_params': (
                "🏪 Elektronikmarkt (simuliert)", 
                11,  # Different n to trigger regeneration
                0.6,
                0.52,
                0.4,
                42
            ),
            'simple_data_temp': {}
        }
        
        with patch.object(st, 'session_state', session_state):
            result = prepare_simple_regression_data(
                dataset_choice="🏪 Elektronikmarkt (simuliert)",
                n=12,
                true_intercept=0.6,
                true_beta=0.52,
                noise_level=0.4,
                seed=42
            )
        
        assert 'x' in result
        assert 'y' in result
        assert 'x_label' in result
        assert 'y_label' in result


class TestTabDatasets:
    """Tests for tabs/tab_datasets.py module."""
    
    @patch('src.tabs.tab_datasets.st.markdown')
    @patch('src.tabs.tab_datasets.st.columns')
    @patch('src.tabs.tab_datasets.st.info')
    @patch('src.tabs.tab_datasets.st.dataframe')
    def test_render_tab(self, mock_dataframe, mock_info, mock_columns, mock_markdown):
        """Test datasets tab rendering."""
        from src.tabs.tab_datasets import render
        
        # Mock columns
        mock_col = Mock()
        mock_columns.return_value = [mock_col, mock_col]
        mock_col.__enter__ = Mock(return_value=mock_col)
        mock_col.__exit__ = Mock()
        
        render()
        
        # Verify markdown was called for headers
        assert mock_markdown.call_count > 0
        
        # Verify dataframe was called for comparison table
        assert mock_dataframe.call_count >= 1


class TestROutput:
    """Tests for r_output.py module."""
    
    @patch('src.r_output.st.markdown')
    @patch('src.r_output.st.columns')
    @patch('src.r_output.st.info')
    @patch('src.r_output.create_r_output_figure')
    @patch('src.r_output.st.plotly_chart')
    def test_render_r_output_section_with_model(
        self, mock_plotly_chart, mock_create_fig, mock_info, mock_columns, mock_markdown
    ):
        """Test R output rendering with a model."""
        from src.r_output import render_r_output_section
        
        # Setup mocks
        mock_model = Mock()
        mock_fig = Mock()
        mock_create_fig.return_value = mock_fig
        mock_col = Mock()
        mock_columns.return_value = [mock_col, mock_col]
        mock_col.__enter__ = Mock(return_value=mock_col)
        mock_col.__exit__ = Mock()
        
        render_r_output_section(
            model=mock_model,
            feature_names=['x1', 'x2']
        )
        
        # Verify figure was created and displayed
        mock_create_fig.assert_called_once()
        mock_plotly_chart.assert_called_once()
    
    @patch('src.r_output.st.markdown')
    @patch('src.r_output.st.columns')
    @patch('src.r_output.st.info')
    def test_render_r_output_section_without_model(
        self, mock_info, mock_columns, mock_markdown
    ):
        """Test R output rendering without a model."""
        from src.r_output import render_r_output_section
        
        # Setup mocks
        mock_col = Mock()
        mock_columns.return_value = [mock_col, mock_col]
        mock_col.__enter__ = Mock(return_value=mock_col)
        mock_col.__exit__ = Mock()
        
        render_r_output_section(model=None, feature_names=None)
        
        # Verify info message was shown
        assert mock_info.call_count >= 1


# Integration test markers
pytestmark = pytest.mark.unit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
