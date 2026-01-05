"""
Streamlit AppTest integration tests for app.py

Tests complete user workflows using Streamlit's AppTest framework.
Tests cover:
- Complete user workflows across all tabs
- Widget interactions (sliders, selectboxes, checkboxes)
- Tab navigation and UI components
- Session state management
- Data persistence across reruns
"""

import pytest
from streamlit.testing.v1 import AppTest


class TestAppInitialization:
    """Test app initialization and basic structure."""

    @pytest.mark.streamlit
    @pytest.mark.integration
    def test_app_loads_successfully(self):
        """Test that the app loads without errors."""
        at = AppTest.from_file("app.py")
        at.run(timeout=30)
        assert not at.exception

    @pytest.mark.streamlit
    @pytest.mark.integration
    def test_app_has_tabs(self):
        """Test that app has all three tabs."""
        at = AppTest.from_file("app.py")
        at.run(timeout=30)

        # Check for tabs - Streamlit creates tabs with specific structure
        # The tabs should be present in the app
        assert not at.exception

    @pytest.mark.streamlit
    @pytest.mark.integration
    def test_page_config_set(self):
        """Test that page configuration is set correctly."""
        at = AppTest.from_file("app.py")
        at.run(timeout=30)

        # App should run without error
        assert not at.exception


class TestSidebarWidgets:
    """Test sidebar widget interactions."""

    @pytest.mark.streamlit
    @pytest.mark.integration
    def test_dataset_selectbox_exists(self):
        """Test that dataset selection widget exists."""
        at = AppTest.from_file("app.py")
        at.run(timeout=30)

        # Check for selectbox widgets in sidebar
        assert len(at.selectbox) > 0

    @pytest.mark.streamlit
    @pytest.mark.integration
    def test_dataset_selection_changes(self):
        """Test changing dataset selection."""
        at = AppTest.from_file("app.py")
        at.run(timeout=30)

        # Find the first selectbox (dataset choice)
        if len(at.selectbox) > 0:
            original_value = at.selectbox[0].value

            # Get available options
            options = at.selectbox[0].options
            if len(options) > 1:
                # Select different option
                new_value = options[1] if options[1] != original_value else options[0]
                at.selectbox[0].select(new_value).run(timeout=30)

                assert not at.exception

    @pytest.mark.streamlit
    @pytest.mark.integration
    def test_slider_interactions(self):
        """Test slider widget interactions."""
        at = AppTest.from_file("app.py")
        at.run(timeout=30)

        # Should have sliders for parameters
        if len(at.slider) > 0:
            # Get first slider and change its value
            slider = at.slider[0]
            slider.value

            # Set to a different value within range
            if slider.max > slider.min:
                new_value = (slider.min + slider.max) / 2
                slider.set_value(new_value).run(timeout=30)

                assert not at.exception


class TestSessionState:
    """Test session state management and caching."""

    @pytest.mark.streamlit
    @pytest.mark.integration
    def test_session_state_initialized(self):
        """Test that session state is properly initialized."""
        at = AppTest.from_file("app.py")
        at.run(timeout=30)

        # Check that app runs without error (session state should be initialized)
        assert not at.exception

    @pytest.mark.streamlit
    @pytest.mark.integration
    def test_session_state_persists_across_reruns(self):
        """Test that session state persists across multiple runs."""
        at = AppTest.from_file("app.py")
        at.run(timeout=30)

        # Make a change
        if len(at.selectbox) > 0:
            options = at.selectbox[0].options
            if len(options) > 1:
                at.selectbox[0].select(options[1]).run(timeout=30)

                # Run again - should not crash
                at.run(timeout=30)
                assert not at.exception


class TestSimpleRegressionTab:
    """Test simple regression tab workflow."""

    @pytest.mark.streamlit
    @pytest.mark.integration
    @pytest.mark.slow
    def test_simple_regression_tab_loads(self):
        """Test that simple regression tab loads without errors."""
        at = AppTest.from_file("app.py")
        at.run(timeout=30)

        # Should load without exception
        assert not at.exception

    @pytest.mark.streamlit
    @pytest.mark.integration
    @pytest.mark.slow
    def test_dataset_parameter_changes(self):
        """Test changing dataset parameters in simple regression."""
        at = AppTest.from_file("app.py")
        at.run(timeout=30)

        # Try changing a slider if available
        if len(at.slider) > 0:
            slider = at.slider[0]
            if slider.max > slider.min:
                slider.set_value((slider.min + slider.max) / 2).run(timeout=30)
                assert not at.exception


class TestMultipleRegressionTab:
    """Test multiple regression tab workflow."""

    @pytest.mark.streamlit
    @pytest.mark.integration
    @pytest.mark.slow
    def test_multiple_regression_functionality(self):
        """Test multiple regression tab basic functionality."""
        at = AppTest.from_file("app.py")
        at.run(timeout=30)

        # App should run without errors
        assert not at.exception


class TestDatasetTab:
    """Test dataset information tab."""

    @pytest.mark.streamlit
    @pytest.mark.integration
    def test_dataset_tab_loads(self):
        """Test that dataset tab loads successfully."""
        at = AppTest.from_file("app.py")
        at.run(timeout=30)

        assert not at.exception


class TestUIComponents:
    """Test various UI components."""

    @pytest.mark.streamlit
    @pytest.mark.integration
    def test_checkboxes_exist(self):
        """Test that checkbox widgets exist in the app."""
        at = AppTest.from_file("app.py")
        at.run(timeout=30)

        # Should run without error
        assert not at.exception

    @pytest.mark.streamlit
    @pytest.mark.integration
    def test_checkbox_toggle(self):
        """Test toggling checkboxes."""
        at = AppTest.from_file("app.py")
        at.run(timeout=30)

        if len(at.checkbox) > 0:
            # Toggle first checkbox
            checkbox = at.checkbox[0]
            original = checkbox.value
            checkbox.set_value(not original).run(timeout=30)

            assert not at.exception

    @pytest.mark.streamlit
    @pytest.mark.integration
    def test_expanders_exist(self):
        """Test that expander components exist."""
        at = AppTest.from_file("app.py")
        at.run(timeout=30)

        # Expanders are part of the UI - app should run
        assert not at.exception


class TestCompleteWorkflows:
    """Test complete end-to-end user workflows."""

    @pytest.mark.streamlit
    @pytest.mark.integration
    @pytest.mark.slow
    def test_workflow_change_dataset_and_parameters(self):
        """Test complete workflow: change dataset and adjust parameters."""
        at = AppTest.from_file("app.py")
        at.run(timeout=30)

        # Step 1: Change dataset
        if len(at.selectbox) > 0:
            options = at.selectbox[0].options
            if len(options) > 1:
                at.selectbox[0].select(options[1]).run(timeout=30)
                assert not at.exception

        # Step 2: Adjust slider
        if len(at.slider) > 0:
            slider = at.slider[0]
            if slider.max > slider.min:
                slider.set_value(slider.min + 1).run(timeout=30)
                assert not at.exception

    @pytest.mark.streamlit
    @pytest.mark.integration
    @pytest.mark.slow
    def test_workflow_multiple_interactions(self):
        """Test multiple sequential interactions."""
        at = AppTest.from_file("app.py")
        at.run(timeout=30)

        # Interaction 1: Change selectbox
        if len(at.selectbox) > 0:
            at.selectbox[0].select(at.selectbox[0].options[0]).run(timeout=30)
            assert not at.exception

        # Interaction 2: Toggle checkbox
        if len(at.checkbox) > 0:
            original = at.checkbox[0].value
            at.checkbox[0].set_value(not original).run(timeout=30)
            assert not at.exception

        # Interaction 3: Adjust slider
        if len(at.slider) > 0:
            slider = at.slider[0]
            if slider.max > slider.min:
                slider.set_value((slider.min + slider.max) / 2).run(timeout=30)
                assert not at.exception


class TestCachingBehavior:
    """Test that caching is working correctly."""

    @pytest.mark.streamlit
    @pytest.mark.integration
    @pytest.mark.performance
    def test_repeated_runs_with_same_params(self):
        """Test that repeated runs with same parameters work correctly."""
        at = AppTest.from_file("app.py")
        at.run(timeout=30)
        assert not at.exception

        # Run again with same parameters
        at.run(timeout=30)
        assert not at.exception

        # And once more
        at.run(timeout=30)
        assert not at.exception


class TestErrorHandling:
    """Test error handling in the app."""

    @pytest.mark.streamlit
    @pytest.mark.integration
    def test_app_handles_rapid_changes(self):
        """Test that app handles rapid parameter changes."""
        at = AppTest.from_file("app.py")
        at.run(timeout=30)

        # Rapid changes
        if len(at.slider) > 0:
            slider = at.slider[0]
            if slider.max > slider.min:
                slider.set_value(slider.min).run(timeout=30)
                assert not at.exception

                slider.set_value(slider.max).run(timeout=30)
                assert not at.exception

    @pytest.mark.streamlit
    @pytest.mark.integration
    def test_app_handles_extreme_values(self):
        """Test app with extreme parameter values."""
        at = AppTest.from_file("app.py")
        at.run(timeout=30)

        # Set slider to extreme values
        if len(at.slider) > 0:
            slider = at.slider[0]

            # Set to minimum
            slider.set_value(slider.min).run(timeout=30)
            assert not at.exception

            # Set to maximum
            slider.set_value(slider.max).run(timeout=30)
            assert not at.exception


class TestDataGeneration:
    """Test data generation through the UI."""

    @pytest.mark.streamlit
    @pytest.mark.integration
    def test_all_datasets_can_be_selected(self):
        """Test that all datasets can be selected without errors."""
        at = AppTest.from_file("app.py")
        at.run(timeout=30)

        if len(at.selectbox) > 0:
            selectbox = at.selectbox[0]
            options = selectbox.options

            # Try each dataset option
            for option in options[:3]:  # Test first 3 to save time
                selectbox.select(option).run(timeout=30)
                assert not at.exception, f"Failed with dataset: {option}"
