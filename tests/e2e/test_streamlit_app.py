import pytest
from streamlit.testing.v1 import AppTest
import sys
import os

# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

class TestStreamlitApp:
    def test_app_startup(self):
        """Test that the app starts up correctly."""
        # Run from root run.py which sets up paths correctly
        at = AppTest.from_file("run.py")
        at.run()
        
        # Check main title
        assert "Simple Regression" in at.title[0].value
        assert not at.exception
        
    def test_sidebar_interactions(self):
        """Test sidebar interactions and state changes."""
        at = AppTest.from_file("run.py")
        at.run()
        
        # Check default state (Simple Regression)
        assert at.sidebar.radio[0].value == "Simple Regression"
        
        # Switch to Multiple Regression
        at.sidebar.radio[0].set_value("Multiple Regression").run()
        
        # Check title update
        assert "Multiple Regression" in at.title[0].value
        assert not at.exception
        
    def test_dataset_selection(self):
        """Test dataset selection updates."""
        at = AppTest.from_file("run.py")
        at.run()
        
        # Get initial selectbox value
        # There might be multiple selectboxes or radios, we need to be careful
        # radio for analysis type, selectbox for dataset
        initial_dataset = at.sidebar.selectbox[0].value
        
        # Change dataset (assuming there are at least 2 options)
        if len(at.sidebar.selectbox[0].options) > 1:
            new_option = at.sidebar.selectbox[0].options[1]
            at.sidebar.selectbox[0].set_value(new_option).run()
            
            # Verify no exceptions after change
            assert not at.exception
            assert at.sidebar.selectbox[0].value == new_option

    def test_parameter_changes(self):
        """Test parameter sliders."""
        at = AppTest.from_file("run.py")
        at.run()
        
        # Change noise level
        # Sliders: 0 (Samples), 1 (Noise) - based on app.py order
        at.sidebar.slider[1].set_value(1.0).run() 
        assert not at.exception

    def test_ai_interpretation_section(self):
        """Test AI interpretation section presence."""
        at = AppTest.from_file("run.py")
        at.run()
        
        # Check for AI section headers
        assert "AI-Interpretation des R-Outputs" in [h.value for h in at.subheader]
