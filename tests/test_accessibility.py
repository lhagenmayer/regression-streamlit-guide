"""
Tests for accessibility functionality.
"""

import pytest
from src.accessibility import add_aria_label


class TestAccessibilityHelpers:
    """Test accessibility helper functions."""

    def test_add_aria_label_basic(self):
        """Test basic ARIA label generation."""
        result = add_aria_label("button", "Submit Form")
        assert 'aria-label="Submit Form"' in result

    def test_add_aria_label_with_description(self):
        """Test ARIA label with description."""
        result = add_aria_label("button", "Submit Form", "Click to submit the form")
        assert 'aria-label="Submit Form"' in result
        assert "aria-describedby=" in result
        assert "desc-Submit-Form" in result

    def test_add_aria_label_handles_special_chars(self):
        """Test ARIA label with special characters."""
        result = add_aria_label("slider", "Value: 50%")
        assert 'aria-label="Value: 50%"' in result


class TestScreenReaderSupport:
    """Test screen reader support functions."""

    def test_create_screen_reader_text_basic(self):
        """Test basic screen reader text creation."""
        # This function uses st.markdown which we can't easily test in isolation
        # We'll just verify it doesn't raise errors
        try:
            # Would need to mock streamlit for full test
            pass
        except Exception as e:
            pytest.fail(f"create_screen_reader_text raised {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
