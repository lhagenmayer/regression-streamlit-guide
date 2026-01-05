"""
Tests for logging functionality.
"""

import logging
import os
import time
import pytest
from logger import (
    setup_logging,
    get_logger,
    log_function_call,
    log_performance,
    log_data_info,
    log_error_with_context,
    cleanup_old_logs,
)


class TestLoggingSetup:
    """Test logging configuration and setup."""

    def test_setup_logging_creates_logger(self, tmp_path, monkeypatch):
        """Test that setup_logging creates and configures a logger."""
        # Use temporary directory for logs
        log_dir = tmp_path / "logs"
        monkeypatch.setattr("logger.LOG_DIR", log_dir)
        monkeypatch.setattr("logger.LOG_FILE", log_dir / "app.log")
        monkeypatch.setattr("logger.ERROR_LOG_FILE", log_dir / "errors.log")
        monkeypatch.setattr("logger.PERFORMANCE_LOG_FILE", log_dir / "performance.log")

        logger = setup_logging(log_level=logging.DEBUG, enable_file_logging=True)

        assert logger is not None
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) > 0

    def test_setup_logging_without_file_logging(self):
        """Test that logging works without file logging."""
        logger = setup_logging(
            log_level=logging.INFO,
            enable_file_logging=False,
            enable_console_logging=True,
        )

        assert logger is not None
        # Should have at least console handler
        console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(console_handlers) > 0

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a logger instance."""
        logger = get_logger("test_module")

        assert logger is not None
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"


class TestLoggingHelpers:
    """Test logging helper functions."""

    def test_log_function_call(self, caplog):
        """Test logging function calls."""
        logger = get_logger("test")
        logger.setLevel(logging.DEBUG)

        log_function_call(logger, "test_function", param1="value1", param2=42)

        assert "Calling test_function" in caplog.text
        assert "param1=value1" in caplog.text
        assert "param2=42" in caplog.text

    def test_log_performance(self, caplog):
        """Test logging performance metrics."""
        logger = get_logger("test")
        logger.setLevel(logging.INFO)

        log_performance(logger, "data_generation", 1.234)

        assert "Performance" in caplog.text
        assert "data_generation" in caplog.text
        assert "1.234" in caplog.text

    def test_log_data_info(self, caplog):
        """Test logging data information."""
        logger = get_logger("test")
        logger.setLevel(logging.INFO)

        log_data_info(logger, "test_dataset", (100, 5), mean=50.5, std=10.2)

        assert "test_dataset" in caplog.text
        assert "shape=(100, 5)" in caplog.text
        assert "mean=50.5" in caplog.text
        assert "std=10.2" in caplog.text

    def test_log_error_with_context(self, caplog):
        """Test logging errors with context."""
        logger = get_logger("test")
        logger.setLevel(logging.ERROR)

        try:
            raise ValueError("Test error")
        except ValueError as e:
            log_error_with_context(logger, e, "test_operation", param1="value1", param2=42)

        assert "Error in test_operation" in caplog.text
        assert "ValueError" in caplog.text
        assert "Test error" in caplog.text
        assert "param1=value1" in caplog.text


class TestLogCleanup:
    """Test log file cleanup functionality."""

    def test_cleanup_old_logs(self, tmp_path, monkeypatch):
        """Test that old log files are cleaned up."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        monkeypatch.setattr("logger.LOG_DIR", log_dir)

        # Create some old log files
        old_log = log_dir / "old.log"
        old_log.touch()

        # Set modification time to 40 days ago
        old_time = time.time() - (40 * 24 * 60 * 60)
        os.utime(old_log, (old_time, old_time))

        # Create a recent log file
        new_log = log_dir / "new.log"
        new_log.touch()

        # Cleanup logs older than 30 days
        cleanup_old_logs(days=30)

        # Old log should be deleted, new log should remain
        assert not old_log.exists()
        assert new_log.exists()

    def test_cleanup_old_logs_no_dir(self, tmp_path, monkeypatch):
        """Test cleanup when log directory doesn't exist."""
        log_dir = tmp_path / "nonexistent_logs"
        monkeypatch.setattr("logger.LOG_DIR", log_dir)

        # Should not raise an error
        cleanup_old_logs(days=30)


class TestLoggingIntegration:
    """Integration tests for logging."""

    def test_logger_writes_to_file(self, tmp_path, monkeypatch):
        """Test that logger actually writes to file."""
        log_dir = tmp_path / "logs"
        log_file = log_dir / "app.log"
        monkeypatch.setattr("logger.LOG_DIR", log_dir)
        monkeypatch.setattr("logger.LOG_FILE", log_file)
        monkeypatch.setattr("logger.ERROR_LOG_FILE", log_dir / "errors.log")
        monkeypatch.setattr("logger.PERFORMANCE_LOG_FILE", log_dir / "performance.log")

        # Setup logging with file output
        setup_logging(log_level=logging.INFO, enable_file_logging=True)

        # Get a logger and log a message
        logger = get_logger("test")
        logger.info("Test message for file logging")

        # Flush handlers
        for handler in logger.handlers:
            handler.flush()

        # Check that log file was created and contains the message
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message for file logging" in content

    def test_error_logs_to_separate_file(self, tmp_path, monkeypatch):
        """Test that errors are logged to separate error file."""
        log_dir = tmp_path / "logs"
        error_log_file = log_dir / "errors.log"
        monkeypatch.setattr("logger.LOG_DIR", log_dir)
        monkeypatch.setattr("logger.LOG_FILE", log_dir / "app.log")
        monkeypatch.setattr("logger.ERROR_LOG_FILE", error_log_file)
        monkeypatch.setattr("logger.PERFORMANCE_LOG_FILE", log_dir / "performance.log")

        # Setup logging
        setup_logging(log_level=logging.DEBUG, enable_file_logging=True)

        # Get a logger and log an error
        logger = get_logger("test")
        logger.error("Test error message")

        # Flush handlers
        for handler in logger.handlers:
            handler.flush()

        # Check that error log file contains the error
        assert error_log_file.exists()
        content = error_log_file.read_text()
        assert "Test error message" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
