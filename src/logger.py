"""
Logging configuration for the Linear Regression Guide.

This module provides a centralized logging setup with structured logging,
log rotation, and different log levels for various components.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from datetime import datetime


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Log directory and file paths
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "app.log"
ERROR_LOG_FILE = LOG_DIR / "errors.log"
PERFORMANCE_LOG_FILE = LOG_DIR / "performance.log"

# Log levels
DEFAULT_LOG_LEVEL = logging.INFO
CONSOLE_LOG_LEVEL = logging.WARNING
FILE_LOG_LEVEL = logging.DEBUG

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - " "%(funcName)s:%(lineno)d - %(message)s"
CONSOLE_FORMAT = "%(levelname)s - %(name)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log rotation settings
MAX_BYTES = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5  # Keep 5 backup files


def setup_logging(
    log_level: int = DEFAULT_LOG_LEVEL,
    enable_file_logging: bool = True,
    enable_console_logging: bool = True,
) -> logging.Logger:
    """
    Set up application-wide logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_file_logging: Whether to log to files
        enable_console_logging: Whether to log to console

    Returns:
        Configured root logger
    """
    # Create logs directory if it doesn't exist
    if enable_file_logging:
        LOG_DIR.mkdir(exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    console_formatter = logging.Formatter(CONSOLE_FORMAT)

    # Console handler (for warnings and errors)
    if enable_console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(CONSOLE_LOG_LEVEL)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # File handler with rotation (for all logs)
    if enable_file_logging:
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                LOG_FILE,
                maxBytes=MAX_BYTES,
                backupCount=BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setLevel(FILE_LOG_LEVEL)
            file_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(file_handler)

            # Separate error log file
            error_handler = logging.handlers.RotatingFileHandler(
                ERROR_LOG_FILE,
                maxBytes=MAX_BYTES,
                backupCount=BACKUP_COUNT,
                encoding="utf-8",
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(error_handler)

            # Performance log file
            perf_handler = logging.handlers.RotatingFileHandler(
                PERFORMANCE_LOG_FILE,
                maxBytes=MAX_BYTES,
                backupCount=BACKUP_COUNT,
                encoding="utf-8",
            )
            perf_handler.setLevel(logging.INFO)
            perf_handler.setFormatter(detailed_formatter)
            # Add filter to only log performance-related messages
            perf_handler.addFilter(lambda record: "performance" in record.name.lower())
            root_logger.addHandler(perf_handler)

        except (OSError, IOError) as e:
            # If file logging fails, at least log to console
            if enable_console_logging:
                root_logger.warning(f"Could not set up file logging: {e}")
            else:
                print(f"Warning: Could not set up logging: {e}", file=sys.stderr)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_function_call(logger: logging.Logger, func_name: str, **kwargs):
    """
    Log a function call with its parameters.

    Args:
        logger: Logger instance
        func_name: Name of the function being called
        **kwargs: Function parameters to log
    """
    params = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.debug(f"Calling {func_name}({params})")


def log_performance(logger: logging.Logger, operation: str, duration: float):
    """
    Log performance metrics.

    Args:
        logger: Logger instance
        operation: Description of the operation
        duration: Duration in seconds
    """
    logger.info(f"Performance: {operation} took {duration:.3f}s")


def log_data_info(logger: logging.Logger, data_name: str, shape: tuple, **info):
    """
    Log data generation or loading information.

    Args:
        logger: Logger instance
        data_name: Name of the dataset
        shape: Shape of the data (rows, columns)
        **info: Additional information about the data
    """
    info_str = ", ".join(f"{k}={v}" for k, v in info.items())
    logger.info(f"Data '{data_name}': shape={shape}, {info_str}")


def log_error_with_context(logger: logging.Logger, error: Exception, context: str, **details):
    """
    Log an error with contextual information.

    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Description of what was being done when error occurred
        **details: Additional details about the error context
    """
    details_str = ", ".join(f"{k}={v}" for k, v in details.items())
    logger.error(
        f"Error in {context}: {type(error).__name__}: {error}. Details: {details_str}",
        exc_info=True,
    )


def cleanup_old_logs(days: int = 30):
    """
    Clean up log files older than specified days.

    Args:
        days: Number of days to keep logs (default: 30)
    """
    if not LOG_DIR.exists():
        return

    cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)

    for log_file in LOG_DIR.glob("*.log*"):
        try:
            if log_file.stat().st_mtime < cutoff_time:
                log_file.unlink()
                print(f"Deleted old log file: {log_file}")
        except (OSError, IOError) as e:
            print(f"Could not delete log file {log_file}: {e}")


# ============================================================================
# INITIALIZE LOGGING ON MODULE IMPORT
# ============================================================================

# Set up logging when module is imported
# This can be disabled by setting environment variable DISABLE_LOGGING=1
if not os.getenv("DISABLE_LOGGING"):
    setup_logging()
