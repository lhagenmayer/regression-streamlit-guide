# Logging Guide

This document describes the logging system implemented in the Linear Regression Guide application.

## Overview

The application uses Python's built-in `logging` module with a centralized configuration that provides:

- **Structured logging** with different severity levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **Multiple log destinations** (console, general log file, error log file, performance log file)
- **Automatic log rotation** to prevent excessive disk usage
- **Contextual information** for easier debugging and monitoring

## Configuration

### Log Files

All log files are stored in the `logs/` directory (excluded from git):

- `logs/app.log` - All application logs (DEBUG and above)
- `logs/errors.log` - Error logs only (ERROR and above)
- `logs/performance.log` - Performance-related logs

### Log Rotation

- **Max file size**: 10 MB per log file
- **Backup count**: 5 backup files
- When a log file reaches the maximum size, it's rotated (e.g., `app.log` → `app.log.1`, etc.)
- Old backups beyond the count limit are automatically deleted

### Log Levels

The application uses the following log levels:

- **DEBUG**: Detailed information for diagnosing problems (function calls, parameters)
- **INFO**: General informational messages (application startup, data generation)
- **WARNING**: Warning messages for potential issues
- **ERROR**: Error messages for serious problems
- **CRITICAL**: Critical issues that may cause application failure

## Usage in Code

### Import the Logger

```python
from logger import get_logger

logger = get_logger(__name__)
```

### Basic Logging

```python
# Informational messages
logger.info("Processing dataset with 100 observations")

# Warning messages
logger.warning("Large dataset may impact performance")

# Error messages
logger.error("Failed to load data", exc_info=True)
```

### Helper Functions

The logger module provides several helper functions for common logging patterns:

#### Log Function Calls

```python
from logger import log_function_call

log_function_call(logger, "generate_data", n=100, seed=42)
# Output: "Calling generate_data(n=100, seed=42)"
```

#### Log Performance Metrics

```python
from logger import log_performance

start = time.time()
# ... do work ...
duration = time.time() - start
log_performance(logger, "data_generation", duration)
# Output: "Performance: data_generation took 0.123s"
```

#### Log Errors with Context

```python
from logger import log_error_with_context

try:
    result = process_data(df)
except Exception as e:
    log_error_with_context(
        logger, e, "process_data", 
        dataset="cities", n_rows=100
    )
# Output: "Error in process_data: ValueError: Invalid data. Details: dataset=cities, n_rows=100"
```

## Log Cleanup

Old log files are automatically cleaned up based on age. You can manually trigger cleanup:

```python
from logger import cleanup_old_logs

# Delete logs older than 30 days
cleanup_old_logs(days=30)
```

## Disabling Logging

To disable logging (e.g., for tests or specific environments), set the environment variable:

```bash
export DISABLE_LOGGING=1
streamlit run app.py
```

## Viewing Logs

### In Development

Tail the logs in real-time:

```bash
# Watch all logs
tail -f logs/app.log

# Watch only errors
tail -f logs/errors.log

# Search for specific patterns
grep "ERROR" logs/app.log
```

### In Production

Consider using log aggregation tools like:

- **Logstash** + **Elasticsearch** + **Kibana** (ELK Stack)
- **Fluentd** for log collection and forwarding
- **Splunk** for enterprise log management
- **CloudWatch Logs** (AWS) or **Cloud Logging** (GCP)

## Best Practices

1. **Use appropriate log levels**: Don't log everything as ERROR
2. **Include context**: Add relevant parameters and state information
3. **Don't log sensitive data**: Avoid logging passwords, API keys, or personal information
4. **Performance**: Excessive DEBUG logging can impact performance; use INFO in production
5. **Structured data**: Use key-value pairs for easier parsing and searching

## Examples

### Logging Data Generation

```python
logger.info(f"Generating multiple regression data: dataset={dataset_choice}, n={n_samples}")
start_time = time.time()

# Generate data
data = generate_dataset(...)

duration = time.time() - start_time
logger.info(f"Generated data in {duration:.3f}s: shape=({n_samples}, {n_features})")
```

### Logging with Error Handling

```python
try:
    logger.debug("Fetching data from API")
    response = requests.get(api_url, timeout=10)
    response.raise_for_status()
    logger.info(f"Successfully fetched data: {len(response.json())} records")
except requests.Timeout:
    logger.error(f"API request timed out: {api_url}")
except requests.RequestException as e:
    log_error_with_context(logger, e, "API request", url=api_url)
```

## Troubleshooting

### Log Files Not Created

- Check that the application has write permissions for the `logs/` directory
- Verify that `DISABLE_LOGGING` environment variable is not set
- Check console output for errors during logger initialization

### Logs Too Verbose

Adjust the log level in `logger.py`:

```python
DEFAULT_LOG_LEVEL = logging.WARNING  # Instead of INFO
```

### Logs Taking Too Much Space

- Reduce `MAX_BYTES` or `BACKUP_COUNT` in `logger.py`
- Run `cleanup_old_logs()` more frequently
- Set up automated log cleanup in cron or system scheduler

## Future Enhancements

Potential improvements for the logging system:

- [ ] JSON structured logging for easier parsing
- [ ] Integration with external logging services (e.g., Sentry, Datadog)
- [ ] Request ID tracking for distributed tracing
- [ ] User action logging for analytics
- [ ] Log filtering by module or component
- [ ] Real-time log streaming to monitoring dashboard
