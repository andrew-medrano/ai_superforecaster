# AI Superforecaster Tests

This directory contains tests for the AI Superforecaster application.

## API Server Tests

The `test_api_server.py` file contains tests for the FastAPI server that provides a RESTful interface to the forecasting engine. These tests verify:

1. Creating a new forecast session
2. Getting forecast status and results
3. Getting buffer contents
4. End-to-end forecasting flow
5. Forecast completion status
6. Running multiple forecasts simultaneously
7. Error handling
8. Input validation
9. Long-running forecast monitoring

### Real Forecasting Tests

The test suite includes special tests marked with `@pytest.mark.slow` that perform actual end-to-end testing with the real forecasting engine:

- `test_real_forecast_completion`: Waits for a forecast to fully complete, which may take 30+ seconds
- `test_forecast_result_structure`: Validates that the forecast result has the expected structure when completed

These real forecasting tests ensure the entire system works together correctly, from API to forecasting engine and back. They perform actual waits and validate real responses.

## Running Tests

You can run the tests with pytest:

```bash
# Run all tests
python -m pytest

# Run API server tests only
python -m pytest tests/test_api_server.py

# Run a specific test with verbose output
python -m pytest test_api_server.py::test_name -v

# Skip the slow tests that wait for real forecasts
python -m pytest -m "not slow"

# Run only the slow/real forecasting tests
python -m pytest -m "slow"
```

## Test Requirements

Test dependencies are listed in `requirements-test.txt`. Make sure to install them before running tests:

```bash
pip install -r tests/requirements-test.txt
``` 