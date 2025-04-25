import pytest

# Register custom markers
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (tests that wait for real forecasts)"
    ) 