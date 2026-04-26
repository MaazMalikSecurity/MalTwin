"""
conftest.py
===========
Shared pytest configuration and fixtures for MalTwin test suite.
"""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: tests that require the Malimg dataset at data/malimg/"
    )


@pytest.fixture(scope="session")
def device():
    """Return 'cpu' — all unit tests run on CPU regardless of host GPU."""
    return "cpu"


@pytest.fixture(scope="session")
def num_classes():
    return 25


@pytest.fixture(scope="session")
def img_size():
    return 128
