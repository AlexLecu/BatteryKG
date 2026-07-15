"""Shared pytest fixtures."""
import pytest


@pytest.fixture(scope="session")
def neo4j_driver():
    """A live Neo4j driver, or skip the test if the database is unreachable."""
    try:
        from src.kg.connection import get_driver
        driver = get_driver(max_retries=1)
    except Exception as e:  # ConnectionError / RuntimeError
        pytest.skip(f"Neo4j not available: {e}")
    yield driver
    driver.close()
