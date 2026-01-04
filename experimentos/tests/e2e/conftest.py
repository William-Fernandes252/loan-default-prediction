"""Configuration for end-to-end tests."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--num-seeds",
        action="store",
        default="1",
        help="Number of seeds to use for tests.",
    )


@pytest.fixture(scope="session")
def num_seeds(pytestconfig: pytest.Config) -> int:
    """Returns the number of seeds to use for tests."""
    return int(pytestconfig.getoption("num_seeds"))
