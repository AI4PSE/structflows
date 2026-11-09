import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--run-expensive", action="store_true", default=False, help="run expensive tests"
    )

def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-expensive"):
        # --run-expensive given in cli: do not skip expensive tests
        return
    skip_expensive = pytest.mark.skip(reason="need --run-expensive option to run")
    for item in items:
        if "expensive" in item.keywords:
            item.add_marker(skip_expensive)
