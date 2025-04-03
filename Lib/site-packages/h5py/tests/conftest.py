import h5py
import pytest


@pytest.fixture()
def writable_file(tmp_path):
    with h5py.File(tmp_path / 'test.h5', 'w') as f:
        yield f


def pytest_addoption(parser):
    parser.addoption(
        '--no-network', action='store_true', default=False, help='No network access'
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption('--no-network'):
        nonet = pytest.mark.skip(reason='No Internet')
        for item in items:
            if 'nonetwork' in item.keywords:
                item.add_marker(nonet)
