import tempfile

import numpy as np
import pytest

from keras.datasets import boston_housing
from keras.datasets import imdb
from keras.datasets import reuters


@pytest.fixture
def fake_downloaded_boston_path(monkeypatch):
    num_rows = 100
    num_cols = 10
    rng = np.random.RandomState(123)

    x = rng.randint(1, 100, size=(num_rows, num_cols))
    y = rng.normal(loc=100, scale=15, size=num_rows)

    with tempfile.NamedTemporaryFile('wb', delete=True) as f:
        np.savez(f, x=x, y=y)
        monkeypatch.setattr(boston_housing, 'get_file',
                            lambda *args, **kwargs: f.name)
        yield f.name


@pytest.fixture
def fake_downloaded_imdb_path(monkeypatch):
    train_rows = 100
    test_rows = 20
    seq_length = 10
    rng = np.random.RandomState(123)

    x_train = rng.randint(1, 100, size=(train_rows, seq_length))
    y_train = rng.binomial(n=1, p=0.5, size=train_rows)
    x_test = rng.randint(1, 100, size=(test_rows, seq_length))
    y_test = rng.binomial(n=1, p=0.5, size=test_rows)

    with tempfile.NamedTemporaryFile('wb', delete=True) as f:
        np.savez(f, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        monkeypatch.setattr(imdb, 'get_file', lambda *args, **kwargs: f.name)
        yield f.name


@pytest.fixture
def fake_downloaded_reuters_path(monkeypatch):
    num_rows = 100
    seq_length = 10
    rng = np.random.RandomState(123)

    x = rng.randint(1, 100, size=(num_rows, seq_length))
    y = rng.binomial(n=1, p=0.5, size=num_rows)

    with tempfile.NamedTemporaryFile('wb', delete=True) as f:
        np.savez(f, x=x, y=y)
        monkeypatch.setattr(reuters, 'get_file', lambda *args, **kwargs: f.name)
        yield f.name


def test_boston_load_does_not_affect_global_rng(fake_downloaded_boston_path):
    np.random.seed(1337)
    before = np.random.randint(0, 100, size=10)

    np.random.seed(1337)
    boston_housing.load_data(path=fake_downloaded_boston_path, seed=9876)
    after = np.random.randint(0, 100, size=10)

    assert np.array_equal(before, after)


def test_imdb_load_does_not_affect_global_rng(fake_downloaded_imdb_path):
    np.random.seed(1337)
    before = np.random.randint(0, 100, size=10)

    np.random.seed(1337)
    imdb.load_data(path=fake_downloaded_imdb_path, seed=9876)
    after = np.random.randint(0, 100, size=10)

    assert np.array_equal(before, after)


def test_reuters_load_does_not_affect_global_rng(fake_downloaded_reuters_path):
    np.random.seed(1337)
    before = np.random.randint(0, 100, size=10)

    np.random.seed(1337)
    reuters.load_data(path=fake_downloaded_reuters_path, seed=9876)
    after = np.random.randint(0, 100, size=10)

    assert np.array_equal(before, after)
