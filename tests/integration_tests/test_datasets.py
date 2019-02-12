from __future__ import print_function
from mock import patch
import random
import tempfile
import time

import numpy as np
import pytest

from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import reuters
from keras.datasets import imdb
from keras.datasets import mnist
from keras.datasets import boston_housing
from keras.datasets import fashion_mnist


@pytest.fixture
def fake_downloaded_boston_path():
    num_rows = 100
    num_cols = 10
    rng = np.random.RandomState(123)

    x = rng.randint(1, 100, size=(num_rows, num_cols))
    y = rng.normal(loc=100, scale=15, size=num_rows)

    with tempfile.NamedTemporaryFile('wb', delete=True) as f:
        np.savez(f, x=x, y=y)
        with patch('keras.datasets.boston_housing.get_file', return_value=f.name):
            yield f.name


@pytest.fixture
def fake_downloaded_imdb_path():
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
        with patch('keras.datasets.imdb.get_file', return_value=f.name):
            yield f.name


@pytest.fixture
def fake_downloaded_reuters_path():
    num_rows = 100
    seq_length = 10
    rng = np.random.RandomState(123)

    x = rng.randint(1, 100, size=(num_rows, seq_length))
    y = rng.binomial(n=1, p=0.5, size=num_rows)

    with tempfile.NamedTemporaryFile('wb', delete=True) as f:
        np.savez(f, x=x, y=y)
        with patch('keras.datasets.reuters.get_file', return_value=f.name):
            yield f.name


def test_cifar():
    # only run data download tests 20% of the time
    # to speed up frequent testing
    random.seed(time.time())
    if random.random() > 0.8:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        assert len(x_train) == len(y_train) == 50000
        assert len(x_test) == len(y_test) == 10000
        (x_train, y_train), (x_test, y_test) = cifar100.load_data('fine')
        assert len(x_train) == len(y_train) == 50000
        assert len(x_test) == len(y_test) == 10000
        (x_train, y_train), (x_test, y_test) = cifar100.load_data('coarse')
        assert len(x_train) == len(y_train) == 50000
        assert len(x_test) == len(y_test) == 10000


def test_reuters():
    # only run data download tests 20% of the time
    # to speed up frequent testing
    random.seed(time.time())
    if random.random() > 0.8:
        (x_train, y_train), (x_test, y_test) = reuters.load_data()
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        assert len(x_train) + len(x_test) == 11228
        (x_train, y_train), (x_test, y_test) = reuters.load_data(maxlen=10)
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        word_index = reuters.get_word_index()
        assert isinstance(word_index, dict)


def test_mnist():
    # only run data download tests 20% of the time
    # to speed up frequent testing
    random.seed(time.time())
    if random.random() > 0.8:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        assert len(x_train) == len(y_train) == 60000
        assert len(x_test) == len(y_test) == 10000


def test_imdb():
    # only run data download tests 20% of the time
    # to speed up frequent testing
    random.seed(time.time())
    if random.random() > 0.8:
        (x_train, y_train), (x_test, y_test) = imdb.load_data()
        (x_train, y_train), (x_test, y_test) = imdb.load_data(maxlen=40)
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        word_index = imdb.get_word_index()
        assert isinstance(word_index, dict)


def test_boston_housing():
    # only run data download tests 20% of the time
    # to speed up frequent testing
    random.seed(time.time())
    if random.random() > 0.8:
        (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)


def test_fashion_mnist():
    # only run data download tests 20% of the time
    # to speed up frequent testing
    random.seed(time.time())
    if random.random() > 0.8:
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        assert len(x_train) == len(y_train) == 60000
        assert len(x_test) == len(y_test) == 10000


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


if __name__ == '__main__':
    pytest.main([__file__])
