from __future__ import print_function
import pytest
import time
import random
from keras.datasets import cifar10, cifar100, reuters, imdb, mnist


def test_cifar():
    # only run data download tests 20% of the time
    # to speed up frequent testing
    random.seed(time.time())
    if random.random() > 0.8:
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        (X_train, y_train), (X_test, y_test) = cifar100.load_data('fine')
        (X_train, y_train), (X_test, y_test) = cifar100.load_data('coarse')


def test_reuters():
    # only run data download tests 20% of the time
    # to speed up frequent testing
    random.seed(time.time())
    if random.random() > 0.8:
        (X_train, y_train), (X_test, y_test) = reuters.load_data()
        (X_train, y_train), (X_test, y_test) = reuters.load_data(maxlen=10)


def test_mnist():
    # only run data download tests 20% of the time
    # to speed up frequent testing
    random.seed(time.time())
    if random.random() > 0.8:
        (X_train, y_train), (X_test, y_test) = mnist.load_data()


def test_imdb():
    # only run data download tests 20% of the time
    # to speed up frequent testing
    random.seed(time.time())
    if random.random() > 0.8:
        (X_train, y_train), (X_test, y_test) = imdb.load_data()
        (X_train, y_train), (X_test, y_test) = imdb.load_data(maxlen=40)


if __name__ == '__main__':
    pytest.main([__file__])
