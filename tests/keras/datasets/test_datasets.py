from __future__ import print_function
import pytest
from keras.datasets import cifar10, cifar100, reuters, imdb, mnist


def test_cifar():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    (X_train, y_train), (X_test, y_test) = cifar100.load_data('fine')
    (X_train, y_train), (X_test, y_test) = cifar100.load_data('coarse')


def test_reuters():
    (X_train, y_train), (X_test, y_test) = reuters.load_data()
    (X_train, y_train), (X_test, y_test) = reuters.load_data(maxlen=10)


def test_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()


def test_imdb():
    (X_train, y_train), (X_test, y_test) = imdb.load_data()
    (X_train, y_train), (X_test, y_test) = imdb.load_data(maxlen=40)


if __name__ == '__main__':
    pytest.main([__file__])
