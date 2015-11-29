from __future__ import print_function
import unittest
from keras.datasets import cifar10, cifar100, reuters, imdb, mnist


class TestDatasets(unittest.TestCase):
    def test_cifar(self):
        print('cifar10')
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)

        print('cifar100 fine')
        (X_train, y_train), (X_test, y_test) = cifar100.load_data('fine')
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)

        print('cifar100 coarse')
        (X_train, y_train), (X_test, y_test) = cifar100.load_data('coarse')
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)

    def test_reuters(self):
        print('reuters')
        (X_train, y_train), (X_test, y_test) = reuters.load_data()

    def test_mnist(self):
        print('mnist')
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)

    def test_imdb(self):
        print('imdb')
        (X_train, y_train), (X_test, y_test) = imdb.load_data()


if __name__ == '__main__':
    print('Test datasets')
    unittest.main()
