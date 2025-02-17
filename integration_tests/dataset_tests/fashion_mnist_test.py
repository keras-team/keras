import numpy as np

from keras.src import testing
from keras.src.datasets import fashion_mnist


class FashionMnistLoadDataTest(testing.TestCase):
    def test_x_train_shape(self):
        (x_train, _), _ = fashion_mnist.load_data()
        self.assertEqual(x_train.shape, (60000, 28, 28))

    def test_y_train_shape(self):
        (_, y_train), _ = fashion_mnist.load_data()
        self.assertEqual(y_train.shape, (60000,))

    def test_x_test_shape(self):
        _, (x_test, _) = fashion_mnist.load_data()
        self.assertEqual(x_test.shape, (10000, 28, 28))

    def test_y_test_shape(self):
        _, (_, y_test) = fashion_mnist.load_data()
        self.assertEqual(y_test.shape, (10000,))

    def test_x_train_dtype(self):
        (x_train, _), _ = fashion_mnist.load_data()
        self.assertEqual(x_train.dtype, np.uint8)

    def test_y_train_dtype(self):
        (_, y_train), _ = fashion_mnist.load_data()
        self.assertEqual(y_train.dtype, np.uint8)

    def test_x_test_dtype(self):
        _, (x_test, _) = fashion_mnist.load_data()
        self.assertEqual(x_test.dtype, np.uint8)

    def test_y_test_dtype(self):
        _, (_, y_test) = fashion_mnist.load_data()
        self.assertEqual(y_test.dtype, np.uint8)
