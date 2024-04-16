import numpy as np

from keras.src import testing
from keras.src.datasets import cifar100


class Cifar100LoadDataTest(testing.TestCase):
    def test_shapes_fine_label_mode(self):
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(
            label_mode="fine"
        )
        self.assertEqual(x_train.shape, (50000, 32, 32, 3))
        self.assertEqual(y_train.shape, (50000, 1))
        self.assertEqual(x_test.shape, (10000, 32, 32, 3))
        self.assertEqual(y_test.shape, (10000, 1))

    def test_shapes_coarse_label_mode(self):
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(
            label_mode="coarse"
        )
        self.assertEqual(x_train.shape, (50000, 32, 32, 3))
        self.assertEqual(y_train.shape, (50000, 1))
        self.assertEqual(x_test.shape, (10000, 32, 32, 3))
        self.assertEqual(y_test.shape, (10000, 1))

    def test_dtypes(self):
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        self.assertEqual(x_train.dtype, np.uint8)
        self.assertEqual(y_train.dtype, np.uint8)
        self.assertEqual(x_test.dtype, np.uint8)
        self.assertEqual(y_test.dtype, np.uint8)

    def test_invalid_label_mode(self):
        with self.assertRaises(ValueError):
            cifar100.load_data(label_mode="invalid")
