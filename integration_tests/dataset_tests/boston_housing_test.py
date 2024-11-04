from keras.src import testing
from keras.src.datasets import boston_housing


class BostonHousingTest(testing.TestCase):
    def test_load_data(self):
        (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
        self.assertEqual(x_train.shape[1], 13)
        self.assertEqual(x_train.shape[0] + x_test.shape[0], 506)

    def test_seed_reproducibility(self):
        seed = 123
        first_load = boston_housing.load_data(seed=seed)
        second_load = boston_housing.load_data(seed=seed)
        self.assertAllClose(first_load[0][0], second_load[0][0])
        self.assertAllClose(first_load[1][0], second_load[1][0])

    def test_invalid_test_split(self):
        with self.assertRaises(AssertionError):
            boston_housing.load_data(test_split=-0.1)
        with self.assertRaises(AssertionError):
            boston_housing.load_data(test_split=1.0)
