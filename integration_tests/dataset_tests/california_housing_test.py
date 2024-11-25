from keras.src import testing
from keras.src.datasets import california_housing


class CaliforniaHousingTest(testing.TestCase):
    def test_load_data_large(self):
        (x_train, y_train), (x_test, y_test) = california_housing.load_data(
            version="large"
        )
        self.assertEqual(x_train.shape[1], 8)
        # Ensure the dataset contains 20,640 samples as documented
        self.assertEqual(x_train.shape[0] + x_test.shape[0], 20640)

    def test_load_data_small(self):
        (x_train, y_train), (x_test, y_test) = california_housing.load_data(
            version="small"
        )
        self.assertEqual(x_train.shape[1], 8)
        # Ensure the small dataset contains 600 samples as documented
        self.assertEqual(x_train.shape[0] + x_test.shape[0], 600)

    def test_invalid_version(self):
        with self.assertRaises(ValueError):
            california_housing.load_data(version="invalid_version")

    def test_seed_reproducibility(self):
        # Ensure the data is reproducible with the same seed
        seed = 123
        first_load = california_housing.load_data(version="large", seed=seed)
        second_load = california_housing.load_data(version="large", seed=seed)
        self.assertAllClose(first_load[0][0], second_load[0][0])
        self.assertAllClose(first_load[1][0], second_load[1][0])
