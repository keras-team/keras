import numpy as np

from keras.testing import test_case
from keras.testing import test_utils


class GetTestDataTest(test_case.TestCase):
    def setUp(self):
        self.train_samples = 100
        self.test_samples = 50
        self.input_shape = (28, 28)
        self.num_classes = 10

    def test_labels_within_range(self):
        """Tests that the labels y_train and y_test within expected range"""
        _, y_train, _, y_test = test_utils.get_test_data(
            self.train_samples,
            self.test_samples,
            self.input_shape,
            self.num_classes,
        )

        self.assertTrue(np.all(y_train < self.num_classes))
        self.assertTrue(np.all(y_train >= 0))
        self.assertTrue(np.all(y_test < self.num_classes))
        self.assertTrue(np.all(y_test >= 0))

    def test_edge_cases_for_zero_samples(self):
        """Tests edge cases when train_samples or test_samples is zero"""
        x_train, _, x_test, _ = test_utils.get_test_data(
            0, self.test_samples, self.input_shape, self.num_classes
        )
        self.assertEqual(len(x_train), 0)

        x_train, _, x_test, _ = test_utils.get_test_data(
            self.train_samples, 0, self.input_shape, self.num_classes
        )
        self.assertEqual(len(x_test), 0)

    def test_get_test_data_returns_correct_number_of_samples(self):
        """Ensures the function returns the expected number of samples."""
        train_samples = 100
        test_samples = 50
        input_shape = (28, 28)
        num_classes = 10

        x_train, y_train, x_test, y_test = test_utils.get_test_data(
            train_samples, test_samples, input_shape, num_classes
        )

        self.assertEqual(len(x_train), train_samples)
        self.assertEqual(len(y_train), train_samples)
        self.assertEqual(len(x_test), test_samples)
        self.assertEqual(len(y_test), test_samples)

    def test_get_test_data_returns_correct_shape_of_data(self):
        """Ensures the function returns data with the expected shape."""
        train_samples = 100
        test_samples = 50
        input_shape = (28, 28)
        num_classes = 10

        x_train, y_train, x_test, y_test = test_utils.get_test_data(
            train_samples, test_samples, input_shape, num_classes
        )

        self.assertEqual(x_train.shape, (train_samples,) + input_shape)
        self.assertEqual(y_train.shape, (train_samples,))
        self.assertEqual(x_test.shape, (test_samples,) + input_shape)
        self.assertEqual(y_test.shape, (test_samples,))

    def test_get_test_data_returns_different_data_for_different_seeds(self):
        """Ensures different data is returned for different seeds."""
        train_samples = 100
        test_samples = 50
        input_shape = (28, 28)
        num_classes = 10

        x_train_1, y_train_1, x_test_1, y_test_1 = test_utils.get_test_data(
            train_samples, test_samples, input_shape, num_classes, random_seed=1
        )
        x_train_2, y_train_2, x_test_2, y_test_2 = test_utils.get_test_data(
            train_samples, test_samples, input_shape, num_classes, random_seed=2
        )

        self.assertFalse(np.array_equal(x_train_1, x_train_2))
        self.assertFalse(np.array_equal(y_train_1, y_train_2))
        self.assertFalse(np.array_equal(x_test_1, x_test_2))
        self.assertFalse(np.array_equal(y_test_1, y_test_2))

    def test_get_test_data_returns_consistent_data_for_same_seed(self):
        """Ensures same data is returned for the same seed."""
        x_train_1, y_train_1, x_test_1, y_test_1 = test_utils.get_test_data(
            self.train_samples,
            self.test_samples,
            self.input_shape,
            self.num_classes,
            random_seed=1,
        )
        x_train_2, y_train_2, x_test_2, y_test_2 = test_utils.get_test_data(
            self.train_samples,
            self.test_samples,
            self.input_shape,
            self.num_classes,
            random_seed=1,
        )

        self.assertTrue(np.array_equal(x_train_1, x_train_2))
        self.assertTrue(np.array_equal(y_train_1, y_train_2))
        self.assertTrue(np.array_equal(x_test_1, x_test_2))
        self.assertTrue(np.array_equal(y_test_1, y_test_2))

    def test_input_shape_variations(self):
        """Tests with different input shapes."""
        input_shape_3d = (28, 28, 3)
        x_train_3d, _, _, _ = test_utils.get_test_data(
            self.train_samples,
            self.test_samples,
            input_shape_3d,
            self.num_classes,
        )
        self.assertEqual(
            x_train_3d.shape, (self.train_samples,) + input_shape_3d
        )
