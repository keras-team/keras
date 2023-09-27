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
        x_train, _, x_test, _ = test_utils.get_test_data(
            0, self.test_samples, self.input_shape, self.num_classes
        )
        self.assertEqual(len(x_train), 0)

        x_train, _, x_test, _ = test_utils.get_test_data(
            self.train_samples, 0, self.input_shape, self.num_classes
        )
        self.assertEqual(len(x_test), 0)

    def test_get_test_data_returns_correct_number_of_samples(self):
        x_train, y_train, x_test, y_test = test_utils.get_test_data(
            self.train_samples,
            self.test_samples,
            self.input_shape,
            self.num_classes,
        )
        self.assertEqual(len(x_train), self.train_samples)
        self.assertEqual(len(y_train), self.train_samples)
        self.assertEqual(len(x_test), self.test_samples)
        self.assertEqual(len(y_test), self.test_samples)

    def test_get_test_data_returns_correct_shape_of_data(self):
        x_train, y_train, x_test, y_test = test_utils.get_test_data(
            self.train_samples,
            self.test_samples,
            self.input_shape,
            self.num_classes,
        )
        self.assertEqual(
            x_train.shape, (self.train_samples,) + self.input_shape
        )
        self.assertEqual(y_train.shape, (self.train_samples,))
        self.assertEqual(x_test.shape, (self.test_samples,) + self.input_shape)
        self.assertEqual(y_test.shape, (self.test_samples,))

    def test_get_test_data_returns_different_data_for_different_seeds(self):
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
            random_seed=2,
        )
        self.assertFalse(np.array_equal(x_train_1, x_train_2))
        self.assertFalse(np.array_equal(y_train_1, y_train_2))
        self.assertFalse(np.array_equal(x_test_1, x_test_2))
        self.assertFalse(np.array_equal(y_test_1, y_test_2))

    def test_get_test_data_returns_consistent_data_for_same_seed(self):
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

    def test_all_classes_represented(self):
        """Ensure all classes are represented in the data."""
        _, y_train, _, y_test = test_utils.get_test_data(
            self.train_samples,
            self.test_samples,
            self.input_shape,
            self.num_classes,
        )
        self.assertEqual(len(np.unique(y_train)), self.num_classes)
        self.assertEqual(len(np.unique(y_test)), self.num_classes)

    def test_data_type(self):
        """Validate the type of the generated data."""
        x_train, _, x_test, _ = test_utils.get_test_data(
            self.train_samples,
            self.test_samples,
            self.input_shape,
            self.num_classes,
        )
        self.assertEqual(x_train.dtype, np.float32)
        self.assertEqual(x_test.dtype, np.float32)

    def test_label_type(self):
        """Validate the type of the generated labels."""
        _, y_train, _, y_test = test_utils.get_test_data(
            self.train_samples,
            self.test_samples,
            self.input_shape,
            self.num_classes,
        )
        self.assertEqual(y_train.dtype, np.int64)
        self.assertEqual(y_test.dtype, np.int64)
