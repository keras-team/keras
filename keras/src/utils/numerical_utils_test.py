import numpy as np
from absl.testing import parameterized

from keras.src import backend
from keras.src import testing
from keras.src.utils import numerical_utils

NUM_CLASSES = 5


class TestNumericalUtils(testing.TestCase):
    @parameterized.parameters(
        [
            ((1,), (1, NUM_CLASSES)),
            ((3,), (3, NUM_CLASSES)),
            ((4, 3), (4, 3, NUM_CLASSES)),
            ((5, 4, 3), (5, 4, 3, NUM_CLASSES)),
            ((3, 1), (3, NUM_CLASSES)),
            ((3, 2, 1), (3, 2, NUM_CLASSES)),
        ]
    )
    def test_to_categorical(self, shape, expected_shape):
        label = np.random.randint(0, NUM_CLASSES, shape)
        one_hot = numerical_utils.to_categorical(label, NUM_CLASSES)
        # Check shape
        self.assertEqual(one_hot.shape, expected_shape)
        # Make sure there is only one 1 in a row
        self.assertTrue(np.all(one_hot.sum(axis=-1) == 1))
        # Get original labels back from one hots
        self.assertTrue(
            np.all(np.argmax(one_hot, -1).reshape(label.shape) == label)
        )

    def test_to_categorical_without_num_classes(self):
        label = [0, 2, 5]
        one_hot = numerical_utils.to_categorical(label)
        self.assertEqual(one_hot.shape, (3, 5 + 1))

    def test_to_categorical_with_backend_tensor(self):
        label = backend.convert_to_tensor(np.array([0, 2, 1, 3, 4]))
        expected = backend.convert_to_tensor(
            np.array(
                [
                    [1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                ]
            )
        )
        one_hot = numerical_utils.to_categorical(label, NUM_CLASSES)
        assert backend.is_tensor(one_hot)
        self.assertAllClose(one_hot, expected)

    @parameterized.parameters([1, 2, 3])
    def test_normalize(self, order):
        xb = backend.random.uniform((3, 3), seed=1337)
        xnp = backend.convert_to_numpy(xb)

        # Expected result
        l2 = np.atleast_1d(np.linalg.norm(xnp, order, axis=-1))
        l2[l2 == 0] = 1
        expected = xnp / np.expand_dims(l2, axis=-1)

        # Test NumPy
        out = numerical_utils.normalize(xnp, axis=-1, order=order)
        self.assertIsInstance(out, np.ndarray)
        self.assertAllClose(out, expected)

        # Test backend
        out = numerical_utils.normalize(xb, axis=-1, order=order)
        self.assertTrue(backend.is_tensor(out))
        self.assertAllClose(backend.convert_to_numpy(out), expected)

    def test_build_pos_neg_masks(self):
        query_labels = np.array([0, 1, 2, 2, 0])
        key_labels = np.array([0, 1, 2, 0, 2])
        expected_shape = (len(query_labels), len(key_labels))

        positive_mask, negative_mask = numerical_utils.build_pos_neg_masks(
            query_labels, key_labels, remove_diagonal=False
        )

        positive_mask = backend.convert_to_numpy(positive_mask)
        negative_mask = backend.convert_to_numpy(negative_mask)
        self.assertEqual(positive_mask.shape, expected_shape)
        self.assertEqual(negative_mask.shape, expected_shape)
        self.assertTrue(
            np.all(np.logical_not(np.logical_and(positive_mask, negative_mask)))
        )

        expected_positive_mask_keep_diag = np.array(
            [
                [1, 0, 0, 1, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 1],
                [0, 0, 1, 0, 1],
                [1, 0, 0, 1, 0],
            ],
            dtype="bool",
        )
        self.assertTrue(
            np.all(positive_mask == expected_positive_mask_keep_diag)
        )
        self.assertTrue(
            np.all(
                negative_mask
                == np.logical_not(expected_positive_mask_keep_diag)
            )
        )

        positive_mask, negative_mask = numerical_utils.build_pos_neg_masks(
            query_labels, key_labels, remove_diagonal=True
        )
        positive_mask = backend.convert_to_numpy(positive_mask)
        negative_mask = backend.convert_to_numpy(negative_mask)
        self.assertEqual(positive_mask.shape, expected_shape)
        self.assertEqual(negative_mask.shape, expected_shape)
        self.assertTrue(
            np.all(np.logical_not(np.logical_and(positive_mask, negative_mask)))
        )

        expected_positive_mask_with_remove_diag = np.array(
            [
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 1, 0, 1],
                [1, 0, 0, 1, 0],
            ],
            dtype="bool",
        )
        self.assertTrue(
            np.all(positive_mask == expected_positive_mask_with_remove_diag)
        )

        query_labels = np.array([1, 2, 3])
        key_labels = np.array([1, 2, 3, 1])

        positive_mask, negative_mask = numerical_utils.build_pos_neg_masks(
            query_labels, key_labels, remove_diagonal=True
        )
        positive_mask = backend.convert_to_numpy(positive_mask)
        negative_mask = backend.convert_to_numpy(negative_mask)
        expected_shape_diff_sizes = (len(query_labels), len(key_labels))
        self.assertEqual(positive_mask.shape, expected_shape_diff_sizes)
        self.assertEqual(negative_mask.shape, expected_shape_diff_sizes)
        self.assertTrue(
            np.all(np.logical_not(np.logical_and(positive_mask, negative_mask)))
        )
