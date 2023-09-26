import numpy as np
from absl.testing import parameterized

from keras import backend
from keras import testing
from keras.backend.common.variables import KerasVariable
from keras.utils import numerical_utils

num_classes = 5


class TestNumericalUtils(testing.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        [
            ((1,), (1, num_classes)),
            ((3,), (3, num_classes)),
            ((4, 3), (4, 3, num_classes)),
            ((5, 4, 3), (5, 4, 3, num_classes)),
            ((3, 1), (3, num_classes)),
            ((3, 2, 1), (3, 2, num_classes)),
        ]
    )
    def test_to_categorical(self, shape, expected_shape):
        """Test categorical conversion of labels."""
        label = np.random.randint(0, num_classes, shape)
        one_hot = numerical_utils.to_categorical(label, num_classes)
        self.assertEqual(one_hot.shape, expected_shape)
        self.assertTrue(np.all(one_hot.sum(axis=-1) == 1))
        self.assertTrue(
            np.all(np.argmax(one_hot, -1).reshape(label.shape) == label)
        )

    def test_to_categorial_without_num_classes(self):
        """Test conversion without specifying number of classes."""
        label = [0, 2, 5]
        one_hot = numerical_utils.to_categorical(label)
        self.assertEqual(one_hot.shape, (3, 5 + 1))

    def test_to_categorical_with_backend_tensor(self):
        """Test conversion with backend tensors."""
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
        one_hot = numerical_utils.to_categorical(label, num_classes)
        self.assertAllClose(one_hot, expected)

    @parameterized.parameters([1, 2, 3])
    def test_normalize(self, order):
        """Test the normalization function."""
        xb = backend.random.uniform((3, 3), seed=1337)
        xnp = backend.convert_to_numpy(xb)
        l2 = np.atleast_1d(np.linalg.norm(xnp, order, axis=-1))
        l2[l2 == 0] = 1
        expected = xnp / np.expand_dims(l2, axis=-1)
        out = numerical_utils.normalize(xnp, axis=-1, order=order)
        self.assertAllClose(out, expected)
        out = numerical_utils.normalize(xb, axis=-1, order=order)
        self.assertAllClose(backend.convert_to_numpy(out), expected)

    def test_numpy_input_with_num_classes(self):
        """Test numpy input with a specified number of classes."""
        label = np.array([0, 2, 1, 3, 4])
        one_hot = numerical_utils.to_categorical(label, 5)
        expected = np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        )
        np.testing.assert_array_equal(one_hot, expected)

    def test_without_num_classes(self):
        """Test conversion without specifying number of classes."""
        label = np.array([0, 2, 5])
        one_hot = numerical_utils.to_categorical(label)
        expected = np.array(
            [[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1]]
        )
        np.testing.assert_array_equal(one_hot, expected)

    def test_non_standard_labels(self):
        """Test with non-standard labels."""
        label = np.array([2, 4, 8])
        one_hot = numerical_utils.to_categorical(label)
        expected = np.array(
            [
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )
        np.testing.assert_array_equal(one_hot, expected)

    def test_with_backend_tensor(self):
        """Test categorical conversion for backend tensors."""
        label_tensor = backend.convert_to_tensor(np.array([0, 2, 1, 3, 4]))
        one_hot = numerical_utils.to_categorical(label_tensor, 5)
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

        # one_hot_value = one_hot.value
        # expected_value = expected.value
        one_hot_value = one_hot.numpy()
        expected_value = expected.numpy()
        np.testing.assert_array_equal(one_hot_value, expected_value)

    def test_encode_categorical_inputs_basic(self):
        """Test basic encoding of categorical inputs."""
        inputs = backend.convert_to_tensor(np.array([0, 1, 2, 1, 0, 2]))
        output = numerical_utils.encode_categorical_inputs(
            inputs, output_mode="int", depth=3
        )
        expected_output = backend.convert_to_tensor(
            np.array([0, 1, 2, 1, 0, 2])
        )
        self.assertAllClose(output, expected_output)

    def test_encode_categorical_inputs_output_modes(self):
        """Test various output modes for encoding."""
        inputs = backend.convert_to_tensor(np.array([0, 1, 2, 1, 0, 2]))

        output_one_hot = numerical_utils.encode_categorical_inputs(
            inputs, output_mode="one_hot", depth=3
        )
        expected_one_hot = backend.convert_to_tensor(
            np.array(
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 0],
                    [0, 0, 1],
                ]
            )
        )
        self.assertAllClose(output_one_hot, expected_one_hot)

    def test_normalize_invalid_order(self):
        """Test normalization with an invalid order."""
        x = np.array([1, 2, 3])
        with self.assertRaisesRegex(
            ValueError,
            "Argument `order` must be an int >= 1. Received: order=0",
        ):
            numerical_utils.normalize(x, order=0)

    def test_encode_categorical_inputs_multi_hot(self):
        """Test multi-hot encoding of categorical inputs."""
        inputs = backend.convert_to_tensor(np.array([[0, 1], [2, 1], [0, 2]]))
        output_multi_hot = numerical_utils.encode_categorical_inputs(
            inputs, output_mode="multi_hot", depth=3
        )
        expected_multi_hot = backend.convert_to_tensor(
            np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
        )
        self.assertAllClose(output_multi_hot, expected_multi_hot)

    def test_encode_categorical_inputs_invalid_output_mode(self):
        """Test encoding with an invalid output mode."""
        inputs = backend.convert_to_tensor(
            np.array([[[0, 1, 2], [1, 0, 2]], [[1, 2, 0], [2, 0, 1]]])
        )
        with self.assertRaisesRegex(
            ValueError,
            "When output_mode is not `'int'`, maximum supported output rank "
            "is 2. Received output_mode invalid_mode and input shape "
            "\(2, 2, 3\), which would result in output rank 3.",
        ):
            numerical_utils.encode_categorical_inputs(
                inputs, "invalid_mode", depth=3
            )
