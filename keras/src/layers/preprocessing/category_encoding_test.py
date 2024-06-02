import numpy as np
from absl.testing import parameterized
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import testing

TEST_CASES = [{"testcase_name": "dense", "sparse": False}]
if backend.SUPPORTS_SPARSE_TENSORS:
    TEST_CASES += [{"testcase_name": "sparse", "sparse": True}]


class CategoryEncodingTest(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(TEST_CASES)
    def test_count_output(self, sparse):
        input_array = np.array([1, 2, 3, 1])
        expected_output = np.array([0, 2, 1, 1, 0, 0])

        num_tokens = 6
        expected_output_shape = (num_tokens,)

        layer = layers.CategoryEncoding(
            num_tokens=num_tokens, output_mode="count", sparse=sparse
        )
        int_data = layer(input_array)
        self.assertEqual(expected_output_shape, int_data.shape)
        self.assertAllClose(int_data, expected_output)
        self.assertSparse(int_data, sparse)

        # Test symbolic call.
        output = layer(
            layers.Input(batch_shape=input_array.shape, dtype="int32")
        )
        self.assertEqual(expected_output_shape, output.shape)
        self.assertEqual("float32", output.dtype)
        self.assertSparse(output, sparse)

    @parameterized.named_parameters(TEST_CASES)
    def test_count_weighted_output(self, sparse):
        input_array = np.array([[0, 1], [0, 0], [1, 2], [3, 1]])
        count_weights = np.array(
            [[0.1, 0.2], [0.1, 0.1], [0.2, 0.3], [0.4, 0.2]]
        )
        expected_output = np.array(
            [
                [0.1, 0.2, 0.0, 0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.2, 0.3, 0.0, 0.0, 0.0],
                [0.0, 0.2, 0.0, 0.4, 0.0, 0.0],
            ]
        )

        num_tokens = 6
        expected_output_shape = (input_array.shape[0], num_tokens)

        layer = layers.CategoryEncoding(
            num_tokens=num_tokens, output_mode="count", sparse=sparse
        )
        int_data = layer(input_array, count_weights=count_weights)
        self.assertEqual(expected_output_shape, int_data.shape)
        self.assertAllClose(int_data, expected_output)
        self.assertSparse(int_data, sparse)

        # Test symbolic call.
        output = layer(
            layers.Input(batch_shape=input_array.shape, dtype="int32"),
            count_weights=layers.Input(
                batch_shape=input_array.shape, dtype="float32"
            ),
        )
        self.assertEqual(expected_output_shape, output.shape)
        self.assertEqual("float32", output.dtype)
        self.assertSparse(output, sparse)

    @parameterized.named_parameters(TEST_CASES)
    def test_batched_count_output(self, sparse):
        input_array = np.array([[1, 2, 3, 1], [0, 3, 1, 0]])
        expected_output = np.array([[0, 2, 1, 1, 0, 0], [2, 1, 0, 1, 0, 0]])

        num_tokens = 6
        expected_output_shape = (2, num_tokens)

        layer = layers.CategoryEncoding(
            num_tokens=num_tokens, output_mode="count", sparse=sparse
        )
        int_data = layer(input_array)
        self.assertEqual(expected_output_shape, int_data.shape)
        self.assertAllClose(int_data, expected_output)
        self.assertSparse(int_data, sparse)

        # Test symbolic call.
        output = layer(
            layers.Input(batch_shape=input_array.shape, dtype="int32")
        )
        self.assertEqual(expected_output_shape, output.shape)
        self.assertEqual("float32", output.dtype)
        self.assertSparse(output, sparse)

    @parameterized.named_parameters(TEST_CASES)
    def test_multi_hot(self, sparse):
        input_data = np.array([3, 2, 0, 1])
        expected_output = np.array([1, 1, 1, 1, 0, 0])
        num_tokens = 6
        expected_output_shape = (num_tokens,)

        # Test call on layer directly.
        layer = layers.CategoryEncoding(
            num_tokens=num_tokens, output_mode="multi_hot", sparse=sparse
        )
        output_data = layer(input_data)
        self.assertAllClose(expected_output, output_data)
        self.assertEqual(expected_output_shape, output_data.shape)
        self.assertSparse(output_data, sparse)

        # Test symbolic call.
        output = layer(
            layers.Input(batch_shape=input_data.shape, dtype="int32")
        )
        self.assertEqual(expected_output_shape, output.shape)
        self.assertEqual("float32", output.dtype)
        self.assertSparse(output, sparse)

    @parameterized.named_parameters(TEST_CASES)
    def test_batched_multi_hot(self, sparse):
        input_data = np.array([[3, 2, 0, 1], [3, 2, 0, 1]])
        expected_output = np.array([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0]])
        num_tokens = 6
        expected_output_shape = (input_data.shape[0], num_tokens)

        # Test call on layer directly.
        layer = layers.CategoryEncoding(
            num_tokens=num_tokens, output_mode="multi_hot", sparse=sparse
        )
        output_data = layer(input_data)
        self.assertAllClose(expected_output, output_data)
        self.assertEqual(expected_output_shape, output_data.shape)
        self.assertSparse(output_data, sparse)

        # Test symbolic call.
        output = layer(
            layers.Input(batch_shape=input_data.shape, dtype="int32")
        )
        self.assertEqual(expected_output_shape, output.shape)
        self.assertEqual("float32", output.dtype)
        self.assertSparse(output, sparse)

        # Test compute_output_shape
        input_data = np.array((4))
        layer = layers.CategoryEncoding(
            num_tokens=num_tokens, output_mode="multi_hot", sparse=sparse
        )
        self.assertEqual(
            layer(input_data).shape,
            layer.compute_output_shape(input_data.shape),
        )

    @parameterized.named_parameters(TEST_CASES)
    def test_one_hot(self, sparse):
        input_data = np.array([3, 2, 0, 1])
        expected_output = np.array(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
            ]
        )
        num_tokens = 6
        expected_output_shape = (input_data.shape[0], num_tokens)

        # Test call on layer directly.
        layer = layers.CategoryEncoding(
            num_tokens=num_tokens, output_mode="one_hot", sparse=sparse
        )
        output_data = layer(input_data)
        self.assertAllClose(expected_output, output_data)
        self.assertEqual(expected_output_shape, output_data.shape)
        self.assertSparse(output_data, sparse)

        # Test symbolic call.
        output = layer(
            layers.Input(batch_shape=input_data.shape, dtype="int32")
        )
        self.assertEqual(expected_output_shape, output.shape)
        self.assertEqual("float32", output.dtype)
        self.assertSparse(output, sparse)

        # Test compute_output_shape
        layer = layers.CategoryEncoding(
            num_tokens=num_tokens, output_mode="one_hot", sparse=sparse
        )
        self.assertEqual(
            layer(input_data).shape,
            layer.compute_output_shape(input_data.shape),
        )

        # Test compute_output_shape with 1 extra dimension
        input_data = np.array([[3], [2], [0], [1]])
        layer = layers.CategoryEncoding(
            num_tokens=num_tokens, output_mode="one_hot", sparse=sparse
        )
        self.assertEqual(
            layer(input_data).shape,
            layer.compute_output_shape(input_data.shape),
        )

        input_data = np.array((4,))
        layer = layers.CategoryEncoding(
            num_tokens=num_tokens, output_mode="one_hot", sparse=sparse
        )
        self.assertEqual(
            layer(input_data).shape,
            layer.compute_output_shape(input_data.shape),
        )

    @parameterized.named_parameters(TEST_CASES)
    def test_batched_one_hot(self, sparse):
        input_data = np.array([[3, 2, 0, 1], [3, 2, 0, 1]])
        expected_output = np.array(
            [
                [
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                ],
            ]
        )
        num_tokens = 6
        expected_output_shape = input_data.shape[0:2] + (num_tokens,)

        # Test call on layer directly.
        layer = layers.CategoryEncoding(
            num_tokens=num_tokens, output_mode="one_hot", sparse=sparse
        )
        output_data = layer(input_data)
        self.assertAllClose(expected_output, output_data)
        self.assertEqual(expected_output_shape, output_data.shape)
        self.assertSparse(output_data, sparse)

        # Test symbolic call.
        output = layer(
            layers.Input(batch_shape=input_data.shape, dtype="int32")
        )
        self.assertEqual(expected_output_shape, output.shape)
        self.assertEqual("float32", output.dtype)
        self.assertSparse(output, sparse)

    def test_tf_data_compatibility(self):
        layer = layers.CategoryEncoding(
            num_tokens=4, output_mode="one_hot", dtype="int32"
        )
        input_data = np.array([3, 2, 0, 1])
        expected_output = np.array(
            [
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ]
        )
        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(4).map(layer)
        for output in ds.take(1):
            output = output.numpy()
        self.assertAllClose(output, expected_output)

    def test_category_encoding_without_num_tokens(self):
        with self.assertRaisesRegex(
            ValueError, r"num_tokens must be set to use this layer"
        ):
            layers.CategoryEncoding(output_mode="multi_hot")

    def test_category_encoding_with_invalid_num_tokens(self):
        with self.assertRaisesRegex(ValueError, r"`num_tokens` must be >= 1"):
            layers.CategoryEncoding(num_tokens=0, output_mode="multi_hot")

        with self.assertRaisesRegex(ValueError, r"`num_tokens` must be >= 1"):
            layers.CategoryEncoding(num_tokens=-1, output_mode="multi_hot")

    def test_category_encoding_with_unnecessary_count_weights(self):
        layer = layers.CategoryEncoding(num_tokens=4, output_mode="multi_hot")
        input_data = np.array([0, 1, 2, 3])
        count_weights = np.array([0.1, 0.2, 0.3, 0.4])
        with self.assertRaisesRegex(
            ValueError, r"`count_weights` is not used when `output_mode`"
        ):
            layer(input_data, count_weights=count_weights)

    def test_invalid_output_mode_raises_error(self):
        with self.assertRaisesRegex(
            ValueError, r"Unknown arg for output_mode: invalid_mode"
        ):
            layers.CategoryEncoding(num_tokens=4, output_mode="invalid_mode")

    def test_encode_one_hot_single_sample(self):
        layer = layers.CategoryEncoding(num_tokens=4, output_mode="one_hot")
        input_array = np.array([1, 2, 3, 1])
        expected_output = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 1, 0, 0],
            ]
        )
        output = layer._encode(input_array)
        self.assertAllClose(expected_output, output)

    def test_encode_one_hot_batched_samples(self):
        layer = layers.CategoryEncoding(num_tokens=4, output_mode="one_hot")
        input_array = np.array([[3, 2, 0, 1], [3, 2, 0, 1]])
        expected_output = np.array(
            [
                [[0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]],
                [[0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]],
            ]
        )
        output = layer._encode(input_array)
        self.assertAllClose(expected_output, output)

    def test_count_single_sample(self):
        layer = layers.CategoryEncoding(num_tokens=4, output_mode="count")
        input_array = np.array([1, 2, 3, 1])
        expected_output = np.array([0, 2, 1, 1])
        output = layer(input_array)
        self.assertAllClose(expected_output, output)

    def test_count_batched_samples(self):
        layer = layers.CategoryEncoding(num_tokens=4, output_mode="count")
        input_array = np.array([[1, 2, 3, 1], [0, 3, 1, 0]])
        expected_output = np.array([[0, 2, 1, 1], [2, 1, 0, 1]])
        output = layer(input_array)
        self.assertAllClose(expected_output, output)
