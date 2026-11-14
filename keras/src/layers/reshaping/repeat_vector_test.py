import numpy as np

from keras.src import layers
from keras.src import ops
from keras.src import testing


class FlattenTest(testing.TestCase):
    def test_repeat_vector(self):
        inputs = np.random.random((2, 5)).astype("float32")
        expected_output = ops.convert_to_tensor(
            np.repeat(np.reshape(inputs, (2, 1, 5)), 3, axis=1)
        )
        self.run_layer_test(
            layers.RepeatVector,
            init_kwargs={"n": 3},
            input_data=inputs,
            expected_output=expected_output,
        )

    def test_repeat_vector_with_dynamic_batch_size(self):
        input_layer = layers.Input(batch_shape=(None, 5))
        repeated = layers.RepeatVector(n=3)(input_layer)
        self.assertEqual(repeated.shape, (None, 3, 5))

    def test_repeat_vector_with_dynamic_dimension(self):
        input_layer = layers.Input(batch_shape=(2, None))
        repeated = layers.RepeatVector(n=3)(input_layer)
        self.assertEqual(repeated.shape, (2, 3, None))

    def test_repeat_vector_with_invalid_n(self):
        for n in ("3", 3.5, [3], True):
            with self.assertRaisesRegex(
                TypeError, "Expected `n` to be an integer"
            ):
                layers.RepeatVector(n=n)

    def test_repeat_vector_with_non_positive_n(self):
        for n in (0, -3):
            with self.assertRaisesRegex(
                ValueError, "Expected `n` to be an integer >= 1"
            ):
                layers.RepeatVector(n=n)

    def test_numpy_integer_n(self):
        layer = layers.RepeatVector(n=np.int64(3))
        self.assertIs(type(layer.n), int)
        self.assertEqual(layer.n, 3)
