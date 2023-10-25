import numpy as np
import pytest

from keras import layers
from keras import ops
from keras import testing


class FlattenTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
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
        with self.assertRaisesRegex(
            TypeError, "Expected an integer value for `n`"
        ):
            layers.RepeatVector(n="3")

        with self.assertRaisesRegex(
            TypeError, "Expected an integer value for `n`"
        ):
            layers.RepeatVector(n=3.5)

        with self.assertRaisesRegex(
            TypeError, "Expected an integer value for `n`"
        ):
            layers.RepeatVector(n=[3])
