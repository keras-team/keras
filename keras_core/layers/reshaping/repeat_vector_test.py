import numpy as np
import pytest

from keras_core import backend
from keras_core import layers
from keras_core import ops
from keras_core import testing


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

    @pytest.mark.skipif(
        not backend.DYNAMIC_SHAPES_OK,
        reason="Backend does not support dynamic shapes",
    )
    def test_repeat_vector_with_dynamic_batch_size(self):
        input_layer = layers.Input(batch_shape=(None, 5))
        repeated = layers.RepeatVector(n=3)(input_layer)
        self.assertEqual(repeated.shape, (None, 3, 5))

    @pytest.mark.skipif(
        not backend.DYNAMIC_SHAPES_OK,
        reason="Backend does not support dynamic shapes",
    )
    def test_repeat_vector_with_dynamic_dimension(self):
        input_layer = layers.Input(batch_shape=(2, None))
        repeated = layers.RepeatVector(n=3)(input_layer)
        self.assertEqual(repeated.shape, (2, 3, None))
