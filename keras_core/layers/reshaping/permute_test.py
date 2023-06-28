import numpy as np
import pytest

from keras_core import backend
from keras_core import layers
from keras_core import ops
from keras_core import testing


class PermuteTest(testing.TestCase):
    def test_permute(self):
        inputs = np.random.random((2, 3, 5)).astype("float32")
        expected_output = ops.convert_to_tensor(
            np.transpose(inputs, axes=(0, 2, 1))
        )
        self.run_layer_test(
            layers.Permute,
            init_kwargs={"dims": (2, 1)},
            input_data=inputs,
            expected_output=expected_output,
        )

    @pytest.mark.skipif(
        not backend.DYNAMIC_SHAPES_OK,
        reason="Backend does not support dynamic shapes",
    )
    def test_permute_with_dynamic_batch_size(self):
        input_layer = layers.Input(batch_shape=(None, 3, 5))
        permuted = layers.Permute((2, 1))(input_layer)
        self.assertEqual(permuted.shape, (None, 5, 3))

    def test_permute_errors_on_invalid_starting_dims_index(self):
        with self.assertRaisesRegex(
            ValueError, r"Invalid permutation .*dims.*"
        ):
            self.run_layer_test(
                layers.Permute,
                init_kwargs={"dims": (0, 1, 2)},
                input_shape=(3, 2, 4),
            )

    def test_permute_errors_on_invalid_set_of_dims_indices(self):
        with self.assertRaisesRegex(
            ValueError, r"Invalid permutation .*dims.*"
        ):
            self.run_layer_test(
                layers.Permute,
                init_kwargs={"dims": (1, 4, 2)},
                input_shape=(3, 2, 4),
            )
