import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import ops
from keras.src import testing


class PermuteTest(testing.TestCase):
    @parameterized.named_parameters(
        [
            {"testcase_name": "dense", "sparse": False},
            {"testcase_name": "sparse", "sparse": True},
        ]
    )
    @pytest.mark.requires_trainable_backend
    def test_permute(self, sparse):
        if sparse and not backend.SUPPORTS_SPARSE_TENSORS:
            pytest.skip("Backend does not support sparse tensors.")

        inputs = np.random.random((10, 3, 5, 5)).astype("float32")
        # Make the ndarray relatively sparse
        inputs = np.multiply(inputs, inputs >= 0.8)
        expected_output = ops.convert_to_tensor(
            np.transpose(inputs, axes=(0, 3, 1, 2))
        )
        if sparse:
            if backend.backend() == "tensorflow":
                import tensorflow as tf

                inputs = tf.sparse.from_dense(inputs)
                expected_output = tf.sparse.from_dense(expected_output)
            elif backend.backend() == "jax":
                import jax.experimental.sparse as jax_sparse

                inputs = jax_sparse.BCOO.fromdense(inputs)
                expected_output = jax_sparse.BCOO.fromdense(expected_output)
            else:
                self.fail(
                    f"Backend {backend.backend()} does not support sparse"
                )

        self.run_layer_test(
            layers.Permute,
            init_kwargs={"dims": (3, 1, 2)},
            input_data=inputs,
            input_sparse=sparse,
            expected_output=expected_output,
            expected_output_sparse=sparse,
            run_training_check=not sparse,
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
