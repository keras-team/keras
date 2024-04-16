import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import testing
from keras.src.backend.common.keras_tensor import KerasTensor


class ReshapeTest(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        [
            {"testcase_name": "dense", "sparse": False},
            {"testcase_name": "sparse", "sparse": True},
        ]
    )
    @pytest.mark.requires_trainable_backend
    def test_reshape(self, sparse):
        if sparse and not backend.SUPPORTS_SPARSE_TENSORS:
            pytest.skip("Backend does not support sparse tensors.")

        self.run_layer_test(
            layers.Reshape,
            init_kwargs={"target_shape": (8, 1)},
            input_shape=(3, 2, 4),
            input_sparse=sparse,
            expected_output_shape=(3, 8, 1),
            expected_output_sparse=sparse,
            run_training_check=not sparse,
        )

        self.run_layer_test(
            layers.Reshape,
            init_kwargs={"target_shape": (8,)},
            input_shape=(3, 2, 4),
            input_sparse=sparse,
            expected_output_shape=(3, 8),
            expected_output_sparse=sparse,
            run_training_check=not sparse,
        )

        self.run_layer_test(
            layers.Reshape,
            init_kwargs={"target_shape": (2, 4)},
            input_shape=(3, 8),
            input_sparse=sparse,
            expected_output_shape=(3, 2, 4),
            expected_output_sparse=sparse,
            run_training_check=not sparse,
        )
        self.run_layer_test(
            layers.Reshape,
            init_kwargs={"target_shape": (-1, 1)},
            input_shape=(3, 2, 4),
            input_sparse=sparse,
            expected_output_shape=(3, 8, 1),
            expected_output_sparse=sparse,
            run_training_check=not sparse,
        )

        self.run_layer_test(
            layers.Reshape,
            init_kwargs={"target_shape": (1, -1)},
            input_shape=(3, 2, 4),
            input_sparse=sparse,
            expected_output_shape=(3, 1, 8),
            expected_output_sparse=sparse,
            run_training_check=not sparse,
        )

        self.run_layer_test(
            layers.Reshape,
            init_kwargs={"target_shape": (-1,)},
            input_shape=(3, 2, 4),
            input_sparse=sparse,
            expected_output_shape=(3, 8),
            expected_output_sparse=sparse,
            run_training_check=not sparse,
        )

        self.run_layer_test(
            layers.Reshape,
            init_kwargs={"target_shape": (2, -1)},
            input_shape=(3, 2, 4),
            input_sparse=sparse,
            expected_output_shape=(3, 2, 4),
            expected_output_sparse=sparse,
            run_training_check=not sparse,
        )

    def test_reshape_with_dynamic_batch_size(self):
        input_layer = layers.Input(shape=(2, 4))
        reshaped = layers.Reshape((8,))(input_layer)
        self.assertEqual(reshaped.shape, (None, 8))

    def test_reshape_with_dynamic_batch_size_and_minus_one(self):
        input = KerasTensor((None, 6, 4))
        layer = layers.Reshape((-1, 8))
        layer.build(input.shape)
        reshaped = backend.compute_output_spec(layer.__call__, input)
        self.assertEqual(reshaped.shape, (None, 3, 8))

    def test_reshape_with_dynamic_dim_and_minus_one(self):
        input = KerasTensor((4, 6, None, 3))
        layer = layers.Reshape((-1, 3))
        layer.build(input.shape)
        reshaped = backend.compute_output_spec(layer.__call__, input)
        self.assertEqual(reshaped.shape, (4, None, 3))

    def test_reshape_sets_static_shape(self):
        input_layer = layers.Input(batch_shape=(2, None))
        reshaped = layers.Reshape((3, 5))(input_layer)
        # Also make sure the batch dim is not lost after reshape.
        self.assertEqual(reshaped.shape, (2, 3, 5))
