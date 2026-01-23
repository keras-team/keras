import os

import numpy as np
import pytest
from absl.testing import parameterized
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.saving import saving_api
from keras.src.testing.test_utils import named_product


class DiscretizationTest(testing.TestCase):
    def test_discretization_basics(self):
        self.run_layer_test(
            layers.Discretization,
            init_kwargs={
                "bin_boundaries": [0.0, 0.5, 1.0],
            },
            input_shape=(2, 3),
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
            run_training_check=False,
        )

    def test_adapt_flow(self):
        layer = layers.Discretization(num_bins=4)
        layer.adapt(
            np.random.random((32, 3)),
        )
        output = layer(np.array([[0.0, 0.1, 0.3]]))
        self.assertTrue(output.dtype, "int32")

    @parameterized.named_parameters(
        named_product(
            [
                {
                    "testcase_name": "int",
                    "output_mode": "int",
                    "input_array": [[-1.0, 0.0, 0.1, 0.8, 1.2]],
                    "expected_output": [[0, 1, 1, 2, 3]],
                },
                {
                    "testcase_name": "one_hot_rank_1",
                    "output_mode": "one_hot",
                    "input_array": [0.1, 0.8],
                    "expected_output": [[0, 1, 0, 0], [0, 0, 1, 0]],
                },
                {
                    "testcase_name": "multi_hot_rank_2",
                    "output_mode": "multi_hot",
                    "input_array": [[0.1, 0.8]],
                    "expected_output": [[0, 1, 1, 0]],
                },
                {
                    "testcase_name": "one_hot_rank_3",
                    "output_mode": "one_hot",
                    "input_array": [[[0.15, 0.75], [0.85, 0.45]]],
                    "expected_output": [
                        [
                            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                            [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                        ]
                    ],
                },
                {
                    "testcase_name": "multi_hot_rank_3",
                    "output_mode": "multi_hot",
                    "input_array": [[[0.15, 0.75], [0.85, 0.45]]],
                    "expected_output": [
                        [[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]]
                    ],
                },
                {
                    "testcase_name": "count",
                    "output_mode": "count",
                    "input_array": [[0.1, 0.8, 0.9]],
                    "expected_output": [[0, 1, 2, 0]],
                },
            ],
            sparse=(
                [True, False] if backend.SUPPORTS_SPARSE_TENSORS else [False]
            ),
        )
    )
    def test_correctness(
        self, output_mode, input_array, expected_output, sparse
    ):
        if output_mode == "int" and sparse:
            pytest.skip("sparse=True cannot be combined with output_mode=int")

        input_array = np.array(input_array)
        expected_output = np.array(expected_output)

        layer = layers.Discretization(
            bin_boundaries=[0.0, 0.5, 1.0],
            output_mode=output_mode,
            sparse=sparse,
        )
        output = layer(input_array)
        self.assertSparse(output, sparse)
        self.assertTrue(backend.is_tensor(output))
        self.assertAllClose(output, expected_output)

    def test_tf_data_compatibility(self):
        # With fixed bins
        layer = layers.Discretization(
            bin_boundaries=[0.0, 0.35, 0.5, 1.0], dtype="float32"
        )
        x = np.array([[-1.0, 0.0, 0.1, 0.2, 0.4, 0.5, 1.0, 1.2, 0.98]])
        self.assertAllClose(layer(x), np.array([[0, 1, 1, 1, 2, 3, 4, 4, 3]]))
        ds = tf_data.Dataset.from_tensor_slices(x).batch(1).map(layer)
        for output in ds.take(1):
            output = output.numpy()
        self.assertAllClose(output, np.array([[0, 1, 1, 1, 2, 3, 4, 4, 3]]))

        # With adapt flow
        layer = layers.Discretization(num_bins=4)
        layer.adapt(
            np.random.random((32, 3)),
        )
        x = np.array([[0.0, 0.1, 0.3]])
        ds = tf_data.Dataset.from_tensor_slices(x).batch(1).map(layer)
        for output in ds.take(1):
            output.numpy()

    def test_serialization(self):
        layer = layers.Discretization(num_bins=5)

        # Serialization before `adapt` is called.
        config = layer.get_config()
        revived_layer = layers.Discretization.from_config(config)
        self.assertEqual(config, revived_layer.get_config())

        # Serialization after `adapt` is called but `num_bins` was not reached.
        layer.adapt(np.array([0.0, 1.0, 5.0]))
        config = layer.get_config()
        revived_layer = layers.Discretization.from_config(config)
        self.assertEqual(config, revived_layer.get_config())

        # Serialization after `adapt` is called and `num_bins` is reached.
        layer.adapt(np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
        config = layer.get_config()
        revived_layer = layers.Discretization.from_config(config)
        self.assertEqual(config, revived_layer.get_config())

        # Serialization with `bin_boundaries`.
        layer = layers.Discretization(bin_boundaries=[0.0, 0.35, 0.5, 1.0])
        config = layer.get_config()
        revived_layer = layers.Discretization.from_config(config)
        self.assertEqual(config, revived_layer.get_config())

    def test_saving(self):
        # With fixed bins
        layer = layers.Discretization(bin_boundaries=[0.0, 0.35, 0.5, 1.0])
        model = models.Sequential(
            [
                layers.Input((2,)),
                layer,
            ]
        )
        fpath = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(fpath)
        model = saving_api.load_model(fpath)
        x = np.array([[-1.0, 0.0, 0.1, 0.2, 0.4, 0.5, 1.0, 1.2, 0.98]])
        self.assertAllClose(layer(x), np.array([[0, 1, 1, 1, 2, 3, 4, 4, 3]]))

        # With adapt flow
        layer = layers.Discretization(num_bins=4)
        layer.adapt(
            np.random.random((32, 3)),
        )
        ref_input = np.random.random((1, 2))
        ref_output = layer(ref_input)
        model = models.Sequential(
            [
                layers.Input((2,)),
                layer,
            ]
        )
        fpath = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(fpath)
        model = saving_api.load_model(fpath)
        self.assertAllClose(layer(ref_input), ref_output)

    def test_init_num_bins_and_bin_boundaries_raises(self):
        with self.assertRaisesRegex(
            ValueError, "Both `num_bins` and `bin_boundaries`"
        ):
            layers.Discretization(num_bins=3, bin_boundaries=[0.0, 1.0])

        with self.assertRaisesRegex(
            ValueError, "either `num_bins` or `bin_boundaries`"
        ):
            layers.Discretization()

    def test_call_before_adapt_raises(self):
        layer = layers.Discretization(num_bins=3)
        with self.assertRaisesRegex(ValueError, "You need .* call .*adapt"):
            layer([[0.1, 0.8, 0.9]])

    def test_model_call_vs_predict_consistency(self):
        """Test that model(input) and model.predict(input) produce consistent outputs."""  # noqa: E501
        # Test with int output mode
        layer = layers.Discretization(
            bin_boundaries=[-0.5, 0, 0.1, 0.2, 3],
            output_mode="int",
        )
        x = np.array([[0.0, 0.15, 0.21, 0.3], [0.0, 0.17, 0.451, 7.8]])

        # Create model
        inputs = layers.Input(shape=(4,), dtype="float32")
        outputs = layer(inputs)
        model = models.Model(inputs=inputs, outputs=outputs)

        # Test both execution modes
        model_call_output = model(x)
        predict_output = model.predict(x)

        # Check consistency
        self.assertAllClose(model_call_output, predict_output)
        self.assertEqual(
            backend.standardize_dtype(model_call_output.dtype),
            backend.standardize_dtype(predict_output.dtype),
        )
        self.assertTrue(backend.is_int_dtype(model_call_output.dtype))

    def test_symbolic_tensor_output_shape(self):
        """Test symbolic tensors have correct output shape for different modes.

        This test ensures that issue #22044 is fixed - Discretization layer
        with output_mode='one_hot' should produce correct output shape for
        symbolic tensors.
        """
        # Test one_hot mode - should add depth dimension
        layer_one_hot = layers.Discretization(
            bin_boundaries=[0.0, 1.0, 2.0], output_mode="one_hot"
        )
        symbolic_input = layers.Input(shape=(3, 4))
        symbolic_output = layer_one_hot(symbolic_input)
        # 3 boundaries create 4 bins, so output should be (None, 3, 4, 4)
        self.assertEqual(symbolic_output.shape, (None, 3, 4, 4))

        # Test multi_hot mode - should replace last dimension with depth
        layer_multi_hot = layers.Discretization(
            bin_boundaries=[0.0, 1.0, 2.0], output_mode="multi_hot"
        )
        symbolic_output_multi = layer_multi_hot(symbolic_input)
        # multi_hot replaces last dimension, so output should be (None, 3, 4)
        self.assertEqual(symbolic_output_multi.shape, (None, 3, 4))

        # Test count mode - should replace last dimension with depth
        layer_count = layers.Discretization(
            bin_boundaries=[0.0, 1.0, 2.0], output_mode="count"
        )
        symbolic_output_count = layer_count(symbolic_input)
        # count replaces last dimension, so output should be (None, 3, 4)
        self.assertEqual(symbolic_output_count.shape, (None, 3, 4))

        # Test int mode - should keep same shape
        layer_int = layers.Discretization(
            bin_boundaries=[0.0, 1.0, 2.0], output_mode="int"
        )
        symbolic_output_int = layer_int(symbolic_input)
        # int mode keeps same shape, so output should be (None, 3, 4)
        self.assertEqual(symbolic_output_int.shape, (None, 3, 4))

        # Test consistency between eager and symbolic tensors for one_hot mode
        eager_input = np.random.uniform(0, 3, size=(2, 3, 4))
        eager_output = layer_one_hot(eager_input)
        # Shapes should be consistent (ignoring batch dimension)
        self.assertEqual(eager_output.shape[1:], symbolic_output.shape[1:])

    def test_compute_output_shape_edge_cases(self):
        """Test edge cases in compute_output_shape to improve coverage."""

        # Test with num_bins instead of bin_boundaries
        layer_num_bins = layers.Discretization(
            num_bins=5, output_mode="one_hot"
        )
        shape = layer_num_bins.compute_output_shape((None, 3, 4))
        expected = (None, 3, 4, 5)  # num_bins=5
        self.assertEqual(shape, expected)

        # Test different output modes
        modes_and_shapes = [
            ("int", (None, 3, 4), (None, 3, 4)),  # int mode - no change
            ("one_hot", (None, 3, 4), (None, 3, 4, 3)),  # one_hot - add dim
            ("multi_hot", (None, 3, 4), (None, 3, 3)),  # multi_hot - replace
            ("count", (None, 3, 4), (None, 3, 3)),  # count - replace
        ]

        for mode, input_shape, expected_shape in modes_and_shapes:
            layer = layers.Discretization(
                bin_boundaries=[0.0, 1.0], output_mode=mode
            )
            result_shape = layer.compute_output_shape(input_shape)
            self.assertEqual(result_shape, expected_shape)

        # Test edge case - last dimension is 1 with one_hot
        layer_one_hot = layers.Discretization(
            bin_boundaries=[0.0, 1.0], output_mode="one_hot"
        )

        # Should replace last dimension of 1 with depth
        shape = layer_one_hot.compute_output_shape((None, 5, 1))
        expected = (None, 5, 3)  # 2 boundaries = 3 bins, replace last dim
        self.assertEqual(shape, expected)

        # Test empty input shape
        shape = layer_one_hot.compute_output_shape(())
        expected = (3,)  # Just depth
        self.assertEqual(shape, expected)

    def test_compute_output_spec_method(self):
        """Test compute_output_spec method directly."""

        layer = layers.Discretization(
            bin_boundaries=[0.0, 1.0, 2.0], output_mode="one_hot"
        )

        # Create a KerasTensor input
        input_tensor = backend.KerasTensor(shape=(None, 3, 4), dtype="float32")

        # Test compute_output_spec
        output_spec = layer.compute_output_spec(input_tensor)

        # Verify shape and dtype
        expected_shape = (None, 3, 4, 4)  # 3 boundaries = 4 bins
        self.assertEqual(output_spec.shape, expected_shape)
        self.assertEqual(output_spec.dtype, layer.output_dtype)
