import numpy as np
import pytest
from absl.testing import parameterized
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import testing


class NormalizationTest(testing.TestCase, parameterized.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_normalization_basics(self):
        self.run_layer_test(
            layers.Normalization,
            init_kwargs={
                "axis": -1,
            },
            input_shape=(2, 3),
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=3,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.Normalization,
            init_kwargs={
                "axis": -1,
                "mean": np.array([0.5, 0.2, -0.1]),
                "variance": np.array([0.1, 0.2, 0.3]),
            },
            input_shape=(2, 3),
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.Normalization,
            init_kwargs={
                "axis": -1,
                "mean": np.array([0.5, 0.2, -0.1]),
                "variance": np.array([0.1, 0.2, 0.3]),
                "invert": True,
            },
            input_shape=(2, 3),
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )

    @parameterized.parameters([("np",), ("tensor",), ("tf.data")])
    def test_normalization_adapt(self, input_type):
        x = np.random.random((32, 4))
        if input_type == "np":
            data = x
        elif input_type == "tensor":
            data = backend.convert_to_tensor(x)
        elif input_type == "tf.data":
            data = tf_data.Dataset.from_tensor_slices(x).batch(8)

        layer = layers.Normalization()
        layer.adapt(data)
        self.assertTrue(layer.built)
        output = layer(x)
        output = backend.convert_to_numpy(output)
        self.assertAllClose(np.var(output, axis=0), 1.0, atol=1e-5)
        self.assertAllClose(np.mean(output, axis=0), 0.0, atol=1e-5)

        # Test in high-dim and with tuple axis.
        x = np.random.random((32, 4, 3, 5))
        if input_type == "np":
            data = x
        elif input_type == "tensor":
            data = backend.convert_to_tensor(x)
        elif input_type == "tf.data":
            data = tf_data.Dataset.from_tensor_slices(x).batch(8)

        layer = layers.Normalization(axis=(1, 2))
        layer.adapt(data)
        self.assertTrue(layer.built)
        output = layer(x)
        output = backend.convert_to_numpy(output)
        self.assertAllClose(np.var(output, axis=(0, 3)), 1.0, atol=1e-5)
        self.assertAllClose(np.mean(output, axis=(0, 3)), 0.0, atol=1e-5)

    @pytest.mark.skipif(
        backend.backend() != "torch",
        reason="Test symbolic call for torch meta device.",
    )
    def test_call_on_meta_device_after_built(self):
        from keras.src.backend.torch import core

        layer = layers.Normalization()
        data = np.random.random((32, 4))
        layer.adapt(data)
        with core.device_scope("meta"):
            layer(data)

    def test_normalization_with_mean_only_raises_error(self):
        # Test error when only `mean` is provided
        with self.assertRaisesRegex(
            ValueError, "both `mean` and `variance` must be set"
        ):
            layers.Normalization(mean=0.5)

    def test_normalization_with_variance_only_raises_error(self):
        # Test error when only `variance` is provided
        with self.assertRaisesRegex(
            ValueError, "both `mean` and `variance` must be set"
        ):
            layers.Normalization(variance=0.1)

    def test_normalization_axis_too_high(self):
        with self.assertRaisesRegex(
            ValueError, "All `axis` values must be in the range"
        ):
            layer = layers.Normalization(axis=3)
            layer.build((2, 2))

    def test_normalization_axis_too_low(self):
        with self.assertRaisesRegex(
            ValueError, "All `axis` values must be in the range"
        ):
            layer = layers.Normalization(axis=-4)
            layer.build((2, 3, 4))

    def test_normalization_unknown_axis_shape(self):
        with self.assertRaisesRegex(ValueError, "All `axis` values to be kept"):
            layer = layers.Normalization(axis=1)
            layer.build((None, None))

    def test_normalization_adapt_with_incompatible_shape(self):
        layer = layers.Normalization(axis=-1)
        initial_shape = (10, 5)
        layer.build(initial_shape)
        new_shape_data = np.random.random((10, 3))
        with self.assertRaisesRegex(ValueError, "an incompatible shape"):
            layer.adapt(new_shape_data)

    def test_tf_data_compatibility(self):
        x = np.random.random((32, 3))
        ds = tf_data.Dataset.from_tensor_slices(x).batch(1)

        # With built-in values
        layer = layers.Normalization(
            mean=[0.1, 0.2, 0.3], variance=[0.1, 0.2, 0.3], axis=-1
        )
        layer.build((None, 3))
        for output in ds.map(layer).take(1):
            output.numpy()

        # With adapt flow
        layer = layers.Normalization(axis=-1)
        layer.adapt(
            np.random.random((32, 3)),
        )
        for output in ds.map(layer).take(1):
            output.numpy()
