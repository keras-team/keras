import numpy as np
import pytest
from absl.testing import parameterized
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import testing
from keras.src.trainers.data_adapters.py_dataset_adapter import PyDataset


class NormalizationTest(testing.TestCase):
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
        else:
            raise NotImplementedError(input_type)

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
        layer = layers.Normalization()
        data = np.random.random((32, 4))
        layer.adapt(data)
        with backend.device("meta"):
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

    def test_normalization_with_scalar_mean_var(self):
        input_data = np.array([[1, 2, 3]], dtype="float32")
        layer = layers.Normalization(mean=3.0, variance=2.0)
        layer(input_data)

    @parameterized.parameters([("x",), ("x_and_y",), ("x_y_and_weights",)])
    def test_adapt_pydataset_compat(self, pydataset_type):
        import keras

        class CustomDataset(PyDataset):
            def __len__(self):
                return 100

            def __getitem__(self, idx):
                x = np.random.rand(32, 32, 3)
                y = np.random.randint(0, 10, size=(1,))
                weights = np.random.randint(0, 10, size=(1,))
                if pydataset_type == "x":
                    return x
                elif pydataset_type == "x_and_y":
                    return x, y
                elif pydataset_type == "x_y_and_weights":
                    return x, y, weights
                else:
                    raise NotImplementedError(pydataset_type)

        normalizer = keras.layers.Normalization()
        normalizer.adapt(CustomDataset())
        self.assertTrue(normalizer.built)
        self.assertIsNotNone(normalizer.mean)
        self.assertIsNotNone(normalizer.variance)
        self.assertEqual(normalizer.mean.shape[-1], 3)
        self.assertEqual(normalizer.variance.shape[-1], 3)
        sample_input = np.random.rand(1, 32, 32, 3)
        output = normalizer(sample_input)
        self.assertEqual(output.shape, (1, 32, 32, 3))
