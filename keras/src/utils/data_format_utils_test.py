import numpy as np
import pytest

import keras
from keras.src import backend
from keras.src import testing
from keras.src.utils.data_format_utils import _convert_axis
from keras.src.utils.data_format_utils import _permute_shape
from keras.src.utils.data_format_utils import convert_data_format
from keras.src.utils.data_format_utils import convert_to_channels_first
from keras.src.utils.data_format_utils import convert_to_channels_last


class PermuteShapeTest(testing.TestCase):
    def test_channels_last_to_first_2d(self):
        # Image shape: (H, W, C) -> (C, H, W)
        shape = (224, 224, 3)
        result = _permute_shape(shape, "channels_last", "channels_first")
        self.assertEqual(result, (3, 224, 224))

    def test_channels_first_to_last_2d(self):
        # Image shape: (C, H, W) -> (H, W, C)
        shape = (3, 224, 224)
        result = _permute_shape(shape, "channels_first", "channels_last")
        self.assertEqual(result, (224, 224, 3))

    def test_channels_last_to_first_3d(self):
        # Volume shape: (D, H, W, C) -> (C, D, H, W)
        shape = (32, 64, 64, 16)
        result = _permute_shape(shape, "channels_last", "channels_first")
        self.assertEqual(result, (16, 32, 64, 64))

    def test_channels_first_to_last_3d(self):
        # Volume shape: (C, D, H, W) -> (D, H, W, C)
        shape = (16, 32, 64, 64)
        result = _permute_shape(shape, "channels_first", "channels_last")
        self.assertEqual(result, (32, 64, 64, 16))

    def test_same_format_no_change(self):
        shape = (224, 224, 3)
        result = _permute_shape(shape, "channels_last", "channels_last")
        self.assertEqual(result, shape)

    def test_none_shape(self):
        result = _permute_shape(None, "channels_last", "channels_first")
        self.assertIsNone(result)

    def test_1d_shape_no_change(self):
        # 1D shapes (like Dense layer output) shouldn't change
        shape = (10,)
        result = _permute_shape(shape, "channels_last", "channels_first")
        self.assertEqual(result, (10,))


class ConvertAxisTest(testing.TestCase):
    def test_channels_last_to_first_4d(self):
        # axis=-1 (channel axis in channels_last) -> axis=1 in channels_first
        result = _convert_axis(-1, 4, "channels_last", "channels_first")
        self.assertEqual(result, 1)

    def test_channels_first_to_last_4d(self):
        # axis=1 (channel axis in channels_first) -> axis=-1 in channels_last
        result = _convert_axis(1, 4, "channels_first", "channels_last")
        self.assertEqual(result, -1)

    def test_same_format_no_change(self):
        result = _convert_axis(-1, 4, "channels_last", "channels_last")
        self.assertEqual(result, -1)


class ConvertDataFormatSequentialTest(testing.TestCase):
    def test_sequential_conv2d_channels_last_to_first(self):
        # Create a simple channels_last model
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(32, 32, 3)),
                keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
                keras.layers.MaxPooling2D(2),
                keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
                keras.layers.GlobalAveragePooling2D(),
                keras.layers.Dense(10),
            ]
        )

        # Convert to channels_first
        converted = convert_to_channels_first(model)

        # Check input shape
        self.assertEqual(converted.input_shape, (None, 3, 32, 32))

        # Check that conv layers have channels_first
        for layer in converted.layers:
            if hasattr(layer, "data_format"):
                self.assertEqual(layer.data_format, "channels_first")

        # Verify weights are preserved
        for old_layer, new_layer in zip(model.layers, converted.layers):
            if old_layer.weights:
                for old_w, new_w in zip(
                    old_layer.get_weights(), new_layer.get_weights()
                ):
                    np.testing.assert_array_equal(old_w, new_w)

    def test_sequential_conv2d_channels_first_to_last(self):
        # Create a channels_first model
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(3, 32, 32)),
                keras.layers.Conv2D(
                    16, 3, padding="same", data_format="channels_first"
                ),
                keras.layers.MaxPooling2D(2, data_format="channels_first"),
                keras.layers.GlobalAveragePooling2D(
                    data_format="channels_first"
                ),
                keras.layers.Dense(10),
            ]
        )

        # Convert to channels_last
        converted = convert_to_channels_last(model)

        # Check input shape
        self.assertEqual(converted.input_shape, (None, 32, 32, 3))

        # Check that conv layers have channels_last
        for layer in converted.layers:
            if hasattr(layer, "data_format"):
                self.assertEqual(layer.data_format, "channels_last")

    def test_with_batch_normalization(self):
        # Create a model with batch normalization
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(32, 32, 3)),
                keras.layers.Conv2D(16, 3, padding="same"),
                keras.layers.BatchNormalization(),
                keras.layers.GlobalAveragePooling2D(),
                keras.layers.Dense(10),
            ]
        )

        # Convert to channels_first
        converted = convert_to_channels_first(model)

        # Check batch norm axis
        for layer in converted.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                self.assertEqual(layer.axis, 1)

    def test_roundtrip_conversion(self):
        # Create a channels_last model
        original = keras.Sequential(
            [
                keras.layers.Input(shape=(28, 28, 1)),
                keras.layers.Conv2D(8, 3, padding="same"),
                keras.layers.MaxPooling2D(2),
                keras.layers.Flatten(),
                keras.layers.Dense(10),
            ]
        )

        # Set some weights
        original.build((None, 28, 28, 1))

        # Convert to channels_first and back
        channels_first = convert_to_channels_first(original)
        roundtrip = convert_to_channels_last(channels_first)

        # Check that input shape is restored
        self.assertEqual(roundtrip.input_shape, original.input_shape)

        # Check that weights are preserved through roundtrip
        for orig_layer, rt_layer in zip(original.layers, roundtrip.layers):
            if orig_layer.weights:
                for orig_w, rt_w in zip(
                    orig_layer.get_weights(), rt_layer.get_weights()
                ):
                    np.testing.assert_array_equal(orig_w, rt_w)


class ConvertDataFormatFunctionalTest(testing.TestCase):
    def test_functional_model_conversion(self):
        # Create a functional model
        inputs = keras.layers.Input(shape=(64, 64, 3))
        x = keras.layers.Conv2D(32, 3, padding="same")(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(2)(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(5)(x)
        model = keras.Model(inputs, outputs)

        # Convert to channels_first
        converted = convert_to_channels_first(model)

        # Verify input shape changed
        self.assertEqual(converted.input_shape, (None, 3, 64, 64))

        # Verify data_format changed
        for layer in converted.layers:
            if hasattr(layer, "data_format"):
                self.assertEqual(layer.data_format, "channels_first")


class ConvertDataFormatOutputTest(testing.TestCase):
    @pytest.mark.skipif(
        backend.backend() == "tensorflow",
        reason="TensorFlow CPU does not support channels_first (NCHW) format",
    )
    def test_output_equivalence(self):
        # Create a simple model
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(16, 16, 3)),
                keras.layers.Conv2D(8, 3, padding="same", activation="relu"),
                keras.layers.GlobalAveragePooling2D(),
                keras.layers.Dense(4),
            ]
        )

        # Create test input
        x_channels_last = np.random.randn(2, 16, 16, 3).astype("float32")
        x_channels_first = np.transpose(x_channels_last, (0, 3, 1, 2))

        # Get output from original model
        output_original = model.predict(x_channels_last, verbose=0)

        # Convert model and get output
        converted = convert_to_channels_first(model)
        output_converted = converted.predict(x_channels_first, verbose=0)

        # Outputs should be the same (up to numerical precision)
        np.testing.assert_allclose(
            output_original, output_converted, rtol=1e-5, atol=1e-5
        )


class ConvertDataFormatEdgeCasesTest(testing.TestCase):
    def test_invalid_target_format(self):
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(32, 32, 3)),
                keras.layers.Conv2D(16, 3),
            ]
        )

        with self.assertRaisesRegex(ValueError, "Invalid target_data_format"):
            convert_data_format(model, "invalid_format")

    def test_invalid_source_format(self):
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(32, 32, 3)),
                keras.layers.Conv2D(16, 3),
            ]
        )

        with self.assertRaisesRegex(ValueError, "Invalid source_data_format"):
            convert_data_format(
                model, "channels_first", source_data_format="invalid"
            )

    def test_cannot_infer_source_format(self):
        # Model with no data_format layers
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(10,)),
                keras.layers.Dense(5),
            ]
        )

        with self.assertRaisesRegex(ValueError, "Could not infer"):
            convert_data_format(model, "channels_first")

    def test_same_format_clones_model(self):
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(32, 32, 3)),
                keras.layers.Conv2D(16, 3),
            ]
        )

        # Converting to same format should just clone
        converted = convert_data_format(model, "channels_last")

        # Should be a different model instance
        self.assertIsNot(converted, model)

        # But with same structure
        self.assertEqual(converted.input_shape, model.input_shape)


class ConvertDataFormat1DTest(testing.TestCase):
    def test_conv1d_conversion(self):
        # Create a 1D conv model (for sequences)
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(100, 16)),  # (timesteps, features)
                keras.layers.Conv1D(32, 3, padding="same"),
                keras.layers.MaxPooling1D(2),
                keras.layers.GlobalAveragePooling1D(),
                keras.layers.Dense(10),
            ]
        )

        # Convert to channels_first
        converted = convert_to_channels_first(model)

        # Check input shape: (timesteps, features) -> (features, timesteps)
        self.assertEqual(converted.input_shape, (None, 16, 100))

        # Check conv layers have channels_first
        for layer in converted.layers:
            if hasattr(layer, "data_format"):
                self.assertEqual(layer.data_format, "channels_first")


class ConvertDataFormat3DTest(testing.TestCase):
    def test_conv3d_conversion(self):
        # Create a 3D conv model (for volumes)
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(16, 16, 16, 3)),  # (D, H, W, C)
                keras.layers.Conv3D(8, 3, padding="same"),
                keras.layers.MaxPooling3D(2),
                keras.layers.GlobalAveragePooling3D(),
                keras.layers.Dense(5),
            ]
        )

        # Convert to channels_first
        converted = convert_to_channels_first(model)

        # Check input shape: (D, H, W, C) -> (C, D, H, W)
        self.assertEqual(converted.input_shape, (None, 3, 16, 16, 16))

        # Check conv layers have channels_first
        for layer in converted.layers:
            if hasattr(layer, "data_format"):
                self.assertEqual(layer.data_format, "channels_first")
