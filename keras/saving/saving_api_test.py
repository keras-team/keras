import os
import unittest.mock as mock

import numpy as np
from absl import logging

from keras import layers
from keras.models import Sequential
from keras.saving import saving_api
from keras.testing import test_case


class SaveModelTests(test_case.TestCase):
    def get_model(self):
        return Sequential(
            [
                layers.Dense(5, input_shape=(3,)),
                layers.Softmax(),
            ]
        )

    def test_basic_saving(self):
        """Test basic model saving and loading."""
        model = self.get_model()
        filepath = os.path.join(self.get_temp_dir(), "test_model.keras")
        saving_api.save_model(model, filepath)

        loaded_model = saving_api.load_model(filepath)
        x = np.random.uniform(size=(10, 3))
        self.assertTrue(np.allclose(model.predict(x), loaded_model.predict(x)))

    def test_invalid_save_format(self):
        """Test deprecated save_format argument."""
        model = self.get_model()
        with self.assertRaisesRegex(
            ValueError, "The `save_format` argument is deprecated"
        ):
            saving_api.save_model(model, "model.txt", save_format=True)

    def test_unsupported_arguments(self):
        """Test unsupported argument during model save."""
        model = self.get_model()
        filepath = os.path.join(self.get_temp_dir(), "test_model.keras")
        with self.assertRaisesRegex(
            ValueError, r"The following argument\(s\) are not supported"
        ):
            saving_api.save_model(model, filepath, random_arg=True)

    def test_save_h5_format(self):
        """Test saving model in h5 format."""
        model = self.get_model()
        filepath_h5 = os.path.join(self.get_temp_dir(), "test_model.h5")
        saving_api.save_model(model, filepath_h5)
        self.assertTrue(os.path.exists(filepath_h5))
        os.remove(filepath_h5)

    def test_save_unsupported_extension(self):
        """Test saving model with unsupported extension."""
        model = self.get_model()
        with self.assertRaisesRegex(
            ValueError, "Invalid filepath extension for saving"
        ):
            saving_api.save_model(model, "model.png")


class LoadModelTests(test_case.TestCase):
    def get_model(self):
        return Sequential(
            [
                layers.Dense(5, input_shape=(3,)),
                layers.Softmax(),
            ]
        )

    def test_basic_load(self):
        """Test basic model loading."""
        model = self.get_model()
        filepath = os.path.join(self.get_temp_dir(), "test_model.keras")
        saving_api.save_model(model, filepath)

        loaded_model = saving_api.load_model(filepath)
        x = np.random.uniform(size=(10, 3))
        self.assertTrue(np.allclose(model.predict(x), loaded_model.predict(x)))

    def test_load_unsupported_format(self):
        """Test loading model with unsupported format."""
        with self.assertRaisesRegex(ValueError, "File format not supported"):
            saving_api.load_model("model.pkl")

    def test_load_keras_not_zip(self):
        """Test loading keras file that's not a zip."""
        with self.assertRaisesRegex(ValueError, "File not found"):
            saving_api.load_model("not_a_zip.keras")

    def test_load_h5_format(self):
        """Test loading model in h5 format."""
        model = self.get_model()
        filepath_h5 = os.path.join(self.get_temp_dir(), "test_model.h5")
        saving_api.save_model(model, filepath_h5)
        loaded_model = saving_api.load_model(filepath_h5)
        x = np.random.uniform(size=(10, 3))
        self.assertTrue(np.allclose(model.predict(x), loaded_model.predict(x)))
        os.remove(filepath_h5)

    def test_load_model_with_custom_objects(self):
        """Test loading model with custom objects."""

        class CustomLayer(layers.Layer):
            def call(self, inputs):
                return inputs

        model = Sequential([CustomLayer(input_shape=(3,))])
        filepath = os.path.join(self.get_temp_dir(), "custom_model.keras")
        model.save(filepath)
        loaded_model = saving_api.load_model(
            filepath, custom_objects={"CustomLayer": CustomLayer}
        )
        self.assertIsInstance(loaded_model.layers[0], CustomLayer)
        os.remove(filepath)


class LoadWeightsTests(test_case.TestCase):
    def get_model(self):
        return Sequential(
            [
                layers.Dense(5, input_shape=(3,)),
                layers.Softmax(),
            ]
        )

    def test_load_keras_weights(self):
        """Test loading keras weights."""
        model = self.get_model()
        filepath = os.path.join(self.get_temp_dir(), "test_weights.weights.h5")
        model.save_weights(filepath)
        original_weights = model.get_weights()
        model.load_weights(filepath)
        loaded_weights = model.get_weights()
        for orig, loaded in zip(original_weights, loaded_weights):
            self.assertTrue(np.array_equal(orig, loaded))

    def test_load_h5_weights_by_name(self):
        """Test loading h5 weights by name."""
        model = self.get_model()
        filepath = os.path.join(self.get_temp_dir(), "test_weights.weights.h5")
        model.save_weights(filepath)
        with self.assertRaisesRegex(ValueError, "Invalid keyword arguments"):
            model.load_weights(filepath, by_name=True)

    def test_load_weights_invalid_extension(self):
        """Test loading weights with unsupported extension."""
        model = self.get_model()
        with self.assertRaisesRegex(ValueError, "File format not supported"):
            model.load_weights("invalid_extension.pkl")


class SaveModelTestsWarning(test_case.TestCase):
    def get_model(self):
        return Sequential(
            [
                layers.Dense(5, input_shape=(3,)),
                layers.Softmax(),
            ]
        )

    def test_h5_deprecation_warning(self):
        """Test deprecation warning for h5 format."""
        model = self.get_model()
        filepath = os.path.join(self.get_temp_dir(), "test_model.h5")

        with mock.patch.object(logging, "warning") as mock_warn:
            saving_api.save_model(model, filepath)
            mock_warn.assert_called_once_with(
                "You are saving your model as an HDF5 file via "
                "`model.save()` or `keras.saving.save_model(model)`. "
                "This file format is considered legacy. "
                "We recommend using instead the native Keras format, "
                "e.g. `model.save('my_model.keras')` or "
                "`keras.saving.save_model(model, 'my_model.keras')`. "
            )
