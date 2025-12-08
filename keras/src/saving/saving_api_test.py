import os
import pathlib
import unittest.mock as mock

import numpy as np
from absl import logging
from absl.testing import parameterized

from keras.src import layers
from keras.src.legacy.saving.legacy_h5_format import save_model_to_hdf5
from keras.src.models import Sequential
from keras.src.saving import saving_api
from keras.src.testing import test_case
from keras.src.testing.test_utils import named_product


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

        # Verify the warning.
        with mock.patch.object(logging, "warning") as mock_warn:
            saving_api.save_model(model, filepath_h5)
            mock_warn.assert_called_once_with(
                "You are saving your model as an HDF5 file via "
                "`model.save()` or `keras.saving.save_model(model)`. "
                "This file format is considered legacy. "
                "We recommend using instead the native Keras format, "
                "e.g. `model.save('my_model.keras')` or "
                "`keras.saving.save_model(model, 'my_model.keras')`. "
            )
        self.assertTrue(os.path.exists(filepath_h5))
        os.remove(filepath_h5)

    def test_save_unsupported_extension(self):
        """Test saving model with unsupported extension."""
        model = self.get_model()
        with self.assertRaisesRegex(
            ValueError, "Invalid filepath extension for saving"
        ):
            saving_api.save_model(model, "model.png")

    def test_objects_to_skip(self):
        model = Sequential(
            [
                layers.Input((3,)),
                layers.Dense(5),
                layers.Dense(5),
            ]
        )
        skip = model.layers[0]
        filepath = os.path.join(self.get_temp_dir(), "test_model.weights.h5")
        saving_api.save_weights(model, filepath, objects_to_skip=[skip])
        new_model = Sequential(
            [
                layers.Input((3,)),
                layers.Dense(5),
                layers.Dense(5),
            ]
        )
        new_model.load_weights(filepath, objects_to_skip=[new_model.layers[0]])
        self.assertNotAllClose(
            new_model.layers[0].get_weights()[0],
            model.layers[0].get_weights()[0],
        )
        self.assertAllClose(
            new_model.layers[0].get_weights()[1],
            model.layers[0].get_weights()[1],
        )
        saving_api.save_weights(model, filepath)
        new_model.load_weights(filepath, objects_to_skip=[new_model.layers[0]])
        self.assertNotAllClose(
            new_model.layers[0].get_weights()[0],
            model.layers[0].get_weights()[0],
        )
        self.assertAllClose(
            new_model.layers[0].get_weights()[1],
            model.layers[0].get_weights()[1],
        )


class LoadModelTests(test_case.TestCase):
    def get_model(self, dtype=None):
        return Sequential(
            [
                layers.Dense(5, input_shape=(3,), dtype=dtype),
                layers.Softmax(),
            ]
        )

    @parameterized.named_parameters(
        [
            {"testcase_name": "bfloat16", "dtype": "bfloat16"},
            {"testcase_name": "float16", "dtype": "float16"},
            {"testcase_name": "float32", "dtype": "float32"},
            {"testcase_name": "float64", "dtype": "float64"},
        ]
    )
    def test_basic_load(self, dtype):
        """Test basic model loading."""
        model = self.get_model(dtype)
        filepath = os.path.join(self.get_temp_dir(), "test_model.keras")
        saving_api.save_model(model, filepath)

        loaded_model = saving_api.load_model(filepath)
        x = np.random.uniform(size=(10, 3))
        self.assertEqual(loaded_model.weights[0].dtype, dtype)
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

    def test_save_unzipped(self):
        """Test saving/loading an unzipped model dir."""
        model = self.get_model()

        # Test error with keras extension
        bad_filepath = os.path.join(self.get_temp_dir(), "test_model.keras")
        with self.assertRaisesRegex(ValueError, "should not end in"):
            saving_api.save_model(model, bad_filepath, zipped=False)

        filepath = os.path.join(self.get_temp_dir(), "test_model_dir")
        saving_api.save_model(model, filepath, zipped=False)

        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(os.path.isdir(filepath))
        config_filepath = os.path.join(filepath, "config.json")
        weights_filepath = os.path.join(filepath, "model.weights.h5")
        self.assertTrue(os.path.exists(config_filepath))
        self.assertTrue(os.path.exists(weights_filepath))

        loaded_model = saving_api.load_model(filepath)
        x = np.random.uniform(size=(10, 3))
        self.assertTrue(np.allclose(model.predict(x), loaded_model.predict(x)))


class LoadWeightsTests(test_case.TestCase):
    def get_model(self, dtype=None):
        return Sequential(
            [
                layers.Dense(5, input_shape=(3,), dtype=dtype),
                layers.Softmax(),
            ]
        )

    @parameterized.named_parameters(
        named_product(
            save_format=["keras", "weights.h5", "h5"],
            source_dtype=["float64", "float32", "float16", "bfloat16"],
            dest_dtype=["float64", "float32", "float16", "bfloat16"],
        )
    )
    def test_load_weights(self, save_format, source_dtype, dest_dtype):
        """Test loading keras weights."""
        src_model = self.get_model(dtype=source_dtype)
        if save_format == "keras":
            filepath = os.path.join(self.get_temp_dir(), "test_weights.keras")
            src_model.save(filepath)
        elif save_format == "weights.h5":
            filepath = os.path.join(
                self.get_temp_dir(), "test_weights.weights.h5"
            )
            src_model.save_weights(filepath)
        elif save_format == "h5":
            if "bfloat16" in (source_dtype, dest_dtype):
                raise self.skipTest(
                    "bfloat16 dtype is not supported in legacy h5 format."
                )
            filepath = os.path.join(self.get_temp_dir(), "test_weights.h5")
            save_model_to_hdf5(src_model, filepath)
        else:
            raise ValueError(f"Unsupported save format: {save_format}")

        dest_model = self.get_model(dtype=dest_dtype)
        dest_model.load_weights(filepath)

        src_weights = src_model.get_weights()
        dest_weights = dest_model.get_weights()
        for orig, loaded in zip(src_weights, dest_weights):
            self.assertAllClose(
                orig.astype("float32"),
                loaded.astype("float32"),
                atol=0.001,
                rtol=0.01,
            )

    def test_load_weights_invalid_kwargs(self):
        src_model = self.get_model()
        keras_filepath = os.path.join(self.get_temp_dir(), "test_weights.keras")
        weight_h5_filepath = os.path.join(
            self.get_temp_dir(), "test_weights.weights.h5"
        )
        legacy_h5_filepath = os.path.join(
            self.get_temp_dir(), "test_weights.h5"
        )
        src_model.save(keras_filepath)
        src_model.save_weights(weight_h5_filepath)
        save_model_to_hdf5(src_model, legacy_h5_filepath)

        dest_model = self.get_model()
        # Test keras file.
        with self.assertRaisesRegex(
            ValueError, r"only supports loading '.weights.h5' files."
        ):
            dest_model.load_weights(keras_filepath, objects_to_skip=[])
        with self.assertRaisesRegex(
            ValueError, r"only supports loading legacy '.h5' or '.hdf5' files."
        ):
            dest_model.load_weights(keras_filepath, by_name=True)
        with self.assertRaisesRegex(ValueError, r"Invalid keyword arguments"):
            dest_model.load_weights(keras_filepath, bad_kwarg=None)
        # Test weights.h5 file.
        with self.assertRaisesRegex(
            ValueError, r"only supports loading legacy '.h5' or '.hdf5' files."
        ):
            dest_model.load_weights(weight_h5_filepath, by_name=True)
        # Test h5 file.
        with self.assertRaisesRegex(
            ValueError, r"only supports loading '.weights.h5' files."
        ):
            dest_model.load_weights(legacy_h5_filepath, objects_to_skip=[])

    def test_load_weights_invalid_extension(self):
        """Test loading weights with unsupported extension."""
        model = self.get_model()
        with self.assertRaisesRegex(ValueError, "File format not supported"):
            model.load_weights("invalid_extension.pkl")

    def test_load_sharded_weights(self):
        src_model = self.get_model()
        temp_filepath = pathlib.Path(
            os.path.join(self.get_temp_dir(), "test_weights.weights.json")
        )
        src_model.save_weights(temp_filepath, max_shard_size=1)
        self.assertLen(os.listdir(temp_filepath.parent), 2)
        src_weights = src_model.get_weights()
        dest_model = self.get_model()
        dest_model.load_weights(temp_filepath)
        dest_weights = dest_model.get_weights()
        for orig, loaded in zip(src_weights, dest_weights):
            self.assertAllClose(orig, loaded)
