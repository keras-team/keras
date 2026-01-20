import os

import numpy as np
import pytest
import tensorflow as tf

import keras
from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src import utils
from keras.src.export import saved_model
from keras.src.export import tfsm_layer
from keras.src.export.saved_model_test import get_model
from keras.src.saving import saving_lib


@pytest.mark.skipif(
    backend.backend() != "tensorflow",
    reason="TFSM Layer reloading is only for the TF backend.",
)
class TestTFSMLayer(testing.TestCase):
    def setUp(self):
        super().setUp()
        # Enable unsafe deserialization for tests that load external SavedModels
        keras.config.enable_unsafe_deserialization()

    def test_reloading_export_archive(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model()
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input)

        saved_model.export_saved_model(model, temp_filepath)
        reloaded_layer = tfsm_layer.TFSMLayer(temp_filepath)
        self.assertAllClose(reloaded_layer(ref_input), ref_output, atol=1e-7)
        self.assertLen(reloaded_layer.weights, len(model.weights))
        self.assertLen(
            reloaded_layer.trainable_weights, len(model.trainable_weights)
        )
        self.assertLen(
            reloaded_layer.non_trainable_weights,
            len(model.non_trainable_weights),
        )

    def test_reloading_default_saved_model(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model()
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input)

        tf.saved_model.save(model, temp_filepath)
        reloaded_layer = tfsm_layer.TFSMLayer(
            temp_filepath, call_endpoint="serving_default"
        )
        # The output is a dict, due to the nature of SavedModel saving.
        new_output = reloaded_layer(ref_input)
        self.assertAllClose(
            new_output[list(new_output.keys())[0]],
            ref_output,
            atol=1e-7,
        )
        self.assertLen(reloaded_layer.weights, len(model.weights))
        self.assertLen(
            reloaded_layer.trainable_weights, len(model.trainable_weights)
        )
        self.assertLen(
            reloaded_layer.non_trainable_weights,
            len(model.non_trainable_weights),
        )
        for keras_var in reloaded_layer.weights:
            self.assertIsInstance(keras_var, backend.Variable)

    def test_call_training(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        utils.set_random_seed(1337)
        model = models.Sequential(
            [
                layers.Input((10,)),
                layers.Dense(10),
                layers.Dropout(0.99999),
            ]
        )
        export_archive = saved_model.ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            name="call_inference",
            fn=lambda x: model(x, training=False),
            input_signature=[tf.TensorSpec(shape=(None, 10), dtype=tf.float32)],
        )
        export_archive.add_endpoint(
            name="call_training",
            fn=lambda x: model(x, training=True),
            input_signature=[tf.TensorSpec(shape=(None, 10), dtype=tf.float32)],
        )
        export_archive.write_out(temp_filepath)
        reloaded_layer = tfsm_layer.TFSMLayer(
            temp_filepath,
            call_endpoint="call_inference",
            call_training_endpoint="call_training",
        )
        inference_output = reloaded_layer(
            tf.random.normal((1, 10)), training=False
        )
        training_output = reloaded_layer(
            tf.random.normal((1, 10)), training=True
        )
        self.assertAllClose(np.mean(training_output), 0.0, atol=1e-7)
        self.assertNotAllClose(np.mean(inference_output), 0.0, atol=1e-7)

    def test_serialization(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model()
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input)

        saved_model.export_saved_model(model, temp_filepath)
        reloaded_layer = tfsm_layer.TFSMLayer(temp_filepath)

        # Test reinstantiation from config
        config = reloaded_layer.get_config()
        rereloaded_layer = tfsm_layer.TFSMLayer.from_config(config)
        self.assertAllClose(rereloaded_layer(ref_input), ref_output, atol=1e-7)

        # Test whole model saving with reloaded layer inside
        model = models.Sequential([reloaded_layer])
        temp_model_filepath = os.path.join(self.get_temp_dir(), "m.keras")
        model.save(temp_model_filepath, save_format="keras_v3")
        reloaded_model = saving_lib.load_model(
            temp_model_filepath,
            custom_objects={"TFSMLayer": tfsm_layer.TFSMLayer},
        )
        self.assertAllClose(reloaded_model(ref_input), ref_output, atol=1e-7)

    def test_errors(self):
        # Test missing call endpoint
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = models.Sequential([layers.Input((2,)), layers.Dense(3)])
        saved_model.export_saved_model(model, temp_filepath)
        with self.assertRaisesRegex(ValueError, "The endpoint 'wrong'"):
            tfsm_layer.TFSMLayer(temp_filepath, call_endpoint="wrong")

        # Test missing call training endpoint
        with self.assertRaisesRegex(ValueError, "The endpoint 'wrong'"):
            tfsm_layer.TFSMLayer(
                temp_filepath,
                call_endpoint="serve",
                call_training_endpoint="wrong",
            )

    def test_safe_mode_direct_instantiation_allowed(self):
        """Test that direct TFSMLayer instantiation works.

        Direct instantiation works regardless of safe_mode.

        Direct instantiation is allowed because the user is explicitly creating
        the layer - the security concern is only during deserialization of
        untrusted .keras files.
        """
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model()
        model(tf.random.normal((1, 10)))
        saved_model.export_saved_model(model, temp_filepath)

        # Direct instantiation should work even without enabling unsafe mode
        # (setUp enables it, so we test the behavior is working)
        reloaded_layer = tfsm_layer.TFSMLayer(temp_filepath)
        self.assertIsNotNone(reloaded_layer)

    def test_safe_mode_from_config_blocked(self):
        """Test that from_config() raises ValueError when safe_mode=True.

        This is the core security fix: when deserializing an untrusted .keras
        model, from_config() is called and should block TFSMLayer loading
        to prevent attacker-controlled SavedModels from being loaded.
        """
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model()
        model(tf.random.normal((1, 10)))
        saved_model.export_saved_model(model, temp_filepath)

        # Create a TFSMLayer and get its config
        reloaded_layer = tfsm_layer.TFSMLayer(temp_filepath)
        config = reloaded_layer.get_config()

        # from_config with explicit safe_mode=True should raise
        with self.assertRaisesRegex(
            ValueError,
            "Loading a TFSMLayer from config is disallowed.*safe_mode=True",
        ):
            tfsm_layer.TFSMLayer.from_config(config, safe_mode=True)

    def test_safe_mode_from_config_allowed_when_disabled(self):
        """Test that from_config() works when safe_mode=False.

        When the user explicitly passes safe_mode=False or calls
        enable_unsafe_deserialization(), loading should succeed.
        """
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model()
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input)

        saved_model.export_saved_model(model, temp_filepath)
        reloaded_layer = tfsm_layer.TFSMLayer(temp_filepath)
        config = reloaded_layer.get_config()

        # from_config with explicit safe_mode=False should work
        rereloaded_layer = tfsm_layer.TFSMLayer.from_config(
            config, safe_mode=False
        )
        self.assertAllClose(rereloaded_layer(ref_input), ref_output, atol=1e-7)

    def test_safe_mode_model_loading_blocked(self):
        """Test that loading a .keras model with TFSMLayer is blocked.

        Loading is blocked when safe_mode=True.

        This tests the full attack scenario: an attacker creates a malicious
        .keras file containing a TFSMLayer pointing to a malicious SavedModel.
        When a victim loads this with safe_mode=True (the default), it should
        be blocked.
        """
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model()
        model(tf.random.normal((1, 10)))
        saved_model.export_saved_model(model, temp_filepath)

        # Create and save a model containing TFSMLayer
        reloaded_layer = tfsm_layer.TFSMLayer(temp_filepath)
        wrapper_model = models.Sequential([reloaded_layer])
        model_path = os.path.join(self.get_temp_dir(), "tfsm_model.keras")
        wrapper_model.save(model_path, save_format="keras_v3")

        # Loading with safe_mode=True should raise
        with self.assertRaisesRegex(
            ValueError,
            "Loading a TFSMLayer from config is disallowed.*safe_mode=True",
        ):
            saving_lib.load_model(
                model_path,
                custom_objects={"TFSMLayer": tfsm_layer.TFSMLayer},
                safe_mode=True,
            )
