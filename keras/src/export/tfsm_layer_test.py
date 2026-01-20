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
from keras.src.saving import serialization_lib


@pytest.mark.skipif(
    backend.backend() != "tensorflow",
    reason="TFSM Layer reloading is only for the TF backend.",
)
class TestTFSMLayer(testing.TestCase):
    def setUp(self):
        super().setUp()
        self._prev_safe_mode = serialization_lib.in_safe_mode()

    def tearDown(self):
        super().tearDown()
        if self._prev_safe_mode:
            keras.config.disable_unsafe_deserialization()
        else:
            keras.config.enable_unsafe_deserialization()

    def test_reloading_export_archive(self):
        keras.config.enable_unsafe_deserialization()

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model()
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input)

        saved_model.export_saved_model(model, temp_filepath)
        reloaded_layer = tfsm_layer.TFSMLayer(temp_filepath)

        self.assertAllClose(reloaded_layer(ref_input), ref_output, atol=1e-7)

    def test_reloading_default_saved_model(self):
        keras.config.enable_unsafe_deserialization()

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model()
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input)

        tf.saved_model.save(model, temp_filepath)
        reloaded_layer = tfsm_layer.TFSMLayer(
            temp_filepath, call_endpoint="serving_default"
        )
        new_output = reloaded_layer(ref_input)

        self.assertAllClose(
            new_output[list(new_output.keys())[0]],
            ref_output,
            atol=1e-7,
        )

    def test_call_training(self):
        keras.config.enable_unsafe_deserialization()

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
        keras.config.enable_unsafe_deserialization()

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model()
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input)

        saved_model.export_saved_model(model, temp_filepath)
        reloaded_layer = tfsm_layer.TFSMLayer(temp_filepath)

        config = reloaded_layer.get_config()
        rereloaded_layer = tfsm_layer.TFSMLayer.from_config(
            config, safe_mode=False
        )

        self.assertAllClose(rereloaded_layer(ref_input), ref_output, atol=1e-7)

        model = models.Sequential([reloaded_layer])
        temp_model_filepath = os.path.join(self.get_temp_dir(), "m.keras")
        model.save(temp_model_filepath, save_format="keras_v3")

        reloaded_model = saving_lib.load_model(
            temp_model_filepath,
            custom_objects={"TFSMLayer": tfsm_layer.TFSMLayer},
            safe_mode=False,
        )

        self.assertAllClose(reloaded_model(ref_input), ref_output, atol=1e-7)

    def test_safe_mode_direct_instantiation_allowed(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model()
        model(tf.random.normal((1, 10)))
        saved_model.export_saved_model(model, temp_filepath)

        reloaded_layer = tfsm_layer.TFSMLayer(temp_filepath)
        self.assertIsNotNone(reloaded_layer)

    def test_safe_mode_from_config_blocked(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model()
        model(tf.random.normal((1, 10)))
        saved_model.export_saved_model(model, temp_filepath)

        reloaded_layer = tfsm_layer.TFSMLayer(temp_filepath)
        config = reloaded_layer.get_config()

        with self.assertRaisesRegex(
            ValueError,
            (
                "Loading a TFSMLayer from config is disallowed"
                ".*safe_mode=True"
            ),
        ):
            tfsm_layer.TFSMLayer.from_config(config, safe_mode=True)

    def test_safe_mode_from_config_allowed_when_disabled(self):
        keras.config.enable_unsafe_deserialization()

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model()
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input)

        saved_model.export_saved_model(model, temp_filepath)
        reloaded_layer = tfsm_layer.TFSMLayer(temp_filepath)
        config = reloaded_layer.get_config()

        rereloaded_layer = tfsm_layer.TFSMLayer.from_config(
            config, safe_mode=False
        )

        self.assertAllClose(rereloaded_layer(ref_input), ref_output, atol=1e-7)

    def test_safe_mode_model_loading_blocked(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model()
        model(tf.random.normal((1, 10)))
        saved_model.export_saved_model(model, temp_filepath)

        reloaded_layer = tfsm_layer.TFSMLayer(temp_filepath)
        wrapper_model = models.Sequential([reloaded_layer])

        model_path = os.path.join(self.get_temp_dir(), "tfsm_model.keras")
        wrapper_model.save(model_path, save_format="keras_v3")

        with self.assertRaisesRegex(
            ValueError,
            (
                "Loading a TFSMLayer from config is disallowed"
                ".*safe_mode=True"
            ),
        ):
            saving_lib.load_model(
                model_path,
                custom_objects={"TFSMLayer": tfsm_layer.TFSMLayer},
                safe_mode=True,
            )
