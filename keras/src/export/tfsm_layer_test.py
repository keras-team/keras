import os

import numpy as np
import pytest
import tensorflow as tf

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
    def test_reloading_export_archive(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model()
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input)

        saved_model.export_saved_model(model, temp_filepath)
        reloaded_layer = tfsm_layer.TFSMLayer(temp_filepath)

        self.assertAllClose(
            reloaded_layer(ref_input),
            ref_output,
            atol=1e-7,
        )
        self.assertLen(reloaded_layer.weights, len(model.weights))
        self.assertLen(
            reloaded_layer.trainable_weights,
            len(model.trainable_weights),
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
            temp_filepath,
            call_endpoint="serving_default",
        )

        new_output = reloaded_layer(ref_input)
        self.assertAllClose(
            new_output[list(new_output.keys())[0]],
            ref_output,
            atol=1e-7,
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
            input_signature=[
                tf.TensorSpec(shape=(None, 10), dtype=tf.float32)
            ],
        )
        export_archive.add_endpoint(
            name="call_training",
            fn=lambda x: model(x, training=True),
            input_signature=[
                tf.TensorSpec(shape=(None, 10), dtype=tf.float32)
            ],
        )

        export_archive.write_out(temp_filepath)

        reloaded_layer = tfsm_layer.TFSMLayer(
            temp_filepath,
            call_endpoint="call_inference",
            call_training_endpoint="call_training",
        )

        inference_output = reloaded_layer(
            tf.random.normal((1, 10)),
            training=False,
        )
        training_output = reloaded_layer(
            tf.random.normal((1, 10)),
            training=True,
        )

        self.assertAllClose(
            np.mean(training_output),
            0.0,
            atol=1e-7,
        )
        self.assertNotAllClose(
            np.mean(inference_output),
            0.0,
            atol=1e-7,
        )

    def test_serialization(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model()

        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input)

        saved_model.export_saved_model(model, temp_filepath)
        reloaded_layer = tfsm_layer.TFSMLayer(temp_filepath)

        # Recreate via from_config (explicit opt-out)
        config = reloaded_layer.get_config()
        rereloaded_layer = tfsm_layer.TFSMLayer.from_config(
            config,
            safe_mode=False,
        )

        self.assertAllClose(
            rereloaded_layer(ref_input),
            ref_output,
            atol=1e-7,
        )

        # Test full model serialization
        model = models.Sequential([reloaded_layer])
        temp_model_filepath = os.path.join(
            self.get_temp_dir(),
            "model.keras",
        )
        model.save(temp_model_filepath)

        reloaded_model = saving_lib.load_model(
            temp_model_filepath,
            custom_objects={"TFSMLayer": tfsm_layer.TFSMLayer},
            safe_mode=False,
        )

        self.assertAllClose(
            reloaded_model(ref_input),
            ref_output,
            atol=1e-7,
        )

    def test_safe_mode_blocks_deserialization(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")

        model = models.Sequential(
            [
                layers.Input((3,)),
                layers.Lambda(lambda x: x * 2),
            ]
        )
        saved_model.export_saved_model(model, temp_filepath)

        layer = tfsm_layer.TFSMLayer(temp_filepath)
        config = layer.get_config()

        # Safe mode blocks implicit deserialization
        with self.assertRaisesRegex(
            ValueError,
            "arbitrary code execution",
        ):
            tfsm_layer.TFSMLayer.from_config(config)

        # Explicit opt-out allows deserialization
        layer = tfsm_layer.TFSMLayer.from_config(
            config,
            safe_mode=False,
        )
        x = tf.random.normal((2, 3))
        self.assertAllClose(layer(x), x * 2)

    def test_errors(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = models.Sequential(
            [
                layers.Input((2,)),
                layers.Dense(3),
            ]
        )
        saved_model.export_saved_model(model, temp_filepath)

        with self.assertRaisesRegex(ValueError, "The endpoint 'wrong'"):
            tfsm_layer.TFSMLayer(
                temp_filepath,
                call_endpoint="wrong",
            )

        with self.assertRaisesRegex(ValueError, "The endpoint 'wrong'"):
            tfsm_layer.TFSMLayer(
                temp_filepath,
                call_endpoint="serve",
                call_training_endpoint="wrong",
            )
