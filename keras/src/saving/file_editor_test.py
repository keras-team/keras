import os

import numpy as np
import pytest

import keras
from keras.src import testing
from keras.src.saving.file_editor import KerasFileEditor


def get_source_model():
    inputs = keras.Input((2,))
    x = keras.layers.Dense(3, name="mydense")(inputs)
    outputs = keras.layers.Dense(3, name="output_layer")(x)
    model = keras.Model(inputs, outputs)
    return model


def get_target_model():
    inputs = keras.Input((2,))
    x = keras.layers.Dense(3, name="mydense")(inputs)
    x = keras.layers.Dense(3, name="myotherdense")(x)
    outputs = keras.layers.Dense(3, name="output_layer")(x)
    model = keras.Model(inputs, outputs)
    return model


class SavingTest(testing.TestCase):
    def test_basics(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "my_model.keras")

        model = get_source_model()
        model.save(temp_filepath)

        editor = KerasFileEditor(temp_filepath)
        editor.summary()

        target_model = get_target_model()

        out = editor.compare(model)  # Succeeds
        self.assertEqual(out["status"], "success")
        out = editor.compare(target_model)  # Fails

        editor.add_object(
            "layers/dense_3", weights={"0": np.random.random((3, 3))}
        )
        out = editor.compare(target_model)  # Fails
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error_count"], 2)

        editor.rename_object("dense_3", "dense_4")
        editor.rename_object("layers/dense_4", "dense_2")
        editor.add_weights("dense_2", weights={"1": np.random.random((3,))})
        out = editor.compare(target_model)  # Succeeds
        self.assertEqual(out["status"], "success")

        editor.add_object(
            "layers/dense_3", weights={"0": np.random.random((3, 3))}
        )
        out = editor.compare(target_model)  # Fails
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error_count"], 1)

        editor.delete_object("layers/dense_3")
        out = editor.compare(target_model)  # Succeeds
        self.assertEqual(out["status"], "success")
        editor.summary()

        temp_filepath = os.path.join(self.get_temp_dir(), "resaved.weights.h5")
        editor.save(temp_filepath)
        target_model.load_weights(temp_filepath)

        editor = KerasFileEditor(temp_filepath)
        editor.summary()
        out = editor.compare(target_model)  # Succeeds
        self.assertEqual(out["status"], "success")

        editor.delete_weight("dense_2", "1")
        out = editor.compare(target_model)  # Fails
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error_count"], 1)

        editor.add_weights("dense_2", {"1": np.zeros((7,))})
        out = editor.compare(target_model)  # Fails
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error_count"], 1)

        editor.delete_weight("dense_2", "1")
        editor.add_weights("dense_2", {"1": np.zeros((3,))})
        out = editor.compare(target_model)  # Succeeds
        self.assertEqual(out["status"], "success")

    @pytest.mark.requires_trainable_backend
    def test_scalar_weight(self):
        model = keras.Sequential(name="my_sequential")
        model.add(keras.Input(shape=(1,), name="my_input"))
        model.add(keras.layers.Dense(1, activation="sigmoid", name="my_dense"))
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        model.fit(np.array([[1]]), np.array([[1]]), verbose=0)
        model_fpath = os.path.join(self.get_temp_dir(), "model.keras")
        weights_fpath = os.path.join(self.get_temp_dir(), "model.weights.h5")
        model.save(model_fpath)
        model.save_weights(weights_fpath)

        model_editor = KerasFileEditor(model_fpath)
        self.assertEqual(
            len(keras.src.tree.flatten(model_editor.weights_dict)), 8
        )
        model_weights_editor = KerasFileEditor(weights_fpath)
        self.assertEqual(
            len(keras.src.tree.flatten(model_weights_editor.weights_dict)), 8
        )
