import os

import numpy as np

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

        out = editor.compare_to(model)  # Succeeds
        self.assertEqual(out["status"], "success")
        out = editor.compare_to(target_model)  # Fails

        editor.add_object(
            "layers/dense_3", weights={"0": np.random.random((3, 3))}
        )
        out = editor.compare_to(target_model)  # Fails
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error_count"], 2)

        editor.rename_object("dense_3", "dense_4")
        editor.rename_object("layers/dense_4", "dense_2")
        editor.add_weights("dense_2", weights={"1": np.random.random((3,))})
        out = editor.compare_to(target_model)  # Succeeds
        self.assertEqual(out["status"], "success")

        editor.add_object(
            "layers/dense_3", weights={"0": np.random.random((3, 3))}
        )
        out = editor.compare_to(target_model)  # Fails
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error_count"], 1)

        editor.delete_object("layers/dense_3")
        out = editor.compare_to(target_model)  # Succeeds
        self.assertEqual(out["status"], "success")
        editor.summary()

        temp_filepath = os.path.join(self.get_temp_dir(), "resaved.weights.h5")
        editor.resave_weights(temp_filepath)
        target_model.load_weights(temp_filepath)

        editor = KerasFileEditor(temp_filepath)
        editor.summary()
        out = editor.compare_to(target_model)  # Succeeds
        self.assertEqual(out["status"], "success")
