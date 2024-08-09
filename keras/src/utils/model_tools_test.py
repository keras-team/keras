import contextlib
import os
import tempfile
import unittest
from io import StringIO
from unittest.mock import patch

import numpy as np

from keras.src.layers import Conv2D
from keras.src.layers import Dense
from keras.src.layers import Dropout
from keras.src.layers import Flatten
from keras.src.layers import Input
from keras.src.models import Model
from keras.src.saving import load_model
from keras.src.utils import model_tools


class TestModelComparison(unittest.TestCase):
    """Tests for the model comparison functions.
    Contains four unit tests:
    - test_identical_models: Tests the comparison of two identical models.
    - test_structural_difference: Tests the comparison of two models with
    structural differences.
    - test_missing_layers: Tests the comparison of two models
    with missing layers.
    - test_significant_differences: Tests the comparison of two models
    with significant differences.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_model(self, path, units1, units2, extra_layer=False):
        input_layer = Input(shape=(20,), name="input")
        x = Dense(units1, activation="relu", name="dense_1")(input_layer)
        x = Dense(units2, activation="relu", name="dense_2")(x)
        if extra_layer:
            x = Dense(3, activation="relu", name="extra_layer")(x)
        output_layer = Dense(1, activation="sigmoid", name="output")(x)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.save(path)

    def create_complex_model(self, path, layers):
        input_layer = Input(shape=(64, 64, 3), name="input")
        x = input_layer
        for layer in layers:
            x = layer(x)
        output_layer = Dense(10, activation="softmax", name="output")(x)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.save(path)

    def test_identical_models(self):
        model1_path = os.path.join(self.temp_dir.name, "model1.keras")
        model2_path = os.path.join(self.temp_dir.name, "model2.keras")
        self.create_model(model1_path, 10, 5)
        self.create_model(model2_path, 10, 5)

        console_output = StringIO()
        with contextlib.redirect_stdout(console_output):
            model_tools.compare_models(model1_path, model2_path)

        output = console_output.getvalue()

        self.assertIn("Matching Layers", output)
        self.assertIn("input", output)
        self.assertIn("Identical layer:", output)
        self.assertIn("Weights: 0", output)
        self.assertIn("Sublayers: 0", output)
        self.assertIn("dense_1", output)
        self.assertIn("Identical layer:", output)
        self.assertIn("output", output)
        self.assertIn("Identical layer:", output)
        self.assertIn("dense_2", output)
        self.assertIn("Identical layer:", output)
        self.assertNotIn("Non matching Layers", output)

    def test_structural_difference(self):
        model1_path = os.path.join(self.temp_dir.name, "model1.keras")
        model2_path = os.path.join(self.temp_dir.name, "model2.keras")
        self.create_model(model1_path, 10, 5)
        self.create_model(model2_path, 12, 5)

        console_output = StringIO()
        with contextlib.redirect_stdout(console_output):
            model_tools.compare_models(model1_path, model2_path)

        output = console_output.getvalue()
        self.assertIn("Matching Layers", output)
        self.assertIn("input", output)
        self.assertIn("Identical layer:", output)
        self.assertIn("dense_1", output)
        self.assertIn("Shape:", output)
        self.assertIn("(20, 10)", output)
        self.assertIn("(20, 12)", output)
        self.assertIn("Dtype:", output)
        self.assertIn("float32", output)
        self.assertIn("float32", output)

    def test_missing_layers(self):
        model1_path = os.path.join(self.temp_dir.name, "model1.keras")
        model2_path = os.path.join(self.temp_dir.name, "model2.keras")
        self.create_model(model1_path, 10, 5)
        self.create_model(model2_path, 10, 5, extra_layer=True)

        console_output = StringIO()
        with contextlib.redirect_stdout(console_output):
            model_tools.compare_models(model1_path, model2_path)

        output = console_output.getvalue()
        self.assertIn("Non matching Layers", output)
        self.assertIn("extra_layer", output)

        self.assertIn("input", output)
        self.assertIn("Identical layer:", output)
        self.assertIn("Weights: 0", output)
        self.assertIn("Sublayers: 0", output)
        self.assertIn("dense_1", output)
        self.assertIn("Identical layer:", output)
        self.assertIn("Weights: 2", output)
        self.assertIn("Sublayers: 0", output)

    def test_significant_differences(self):
        model1_path = os.path.join(self.temp_dir.name, "model1.keras")
        model2_path = os.path.join(self.temp_dir.name, "model2.keras")

        model1_layers = [
            Conv2D(32, (3, 3), activation="relu", name="conv_1"),
            Conv2D(64, (3, 3), activation="relu", name="conv_2"),
            Flatten(name="flatten"),
            Dense(128, activation="relu", name="dense_1"),
        ]
        model2_layers = [
            Conv2D(32, (5, 5), activation="relu", name="conv_1"),
            Conv2D(128, (3, 3), activation="relu", name="conv_2"),
            Conv2D(64, (3, 3), activation="relu", name="conv_3"),
            Dropout(0.5, name="dropout"),
            Flatten(name="flatten"),
            Dense(256, activation="relu", name="dense_1"),
        ]

        self.create_complex_model(model1_path, model1_layers)
        self.create_complex_model(model2_path, model2_layers)

        console_output = StringIO()
        with contextlib.redirect_stdout(console_output):
            model_tools.compare_models(model1_path, model2_path)

        output = console_output.getvalue()
        self.assertIn("Non matching Layers", output)
        self.assertIn("conv_3", output)
        self.assertIn("dropout", output)

        self.assertIn("Matching Layers", output)
        self.assertIn("dense_1", output)
        self.assertIn("Shape:", output)
        self.assertIn("(230400, 128)", output)
        self.assertIn("(200704, 256)", output)
        self.assertIn("Dtype:", output)
        self.assertIn("float32", output)
        self.assertIn("float32", output)

        self.assertIn("conv_2", output)
        self.assertIn("Shape:", output)
        self.assertIn("(3, 3, 32, 64)", output)
        self.assertIn("(3, 3, 32, 128)", output)
        self.assertIn("Dtype:", output)
        self.assertIn("float32", output)
        self.assertIn("float32", output)

        self.assertIn("flatten", output)
        self.assertIn("Identical layer:", output)
        self.assertIn("Weights: 0", output)
        self.assertIn("Sublayers: 0", output)


class InspectModelsTest(unittest.TestCase):

    def setUp(self):
        self.temp_filepath = os.path.join(os.getcwd(), "my_model.keras")

    def tearDown(self):
        if os.path.exists(self.temp_filepath):
            os.remove(self.temp_filepath)

    def create_simple_model(self):
        inputs = Input(shape=(4,))
        outputs = Dense(1)(inputs)
        model = Model(inputs, outputs)
        model.save(self.temp_filepath)

    def test_inspect_keras_model_shell(self):
        with patch(
            "keras.src.utils.model_tools.is_notebook", return_value=False
        ):
            self.create_simple_model()

            console_output = StringIO()
            with contextlib.redirect_stdout(console_output):
                model_tools.inspect_file(self.temp_filepath)

            output = console_output.getvalue().replace("\n", "")
            self.assertIn("Keras model file", output)
            self.assertIn("Model: Functional", output)
            self.assertIn("Saved with Keras", output)
            self.assertIn("Weights file:", output)
            self.assertIn("dense", output)
            self.assertIn("(4, 1) float32", output)
            self.assertIn("(1,) float32", output)

    @patch("keras.src.utils.model_tools.is_notebook", return_value=True)
    @patch("keras.src.utils.model_tools.display")
    def test_inspect_keras_model_notebook(self, mock_display, mock_is_notebook):
        self.create_simple_model()

        model_tools.inspect_file(self.temp_filepath)
        display_args = mock_display.call_args[0][0].data
        display_args = display_args.replace("\n", "")
        self.assertIn("Keras model file", display_args)
        self.assertIn("Model: Functional", display_args)
        self.assertIn("Saved with Keras", display_args)
        self.assertIn("Weights file:", display_args)
        self.assertIn("dense", display_args)
        self.assertIn("0: (4, 1) float32", display_args)
        self.assertIn("1: (1,) float32", display_args)


class TestKerasFileEditor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.filepath = os.path.join(self.temp_dir.name, "test_model.keras")
        self.editor = None
        self.create_model()

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_model(self):
        inputs = Input(shape=(4,), name="input")
        x = Dense(3, activation="relu", name="dense_1")(inputs)
        outputs = Dense(1, activation="sigmoid", name="output")(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.save(self.filepath)
        self.editor = model_tools.KerasFileEditor(self.filepath)

    def test_initialization(self):
        self.assertIsNotNone(self.editor.model)
        self.assertEqual(self.editor.filepath, self.filepath)

    def test_list_layer_paths(self):
        with patch("builtins.print") as mocked_print:
            self.editor.list_layer_paths()
            mocked_print.assert_any_call("input")
            mocked_print.assert_any_call("dense_1")
            mocked_print.assert_any_call("output")

    def test_layer_info(self):
        with patch("builtins.print") as mocked_print:
            self.editor.layer_info("dense_1")
            mocked_print.assert_any_call("kernel", (4, 3))
            mocked_print.assert_any_call("bias", (3,))

    def test_edit_layer_name(self):
        new_name = "new_dense_1"
        self.editor.edit_layer("dense_1", new_name=new_name)
        self.assertEqual(
            self.editor.model.get_layer(name=new_name).name, new_name
        )

    def test_edit_layer_weights(self):
        layer_name = "dense_1"
        layer = self.editor.model.get_layer(name=layer_name)
        new_weights = [np.ones_like(w) for w in layer.get_weights()]
        self.editor.edit_layer(layer_name, new_vars=new_weights)

        updated_layer = self.editor.model.get_layer(name=layer_name)
        for original, updated in zip(new_weights, updated_layer.get_weights()):
            np.testing.assert_array_equal(original, updated)

    def test_edit_layer_mismatch_weights(self):
        layer_name = "dense_1"
        layer = self.editor.model.get_layer(name=layer_name)
        new_weights = [np.ones_like(layer.get_weights()[0])]

        with patch("builtins.print") as mocked_print:
            self.editor.edit_layer(layer_name, new_vars=new_weights)
            mocked_print.assert_called_with(
                "Number of new variables (1) does \
                        not match the number of weights in the layer \
                            (2)."
            )

    def test_write_out(self):
        new_filepath = os.path.join(self.temp_dir.name, "new_model.keras")
        self.editor.write_out(new_filepath)
        self.assertTrue(os.path.exists(new_filepath))

        loaded_model = load_model(new_filepath)
        self.assertEqual(
            len(loaded_model.layers), len(self.editor.model.layers)
        )
        for layer_orig, layer_loaded in zip(
            self.editor.model.layers, loaded_model.layers
        ):
            self.assertEqual(layer_orig.name, layer_loaded.name)
