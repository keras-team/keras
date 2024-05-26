import unittest
import os

from unittest.mock import patch

from keras.src.optimizers import Adam
from keras.src.layers import Dense, Input
from keras.src.models import Model

from keras.src.saving import load_model
from keras.src.saving.saving_lib import save_model
from keras.src.utils import model_tools


class DiffModelsTest(unittest.TestCase):

    def setUp(self):
        ref_name = 'reference_model.keras'
        diff_name = 'diff_model.keras'
        self.reference_model_path = os.path.join(os.getcwd(), ref_name)
        self.diff_model_path = os.path.join(os.getcwd(), diff_name)

    def tearDown(self):
        if os.path.exists(self.reference_model_path):
            os.remove(self.reference_model_path)
        if os.path.exists(self.diff_model_path):
            os.remove(self.diff_model_path)

    def create_reference_model(self, units, name):
        input_layer = Input(shape=(20,), name='input')
        x = Dense(units, activation='relu', name='dense_1')(input_layer)
        x = Dense(5, activation='relu', name='dense_2')(x)
        output_layer = Dense(1, activation='sigmoid', name='output')(x)
        model = Model(inputs=input_layer, outputs=output_layer, name=name)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    @patch('keras.src.utils.model_tools.display')
    def test_equal_models(self, mock_display):
        model1 = self.create_reference_model(10, 'reference_model')
        save_model(model1, self.reference_model_path)
        model2 = self.create_reference_model(10, 'reference_model')
        save_model(model2, self.diff_model_path)

        model_tools.compare_models(self.reference_model_path,
                                   self.diff_model_path)
        display_args = mock_display.call_args[0][0].data
        # no differences between equal models
        self.assertIn("No differences found", display_args)

    @patch('keras.src.utils.model_tools.display')
    def test_missing_layer_models(self, mock_display):
        style1 = "background-color:rgba(248, 215, 218, 0.5); color:#721c24;"
        style2 = "background-color:rgba(212, 237, 218, 0.5); color:#155724;"
        model1 = self.create_reference_model(10, 'reference_model')
        save_model(model1, self.reference_model_path)

        input_layer = Input(shape=(20,), name='input')
        x = Dense(10, activation='relu', name='dense_1')(input_layer)
        x = Dense(5, activation='relu', name='dense_2')(x)
        x = Dense(3, activation='relu', name='extra_layer')(x)  # extra layer
        output_layer = Dense(1, activation='sigmoid', name='output')(x)
        model2 = Model(inputs=input_layer, outputs=output_layer,
                       name='diff_model')
        model2.compile(optimizer=Adam(), loss='binary_crossentropy')
        save_model(model2, self.diff_model_path)
        model_tools.compare_models(self.reference_model_path,
                                   self.diff_model_path)
        display_args = mock_display.call_args[0][0].data
        # output layer (model in the left) is highlighted in red
        self.assertIn(f'"name": "<span style=\'{style1}\'>output"',
                      display_args)
        # extra_layer (model in the right) is highlighted in green
        self.assertIn(f'"name": "<span style=\'{style2}\'>extra_layer"',
                      display_args)
        self.assertIn('"units": 1,', display_args)
        self.assertIn('"units": 3,', display_args)
        self.assertIn('"activation": "sigmoid"', display_args)
        self.assertIn('"activation": "relu"', display_args)


class InspectModelsTest(unittest.TestCase):

    def setUp(self):
        self.temp_filepath = os.path.join(os.getcwd(), "my_model.keras")

    def tearDown(self):
        if os.path.exists(self.temp_filepath):
            os.remove(self.temp_filepath)

    @patch('keras.src.utils.model_tools.display')
    def test_inspect_keras_model(self, mock_display):
        inputs = Input(shape=(4,))
        outputs = Dense(1)(inputs)
        model = Model(inputs, outputs)
        model.save(self.temp_filepath)

        model_tools.inspect_file(self.temp_filepath)
        display_args = mock_display.call_args[0][0].data
        display_args.replace('\n', '')
        self.assertIn("Keras model file", display_args)
        self.assertIn("Model: Functional", display_args)
        self.assertIn("name='functional_1'", display_args)
        self.assertIn("Saved with Keras", display_args)
        self.assertIn("Weights file:", display_args)
        self.assertIn("dense", display_args)
        self.assertIn("0: (4, 1) float32", display_args)
        self.assertIn("1: (1,) float32", display_args)


class TestModelPatching(unittest.TestCase):
    def setUp(self):
        input_layer = Input(shape=(20,), name='input')
        x = Dense(10, activation='relu', name='dense_1')(input_layer)
        output_layer = Dense(1, activation='sigmoid', name='output')(x)
        self.model = Model(inputs=input_layer, outputs=output_layer,
                           name='test_model')
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

        self.model_path = 'test_model.keras'
        save_model(self.model, self.model_path)

        self.patched_model_path = 'patched_model.keras'

    def tearDown(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        if os.path.exists(self.patched_model_path):
            os.remove(self.patched_model_path)

    def test_patch_weight(self):
        weight_index = (0, 0, 1)
        new_value = 0.5

        original_weights = self.model.get_layer(name='dense_1').get_weights()
        original_weight_value = original_weights[0][0][1]

        model_tools.patch_weight(self.model_path, 'dense_1', weight_index,
                                 new_value, self.patched_model_path)

        patched_model = load_model(self.patched_model_path)
        patched_weights = patched_model.get_layer(name='dense_1').get_weights()
        print("patched_weights: ", patched_weights[0][0][1])
        patched_weight_value = patched_weights[0][0][1]

        self.assertNotEqual(original_weight_value, patched_weight_value)
        self.assertEqual(patched_weight_value, new_value)

    # we use the decorator to mock the input function
    @patch('builtins.input', side_effect=[
            'dense_1',
            'renamed_dense_1',
            'patched_model.keras'
        ])
    def test_change_layer_name(self, mock_input):
        model_tools.change_layer_name(self.model_path)

        patched_model = load_model(self.patched_model_path)
        layer_names = [layer.name for layer in patched_model.layers]
        self.assertIn('renamed_dense_1', layer_names)
        self.assertNotIn('dense_1', layer_names)
