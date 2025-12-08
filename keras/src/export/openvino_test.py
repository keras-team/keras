import os

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src import tree
from keras.src.export import openvino
from keras.src.saving import saving_lib
from keras.src.testing.test_utils import named_product

try:
    import openvino as ov
except ImportError:
    ov = None


class CustomModel(models.Model):
    def __init__(self, layer_list):
        super().__init__()
        self.layer_list = layer_list

    def call(self, input):
        output = input
        for layer in self.layer_list:
            output = layer(output)
        return output


def get_model(type="sequential", input_shape=(10,), layer_list=None):
    layer_list = layer_list or [
        layers.Dense(10, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(1, activation="sigmoid"),
    ]
    if type == "sequential":
        return models.Sequential(layer_list)
    elif type == "functional":
        input = output = tree.map_shape_structure(layers.Input, input_shape)
        for layer in layer_list:
            output = layer(output)
        return models.Model(inputs=input, outputs=output)
    elif type == "subclass":
        return CustomModel(layer_list)
    elif type == "lstm":
        # https://github.com/keras-team/keras/issues/21390
        inputs = layers.Input((4, 10))
        x = layers.Bidirectional(
            layers.LSTM(
                10,
                kernel_initializer="he_normal",
                return_sequences=True,
                kernel_regularizer=None,
            ),
            merge_mode="sum",
        )(inputs)
        outputs = layers.Bidirectional(
            layers.LSTM(
                10,
                kernel_initializer="he_normal",
                return_sequences=True,
                kernel_regularizer=None,
            ),
            merge_mode="concat",
        )(x)
        return models.Model(inputs=inputs, outputs=outputs)


@pytest.mark.skipif(ov is None, reason="OpenVINO is not installed")
@pytest.mark.skipif(
    backend.backend() not in ("tensorflow", "openvino", "jax", "torch"),
    reason=(
        "`export_openvino` only currently supports"
        "the tensorflow, jax, torch and openvino backends."
    ),
)
@pytest.mark.skipif(testing.jax_uses_gpu(), reason="Leads to core dumps on CI")
@pytest.mark.skipif(
    testing.tensorflow_uses_gpu(), reason="Leads to core dumps on CI"
)
class ExportOpenVINOTest(testing.TestCase):
    @parameterized.named_parameters(
        named_product(
            model_type=["sequential", "functional", "subclass", "lstm"]
        )
    )
    def test_standard_model_export(self, model_type):
        if model_type == "lstm":
            self.skipTest(
                "LSTM export not supported - unimplemented QR operation"
            )

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model.xml")
        model = get_model(model_type)
        batch_size = 3
        if model_type == "lstm":
            ref_input = np.random.normal(size=(batch_size, 4, 10))
        else:
            ref_input = np.random.normal(size=(batch_size, 10))
        ref_input = ref_input.astype("float32")
        ref_output = model(ref_input)

        try:
            openvino.export_openvino(model, temp_filepath)
        except Exception as e:
            if "XlaCallModule" in str(e):
                self.skipTest("OpenVINO does not support XlaCallModule yet")
            raise e

        # Load and run inference with OpenVINO
        core = ov.Core()
        ov_model = core.read_model(temp_filepath)
        compiled_model = core.compile_model(ov_model, "CPU")

        ov_output = compiled_model([ref_input])[compiled_model.output(0)]

        self.assertAllClose(ref_output, ov_output)

        larger_input = np.concatenate([ref_input, ref_input], axis=0)
        compiled_model([larger_input])

    @parameterized.named_parameters(
        named_product(struct_type=["tuple", "array", "dict"])
    )
    def test_model_with_input_structure(self, struct_type):
        class TupleModel(models.Model):
            def call(self, inputs):
                x, y = inputs
                return ops.add(x, y)

        class ArrayModel(models.Model):
            def call(self, inputs):
                x = inputs[0]
                y = inputs[1]
                return ops.add(x, y)

        class DictModel(models.Model):
            def call(self, inputs):
                x = inputs["x"]
                y = inputs["y"]
                return ops.add(x, y)

        batch_size = 3
        ref_input = np.random.normal(size=(batch_size, 10)).astype("float32")
        if struct_type == "tuple":
            model = TupleModel()
            ref_input = (ref_input, ref_input * 2)
        elif struct_type == "array":
            model = ArrayModel()
            ref_input = [ref_input, ref_input * 2]
        elif struct_type == "dict":
            model = DictModel()
            ref_input = {"x": ref_input, "y": ref_input * 2}

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model.xml")
        ref_output = model(tree.map_structure(ops.convert_to_tensor, ref_input))

        try:
            openvino.export_openvino(model, temp_filepath)
        except Exception as e:
            if "XlaCallModule" in str(e):
                self.skipTest("OpenVINO does not support XlaCallModule yet")
            raise e

        # Load and run inference with OpenVINO
        core = ov.Core()
        ov_model = core.read_model(temp_filepath)
        compiled_model = core.compile_model(ov_model, "CPU")

        if isinstance(ref_input, dict):
            ov_inputs = [ref_input[key] for key in ref_input.keys()]
        else:
            ov_inputs = list(ref_input)

        ov_output = compiled_model(ov_inputs)[compiled_model.output(0)]
        self.assertAllClose(ref_output, ov_output)

        # Test with keras.saving_lib
        temp_filepath = os.path.join(
            self.get_temp_dir(), "exported_model.keras"
        )
        saving_lib.save_model(model, temp_filepath)
        revived_model = saving_lib.load_model(
            temp_filepath,
            {
                "TupleModel": TupleModel,
                "ArrayModel": ArrayModel,
                "DictModel": DictModel,
            },
        )
        self.assertAllClose(ref_output, revived_model(ref_input))
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model2.xml")
        try:
            openvino.export_openvino(revived_model, temp_filepath)
        except Exception as e:
            if "XlaCallModule" in str(e):
                self.skipTest("OpenVINO does not support XlaCallModule yet")
            raise e

        bigger_ref_input = tree.map_structure(
            lambda x: np.concatenate([x, x], axis=0), ref_input
        )
        if isinstance(bigger_ref_input, dict):
            bigger_ov_inputs = [
                bigger_ref_input[key] for key in bigger_ref_input.keys()
            ]
        else:
            bigger_ov_inputs = list(bigger_ref_input)
        compiled_model(bigger_ov_inputs)

    def test_model_with_multiple_inputs(self):
        class TwoInputsModel(models.Model):
            def call(self, x, y):
                return x + y

            def build(self, y_shape, x_shape):
                self.built = True

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model.xml")
        model = TwoInputsModel()
        batch_size = 3
        ref_input_x = np.random.normal(size=(batch_size, 10)).astype("float32")
        ref_input_y = np.random.normal(size=(batch_size, 10)).astype("float32")
        ref_output = model(ref_input_x, ref_input_y)

        try:
            openvino.export_openvino(model, temp_filepath)
        except Exception as e:
            if "XlaCallModule" in str(e):
                self.skipTest("OpenVINO does not support XlaCallModule yet")
            raise e

        # Load and run inference with OpenVINO
        core = ov.Core()
        ov_model = core.read_model(temp_filepath)
        compiled_model = core.compile_model(ov_model, "CPU")

        ov_output = compiled_model([ref_input_x, ref_input_y])[
            compiled_model.output(0)
        ]
        self.assertAllClose(ref_output, ov_output)
        larger_input_x = np.concatenate([ref_input_x, ref_input_x], axis=0)
        larger_input_y = np.concatenate([ref_input_y, ref_input_y], axis=0)
        compiled_model([larger_input_x, larger_input_y])
