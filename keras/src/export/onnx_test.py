"""Tests for ONNX exporting utilities."""

import os

import numpy as np
import onnxruntime
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src import tree
from keras.src.export import onnx
from keras.src.layers.input_spec import InputSpec as InputSpec
from keras.src.saving import saving_lib
from keras.src.testing.test_utils import named_product


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


@pytest.mark.skipif(
    backend.backend() not in ("tensorflow", "jax", "torch"),
    reason=(
        "`export_onnx` only currently supports the tensorflow, jax and torch "
        "backends."
    ),
)
@pytest.mark.skipif(
    testing.jax_uses_gpu()
    or testing.tensorflow_uses_gpu()
    or testing.torch_uses_gpu(),
    reason="Fails on GPU",
)
@pytest.mark.skipif(
    np.version.version.startswith("2."),
    reason="ONNX export is currently incompatible with NumPy 2.0",
)
class ExportONNXTest(testing.TestCase):
    @parameterized.named_parameters(
        named_product(
            model_type=["sequential", "functional", "subclass", "lstm"]
        )
    )
    def test_standard_model_export(self, model_type):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model(model_type)
        batch_size = 3 if backend.backend() != "torch" else 1
        if model_type == "lstm":
            ref_input = np.random.normal(size=(batch_size, 4, 10))
        else:
            ref_input = np.random.normal(size=(batch_size, 10))
        ref_input = ref_input.astype("float32")
        ref_output = model(ref_input)

        onnx.export_onnx(model, temp_filepath)
        ort_session = onnxruntime.InferenceSession(temp_filepath)
        ort_inputs = {
            k.name: v for k, v in zip(ort_session.get_inputs(), [ref_input])
        }
        self.assertAllClose(ref_output, ort_session.run(None, ort_inputs)[0])
        # Test with a different batch size
        if backend.backend() == "torch":
            # TODO: Dynamic shape is not supported yet in the torch backend
            return
        ort_inputs = {
            k.name: v
            for k, v in zip(
                ort_session.get_inputs(),
                [np.concatenate([ref_input, ref_input], axis=0)],
            )
        }
        ort_session.run(None, ort_inputs)

    @parameterized.named_parameters(
        named_product(struct_type=["tuple", "array", "dict"])
    )
    def test_model_with_input_structure(self, struct_type):
        if backend.backend() == "torch" and struct_type == "dict":
            self.skipTest("The torch backend doesn't support the dict model.")

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

        batch_size = 3 if backend.backend() != "torch" else 1
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

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        ref_output = model(tree.map_structure(ops.convert_to_tensor, ref_input))

        onnx.export_onnx(model, temp_filepath)
        ort_session = onnxruntime.InferenceSession(temp_filepath)
        if isinstance(ref_input, dict):
            ort_inputs = {
                k.name: v
                for k, v in zip(ort_session.get_inputs(), ref_input.values())
            }
        else:
            ort_inputs = {
                k.name: v for k, v in zip(ort_session.get_inputs(), ref_input)
            }
        self.assertAllClose(ref_output, ort_session.run(None, ort_inputs)[0])

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
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model2")
        onnx.export_onnx(revived_model, temp_filepath)

        # Test with a different batch size
        if backend.backend() == "torch":
            # TODO: Dynamic shape is not supported yet in the torch backend
            return
        bigger_ref_input = tree.map_structure(
            lambda x: np.concatenate([x, x], axis=0), ref_input
        )
        if isinstance(bigger_ref_input, dict):
            bigger_ort_inputs = {
                k.name: v
                for k, v in zip(
                    ort_session.get_inputs(), bigger_ref_input.values()
                )
            }
        else:
            bigger_ort_inputs = {
                k.name: v
                for k, v in zip(ort_session.get_inputs(), bigger_ref_input)
            }
        ort_session.run(None, bigger_ort_inputs)

    def test_model_with_multiple_inputs(self):
        class TwoInputsModel(models.Model):
            def call(self, x, y):
                return x + y

            def build(self, y_shape, x_shape):
                self.built = True

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = TwoInputsModel()
        batch_size = 3 if backend.backend() != "torch" else 1
        ref_input_x = np.random.normal(size=(batch_size, 10)).astype("float32")
        ref_input_y = np.random.normal(size=(batch_size, 10)).astype("float32")
        ref_output = model(ref_input_x, ref_input_y)

        onnx.export_onnx(model, temp_filepath)
        ort_session = onnxruntime.InferenceSession(temp_filepath)
        ort_inputs = {
            k.name: v
            for k, v in zip(
                ort_session.get_inputs(), [ref_input_x, ref_input_y]
            )
        }
        self.assertAllClose(ref_output, ort_session.run(None, ort_inputs)[0])
        # Test with a different batch size
        if backend.backend() == "torch":
            # TODO: Dynamic shape is not supported yet in the torch backend
            return
        ort_inputs = {
            k.name: v
            for k, v in zip(
                ort_session.get_inputs(),
                [
                    np.concatenate([ref_input_x, ref_input_x], axis=0),
                    np.concatenate([ref_input_y, ref_input_y], axis=0),
                ],
            )
        }
        ort_session.run(None, ort_inputs)

    @parameterized.named_parameters(named_product(opset_version=[None, 17]))
    def test_export_with_opset_version(self, opset_version):
        import onnx as onnx_lib

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model("sequential")
        batch_size = 3 if backend.backend() != "torch" else 1
        ref_input = np.random.normal(size=(batch_size, 10))
        ref_input = ref_input.astype("float32")
        ref_output = model(ref_input)

        onnx.export_onnx(
            model, temp_filepath, opset_version=opset_version, verbose=True
        )
        ort_session = onnxruntime.InferenceSession(temp_filepath)
        ort_inputs = {
            k.name: v for k, v in zip(ort_session.get_inputs(), [ref_input])
        }
        self.assertAllClose(ref_output, ort_session.run(None, ort_inputs)[0])

        if opset_version is not None:
            onnx_model = onnx_lib.load(temp_filepath)
            self.assertEqual(onnx_model.opset_import[0].version, opset_version)

    def test_export_with_input_names(self):
        """Test ONNX export uses InputSpec.name for input names."""
        import onnx as onnx_lib

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model("sequential")
        batch_size = 3 if backend.backend() != "torch" else 1
        ref_input = np.random.normal(size=(batch_size, 10)).astype("float32")
        ref_output = model(ref_input)

        # Test with custom input name
        input_spec = [
            InputSpec(
                name="custom_input", shape=(batch_size, 10), dtype="float32"
            )
        ]
        onnx.export_onnx(model, temp_filepath, input_signature=input_spec)

        onnx_model = onnx_lib.load(temp_filepath)
        input_names = [input.name for input in onnx_model.graph.input]
        self.assertIn("custom_input", input_names)

        ort_session = onnxruntime.InferenceSession(temp_filepath)
        ort_inputs = {
            k.name: v for k, v in zip(ort_session.get_inputs(), [ref_input])
        }
        self.assertAllClose(ref_output, ort_session.run(None, ort_inputs)[0])
