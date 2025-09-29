"""Tests for LiteRT exporting utilities."""

import os

import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized

# Try to use AI Edge LiteRT interpreter, fallback to TensorFlow Lite
try:
    from ai_edge_litert.interpreter import Interpreter as LiteRtInterpreter
    litert_available = True
    print("Using AI Edge LiteRT interpreter")
except ImportError:
    # Fallback to TensorFlow Lite interpreter
    LiteRtInterpreter = tf.lite.Interpreter
    litert_available = True
    print("Using TensorFlow Lite interpreter as fallback")

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src import tree
from keras.src.export import export_litert
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
        model = models.Sequential(layer_list)
        model.build(input_shape=(None,) + input_shape)
        return model
    if type == "functional":
        input = output = tree.map_shape_structure(layers.Input, input_shape)
        for layer in layer_list:
            output = layer(output)
        return models.Model(inputs=input, outputs=output)
    if type == "subclass":
        model = CustomModel(layer_list)
        model.build(input_shape=(None,) + input_shape)
        # Trace the model with dummy data to ensure it's properly built for export
        dummy_input = np.zeros((1,) + input_shape, dtype=np.float32)
        _ = model(dummy_input)  # This traces the model
        return model
    if type == "lstm":
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
    raise ValueError(f"Unknown model type: {type}")





def _convert_to_numpy(structure):
    return tree.map_structure(
        lambda x: x.numpy() if hasattr(x, "numpy") else np.array(x), structure
    )


def _normalize_name(name):
    normalized = name.split(":")[0]
    if normalized.startswith("serving_default_"):
        normalized = normalized[len("serving_default_") :]
    return normalized


def _set_interpreter_inputs(interpreter, inputs):
    input_details = interpreter.get_input_details()
    if isinstance(inputs, dict):
        for detail in input_details:
            key = _normalize_name(detail["name"])
            if key in inputs:
                value = inputs[key]
            else:
                matched_key = None
                for candidate in inputs:
                    if key.endswith(candidate) or candidate.endswith(key):
                        matched_key = candidate
                        break
                if matched_key is None:
                    raise KeyError(
                        f"Unable to match input '{detail['name']}' in provided inputs"
                    )
                value = inputs[matched_key]
            interpreter.set_tensor(detail["index"], value)
    else:
        values = inputs
        if not isinstance(values, (list, tuple)):
            values = [values]
        if len(values) != len(input_details):
            raise ValueError(
                "Number of provided inputs does not match interpreter signature"
            )
        for detail, value in zip(input_details, values):
            interpreter.set_tensor(detail["index"], value)


def _get_interpreter_outputs(interpreter):
    output_details = interpreter.get_output_details()
    outputs = [interpreter.get_tensor(detail["index"]) for detail in output_details]
    return outputs[0] if len(outputs) == 1 else outputs


@pytest.mark.skipif(
    backend.backend() != "tensorflow",
    reason="`export_litert` currently supports the tensorflow backend only.",
)
@pytest.mark.skipif(
    testing.tensorflow_uses_gpu(),
    reason="LiteRT export tests are only run on CPU to avoid CI issues.",
)
# Note: Tests use AI Edge LiteRT interpreter when available, 
# fallback to TensorFlow Lite interpreter otherwise
class ExportLitertTest(testing.TestCase):
    @parameterized.named_parameters(
        named_product(
            model_type=["sequential", "functional", "lstm"]
        )
    )
    def test_standard_model_export(self, model_type):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = get_model(model_type)
        batch_size = 1  # TFLite expects batch_size=1
        if model_type == "lstm":
            ref_input = np.random.normal(size=(batch_size, 4, 10))
        else:
            ref_input = np.random.normal(size=(batch_size, 10))
        ref_input = ref_input.astype("float32")
        ref_output = _convert_to_numpy(model(ref_input))

        export_path = export_litert(model, temp_filepath)
        self.assertTrue(export_path.endswith(".tflite"))
        self.assertTrue(os.path.exists(export_path))

        interpreter = LiteRtInterpreter(model_path=export_path)
        interpreter.allocate_tensors()
        _set_interpreter_inputs(interpreter, ref_input)
        interpreter.invoke()
        litert_output = _get_interpreter_outputs(interpreter)

        self.assertAllClose(ref_output, litert_output, atol=1e-4, rtol=1e-4)

    @parameterized.named_parameters(
        named_product(struct_type=["tuple", "array", "dict"])
    )
    def test_model_with_input_structure(self, struct_type):
        batch_size = 1  # TFLite expects batch_size=1
        base_input = np.random.normal(size=(batch_size, 10)).astype("float32")
        
        if struct_type == "tuple":
            # Use Functional API for proper Input layer handling
            input1 = layers.Input(shape=(10,), name="input_1")
            input2 = layers.Input(shape=(10,), name="input_2")
            output = layers.Add()([input1, input2])
            model = models.Model(inputs=[input1, input2], outputs=output)
            ref_input = (base_input, base_input * 2)
        elif struct_type == "array":
            # Use Functional API for proper Input layer handling
            input1 = layers.Input(shape=(10,), name="input_1")
            input2 = layers.Input(shape=(10,), name="input_2")
            output = layers.Add()([input1, input2])
            model = models.Model(inputs=[input1, input2], outputs=output)
            ref_input = [base_input, base_input * 2]
        elif struct_type == "dict":
            # Use Functional API for proper Input layer handling
            input1 = layers.Input(shape=(10,), name="x")
            input2 = layers.Input(shape=(10,), name="y")
            output = layers.Add()([input1, input2])
            model = models.Model(inputs={"x": input1, "y": input2}, outputs=output)
            ref_input = {"x": base_input, "y": base_input * 2}
        else:
            raise AssertionError("Unexpected structure type")

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        ref_output = _convert_to_numpy(
            model(tree.map_structure(ops.convert_to_tensor, ref_input))
        )

        export_path = export_litert(model, temp_filepath)
        interpreter = LiteRtInterpreter(model_path=export_path)
        interpreter.allocate_tensors()

        feed_inputs = ref_input
        if isinstance(feed_inputs, tuple):
            feed_inputs = list(feed_inputs)
        _set_interpreter_inputs(interpreter, feed_inputs)
        interpreter.invoke()
        litert_output = _get_interpreter_outputs(interpreter)

        self.assertAllClose(ref_output, litert_output, atol=1e-4, rtol=1e-4)

        # Verify export still works after saving/loading via saving_lib.
        archive_path = os.path.join(self.get_temp_dir(), "revived.keras")
        saving_lib.save_model(model, archive_path)
        revived_model = saving_lib.load_model(archive_path)
        revived_output = _convert_to_numpy(revived_model(ref_input))
        self.assertAllClose(ref_output, revived_output)

    def test_model_with_multiple_inputs(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        
        # Use Functional API for proper Input layer handling
        input_x = layers.Input(shape=(10,), name="x")
        input_y = layers.Input(shape=(10,), name="y")
        output = layers.Add()([input_x, input_y])
        model = models.Model(inputs=[input_x, input_y], outputs=output)
        
        batch_size = 1  # TFLite expects batch_size=1
        ref_input_x = np.random.normal(size=(batch_size, 10)).astype("float32")
        ref_input_y = np.random.normal(size=(batch_size, 10)).astype("float32")
        ref_output = _convert_to_numpy(model([ref_input_x, ref_input_y]))

        export_path = export_litert(model, temp_filepath)
        interpreter = LiteRtInterpreter(model_path=export_path)
        interpreter.allocate_tensors()

        _set_interpreter_inputs(interpreter, [ref_input_x, ref_input_y])
        interpreter.invoke()
        litert_output = _get_interpreter_outputs(interpreter)

        self.assertAllClose(ref_output, litert_output, atol=1e-4, rtol=1e-4)

        # Test with a different batch size by resizing interpreter inputs.
        larger_x = np.concatenate([ref_input_x, ref_input_x], axis=0)
        larger_y = np.concatenate([ref_input_y, ref_input_y], axis=0)
        input_details = interpreter.get_input_details()
        interpreter.resize_tensor_input(input_details[0]["index"], larger_x.shape)
        interpreter.resize_tensor_input(input_details[1]["index"], larger_y.shape)
        interpreter.allocate_tensors()
        _set_interpreter_inputs(interpreter, [larger_x, larger_y])
        interpreter.invoke()
        larger_output = _get_interpreter_outputs(interpreter)
        larger_ref_output = _convert_to_numpy(model([larger_x, larger_y]))
        self.assertAllClose(larger_ref_output, larger_output, atol=1e-4, rtol=1e-4)

    def test_export_with_custom_input_signature(self):
        model = get_model("sequential")
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        input_signature = [layers.InputSpec(shape=(None, 10), dtype="float32")]

        export_path = export_litert(
            model,
            temp_filepath,
            input_signature=input_signature,
        )
        self.assertTrue(os.path.exists(export_path))

        interpreter = LiteRtInterpreter(model_path=export_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertEqual(len(input_details), 1)
        self.assertEqual(tuple(input_details[0]["shape"][1:]), (10,))