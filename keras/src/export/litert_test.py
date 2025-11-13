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
from keras.src.saving import saving_lib
from keras.src.testing.test_utils import named_product
from keras.src.utils.module_utils import litert
from keras.src.utils.module_utils import tensorflow

# Set up LiteRT interpreter with fallback logic:
# 1. Try AI Edge LiteRT interpreter (preferred)
# 2. Fall back to TensorFlow Lite interpreter if AI Edge LiteRT unavailable
AI_EDGE_LITERT_AVAILABLE = False
LiteRTInterpreter = None

if backend.backend() == "tensorflow":
    if litert.available:
        try:
            from ai_edge_litert.interpreter import (
                Interpreter as LiteRTInterpreter,
            )

            AI_EDGE_LITERT_AVAILABLE = True
        except (ImportError, OSError):
            LiteRTInterpreter = tensorflow.lite.Interpreter
    else:
        LiteRTInterpreter = tensorflow.lite.Interpreter

# Model types to test (LSTM only if AI Edge LiteRT is available)
model_types = ["sequential", "functional"]
if AI_EDGE_LITERT_AVAILABLE:
    model_types.append("lstm")


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
        # Trace the model with dummy data to ensure it's properly built for
        # export
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
    if type == "multi_input":
        input1 = layers.Input(shape=input_shape, name="input1")
        input2 = layers.Input(shape=input_shape, name="input2")
        x1 = layers.Dense(10, activation="relu")(input1)
        x2 = layers.Dense(10, activation="relu")(input2)
        combined = layers.concatenate([x1, x2])
        output = layers.Dense(1, activation="sigmoid")(combined)
        return models.Model(inputs=[input1, input2], outputs=output)
    if type == "multi_output":
        inputs = layers.Input(shape=input_shape)
        shared = layers.Dense(20, activation="relu")(inputs)
        output1 = layers.Dense(1, activation="sigmoid", name="output1")(shared)
        output2 = layers.Dense(3, activation="softmax", name="output2")(shared)
        return models.Model(inputs=inputs, outputs=[output1, output2])
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
                        f"Unable to match input '{detail['name']}' in provided "
                        f"inputs"
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
    outputs = [
        interpreter.get_tensor(detail["index"]) for detail in output_details
    ]
    return outputs[0] if len(outputs) == 1 else outputs


@pytest.mark.skipif(
    backend.backend() != "tensorflow",
    reason="`export_litert` currently supports the TensorFlow backend only.",
)
class ExportLitertTest(testing.TestCase):
    """Test suite for LiteRT (TFLite) model export functionality.

    Tests use AI Edge LiteRT interpreter when available, otherwise fall back
    to TensorFlow Lite interpreter for validation.
    """

    @parameterized.named_parameters(named_product(model_type=model_types))
    def test_standard_model_export(self, model_type):
        """Test exporting standard model types to LiteRT format."""
        if model_type == "lstm" and not AI_EDGE_LITERT_AVAILABLE:
            self.skipTest("LSTM models require AI Edge LiteRT interpreter.")

        temp_filepath = os.path.join(
            self.get_temp_dir(), "exported_model.tflite"
        )
        model = get_model(model_type)
        batch_size = 1  # LiteRT expects batch_size=1
        if model_type == "lstm":
            ref_input = np.random.normal(size=(batch_size, 4, 10))
        else:
            ref_input = np.random.normal(size=(batch_size, 10))
        ref_input = ref_input.astype("float32")
        ref_output = _convert_to_numpy(model(ref_input))

        # Test with model.export()
        model.export(temp_filepath, format="litert")
        self.assertTrue(os.path.exists(temp_filepath))

        interpreter = LiteRTInterpreter(model_path=temp_filepath)
        interpreter.allocate_tensors()
        _set_interpreter_inputs(interpreter, ref_input)
        interpreter.invoke()
        litert_output = _get_interpreter_outputs(interpreter)

        self.assertAllClose(ref_output, litert_output, atol=1e-4, rtol=1e-4)

    @parameterized.named_parameters(
        named_product(struct_type=["tuple", "array", "dict"])
    )
    def test_model_with_input_structure(self, struct_type):
        """Test exporting models with structured inputs (tuple/array/dict)."""
        batch_size = 1  # LiteRT expects batch_size=1
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
            model = models.Model(
                inputs={"x": input1, "y": input2}, outputs=output
            )
            ref_input = {"x": base_input, "y": base_input * 2}
        else:
            raise AssertionError("Unexpected structure type")

        temp_filepath = os.path.join(
            self.get_temp_dir(), "exported_model.tflite"
        )
        ref_output = _convert_to_numpy(
            model(tree.map_structure(ops.convert_to_tensor, ref_input))
        )

        # Test with model.export()
        model.export(temp_filepath, format="litert")
        export_path = temp_filepath
        interpreter = LiteRTInterpreter(model_path=export_path)
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
        """Test exporting models with multiple inputs and batch resizing."""
        temp_filepath = os.path.join(
            self.get_temp_dir(), "exported_model.tflite"
        )

        # Use Functional API for proper Input layer handling
        input_x = layers.Input(shape=(10,), name="x")
        input_y = layers.Input(shape=(10,), name="y")
        output = layers.Add()([input_x, input_y])
        model = models.Model(inputs=[input_x, input_y], outputs=output)

        batch_size = 1  # LiteRT expects batch_size=1
        ref_input_x = np.random.normal(size=(batch_size, 10)).astype("float32")
        ref_input_y = np.random.normal(size=(batch_size, 10)).astype("float32")
        ref_output = _convert_to_numpy(model([ref_input_x, ref_input_y]))

        # Test with model.export()
        model.export(temp_filepath, format="litert")
        export_path = temp_filepath
        interpreter = LiteRTInterpreter(model_path=export_path)
        interpreter.allocate_tensors()

        _set_interpreter_inputs(interpreter, [ref_input_x, ref_input_y])
        interpreter.invoke()
        litert_output = _get_interpreter_outputs(interpreter)

        self.assertAllClose(ref_output, litert_output, atol=1e-4, rtol=1e-4)

        # Test with a different batch size by resizing interpreter inputs.
        larger_x = np.concatenate([ref_input_x, ref_input_x], axis=0)
        larger_y = np.concatenate([ref_input_y, ref_input_y], axis=0)
        input_details = interpreter.get_input_details()
        interpreter.resize_tensor_input(
            input_details[0]["index"], larger_x.shape
        )
        interpreter.resize_tensor_input(
            input_details[1]["index"], larger_y.shape
        )
        interpreter.allocate_tensors()
        _set_interpreter_inputs(interpreter, [larger_x, larger_y])
        interpreter.invoke()
        larger_output = _get_interpreter_outputs(interpreter)
        larger_ref_output = _convert_to_numpy(model([larger_x, larger_y]))
        self.assertAllClose(
            larger_ref_output, larger_output, atol=1e-4, rtol=1e-4
        )

    def test_export_with_custom_input_signature(self):
        """Test exporting with custom input signature specification."""
        model = get_model("sequential")
        temp_filepath = os.path.join(
            self.get_temp_dir(), "exported_model.tflite"
        )
        input_signature = [layers.InputSpec(shape=(None, 10), dtype="float32")]

        # Test with model.export()
        model.export(
            temp_filepath,
            format="litert",
            input_signature=input_signature,
        )
        export_path = temp_filepath
        self.assertTrue(os.path.exists(export_path))

        interpreter = LiteRTInterpreter(model_path=export_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertEqual(len(input_details), 1)
        self.assertEqual(tuple(input_details[0]["shape"][1:]), (10,))

    def test_multi_output_model_export(self):
        """Test exporting multi-output models."""
        model = get_model("multi_output")

        # Build the model
        ref_input = np.random.normal(size=(3, 10)).astype("float32")
        model(ref_input)

        temp_filepath = os.path.join(
            self.get_temp_dir(), "exported_model.tflite"
        )
        model.export(temp_filepath, format="litert")

        tflite_path = temp_filepath
        self.assertTrue(os.path.exists(tflite_path))

        # Test inference
        interpreter = LiteRTInterpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        self.assertEqual(len(output_details), 2)

        test_input = np.random.random(input_details[0]["shape"]).astype(
            np.float32
        )
        interpreter.set_tensor(input_details[0]["index"], test_input)
        interpreter.invoke()

        for detail in output_details:
            output = interpreter.get_tensor(detail["index"])
            self.assertIsInstance(output, np.ndarray)

    def test_export_with_verbose(self):
        """Test export with verbose output."""
        model = get_model("sequential")
        dummy_input = np.random.random((3, 10)).astype(np.float32)
        model(dummy_input)

        temp_filepath = os.path.join(
            self.get_temp_dir(), "exported_model.tflite"
        )

        # Export with verbose=True
        model.export(temp_filepath, format="litert", verbose=True)

        tflite_path = temp_filepath
        self.assertTrue(os.path.exists(tflite_path))

        # Verify the exported model works
        interpreter = LiteRTInterpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        self.assertEqual(len(input_details), 1)

    def test_export_error_handling(self):
        """Test error handling in export API."""
        model = get_model("sequential")
        dummy_input = np.random.random((3, 10)).astype(np.float32)
        model(dummy_input)

        temp_filepath = os.path.join(
            self.get_temp_dir(), "exported_model.tflite"
        )

        # Test with invalid format
        with self.assertRaises(ValueError):
            model.export(temp_filepath, format="invalid_format")

    def test_export_invalid_filepath(self):
        """Test that export fails with invalid file extension."""
        model = get_model("sequential")
        dummy_input = np.random.random((3, 10)).astype(np.float32)
        model(dummy_input)

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model.txt")

        # Should raise ValueError for wrong extension
        with self.assertRaises(ValueError):
            model.export(temp_filepath, format="litert")

    def test_export_subclass_model(self):
        """Test exporting subclass models (uses wrapper conversion path)."""
        if LiteRTInterpreter is None:
            self.skipTest("No LiteRT interpreter available")

        model = get_model("subclass")
        temp_filepath = os.path.join(
            self.get_temp_dir(), "exported_model.tflite"
        )

        batch_size = 1
        ref_input = np.random.normal(size=(batch_size, 10)).astype("float32")
        ref_output = _convert_to_numpy(model(ref_input))

        # Export subclass model - this tests wrapper-based conversion
        model.export(temp_filepath, format="litert")
        self.assertTrue(os.path.exists(temp_filepath))

        # Verify inference
        interpreter = LiteRTInterpreter(model_path=temp_filepath)
        interpreter.allocate_tensors()
        _set_interpreter_inputs(interpreter, ref_input)
        interpreter.invoke()
        litert_output = _get_interpreter_outputs(interpreter)

        self.assertAllClose(ref_output, litert_output, atol=1e-4, rtol=1e-4)

    def test_export_with_optimizations_default(self):
        """Test export with DEFAULT optimization."""
        if LiteRTInterpreter is None:
            self.skipTest("No LiteRT interpreter available")

        model = get_model("sequential")
        temp_filepath = os.path.join(
            self.get_temp_dir(), "optimized_default.tflite"
        )

        batch_size = 1
        ref_input = np.random.normal(size=(batch_size, 10)).astype("float32")
        ref_output = _convert_to_numpy(model(ref_input))

        # Export with DEFAULT optimization
        model.export(
            temp_filepath,
            format="litert",
            optimizations=[tensorflow.lite.Optimize.DEFAULT],
        )
        self.assertTrue(os.path.exists(temp_filepath))

        # Verify inference still works
        interpreter = LiteRTInterpreter(model_path=temp_filepath)
        interpreter.allocate_tensors()
        _set_interpreter_inputs(interpreter, ref_input)
        interpreter.invoke()
        litert_output = _get_interpreter_outputs(interpreter)

        # Quantized model should be close but not exact
        self.assertAllClose(ref_output, litert_output, atol=1e-2, rtol=1e-2)

    def test_export_with_optimizations_sparsity(self):
        """Test export with EXPERIMENTAL_SPARSITY optimization."""
        if LiteRTInterpreter is None:
            self.skipTest("No LiteRT interpreter available")

        model = get_model("functional")
        temp_filepath = os.path.join(
            self.get_temp_dir(), "optimized_sparsity.tflite"
        )

        batch_size = 1
        ref_input = np.random.normal(size=(batch_size, 10)).astype("float32")

        # Export with EXPERIMENTAL_SPARSITY optimization
        model.export(
            temp_filepath,
            format="litert",
            optimizations=[tensorflow.lite.Optimize.EXPERIMENTAL_SPARSITY],
        )
        self.assertTrue(os.path.exists(temp_filepath))

        # Verify the model can run inference
        interpreter = LiteRTInterpreter(model_path=temp_filepath)
        interpreter.allocate_tensors()
        _set_interpreter_inputs(interpreter, ref_input)
        interpreter.invoke()
        litert_output = _get_interpreter_outputs(interpreter)

        # Output should have valid shape
        self.assertEqual(litert_output.shape, (batch_size, 1))

    def test_export_with_optimizations_size(self):
        """Test export with OPTIMIZE_FOR_SIZE optimization."""
        if LiteRTInterpreter is None:
            self.skipTest("No LiteRT interpreter available")

        model = get_model("sequential")
        temp_filepath = os.path.join(
            self.get_temp_dir(), "optimized_size.tflite"
        )

        batch_size = 1
        ref_input = np.random.normal(size=(batch_size, 10)).astype("float32")

        # Export with OPTIMIZE_FOR_SIZE
        model.export(
            temp_filepath,
            format="litert",
            optimizations=[tensorflow.lite.Optimize.OPTIMIZE_FOR_SIZE],
        )
        self.assertTrue(os.path.exists(temp_filepath))

        # Verify the model can run inference
        interpreter = LiteRTInterpreter(model_path=temp_filepath)
        interpreter.allocate_tensors()
        _set_interpreter_inputs(interpreter, ref_input)
        interpreter.invoke()
        litert_output = _get_interpreter_outputs(interpreter)

        self.assertEqual(litert_output.shape, (batch_size, 1))

    def test_export_with_optimizations_latency(self):
        """Test export with OPTIMIZE_FOR_LATENCY optimization."""
        if LiteRTInterpreter is None:
            self.skipTest("No LiteRT interpreter available")

        model = get_model("functional")
        temp_filepath = os.path.join(
            self.get_temp_dir(), "optimized_latency.tflite"
        )

        batch_size = 1
        ref_input = np.random.normal(size=(batch_size, 10)).astype("float32")

        # Export with OPTIMIZE_FOR_LATENCY
        model.export(
            temp_filepath,
            format="litert",
            optimizations=[tensorflow.lite.Optimize.OPTIMIZE_FOR_LATENCY],
        )
        self.assertTrue(os.path.exists(temp_filepath))

        # Verify the model can run inference
        interpreter = LiteRTInterpreter(model_path=temp_filepath)
        interpreter.allocate_tensors()
        _set_interpreter_inputs(interpreter, ref_input)
        interpreter.invoke()
        litert_output = _get_interpreter_outputs(interpreter)

        self.assertEqual(litert_output.shape, (batch_size, 1))

    def test_export_with_multiple_optimizations(self):
        """Test export with multiple optimization options combined."""
        if LiteRTInterpreter is None:
            self.skipTest("No LiteRT interpreter available")

        model = get_model("sequential")
        temp_filepath = os.path.join(
            self.get_temp_dir(), "optimized_multiple.tflite"
        )

        batch_size = 1
        ref_input = np.random.normal(size=(batch_size, 10)).astype("float32")

        # Export with multiple optimizations
        model.export(
            temp_filepath,
            format="litert",
            optimizations=[
                tensorflow.lite.Optimize.DEFAULT,
                tensorflow.lite.Optimize.EXPERIMENTAL_SPARSITY,
            ],
        )
        self.assertTrue(os.path.exists(temp_filepath))

        # Verify the model can run inference
        interpreter = LiteRTInterpreter(model_path=temp_filepath)
        interpreter.allocate_tensors()
        _set_interpreter_inputs(interpreter, ref_input)
        interpreter.invoke()
        litert_output = _get_interpreter_outputs(interpreter)

        self.assertEqual(litert_output.shape, (batch_size, 1))

    def test_export_with_representative_dataset(self):
        """Test export with representative dataset for better quantization."""
        if LiteRTInterpreter is None:
            self.skipTest("No LiteRT interpreter available")

        model = get_model("functional")
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_model.tflite"
        )

        # Create representative dataset
        def representative_dataset():
            for _ in range(10):
                yield [np.random.normal(size=(1, 10)).astype("float32")]

        # Export with optimizations and representative dataset
        model.export(
            temp_filepath,
            format="litert",
            optimizations=[tensorflow.lite.Optimize.DEFAULT],
            representative_dataset=representative_dataset,
        )
        self.assertTrue(os.path.exists(temp_filepath))

        # Verify the model can run inference
        interpreter = LiteRTInterpreter(model_path=temp_filepath)
        interpreter.allocate_tensors()

        batch_size = 1
        ref_input = np.random.normal(size=(batch_size, 10)).astype("float32")
        _set_interpreter_inputs(interpreter, ref_input)
        interpreter.invoke()
        litert_output = _get_interpreter_outputs(interpreter)

        # Output should have valid shape
        self.assertEqual(litert_output.shape, (batch_size, 1))

    def test_export_with_multiple_kwargs(self):
        """Test export with multiple converter kwargs."""
        if LiteRTInterpreter is None:
            self.skipTest("No LiteRT interpreter available")

        # Create a larger model for quantization testing
        inputs = layers.Input(shape=(28, 28, 3))
        x = layers.Conv2D(32, 3, activation="relu")(inputs)
        x = layers.MaxPooling2D()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(10, activation="softmax")(x)
        model = models.Model(inputs, x)

        temp_filepath = os.path.join(
            self.get_temp_dir(), "multi_kwargs_model.tflite"
        )

        # Create representative dataset
        def representative_dataset():
            for _ in range(5):
                yield [np.random.normal(size=(1, 28, 28, 3)).astype("float32")]

        # Export with multiple kwargs
        model.export(
            temp_filepath,
            format="litert",
            optimizations=[tensorflow.lite.Optimize.DEFAULT],
            representative_dataset=representative_dataset,
            experimental_new_quantizer=True,
        )
        self.assertTrue(os.path.exists(temp_filepath))

        # Verify file size is reduced compared to non-quantized
        file_size = os.path.getsize(temp_filepath)
        self.assertGreater(file_size, 0)

    def test_export_optimization_file_size_comparison(self):
        """Test that optimizations reduce file size."""
        if LiteRTInterpreter is None:
            self.skipTest("No LiteRT interpreter available")

        # Create a larger model to see size differences
        inputs = layers.Input(shape=(28, 28, 3))
        x = layers.Conv2D(64, 3, activation="relu")(inputs)
        x = layers.Conv2D(64, 3, activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(10, activation="softmax")(x)
        model = models.Model(inputs, x)

        # Export without optimization
        filepath_no_opt = os.path.join(
            self.get_temp_dir(), "model_no_opt.tflite"
        )
        model.export(filepath_no_opt, format="litert")

        # Export with optimization
        filepath_with_opt = os.path.join(
            self.get_temp_dir(), "model_with_opt.tflite"
        )
        model.export(
            filepath_with_opt,
            format="litert",
            optimizations=[tensorflow.lite.Optimize.DEFAULT],
        )

        # Optimized model should be smaller
        size_no_opt = os.path.getsize(filepath_no_opt)
        size_with_opt = os.path.getsize(filepath_with_opt)

        self.assertLess(
            size_with_opt,
            size_no_opt,
            f"Optimized model ({size_with_opt} bytes) should be smaller "
            f"than non-optimized ({size_no_opt} bytes)",
        )

        # Typically expect ~75% size reduction with quantization
        reduction_ratio = size_with_opt / size_no_opt
        self.assertLess(
            reduction_ratio,
            0.5,  # Should be less than 50% of original size
            f"Expected significant size reduction, got {reduction_ratio:.2%}",
        )

    def test_signature_def_with_named_model(self):
        """Test that exported models have SignatureDef with input names."""
        if LiteRTInterpreter is None:
            self.skipTest("No LiteRT interpreter available")

        # Build a model with explicit layer names
        inputs = layers.Input(shape=(10,), name="feature_input")
        x = layers.Dense(32, activation="relu", name="encoder")(inputs)
        x = layers.Dense(16, activation="relu", name="bottleneck")(x)
        outputs = layers.Dense(
            1, activation="sigmoid", name="prediction_output"
        )(x)
        model = models.Model(inputs=inputs, outputs=outputs, name="named_model")

        temp_filepath = os.path.join(self.get_temp_dir(), "named_model.tflite")

        # Export the model
        model.export(temp_filepath, format="litert")
        self.assertTrue(os.path.exists(temp_filepath))

        # Load and check SignatureDef
        interpreter = LiteRTInterpreter(model_path=temp_filepath)
        interpreter.allocate_tensors()

        # Get SignatureDef information
        signature_defs = interpreter.get_signature_list()
        self.assertIn("serving_default", signature_defs)

        serving_sig = signature_defs["serving_default"]
        sig_inputs = serving_sig.get("inputs", [])
        sig_outputs = serving_sig.get("outputs", [])

        # Verify SignatureDef has inputs and outputs
        self.assertGreater(
            len(sig_inputs), 0, "Should have at least one input in SignatureDef"
        )
        self.assertGreater(
            len(sig_outputs),
            0,
            "Should have at least one output in SignatureDef",
        )

        # Verify input names are preserved (they should match Keras input names)
        self.assertIn(
            "feature_input",
            sig_inputs,
            f"Input name 'feature_input' should be in SignatureDef inputs: "
            f"{sig_inputs}",
        )

        # Verify inference works using signature runner
        batch_size = 1
        ref_input = np.random.normal(size=(batch_size, 10)).astype("float32")
        ref_output = _convert_to_numpy(model(ref_input))

        # Note: For single-output Functional models, Keras returns a tensor
        # (not dict). SignatureDef will have generic output names like
        # 'output_0'.
        # Only multi-output models or models with explicit dict returns have
        # named outputs

        # Test inference using signature runner for better output name handling
        signature_runner = interpreter.get_signature_runner("serving_default")
        sig_output = signature_runner(feature_input=ref_input)

        # sig_output should be a dict with meaningful output names
        self.assertIsInstance(sig_output, dict)
        self.assertGreater(
            len(sig_output), 0, "Should have at least one output"
        )

        # For single output, extract the value
        if len(sig_output) == 1:
            litert_output = list(sig_output.values())[0]
        else:
            litert_output = list(sig_output.values())

        self.assertAllClose(ref_output, litert_output, atol=1e-4, rtol=1e-4)

    def test_signature_def_with_functional_model(self):
        """Test that SignatureDef preserves input/output names for
        Functional models."""
        if LiteRTInterpreter is None:
            self.skipTest("No LiteRT interpreter available")

        # Create a Functional model with named inputs and outputs
        inputs = layers.Input(shape=(10,), name="input_layer")
        x = layers.Dense(32, activation="relu", name="hidden_layer")(inputs)
        outputs = layers.Dense(1, activation="sigmoid", name="output_layer")(x)
        model = models.Model(
            inputs=inputs, outputs=outputs, name="functional_model"
        )

        temp_filepath = os.path.join(
            self.get_temp_dir(), "functional_model.tflite"
        )

        # Export the model
        model.export(temp_filepath, format="litert")
        self.assertTrue(os.path.exists(temp_filepath))

        # Load and check SignatureDef
        interpreter = LiteRTInterpreter(model_path=temp_filepath)
        interpreter.allocate_tensors()

        # Get SignatureDef information
        signature_defs = interpreter.get_signature_list()
        self.assertIn("serving_default", signature_defs)

        serving_sig = signature_defs["serving_default"]
        sig_inputs = serving_sig.get("inputs", [])
        sig_outputs = serving_sig.get("outputs", [])

        # Verify SignatureDef has inputs and outputs
        self.assertGreater(
            len(sig_inputs), 0, "Should have at least one input in SignatureDef"
        )
        self.assertGreater(
            len(sig_outputs),
            0,
            "Should have at least one output in SignatureDef",
        )

        # Verify that input names are preserved
        self.assertIn(
            "input_layer",
            sig_inputs,
            f"Input name 'input_layer' should be in SignatureDef inputs: "
            f"{sig_inputs}",
        )

        # Test inference using signature runner for named outputs
        batch_size = 1
        ref_input = np.random.normal(size=(batch_size, 10)).astype("float32")
        ref_output = _convert_to_numpy(model(ref_input))

        # Use signature runner to get outputs with meaningful names
        signature_runner = interpreter.get_signature_runner("serving_default")
        sig_output = signature_runner(input_layer=ref_input)

        # sig_output should be a dict with output names
        self.assertIsInstance(sig_output, dict)
        self.assertGreater(
            len(sig_output), 0, "Should have at least one output"
        )

        # For single output, TFLite typically uses generic names like 'output_0'
        # Extract the single output value
        if len(sig_output) == 1:
            litert_output = list(sig_output.values())[0]
        else:
            litert_output = list(sig_output.values())

        self.assertAllClose(ref_output, litert_output, atol=1e-4, rtol=1e-4)

    def test_signature_def_with_multi_input_model(self):
        """Test that SignatureDef preserves names for multi-input models."""
        if LiteRTInterpreter is None:
            self.skipTest("No LiteRT interpreter available")

        # Create a multi-input model
        input1 = layers.Input(shape=(10,), name="input_1")
        input2 = layers.Input(shape=(5,), name="input_2")
        concat = layers.Concatenate(name="concat_layer")([input1, input2])
        outputs = layers.Dense(1, activation="sigmoid", name="output")(concat)
        model = models.Model(
            inputs=[input1, input2], outputs=outputs, name="multi_input_model"
        )

        temp_filepath = os.path.join(
            self.get_temp_dir(), "multi_input_model.tflite"
        )

        # Export the model
        model.export(temp_filepath, format="litert")
        self.assertTrue(os.path.exists(temp_filepath))

        # Load and check SignatureDef
        interpreter = LiteRTInterpreter(model_path=temp_filepath)
        interpreter.allocate_tensors()

        # Get SignatureDef information
        signature_defs = interpreter.get_signature_list()
        self.assertIn("serving_default", signature_defs)

        serving_sig = signature_defs["serving_default"]
        sig_inputs = serving_sig.get("inputs", [])
        sig_outputs = serving_sig.get("outputs", [])

        # Verify SignatureDef has correct number of inputs and outputs
        self.assertEqual(
            len(sig_inputs), 2, "Should have 2 inputs in SignatureDef"
        )
        self.assertGreater(
            len(sig_outputs),
            0,
            "Should have at least one output in SignatureDef",
        )

        # Verify that input names are preserved
        self.assertIn(
            "input_1",
            sig_inputs,
            f"Input name 'input_1' should be in SignatureDef inputs: "
            f"{sig_inputs}",
        )
        self.assertIn(
            "input_2",
            sig_inputs,
            f"Input name 'input_2' should be in SignatureDef inputs: "
            f"{sig_inputs}",
        )

        # Test inference using signature runner
        batch_size = 1
        ref_input1 = np.random.normal(size=(batch_size, 10)).astype("float32")
        ref_input2 = np.random.normal(size=(batch_size, 5)).astype("float32")
        ref_inputs = [ref_input1, ref_input2]
        ref_output = _convert_to_numpy(model(ref_inputs))

        # Use signature runner with named inputs
        signature_runner = interpreter.get_signature_runner("serving_default")
        sig_output = signature_runner(input_1=ref_input1, input_2=ref_input2)

        # sig_output should be a dict with output names
        self.assertIsInstance(sig_output, dict)
        self.assertGreater(
            len(sig_output), 0, "Should have at least one output"
        )

        # For single output, TFLite uses generic names like 'output_0'
        # Extract the single output value
        if len(sig_output) == 1:
            litert_output = list(sig_output.values())[0]
        else:
            litert_output = list(sig_output.values())

        self.assertAllClose(ref_output, litert_output, atol=1e-4, rtol=1e-4)

    def test_signature_def_with_multi_output_model(self):
        """Test that SignatureDef handles multi-output models correctly."""
        if LiteRTInterpreter is None:
            self.skipTest("No LiteRT interpreter available")

        # Create a multi-output model
        inputs = layers.Input(shape=(10,), name="input_layer")
        x = layers.Dense(32, activation="relu", name="shared_layer")(inputs)
        output1 = layers.Dense(1, activation="sigmoid", name="output_1")(x)
        output2 = layers.Dense(2, activation="softmax", name="output_2")(x)
        model = models.Model(
            inputs=inputs, outputs=[output1, output2], name="multi_output_model"
        )

        temp_filepath = os.path.join(
            self.get_temp_dir(), "multi_output_model.tflite"
        )

        # Export the model
        model.export(temp_filepath, format="litert")
        self.assertTrue(os.path.exists(temp_filepath))

        # Load and check SignatureDef
        interpreter = LiteRTInterpreter(model_path=temp_filepath)
        interpreter.allocate_tensors()

        # Get SignatureDef information
        signature_defs = interpreter.get_signature_list()
        self.assertIn("serving_default", signature_defs)

        serving_sig = signature_defs["serving_default"]
        sig_inputs = serving_sig.get("inputs", [])
        sig_outputs = serving_sig.get("outputs", [])

        # Verify SignatureDef structure
        self.assertGreater(
            len(sig_inputs), 0, "Should have at least one input in SignatureDef"
        )
        self.assertEqual(
            len(sig_outputs), 2, "Should have 2 outputs in SignatureDef"
        )

        # Test inference using signature runner
        batch_size = 1
        ref_input = np.random.normal(size=(batch_size, 10)).astype("float32")
        ref_outputs = _convert_to_numpy(model(ref_input))

        # Use signature runner
        signature_runner = interpreter.get_signature_runner("serving_default")
        sig_output = signature_runner(input_layer=ref_input)

        # sig_output should be a dict with output names
        self.assertIsInstance(sig_output, dict)
        self.assertEqual(len(sig_output), 2, "Should have 2 outputs")

        # Note: TFLite uses generic names like 'output_0', 'output_1' for
        # SignatureDef outputs. These don't match the Keras layer names
        # ('output_1', 'output_2') - this is expected. The names come from
        # TensorFlow's symbolic tracing, not from our exporter code.
        # Verify outputs match by position
        sig_output_values = list(sig_output.values())
        for i, ref_out in enumerate(ref_outputs):
            self.assertAllClose(
                ref_out, sig_output_values[i], atol=1e-4, rtol=1e-4
            )

    def test_dict_input_adapter_creation(self):
        """Test that dict input adapter is created and works correctly."""

        # Create a model with dictionary inputs
        input1 = layers.Input(shape=(10,), name="x")
        input2 = layers.Input(shape=(10,), name="y")
        output = layers.Add()([input1, input2])
        model = models.Model(inputs={"x": input1, "y": input2}, outputs=output)

        temp_filepath = os.path.join(
            self.get_temp_dir(), "dict_adapter_model.tflite"
        )

        # Export with verbose to verify adapter creation messages
        model.export(temp_filepath, format="litert", verbose=True)

        # Verify the file was created
        self.assertTrue(os.path.exists(temp_filepath))

        # Load and test the model
        interpreter = LiteRTInterpreter(model_path=temp_filepath)
        interpreter.allocate_tensors()

        # Check input details - should have 2 inputs in list form
        input_details = interpreter.get_input_details()
        self.assertEqual(len(input_details), 2)

        # Test inference
        batch_size = 1
        x_val = np.random.normal(size=(batch_size, 10)).astype("float32")
        y_val = np.random.normal(size=(batch_size, 10)).astype("float32")

        ref_output = _convert_to_numpy(
            model(
                {
                    "x": ops.convert_to_tensor(x_val),
                    "y": ops.convert_to_tensor(y_val),
                }
            )
        )

        # Set inputs as list (adapter converts list to dict internally)
        _set_interpreter_inputs(interpreter, [x_val, y_val])
        interpreter.invoke()
        litert_output = _get_interpreter_outputs(interpreter)

        self.assertAllClose(ref_output, litert_output, atol=1e-4, rtol=1e-4)

    def test_dict_input_signature_inference(self):
        """Test automatic inference of dict input signatures."""

        # Create a model with dictionary inputs (without calling it first)
        input1 = layers.Input(shape=(5,), name="feature_a")
        input2 = layers.Input(shape=(3,), name="feature_b")
        concat = layers.Concatenate()([input1, input2])
        output = layers.Dense(1)(concat)
        model = models.Model(
            inputs={"feature_a": input1, "feature_b": input2}, outputs=output
        )

        temp_filepath = os.path.join(
            self.get_temp_dir(), "inferred_dict_model.tflite"
        )

        # Export without providing input_signature - should be inferred
        model.export(temp_filepath, format="litert")

        # Verify successful export
        self.assertTrue(os.path.exists(temp_filepath))

        # Load and verify structure
        interpreter = LiteRTInterpreter(model_path=temp_filepath)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        self.assertEqual(len(input_details), 2)

        # Verify shapes match expected
        shapes = [tuple(d["shape"][1:]) for d in input_details]
        self.assertIn((5,), shapes)
        self.assertIn((3,), shapes)

    def test_dict_input_with_custom_signature(self):
        """Test dict input export with custom input signature."""

        # Create model with dict inputs
        input1 = layers.Input(shape=(10,), name="input_x")
        input2 = layers.Input(shape=(10,), name="input_y")
        output = layers.Multiply()([input1, input2])
        model = models.Model(
            inputs={"input_x": input1, "input_y": input2}, outputs=output
        )

        temp_filepath = os.path.join(
            self.get_temp_dir(), "dict_custom_sig_model.tflite"
        )

        # Provide custom dict input signature
        input_signature = {
            "input_x": layers.InputSpec(shape=(None, 10), dtype="float32"),
            "input_y": layers.InputSpec(shape=(None, 10), dtype="float32"),
        }

        model.export(
            temp_filepath, format="litert", input_signature=input_signature
        )

        # Verify export
        self.assertTrue(os.path.exists(temp_filepath))

        interpreter = LiteRTInterpreter(model_path=temp_filepath)
        interpreter.allocate_tensors()

        # Test inference
        batch_size = 1
        x_val = np.random.normal(size=(batch_size, 10)).astype("float32")
        y_val = np.random.normal(size=(batch_size, 10)).astype("float32")

        ref_output = _convert_to_numpy(
            model(
                {
                    "input_x": ops.convert_to_tensor(x_val),
                    "input_y": ops.convert_to_tensor(y_val),
                }
            )
        )

        _set_interpreter_inputs(interpreter, [x_val, y_val])
        interpreter.invoke()
        litert_output = _get_interpreter_outputs(interpreter)

        self.assertAllClose(ref_output, litert_output, atol=1e-4, rtol=1e-4)

    def test_dict_input_numerical_accuracy(self):
        """Test numerical accuracy of dict input models with complex ops."""

        # Create a more complex model with dict inputs
        input1 = layers.Input(shape=(20,), name="tokens")
        input2 = layers.Input(shape=(20,), name="mask")

        # Apply some transformations
        x1 = layers.Dense(16, activation="relu")(input1)
        x2 = layers.Dense(16, activation="relu")(input2)

        # Combine
        combined = layers.Multiply()([x1, x2])
        output = layers.Dense(1, activation="sigmoid")(combined)

        model = models.Model(
            inputs={"tokens": input1, "mask": input2}, outputs=output
        )

        temp_filepath = os.path.join(
            self.get_temp_dir(), "dict_numerical_model.tflite"
        )

        model.export(temp_filepath, format="litert")

        # Test with multiple samples
        batch_size = 1
        tokens_val = np.random.normal(size=(batch_size, 20)).astype("float32")
        mask_val = np.random.normal(size=(batch_size, 20)).astype("float32")

        ref_output = _convert_to_numpy(
            model(
                {
                    "tokens": ops.convert_to_tensor(tokens_val),
                    "mask": ops.convert_to_tensor(mask_val),
                }
            )
        )

        interpreter = LiteRTInterpreter(model_path=temp_filepath)
        interpreter.allocate_tensors()
        _set_interpreter_inputs(interpreter, [tokens_val, mask_val])
        interpreter.invoke()
        litert_output = _get_interpreter_outputs(interpreter)

        # Should have good numerical accuracy
        self.assertAllClose(ref_output, litert_output, atol=1e-5, rtol=1e-5)

    def test_dict_input_preserves_variable_sharing(self):
        """Test that adapter preserves variable sharing from original model."""

        # Create model with shared layers
        shared_dense = layers.Dense(8, activation="relu")

        input1 = layers.Input(shape=(10,), name="branch_a")
        input2 = layers.Input(shape=(10,), name="branch_b")

        # Both inputs go through same shared layer
        x1 = shared_dense(input1)
        x2 = shared_dense(input2)

        output = layers.Add()([x1, x2])
        model = models.Model(
            inputs={"branch_a": input1, "branch_b": input2}, outputs=output
        )

        # Train briefly to ensure weights are meaningful
        model.compile(optimizer="adam", loss="mse")
        x_train = {
            "branch_a": np.random.normal(size=(5, 10)).astype("float32"),
            "branch_b": np.random.normal(size=(5, 10)).astype("float32"),
        }
        y_train = np.random.normal(size=(5, 8)).astype("float32")
        model.fit(x_train, y_train, epochs=1, verbose=0)

        temp_filepath = os.path.join(
            self.get_temp_dir(), "dict_shared_vars_model.tflite"
        )

        model.export(temp_filepath, format="litert")

        # Verify export works and inference matches
        interpreter = LiteRTInterpreter(model_path=temp_filepath)
        interpreter.allocate_tensors()

        batch_size = 1
        a_val = np.random.normal(size=(batch_size, 10)).astype("float32")
        b_val = np.random.normal(size=(batch_size, 10)).astype("float32")

        ref_output = _convert_to_numpy(
            model(
                {
                    "branch_a": ops.convert_to_tensor(a_val),
                    "branch_b": ops.convert_to_tensor(b_val),
                }
            )
        )

        _set_interpreter_inputs(interpreter, [a_val, b_val])
        interpreter.invoke()
        litert_output = _get_interpreter_outputs(interpreter)

        self.assertAllClose(ref_output, litert_output, atol=1e-4, rtol=1e-4)

    def test_dict_input_multi_output_model(self):
        """Test dict input model with multiple outputs exports successfully."""

        # Create model with dict inputs and multiple outputs
        input1 = layers.Input(shape=(10,), name="feature_1")
        input2 = layers.Input(shape=(10,), name="feature_2")

        # Two output branches
        output1 = layers.Dense(5, name="output_a")(input1)
        output2 = layers.Dense(3, name="output_b")(input2)

        model = models.Model(
            inputs={"feature_1": input1, "feature_2": input2},
            outputs=[output1, output2],
        )

        temp_filepath = os.path.join(
            self.get_temp_dir(), "dict_multi_output_model.tflite"
        )

        # Main test: export should succeed with dict inputs + multi outputs
        model.export(temp_filepath, format="litert")

        # Verify file was created
        self.assertTrue(os.path.exists(temp_filepath))

        # Verify structure
        interpreter = LiteRTInterpreter(model_path=temp_filepath)
        interpreter.allocate_tensors()

        # Should have 2 inputs (from dict)
        input_details = interpreter.get_input_details()
        self.assertEqual(len(input_details), 2)

        # Should have 2 outputs
        output_details = interpreter.get_output_details()
        self.assertEqual(len(output_details), 2)

        # Verify shapes
        output_shapes = [tuple(d["shape"][1:]) for d in output_details]
        self.assertIn((5,), output_shapes)
        self.assertIn((3,), output_shapes)
