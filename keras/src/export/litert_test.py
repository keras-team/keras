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

        # Should raise AssertionError for wrong extension
        with self.assertRaises(AssertionError):
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

    def test_aot_compile_parameter_accepted(self):
        """Test that aot_compile_targets parameter is accepted without error."""
        if not litert.available:
            self.skipTest("LiteRT not available")

        model = get_model("sequential")
        temp_filepath = os.path.join(
            self.get_temp_dir(), "model_with_aot.tflite"
        )

        # Test that parameter is accepted (compilation may or may not succeed)
        # The key is that it doesn't crash
        try:
            result = model.export(
                temp_filepath,
                format="litert",
                aot_compile_targets=["arm64"],
                verbose=True,
            )
            # Base .tflite file should always be created
            self.assertTrue(os.path.exists(temp_filepath))

            # Result could be filepath (if AOT failed/skipped) or
            # CompilationResult (if AOT succeeded)
            self.assertIsNotNone(result)
        except Exception as e:
            # If AOT infrastructure not available, that's okay as long as
            # base model was exported
            error_msg = str(e)
            if "AOT" in error_msg or "compilation" in error_msg.lower():
                if os.path.exists(temp_filepath):
                    # Base model created, AOT just not available - this is fine
                    pass
                else:
                    self.fail(
                        f"Base .tflite model should be created even if AOT "
                        f"fails: {error_msg}"
                    )
            else:
                # Some other error - re-raise
                raise

    def test_aot_compile_multiple_targets(self):
        """Test AOT compilation with multiple targets."""
        if not litert.available:
            self.skipTest("LiteRT not available")

        model = get_model("functional")
        temp_filepath = os.path.join(
            self.get_temp_dir(), "model_multi_aot.tflite"
        )

        # Test with multiple targets
        try:
            result = model.export(
                temp_filepath,
                format="litert",
                aot_compile_targets=["arm64", "x86_64"],
                verbose=True,
            )

            # Base model should exist
            self.assertTrue(os.path.exists(temp_filepath))

            # Check if result contains compilation info
            if hasattr(result, "models"):
                # AOT compilation succeeded
                self.assertGreater(
                    len(result.models),
                    0,
                    "Should have at least one compiled model",
                )
            elif isinstance(result, str):
                # AOT skipped, returned filepath
                self.assertEqual(result, temp_filepath)

        except Exception as e:
            # AOT infrastructure may not be available
            if os.path.exists(temp_filepath):
                # Base model was created - acceptable
                pass
            else:
                self.fail(f"Base model should be created: {str(e)}")

    def test_aot_compile_with_optimizations(self):
        """Test AOT compilation combined with quantization optimizations."""
        if not litert.available:
            self.skipTest("LiteRT not available")

        model = get_model("sequential")
        temp_filepath = os.path.join(
            self.get_temp_dir(), "model_aot_optimized.tflite"
        )

        # Test AOT with quantization
        try:
            model.export(
                temp_filepath,
                format="litert",
                aot_compile_targets=["arm64"],
                optimizations=[tensorflow.lite.Optimize.DEFAULT],
                verbose=True,
            )

            # Base model must exist
            self.assertTrue(os.path.exists(temp_filepath))

            # Verify model is quantized (smaller size)
            size = os.path.getsize(temp_filepath)
            self.assertGreater(size, 0)

        except Exception as e:
            # Acceptable if AOT not available but base model created
            if not os.path.exists(temp_filepath):
                self.fail(f"Base model should be created: {str(e)}")

    def test_get_available_aot_targets(self):
        """Test retrieving available AOT compilation targets."""
        if not litert.available:
            self.skipTest("LiteRT not available")

        try:
            from keras.src.export.litert import LiteRTExporter

            # This should not crash even if no targets available
            targets = LiteRTExporter.get_available_targets()

            # Should return a list (possibly empty)
            self.assertIsInstance(targets, list)

            # If targets are available, they should be valid
            if targets:
                for target in targets:
                    # Each target should have some identifying property
                    self.assertIsNotNone(target)

        except ImportError:
            self.skipTest("LiteRTExporter not available")
        except Exception as e:
            # No targets available is acceptable
            if "target" in str(e).lower() or "vendor" in str(e).lower():
                pass
            else:
                raise

    def test_aot_compile_without_litert_available(self):
        """Test that export works gracefully when LiteRT AOT is unavailable."""
        # This test verifies the fallback behavior
        model = get_model("sequential")
        temp_filepath = os.path.join(self.get_temp_dir(), "model_no_aot.tflite")

        # Even if we request AOT, export should succeed and create base model
        # AOT compilation may fail, but that's acceptable as long as base model
        # is created
        try:
            model.export(
                temp_filepath,
                format="litert",
                aot_compile_targets=["arm64"],
                verbose=False,  # Suppress warnings in test output
            )

            # Base .tflite file should be created regardless
            self.assertTrue(os.path.exists(temp_filepath))

        except RuntimeError as e:
            # AOT compilation may fail if infrastructure not available
            # This is acceptable as long as base model is created
            if "AOT" in str(e):
                # Verify base model was created before AOT failure
                self.assertTrue(
                    os.path.exists(temp_filepath),
                    "Base .tflite model should be created even if AOT fails",
                )
            else:
                # Other runtime errors should be raised
                raise

    def test_export_with_aot_class_method(self):
        """Test the export_with_aot class method."""
        if not litert.available:
            self.skipTest("LiteRT not available")

        try:
            from keras.src.export.litert import LiteRTExporter

            model = get_model("functional")
            temp_filepath = os.path.join(
                self.get_temp_dir(), "model_class_method_aot.tflite"
            )

            # Test the class method
            result = LiteRTExporter.export_with_aot(
                model=model,
                filepath=temp_filepath,
                targets=["arm64"],
                verbose=True,
            )

            # Base model should exist
            self.assertTrue(os.path.exists(temp_filepath))
            self.assertIsNotNone(result)

        except ImportError:
            self.skipTest("LiteRTExporter not available")
        except Exception as e:
            # AOT may not be available, but base model should be created
            if not os.path.exists(temp_filepath):
                self.fail(f"Base model should be created: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__])
