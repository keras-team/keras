"""Tests for LiteRT export via the PyTorch backend (litert-torch)."""

import contextlib
import io
import os

import jax
import numpy as np
import pytest
import tensorflow as tf
import torch

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.export.litert import _preserve_jax_x64_state

# Importing `litert_torch` has the module-level side effect of calling
# `jax.config.update("jax_enable_x64", True)` (from
# `litert_torch/backend/jax_bridge/_wrap.py`). We wrap the import with
# `_preserve_jax_x64_state` so the rest of the test session sees the
# original behavior.
with _preserve_jax_x64_state():
    try:
        import litert_torch  # noqa: F401

        _HAS_LITERT_TORCH = True
    except (ImportError, ModuleNotFoundError):
        _HAS_LITERT_TORCH = False


def _has_litert_torch():
    return _HAS_LITERT_TORCH


def _get_interpreter(filepath):
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=filepath)
    interp.allocate_tensors()
    return interp


def _run_litert_inference(interpreter, input_arrays):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for detail, arr in zip(input_details, input_arrays):
        interpreter.set_tensor(detail["index"], arr)

    interpreter.invoke()

    outputs = [interpreter.get_tensor(d["index"]) for d in output_details]
    return outputs if len(outputs) > 1 else outputs[0]


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Requires torch backend"
)
@pytest.mark.skipif(
    not _has_litert_torch(), reason="Requires litert-torch and torch"
)
class LiteRTTorchExportTest(testing.TestCase):
    LITERT_ATOL = 1e-4
    TORCH_ATOL = 1e-5

    def _verify_litert_export(
        self, model, ref_input, filepath=None, **export_kwargs
    ):
        if filepath is None:
            filepath = os.path.join(self.get_temp_dir(), "model.tflite")

        keras_output = backend.convert_to_numpy(model(ref_input))

        model.export(filepath, format="litert", **export_kwargs)
        self.assertTrue(os.path.exists(filepath))

        input_arrays = ref_input if isinstance(ref_input, list) else [ref_input]
        litert_output = _run_litert_inference(
            _get_interpreter(filepath), input_arrays
        )

        self.assertAllClose(keras_output, litert_output, atol=self.LITERT_ATOL)

        return keras_output, litert_output

    def test_sequential_model(self):
        model = models.Sequential(
            [
                layers.Dense(16, activation="relu", input_shape=(10,)),
                layers.Dense(1),
            ]
        )

        x = np.random.normal(size=(1, 10)).astype("float32")
        self._verify_litert_export(model, x)

    def test_functional_model(self):
        inp = layers.Input(shape=(10,))
        x = layers.Dense(16, activation="relu")(inp)
        out = layers.Dense(1)(x)
        model = models.Model(inputs=inp, outputs=out)

        x = np.random.normal(size=(1, 10)).astype("float32")
        self._verify_litert_export(model, x)

    def test_subclass_model(self):
        class TinyModel(models.Model):
            def __init__(self):
                super().__init__()
                self.dense1 = layers.Dense(16, activation="relu")
                self.dense2 = layers.Dense(1)

            def call(self, x):
                return self.dense2(self.dense1(x))

        model = TinyModel()
        model(np.zeros((1, 10), dtype="float32"))  # Build

        x = np.random.normal(size=(1, 10)).astype("float32")
        self._verify_litert_export(model, x)

    def test_conv_model(self):
        model = models.Sequential(
            [
                layers.Conv2D(8, 3, activation="relu", input_shape=(8, 8, 3)),
                layers.Flatten(),
                layers.Dense(1),
            ]
        )

        x = np.random.normal(size=(1, 8, 8, 3)).astype("float32")
        self._verify_litert_export(model, x)

    def test_multi_input_model(self):
        inp_a = layers.Input(shape=(10,), name="input_a")
        inp_b = layers.Input(shape=(10,), name="input_b")
        merged = layers.Concatenate()([inp_a, inp_b])
        out = layers.Dense(1)(merged)
        model = models.Model(inputs=[inp_a, inp_b], outputs=out)

        a = np.random.normal(size=(1, 10)).astype("float32")
        b = np.random.normal(size=(1, 10)).astype("float32")
        self._verify_litert_export(model, [a, b])

    def test_multi_output_model(self):
        inp = layers.Input(shape=(10,))
        x = layers.Dense(16, activation="relu")(inp)
        out_a = layers.Dense(1, name="out_a")(x)
        out_b = layers.Dense(1, name="out_b")(x)
        model = models.Model(inputs=inp, outputs=[out_a, out_b])

        tflite_path = os.path.join(self.get_temp_dir(), "multi_out.tflite")
        x = np.random.normal(size=(1, 10)).astype("float32")

        keras_outs = model(x)
        if not isinstance(keras_outs, (list, tuple)):
            keras_outs = [keras_outs]
        keras_outs = [backend.convert_to_numpy(o) for o in keras_outs]

        model.export(tflite_path, format="litert")
        self.assertTrue(os.path.exists(tflite_path))

        litert_outs = _run_litert_inference(_get_interpreter(tflite_path), [x])
        if not isinstance(litert_outs, list):
            litert_outs = [litert_outs]

        self.assertEqual(len(keras_outs), len(litert_outs))
        for k_out in keras_outs:
            matched = False
            for l_out in litert_outs:
                if k_out.shape == l_out.shape and np.allclose(
                    k_out, l_out, atol=1e-3
                ):
                    matched = True
                    break
            self.assertTrue(
                matched, f"No matching LiteRT output for keras output {k_out}"
            )

    def test_torch_export_numeric_parity(self):
        model = models.Sequential(
            [
                layers.Dense(16, activation="relu", input_shape=(10,)),
                layers.Dense(1),
            ]
        )

        pt2_path = os.path.join(self.get_temp_dir(), "model.pt2")
        model.export(pt2_path, format="torch")

        x_np = np.random.normal(size=(1, 10)).astype("float32")
        keras_out = backend.convert_to_numpy(model(x_np))

        program = torch.export.load(pt2_path)
        module = program.module()
        device = next(module.parameters()).device
        pt2_out = backend.convert_to_numpy(
            module(torch.tensor(x_np, device=device))
        )

        self.assertAllClose(keras_out, pt2_out, atol=self.TORCH_ATOL)

    def test_full_pipeline_numeric_parity(self):
        model = models.Sequential(
            [
                layers.Dense(16, activation="relu", input_shape=(10,)),
                layers.Dense(1),
            ]
        )

        pt2_path = os.path.join(self.get_temp_dir(), "model.pt2")
        tflite_path = os.path.join(self.get_temp_dir(), "model.tflite")

        model.export(pt2_path, format="torch")
        model.export(tflite_path, format="litert")

        x_np = np.random.normal(size=(1, 10)).astype("float32")
        keras_out = backend.convert_to_numpy(model(x_np))

        program = torch.export.load(pt2_path)
        module = program.module()
        device = next(module.parameters()).device
        pt2_out = backend.convert_to_numpy(
            module(torch.tensor(x_np, device=device))
        )

        # LiteRT .tflite
        litert_out = _run_litert_inference(
            _get_interpreter(tflite_path), [x_np]
        )

        self.assertAllClose(
            keras_out, pt2_out, atol=self.TORCH_ATOL, msg="Keras vs Torch"
        )
        self.assertAllClose(
            keras_out, litert_out, atol=self.LITERT_ATOL, msg="Keras vs LiteRT"
        )

    def test_export_with_optimizations_default(self):
        model = models.Sequential(
            [
                layers.Dense(16, activation="relu", input_shape=(10,)),
                layers.Dense(8, activation="relu"),
                layers.Dense(1),
            ]
        )

        path = os.path.join(self.get_temp_dir(), "quantized.tflite")

        model.export(
            path,
            format="litert",
            optimizations=[tf.lite.Optimize.DEFAULT],
        )

        self.assertTrue(os.path.exists(path))

        # Verify inference still works post-quantization
        x_np = np.random.normal(size=(1, 10)).astype("float32")
        keras_out = backend.convert_to_numpy(model(x_np))
        litert_out = _run_litert_inference(_get_interpreter(path), [x_np])

        # Quantized model has reduced precision
        self.assertAllClose(keras_out, litert_out, atol=1e-1, rtol=1e-1)

    def test_export_with_direct_quant_config(self):
        """Test passing litert_torch quant_config directly."""

        try:
            from litert_torch.quantize.pt2e_quantizer import PT2EQuantizer
            from litert_torch.quantize.pt2e_quantizer import (
                get_symmetric_quantization_config,
            )
            from litert_torch.quantize.quant_config import QuantConfig
        except ImportError:
            self.skipTest("litert_torch quantization modules unavailable")

        model = models.Sequential(
            [
                layers.Dense(16, activation="relu", input_shape=(10,)),
                layers.Dense(1),
            ]
        )

        path = os.path.join(self.get_temp_dir(), "quant_direct.tflite")

        # Create quantization config directly
        quant_config_obj = get_symmetric_quantization_config(
            is_per_channel=True, is_dynamic=False, is_qat=False
        )
        quantizer = PT2EQuantizer()
        quantizer.set_global(quant_config_obj)
        quant_cfg = QuantConfig(pt2e_quantizer=quantizer)

        model.export(path, format="litert", quant_config=quant_cfg)
        self.assertTrue(os.path.exists(path))

    def test_litert_interpreter_runtime_resize(self):
        """Test that the exported model supports runtime input resizing."""

        model = models.Sequential(
            [
                layers.Dense(8, activation="relu", input_shape=(10,)),
                layers.Dense(1),
            ]
        )

        path = os.path.join(self.get_temp_dir(), "dynamic.tflite")

        # Export with static shape (batch=1)
        model.export(path, format="litert")

        # Test runtime dynamic shapes by resizing interpreter inputs
        interpreter = _get_interpreter(path)
        input_details = interpreter.get_input_details()

        # Test with different batch sizes at runtime
        for batch_size in [1, 2, 5]:
            # Resize the input tensor at runtime
            interpreter.resize_tensor_input(
                input_details[0]["index"], [batch_size, 10]
            )
            interpreter.allocate_tensors()

            # Run inference with the new batch size
            x = np.random.normal(size=(batch_size, 10)).astype("float32")

            # Set input and invoke
            interpreter.set_tensor(input_details[0]["index"], x)
            interpreter.invoke()

            # Get output
            output_details = interpreter.get_output_details()
            output = interpreter.get_tensor(output_details[0]["index"])

            # Verify output has correct batch dimension
            self.assertEqual(output.shape[0], batch_size)

    def test_export_with_multiple_kwargs(self):
        """Test export with multiple litert_torch kwargs."""

        model = models.Sequential(
            [
                layers.Dense(8, activation="relu", input_shape=(10,)),
                layers.Dense(1),
            ]
        )

        path = os.path.join(self.get_temp_dir(), "multi_kwargs.tflite")

        model.export(
            path,
            format="litert",
            strict_export=False,
            optimizations=[
                tf.lite.Optimize.DEFAULT
            ],  # Translated to quant_config
            # representative_dataset would be ignored (TFLite-specific)
        )

        self.assertTrue(os.path.exists(path))

    def test_export_with_additional_litert_torch_kwargs(self):
        model = models.Sequential(
            [
                layers.Dense(8, activation="relu", input_shape=(10,)),
                layers.Dense(1),
            ]
        )

        path = os.path.join(self.get_temp_dir(), "extra_kwargs.tflite")

        model.export(
            path,
            format="litert",
            lightweight_conversion=True,
            runtime_constant_folding=True,
            enable_x64=False,
        )

        self.assertTrue(os.path.exists(path))

    def test_export_with_unsupported_kwargs(self):
        model = models.Sequential([layers.Dense(1, input_shape=(5,))])
        path = os.path.join(self.get_temp_dir(), "unsupported_kwarg.tflite")

        with self.assertRaisesRegex(
            ValueError,
            "Unsupported arguments for LiteRT export with the PyTorch backend",
        ):
            model.export(
                path,
                format="litert",
                allow_custom_ops=True,
            )

    def test_export_preserves_model_device(self):
        """Verify model tensors return to original device after export."""

        # Create model on CPU initially
        model = models.Sequential(
            [
                layers.Dense(8, activation="relu", input_shape=(5,)),
                layers.Dense(1),
            ]
        )
        model.build((None, 5))

        # Get original devices
        original_devices = [str(p.device) for _, p in model.named_parameters()]

        path = os.path.join(self.get_temp_dir(), "device_test.tflite")
        model.export(path, format="litert")

        # Check devices are preserved
        final_devices = [str(p.device) for _, p in model.named_parameters()]
        self.assertEqual(original_devices, final_devices)

    def test_export_with_verbose(self):
        """Test verbose output during export."""

        model = models.Sequential([layers.Dense(1, input_shape=(5,))])
        path = os.path.join(self.get_temp_dir(), "verbose_test.tflite")

        # Should print message to stdout
        model.export(path, format="litert", verbose=True)
        self.assertTrue(os.path.exists(path))

        # Verify inference works
        x_np = np.random.normal(size=(1, 5)).astype("float32")
        keras_out = backend.convert_to_numpy(model(x_np))
        litert_out = _run_litert_inference(_get_interpreter(path), [x_np])
        self.assertAllClose(keras_out, litert_out, atol=self.LITERT_ATOL)

    def test_export_with_verbose_false(self):
        model = models.Sequential([layers.Dense(1, input_shape=(5,))])
        path = os.path.join(self.get_temp_dir(), "quiet_test.tflite")

        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            model.export(path, format="litert", verbose=False)

        self.assertEqual(captured.getvalue(), "")
        self.assertTrue(os.path.exists(path))

    def test_export_invalid_filepath(self):
        model = models.Sequential([layers.Dense(1, input_shape=(5,))])
        path = os.path.join(self.get_temp_dir(), "bad_extension.bin")

        with self.assertRaises(ValueError):
            model.export(path, format="litert")

    def test_export_restores_jax_x64_setting(self):
        model = models.Sequential([layers.Dense(1, input_shape=(5,))])
        path = os.path.join(self.get_temp_dir(), "jax_restore.tflite")

        jax.config.update("jax_enable_x64", False)
        model.export(path, format="litert")

        self.assertFalse(jax.config.jax_enable_x64)

    def test_export_empty_optimizations_list(self):
        """Test export with empty optimizations list (no quantization)."""

        model = models.Sequential([layers.Dense(1, input_shape=(5,))])
        path = os.path.join(self.get_temp_dir(), "no_quant.tflite")

        # Empty list should not create quant_config
        model.export(path, format="litert", optimizations=[])
        self.assertTrue(os.path.exists(path))

    def test_export_with_normalization_layer(self):
        """Test export with a layer that has non-trainable tensor attributes.

        `Normalization` stores `mean` and `variance` as non-trainable tensors
        after `adapt()`. This ensures such attributes are correctly captured
        during export, which is distinct coverage from simple `Dense` layers.
        """

        model = models.Sequential(
            [
                layers.Normalization(axis=-1, input_shape=(5,)),
                layers.Dense(8, activation="relu"),
                layers.Dense(1),
            ]
        )

        # Adapt the normalization layer
        data = np.random.normal(size=(100, 5)).astype("float32")
        model.layers[0].adapt(data)

        path = os.path.join(self.get_temp_dir(), "norm_test.tflite")
        model.export(path, format="litert")

        self.assertTrue(os.path.exists(path))

        # Verify inference
        x_np = np.random.normal(size=(1, 5)).astype("float32")
        keras_out = backend.convert_to_numpy(model(x_np))
        litert_out = _run_litert_inference(_get_interpreter(path), [x_np])
        self.assertAllClose(keras_out, litert_out, atol=1e-4)
