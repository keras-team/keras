"""Tests for LiteRT export via the PyTorch backend (litert-torch)."""

import os

import numpy as np
import pytest

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing


def _requires_litert_torch():
    """Skip helper — call at the top of every test."""
    try:
        import litert_torch  # noqa: F401
        import torch  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        pytest.skip("litert-torch or torch not installed")


def _get_interpreter(filepath):
    """Return an allocated LiteRT interpreter for *filepath*."""
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=filepath)
    interp.allocate_tensors()
    return interp


def _run_litert_inference(interpreter, input_arrays):
    """Feed *input_arrays* (list of np arrays) through the interpreter."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for detail, arr in zip(input_details, input_arrays):
        interpreter.set_tensor(detail["index"], arr)

    interpreter.invoke()

    outputs = [interpreter.get_tensor(d["index"]) for d in output_details]
    return outputs if len(outputs) > 1 else outputs[0]


def _to_numpy(x):
    """Convert a Keras / torch tensor to numpy."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Requires torch backend"
)
class LiteRTTorchExportTest(testing.TestCase):
    """End-to-end tests: Keras → Torch (.pt2) → LiteRT (.tflite)."""

    # ------------------------------------------------------------------ #
    #  Sequential model
    # ------------------------------------------------------------------ #
    def test_sequential_model(self):
        _requires_litert_torch()

        model = models.Sequential(
            [
                layers.Dense(16, activation="relu", input_shape=(10,)),
                layers.Dense(1),
            ]
        )

        tflite_path = os.path.join(self.get_temp_dir(), "seq.tflite")
        model.export(tflite_path, format="litert")
        self.assertTrue(os.path.exists(tflite_path))

        # Numeric check
        x = np.random.normal(size=(1, 10)).astype("float32")
        keras_out = _to_numpy(model(x))
        litert_out = _run_litert_inference(_get_interpreter(tflite_path), [x])
        self.assertAllClose(keras_out, litert_out, atol=1e-4)

    # ------------------------------------------------------------------ #
    #  Functional model
    # ------------------------------------------------------------------ #
    def test_functional_model(self):
        _requires_litert_torch()

        inp = layers.Input(shape=(10,))
        x = layers.Dense(16, activation="relu")(inp)
        out = layers.Dense(1)(x)
        model = models.Model(inputs=inp, outputs=out)

        tflite_path = os.path.join(self.get_temp_dir(), "func.tflite")
        model.export(tflite_path, format="litert")
        self.assertTrue(os.path.exists(tflite_path))

        x = np.random.normal(size=(1, 10)).astype("float32")
        keras_out = _to_numpy(model(x))
        litert_out = _run_litert_inference(_get_interpreter(tflite_path), [x])
        self.assertAllClose(keras_out, litert_out, atol=1e-4)

    # ------------------------------------------------------------------ #
    #  Subclass model
    # ------------------------------------------------------------------ #
    def test_subclass_model(self):
        _requires_litert_torch()

        class TinyModel(models.Model):
            def __init__(self):
                super().__init__()
                self.dense1 = layers.Dense(16, activation="relu")
                self.dense2 = layers.Dense(1)

            def call(self, x):
                return self.dense2(self.dense1(x))

        model = TinyModel()
        model(np.zeros((1, 10), dtype="float32"))  # build

        tflite_path = os.path.join(self.get_temp_dir(), "subclass.tflite")
        model.export(tflite_path, format="litert")
        self.assertTrue(os.path.exists(tflite_path))

        x = np.random.normal(size=(1, 10)).astype("float32")
        keras_out = _to_numpy(model(x))
        litert_out = _run_litert_inference(_get_interpreter(tflite_path), [x])
        self.assertAllClose(keras_out, litert_out, atol=1e-4)

    # ------------------------------------------------------------------ #
    #  Conv model
    # ------------------------------------------------------------------ #
    def test_conv_model(self):
        _requires_litert_torch()

        model = models.Sequential(
            [
                layers.Conv2D(8, 3, activation="relu", input_shape=(8, 8, 3)),
                layers.Flatten(),
                layers.Dense(1),
            ]
        )

        tflite_path = os.path.join(self.get_temp_dir(), "conv.tflite")
        model.export(tflite_path, format="litert")
        self.assertTrue(os.path.exists(tflite_path))

        x = np.random.normal(size=(1, 8, 8, 3)).astype("float32")
        keras_out = _to_numpy(model(x))
        litert_out = _run_litert_inference(_get_interpreter(tflite_path), [x])
        self.assertAllClose(keras_out, litert_out, atol=1e-4)

    # ------------------------------------------------------------------ #
    #  Multi-input model
    # ------------------------------------------------------------------ #
    def test_multi_input_model(self):
        _requires_litert_torch()

        inp_a = layers.Input(shape=(10,), name="input_a")
        inp_b = layers.Input(shape=(10,), name="input_b")
        merged = layers.Concatenate()([inp_a, inp_b])
        out = layers.Dense(1)(merged)
        model = models.Model(inputs=[inp_a, inp_b], outputs=out)

        tflite_path = os.path.join(self.get_temp_dir(), "multi_in.tflite")
        model.export(tflite_path, format="litert")
        self.assertTrue(os.path.exists(tflite_path))

        a = np.random.normal(size=(1, 10)).astype("float32")
        b = np.random.normal(size=(1, 10)).astype("float32")
        keras_out = _to_numpy(model([a, b]))
        litert_out = _run_litert_inference(
            _get_interpreter(tflite_path), [a, b]
        )
        self.assertAllClose(keras_out, litert_out, atol=1e-4)

    # ------------------------------------------------------------------ #
    #  Multi-output model
    # ------------------------------------------------------------------ #
    def test_multi_output_model(self):
        _requires_litert_torch()

        inp = layers.Input(shape=(10,))
        x = layers.Dense(16, activation="relu")(inp)
        out_a = layers.Dense(1, name="out_a")(x)
        out_b = layers.Dense(1, name="out_b")(x)
        model = models.Model(inputs=inp, outputs=[out_a, out_b])

        tflite_path = os.path.join(self.get_temp_dir(), "multi_out.tflite")
        model.export(tflite_path, format="litert")
        self.assertTrue(os.path.exists(tflite_path))

        x = np.random.normal(size=(1, 10)).astype("float32")
        keras_outs = model(x)
        if not isinstance(keras_outs, (list, tuple)):
            keras_outs = [keras_outs]
        keras_outs = [_to_numpy(o) for o in keras_outs]

        litert_outs = _run_litert_inference(_get_interpreter(tflite_path), [x])
        if not isinstance(litert_outs, list):
            litert_outs = [litert_outs]

        self.assertEqual(len(keras_outs), len(litert_outs))
        # LiteRT may reorder outputs; match by closest value
        for k_out in keras_outs:
            matched = False
            for l_out in litert_outs:
                if k_out.shape == l_out.shape and np.allclose(
                    k_out, l_out, atol=1e-3
                ):
                    matched = True
                    break
            self.assertTrue(
                matched,
                f"No matching LiteRT output for keras output {k_out}",
            )

    # ------------------------------------------------------------------ #
    #  Torch (.pt2) numeric parity — Keras vs loaded ExportedProgram
    # ------------------------------------------------------------------ #
    def test_torch_export_numeric_parity(self):
        """Verify Keras output == loaded .pt2 output (no LiteRT)."""
        _requires_litert_torch()
        import torch

        model = models.Sequential(
            [
                layers.Dense(16, activation="relu", input_shape=(10,)),
                layers.Dense(1),
            ]
        )

        pt2_path = os.path.join(self.get_temp_dir(), "model.pt2")
        model.export(pt2_path, format="torch")

        x_np = np.random.normal(size=(1, 10)).astype("float32")
        keras_out = _to_numpy(model(x_np))

        program = torch.export.load(pt2_path)
        module = program.module()
        # Move input to same device as model (may be MPS on Apple Silicon)
        device = next(module.parameters()).device
        pt2_out = _to_numpy(module(torch.tensor(x_np, device=device)))
        self.assertAllClose(keras_out, pt2_out, atol=1e-5)

    # ------------------------------------------------------------------ #
    #  Full pipeline parity — Keras vs Torch vs LiteRT
    # ------------------------------------------------------------------ #
    def test_full_pipeline_numeric_parity(self):
        """Verify Keras == Torch (.pt2) == LiteRT (.tflite)."""
        _requires_litert_torch()
        import torch

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

        # 1. Keras reference
        keras_out = _to_numpy(model(x_np))

        # 2. Torch .pt2
        program = torch.export.load(pt2_path)
        module = program.module()
        device = next(module.parameters()).device
        pt2_out = _to_numpy(module(torch.tensor(x_np, device=device)))

        # 3. LiteRT .tflite
        litert_out = _run_litert_inference(
            _get_interpreter(tflite_path), [x_np]
        )

        self.assertAllClose(
            keras_out, pt2_out, atol=1e-5, msg="Keras vs Torch mismatch"
        )
        self.assertAllClose(
            keras_out, litert_out, atol=1e-4, msg="Keras vs LiteRT mismatch"
        )

    # ------------------------------------------------------------------ #
    #  Error handling — missing litert-torch
    # ------------------------------------------------------------------ #
    def test_import_error_without_litert_torch(self):
        """Mocking away litert_torch should raise a helpful ImportError."""
        import sys
        import unittest.mock

        with unittest.mock.patch.dict(sys.modules, {"litert_torch": None}):
            model = models.Sequential([layers.Dense(1)])
            model.build((None, 10))
            with self.assertRaisesRegex(ImportError, "litert-torch"):
                model.export("dummy.tflite", format="litert")
