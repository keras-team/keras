import os
import glob
import tempfile
import warnings

from contextlib import redirect_stdout
from io import StringIO
from importlib import reload
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from keras.src.callbacks.memory_usage_callback import MemoryUsageCallback
from keras.src.layers import Dense
from keras.src.models import Sequential
from keras.src.testing import TestCase
from keras.src import backend as K

# Skip all tests if psutil is not installed
try:
    import psutil
except ImportError:
    psutil = None


@pytest.mark.skipif(
    psutil is None, reason="psutil is required for MemoryUsageCallback tests."
)
class MemoryUsageCallbackTest(TestCase):
    def setUp(self):
        super().setUp()
        self.x = np.random.random((20, 10)).astype(np.float32)
        self.y = np.random.randint(0, 2, (20, 1)).astype(np.float32)
        self.model = Sequential(
            [
                Dense(5, activation="relu", input_shape=(10,)),
                Dense(1, activation="sigmoid"),
            ]
        )
        self.model.compile(optimizer="adam", loss="binary_crossentropy")
        self.epochs = 2
        self.batch_size = 5
        self.total_batches = self.epochs * (len(self.x) // self.batch_size)

    @pytest.mark.requires_trainable_backend
    def test_epoch_and_batch_stdout(self):
        out = StringIO()
        with redirect_stdout(out):
            # Mock GPU memory for predictability
            with patch.object(
                MemoryUsageCallback, "_get_gpu_memory", return_value=42.0
            ):
                cb = MemoryUsageCallback(monitor_gpu=True, log_every_batch=True)
                self.model.fit(
                    self.x,
                    self.y,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    callbacks=[cb],
                    verbose=0,
                )
        log = out.getvalue().splitlines()
        # Check epoch logs
        for i in range(self.epochs):
            assert any(f"Epoch {i} start" in line for line in log)
            assert any(f"Epoch {i} end" in line for line in log)
        # Check batch logs count
        batch_lines = [l for l in log if l.startswith("Batch")]
        assert len(batch_lines) == self.total_batches
        # Confirm GPU part present
        assert any("GPU Memory: 42.00 MB" in l for l in log)

    @pytest.mark.requires_trainable_backend
    def test_tensorboard_file_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tb_dir = os.path.join(tmpdir, "tb")
            # Mock CPU/GPU memory
            with patch.object(
                MemoryUsageCallback, "_get_gpu_memory", return_value=10.0
            ), patch.object(MemoryUsageCallback, "_get_cpu_memory", return_value=5.0):
                cb = MemoryUsageCallback(
                    monitor_gpu=True,
                    log_every_batch=False,
                    tensorboard_log_dir=tb_dir,
                )
                assert os.path.isdir(tb_dir)
                self.model.fit(
                    self.x,
                    self.y,
                    epochs=1,
                    batch_size=self.batch_size,
                    callbacks=[cb],
                    verbose=0,
                )
            events = glob.glob(os.path.join(tb_dir, "events.out.tfevents.*"))
            assert events, "No TensorBoard event file found."

    def test_import_error_without_psutil(self):
        import sys
        import keras.src.callbacks.memory_usage_callback as mod

        orig = getattr(mod, "psutil", None)
        with patch.dict(sys.modules, {"psutil": None}):
            with pytest.raises(ImportError):
                reload(mod)
                _ = mod.MemoryUsageCallback()
        # restore
        if orig is not None:
            sys.modules["psutil"] = orig
            reload(mod)


# Backend-specific tests
@pytest.mark.requires_trainable_backend
def test_torch_gpu_memory(monkeypatch):
    monkeypatch.setattr(K, "backend", lambda: "torch")
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.device_count.return_value = 2
    fake_torch.cuda.memory_allocated.side_effect = [100 * 1024**2, 150 * 1024**2]
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    cb = MemoryUsageCallback(monitor_gpu=True)
    assert pytest.approx(250, rel=1e-6) == cb._get_gpu_memory()


@pytest.mark.requires_trainable_backend
def test_jax_gpu_memory(monkeypatch):
    monkeypatch.setattr(K, "backend", lambda: "jax")

    class Dev:
        platform = "gpu"

        def memory_stats(self):
            return {"bytes_in_use": 200 * 1024**2}

    fake_jax = MagicMock()
    fake_jax.devices.return_value = [Dev(), Dev()]
    monkeypatch.setitem(__import__("sys").modules, "jax", fake_jax)
    cb = MemoryUsageCallback(monitor_gpu=True)
    assert pytest.approx(400, rel=1e-6) == cb._get_gpu_memory()
