import glob
import os
import re
import sys
import tempfile
from contextlib import redirect_stdout
from importlib import reload
from io import StringIO
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

from keras.src.callbacks.memory_usage_callback import MemoryUsageCallback
from keras.src.layers import Dense
from keras.src.models import Sequential
from keras.src.testing import TestCase

try:
    import psutil
except ImportError:
    psutil = None


@pytest.mark.skipif(psutil is None, reason="psutil is required")
class MemoryUsageCallbackTest(TestCase):
    def setUp(self):
        super().setUp()
        # Prepare 20 samples of 10-dim data → 4 batches @ bs=5

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
        self.bs = 5
        self.steps = len(self.x) // self.bs

    @pytest.mark.requires_trainable_backend
    def test_epoch_logging_stdout(self):
        """Epoch-level logs appear with correct format."""
        buf = StringIO()
        with redirect_stdout(buf):
            cb = MemoryUsageCallback(monitor_gpu=False)
            self.model.fit(
                self.x,
                self.y,
                epochs=self.epochs,
                batch_size=self.bs,
                callbacks=[cb],
                verbose=0,
            )
        out = buf.getvalue()
        for e in range(self.epochs):
            assert f"Epoch {e} start" in out
            assert f"Epoch {e} end" in out
            assert re.search(rf"Epoch {e} start - CPU Memory: [\d\.]+ MB", out)
            assert re.search(rf"Epoch {e} end - CPU Memory: [\d\.]+ MB", out)

    @pytest.mark.requires_trainable_backend
    def test_batch_logging_stdout(self):
        """Batch-level logs appear when log_every_batch=True."""
        buf = StringIO()
        with redirect_stdout(buf):
            cb = MemoryUsageCallback(monitor_gpu=False, log_every_batch=True)
            self.model.fit(
                self.x,
                self.y,
                epochs=1,
                batch_size=self.bs,
                callbacks=[cb],
                verbose=0,
            )
        lines = buf.getvalue().splitlines()
        batch_lines = [l for l in lines if l.startswith("Batch ")]
        assert len(batch_lines) == self.steps
        assert all(
            re.match(r"Batch \d+ end - CPU Memory: [\d\.]+ MB", l)
            for l in batch_lines
        )

    @pytest.mark.requires_trainable_backend
    def test_tensorboard_writes_files(self):
        """TensorBoard event files are created."""
        tmp = tempfile.TemporaryDirectory()
        logdir = os.path.join(tmp.name, "tb")
        buf = StringIO()
        with redirect_stdout(buf):
            cb = MemoryUsageCallback(
                monitor_gpu=False, tensorboard_log_dir=logdir
            )
            self.model.fit(
                self.x,
                self.y,
                epochs=1,
                batch_size=self.bs,
                callbacks=[cb],
                verbose=0,
            )
        files = glob.glob(os.path.join(logdir, "events.out.tfevents.*"))
        assert files, "No TensorBoard event files generated"

    @pytest.mark.requires_trainable_backend
    def test_missing_psutil_raises(self):
        """Constructor raises if psutil is missing."""
        mod = sys.modules["keras.src.callbacks.memory_usage_callback"]
        orig = getattr(mod, "psutil", None)
        with patch.dict(sys.modules, {"psutil": None}):
            reload(mod)
            with pytest.raises(ImportError):
                _ = mod.MemoryUsageCallback(monitor_gpu=False)
        # restore

        if orig is not None:
            sys.modules["psutil"] = orig
            reload(mod)


@pytest.mark.requires_trainable_backend
def test_torch_backend_gpu_memory(monkeypatch):
    """Simulate PyTorch backend and verify GPU memory sum."""
    import keras.src.backend as B

    monkeypatch.setattr(B, "backend", lambda: "torch")

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.device_count.return_value = 2
    fake_torch.cuda.memory_allocated.side_effect = [
        100 * 1024**2,
        150 * 1024**2,
    ]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    cb = MemoryUsageCallback(monitor_gpu=True)
    mem = cb._get_gpu_memory()
    assert pytest.approx(250, rel=1e-6) == mem


@pytest.mark.requires_trainable_backend
def test_jax_backend_gpu_memory(monkeypatch):
    """Simulate JAX backend and verify GPU memory sum."""
    import keras.src.backend as B

    monkeypatch.setattr(B, "backend", lambda: "jax")

    class FakeDev:
        platform = "gpu"

        def memory_stats(self):
            return {"bytes_in_use": 200 * 1024**2}

    fake_jax = MagicMock()
    fake_jax.devices.return_value = [FakeDev(), FakeDev()]
    monkeypatch.setitem(sys.modules, "jax", fake_jax)

    cb = MemoryUsageCallback(monitor_gpu=True)
    mem = cb._get_gpu_memory()
    assert pytest.approx(400, rel=1e-6) == mem
