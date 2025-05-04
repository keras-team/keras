import os
import glob
import sys
import tempfile
from importlib import reload
from io import StringIO
from contextlib import redirect_stdout
from unittest.mock import patch, MagicMock

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
    def test_epoch_logging(self):
        out = StringIO()
        with redirect_stdout(out):

            for gpu_val in (None, 42.0):
                with patch.object(
                    MemoryUsageCallback, "_get_gpu_memory", return_value=gpu_val
                ):
                    cb = MemoryUsageCallback(monitor_gpu=True, log_every_batch=False)
                    self.model.fit(
                        self.x,
                        self.y,
                        epochs=self.epochs,
                        batch_size=self.bs,
                        callbacks=[cb],
                        verbose=0,
                    )
        log = out.getvalue()
        # must see epoch start/end lines

        for e in range(self.epochs):
            assert f"Epoch {e} start" in log
            assert f"Epoch {e} end" in log
        # must see GPU segment at least once if gpu_val not None

        assert "; GPU" in log

    @pytest.mark.requires_trainable_backend
    def test_batch_logging(self):
        out = StringIO()
        with redirect_stdout(out):
            with patch.object(
                MemoryUsageCallback, "_get_gpu_memory", return_value=None
            ):
                cb = MemoryUsageCallback(monitor_gpu=True, log_every_batch=True)
                self.model.fit(
                    self.x,
                    self.y,
                    epochs=1,
                    batch_size=self.bs,
                    callbacks=[cb],
                    verbose=0,
                )
        lines = [ln for ln in out.getvalue().splitlines() if ln.startswith("Batch ")]
        assert len(lines) == self.steps

    @pytest.mark.requires_trainable_backend
    def test_tensorboard_events(self):
        tmp = tempfile.TemporaryDirectory()
        logdir = os.path.join(tmp.name, "logs")
        cb = MemoryUsageCallback(
            monitor_gpu=False, log_every_batch=False, tensorboard_log_dir=logdir
        )
        assert os.path.isdir(logdir)
        self.model.fit(
            self.x, self.y, epochs=1, batch_size=self.bs, callbacks=[cb], verbose=0
        )
        files = glob.glob(os.path.join(logdir, "events.out.tfevents.*"))
        assert len(files) > 0
        tmp.cleanup()

    def test_missing_psutil(self):
        mod = sys.modules["keras.src.callbacks.memory_usage_callback"]
        orig = getattr(mod, "psutil", None)
        with patch.dict(sys.modules, {"psutil": None}):
            with pytest.raises(ImportError):
                reload(mod)
                _ = mod.MemoryUsageCallback()
        if orig is not None:
            sys.modules["psutil"] = orig
            reload(mod)


@pytest.mark.requires_trainable_backend
def test_torch_gpu(monkeypatch):
    import keras.src.backend as B

    monkeypatch.setattr(B, "backend", lambda: "torch")
    fake = MagicMock()
    fake.cuda.is_available.return_value = True
    fake.cuda.device_count.return_value = 2
    fake.cuda.memory_allocated.side_effect = [50 * 1024**2, 70 * 1024**2]
    monkeypatch.setitem(sys.modules, "torch", fake)
    cb = MemoryUsageCallback(monitor_gpu=True)
    assert pytest.approx(120, rel=1e-6) == cb._get_gpu_memory()


@pytest.mark.requires_trainable_backend
def test_jax_gpu(monkeypatch):
    import keras.src.backend as B

    monkeypatch.setattr(B, "backend", lambda: "jax")

    class Dev:
        platform = "GPU"

        def memory_stats(self):
            return {"bytes_in_use": 30 * 1024**2}

    fake = MagicMock(devices=lambda: [Dev(), Dev()])
    monkeypatch.setitem(sys.modules, "jax", fake)
    cb = MemoryUsageCallback(monitor_gpu=True)
    assert pytest.approx(60, rel=1e-6) == cb._get_gpu_memory()


@pytest.mark.requires_trainable_backend
def test_openvino_gpu(monkeypatch):
    import keras.src.backend as B

    monkeypatch.setattr(B, "backend", lambda: "openvino")

    class Core:
        def get_property(self, device, name):
            return {"deviceUsedBytes": 25 * 1024**2}

        available_devices = ["GPU"]

    fake = MagicMock(Core=lambda: Core())
    monkeypatch.setitem(sys.modules, "openvino", fake)
    cb = MemoryUsageCallback(monitor_gpu=True)
    assert pytest.approx(25, rel=1e-6) == cb._get_gpu_memory()
