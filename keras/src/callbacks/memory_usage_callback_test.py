import glob
import os
import sys
import tempfile
from contextlib import redirect_stdout
from importlib import reload
from io import StringIO
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

import keras.src.callbacks.memory_usage_callback as muc_module
from keras.src.callbacks.memory_usage_callback import MemoryUsageCallback

try:
    import psutil as real_psutil
except ImportError:
    real_psutil = None

from keras.src.layers import Dense
from keras.src.models import Sequential
from keras.src.testing import TestCase


@pytest.mark.skipif(
    real_psutil is None,
    reason="psutil is required for MemoryUsageCallback tests.",
)
class MemoryUsageCallbackTest(TestCase):
    def setUp(self):
        super().setUp()
        self.x_train = np.random.random((16, 8)).astype(np.float32)
        self.y_train = np.random.randint(0, 2, (16, 1)).astype(np.float32)

        self.model = Sequential(
            [
                Dense(4, activation="relu", input_shape=(8,)),
                Dense(1, activation="sigmoid"),
            ]
        )
        self.model.compile(optimizer="adam", loss="binary_crossentropy")

        self.epochs = 2
        self.batch_size = 4
        self.steps_per_epoch = len(self.x_train) // self.batch_size

    def test_cpu_only_epoch_logging(self):
        with patch.object(
            muc_module.K, "backend", return_value="unsupported_backend"
        ):
            out = StringIO()
            with redirect_stdout(out):
                cb = MemoryUsageCallback(log_every_batch=False)
                self.model.fit(
                    self.x_train,
                    self.y_train,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    callbacks=[cb],
                    verbose=0,
                )
            lines = [
                l.strip() for l in out.getvalue().splitlines() if l.strip()
            ]
            epoch_lines = [l for l in lines if l.startswith("Epoch")]
            assert len(epoch_lines) == 4
            for i in range(self.epochs):
                assert epoch_lines[2 * i].startswith(
                    f"Epoch {i} start - CPU Memory:"
                )
                assert epoch_lines[2 * i + 1].startswith(
                    f"Epoch {i} end - CPU Memory:"
                )

    def test_log_every_batch(self):
        with patch.object(
            muc_module.K, "backend", return_value="unsupported_backend"
        ):
            out = StringIO()
            with redirect_stdout(out):
                cb = MemoryUsageCallback(log_every_batch=True)
                self.model.fit(
                    self.x_train,
                    self.y_train,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    callbacks=[cb],
                    verbose=0,
                )
            lines = [
                l.strip() for l in out.getvalue().splitlines() if l.strip()
            ]
            batch_lines = [l for l in lines if l.startswith("Batch")]
            assert len(batch_lines) == self.epochs * self.steps_per_epoch

    def test_tensorboard_log_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.object(
                muc_module.K, "backend", return_value="unsupported_backend"
            ):
                cb = MemoryUsageCallback(
                    log_every_batch=False, tensorboard_log_dir=tmp_dir
                )
                self.model.fit(
                    self.x_train,
                    self.y_train,
                    epochs=1,
                    batch_size=self.batch_size,
                    callbacks=[cb],
                    verbose=0,
                )
            files = glob.glob(os.path.join(tmp_dir, "events.out.tfevents.*"))
            assert len(files) > 0, (
                f"No TensorBoard event file found in {tmp_dir}"
            )

    def test_psutil_missing(self):
        """
        Temporarily override the module's `psutil` to None so
        that instantiating MemoryUsageCallback raises ImportError.
        """
        original_psutil = muc_module.psutil
        try:
            muc_module.psutil = None
            with pytest.raises(
                ImportError,
                match="MemoryUsageCallback requires the 'psutil' library",
            ):
                _ = muc_module.MemoryUsageCallback()
        finally:
            muc_module.psutil = original_psutil
            reload(muc_module)


def test_gpu_memory_tensorflow(monkeypatch):
    """
    Simulate TensorFlow backend with one GPU device named "GPU:0"
    whose memory_info()['current'] is 150 MiB. After reload, _get_gpu_memory()
    must return 150.0 (MB).
    """

    if real_psutil:
        sys.modules["psutil"] = real_psutil

    class FakeDevice:
        def __init__(self, name):
            self.name = name

    fake_tf = MagicMock()
    fake_tf.config.list_physical_devices.return_value = [FakeDevice("GPU:0")]
    fake_tf.config.experimental.get_memory_info.return_value = {
        "current": 150 * 1024**2
    }

    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    monkeypatch.setattr(
        "keras.src.callbacks.memory_usage_callback.K.backend",
        lambda: "tensorflow",
    )

    reload(muc_module)

    cb = muc_module.MemoryUsageCallback()
    mem_mb = cb._get_gpu_memory()
    assert pytest.approx(150.0, rel=1e-6) == mem_mb


def test_gpu_memory_torch(monkeypatch):
    """
    Simulate PyTorch backend with 2 GPUs that allocate 100 MiB and 200 MiB.
    After reload, _get_gpu_memory() should return 300.0 (MB).
    """
    if real_psutil:
        sys.modules["psutil"] = real_psutil

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.device_count.return_value = 2
    fake_torch.cuda.memory_allocated.side_effect = [
        100 * 1024**2,
        200 * 1024**2,
    ]

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        "keras.src.callbacks.memory_usage_callback.K.backend", lambda: "torch"
    )

    reload(muc_module)

    cb = muc_module.MemoryUsageCallback()
    mem_mb = cb._get_gpu_memory()
    assert pytest.approx(300.0, rel=1e-6) == mem_mb


def test_gpu_memory_jax(monkeypatch):
    """
    Simulate JAX backend with two GPU devices each reporting
    bytes_in_use=220 MiB. Expect 440.0 (MB).
    """
    if real_psutil:
        sys.modules["psutil"] = real_psutil

    class FakeDevice:
        platform = "GPU"

        def memory_stats(self):
            return {"bytes_in_use": 220 * 1024**2}

    fake_jax = MagicMock()
    fake_jax.devices.return_value = [FakeDevice(), FakeDevice()]

    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setattr(
        "keras.src.callbacks.memory_usage_callback.K.backend", lambda: "jax"
    )

    reload(muc_module)

    cb = muc_module.MemoryUsageCallback()
    mem_mb = cb._get_gpu_memory()
    assert pytest.approx(440.0, rel=1e-6) == mem_mb
