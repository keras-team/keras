import os
import glob
import re
import sys
import tempfile

import numpy as np
import pytest
import tensorflow as tf

from io import StringIO
from contextlib import redirect_stdout
from importlib import reload
from unittest.mock import patch, MagicMock

from keras.src.models import Sequential
from keras.src.layers import Dense
from keras.src.testing import TestCase
from keras.src.callbacks.memory_usage_callback import MemoryUsageCallback

# Skip the class entirely if psutil is missing
try:
    import psutil
except ImportError:
    psutil = None


@pytest.mark.skipif(psutil is None, reason="psutil is required for MemoryUsageCallback tests.")
class MemoryUsageCallbackTest(TestCase):
    def setUp(self):
        super().setUp()
        # Dummy data
        self.x_train = np.random.random((20, 10)).astype(np.float32)
        self.y_train = np.random.randint(0, 2, (20, 1)).astype(np.float32)
        # Simple model
        self.model = Sequential([
            Dense(5, activation="relu", input_shape=(10,)),
            Dense(1, activation="sigmoid")
        ])
        self.model.compile(optimizer="adam", loss="binary_crossentropy")

        self.epochs = 2
        self.batch_size = 5
        self.steps_per_epoch = len(self.x_train) // self.batch_size

    @pytest.mark.requires_trainable_backend
    def test_callback_logs_stdout(self):
        """Epoch‐level stdout logging works as expected."""
        out = StringIO()
        with redirect_stdout(out):
            gpu_avail = bool(tf.config.list_physical_devices("GPU"))
            cb = MemoryUsageCallback(monitor_gpu=gpu_avail, log_every_batch=False)
            self.model.fit(
                self.x_train, self.y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[cb],
                verbose=0
            )
        cap = out.getvalue()
        for i in range(self.epochs):
            self.assertIn(f"Epoch {i} start - CPU Memory:", cap)
            self.assertIn(f"Epoch {i} end - CPU Memory:", cap)
            self.assertRegex(cap, rf"Epoch {i} start - CPU Memory: [\d.e+-]+ MB")
            self.assertRegex(cap, rf"Epoch {i} end - CPU Memory: [\d.e+-]+ MB")
        if gpu_avail:
            self.assertIn("GPU Memory:", cap)
        else:
            self.assertNotIn("GPU Memory:", cap)

    @pytest.mark.requires_trainable_backend
    def test_log_every_batch_stdout(self):
        """Batch‐level stdout logging works when enabled."""
        out = StringIO()
        with redirect_stdout(out):
            gpu_avail = bool(tf.config.list_physical_devices("GPU"))
            cb = MemoryUsageCallback(monitor_gpu=gpu_avail, log_every_batch=True)
            self.model.fit(
                self.x_train, self.y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[cb],
                verbose=0
            )
        lines = out.getvalue().splitlines()
        total_batches = self.epochs * self.steps_per_epoch
        batch_regex = r"Batch \d+ end - CPU Memory: [\d.e+-]+ MB"
        count = sum(1 for line in lines if re.search(batch_regex, line))
        self.assertEqual(count, total_batches)

    @pytest.mark.requires_trainable_backend
    def test_tensorboard_logging_file_creation(self):
        """TensorBoard writer creates event files in given directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            gpu_avail = bool(tf.config.list_physical_devices("GPU"))
            log_dir = os.path.join(tmp_dir, "tb_logs")
            cb = MemoryUsageCallback(
                monitor_gpu=gpu_avail,
                log_every_batch=True,
                tensorboard_log_dir=log_dir
            )
            # The directory should be created by the callback __init__
            assert os.path.isdir(log_dir)
            self.model.fit(
                self.x_train, self.y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[cb],
                verbose=0
            )
            event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
            self.assertGreater(len(event_files), 0)

    @pytest.mark.requires_trainable_backend
    def test_get_gpu_memory(self):
        """_get_gpu_memory returns float or None depending on availability."""
        cb_gpu = MemoryUsageCallback(monitor_gpu=True, log_every_batch=False)
        mem = cb_gpu._get_gpu_memory()
        if tf.config.list_physical_devices("GPU"):
            assert isinstance(mem, float) and mem >= 0.0
        else:
            assert mem is None

        cb_no = MemoryUsageCallback(monitor_gpu=False, log_every_batch=False)
        assert cb_no._get_gpu_memory() is None

    def test_raises_if_psutil_missing(self):
        """Constructor raises ImportError when psutil is unavailable."""
        import keras.src.callbacks.memory_usage_callback as mod
        orig = getattr(mod, 'psutil', None)
        with patch.dict(sys.modules, {"psutil": None}):
            with pytest.raises(ImportError):
                reload(mod)
                _ = mod.MemoryUsageCallback()
        # Restore
        if orig is not None:
            sys.modules["psutil"] = orig
            reload(mod)




@pytest.mark.requires_trainable_backend
def test_torch_backend_gpu_memory(monkeypatch):
    """Simulate PyTorch backend and verify GPU memory sum."""
    import keras.src.backend as B
    monkeypatch.setattr(B, "backend", lambda: "torch")

    # Create a fake torch module
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.device_count.return_value = 2
    # Each device allocates 100 MB and 150 MB
    fake_torch.cuda.memory_allocated.side_effect = [100 * 1024**2, 150 * 1024**2]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    cb = MemoryUsageCallback(monitor_gpu=True)
    mem = cb._get_gpu_memory()
    # Expect (100 + 150) MB
    assert pytest.approx(250, rel=1e-6) == mem


@pytest.mark.requires_trainable_backend
def test_jax_backend_gpu_memory(monkeypatch):
    """Simulate JAX backend and verify GPU memory sum."""
    import keras.src.backend as B
    monkeypatch.setattr(B, "backend", lambda: "jax")

    # Fake JAX devices
    class FakeDevice:
        platform = "gpu"
        def memory_stats(self):
            return {"bytes_in_use": 200 * 1024**2}

    fake_jax = MagicMock()
    fake_jax.devices.return_value = [FakeDevice(), FakeDevice()]
    monkeypatch.setitem(sys.modules, "jax", fake_jax)

    cb = MemoryUsageCallback(monitor_gpu=True)
    mem = cb._get_gpu_memory()
    # Expect 2 * 200 MB
    assert pytest.approx(400, rel=1e-6) == mem
