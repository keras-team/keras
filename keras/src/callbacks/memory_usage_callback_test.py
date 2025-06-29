import os
import glob
import sys
import tempfile
import re

import numpy as np
import pytest
import tensorflow as tf

from io import StringIO
from contextlib import redirect_stdout
from importlib import reload
from unittest.mock import patch, MagicMock

from keras.src import backend as K
from keras.src.callbacks.memory_usage_callback import (
    MemoryUsageCallback,
    running_on_gpu,
    running_on_tpu,
)
from keras.src.models import Sequential
from keras.src.layers import Dense

try:
    import psutil
except ImportError:
    psutil = None


@pytest.mark.skipif(psutil is None, reason="psutil is required for MemoryUsageCallback tests.")
class TestMemoryUsageCallback:
    @pytest.fixture(autouse=True)
    def setup_model(self):
        self.x_train = np.random.random((20, 10)).astype(np.float32)
        self.y_train = np.random.randint(0, 2, (20, 1)).astype(np.float32)

        self.model = Sequential([
            Dense(5, activation="relu", input_shape=(10,)),
            Dense(1, activation="sigmoid")
        ])
        self.model.compile(optimizer="adam", loss="binary_crossentropy")

        self.epochs = 2
        self.batch_size = 5
        self.steps_per_epoch = len(self.x_train) // self.batch_size
        yield

    @pytest.mark.requires_trainable_backend
    def test_cpu_only_epoch_logging(self):
        # Force TF backend and no GPU/TPU
        monkey = patch.object(K, "backend", lambda: "tensorflow")
        with monkey:
            out = StringIO()
            with redirect_stdout(out), \
                 patch("keras.src.callbacks.memory_usage_callback.running_on_gpu", return_value=False), \
                 patch("keras.src.callbacks.memory_usage_callback.running_on_tpu", return_value=False):
                cb = MemoryUsageCallback(log_every_batch=False)
                self.model.fit(self.x_train, self.y_train,
                               epochs=self.epochs,
                               batch_size=self.batch_size,
                               callbacks=[cb],
                               verbose=0)

            lines = out.getvalue().splitlines()
            start = [ln for ln in lines if re.match(r"^Epoch \d+ start - CPU Memory:", ln)]
            end   = [ln for ln in lines if re.match(r"^Epoch \d+ end - CPU Memory:", ln)]
            assert len(start) == self.epochs
            assert len(end)   == self.epochs
            assert not any("GPU Memory" in ln or "TPU Memory" in ln for ln in lines)

    @pytest.mark.requires_trainable_backend
    def test_log_every_batch(self):
        monkey = patch.object(K, "backend", lambda: "tensorflow")
        with monkey:
            out = StringIO()
            with redirect_stdout(out), \
                 patch("keras.src.callbacks.memory_usage_callback.running_on_gpu", return_value=False), \
                 patch("keras.src.callbacks.memory_usage_callback.running_on_tpu", return_value=False):
                cb = MemoryUsageCallback(log_every_batch=True)
                self.model.fit(self.x_train, self.y_train,
                               epochs=self.epochs,
                               batch_size=self.batch_size,
                               callbacks=[cb],
                               verbose=0)

            lines = out.getvalue().splitlines()
            batches = [ln for ln in lines if re.match(r"^Batch \d+ end - CPU Memory:", ln)]
            assert len(batches) == self.epochs * self.steps_per_epoch

    @pytest.mark.requires_trainable_backend
    def test_tensorboard_log_dir(self):
        monkey = patch.object(K, "backend", lambda: "tensorflow")
        with monkey:
            with tempfile.TemporaryDirectory() as tmpdir:
                log_dir = os.path.join(tmpdir, "tb_logs")
                with patch("keras.src.callbacks.memory_usage_callback.running_on_gpu", return_value=False), \
                     patch("keras.src.callbacks.memory_usage_callback.running_on_tpu", return_value=False):
                    cb = MemoryUsageCallback(log_every_batch=True, tensorboard_log_dir=log_dir)
                    assert os.path.isdir(log_dir)
                    self.model.fit(self.x_train, self.y_train,
                                   epochs=self.epochs,
                                   batch_size=self.batch_size,
                                   callbacks=[cb],
                                   verbose=0)
                files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
                assert files, f"No events files under {log_dir}"

    @pytest.mark.requires_trainable_backend
    def test_get_gpu_memory_tensorflow(self):
        patch_backend = patch.object(K, "backend", lambda: "tensorflow")
        fake_tf = MagicMock()
        # mock physical devices
        fake_tf.config.list_physical_devices.return_value = ["GPU:0"]
        fake_tf.config.experimental.get_memory_info.return_value = {"current": 150 * 1024**2}

        with patch_backend, \
             patch.dict(sys.modules, {
                 "tensorflow": fake_tf,
                 "tensorflow.config": fake_tf.config,
                 "tensorflow.config.experimental": fake_tf.config.experimental
             }):
            cb = MemoryUsageCallback()
            assert pytest.approx(150.0, rel=1e-6) == cb._get_gpu_memory()

    @pytest.mark.requires_trainable_backend
    def test_get_gpu_memory_torch(self):
        patch_backend = patch.object(K, "backend", lambda: "torch")
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.device_count.return_value = 2
        # return 100MB then 200MB
        fake_torch.cuda.memory_allocated.side_effect = [100 * 1024**2, 200 * 1024**2]

        with patch_backend, \
             patch.dict(sys.modules, {"torch": fake_torch, "torch.cuda": fake_torch.cuda}):
            cb = MemoryUsageCallback()
            assert pytest.approx(300.0, rel=1e-6) == cb._get_gpu_memory()

    @pytest.mark.requires_trainable_backend
    def test_get_gpu_memory_jax(self):
        patch_backend = patch.object(K, "backend", lambda: "jax")
        class Dev:
            platform = "gpu"
            def memory_stats(self): return {"bytes_in_use": 200 * 1024**2}
        fake_jax = MagicMock()
        fake_jax.devices.return_value = [Dev(), Dev()]

        with patch_backend, patch.dict(sys.modules, {"jax": fake_jax}):
            cb = MemoryUsageCallback()
            assert pytest.approx(400.0, rel=1e-6) == cb._get_gpu_memory()

    def test_running_on_gpu_and_tpu_flags(self):
        g = running_on_gpu(); t = running_on_tpu()
        assert isinstance(g, bool) and isinstance(t, bool)

    def test_psutil_missing(self):
        # ensure ImportError if psutil absent
        orig = sys.modules.pop("psutil", None)
        try:
            import keras.src.callbacks.memory_usage_callback as mod
            with patch.dict(sys.modules, {"psutil": None}):
                with pytest.raises(ImportError):
                    reload(mod)
                    _ = mod.MemoryUsageCallback()
        finally:
            if orig is not None:
                sys.modules["psutil"] = orig
                reload(sys.modules["keras.src.callbacks.memory_usage_callback"])
