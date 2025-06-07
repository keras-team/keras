import os
import glob
import sys
import tempfile
import re

import numpy as np
import pytest

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


@pytest.mark.skipif(
    psutil is None, reason="psutil is required for MemoryUsageCallback tests."
)
class TestMemoryUsageCallback:
    """
    Test suite for MemoryUsageCallback.  We explicitly patch `K.backend()` → "tensorflow"
    whenever we call `model.fit(...)`, so that the callback’s logging logic actually runs.
    Otherwise, on the “NumPy” backend, `.fit(…)` isn’t implemented and nothing is printed.
    """

    @pytest.fixture(autouse=True)
    def setup_model(self):
        self.x_train = np.random.random((20, 10)).astype(np.float32)
        self.y_train = np.random.randint(0, 2, (20, 1)).astype(np.float32)

        self.model = Sequential(
            [
                Dense(5, activation="relu", input_shape=(10,)),
                Dense(1, activation="sigmoid"),
            ]
        )
        self.model.compile(optimizer="adam", loss="binary_crossentropy")

        self.epochs = 2
        self.batch_size = 5
        self.steps_per_epoch = len(self.x_train) // self.batch_size

        yield

    @pytest.mark.requires_trainable_backend
    def test_cpu_only_epoch_logging(self, monkeypatch):
        monkeypatch.setattr(K, "backend", lambda: "tensorflow")

        out = StringIO()
        with redirect_stdout(out):
            with patch(
                "keras.src.callbacks.memory_usage_callback.running_on_gpu",
                return_value=False,
            ), patch(
                "keras.src.callbacks.memory_usage_callback.running_on_tpu",
                return_value=False,
            ):
                cb = MemoryUsageCallback(log_every_batch=False)
                self.model.fit(
                    self.x_train,
                    self.y_train,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    callbacks=[cb],
                    verbose=0,
                )
        lines = out.getvalue().splitlines()
        start_lines = [
            ln
            for ln in lines
            if re.match(r"^Epoch \d+ start - CPU Memory: [\d\.]+ MB$", ln)
        ]
        end_lines = [
            ln
            for ln in lines
            if re.match(r"^Epoch \d+ end - CPU Memory: [\d\.]+ MB$", ln)
        ]

        assert len(start_lines) == self.epochs
        assert len(end_lines) == self.epochs
        assert all("GPU Memory" not in ln for ln in lines)
        assert all("TPU Memory" not in ln for ln in lines)

    @pytest.mark.requires_trainable_backend
    def test_log_every_batch(self, monkeypatch):
        monkeypatch.setattr(K, "backend", lambda: "tensorflow")

        out = StringIO()
        with redirect_stdout(out):
            with patch(
                "keras.src.callbacks.memory_usage_callback.running_on_gpu",
                return_value=False,
            ), patch(
                "keras.src.callbacks.memory_usage_callback.running_on_tpu",
                return_value=False,
            ):
                cb = MemoryUsageCallback(log_every_batch=True)
                self.model.fit(
                    self.x_train,
                    self.y_train,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    callbacks=[cb],
                    verbose=0,
                )
        lines = out.getvalue().splitlines()
        batch_lines = [
            ln
            for ln in lines
            if re.match(r"^Batch \d+ end - CPU Memory: [\d\.]+ MB$", ln)
        ]
        expected_batches = self.epochs * self.steps_per_epoch
        assert len(batch_lines) == expected_batches

    @pytest.mark.requires_trainable_backend
    def test_tensorboard_log_dir(self, monkeypatch):
        monkeypatch.setattr(K, "backend", lambda: "tensorflow")

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "tb_logs")
            with patch(
                "keras.src.callbacks.memory_usage_callback.running_on_gpu",
                return_value=False,
            ), patch(
                "keras.src.callbacks.memory_usage_callback.running_on_tpu",
                return_value=False,
            ):
                cb = MemoryUsageCallback(
                    log_every_batch=True, tensorboard_log_dir=log_dir
                )
                assert os.path.isdir(log_dir)

                self.model.fit(
                    self.x_train,
                    self.y_train,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    callbacks=[cb],
                    verbose=0,
                )

                event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
                assert event_files, f"No TensorBoard events under {log_dir}"

    @pytest.mark.requires_trainable_backend
    def test_get_gpu_memory_tensorflow(self, monkeypatch):
        monkeypatch.setattr(K, "backend", lambda: "tensorflow")
        fake_tf = MagicMock()
        fake_tf.config.list_physical_devices.return_value = ["GPU:0"]
        fake_tf.config.experimental.get_memory_info.return_value = {
            "current": 150 * 1024**2
        }
        monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
        monkeypatch.setitem(sys.modules, "tensorflow.config", fake_tf.config)
        monkeypatch.setitem(
            sys.modules, "tensorflow.config.experimental", fake_tf.config.experimental
        )

        cb = MemoryUsageCallback()
        assert pytest.approx(150.0, rel=1e-6) == cb._get_gpu_memory()

    @pytest.mark.requires_trainable_backend
    def test_get_gpu_memory_torch(self, monkeypatch):
        monkeypatch.setattr(K, "backend", lambda: "torch")
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.device_count.return_value = 2
        fake_torch.cuda.memory_allocated.side_effect = [100 * 1024**2, 200 * 1024**2]
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "torch.cuda", fake_torch.cuda)

        cb = MemoryUsageCallback()
        assert pytest.approx(300.0, rel=1e-6) == cb._get_gpu_memory()

    @pytest.mark.requires_trainable_backend
    def test_get_gpu_memory_jax(self, monkeypatch):
        monkeypatch.setattr(K, "backend", lambda: "jax")

        class FakeDev:
            platform = "gpu"

            def memory_stats(self):
                return {"bytes_in_use": 200 * 1024**2}

        fake_jax = MagicMock()
        fake_jax.devices.return_value = [FakeDev(), FakeDev()]
        monkeypatch.setitem(sys.modules, "jax", fake_jax)

        cb = MemoryUsageCallback()
        assert pytest.approx(400.0, rel=1e-6) == cb._get_gpu_memory()

    def test_running_on_gpu_and_tpu_flags(self):
        val_gpu = running_on_gpu()
        val_tpu = running_on_tpu()
        assert isinstance(val_gpu, bool)
        assert isinstance(val_tpu, bool)

    def test_psutil_missing(self):
        orig = sys.modules.pop("psutil", None)
        try:
            import keras.src.callbacks.memory_usage_callback as mod

            with patch.dict(sys.modules, {"psutil": None}):
                with pytest.raises(
                    ImportError,
                    match="MemoryUsageCallback requires the 'psutil' library",
                ):
                    reload(mod)
                    _ = mod.MemoryUsageCallback()
        finally:
            if orig is not None:
                sys.modules["psutil"] = orig
                from importlib import reload as _r

                _r(sys.modules["keras.src.callbacks.memory_usage_callback"])

    @pytest.mark.requires_trainable_backend
    def test_running_on_tpu_true(self, monkeypatch):
        monkeypatch.setattr(K, "backend", lambda: "tensorflow")
        fake_tf = MagicMock()
        fake_tf.config.list_logical_devices.return_value = ["TPU:0"]
        fake_tf.distribute.cluster_resolver.TPUClusterResolver = lambda: MagicMock()
        fake_tf.config.experimental_connect_to_cluster = lambda resolver: None
        fake_tf.tpu.experimental.initialize_tpu_system = lambda resolver: None
        monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)

        assert running_on_tpu() is True

    @pytest.mark.requires_trainable_backend
    def test_get_tpu_memory_tensorflow(self, monkeypatch):
        monkeypatch.setattr(K, "backend", lambda: "tensorflow")
        fake_tf = MagicMock()
        fake_tf.config.list_logical_devices.return_value = ["TPU:0"]
        fake_tf.distribute.cluster_resolver.TPUClusterResolver = lambda: MagicMock()
        fake_tf.config.experimental_connect_to_cluster = lambda resolver: None
        fake_tf.tpu.experimental.initialize_tpu_system = lambda resolver: None
        monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)

        cb = MemoryUsageCallback()
        assert cb._get_tpu_memory() is None

    @pytest.mark.requires_trainable_backend
    def test_get_tpu_memory_jax(self, monkeypatch):
        monkeypatch.setattr(K, "backend", lambda: "jax")

        class FakeTpuDev:
            platform = "tpu"

            def memory_stats(self):
                return {"bytes_in_use": 250 * 1024**2}

        fake_jax = MagicMock()
        fake_jax.devices.return_value = [FakeTpuDev()]
        monkeypatch.setitem(sys.modules, "jax", fake_jax)

        cb = MemoryUsageCallback()
        assert pytest.approx(250.0, rel=1e-6) == cb._get_tpu_memory()
