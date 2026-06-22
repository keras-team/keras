"""Tests for MemoryUsageCallback."""

from unittest.mock import MagicMock, patch

import pytest

from keras.src.callbacks.memory_usage_callback import MemoryUsageCallback


class TestMemoryUsageCallback:
    """Unit tests for MemoryUsageCallback."""

    # ------------------------------------------------------------------
    # Construction & attribute defaults
    # ------------------------------------------------------------------

    def test_default_attributes(self):
        cb = MemoryUsageCallback()
        assert cb.log_per_batch is False
        assert cb.verbose == 0

    def test_custom_attributes(self):
        cb = MemoryUsageCallback(log_per_batch=True, verbose=1)
        assert cb.log_per_batch is True
        assert cb.verbose == 1

    # ------------------------------------------------------------------
    # _cpu_memory_gb
    # ------------------------------------------------------------------

    def test_cpu_memory_gb_returns_float_when_psutil_available(self):
        cb = MemoryUsageCallback()
        cb._psutil_available = True
        result = cb._cpu_memory_gb()
        # psutil should be available in the test environment
        if result is not None:
            assert isinstance(result, float)
            assert result > 0

    def test_cpu_memory_gb_returns_none_when_psutil_unavailable(self):
        cb = MemoryUsageCallback()
        cb._psutil_available = False
        assert cb._cpu_memory_gb() is None

    # ------------------------------------------------------------------
    # _collect_stats
    # ------------------------------------------------------------------

    def test_collect_stats_includes_cpu_key_when_psutil_available(self):
        cb = MemoryUsageCallback()
        cb._psutil_available = True
        with patch.object(cb, "_gpu_memory_gb", return_value=None):
            stats = cb._collect_stats()
        assert "memory/cpu_used_gb" in stats
        assert isinstance(stats["memory/cpu_used_gb"], float)

    def test_collect_stats_includes_gpu_key_when_gpu_available(self):
        cb = MemoryUsageCallback()
        cb._psutil_available = False
        with patch.object(cb, "_gpu_memory_gb", return_value=1.5):
            stats = cb._collect_stats()
        assert "memory/gpu_used_gb" in stats
        assert stats["memory/gpu_used_gb"] == 1.5

    def test_collect_stats_omits_gpu_key_when_no_gpu(self):
        cb = MemoryUsageCallback()
        cb._psutil_available = False
        with patch.object(cb, "_gpu_memory_gb", return_value=None):
            stats = cb._collect_stats()
        assert "memory/gpu_used_gb" not in stats

    def test_collect_stats_rounds_to_four_decimal_places(self):
        cb = MemoryUsageCallback()
        cb._psutil_available = False
        with patch.object(cb, "_gpu_memory_gb", return_value=1.23456789):
            stats = cb._collect_stats()
        assert stats["memory/gpu_used_gb"] == round(1.23456789, 4)

    # ------------------------------------------------------------------
    # on_epoch_end
    # ------------------------------------------------------------------

    def test_on_epoch_end_updates_logs(self):
        cb = MemoryUsageCallback()
        fake_stats = {"memory/cpu_used_gb": 1.0, "memory/gpu_used_gb": 0.5}
        with patch.object(cb, "_collect_stats", return_value=fake_stats):
            logs = {"loss": 0.3}
            cb.on_epoch_end(0, logs=logs)
        assert logs["memory/cpu_used_gb"] == 1.0
        assert logs["memory/gpu_used_gb"] == 0.5
        assert logs["loss"] == 0.3  # existing keys preserved

    def test_on_epoch_end_handles_none_logs(self):
        cb = MemoryUsageCallback()
        fake_stats = {"memory/cpu_used_gb": 1.0}
        with patch.object(cb, "_collect_stats", return_value=fake_stats):
            cb.on_epoch_end(0, logs=None)  # must not raise

    def test_on_epoch_end_prints_when_verbose(self, capsys):
        cb = MemoryUsageCallback(verbose=1)
        fake_stats = {"memory/cpu_used_gb": 1.2345}
        with patch.object(cb, "_collect_stats", return_value=fake_stats):
            cb.on_epoch_end(2, logs={})
        captured = capsys.readouterr()
        assert "Epoch 3" in captured.out
        assert "memory/cpu_used_gb" in captured.out

    def test_on_epoch_end_silent_when_not_verbose(self, capsys):
        cb = MemoryUsageCallback(verbose=0)
        fake_stats = {"memory/cpu_used_gb": 1.0}
        with patch.object(cb, "_collect_stats", return_value=fake_stats):
            cb.on_epoch_end(0, logs={})
        captured = capsys.readouterr()
        assert captured.out == ""

    # ------------------------------------------------------------------
    # on_train_batch_end
    # ------------------------------------------------------------------

    def test_on_train_batch_end_updates_logs_when_log_per_batch(self):
        cb = MemoryUsageCallback(log_per_batch=True)
        fake_stats = {"memory/cpu_used_gb": 0.8}
        with patch.object(cb, "_collect_stats", return_value=fake_stats):
            logs = {}
            cb.on_train_batch_end(0, logs=logs)
        assert "memory/cpu_used_gb" in logs

    def test_on_train_batch_end_skips_when_not_log_per_batch(self):
        cb = MemoryUsageCallback(log_per_batch=False)
        mock_collect = MagicMock(return_value={"memory/cpu_used_gb": 0.8})
        with patch.object(cb, "_collect_stats", mock_collect):
            logs = {}
            cb.on_train_batch_end(0, logs=logs)
        mock_collect.assert_not_called()
        assert logs == {}

    # ------------------------------------------------------------------
    # Backend-specific GPU helpers (smoke tests)
    # ------------------------------------------------------------------

    def test_gpu_memory_tf_returns_none_when_no_gpus(self):
        tf_mock = MagicMock()
        tf_mock.config.list_physical_devices.return_value = []
        with patch.dict("sys.modules", {"tensorflow": tf_mock}):
            result = MemoryUsageCallback._gpu_memory_tf()
        assert result is None

    def test_gpu_memory_torch_returns_none_when_no_cuda_or_mps(self):
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = False
        torch_mock.backends.mps.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": torch_mock}):
            result = MemoryUsageCallback._gpu_memory_torch()
        assert result is None

    def test_gpu_memory_torch_cuda(self):
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = True
        torch_mock.cuda.memory_allocated.return_value = 1024**3  # 1 GiB
        with patch.dict("sys.modules", {"torch": torch_mock}):
            result = MemoryUsageCallback._gpu_memory_torch()
        assert result == pytest.approx(1.0)

    def test_gpu_memory_jax_returns_none_on_cpu_platform(self):
        jax_mock = MagicMock()
        cpu_device = MagicMock()
        cpu_device.platform = "cpu"
        jax_mock.devices.return_value = [cpu_device]
        with patch.dict("sys.modules", {"jax": jax_mock}):
            result = MemoryUsageCallback._gpu_memory_jax()
        assert result is None
