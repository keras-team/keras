import warnings

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback
from keras.src.utils import io_utils


@keras_export("keras.callbacks.MemoryUsageCallback")
class MemoryUsageCallback(Callback):
    """Callback that logs CPU and GPU/accelerator memory usage during training.

    This callback tracks memory consumption at the end of each epoch (and
    optionally each batch) and injects the measurements into the `logs`
    dictionary so they are visible in the progress bar, accessible via the
    `History` callback, and available to other callbacks such as
    `keras.callbacks.CSVLogger`.

    Reported metrics:

    - **``memory/cpu_used_gb``** – process RSS (Resident Set Size) in GiB,
      sampled via `psutil`.
    - **``memory/gpu_used_gb``** – current device memory allocated on the
      active accelerator in GiB.  The source depends on the Keras backend:

      * **TensorFlow** – `tf.config.experimental.get_memory_info`
      * **PyTorch** – `torch.cuda.memory_allocated` /
        `torch.mps.driver_allocated_memory`
      * **JAX** – `jax.devices()[0]` live-buffer query via
        `jax.device_put` round-trip (best-effort; ``0`` when unsupported)

      When no GPU / accelerator is available the metric is omitted.

    The callback degrades gracefully: if `psutil` is not installed only GPU
    memory is reported; if no accelerator is present only CPU memory is
    reported.

    Args:
        log_per_batch: Boolean.  When ``True``, memory is also sampled at the
            end of each *training* batch and added to the batch-level logs.
            Default is ``False`` (epoch-level only) to minimise the overhead
            in tight training loops.
        verbose: Integer, 0 or 1.  When ``1``, a summary line is printed to
            stdout at the end of each epoch.  Default is ``0``.

    Example:

    ```python
    callback = keras.callbacks.MemoryUsageCallback(verbose=1)
    model.fit(X_train, y_train, epochs=5, callbacks=[callback])
    # Epoch 1/5  …  memory/cpu_used_gb: 1.23  memory/gpu_used_gb: 0.87
    ```

    Accessing logged values via the History callback:

    ```python
    history = model.fit(X_train, y_train, callbacks=[
        keras.callbacks.MemoryUsageCallback()
    ])
    print(history.history["memory/cpu_used_gb"])
    ```
    """

    def __init__(self, log_per_batch: bool = False, verbose: int = 0):
        super().__init__()
        self.log_per_batch = log_per_batch
        self.verbose = verbose

        self._psutil_available = self._check_psutil()
        self._backend = backend.backend()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_psutil() -> bool:
        try:
            import psutil  # noqa: F401

            return True
        except ImportError:
            warnings.warn(
                "MemoryUsageCallback: `psutil` is not installed. "
                "CPU memory usage will not be reported. "
                "Install it with `pip install psutil`.",
                stacklevel=3,
            )
            return False

    def _cpu_memory_gb(self) -> float | None:
        """Return process RSS in GiB, or None if psutil is unavailable."""
        if not self._psutil_available:
            return None
        import os

        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024**3)

    def _gpu_memory_gb(self) -> float | None:
        """Return current accelerator memory usage in GiB, or None."""
        try:
            if self._backend == "tensorflow":
                return self._gpu_memory_tf()
            elif self._backend == "torch":
                return self._gpu_memory_torch()
            elif self._backend == "jax":
                return self._gpu_memory_jax()
        except Exception:
            pass
        return None

    @staticmethod
    def _gpu_memory_tf() -> float | None:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return None
        info = tf.config.experimental.get_memory_info("GPU:0")
        return info["current"] / (1024**3)

    @staticmethod
    def _gpu_memory_torch() -> float | None:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            return torch.mps.driver_allocated_memory() / (1024**3)
        return None

    @staticmethod
    def _gpu_memory_jax() -> float | None:
        try:
            import jax

            devices = jax.devices()
            if not devices or devices[0].platform == "cpu":
                return None
            # Sum live buffer memory across all devices of the first platform
            total = sum(
                buf.nbytes
                for device in devices
                for buf in device.live_buffers()
            )
            return total / (1024**3)
        except Exception:
            return None

    def _collect_stats(self) -> dict:
        """Collect all available memory stats into a flat dict."""
        stats = {}
        cpu = self._cpu_memory_gb()
        if cpu is not None:
            stats["memory/cpu_used_gb"] = round(cpu, 4)
        gpu = self._gpu_memory_gb()
        if gpu is not None:
            stats["memory/gpu_used_gb"] = round(gpu, 4)
        return stats

    def _maybe_print(self, epoch: int, stats: dict) -> None:
        if self.verbose and stats:
            parts = "  ".join(f"{k}: {v:.4f}" for k, v in stats.items())
            io_utils.print_msg(f"Epoch {epoch + 1}: {parts}")

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def on_epoch_end(self, epoch, logs=None):
        """Inject memory stats into epoch-level logs."""
        stats = self._collect_stats()
        if logs is not None:
            logs.update(stats)
        self._maybe_print(epoch, stats)

    def on_train_batch_end(self, batch, logs=None):
        """Optionally inject memory stats into batch-level logs."""
        if not self.log_per_batch:
            return
        stats = self._collect_stats()
        if logs is not None:
            logs.update(stats)
