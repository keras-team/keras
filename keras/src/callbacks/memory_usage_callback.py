import os
import warnings
from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback
from keras.src import backend as K

# Attempt to import psutil for memory monitoring
try:
    import psutil
except ImportError:
    psutil = None  

@keras_export("keras.callbacks.MemoryUsageCallback")
class MemoryUsageCallback(Callback):
    """Monitors and logs memory usage (CPU + optional GPU/TPU) during training.

    This callback measures:

      - **CPU**: via psutil.Process().memory_info().rss
      - **GPU/TPU**: via backend‐specific APIs (TensorFlow, PyTorch, JAX, OpenVINO)

    Logs are printed to stdout at the start/end of each epoch and,
    if `log_every_batch=True`, after every batch.  If `tensorboard_log_dir`
    is provided, scalars are also written via `tf.summary` (TensorBoard).

    Args:
        monitor_gpu (bool): If True, attempt to measure accelerator memory.
        log_every_batch (bool): If True, also log after each batch.
        tensorboard_log_dir (str|None): Directory for TensorBoard logs;
            if None, no TF summary writer is created.

    Raises:
        ImportError: If `psutil` is not installed (required for CPU logging).

    Example:

    ```python
    from keras.callbacks import MemoryUsageCallback
    # ...
    cb = MemoryUsageCallback(
        monitor_gpu=True,
        log_every_batch=False,
        tensorboard_log_dir="./logs/memory"
    )
    model.fit(X, y, callbacks=[cb])
    ```
    """

    def __init__(
        self, monitor_gpu=True, log_every_batch=False, tensorboard_log_dir=None
    ):
        super().__init__()
        if psutil is None:
            raise ImportError(
                "MemoryUsageCallback requires the 'psutil' library. "
                "Install via `pip install psutil`."
            )
        self.monitor_gpu = monitor_gpu
        self.log_every_batch = log_every_batch
        self._proc = psutil.Process()
        self._step_counter = 0
        self._writer = None

        if tensorboard_log_dir:
            try:
                import tensorflow as tf  

                logdir = os.path.expanduser(tensorboard_log_dir)
                self._writer = tf.summary.create_file_writer(logdir)
                print(f"MemoryUsageCallback: TensorBoard logs → {logdir}")
            except Exception as e:
                warnings.warn(
                    f"Could not initialize TensorBoard writer: {e}", RuntimeWarning
                )
                self._writer = None

    def on_train_begin(self, logs=None):
        self._step_counter = 0

    def on_epoch_begin(self, epoch, logs=None):
        self._log_epoch("start", epoch)

    def on_epoch_end(self, epoch, logs=None):
        self._log_epoch("end", epoch, offset=1)

    def on_batch_end(self, batch, logs=None):
        if self.log_every_batch:
            self._log_step(f"Batch {self._step_counter} end", self._step_counter)
        self._step_counter += 1

    def on_train_end(self, logs=None):
        if self._writer:
            self._writer.close()

    def _log_epoch(self, when, epoch, offset=0):
        label = f"Epoch {epoch} {when}"
        step = epoch + offset
        self._log_step(label, step)

    def _log_step(self, label, step):
        cpu_mb = self._get_cpu_memory()
        gpu_mb = self._get_gpu_memory() if self.monitor_gpu else None

        msg = f"{label} - CPU Memory: {cpu_mb:.2f} MB"
        if gpu_mb is not None:
            msg += f"; GPU Memory: {gpu_mb:.2f} MB"
        print(msg)

        if self._writer:
            import tensorflow as tf  # noqa: E501

            with self._writer.as_default(step=int(step)):
                tf.summary.scalar("Memory/CPU_MB", cpu_mb)
                if gpu_mb is not None:
                    tf.summary.scalar("Memory/GPU_MB", gpu_mb)
            self._writer.flush()

    def _get_cpu_memory(self):
        return self._proc.memory_info().rss / (1024**2)

    def _get_gpu_memory(self):
        backend_name = K.backend()
        try:
            if backend_name == "tensorflow":
                import tensorflow as tf  

                gpus = tf.config.list_physical_devices("GPU")
                if not gpus:
                    return None
                total = sum(
                    tf.config.experimental.get_memory_info(g.name)["current"]
                    for g in gpus
                )
                return total / (1024**2)
            elif backend_name == "torch":
                import torch

                if not torch.cuda.is_available():
                    return None
                total = sum(
                    torch.cuda.memory_allocated(i)
                    for i in range(torch.cuda.device_count())
                )
                return total / (1024**2)
            elif backend_name == "jax":
                import jax

                devs = [d for d in jax.devices() if d.platform.upper() == "GPU"]
                if not devs:
                    return None
                total = 0
                for d in devs:
                    stats = getattr(d, "memory_stats", lambda: {})()
                    total += stats.get("bytes_in_use", 0)
                return total / (1024**2)
            else:
                # OpenVINO and others fall back to unsupported

                if not hasattr(self, "_warn_backend"):
                    warnings.warn(
                        f"MemoryUsageCallback: unsupported backend '{backend_name}'",
                        RuntimeWarning,
                    )
                    self._warn_backend = True
                return None
        except ImportError as imp_err:
            if not hasattr(self, "_warn_import"):
                warnings.warn(
                    f"Could not import for backend '{backend_name}': {imp_err}",
                    RuntimeWarning,
                )
                self._warn_import = True
            return None
        except Exception as exc:
            if not hasattr(self, "_warn_exc"):
                warnings.warn(f"Error retrieving GPU memory: {exc}", RuntimeWarning)
                self._warn_exc = True
            return None
