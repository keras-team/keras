import os
import warnings

from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback
from keras.src import backend as K

# Attempt to import psutil for CPU memory
try:
    import psutil
except ImportError:
    psutil = None


@keras_export("keras.callbacks.MemoryUsageCallback")
class MemoryUsageCallback(Callback):
    """
    Monitors CPU and GPU memory across backends and logs to stdout and TensorBoard.

    Example:
    ```python
    from keras.callbacks import MemoryUsageCallback
    callback = MemoryUsageCallback(
        monitor_gpu=True,
        log_every_batch=False,
        tensorboard_log_dir="./logs"
    )
    model.fit(..., callbacks=[callback])
    ```

    Args:
        monitor_gpu (bool): Whether to log GPU memory. Defaults to True.
        log_every_batch (bool): Whether to log after every batch. Defaults to False.
        tensorboard_log_dir (str): Directory for TensorBoard logs; None disables. Defaults to None.

    Raises:
        ImportError: If psutil is not installed.
    """

    def __init__(
        self,
        monitor_gpu=True,
        log_every_batch=False,
        tensorboard_log_dir=None,
    ):
        super().__init__()
        if psutil is None:
            raise ImportError(
                "MemoryUsageCallback requires `psutil`; install via `pip install psutil`."
            )
        self.monitor_gpu = monitor_gpu
        self.log_every_batch = log_every_batch
        self.process = psutil.Process()
        self.tb_writer = None
        self._batch_count = 0

        if tensorboard_log_dir:
            try:
                import tensorflow as tf  

                logdir = os.path.expanduser(tensorboard_log_dir)
                self.tb_writer = tf.summary.create_file_writer(logdir)
            except ImportError as e:
                warnings.warn(f"TensorBoard disabled (no TF): {e}", RuntimeWarning)
            except Exception as e:
                warnings.warn(
                    f"Failed to init TB writer at {tensorboard_log_dir}: {e}",
                    RuntimeWarning,
                )

    def _get_cpu_memory(self):
        """Return resident set size in MB."""
        return self.process.memory_info().rss / (1024**2)

    def _get_gpu_memory(self):
        """Return GPU memory usage in MB or None."""
        if not self.monitor_gpu:
            return None
        backend = K.backend()
        try:
            if backend == "tensorflow":
                import tensorflow as tf  

                gpus = tf.config.list_physical_devices("GPU")
                if not gpus:
                    return None
                total = 0
                for gpu in gpus:
                    info = tf.config.experimental.get_memory_info(gpu.name)
                    total += info.get("current", 0)
                return total / (1024**2)

            if backend == "torch":
                import torch  

                if not torch.cuda.is_available():
                    return None
                total = sum(
                    torch.cuda.memory_allocated(i)
                    for i in range(torch.cuda.device_count())
                )
                return total / (1024**2)

            if backend == "jax":
                import jax  

                devs = [d for d in jax.devices() if d.platform == "gpu"]
                if not devs:
                    return None
                total = 0
                for d in devs:
                    stats = getattr(d, "memory_stats", lambda: {})()
                    total += stats.get("bytes_in_use", stats.get("allocated_bytes", 0))
                return total / (1024**2)

            if not hasattr(self, "_warned_backend"):
                warnings.warn(
                    f"Backend '{backend}' not supported for GPU memory.",
                    RuntimeWarning,
                )
                self._warned_backend = True
            return None

        except ImportError as e:
            warnings.warn(
                f"Could not import backend lib ({e}); GPU disabled.",
                RuntimeWarning,
            )
            return None
        except Exception as e:
            warnings.warn(f"Error retrieving GPU memory ({e}).", RuntimeWarning)
            return None

    def _log(self, label, step):
        cpu = self._get_cpu_memory()
        gpu = self._get_gpu_memory()
        msg = f"{label} - CPU Memory: {cpu:.2f} MB"
        if gpu is not None:
            msg += f"; GPU Memory: {gpu:.2f} MB"
        print(msg)
        if self.tb_writer:
            import tensorflow as tf  

            with self.tb_writer.as_default(step=int(step)):
                tf.summary.scalar("Memory/CPU_MB", cpu)
                if gpu is not None:
                    tf.summary.scalar("Memory/GPU_MB", gpu)
            self.tb_writer.flush()

    def on_train_begin(self, logs=None):
        self._batch_count = 0

    def on_epoch_begin(self, epoch, logs=None):
        self._log(f"Epoch {epoch} start", epoch)

    def on_epoch_end(self, epoch, logs=None):
        self._log(f"Epoch {epoch} end", epoch + 1)

    def on_batch_end(self, batch, logs=None):
        if self.log_every_batch:
            self._log(f"Batch {self._batch_count} end", self._batch_count)
        self._batch_count += 1

    def on_train_end(self, logs=None):
        if self.tb_writer:
            self.tb_writer.close()
