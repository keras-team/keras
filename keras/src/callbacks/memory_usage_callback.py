
import os
import warnings
import time

from keras.src import backend as K
from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback

try:
    import psutil
except ImportError:
    psutil = None


def running_on_gpu():
    """Detect if any GPU is available on the current Keras backend."""
    backend_name = K.backend()
    if backend_name == "tensorflow":
        import tensorflow as tf
        return bool(tf.config.list_logical_devices("GPU"))
    elif backend_name == "torch":
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    elif backend_name == "jax":
        try:
            import jax
            return any(d.platform.upper() == "GPU" for d in jax.devices())
        except ImportError:
            return False
    return False


def running_on_tpu():
    """Detect if any TPU is available on the current Keras backend."""
    backend_name = K.backend()
    if backend_name == "tensorflow":
        import tensorflow as tf
        return bool(tf.config.list_logical_devices("TPU"))
    elif backend_name == "jax":
        try:
            import jax
            return any(d.platform.upper() == "TPU" for d in jax.devices())
        except ImportError:
            return False
    return False


@keras_export("keras.callbacks.MemoryUsageCallback")
class MemoryUsageCallback(Callback):
    """
    Monitors and logs memory usage (CPU + optional GPU/TPU) during training.

    This callback measures:
      - **CPU**: via psutil.Process().memory_info().rss
      - **GPU**: if a GPU is detected, via backend-specific APIs
        (TensorFlow, PyTorch, JAX)
      - **TPU**: if a TPU is detected, via backend-specific APIs
        (TensorFlow, JAX)

    Logs are printed to stdout at the start and end of each epoch
    (with a leading newline to avoid clobbering the progress bar),
    and, if `log_every_batch=True`, after every batch.
    If `tensorboard_log_dir` is provided, scalars are also written
    via tf.summary (TensorBoard).

    Args:
        log_every_batch (bool): If True, also log after each batch. Defaults to False
            (i.e., log only at epoch start and end).
        tensorboard_log_dir (str|None): Directory for TensorBoard logs; if None,
            no TF summary writer is created.

    Raises:
        ImportError: If `psutil` is not installed (required for CPU logging).
    """

    def __init__(
        self,
        log_every_batch=False,
        tensorboard_log_dir=None,
    ):
        super().__init__()

        if psutil is None:
            raise ImportError(
                "MemoryUsageCallback requires the 'psutil' library. "
                "To install, please use: pip install psutil"
            )

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
                    f"Could not initialize TensorBoard writer: {e}",
                    RuntimeWarning,
                )
                self._writer = None

    def on_train_begin(self, logs=None):
        self._step_counter = 0

    def on_epoch_begin(self, epoch, logs=None):
        print()
        self._log_epoch("start", epoch)

    def on_epoch_end(self, epoch, logs=None):
        print()
        self._log_epoch("end", epoch, offset=1)

    def on_batch_end(self, batch, logs=None):
        if self.log_every_batch:
            print()
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
        """
        Internal helper to measure and print CPU/GPU/TPU memory.
        Inserts a short delay (time.sleep(0)) to let stdout flush cleanly.
        """
        cpu_mb = self._get_cpu_memory()
        gpu_mb = self._get_gpu_memory()
        tpu_mb = self._get_tpu_memory()

        msg = f"{label} - CPU Memory: {cpu_mb:.2f} MB"
        if gpu_mb is not None:
            msg += f"; GPU Memory: {gpu_mb:.2f} MB"
        if tpu_mb is not None:
            msg += f"; TPU Memory: {tpu_mb:.2f} MB"

        print(msg)
        time.sleep(0)

        if self._writer:
            import tensorflow as tf

            with self._writer.as_default(step=int(step)):
                tf.summary.scalar("Memory/CPU_MB", cpu_mb)
                if gpu_mb is not None:
                    tf.summary.scalar("Memory/GPU_MB", gpu_mb)
                if tpu_mb is not None:
                    tf.summary.scalar("Memory/TPU_MB", tpu_mb)

    def _get_cpu_memory(self):
        """Return current process CPU memory usage in MB."""
        return self._proc.memory_info().rss / (1024**2)

    def _get_gpu_memory(self):
        """
        Return current GPU memory usage in MB for the detected backend,
        or None if no GPU is present or if measurement fails.
        """
        if not running_on_gpu():
            return None

        backend_name = K.backend()
        try:
            if backend_name == "tensorflow":
                import tensorflow as tf
                try:
                    mem_info = tf.config.experimental.get_memory_info("GPU:0")
                    return mem_info["current"] / (1024**2)
                except Exception:
                    gpus = tf.config.list_physical_devices("GPU")
                    if not gpus:
                        return None
                    total = 0
                    for i, _ in enumerate(gpus):
                        try:
                            info = tf.config.experimental.get_memory_info(f"GPU:{i}")
                            total += info.get("current", 0)
                        except Exception:
                            continue
                    return total / (1024**2)

            elif backend_name == "torch":
                import torch

                if not torch.cuda.is_available():
                    return None

                device_count = torch.cuda.device_count()
                total_bytes = 0
                for i in range(device_count):
                    total_bytes += torch.cuda.memory_allocated(i)
                return total_bytes / (1024**2)

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

            return None

        except ImportError as imp_err:
            if not hasattr(self, "_warn_import"):
                warnings.warn(
                    f"Could not import library for GPU memory tracking ({backend_name}): {imp_err}",
                    RuntimeWarning,
                )
                self._warn_import = True
            return None

        except Exception as exc:
            if not hasattr(self, "_warn_exc"):
                warnings.warn(
                    f"Error retrieving GPU memory: {exc}", RuntimeWarning
                )
                self._warn_exc = True
            return None

    def _get_tpu_memory(self):
        """
        Return current TPU memory usage in MB for the detected backend,
        or None if no TPU is present or if measurement fails.
        Note: TPU memory APIs vary; here we attempt best‐effort.
        """
        if not running_on_tpu():
            return None

        backend_name = K.backend()
        try:
            if backend_name == "tensorflow":
                import tensorflow as tf

                if not hasattr(self, "_warn_tpu_tf"):
                    warnings.warn(
                        "TensorFlow TPU memory info is not directly available; returning None.",
                        RuntimeWarning,
                    )
                    self._warn_tpu_tf = True
                return None

            elif backend_name == "jax":
                import jax

                devs = [d for d in jax.devices() if d.platform.upper() == "TPU"]
                if not devs:
                    return None
                try:
                    stats = devs[0].memory_stats()
                    tpu_bytes = stats.get("bytes_in_use", stats.get("allocated_bytes", 0))
                    return tpu_bytes / (1024**2)
                except Exception:
                    if not hasattr(self, "_warn_tpu_jax"):
                        warnings.warn(
                            "Failed to retrieve JAX TPU memory stats; returning None.",
                            RuntimeWarning,
                        )
                        self._warn_tpu_jax = True
                    return None

            return None

        except ImportError as imp_err:
            if not hasattr(self, "_warn_tpu_imp"):
                warnings.warn(
                    f"Could not import library for TPU memory tracking ({backend_name}): {imp_err}",
                    RuntimeWarning,
                )
                self._warn_tpu_imp = True
            return None

        except Exception as exc:
            if not hasattr(self, "_warn_tpu_exc"):
                warnings.warn(
                    f"Error retrieving TPU memory: {exc}", RuntimeWarning
                )
                self._warn_tpu_exc = True
            return None
