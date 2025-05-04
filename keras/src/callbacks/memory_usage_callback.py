import os
import warnings

from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback
from keras.src import backend as K

try:
    import psutil
except ImportError:
    psutil = None


@keras_export("keras.callbacks.MemoryUsageCallback")
class MemoryUsageCallback(Callback):
    """Monitor CPU/GPU/TPU/OpenVINO memory during training.

    Tracks:
        - CPU memory via `psutil.Process().memory_info().rss`.
        - GPU memory via backend APIs (TF, Torch, JAX, OpenVINO).
        - Logs to stdout and, optionally, to TensorBoard.

    Args:
        monitor_gpu: Bool. If True, query GPU/accelerator memory.
        log_every_batch: Bool. If True, log after each batch.
        tensorboard_log_dir: str or None. If set, use TF summary writer.

    Raises:
        ImportError: If `psutil` is missing.
    """

    def __init__(
        self, monitor_gpu=True, log_every_batch=False, tensorboard_log_dir=None
    ):
        super().__init__()
        if psutil is None:
            raise ImportError("MemoryUsageCallback requires the 'psutil' library.")
        self.monitor_gpu = monitor_gpu
        self.log_every_batch = log_every_batch
        self.process = psutil.Process()
        self.tb_writer = None
        self._batches_seen = 0

        if tensorboard_log_dir:
            try:
                import tensorflow as tf 

                logdir = os.path.expanduser(tensorboard_log_dir)
                self.tb_writer = tf.summary.create_file_writer(logdir)
            except Exception as e:
                warnings.warn(f"TB init error: {e}", RuntimeWarning)

    def on_train_begin(self, logs=None):
        self._batches_seen = 0

    def on_epoch_begin(self, epoch, logs=None):
        cpu = self._cpu_mem_mb()
        gpu = self._get_gpu_memory()
        self._log("Epoch %d start" % epoch, epoch, cpu, gpu)

    def on_epoch_end(self, epoch, logs=None):
        cpu = self._cpu_mem_mb()
        gpu = self._get_gpu_memory()
        self._log("Epoch %d end" % epoch, epoch + 1, cpu, gpu)

    def on_batch_end(self, batch, logs=None):
        if self.log_every_batch:
            cpu = self._cpu_mem_mb()
            gpu = self._get_gpu_memory()
            self._log(f"Batch {self._batches_seen} end", self._batches_seen, cpu, gpu)
        self._batches_seen += 1

    def on_train_end(self, logs=None):
        if self.tb_writer:
            self.tb_writer.close()

    def _cpu_mem_mb(self):
        return self.process.memory_info().rss / (1024**2)

    def _get_gpu_memory(self):
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
                for g in gpus:
                    info = tf.config.experimental.get_memory_info(g.name)
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

                devs = [d for d in jax.devices() if d.platform.upper() == "GPU"]
                if not devs:
                    return None
                total = 0
                for d in devs:
                    stats = d.memory_stats()
                    total += stats.get("bytes_in_use", 0)
                return total / (1024**2)
            if backend == "openvino":
                try:
                    import openvino as ov  

                    core = ov.Core()
                    devices = core.available_devices
                    total = 0
                    for dev in devices:
                        stats = core.get_property(dev, "DEVICE_MEMORY_STATISTICS")
                        total += stats.get("deviceUsedBytes", 0)
                    return total / (1024**2)
                except Exception as e:
                    warnings.warn(f"OVINO mem err: {e}", RuntimeWarning)
                    return None
        except ImportError as e:
            warnings.warn(f"Import err for {backend}: {e}", RuntimeWarning)
            return None
        warnings.warn(f"Unsupported backend '{backend}'", RuntimeWarning)
        return None

    def _log(self, label, step, cpu, gpu):
        msg = f"{label} - CPU: {cpu:.2f} MB"
        if gpu is not None:
            msg += f"; GPU: {gpu:.2f} MB"
        print(msg)
        if self.tb_writer:
            import tensorflow as tf  

            with self.tb_writer.as_default(step=step):
                tf.summary.scalar("Memory/CPU_MB", cpu)
                if gpu is not None:
                    tf.summary.scalar("Memory/GPU_MB", gpu)
            self.tb_writer.flush()
