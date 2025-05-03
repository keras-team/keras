import os
import warnings
import tensorflow as tf  # Ensure TF is imported for tf.summary
from keras.src import backend as K
from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback

# Attempt to import psutil and warn if unavailable.
try:
    import psutil
except ImportError:
    psutil = None

@keras_export("keras.callbacks.MemoryUsageCallback")
class MemoryUsageCallback(Callback):
    """Callback for enhanced monitoring of memory usage during training.

    This callback tracks CPU memory usage via `psutil.Process().memory_info().rss`
    and optionally GPU memory usage via backend-specific APIs (TensorFlow, PyTorch, JAX).
    Memory statistics are logged to stdout at the start and end of each epoch and,
    optionally, after every batch. Additionally, metrics are logged to TensorBoard
    if a log directory is provided, using integer steps for proper visualization.

    Note: GPU memory reporting consistency across backends (TF, PyTorch, JAX)
    may vary, as they use different underlying mechanisms to measure usage
    (e.g., framework overhead vs. purely tensor allocations).

    Args:
        monitor_gpu (bool): Whether to monitor GPU memory. Defaults to True.
            Requires appropriate backend (TensorFlow, PyTorch, JAX) with GPU
            support and necessary drivers/libraries installed.
        log_every_batch (bool): Whether to log memory usage after each batch
            in addition to epoch start/end. Defaults to False.
        tensorboard_log_dir (str, optional): Path to the directory where TensorBoard
            logs will be written using `tf.summary`. If None, TensorBoard logging
            is disabled. Defaults to None. Requires TensorFlow to be installed.

    Raises:
        ImportError: If `psutil` is not installed.

    Example:
    ```python
    import tensorflow as tf
    import keras
    from keras.callbacks import MemoryUsageCallback # Use public API path
    import numpy as np

    # Ensure psutil is installed: pip install psutil

    memory_callback = MemoryUsageCallback(
        monitor_gpu=True, # Set based on GPU availability and backend support
        log_every_batch=False,
        tensorboard_log_dir="~/logs/memory_usage" # Needs TF installed
    )

    model = keras.models.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(100,)),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    x_train = np.random.random((100, 100))
    y_train = keras.utils.to_categorical(
        np.random.randint(10, size=(100, 1)), num_classes=10
    )
    model.fit(x_train, y_train, epochs=5, batch_size=32, callbacks=[memory_callback])
    ```
    """
    def __init__(self, monitor_gpu=True, log_every_batch=False, tensorboard_log_dir=None):
        super().__init__()
        if psutil is None:
            raise ImportError(
                "MemoryUsageCallback requires the 'psutil' library. "
                "Please install it using 'pip install psutil'."
            )
        self.monitor_gpu = monitor_gpu
        self.log_every_batch = log_every_batch
        self.process = psutil.Process()
        self.tb_writer = None
        self._total_batches_seen = 0  # For TensorBoard step counting

        if tensorboard_log_dir:
            # tf.summary requires TensorFlow installed.
            if tf is None:
                 warnings.warn(
                     "MemoryUsageCallback: TensorFlow is required for TensorBoard logging. "
                     "Please install TensorFlow.", ImportWarning
                 )
                 self.tb_writer = None
            else:
                try:
                    log_dir = os.path.expanduser(tensorboard_log_dir)
                    # Use tf.summary for robust integration
                    self.tb_writer = tf.summary.create_file_writer(log_dir)
                    print(f"MemoryUsageCallback: TensorBoard logging initialized at {log_dir}")
                except Exception as e:
                    warnings.warn(f"Error initializing TensorBoard writer: {e}", RuntimeWarning)
                    self.tb_writer = None

    def on_train_begin(self, logs=None):
        """Reset batch counter at the start of training."""
        self._total_batches_seen = 0

    def on_epoch_begin(self, epoch, logs=None):
        """Log memory usage at the beginning of each epoch."""
        cpu_mem = self._get_cpu_memory()
        gpu_mem = self._get_gpu_memory()
        self._log_memory(
            label=f"Epoch {epoch} start",
            step=epoch, # Use epoch number for TB step
            cpu_mem=cpu_mem,
            gpu_mem=gpu_mem
        )

    def on_epoch_end(self, epoch, logs=None):
        """Log memory usage at the end of each epoch."""
        cpu_mem = self._get_cpu_memory()
        gpu_mem = self._get_gpu_memory()
        # Use epoch + 1 for TB step to mark the end point distinctly
        self._log_memory(
            label=f"Epoch {epoch} end",
            step=epoch + 1,
            cpu_mem=cpu_mem,
            gpu_mem=gpu_mem
        )

    def on_batch_end(self, batch, logs=None):
        """If enabled, log memory usage at the end of each batch."""
        if self.log_every_batch:
            cpu_mem = self._get_cpu_memory()
            gpu_mem = self._get_gpu_memory()
            # Use the total batches seen count for a continuous TB step
            self._log_memory(
                label=f"Batch {self._total_batches_seen} end",
                step=self._total_batches_seen,
                cpu_mem=cpu_mem,
                gpu_mem=gpu_mem
            )
        # Always increment, even if not logging
        self._total_batches_seen += 1

    def on_train_end(self, logs=None):
        """Clean up the TensorBoard writer."""
        if self.tb_writer:
            self.tb_writer.close()
            self.tb_writer = None

    def _get_cpu_memory(self):
        """Return current process CPU memory usage in MB."""
        return self.process.memory_info().rss / (1024 ** 2)

    def _get_gpu_memory(self):
        """Return current GPU memory usage in MB based on backend."""
        if not self.monitor_gpu:
            return None

        backend = K.backend()
        gpu_mem_mb = None
        try:
            if backend == "tensorflow":
                gpus = tf.config.list_physical_devices("GPU")
                if not gpus: return None
                total_mem_bytes = 0
                for gpu in gpus:
                    mem_info = tf.config.experimental.get_memory_info(gpu.name)
                    total_mem_bytes += mem_info.get("current", 0)
                gpu_mem_mb = total_mem_bytes / (1024 ** 2)

            elif backend == "torch":
                # Note: memory_allocated() tracks only tensors, might differ from TF.
                import torch
                if not torch.cuda.is_available(): return None
                # Sum memory allocated across all visible GPUs
                total_mem_bytes = sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count()))
                gpu_mem_mb = total_mem_bytes / (1024 ** 2)

            elif backend == "jax":
                # Note: JAX memory stats might also differ from TF/Torch in scope.
                import jax
                devices = jax.devices()
                gpu_devices = [d for d in devices if d.platform.upper() == 'GPU'] # Filter for GPU devices
                if not gpu_devices: return None
                total_mem_bytes = 0
                for device in gpu_devices:
                    try:
                        # memory_stats() might not be available or API could change
                        stats = device.memory_stats()
                        total_mem_bytes += stats.get("bytes_in_use", stats.get("allocated_bytes", 0)) # Try common keys
                    except Exception:
                        # Ignore if stats are unavailable for a device
                        pass
                gpu_mem_mb = total_mem_bytes / (1024 ** 2)

            else:
                if not hasattr(self, '_backend_warned'):
                    warnings.warn(f"Unsupported backend '{backend}' for GPU memory monitoring.", RuntimeWarning)
                    self._backend_warned = True
                return None

        except ImportError as e:
            # Backend library might not be installed
             if not hasattr(self, f'_{backend}_import_warned'):
                 warnings.warn(f"MemoryUsageCallback: Could not import library for backend '{backend}': {e}. "
                               f"GPU monitoring disabled for this backend.", RuntimeWarning)
                 setattr(self, f'_{backend}_import_warned', True)
             return None
        except Exception as e:
            # Catch other potential errors during memory retrieval
            if not hasattr(self, f'_{backend}_error_warned'):
                warnings.warn(f"MemoryUsageCallback: Error retrieving GPU memory info for backend '{backend}': {e}", RuntimeWarning)
                setattr(self, f'_{backend}_error_warned', True)
            return None

        return gpu_mem_mb


    def _log_memory(self, label, step, cpu_mem, gpu_mem):
        """Log memory metrics to stdout and potentially TensorBoard."""
        message = f"{label} - CPU Memory: {cpu_mem:.2f} MB"
        if gpu_mem is not None:
            message += f"; GPU Memory: {gpu_mem:.2f} MB"
        print(message) # Log to stdout

        # Log to TensorBoard if writer is configured
        if self.tb_writer:
            try:
                with self.tb_writer.as_default(step=int(step)):
                    tf.summary.scalar("Memory/CPU_MB", cpu_mem)
                    if gpu_mem is not None:
                        tf.summary.scalar("Memory/GPU_MB", gpu_mem)
                self.tb_writer.flush()
            except Exception as e:
                 # Catch potential errors during logging (e.g., writer closed unexpectedly)
                 if not hasattr(self, '_tb_log_error_warned'):
                     warnings.warn(f"MemoryUsageCallback: Error writing to TensorBoard: {e}", RuntimeWarning)
                     self._tb_log_error_warned = True
                 # Optionally disable writer if logging fails persistently
                 # self.tb_writer = None