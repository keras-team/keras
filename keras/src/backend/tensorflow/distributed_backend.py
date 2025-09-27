import logging
from typing import Any
from typing import List

import tensorflow as tf

import keras
from keras.src.backend.distributed.base import BaseDistributedBackend

logger = logging.getLogger(__name__)


class TensorflowDistributedBackend(BaseDistributedBackend):
    """TensorFlow-specific implementation of distributed operations."""

    def get_tensor_lib(self):
        return tf

    def convert_to_backend_tensor(self, tensor: Any) -> Any:
        if hasattr(tensor, "numpy"):
            return tf.convert_to_tensor(tensor.numpy())
        else:
            return tf.convert_to_tensor(tensor)

    def compute_gradients(
        self, loss: Any, trainable_vars: List[Any]
    ) -> List[Any]:
        with tf.GradientTape() as tape:
            # TensorFlow's tape automatically watches trainable variables,
            # but explicit watching is safer.
            for var in trainable_vars:
                tape.watch(var)

        try:
            # Assuming loss is already a tensor computed from watched variables
            gradients = tape.gradient(loss, trainable_vars)
            logger.info("   - TensorFlow gradient computation successful")
            return gradients
        except Exception as e:
            logger.warning(
                f"TensorFlow gradient computation failed: {e}, using fallback"
            )
            return [tf.zeros_like(var) for var in trainable_vars]

    def apply_gradients(
        self,
        gradients: List[Any],
        trainable_vars: List[Any],
        learning_rate: float = 0.001,
    ) -> None:
        for grad, var in zip(gradients, trainable_vars):
            if grad is not None:
                new_value = var - (learning_rate * grad)
                var.assign(new_value)

    def create_optimizer(self, optimizer_class: str, **kwargs):
        if optimizer_class.lower() == "adam":
            return tf.keras.optimizers.Adam(**kwargs)
        elif optimizer_class.lower() == "sgd":
            return tf.keras.optimizers.SGD(**kwargs)
        else:
            return tf.keras.optimizers.Adam(learning_rate=0.001)

    def get_device_info(self) -> dict:
        info = {"backend": "tensorflow", "devices": [], "device_count": 0}
        try:
            info["devices"] = [
                d.name for d in tf.config.list_physical_devices()
            ]
            info["device_count"] = len(tf.config.list_physical_devices())
        except Exception as e:
            logger.warning(f"Could not get device info for TensorFlow: {e}")
            info["devices"] = ["cpu"]
            info["device_count"] = 1
        return info

    def is_multi_device_capable(self) -> bool:
        return self.get_device_info()["device_count"] > 1

    def get_communication_ops(self) -> dict:
        def all_reduce_tf(x, op="sum"):
            strategy = tf.distribute.get_strategy()
            return strategy.reduce(tf.distribute.ReduceOp.SUM, x, axis=0)

        def all_gather_tf(x, axis=0):
            strategy = tf.distribute.get_strategy()
            return tf.raw_ops.AllGather(
                input=x,
                group_assignment=[
                    [i for i in range(strategy.num_replicas_in_sync)]
                ],
                group_size=strategy.num_replicas_in_sync,
            )

        def broadcast_tf(x, root=0):
            strategy = tf.distribute.get_strategy()
            return strategy.broadcast(x)

        def scatter_tf(x):
            strategy = tf.distribute.get_strategy()
            return strategy.scatter(x, axis=0)

        def all_reduce_simulated(x, op="sum"):
            return keras.ops.sum(x, axis=0)

        def all_gather_simulated(x, axis=0):
            return keras.ops.concatenate([x, x], axis=axis)

        def broadcast_simulated(x):
            return x

        def scatter_simulated(x, num_devices):
            return keras.ops.split(x, num_devices, axis=0)

        try:
            strategy = tf.distribute.get_strategy()
            if not isinstance(
                strategy,
                (
                    tf.distribute.MirroredStrategy,
                    tf.distribute.MultiWorkerMirroredStrategy,
                ),
            ):
                raise RuntimeError("No active `tf.distribute` strategy found.")
            logger.info("Using real TensorFlow `tf.distribute` collective ops.")
            return {
                "all_reduce": all_reduce_tf,
                "all_gather": all_gather_tf,
                "broadcast": broadcast_tf,
                "scatter": scatter_tf,
            }
        except (ImportError, RuntimeError) as e:
            logger.warning(f"TensorFlow collective ops not available: {e}.")
            return {
                "all_reduce": all_reduce_simulated,
                "all_gather": all_gather_simulated,
                "broadcast": broadcast_simulated,
                "scatter": scatter_simulated,
            }
