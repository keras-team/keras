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
        if hasattr(tensor, "cpu") and hasattr(tensor, "numpy"):
            return tf.convert_to_tensor(tensor.cpu().numpy())
        return tf.convert_to_tensor(tensor)

    def compute_gradients(
        self, loss: Any, trainable_vars: List[Any]
    ) -> List[Any]:
        with tf.GradientTape() as tape:
            for var in trainable_vars:
                tape.watch(var)

        try:
            gradients = tape.gradient(loss, trainable_vars)
            logger.info("   - TensorFlow gradient computation successful")
            return gradients
        except Exception:
            logger.warning(
                "TensorFlow gradient computation resulted in None gradients, "
                "using zero-filled fallback for affected variables."
            )
            return [
                tf.zeros_like(var) if g is None else g
                for var, g in zip(trainable_vars, gradients)
            ]
        return gradients

    def apply_gradients(
        self,
        gradients: List[Any],
        trainable_vars: List[Any],
        learning_rate: float = 0.001,
    ) -> None:
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        optimizer.apply_gradients(zip(gradients, trainable_vars))

    def create_optimizer(self, optimizer_class: str, **kwargs):
        if optimizer_class.lower() == "adam":
            return tf.keras.optimizers.Adam(**kwargs)
        elif optimizer_class.lower() == "sgd":
            return tf.keras.optimizers.SGD(**kwargs)
        else:
            return tf.keras.optimizers.Adam(learning_rate=0.001, **kwargs)

    def get_device_info(self) -> dict:
        info = {"backend": "tensorflow", "devices": [], "device_count": 0}
        try:
            physical_devices = tf.config.list_physical_devices()
            info["devices"] = [d.name for d in physical_devices]
            info["device_count"] = len(physical_devices)
        except Exception as e:
            logger.warning(f"Could not get device info for TensorFlow: {e}")
            info["devices"] = ["/physical_device:CPU:0"]
            info["device_count"] = 1
        return info

    def is_multi_device_capable(self) -> bool:
        return self.get_device_info()["device_count"] > 1

    def get_communication_ops(self) -> dict:
        def all_reduce_tf(x, op="sum"):
            strategy = tf.distribute.get_strategy()
            if op == "sum":
                reduce_op = tf.distribute.ReduceOp.SUM
            elif op == "mean":
                reduce_op = tf.distribute.ReduceOp.MEAN
            else:
                raise ValueError(f"Unsupported all_reduce op: {op}")
            return strategy.reduce(reduce_op, x, axis=None)

        def all_gather_tf(x, axis=0):
            strategy = tf.distribute.get_strategy()
            return strategy.gather(x, axis=axis)

        def broadcast_tf(x, root=0):
            strategy = tf.distribute.get_strategy()
            return strategy.broadcast(x, destination=None)

        def scatter_tf(x, root=0):
            strategy = tf.distribute.get_strategy()
            return strategy.experimental_distribute_values_from_function(
                lambda _: x
            )

        try:
            strategy = tf.distribute.get_strategy()
            if strategy.num_replicas_in_sync <= 1:
                raise RuntimeError("No active multi-device strategy found.")
            logger.info("Using real TensorFlow `tf.distribute` collective ops.")
            return {
                "all_reduce": all_reduce_tf,
                "all_gather": all_gather_tf,
                "broadcast": broadcast_tf,
                "scatter": scatter_tf,
            }
        except (ImportError, RuntimeError, ValueError) as e:
            logger.warning(
                f"TensorFlow collective ops not available: {e}. "
                "Using SIMULATED ops."
            )

            device_info = self.get_device_info()
            simulated_world_size = device_info.get("device_count", 1)
            if simulated_world_size == 0:
                simulated_world_size = 1

            logger.info(
                f"Simulating with world_size={simulated_world_size} "
                "based on available devices."
            )

            def all_reduce_simulated(x, op="sum"):
                if simulated_world_size <= 1:
                    return x
                if op == "sum":
                    return keras.ops.multiply(x, simulated_world_size)
                elif op == "mean":
                    return x
                else:
                    raise ValueError(f"Unsupported all_reduce op: {op}")

            def all_gather_simulated(x, axis=0):
                if simulated_world_size <= 1:
                    return x
                tensor_list = [x] * simulated_world_size
                return keras.ops.concatenate(tensor_list, axis=axis)

            def broadcast_simulated(x, root=0):
                return x

            def scatter_simulated(x, root=0):
                if simulated_world_size <= 1:
                    return x
                if keras.ops.shape(x)[0] % simulated_world_size != 0:
                    raise ValueError(
                        "For simulation, the first dimension of tensor must "
                        f"be divisible by the simulated world size "
                        f"({simulated_world_size})."
                    )
                chunks = keras.ops.split(x, simulated_world_size, axis=0)
                return chunks[0]

            return {
                "all_reduce": all_reduce_simulated,
                "all_gather": all_gather_simulated,
                "broadcast": broadcast_simulated,
                "scatter": scatter_simulated,
            }
