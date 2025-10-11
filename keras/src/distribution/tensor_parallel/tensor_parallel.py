import re
from typing import Collection, Optional, Sequence, Union

import numpy as np
import tensorflow as tf
import keras
from keras import ops
from keras.src.distribution.tensor_parallel.autoconfig import (
    get_default_config_keras,
)
from keras.src.distribution.tensor_parallel.parameter_sharding import (
    make_parameter_sharded_model,
)
from keras.src.distribution.tensor_parallel.sharding_keras import ShardedKeras

from keras.src.distribution.tensor_parallel.coordinated_optimizer import TensorParallelOptimizer


from keras.src.models import Model


class TensorParallelKeras(Model):
    """A Keras Model wrapper for implementing tensor parallelism.

    This class takes a standard Keras model and shards its weights across
    multiple devices (`world_size`). It automatically handles the sharding of
    parameters, communication between devices, and construction of a unified
    computational graph. The result is a model that can be trained and used
    like a regular Keras model but leverages multiple accelerators to fit
    larger models into memory.

    Args:
        model (keras.Model): The Keras model to be parallelized.
        world_size (int, optional): The total number of devices to shard the
            model across. If `None`, it will be auto-detected. Defaults to `None`.
        device_ids (Sequence[str], optional): A sequence of specific device IDs
            (e.g., `['/gpu:0', '/gpu:1']`) to use. If `None`, devices will be
            auto-configured. Defaults to `None`.
        distributed_backend (str, optional): The backend to use for distributed
            communication. Defaults to "auto".
        **kwargs: Additional keyword arguments passed to the `keras.Model`
            base class constructor.
    """
    def __init__(
        self,
        model,
        world_size=None,
        device_ids=None,
        distributed_backend="auto",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._original_model = model

        if world_size is None:
            world_size, device_ids = self._auto_detect_parallelism()
        elif device_ids is None:
            device_ids = self._auto_configure_devices(
                world_size, distributed_backend
            )

        self.world_size = world_size
        self.device_ids = device_ids
        self.sharding_strategy = "auto"
        self.distributed_backend = distributed_backend

        self.tensor_parallel_config = None
        self.distributed = True

        self.sharded_models = [self._original_model]
        original_params = 0
        for p in model.weights:
            if hasattr(p, "shape") and hasattr(p.shape, "num_elements"):
                original_params += p.shape.num_elements()
            elif hasattr(p, "shape") and hasattr(p.shape, "__iter__"):
                original_params += np.prod(p.shape)
            else:
                original_params += np.prod(p.shape)

        device_ids = list(self.check_device_ids(device_ids))

        accel_devices = self._discover_devices()

        if accel_devices:
            backend_name = keras.backend.backend()

            if len(accel_devices) >= world_size:
                device_ids = accel_devices[:world_size]
            else:
                world_size = len(accel_devices)
                device_ids = accel_devices[:world_size]

        if not device_ids:
            device_ids = self._auto_configure_devices(
                world_size, distributed_backend
            )

        if len(device_ids) != world_size:
            device_ids = self._adjust_device_list(device_ids, world_size)

        self.devices = device_ids
        self.world_size = world_size
        self.sharding_manager = None
        
        if self.world_size <= 1:
            self.model_shards = [model]
            self.distributed = False
            if len(self.devices) == 1:
                from keras import device
                with device(self.devices[0]):
                    self.model_shards[0] = model
            self.built = True
            self.assembled_model = self._original_model
            return

        if self.tensor_parallel_config is None:
            device_names = [str(d) for d in self.devices]
            self.tensor_parallel_config = get_default_config_keras(
                model, device_names
            )
        config_with_ops = self.tensor_parallel_config.create_collective_ops(
            self.devices
        )

        self._is_multi_layer_model = len(model.layers) > 2

        self.model_shards = []
        self.modified_parameters_names = set()


        for rank, device_id in enumerate(self.devices):
            shard, modified_parameters_names = make_parameter_sharded_model(
                model,
                config_with_ops,
                rank=rank,
                world_size=self.world_size,
                device_id=device_id,
            )
            self.model_shards.append(shard)
            self.modified_parameters_names.update(modified_parameters_names)

        params_per_shard = []
        for i, shard in enumerate(self.model_shards):
            total_params = 0
            for p in shard.weights:
                if hasattr(p, "num_elements"):
                    total_params += p.num_elements()
                elif hasattr(p, "numel"):
                    total_params += p.numel()
                elif hasattr(p.shape, "num_elements"):
                    total_params += p.shape.num_elements()
                else:
                    total_params += np.prod(p.shape)

            params_per_shard.append(int(total_params))

        self.distributed_backend_name = distributed_backend
        from keras.src.distribution import distributed_backend

        self.distributed_backend = distributed_backend
        self.built = True
        if self.distributed:
            self.assembled_model = self.build_assembled_model()
        else:
            self.assembled_model = self._original_model

    @property
    def variables(self):
        """Returns a unique list of all variables from all model shards."""
        unique_vars = {}
        for shard in self.model_shards:
            for var in shard.variables:
                if id(var) not in unique_vars:
                    unique_vars[id(var)] = var
        return list(unique_vars.values())

    @property
    def trainable_variables(self):
        """Returns a unique list of all trainable variables from all model shards."""
        unique_vars = {}
        for shard in self.model_shards:
            for var in shard.trainable_variables:
                if id(var) not in unique_vars:
                    unique_vars[id(var)] = var
        return list(unique_vars.values())

    @property
    def non_trainable_variables(self):
        """Returns a unique list of all non-trainable variables from all model shards."""
        unique_vars = {}
        for shard in self.model_shards:
            for var in shard.non_trainable_variables:
                if id(var) not in unique_vars:
                    unique_vars[id(var)] = var
        return list(unique_vars.values())

    @property
    def weights(self):
        """Returns a unique list of all weights from all model shards."""
        unique_vars = {}
        for shard in self.model_shards:
            for var in shard.weights:
                if id(var) not in unique_vars:
                    unique_vars[id(var)] = var
        return list(unique_vars.values())

    @property
    def trainable_weights(self):
        """Returns a unique list of all trainable weights from all model shards."""
        unique_vars = {}
        for shard in self.model_shards:
            for var in shard.trainable_weights:
                if id(var) not in unique_vars:
                    unique_vars[id(var)] = var
        return list(unique_vars.values())

    @property
    def non_trainable_weights(self):
        """Returns a unique list of all non-trainable weights from all model shards."""
        unique_vars = {}
        for shard in self.model_shards:
            for var in shard.non_trainable_weights:
                if id(var) not in unique_vars:
                    unique_vars[id(var)] = var
        return list(unique_vars.values())

    def _discover_devices(self):
        """Discovers available accelerator devices for the current backend.

        Returns:
            list: A list of strings representing the available device names.
        """
        backend = keras.backend.backend()
        devices = []

        if backend == "jax":
            import jax
            all_devices = jax.devices()
            for platform in ("tpu", "gpu", "cpu"):
                platform_devices = [
                    d for d in all_devices if d.platform == platform
                ]
                if platform_devices:
                    devices = platform_devices
                    break
        elif backend == "tensorflow":
            import tensorflow as tf
            gpus = tf.config.list_logical_devices("GPU")
            if gpus:
                devices = [d.name for d in gpus]
            else:
                cpus = tf.config.list_logical_devices("CPU")
                devices = [d.name for d in cpus]
        elif backend == "torch":
            import torch
            if torch.cuda.is_available():
                devices = [
                    f"cuda:{i}" for i in range(torch.cuda.device_count())
                ]
            elif torch.backends.mps.is_available():
                devices = ["mps"]
            else:
                devices = ["cpu"]
                
        return devices

    def _auto_detect_parallelism(self):
        """Auto-detects world_size and device_ids based on available hardware.

        Returns:
            tuple: A tuple containing the world size (int) and a list of
                   device IDs (list[str]).
        """
        from keras.src.distribution import get_best_devices
        from keras.src.distribution import list_devices

        available_devices = list_devices()
        world_size = len(available_devices)

        device_ids = get_best_devices(world_size)

        return world_size, device_ids

    def _adjust_device_list(self, device_ids, target_world_size):
        """Adjusts the device list to match the target world size.

        If the list is longer, it's truncated. If it's shorter, it's
        extended, attempting to follow the pattern of existing devices or
        falling back to CPUs.

        Args:
            device_ids (list): The current list of device IDs.
            target_world_size (int): The desired number of devices.

        Returns:
            list: The adjusted list of device IDs.
        """
        current_size = len(device_ids)
        if current_size >= target_world_size:
            return device_ids[:target_world_size]

        num_to_add = target_world_size - current_size

        if not device_ids:
            return [f"cpu:{i}" for i in range(target_world_size)]

        base_device = device_ids[0]
        if isinstance(base_device, str) and ":" in base_device:
            device_type, index_str = base_device.rsplit(":", 1)
            if index_str.isdigit():
                additional_devices = [
                    f"{device_type}:{current_size + i}" for i in range(num_to_add)
                ]
                return device_ids + additional_devices

        additional_devices = [f"cpu:{current_size + i}" for i in range(num_to_add)]
        return device_ids + additional_devices

    def _auto_configure_devices(self, world_size, distributed_backend):
        """Automatically configures a list of devices to use.

        It prioritizes available accelerators.

        Args:
            world_size (int): The number of devices to configure.
            distributed_backend (str): The name of the distributed backend.

        Returns:
            list: A list of device ID strings.
        """
        from keras.src.distribution import list_devices

        available_devices = list_devices()

        if available_devices:
            devices = available_devices[:world_size]
            return devices
        else:
            return ["cpu:0"]

    def check_device_ids(
        self, device_ids: Optional[Sequence[str]]
    ) -> Sequence[str]:
        """Validates and normalizes a sequence of device IDs.

        Args:
            device_ids (Sequence[str], optional): The input device IDs.

        Returns:
            Sequence[str]: A tuple of canonicalized device ID strings.
        """
        if device_ids is None:
            device_ids = self._get_all_device_indices()

        device_ids = list(device_ids)

        canonical_ids = []
        for device_id in device_ids:
            if isinstance(device_id, str):
                canonical_ids.append(self.canonicalize_device(device_id))
            else:
                canonical_ids.append(device_id)

        return tuple(canonical_ids)

    def _get_all_device_indices(self) -> Sequence[str]:
        """Gets all available device identifiers from the distribution backend.

        Returns:
            Sequence[str]: A sequence of available device names.
        """
        from keras.src.distribution import list_devices

        devices = list_devices()
        return devices

    def build_assembled_model(self):
        """Builds a single Keras Functional model that encapsulates the parallel logic.

        This method creates a unified computation graph that takes the user's
        inputs, passes them to each model shard in parallel, and then correctly
        combines the outputs from each shard based on the sharding strategy of
        the final layer (e.g., concatenation for column-parallel, summation for
        row-parallel).

        This approach provides a simple, high-level interface for both inference
        and training and is more amenable to JIT compilation.

        Returns:
            keras.Model: The assembled functional model representing the entire
                         tensor-parallel computation.
        """
        if not self.distributed:
            return self._original_model

        input_layers = {
            inp.name.split(":")[0]: keras.Input(
                shape=inp.shape[1:],
                dtype=inp.dtype,
                name=inp.name.split(":")[0],
            )
            for inp in self._original_model.inputs
        }

        partial_outputs = [model(input_layers) for model in self.sharded_models]

        final_layer = self._original_model.layers[-1]
        sharding_type = "unknown"
        final_kernel_name = f"{final_layer.name}.kernel"
        if hasattr(self._original_model, "name") and self._original_model.name:
            final_kernel_name = (
                f"{self._original_model.name}.{final_kernel_name}"
            )

        for pattern, action in self.tensor_parallel_config.state_rules.items():
            if re.search(pattern, final_kernel_name):
                if hasattr(action, "sharding_type"):
                    sharding_type = action.sharding_type
                break

        if sharding_type == "column":
            final_output = ops.concatenate(partial_outputs, axis=-1)
            original_output_dim = self._original_model.output_shape[-1]
            if final_output.shape[-1] != original_output_dim:
                final_output = keras.layers.Lambda(
                    lambda x: x[..., :original_output_dim]
                )(final_output)
        elif sharding_type == "row":
            if len(partial_outputs) > 1:
                summed_output = keras.layers.Add()(partial_outputs)
            else:
                summed_output = partial_outputs[0]

            if final_layer.use_bias:
                bias = final_layer.bias
                final_output = keras.layers.Lambda(
                    lambda x: x - bias * (self.world_size - 1)
                )(summed_output)
            else:
                final_output = summed_output
        else:
            final_output = partial_outputs[0]

        assembled_model = keras.Model(
            inputs=list(input_layers.values()), outputs=final_output
        )
        return assembled_model

    def canonicalize_device(self, device_spec: Union[str, int]) -> str:
        """Converts a device specification to a canonical string format.

        For example, `1` -> `"gpu:1"`, `"cuda:1"` -> `"gpu:1"`.

        Args:
            device_spec (Union[str, int]): The device identifier.

        Returns:
            str: The canonical device string.
        """
        if isinstance(device_spec, int):
            if device_spec == -1:
                return "cpu"
            else:
                return f"gpu:{device_spec}"
        elif isinstance(device_spec, str):
            if device_spec == "cpu":
                return "cpu"
            elif device_spec.startswith("gpu:"):
                return device_spec
            elif device_spec.startswith("cuda:"):
                return f"gpu:{device_spec.split(':')[1]}"
            else:
                return device_spec
        else:
            return "cpu"

    def apply_sharding(
        self, replicated_param_names: Optional[Collection[str]] = None
    ):
        """Applies the sharding strategy to the model parameters.

        This method is typically called internally but can be used to manually
        trigger the sharding process.

        Args:
            replicated_param_names (Collection[str], optional): A collection of
                parameter names that should be replicated across all devices
                instead of sharded. Defaults to `self.modified_parameters_names`.
        """
        if replicated_param_names is None:
            replicated_param_names = self.modified_parameters_names

        self.sharding_manager = ShardedKeras(
            self.model_shards,
            replicated_param_names,
            self.tensor_parallel_config,
            self.devices,
            0,
        )

    def call(self, inputs, training=None, **kwargs):
        """Defines the forward pass of the tensor-parallel model.

        This method delegates the call to the `assembled_model`, which contains
        the complete, unified computation graph for the parallel execution.

        Args:
            inputs: The input tensor(s).
            training (bool, optional): Indicates whether the model is in
                training mode. Defaults to None.
            **kwargs: Additional arguments for the forward pass.

        Returns:
            The output tensor(s) of the model.
        """
        return self.assembled_model(inputs, training=training, **kwargs)

    def _apply_forward_communication(self, inputs, training=None, **kwargs):
        """
        (Internal) Applies forward pass communication based on the conjugate rule.

        Note: This method's logic is typically encapsulated within the
        `assembled_model` and may not be called directly during a standard
        forward pass.

        Args:
            inputs: Input tensors.
            training (bool, optional): Training mode flag.
            **kwargs: Additional arguments.

        Returns:
            The combined output tensor after communication.
        """
        if (
            not hasattr(self, "tensor_parallel_config")
            or self.tensor_parallel_config is None
        ):
            return self.shard_outputs[0]

        output_rules = self.tensor_parallel_config.output_rules

        if not output_rules:
            return self.shard_outputs[0]

        from keras.src.distribution.tensor_parallel.communications import (
            TensorParallelCommunicator,
        )

        communicator = TensorParallelCommunicator(self.world_size, rank=0)

        if hasattr(self, "_is_mlp_model") and self._is_mlp_model:
            return self._handle_mlp_forward_communication(communicator)
        else:
            return self._handle_single_layer_forward_communication(
                communicator, output_rules
            )

    def _handle_mlp_forward_communication(self, communicator):
        """
        (Internal) Handles MLP-specific forward communication with handshake optimization.

        Args:
            communicator (TensorParallelCommunicator): The communication handler.

        Returns:
            The final output tensor.
        """
        up_outputs = []
        down_outputs = []

        for i in range(self.world_size):
            if i in self.shard_outputs:
                up_outputs.append(self.shard_outputs[i])
                down_outputs.append(self.shard_outputs[i])

        final_up, final_down = communicator.handle_mlp_handshake(
            up_outputs, down_outputs
        )

        return final_down[0] if isinstance(final_down, list) else final_down

    def _handle_single_layer_forward_communication(
        self, communicator, output_rules
    ):
        """
        (Internal) Handles forward communication for a single sharded layer.

        Args:
            communicator (TensorParallelCommunicator): The communication handler.
            output_rules (dict): Rules defining how to handle outputs.

        Returns:
            The final output tensor.
        """
        first_output = self.shard_outputs[0]
        if hasattr(first_output, "shape") and len(first_output.shape) >= 2:
            if (
                hasattr(self, "_is_multi_layer_model")
                and self._is_multi_layer_model
            ):
                return first_output

            partial_outputs = []
            for i in range(self.world_size):
                if i in self.shard_outputs:
                    partial_outputs.append(self.shard_outputs[i])
            return first_output

        return self.shard_outputs[0]

    def compile(self, optimizer=None, loss=None, metrics=None, **kwargs):
        """Compiles the tensor-parallel model.

        This method overrides the standard `compile`. If the model is distributed
        (`world_size > 1`), it wraps the provided optimizer in a
        `TensorParallelOptimizer`. This specialized optimizer is responsible for
        coordinating gradient computation and application across all devices
        during training.

        Args:
            optimizer: The optimizer instance.
            loss: The loss function.
            metrics: A list of metrics to be evaluated by the model.
            **kwargs: Additional arguments passed to `super().compile()`.
        """
        if len(self.model_shards) > 1 and optimizer is not None:
            backend_name = getattr(self, "distributed_backend_name", "auto")

            self.coordinated_optimizer = TensorParallelOptimizer(
                optimizer,
                self.world_size,
                distributed_backend=backend_name,
                tensor_parallel_config=self.tensor_parallel_config,
            )

            super().compile(
                optimizer=self.coordinated_optimizer,
                loss=loss,
                metrics=metrics,
                **kwargs,
            )

        else:
            super().compile(optimizer, loss, metrics, **kwargs)

    def _apply_backward_communication(self, gradients, layer_type="unknown"):
        """
        (Internal) Applies backward pass communication based on the conjugate rule.

        Note: This logic is typically handled by the `TensorParallelOptimizer`
        and may not be called directly during standard training.

        Args:
            gradients: The gradients to be communicated.
            layer_type (str, optional): The type of layer sharding
                (e.g., 'column', 'row'). Defaults to "unknown".

        Returns:
            The communicated gradients.
        """
        if len(self.model_shards) <= 1:
            return gradients

        from keras.src.distribution.tensor_parallel.communications import (
            TensorParallelCommunicator,
        )

        communicator = TensorParallelCommunicator(self.world_size, rank=0)

        if (
            "column" in layer_type.lower()
            or "up_projection" in layer_type.lower()
        ):
            return communicator.backward_column_parallel(
                gradients, op="sum"
            )
        elif (
            "row" in layer_type.lower()
            or "down_projection" in layer_type.lower()
        ):
            gathered = communicator.backward_row_parallel(gradients, dim=-1)
            return [gathered] * self.world_size
        else:
            return gradients

    def _slice_upstream_gradients_for_backward(
        self, full_gradients, sharding_type="unknown"
    ):
        """
        (Internal) Slices the upstream gradients to match each device's shard.

        Note: This logic is typically handled by the `TensorParallelOptimizer`.

        Args:
            full_gradients: The complete upstream gradients.
            sharding_type (str, optional): The sharding type of the layer that
                produced the gradients. Defaults to "unknown".

        Returns:
            list: A list of sliced gradients, one for each device shard.
        """
        if len(self.model_shards) <= 1:
            return [full_gradients]

        from keras.src.distribution.tensor_parallel.communications import (
            TensorParallelCommunicator,
        )

        communicator = TensorParallelCommunicator(self.world_size, rank=0)

        sliced_gradients = []

        for rank in range(self.world_size):
            if sharding_type == "column_parallel":
                sliced_grad = communicator.slice_upstream_gradient_for_column_parallel(
                    full_gradients, rank, self.world_size, dim=-1
                )
            elif sharding_type == "row_parallel":
                sliced_grad = (
                    communicator.slice_upstream_gradient_for_row_parallel(
                        full_gradients, rank, self.world_size, dim=0
                    )
                )
            else:
                sliced_grad = full_gradients

            sliced_gradients.append(sliced_grad)

        return sliced_gradients

    def _compute_shard_gradients_with_sliced_upstream(
        self, shard, sliced_upstream_grad, inputs, training=True
    ):
        """
        (Internal) Computes gradients for a single shard using its sliced upstream gradient.

        Note: This logic is typically handled by the `TensorParallelOptimizer`.

        Args:
            shard (keras.Model): The model shard.
            sliced_upstream_grad: The corresponding slice of the upstream gradient.
            inputs: The inputs to the shard.
            training (bool, optional): Training mode flag. Defaults to True.

        Returns:
            list: The computed gradients for the shard's trainable variables.
        """
        with tf.GradientTape() as tape:
            shard_output = shard(inputs, training=training)
            loss = self._compute_shard_loss(
                shard_output, sliced_upstream_grad
            )

        gradients = tape.gradient(loss, shard.trainable_variables)
        return gradients

    def _compute_shard_loss(self, shard_output, sliced_upstream_grad):
        """
        (Internal) Computes a pseudo-loss to generate correct gradients.

        This function creates a loss whose gradient with respect to the
        `shard_output` is equal to the `sliced_upstream_grad`. A common way
        is to use the mean squared error between the output and the gradient.

        Args:
            shard_output: The output tensor from the model shard.
            sliced_upstream_grad: The target gradient for the shard's output.

        Returns:
            A scalar loss tensor.
        """
        if hasattr(sliced_upstream_grad, "shape") and hasattr(
            shard_output, "shape"
        ):
            target = sliced_upstream_grad
            loss = tf.reduce_mean(tf.square(shard_output - target))
            return loss
        else:
            return tf.reduce_mean(tf.square(shard_output))

    def fit(self, x=None, y=None, **kwargs):
        """Trains the model for a fixed number of epochs (iterations on a dataset).

        This method leverages the standard `keras.Model.fit()` training loop.
        The custom logic provided in `compile()` and the `assembled_model`
        ensures that each training step is executed in a tensor-parallel manner
        correctly.

        Args:
            x: Input data.
            y: Target data.
            **kwargs: Other arguments supported by `keras.Model.fit()`.

        Returns:
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).
        """
        return super().fit(x, y, **kwargs)