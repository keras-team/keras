"""
Tensor Parallel implementation for Keras 3.0
Port of the PyTorch tensor_parallel library
"""

import re

import numpy as np

import keras
from keras import ops
from keras.src.distribution import list_devices
from keras.src.distribution.tensor_parallel.autoconfig import (
    get_default_config_keras,
)
from keras.src.distribution.tensor_parallel.coordinated_optimizer import (
    TensorParallelOptimizer,
)
from keras.src.distribution.tensor_parallel.parameter_sharding import (
    make_parameter_sharded_model,
)
from keras.src.models import Model


class TensorParallelKeras(Model):
    """
    A Keras Model wrapper that implements tensor parallelism.

    This class takes a standard Keras model and shards its weights across
    multiple devices, enabling the model to handle larger sizes than would fit
    on a single device. It automatically handles the sharding, communication,
    and coordination required for training and inference.

    Args:
        model: The Keras model to be parallelized.
        world_size (int, optional): The number of devices to parallelize across.
            If None, it's auto-detected. Defaults to None.
        device_ids (list, optional): A list of device IDs to use. If None,
            they are auto-detected. Defaults to None.
        distributed_backend (str, optional): The backend to use for distributed
            communication. Defaults to "auto".
        **kwargs: Additional arguments passed to the base `keras.Model`.
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

        accel_devices = list_devices()
        device_ids = list(self.check_device_ids(device_ids))

        if accel_devices:
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
            total_params = sum(np.prod(p.shape) for p in shard.weights)
            params_per_shard.append(int(total_params))

        self.distributed_backend_name = distributed_backend
        from keras.src.backend import distributed_backend

        self.distributed_backend = distributed_backend

        self.built = True
        if self.distributed:
            self.assembled_model = self.build_assembled_model()
        else:
            self.assembled_model = self._original_model

    @property
    def variables(self):
        """Returns a list of all unique variables from all model shards."""
        unique_vars = {
            id(var): var
            for shard in self.model_shards
            for var in shard.variables
        }
        return list(unique_vars.values())

    @property
    def trainable_variables(self):
        """Returns list of all unique trainable variables from model shards."""
        unique_vars = {
            id(var): var
            for shard in self.model_shards
            for var in shard.trainable_variables
        }
        return list(unique_vars.values())

    @property
    def non_trainable_variables(self):
        """Returns list of unique non-trainable variables from model shards."""
        unique_vars = {
            id(var): var
            for shard in self.model_shards
            for var in shard.non_trainable_variables
        }
        return list(unique_vars.values())

    @property
    def weights(self):
        """Returns a list of all unique weights from all model shards."""
        unique_vars = {
            id(var): var for shard in self.model_shards for var in shard.weights
        }
        return list(unique_vars.values())

    @property
    def trainable_weights(self):
        """Returns a list of all unique trainable weights from model shards."""
        unique_vars = {
            id(var): var
            for shard in self.model_shards
            for var in shard.trainable_weights
        }
        return list(unique_vars.values())

    @property
    def non_trainable_weights(self):
        """Returns list of unique non-trainable weights from model shards."""
        unique_vars = {
            id(var): var
            for shard in self.model_shards
            for var in shard.non_trainable_weights
        }
        return list(unique_vars.values())

    def _auto_detect_parallelism(self):
        """Auto-detects the number of available devices and sets world size."""
        from keras.src.distribution import get_best_devices

        available_devices = list_devices()
        world_size = len(available_devices)

        device_ids = get_best_devices(world_size)

        return world_size, device_ids

    def _adjust_device_list(self, device_ids, target_world_size):
        """Adjusts the device list to match the target world size."""
        current_size = len(device_ids)
        if current_size >= target_world_size:
            return device_ids[:target_world_size]

        return list(device_ids) + [
            f"cpu:{i}" for i in range(current_size, target_world_size)
        ]

    def _auto_configure_devices(self, world_size, distributed_backend):
        """Automatically configures the devices to be used for parallelism."""
        available_devices = list_devices()
        if available_devices:
            devices = available_devices[:world_size]
            return devices
        else:
            return ["cpu:0"]

    def check_device_ids(self, device_ids):
        """Validates and normalizes a sequence of device IDs."""
        if device_ids is None:
            device_ids = self._get_all_device_indices()

        return tuple(self.canonicalize_device(d) for d in device_ids)

    def _get_all_device_indices(self):
        """Retrieves all available device indices from distribution library."""
        return list_devices()

    def build_assembled_model(self):
        """
        Builds a single Keras Functional model that encapsulates tensor
        parallel logic.

        This method creates unified model that takes original model's inputs,
        distributes the computation across the sharded models, and assembles
        the final output. This assembled model is JIT-compilation friendly.

        Returns:
            A `keras.Model` instance representing the assembled parallel model.
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

    def canonicalize_device(self, device_spec):
        """
        Converts a device specification to its canonical string form.

        Args:
            device_spec: The device identifier (e.g., an int like 0, or a
                         string like "gpu:0", "cuda:0", "cpu").

        Returns:
            A string representing the canonical device name
            (e.g., "gpu:0", "cpu").
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

    def call(self, inputs, training=None, **kwargs):
        """
        Defines the forward pass of the tensor-parallel model.

        This method delegates the call to the internal `assembled_model`,
        which handles the distributed computation.

        Args:
            inputs: Input tensors.
            training (bool, optional): Indicates whether the model is in
                                      training mode. Defaults to None.
            **kwargs: Additional arguments.

        Returns:
            The output tensor(s) of the model.
        """
        return self.assembled_model(inputs, training=training, **kwargs)

    def compile(self, optimizer=None, loss=None, metrics=None, **kwargs):
        """
        Configures the model for training.

        If an optimizer is provided and the model is distributed across more
        than one device, it wraps the optimizer in a `TensorParallelOptimizer`
        to coordinate gradients across all shards.

        Args:
            optimizer: The optimizer instance.
            loss: The loss function.
            metrics: A list of metrics to be evaluated by the model.
            **kwargs: Additional arguments passed to `keras.Model.compile`.
        """
        if len(self.model_shards) > 1 and optimizer is not None:
            backend_name = getattr(self, "distributed_backend_name", "auto")

            self.coordinated_optimizer = TensorParallelOptimizer(
                optimizer,
                self.world_size,
                distributed_backend=backend_name,
                tensor_parallel_config=self.tensor_parallel_config,
            )
            self.coordinated_optimizer._shard_models = self.model_shards

            var_map = {}
            assembled = getattr(self, "assembled_model", None)
            assembled_vars = (
                assembled.variables if assembled is not None else []
            )

            for a_var in assembled_vars:
                key = getattr(a_var, "path", None) or a_var.name
                suffix = key.split("/")[-1]
                per_shard = []
                for shard in self.model_shards:
                    match = next(
                        (v for v in shard.variables if v.name.endswith(suffix)),
                        None,
                    )
                    per_shard.append(match)
                var_map[key] = per_shard

            self.coordinated_optimizer._shard_var_map = var_map
            inner = getattr(
                self.coordinated_optimizer, "coordinated_optimizer", None
            )
            if inner is not None:
                inner._shard_models = self.model_shards
                inner._shard_var_map = var_map

            super().compile(
                optimizer=self.coordinated_optimizer,
                loss=loss,
                metrics=metrics,
                **kwargs,
            )

        else:
            super().compile(optimizer, loss, metrics, **kwargs)

    def fit(self, x=None, y=None, **kwargs):
        """
        Trains the model for a fixed number of epochs (iterations on a dataset).

        This method uses the standard Keras `fit` method, which correctly
        handles the custom `train_step` implicitly managed by compiled model.

        Args:
            x: Input data.
            y: Target data.
            **kwargs: Additional arguments passed to `keras.Model.fit`.

        Returns:
            A `History` object. Its `history` attribute is a record of training
            loss values and metric values at successive epochs.
        """
        return super().fit(x, y, **kwargs)
