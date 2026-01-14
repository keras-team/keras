"""
Tensor Parallel implementation for Keras 3.0
Port of the PyTorch tensor_parallel library
"""

import re
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np

import keras
from keras import ops
from keras.src.distribution import list_devices
from keras.src.distribution.tensor_parallel.autoconfig import get_default_config
from keras.src.distribution.tensor_parallel.parameter_sharding import (
    make_parameter_sharded_model,
)
from keras.src.models import Model


class TensorParallelKeras(Model):
    def __init__(
        self,
        model,
        device_count=None,
        device_ids=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._original_model = model

        if device_count is None:
            device_count, device_ids = self._auto_detect_parallelism()
        elif device_ids is None:
            device_ids = self._auto_configure_devices(device_count)

        self.device_count = device_count
        self.device_ids = device_ids
        self.sharding_strategy = "auto"

        self.tensor_parallel_config = None
        self.distributed = True

        self.sharded_models = [self._original_model]

        accel_devices = list_devices()
        device_ids = list(self.check_device_ids(device_ids))

        if accel_devices:
            if len(accel_devices) >= device_count:
                device_ids = accel_devices[:device_count]
            else:
                device_count = len(accel_devices)
                device_ids = accel_devices[:device_count]

        if not device_ids:
            device_ids = self._auto_configure_devices(device_count)

        if len(device_ids) != device_count:
            device_ids = self._adjust_device_list(device_ids, device_count)

        self.devices = device_ids
        self.device_count = device_count

        if self.device_count <= 1:
            self.model_shards = [model]
            self.distributed = False
            if len(self.devices) == 1:
                from keras import device

                with device(self.devices[0]):
                    self.model_shards[0] = model

            self.assembled_model = self._original_model

            if hasattr(self._original_model, "inputs"):
                self._inputs = self._original_model.inputs
                self._outputs = self._original_model.outputs

            self.built = True
            return

        if self.tensor_parallel_config is None:
            device_names = [str(d) for d in self.devices]
            self.tensor_parallel_config = get_default_config(
                model, device_names
            )

        self._is_multi_layer_model = len(model.layers) > 2

        self.model_shards = []
        self.modified_parameters_names = set()

        for rank, device_id in enumerate(self.devices):
            shard, modified_parameters_names = make_parameter_sharded_model(
                model,
                self.tensor_parallel_config,
                rank=rank,
                device_count=self.device_count,
                device_id=device_id,
            )
            self.model_shards.append(shard)
            self.modified_parameters_names.update(modified_parameters_names)

        params_per_shard = []
        for i, shard in enumerate(self.model_shards):
            total_params = sum(np.prod(p.shape) for p in shard.weights)
            params_per_shard.append(int(total_params))

        self.built = True
        if self.distributed:
            self.assembled_model = self.build_assembled_model()
            self._inputs = self.assembled_model.inputs
            self._outputs = self.assembled_model.outputs
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
        """Auto-detect device_count and device_ids efficiently."""
        from keras.src.distribution import get_best_devices

        available_devices = list_devices()
        device_count = len(available_devices)

        device_ids = get_best_devices(device_count)

        return device_count, device_ids

    def _adjust_device_list(self, device_ids, target_device_count):
        """Adjust device list to match target device_count intelligently."""
        current_size = len(device_ids)
        if current_size >= target_device_count:
            return device_ids[:target_device_count]

        return list(device_ids) + [
            f"cpu:{i}" for i in range(current_size, target_device_count)
        ]

    def _auto_configure_devices(self, device_count):
        """Auto-configure devices - simplified version."""
        available_devices = list_devices()
        if available_devices:
            devices = available_devices[:device_count]
            return devices
        else:
            return ["cpu:0"]

    def check_device_ids(
        self, device_ids: Optional[Sequence[str]]
    ) -> Sequence[str]:
        """Validate and normalize device IDs for Keras."""
        if device_ids is None:
            device_ids = self._get_all_device_indices()

        return tuple(self.canonicalize_device(d) for d in device_ids)

    def _get_all_device_indices(self) -> Sequence[str]:
        """Get all available device indices using distribution library."""
        return list_devices()

    def build_assembled_model(self):
        """
        Builds a single, JIT-friendly Keras Functional model that encapsulates
        the entire tensor parallel logic, correctly handling multiple inputs.
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

        partial_outputs = []
        for shard in self.model_shards:
            shard_inputs = {}
            input_names = getattr(shard, "input_names", None)
            if input_names:
                for name in input_names:
                    clean_name = name.split(":")[0]
                    if clean_name in input_layers:
                        shard_inputs[clean_name] = input_layers[clean_name]
            else:
                for inp in getattr(shard, "inputs", []):
                    clean_name = inp.name.split(":")[0]
                    if clean_name in input_layers:
                        shard_inputs[clean_name] = input_layers[clean_name]

            if not shard_inputs:
                shard_inputs = dict(input_layers)

            partial_outputs.append(shard(shard_inputs))

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
                    lambda x: x - bias * (self.device_count - 1)
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
        """Convert device specification to canonical form."""
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
        Forward pass for the tensor-parallel model.
        """
        return self.assembled_model(inputs, training=training, **kwargs)

    def compile(
        self,
        optimizer=None,
        loss=None,
        metrics=None,
        loss_weights=None,
        **kwargs,
    ):
        """
        Compile the tensor parallel model.
        """
        super().compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights,
            **kwargs,
        )

    def fit(self, x=None, y=None, **kwargs):
        """Use standard Keras training which handles the train_step."""
        return super().fit(x, y, **kwargs)
