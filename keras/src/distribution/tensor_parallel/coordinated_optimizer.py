import re
from typing import Any

import numpy as np

from keras.src import ops
from keras.src import optimizers

from keras.src.backend import distribution_lib


class CoordinatedOptimizer:
    """Manages an optimizer's state for distributed training.
    This class is an internal coordinator that handles the complexities of
    sharding optimizer states across multiple devices (shards) and
    synchronizing gradients according to tensor parallelism rules.
    ...
    Args:
        base_optimizer: The Keras optimizer instance.
        device_count: The total number of devices/processes in the distributed
            setup.
        shard_optimizer_states: If `True`, the optimizer's state variables
            will be partitioned across `device_count` devices. Defaults to `True`.
        tensor_parallel_config: An optional configuration object that defines
            rules for tensor parallelism. Defaults to `None`.
    """

    def __init__(
        self,
        base_optimizer: optimizers.Optimizer,
        device_count: int,
        shard_optimizer_states: bool = True,
        tensor_parallel_config=None,
    ):
        self.base_optimizer = base_optimizer
        self.device_count = device_count
        self.shard_optimizer_states = shard_optimizer_states
        self.tensor_parallel_config = tensor_parallel_config
        self.sharded_states = {}
        self._state_variable_to_parameter = {}
        self._variables = None
        self._variable_to_slot_name = {}

    def _initialize_sharded_states(self):
        """
        Partitions the optimizer's state variables across shards by inspecting
        the variables created by the base optimizer.

        NOTE: Since the Keras BaseOptimizer does not expose a direct mapping 
        from a model parameter to its optimizer state variables, this method 
        infers the mapping by string parsing their paths/names. This addresses
        the collaborator's request for clarity on the path-matching logic.
        """
        if not self.shard_optimizer_states or not self.base_optimizer.built:
            return

        self.sharded_states = {}
        self._state_variable_to_parameter = {}
        self._variable_to_slot_name = {}
        opt_name = self.base_optimizer.name

        normalized_params = sorted(
            [(p.path.replace("/", "_"), p) for p in self._variables],
            key=lambda x: len(x[0]),
            reverse=True,
        )

        for state_var in self.base_optimizer.variables:
            if state_var is self.base_optimizer.iterations:
                continue

            path_parts = state_var.path.split("/")
            if len(path_parts) != 2 or path_parts[0] != opt_name:
                continue

            state_suffix = path_parts[1]

            found_param = None
            slot_name = None
            
            for norm_param_path, param in normalized_params:
                if state_suffix.startswith(norm_param_path):
                    found_param = param
                    slot_suffix = state_suffix[len(norm_param_path) :]
                    slot_name = slot_suffix.strip("_")
                    break

            if found_param is not None and slot_name is not None:
                self._state_variable_to_parameter[state_var.path] = found_param
                self._variable_to_slot_name[state_var.path] = slot_name

                sharding_dim = 0
                if self.tensor_parallel_config:
                    norm_param_name = found_param.path.replace("/", ".")
                    for (
                        p,
                        a,
                    ) in self.tensor_parallel_config.state_rules.items():
                        if re.search(p, norm_param_name) and hasattr(a, "dim"):
                            sharding_dim = a.dim
                            break

                partitioned_state = self._partition_state(
                    state_var, dim=sharding_dim
                )
                self.sharded_states.setdefault(slot_name, {})[
                    found_param.path
                ] = partitioned_state

        if self.base_optimizer.iterations is not None:
            self.sharded_states["iterations"] = self._partition_state(
                self.base_optimizer.iterations, dim=0
            )

    def _partition_state(
        self, state_variable: Any, dim: int
    ) -> list[np.ndarray]:
        """Splits a single state variable numpy array into chunks."""
        state_array = ops.convert_to_numpy(state_variable)
        if (
            state_array.ndim > dim
            and state_array.shape[dim] >= self.device_count
        ):
            return np.array_split(state_array, self.device_count, axis=dim)
        else:
            return [np.copy(state_array) for _ in range(self.device_count)]

    def apply_gradients(
        self, gradients_and_vars: list[list[tuple]], shard_models: list
    ):
        """Coordinates gradient synchronization and application."""
        if len(gradients_and_vars) != self.device_count:
            raise ValueError(
                f"Expected {self.device_count} sets of gradients, "
                f"but received {len(gradients_and_vars)}."
            )

        synchronized_gradients = self._synchronize_gradients(gradients_and_vars)

        if self.shard_optimizer_states:
            self._apply_gradients_with_sharded_states(
                synchronized_gradients, shard_models
            )
        else:
            self._apply_gradients_with_replicated_states(
                synchronized_gradients, shard_models
            )

    def _apply_gradients_with_replicated_states(
        self, synchronized_gradients: list[list[tuple]], shard_models: list
    ):
        """Averages gradients across all shards and applies them once."""
        num_vars = len(synchronized_gradients[0])
        averaged_grads_and_vars = []

        for i in range(num_vars):
            variable = synchronized_gradients[0][i][1]
            grads_for_var = [
                shard_grads[i][0]
                for shard_grads in synchronized_gradients
                if shard_grads[i][0] is not None
            ]

            if not grads_for_var:
                continue

            if len(grads_for_var) > 1:
                stacked_grads = ops.stack(grads_for_var, axis=0)
                averaged_grad = ops.mean(stacked_grads, axis=0)
            else:
                averaged_grad = grads_for_var[0]

            averaged_grads_and_vars.append((averaged_grad, variable))

        if averaged_grads_and_vars:
            self.base_optimizer.apply_gradients(averaged_grads_and_vars)

    def _apply_gradients_with_sharded_states(
        self, synchronized_gradients: list[list[tuple]], shard_models: list
    ):
        """Applies gradients to each shard using its local optimizer state."""
        for shard_idx in range(self.device_count):
            local_states = self._get_local_optimizer_states(shard_idx)
            # Access the base optimizer inside the TensorParallelOptimizer wrapper
            shard_optimizer = shard_models[shard_idx].optimizer.base_optimizer 

            self._update_optimizer_internal_state(
                shard_optimizer, local_states
            )

            shard_grads_and_vars = synchronized_gradients[shard_idx]
            shard_optimizer.apply_gradients(shard_grads_and_vars)

            self._update_global_sharded_states(shard_optimizer, shard_idx)

    def _get_local_optimizer_states(self, shard_idx: int) -> dict[str, Any]:
        """Constructs the state dictionary for a single shard."""
        local_states = {}
        for state_name, state_value in self.sharded_states.items():
            if isinstance(state_value, dict):
                local_states[state_name] = {}
                for param_name, param_states in state_value.items():
                    local_states[state_name][param_name] = param_states[
                        shard_idx
                    ]
            else:
                local_states[state_name] = state_value[shard_idx]
        return local_states

    def _update_optimizer_internal_state(self, optimizer, local_states: dict):
        """Assigns local sharded state values to the optimizer's variables."""
        if not optimizer.built:
            return

        for var in optimizer.variables:
            if var is optimizer.iterations:
                if "iterations" in local_states:
                    var.assign(local_states["iterations"])
                continue

            param = self._state_variable_to_parameter.get(var.path, None)
            slot_name = self._variable_to_slot_name.get(var.path)

            if (
                param
                and slot_name
                and slot_name in local_states
                and param.path in local_states[slot_name]
            ):
                local_param_state = local_states[slot_name][param.path]
                if var.shape == local_param_state.shape:
                    var.assign(local_param_state)

    def _update_global_sharded_states(self, optimizer, shard_idx: int):
        """Updates the main sharded_states dictionary after a gradient step."""
        if not optimizer.built:
            return

        for var in optimizer.variables:
            if var is optimizer.iterations:
                self.sharded_states["iterations"][shard_idx] = (
                    ops.convert_to_numpy(var)
                )
                continue

            param = self._state_variable_to_parameter.get(var.path, None)
            slot_name = self._variable_to_slot_name.get(var.path)

            if (
                param
                and slot_name
                and slot_name in self.sharded_states
                and param.path in self.sharded_states[slot_name]
            ):
                self.sharded_states[slot_name][param.path][shard_idx] = (
                    ops.convert_to_numpy(var)
                )

    def _synchronize_gradients(
        self, gradients_and_vars: list[list[tuple]]
    ) -> list[list[tuple]]:
        """Synchronizes gradients across shards based on tensor parallel rules."""
        if not self.tensor_parallel_config:
            return gradients_and_vars

        rules = self.tensor_parallel_config.state_rules.items()
        column_parallel_patterns = {
            pattern
            for pattern, action in rules
            if hasattr(action, "sharding_type")
            and action.sharding_type == "column"
        }

        if not column_parallel_patterns:
            return gradients_and_vars

        num_weights = len(gradients_and_vars[0])
        for i in range(num_weights):
            variable = gradients_and_vars[0][i][1]
            var_name = getattr(variable, "path", getattr(variable, "name", ""))

            if any(
                re.search(pattern, var_name)
                for pattern in column_parallel_patterns
            ):
                grads_to_reduce = [
                    g_and_v[i][0]
                    for g_and_v in gradients_and_vars
                    if g_and_v[i][0] is not None
                ]
                if grads_to_reduce:
                    synced_grad = self._allreduce_gradients(grads_to_reduce)[0] 
                    for shard_idx in range(self.device_count):
                        if gradients_and_vars[shard_idx][i][0] is not None:
                            gradients_and_vars[shard_idx][i] = (
                                synced_grad,
                                variable,
                            )
        return gradients_and_vars

    def _allreduce_gradients(self, gradients: list[Any]) -> list[Any]:
        """Performs a mean all-reduce operation on a list of gradients.

        This method uses the on-device communication primitive from the backend
        (e.g., JAX's lax.pmean) when multiple devices are detected, resolving
        the critical performance issue related to CPU transfers.
        """
        if not gradients:
            return []

        if distribution_lib.get_device_count() > 1:
            local_grad = gradients[0]
            synced_tensor = distribution_lib.all_reduce(
                local_grad, op="mean", axis_name="model"
            )

            return [synced_tensor for _ in range(self.device_count)]

        if len(gradients) == 1:
            mean_grad = ops.convert_to_tensor(gradients[0])
        else:
            stacked_grads = ops.stack(
                [ops.convert_to_tensor(g) for g in gradients], axis=0
            )
            mean_grad = ops.mean(stacked_grads, axis=0)

        return [mean_grad for _ in range(len(gradients))]

    def get_weights(self) -> list[np.ndarray]:
        """Returns the weights of the base optimizer."""
        return [
            ops.convert_to_numpy(var) for var in self.base_optimizer.variables
        ]

    def set_weights(self, weights: list[np.ndarray]):
        """Sets the weights of the base optimizer."""
        self.base_optimizer.set_weights(weights)

    def enable_optimizer_state_sharding(self, variables: list):
        """Enables and initializes optimizer state sharding."""
        self.shard_optimizer_states = True
        self._variables = variables
        self._initialize_sharded_states()


class TensorParallelOptimizer(optimizers.Optimizer):
    """A Keras Optimizer wrapper for tensor-parallel distributed training.

    This class serves as the public Keras-compliant interface (inherits 
    `optimizers.Optimizer`). It delegates the complex tasks of state 
    management, gradient synchronization, and sharding to the internal 
    `CoordinatedOptimizer` instance. This separation adheres to the 
    principle of keeping the public API clean while encapsulating complex 
    distribution logic.
    
    Args:
        base_optimizer: A Keras optimizer instance or a string identifier.
        device_count: The total number of devices/processes in the distributed
            setup.
        tensor_parallel_config: An optional configuration object. Defaults to `None`.
    """

    def __init__(
        self,
        base_optimizer: optimizers.Optimizer,
        device_count: int,
        tensor_parallel_config=None,
    ):
        if isinstance(base_optimizer, str):
            base_optimizer_instance = optimizers.get(base_optimizer)
        else:
            base_optimizer_instance = base_optimizer

        learning_rate = base_optimizer_instance.learning_rate
        if callable(learning_rate):
            lr_value = float(ops.convert_to_numpy(learning_rate(0)))
        else:
            lr_value = float(ops.convert_to_numpy(learning_rate))

        super().__init__(
            learning_rate=lr_value,
            name=f"TensorParallel_{base_optimizer_instance.name}",
        )

        self.base_optimizer = base_optimizer_instance
        self.device_count = device_count
        self.coordinated_optimizer = CoordinatedOptimizer(
            self.base_optimizer,
            device_count,
            tensor_parallel_config=tensor_parallel_config,
        )

    def apply_gradients(self, grads_and_vars: list, **kwargs):
        """Applies gradients to the model variables."""
        is_sharded_grads = (
            isinstance(grads_and_vars, list)
            and grads_and_vars
            and isinstance(grads_and_vars[0], list)
        )
        if is_sharded_grads:
            if "shard_models" not in kwargs:
                raise ValueError(
                    "The `shard_models` keyword argument is required when "
                    "applying sharded gradients (a list of lists)."
                )
            shard_models = kwargs.get("shard_models")
            self.coordinated_optimizer.apply_gradients(
                grads_and_vars, shard_models
            )
        else:
            self.base_optimizer.apply_gradients(grads_and_vars)

    def get_config(self) -> dict[str, Any]:
        from keras.src import saving

        config = super().get_config()
        config.pop("learning_rate", None)
        config.pop("name", None)

        config.update(
            {
                "base_optimizer": saving.serialize_keras_object(
                    self.base_optimizer
                ),
                "device_count": self.device_count,
                "tensor_parallel_config": self.coordinated_optimizer.tensor_parallel_config,
            }
        )
        return config

    def update_step(self, gradient, variable, *args, **kwargs):
        """Delegates the update step to the base optimizer."""
        if hasattr(self.base_optimizer, "update_step"):
            try:
                return self.base_optimizer.update_step(
                    gradient, variable, *args, **kwargs
                )
            except TypeError:
                return self.base_optimizer.update_step(gradient, variable)

        try:
            return super().update_step(gradient, variable, *args, **kwargs)
        except TypeError:
            return super().update_step(gradient, variable)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TensorParallelOptimizer":
        from keras.src import saving

        base_optimizer_config = config.pop("base_optimizer")
        base_optimizer = saving.deserialize_keras_object(base_optimizer_config)

        init_kwargs = {
            "device_count": config.get("device_count"), 
            "tensor_parallel_config": config.get("tensor_parallel_config"),
        }

        config.pop("device_count", None) 
        config.pop("tensor_parallel_config", None) 
        
        return cls(base_optimizer=base_optimizer, **init_kwargs)

    def build(self, variables: list):
        """Builds the optimizer and initializes sharded states."""
        if self.built:
            return

        self.base_optimizer.build(variables)
        if variables:
            iterations = self.base_optimizer.iterations
            original_iterations_val = None
            if iterations is not None:
                original_iterations_val = ops.convert_to_numpy(
                    iterations.value
                )

            zero_grads = [ops.zeros_like(v) for v in variables]
            self.base_optimizer.apply_gradients(zip(zero_grads, variables))

            if iterations is not None and original_iterations_val is not None:
                iterations.assign(original_iterations_val)

        self.coordinated_optimizer.enable_optimizer_state_sharding(variables)
        super().build(variables)

    def get_weights(self) -> list[np.ndarray]:
        """Returns the weights of the base optimizer."""
        return self.coordinated_optimizer.get_weights()

    def set_weights(self, weights: list[np.ndarray]):
        """Sets the weights of the base optimizer."""
        self.coordinated_optimizer.set_weights(weights)

    @property
    def variables(self) -> list:
        """Returns the list of variables from the base optimizer."""
        return self.base_optimizer.variables

    @property
    def learning_rate(self) -> Any:
        """Provides access to the learning rate of the base optimizer."""
        return self.base_optimizer.learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self.base_optimizer.learning_rate = value

    @property
    def iterations(self):
        """
        Returns the training iteration count directly from the base optimizer.
        """
        return self.base_optimizer.iterations