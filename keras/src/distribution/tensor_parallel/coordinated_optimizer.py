import re
from typing import Any
from typing import Dict
from typing import List

import numpy as np

import keras
from keras.src import ops
from keras.src import optimizers
from keras.src.backend.distributed import backend_resolver


class CoordinatedOptimizer:
    """Manages an optimizer's state for distributed training.

    This class is an internal coordinator that handles the complexities of
    sharding optimizer states across multiple devices (shards) and
    synchronizing gradients according to tensor parallelism rules. It is not
    intended to be used directly by the end-user but is a core component of
    the `TensorParallelOptimizer`.

    Args:
        base_optimizer: The Keras optimizer instance
            (e.g., `keras.optimizers.Adam`) whose state will be managed.
        world_size: The total number of devices/processes in the distributed
            setup.
        distributed_backend: The distributed communication backend to use.
            Defaults to "auto".
        rank: The rank of the current process. Defaults to 0.
        shard_optimizer_states: If `True`, the optimizer's state variables
            (e.g., momentum, velocity) will be partitioned across `world_size`
            devices. Defaults to `True`.
        tensor_parallel_config: An optional configuration object that defines
            rules for tensor parallelism, such as which gradients to
            all-reduce. Defaults to `None`.
    """

    def __init__(
        self,
        base_optimizer: optimizers.Optimizer,
        world_size: int,
        distributed_backend: str = "auto",
        rank: int = 0,
        shard_optimizer_states: bool = True,
        tensor_parallel_config=None,
    ):
        self.base_optimizer = base_optimizer
        self.world_size = world_size
        self.rank = rank
        self.shard_optimizer_states = shard_optimizer_states
        self.tensor_parallel_config = tensor_parallel_config
        self.sharded_states = {}
        self._state_variable_to_parameter = {}
        self.distributed_backend = (
            backend_resolver.get_distributed_backend(distributed_backend)
            if distributed_backend is not None
            else None
        )
        self._variables = None # Will be set when optimizer is built

    # In class CoordinatedOptimizer:

# In class CoordinatedOptimizer:

# In class CoordinatedOptimizer:

# In class CoordinatedOptimizer:

# In class CoordinatedOptimizer:
# In class CoordinatedOptimizer:

    def _get_optimizer_slot_names(self) -> set:
        """
        Deduces the slot names ('m', 'v', etc.) by inspecting the variables
        created by the base optimizer. This is the most robust method.
        """
        slot_names = set()
        # The optimizer's variables have paths like 'Adam/m/dense/kernel'.
        # We can extract the second part as the slot name.
        for var in self.base_optimizer.variables:
            # Skip the iteration counter
            if "iteration" in var.path.lower():
                continue
            path_parts = var.path.split('/')
            if len(path_parts) > 1:
                slot_names.add(path_parts[1])
        return slot_names

# In class CoordinatedOptimizer:

# In class CoordinatedOptimizer:

# In coordinated_optimizer.py -> class CoordinatedOptimizer:

# In coordinated_optimizer.py -> class CoordinatedOptimizer:

# In coordinated_optimizer.py -> class CoordinatedOptimizer:

# In coordinated_optimizer.py -> class CoordinatedOptimizer:

# In coordinated_optimizer.py -> class CoordinatedOptimizer:

# In coordinated_optimizer.py -> class CoordinatedOptimizer:

# In coordinated_optimizer.py -> class CoordinatedOptimizer:

    def _initialize_sharded_states(self):
        """
        Partitions the optimizer's state variables across shards by inspecting
        the variables created by the base optimizer. This version correctly
        parses variable paths like 'optimizer/param_name_slot_name'.
        """
        if not self.shard_optimizer_states or not self.base_optimizer.built:
            return

        self.sharded_states = {}
        self._state_variable_to_parameter = {}
        opt_name = self.base_optimizer.name

        normalized_params = [
            (p.path.replace('/', '_'), p) for p in self._variables
        ]

        for state_var in self.base_optimizer.variables:
            if state_var is self.base_optimizer.iterations:
                continue

            path_parts = state_var.path.split('/')
            if len(path_parts) != 2 or path_parts[0] != opt_name:
                continue
            
            state_suffix = path_parts[1]

            found_param = None
            slot_name = None
            for norm_param_path, param in normalized_params:
                if state_suffix.startswith(norm_param_path):
                    found_param = param
                    slot_suffix = state_suffix[len(norm_param_path):]
                    slot_name = slot_suffix.strip('_')
                    break

            # THE FIX IS HERE: Explicitly check for 'is not None'
            if found_param is not None and slot_name is not None:
                self._state_variable_to_parameter[state_var.path] = found_param

                sharding_dim = 0
                if self.tensor_parallel_config:
                    norm_param_name = found_param.path.replace("/", ".")
                    for (p, a) in self.tensor_parallel_config.state_rules.items():
                        if re.search(p, norm_param_name) and hasattr(a, "dim"):
                            sharding_dim = a.dim
                            break
                
                partitioned_state = self._partition_state(state_var, dim=sharding_dim)
                self.sharded_states.setdefault(slot_name, {})[found_param.path] = partitioned_state

        if self.base_optimizer.iterations is not None:
            self.sharded_states["iterations"] = self._partition_state(
                self.base_optimizer.iterations, dim=0
            )
    def _partition_state(
        self, state_variable: any, dim: int
    ) -> List[np.ndarray]:
        """Splits a single state variable numpy array into chunks.

        If the variable cannot be split along the given dimension, it is
        replicated across all shards.

        Args:
            state_variable: The optimizer state variable.
            dim: The dimension along which to partition the variable.

        Returns:
            A list of NumPy arrays, where each array is a partition of the
            original state variable for a specific shard.
        """
        state_array = keras.ops.convert_to_numpy(state_variable)
        if state_array.ndim > dim and state_array.shape[dim] >= self.world_size:
            return np.array_split(state_array, self.world_size, axis=dim)
        else:
            return [np.copy(state_array) for _ in range(self.world_size)]

    def get_config(self) -> Dict[str, Any]:
        return {
            "base_optimizer": self.base_optimizer.get_config(),
            "world_size": self.world_size,
            "shard_optimizer_states": self.shard_optimizer_states,
        }

    def apply_gradients(
        self, gradients_and_vars: List[List[tuple]], shard_models: List
    ):
        """Coordinates gradient synchronization and application.

        This method first synchronizes gradients across all shards based on
        tensor parallelism rules. Then, it applies the gradients using either
        sharded optimizer states or replicated states.

        Args:
            gradients_and_vars: A list of lists, where each inner list contains
                (gradient, variable) tuples for a specific model shard.
            shard_models: A list of the sharded model instances.

        Raises:
            ValueError: If the number of gradient sets does not match the
                world size.
        """
        if len(gradients_and_vars) != self.world_size:
            error_msg = (
                f"Expected {self.world_size} gradient sets, "
                f"got {len(gradients_and_vars)}"
            )
            raise ValueError(error_msg)

        synchronized_gradients = self._synchronize_gradients(gradients_and_vars)

        if self.shard_optimizer_states and self.sharded_states:
            self._apply_gradients_with_sharded_states(
                synchronized_gradients, shard_models
            )
        else:
            self._apply_gradients_with_replicated_states(
                synchronized_gradients, shard_models
            )

    def _apply_gradients_with_sharded_states(
        self, synchronized_gradients: List[List[tuple]], shard_models: List
    ):
        """Applies gradients to each shard using its local optimizer state.

        For each shard, this method loads the corresponding partition of the
        optimizer state into the base optimizer and then applies the shard's
        gradients.

        Args:
            synchronized_gradients: The gradients after synchronization.
            shard_models: The list of sharded models.
        """
        for shard_idx, shard_grads in enumerate(synchronized_gradients):
            local_states = self._get_local_optimizer_states(shard_idx)
            self._update_optimizer_internal_state(
                self.base_optimizer, local_states
            )
            self.base_optimizer.apply_gradients(shard_grads)

    def _apply_gradients_with_replicated_states(
        self, synchronized_gradients: List[List[tuple]], shard_models: List
    ):
        """Averages gradients across all shards and applies them once.

        This method is used when optimizer state sharding is disabled. It
        calculates the average of the gradients for each variable across all
        shards and applies the averaged gradients using the single, replicated
        optimizer state.

        Args:
            synchronized_gradients: The gradients after synchronization.
            shard_models: The list of sharded models.
        """
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

            summed_grad = grads_for_var[0]
            for grad in grads_for_var[1:]:
                summed_grad += grad
            averaged_grad = summed_grad / len(grads_for_var)
            averaged_grads_and_vars.append((averaged_grad, variable))

        if averaged_grads_and_vars:
            self.base_optimizer.apply_gradients(averaged_grads_and_vars)

    def _get_local_optimizer_states(self, shard_idx: int) -> Dict[str, Any]:
        """Constructs the state dictionary for a single shard.

        Args:
            shard_idx: The index of the shard for which to retrieve the state.

        Returns:
            A dictionary containing the optimizer state variables specific to
            the given shard index.
        """
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

# In coordinated_optimizer.py -> class CoordinatedOptimizer:

    def _update_optimizer_internal_state(self, local_states: dict):
        """Assigns local sharded state values to the optimizer's variables."""
        if not self.base_optimizer.built:
            return

        for var in self.base_optimizer.variables:
            if var is self.base_optimizer.iterations:
                if "iterations" in local_states:
                    var.assign(local_states["iterations"])
                continue

            # THE FIX IS HERE: Use the variable's path for the lookup.
            param = self._state_variable_to_parameter.get(var.path, None)
            
            if param:
                # This internal method is the most reliable way to get the
                # slot name (e.g., "momentum") from the variable object.
                slot_name = (
                    self.base_optimizer._get_slot_name_from_variable(var)
                )
                if (
                    slot_name in local_states
                    and param.path in local_states[slot_name]
                ):
                    local_param_state = local_states[slot_name][param.path]
                    if var.shape == local_param_state.shape:
                        var.assign(local_param_state)

    def _synchronize_gradients(
        self, gradients_and_vars: List[List[tuple]]
    ) -> List[List[tuple]]:
        """Synchronizes gradients across shards based on tensor parallel rules.

        Specifically, it performs an all-reduce operation on gradients of
        weights that are split along a "column" dimension in tensor parallelism.
        Other gradients are passed through unchanged.

        Args:
            gradients_and_vars: The list of (gradient, variable) lists from
                all shards.

        Returns:
            The list of (gradient, variable) lists after synchronization.
        """
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
                    for shard_idx in range(self.world_size):
                        gradients_and_vars[shard_idx][i] = (
                            synced_grad,
                            variable,
                        )
        return gradients_and_vars

    def _allreduce_gradients(self, gradients: List[Any]) -> List[Any]:
        """Performs a mean all-reduce operation on a list of gradients.

        If a distributed backend is available, it uses it. Otherwise, it
        falls back to a local mean calculation.

        Args:
            gradients: A list of gradients (one from each shard) to be averaged.

        Returns:
            A list where each element is the mean of the input gradients.
        """
        if not gradients:
            return []

        if (
            self.distributed_backend is not None
            and self.distributed_backend.is_initialized
        ):
            numpy_grad = keras.ops.convert_to_numpy(gradients[0])
            synced_numpy = self.distributed_backend.allreduce(
                numpy_grad, op="mean"
            )
            synced_tensor = keras.ops.convert_to_tensor(synced_numpy)
            return [synced_tensor for _ in range(self.world_size)]

        stacked_grads = keras.ops.stack(
            [keras.ops.convert_to_tensor(g) for g in gradients], axis=0
        )
        mean_grad = keras.ops.mean(stacked_grads, axis=0)
        return [mean_grad for _ in range(len(gradients))]

    def get_weights(self) -> List[np.ndarray]:
        """Returns the weights of the base optimizer."""
        return self.base_optimizer.get_weights()

    def set_weights(self, weights: List[np.ndarray]):
        """Sets the weights of the base optimizer."""
        self.base_optimizer.set_weights(weights)

    def enable_optimizer_state_sharding(self, variables: List):
        """Enables and initializes optimizer state sharding.

        This method is called from `build()`, which is guarded from running
        multiple times. We can assume this should always execute.
        """
        # The check 'if not self.shard_optimizer_states:' was here and was
        # incorrectly preventing this code from running. It has been removed.
        self.shard_optimizer_states = True
        self._variables = variables
        self._initialize_sharded_states()

    def disable_optimizer_state_sharding(self):
        """Disables sharding and clears any sharded states.

        This reverts the optimizer to using a single, replicated state.
        """
        if self.shard_optimizer_states:
            self.shard_optimizer_states = False
            self.sharded_states = {}


class TensorParallelOptimizer(optimizers.Optimizer):
    """A Keras Optimizer wrapper for tensor-parallel distributed training.

    This optimizer wraps a standard Keras optimizer (e.g., Adam, SGD) and
    delegates the complex tasks of state management and gradient synchronization
    to a `CoordinatedOptimizer` instance. It is designed to work with models
    that have been sharded for tensor parallelism.

    When `apply_gradients` is called with a list of gradient lists (one for each
    model shard), it uses the `CoordinatedOptimizer` to handle synchronization
    and state sharding. Otherwise, it behaves like the base optimizer.

    Args:
        base_optimizer: A Keras optimizer instance or a string identifier
            (e.g., 'adam', 'sgd').
        world_size: The total number of devices/processes in the distributed
            setup.
        distributed_backend: The distributed communication backend to use.
            Defaults to "auto".
        tensor_parallel_config: An optional configuration object that defines
            rules for tensor parallelism. Defaults to `None`.

    Example:

    ```python
    import keras

    # Assume model variables and gradients from 4 shards exist.
    # The structure is: List[List[Tuple[gradient, variable]]]
    trainable_vars = [keras.Variable(1.0), keras.Variable(2.0)]
    sharded_grads_and_vars = [
        [(keras.ops.ones_like(v), v) for v in trainable_vars]
        for _ in range(4)  # 4 shards
    ]

    # 1. Wrap a standard Keras optimizer.
    base_optimizer = keras.optimizers.Adam()
    optimizer = TensorParallelOptimizer(base_optimizer, world_size=4)
    optimizer.build(trainable_vars)

    # 2. Apply the sharded gradients.
    # The optimizer will handle synchronization (e.g., all-reduce) internally.
    optimizer.apply_gradients(sharded_grads_and_vars)
    ```
    """

    def __init__(
        self,
        base_optimizer: optimizers.Optimizer,
        world_size: int,
        distributed_backend: str = "auto",
        tensor_parallel_config=None,
    ):
        if isinstance(base_optimizer, str):
            resolved_base_optimizer = optimizers.get(base_optimizer)
        else:
            resolved_base_optimizer = base_optimizer

        if isinstance(
            resolved_base_optimizer.learning_rate,
            keras.optimizers.schedules.LearningRateSchedule,
        ):
            lr_value = float(
                ops.convert_to_numpy(
                    resolved_base_optimizer.learning_rate.initial_learning_rate
                )
            )
        else:
            lr_value = float(
                ops.convert_to_numpy(resolved_base_optimizer.learning_rate)
            )

        super().__init__(
            learning_rate=lr_value,
            name=f"TensorParallel_{resolved_base_optimizer.name}",
        )

        self.base_optimizer = resolved_base_optimizer
        self.world_size = world_size
        self.distributed_backend = distributed_backend
        self.coordinated_optimizer = CoordinatedOptimizer(
            self.base_optimizer,
            world_size,
            distributed_backend=distributed_backend,
            tensor_parallel_config=tensor_parallel_config,
        )

    def apply_gradients(self, grads_and_vars: List, **kwargs):
        """Applies gradients to the model variables.

        If `grads_and_vars` is a list of lists, it's assumed to be from
        sharded models, and the `CoordinatedOptimizer` is used. Otherwise,
        it calls the `base_optimizer`'s `apply_gradients` directly.

        Args:
            grads_and_vars: A list of (gradient, variable) tuples, or a list
                of such lists if running in a sharded context.
            **kwargs: Additional arguments. `shard_models` can be passed to
                provide the list of model shards.
        """
        if (
            isinstance(grads_and_vars, list)
            and grads_and_vars
            and isinstance(grads_and_vars[0], list)
        ):
            shard_models = kwargs.get("shard_models", [])
            self.coordinated_optimizer.apply_gradients(
                grads_and_vars, shard_models
            )
        else:
            self.base_optimizer.apply_gradients(grads_and_vars)

    def get_config(self) -> Dict[str, Any]:
        from keras.src import saving

        config = super().get_config()
        config.pop("learning_rate", None)
        config.pop("name", None)

        config.update(
            {
                "base_optimizer": saving.serialize_keras_object(
                    self.base_optimizer
                ),
                "world_size": self.world_size,
                "distributed_backend": self.distributed_backend,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TensorParallelOptimizer":
        from keras.src import saving

        base_optimizer_config = config.pop("base_optimizer")
        base_optimizer = saving.deserialize_keras_object(base_optimizer_config)

        init_kwargs = {
            "world_size": config.get("world_size"),
            "distributed_backend": config.get("distributed_backend", "auto"),
            "tensor_parallel_config": config.get("tensor_parallel_config"),
        }

        return cls(base_optimizer=base_optimizer, **init_kwargs)

    def build(self, variables: List):
        """Builds the optimizer and initializes sharded states.

        This method is called the first time the optimizer is used. It builds
        the base optimizer and then triggers the `CoordinatedOptimizer` to
        initialize its sharded states.

        Args:
            variables: A list of model variables to be optimized.
        """
        if self.built:
            return

        # First, build the base optimizer with the variables.
        self.base_optimizer.build(variables)
        print(f"Variables after build: {[v.path for v in self.base_optimizer.variables]}")

        # THE FINAL FIX: Force slot variable creation by applying zero gradients.
        # This is necessary because optimizers create slots lazily on the first
        # call to apply_gradients.
        if variables:  # Only run if there are variables to optimize.
            zero_grads = [ops.zeros_like(v) for v in variables]
            self.base_optimizer.apply_gradients(zip(zero_grads, variables))

            # The dry run increments the iteration counter, so we reset it.
            if self.base_optimizer.iterations is not None:
                self.base_optimizer.iterations.assign(0)

        # Now that all state variables (m, v, etc.) are guaranteed to exist,
        # we can safely initialize sharding.
        self.coordinated_optimizer.enable_optimizer_state_sharding(variables)
        super().build(variables)

    def get_weights(self) -> List[np.ndarray]:
        """Returns the weights of the base optimizer."""
        return self.coordinated_optimizer.get_weights()

    def set_weights(self, weights: List[np.ndarray]):
        """Sets the weights of the base optimizer."""
        self.coordinated_optimizer.set_weights(weights)

    @property
    def variables(self) -> List:
        """Returns the list of variables from the base optimizer."""
        return self.base_optimizer.variables

    @property
    def learning_rate(self) -> Any:
        """Provides access to the learning rate of the base optimizer."""
        return self.base_optimizer.learning_rate
