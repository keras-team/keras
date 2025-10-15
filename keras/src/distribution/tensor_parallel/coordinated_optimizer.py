import re

import numpy as np

import keras
from keras.src import ops
from keras.src import optimizers
from keras.src.backend import distributed_backend


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
        base_optimizer,
        world_size,
        distributed_backend="auto",
        rank=0,
        shard_optimizer_states=True,
        tensor_parallel_config=None,
    ):
        """Initializes the CoordinatedOptimizer."""
        self.base_optimizer = base_optimizer
        self.world_size = world_size
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
                    for p, a in self.tensor_parallel_config.state_rules.items():
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

    def _partition_state(self, state_variable, dim):
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
        state_array = ops.convert_to_numpy(state_variable)
        if state_array.ndim > dim and state_array.shape[dim] >= self.world_size:
            return np.array_split(state_array, self.world_size, axis=dim)
        else:
            return [np.copy(state_array) for _ in range(self.world_size)]

    def apply_gradients(self, grads_and_vars, shard_models):
        """
        Applies gradients to the model variables by first synchronizing them
        and then applying them using either sharded or replicated optimizer
        states.

        Args:
            grads_and_vars: A list of (gradient, variable) lists from all
                shards.
            shard_models: A list of the sharded model instances.
        """
        synchronized_gradients = self._synchronize_gradients(grads_and_vars)

        if self.shard_optimizer_states:
            self._apply_gradients_with_sharded_states(
                synchronized_gradients, shard_models
            )
        else:
            self._apply_gradients_with_replicated_states(
                synchronized_gradients, shard_models
            )

    def _apply_gradients_with_replicated_states(
        self, synchronized_gradients, shard_models
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

            if len(grads_for_var) > 1:
                stacked_grads = ops.stack(grads_for_var, axis=0)
                averaged_grad = ops.mean(stacked_grads, axis=0)
            else:
                averaged_grad = grads_for_var[0]

            averaged_grads_and_vars.append((averaged_grad, variable))

        if averaged_grads_and_vars:
            self.base_optimizer.apply_gradients(averaged_grads_and_vars)

    def _apply_gradients_with_sharded_states(
        self, synchronized_gradients, shard_models
    ):
        """Applies gradients to each shard using its local optimizer state.

        Args:
            synchronized_gradients: The gradients after synchronization.
            shard_models: The list of sharded models.
        """
        for shard_idx in range(self.world_size):
            local_states = self._get_local_optimizer_states(shard_idx)
            shard_optimizer = shard_models[shard_idx].optimizer

            self._update_optimizer_internal_state(shard_optimizer, local_states)

            shard_grads_and_vars = synchronized_gradients[shard_idx]
            shard_optimizer.apply_gradients(shard_grads_and_vars)

            self._update_global_sharded_states(shard_optimizer, shard_idx)

    def _get_local_optimizer_states(self, shard_idx):
        """Constructs the state dictionary for a single shard.

        Args:
            shard_idx: The index of the shard for which to retrieve the state.

        Returns:
            A dictionary containing the local optimizer state for the specified
            shard.
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

    def _update_optimizer_internal_state(self, optimizer, local_states):
        """Assigns local sharded state values to the optimizer's variables.

        Args:
            optimizer: The optimizer instance for a specific shard.
            local_states: The dictionary of local states for that shard.
        """
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

    def _update_global_sharded_states(self, optimizer, shard_idx):
        """Updates the main sharded_states dictionary after a gradient step.

        Args:
            optimizer: The optimizer instance for a specific shard.
            shard_idx: The index of the shard that was updated.
        """
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

    def _synchronize_gradients(self, gradients_and_vars):
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

    def _allreduce_gradients(self, gradients):
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

        if distributed_backend.is_multi_device_capable():
            all_reduce_fn = distributed_backend.get_communication_ops()[
                "all_reduce"
            ]
            numpy_grad = ops.convert_to_numpy(gradients[0])
            synced_numpy = all_reduce_fn(numpy_grad, op="mean")
            synced_tensor = ops.convert_to_tensor(synced_numpy)
            return [synced_tensor for _ in range(self.world_size)]

        stacked_grads = keras.ops.stack(
            [ops.convert_to_tensor(g) for g in gradients], axis=0
        )
        mean_grad = ops.mean(stacked_grads, axis=0)
        return [mean_grad for _ in range(len(gradients))]

    def get_weights(self):
        """Returns the weights of the base optimizer.

        Returns:
            A list of NumPy arrays representing the optimizer's state variables.
        """
        return [
            ops.convert_to_numpy(var) for var in self.base_optimizer.variables
        ]

    def set_weights(self, weights):
        """Sets the weights of the base optimizer.

        Args:
            weights: A list of NumPy arrays to set as the optimizer's state.
        """
        self.base_optimizer.set_weights(weights)

    def enable_optimizer_state_sharding(self, variables):
        """Enables and initializes optimizer state sharding.

        This method is called from `build()`, which is guarded from running
        multiple times. We can assume this should always execute.

        Args:
            variables: A list of model variables to be optimized.
        """
        self.shard_optimizer_states = True
        self._variables = variables
        self._initialize_sharded_states()


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
    # The structure is: list[list[tuple[gradient, variable]]]
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
        base_optimizer,
        world_size,
        distributed_backend="auto",
        tensor_parallel_config=None,
    ):
        """Initializes the TensorParallelOptimizer."""
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
        self.world_size = world_size
        self.distributed_backend = distributed_backend
        self.coordinated_optimizer = CoordinatedOptimizer(
            self.base_optimizer,
            world_size,
            distributed_backend=distributed_backend,
            tensor_parallel_config=tensor_parallel_config,
        )

    def apply_gradients(self, grads_and_vars, **kwargs):
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
        is_sharded_grads = (
            isinstance(grads_and_vars, list)
            and grads_and_vars
            and isinstance(grads_and_vars[0], list)
        )
        if is_sharded_grads:
            shard_models = kwargs.get("shard_models", [])
            self.coordinated_optimizer.apply_gradients(
                grads_and_vars, shard_models
            )
        else:
            self.base_optimizer.apply_gradients(grads_and_vars)

    def get_config(self):
        """Returns the configuration of the optimizer.

        Returns:
            A dictionary containing the optimizer's configuration.
        """
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

    def update_step(self, gradient, variable, *args, **kwargs):
        """Performs a single optimization step.

        Delegates the update step to the base optimizer if it has a custom
        `update_step` implementation, otherwise falls back to the parent
        optimizer's logic.

        Args:
            gradient: The gradient tensor.
            variable: The variable to be updated.
            *args: Positional arguments passed to the update function.
            **kwargs: Keyword arguments passed to the update function.
        """
        if hasattr(self.base_optimizer, "update_step"):
            return self.base_optimizer.update_step(
                gradient, variable, *args, **kwargs
            )

        return super().update_step(gradient, variable, *args, **kwargs)

    @classmethod
    def from_config(cls, config):
        """Creates an optimizer from its configuration.

        Args:
            config: A Python dictionary, typically the output of `get_config`.

        Returns:
            A `TensorParallelOptimizer` instance.
        """
        from keras.src import saving

        base_optimizer_config = config.pop("base_optimizer")
        base_optimizer = saving.deserialize_keras_object(base_optimizer_config)

        init_kwargs = {
            "world_size": config.get("world_size"),
            "distributed_backend": config.get("distributed_backend", "auto"),
            "tensor_parallel_config": config.get("tensor_parallel_config"),
        }

        return cls(base_optimizer=base_optimizer, **init_kwargs)

    def build(self, variables):
        """Builds the optimizer and initializes sharded states.

        This method is called the first time the optimizer is used. It builds
        the base optimizer and then triggers the `CoordinatedOptimizer` to
        initialize its sharded states.

        Args:
            variables: A list of model variables to be optimized.
        """
        if self.built:
            return

        self.base_optimizer.build(variables)
        if variables:
            iterations = self.base_optimizer.iterations
            original_iterations_val = None
            if iterations is not None:
                original_iterations_val = ops.convert_to_numpy(iterations.value)

            zero_grads = [ops.zeros_like(v) for v in variables]
            self.base_optimizer.apply_gradients(zip(zero_grads, variables))

            if iterations is not None and original_iterations_val is not None:
                iterations.assign(original_iterations_val)

        self.coordinated_optimizer.enable_optimizer_state_sharding(variables)
        super().build(variables)

    def get_weights(self):
        """Returns the weights of the base optimizer.

        Returns:
            A list of NumPy arrays representing the optimizer's state variables.
        """
        return self.coordinated_optimizer.get_weights()

    def set_weights(self, weights):
        """Sets the weights of the base optimizer.

        Args:
            weights: A list of NumPy arrays to set as the optimizer's state.
        """
        self.coordinated_optimizer.set_weights(weights)

    @property
    def variables(self):
        """Returns the list of variables from the base optimizer.

        Returns:
            A list of state variables of the base optimizer.
        """
        return self.base_optimizer.variables

    @property
    def learning_rate(self):
        """Provides access to the learning rate of the base optimizer."""
        return self.base_optimizer.learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        """Sets the learning rate of the base optimizer."""
        self.base_optimizer.learning_rate = value

    @property
    def iterations(self):
        """
        Returns the training iteration count directly from the base optimizer.
        """
        return self.base_optimizer.iterations