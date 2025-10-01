import re
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

import keras
from keras import ops
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

        self.distributed_backend = backend_resolver.get_distributed_backend(
            distributed_backend
        )

        if self.shard_optimizer_states:
            if getattr(self.base_optimizer, "built", False):
                self._initialize_sharded_states()
            else:
                self.shard_optimizer_states = False

    def _parse_variable_name(
        self, var_name: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Parses an optimizer variable name to find its type and parameter.

        For ex, it maps 'dense/kernel/_momentum' to
        ('momentum', 'dense/kernel').

        Args:
            var_name: The name of the optimizer state variable.

        Returns:
            A tuple containing the state type (e.g., 'momentum') and the
            associated parameter name. Returns (None, None) if no known
            state suffix is found.
        """
        var_name_lower = var_name.lower()
        state_map = {
            "_momentum": "momentum",
            "_velocity": "velocity",
            "_m": "m",
            "_v": "v",
        }
        for suffix, state_name in state_map.items():
            if var_name_lower.endswith(suffix):
                param_name = var_name[: -len(suffix)]
                return state_name, param_name
        return None, None

    def _get_actual_optimizer_state(self) -> Dict[str, Any]:
        """Extracts and structures the optimizer's state variables.

        This method inspects the `variables` of the `base_optimizer` and
        organizes them into a nested dictionary structure based on their type
        (e.g., learning rate, iteration count, momentum).

        Returns:
            A dictionary containing the optimizer's state, organized by
            state type. For example:
            {'t': <tf.Variable>, 'lr': <tf.Variable>,
             'momentum': {'param1': <tf.Variable>}}.
        """
        state_dict = {}
        for var in self.base_optimizer.variables:
            identifier = getattr(var, "path", getattr(var, "name", str(var)))
            parts = identifier.split("/")
            tail = parts[-1]
            tail_lower = tail.lower()

            if "iteration" in tail_lower or tail_lower in {"iter", "t"}:
                state_dict["t"] = var
                continue
            if "learning_rate" in tail_lower or tail_lower in {"lr"}:
                state_dict["lr"] = var
                continue

            state_name, param_name = self._parse_variable_name(tail)
            if not state_name:
                state_name = "state"
                param_name = tail

            if state_name not in state_dict:
                state_dict[state_name] = {}
            state_dict[state_name][param_name] = var
        return state_dict

    def _initialize_sharded_states(self):
        """Partitions the optimizer's state variables across shards."""
        if not self.shard_optimizer_states or not getattr(
            self.base_optimizer, "built", False
        ):
            return

        base_state = self._get_actual_optimizer_state()
        if not base_state:
            self.shard_optimizer_states = False
            return

        self.sharded_states = {}
        for state_name, state_value in base_state.items():
            if isinstance(state_value, dict):
                self.sharded_states[state_name] = {}
                for param_name, param_state_var in state_value.items():
                    sharding_dim = 0
                    if self.tensor_parallel_config:
                        norm_param = param_name.replace("/", ".")
                        for (
                            p,
                            a,
                        ) in self.tensor_parallel_config.state_rules.items():
                            if re.search(p, norm_param) and hasattr(a, "dim"):
                                sharding_dim = a.dim
                                break
                    self.sharded_states[state_name][param_name] = (
                        self._partition_state(param_state_var, dim=sharding_dim)
                    )
            else:
                self.sharded_states[state_name] = self._partition_state(
                    state_value, dim=0
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

    def _update_optimizer_internal_state(self, optimizer, local_states: dict):
        """Assigns local sharded state values to the optimizer's variables.

        This method updates the `base_optimizer`'s internal state variables
        in-place with the values from a specific shard's state partition.

        Args:
            optimizer: The Keras optimizer instance to update.
            local_states: A dictionary of state values for a single shard.
        """
        if not hasattr(optimizer, "variables") or not optimizer.variables:
            return

        for var in optimizer.variables:
            identifier = getattr(var, "path", getattr(var, "name", str(var)))
            parts = identifier.split("/")
            tail = parts[-1]
            tail_lower = tail.lower()

            if "iteration" in tail_lower or tail_lower in {"iter", "t"}:
                if "t" in local_states:
                    var.assign(local_states["t"])
                continue
            if "learning_rate" in tail_lower or tail_lower in {"lr"}:
                if "lr" in local_states:
                    var.assign(local_states["lr"])
                continue

            state_name, param_name_in_opt = self._parse_variable_name(tail)
            if (
                state_name in local_states
                and param_name_in_opt in local_states[state_name]
            ):
                local_param_state = local_states[state_name][param_name_in_opt]
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

    def enable_optimizer_state_sharding(self):
        """Enables and initializes optimizer state sharding.

        If sharding is not already active, this method sets the flag and
        triggers the partitioning of the optimizer's states.
        """
        if not self.shard_optimizer_states:
            self.shard_optimizer_states = True
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
    """

    def __init__(
        self,
        base_optimizer: optimizers.Optimizer,
        world_size: int,
        distributed_backend: str = "auto",
        tensor_parallel_config=None,
    ):
        if isinstance(base_optimizer, str):
            opt_lower = base_optimizer.lower()
            if opt_lower == "adam":
                resolved_base_optimizer = optimizers.Adam()
            elif opt_lower == "sgd":
                resolved_base_optimizer = optimizers.SGD()
            elif opt_lower == "rmsprop":
                resolved_base_optimizer = optimizers.RMSprop()
            else:
                raise ValueError(f"Unknown optimizer string: {base_optimizer}")
        else:
            resolved_base_optimizer = base_optimizer

        lr_value = float(ops.convert_to_numpy(resolved_base_optimizer.learning_rate))

        super().__init__(
            learning_rate=lr_value,
            name=f"TensorParallel_{resolved_base_optimizer.name}"
        )

        self.base_optimizer = resolved_base_optimizer
        self.world_size = world_size
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
        """Returns the serializable configuration of the optimizer."""
        config = super().get_config()
        config.pop("learning_rate", None)
        config.pop("name", None)
        
        config.update(
            {
                "base_optimizer": saving.serialize_keras_object(self.base_optimizer),
                "world_size": self.world_size,
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
        self.base_optimizer.build(variables)
        self.coordinated_optimizer.enable_optimizer_state_sharding()
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
