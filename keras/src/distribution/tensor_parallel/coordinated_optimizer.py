import numpy as np

from keras.src import backend
from keras.src import ops
from keras.src import optimizers
from keras.src.saving import serialization_lib


class TensorParallelOptimizer(optimizers.Optimizer):
    """An optimizer wrapper for Tensor Parallelism and Optimizer State Sharding.

    This optimizer reduces memory overhead by partitioning optimizer states
    (e.g., momentum, velocity) across multiple devices. It is specifically
    designed for large-scale models where optimizer states can consume
    significantly morememory than the model weights themselves.

    Attributes:
        base_optimizer: The underlying Keras optimizer being wrapped.
        device_count: Total number of accelerator devices (shards).
        shard_optimizer_states: Whether to enable partitioning of states.
        tensor_parallel_config: Configuration object defining how specific
            variables should be sharded.
    """

    def __init__(
        self,
        base_optimizer,
        device_count,
        shard_optimizer_states=True,
        tensor_parallel_config=None,
        name=None,
        **kwargs,
    ):
        """Initializes the TensorParallelOptimizer.

        Args:
            base_optimizer: A Keras optimizer instance, a string identifier,
                or a configuration dictionary.
            device_count: Integer, the number of devices to shard states across.
            shard_optimizer_states: Boolean, if True, partitions optimizer
                variables across devices.
            tensor_parallel_config: Optional object containing sharding rules
                mapping specific variables to axes.
            name: String, name of the optimizer instance.
            **kwargs: Additional arguments passed to the base optimizer.
        """
        if isinstance(base_optimizer, str):
            base_optimizer = optimizers.get(base_optimizer)
        elif isinstance(base_optimizer, dict):
            base_optimizer = serialization_lib.deserialize_keras_object(
                base_optimizer
            )

        lr = getattr(
            base_optimizer, "_learning_rate", base_optimizer.learning_rate
        )
        if hasattr(lr, "numpy") and not callable(lr):
            kwargs["learning_rate"] = float(ops.convert_to_numpy(lr))
        else:
            kwargs["learning_rate"] = lr

        super().__init__(name=name, **kwargs)

        self.base_optimizer = base_optimizer
        self.device_count = device_count
        self.shard_optimizer_states = shard_optimizer_states
        self.tensor_parallel_config = tensor_parallel_config

        self._sharded_states = {}
        self._state_var_to_param = {}
        self._var_to_slot_name = {}
        self._model_variables = None

    def build(self, variables):
        """Creates optimizer variables and initializes the sharded state cache.

        This method initializes the base optimizer and performs a dummy
        application of gradients to force the creation of all optimizer slots
        (momentum, etc.) before partitioning them.

        Args:
            variables: List of model variables (weights) to be optimized.
        """
        if self.built:
            return

        self._model_variables = variables
        self.base_optimizer.build(variables)

        if variables:
            grads = [
                ops.zeros_like(ops.convert_to_tensor(v)) for v in variables
            ]
            self.base_optimizer.apply_gradients(zip(grads, variables))

        if self.shard_optimizer_states:
            self._initialize_sharded_states()

        super().build(variables)

    def _initialize_sharded_states(self):
        """Partitions all optimizer state variables into the sharded
           state cache.

        This method maps every optimizer 'slot' variable back to its
        corresponding model parameter and shards it along the appropriate
        dimension determined by the tensor parallel configuration.
        """
        self._sharded_states = {}

        for state_var in self.base_optimizer.variables:
            path = state_var.path

            if "iteration" in path:
                self._sharded_states["iterations"] = self._partition_state(
                    state_var, 0
                )
                continue

            if "learning_rate" in path:
                continue

            for model_var in self._model_variables:
                m_path_norm = model_var.path.replace("/", "_")
                s_path_norm = path.replace("/", "_")

                if m_path_norm in s_path_norm:
                    remainder = s_path_norm.split(m_path_norm)[-1].strip("_")
                    slot_name = remainder if remainder else "unknown"

                    self._state_var_to_param[path] = model_var
                    self._var_to_slot_name[path] = slot_name

                    dim = self._get_sharding_dim(model_var)
                    partitioned = self._partition_state(state_var, dim)

                    if slot_name not in self._sharded_states:
                        self._sharded_states[slot_name] = {}
                    self._sharded_states[slot_name][model_var.path] = (
                        partitioned
                    )
                    break

    def update_step(self, gradient, variable, learning_rate=None):
        """Performs a single weight update on a local variable.

        Args:
            gradient: The gradient tensor for the variable.
            variable: The weight tensor to update.
            learning_rate: Optional learning rate override.

        Returns:
            The result of the base optimizer's update step.
        """
        return self.base_optimizer.update_step(
            gradient, variable, learning_rate=learning_rate
        )

    def apply_gradients(self, grads_and_vars, **kwargs):
        """Applies gradients across shards or via standard Keras logic.

        If the input is a list of lists (sharded gradients), it iterates
        through each device shard, transfers the corresponding optimizer
        state from the global cache to the local optimizer, performs the
        update, and transfers the updated state back to the cache.

        Args:
            grads_and_vars: List of (gradient, variable) tuples, or a list
                of lists for sharded execution.
            **kwargs: Additional arguments, specifically `shard_models`
                which provides access to sub-optimizers for each shard.

        Raises:
            ValueError: If `grads_and_vars` is sharded but `shard_models`
                is not provided in kwargs.
        """
        is_sharded = (
            isinstance(grads_and_vars, list)
            and len(grads_and_vars) > 0
            and isinstance(grads_and_vars[0], list)
        )

        if not is_sharded:
            return super().apply_gradients(grads_and_vars, **kwargs)

        shard_models = kwargs.get("shard_models")
        if not shard_models:
            raise ValueError(
                "`shard_models` is required for sharded gradients."
            )

        synced_grads_and_vars = self._synchronize_gradients(grads_and_vars)

        for i in range(self.device_count):
            shard_opt = shard_models[i].optimizer.base_optimizer
            self._transfer_state(shard_opt, shard_idx=i, direction="to_local")
            shard_opt.apply_gradients(synced_grads_and_vars[i])
            self._transfer_state(shard_opt, shard_idx=i, direction="to_global")

    def _synchronize_gradients(self, gradients_and_vars):
        """Averages gradients for variables that are not sharded via
           Tensor Parallelism.

        This ensures that data-parallel updates remain consistent across
        different optimizer shards.

        Args:
            gradients_and_vars: Nested list of (gradient, variable) for
            each shard.

        Returns:
            A list of lists containing synchronized gradients.
        """
        if self.tensor_parallel_config:
            return gradients_and_vars

        def sync_variable(shards_for_this_var):
            """Calculates the mean gradient across all shards for a variable."""
            grads = [g for g, v in shards_for_this_var if g is not None]
            if not grads:
                return shards_for_this_var

            reduced_grad = ops.mean(ops.stack(grads), axis=0)
            return [(reduced_grad, v) for _, v in shards_for_this_var]

        return [
            list(shard)
            for shard in zip(
                *[sync_variable(v) for v in zip(*gradients_and_vars)]
            )
        ]

    def get_config(self):
        """Returns the configuration of the optimizer for serialization.

        Returns:
            A Python dictionary containing the optimizer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "base_optimizer": serialization_lib.serialize_keras_object(
                    self.base_optimizer
                ),
                "device_count": self.device_count,
                "shard_optimizer_states": self.shard_optimizer_states,
                "tensor_parallel_config": self.tensor_parallel_config,
            }
        )
        return config

    @property
    def variables(self):
        """Returns the variables of the underlying base optimizer."""
        return self.base_optimizer.variables

    @property
    def iterations(self):
        """Returns the iteration count variable of the base optimizer."""
        return self.base_optimizer.iterations

    def _partition_state(self, state_variable, dim):
        """Splits a state variable into N chunks along a specific dimension.

        Args:
            state_variable: The tensor variable to split.
            dim: The dimension along which to perform the split.

        Returns:
            A list of NumPy arrays representing the shards. If the dimension
            cannot be split, the array is replicated across all shards.
        """
        arr = ops.convert_to_numpy(state_variable)
        if arr.ndim > dim and arr.shape[dim] >= self.device_count:
            return np.array_split(arr, self.device_count, axis=dim)
        return [np.copy(arr) for _ in range(self.device_count)]

    def _get_sharding_dim(self, param):
        """Determines the appropriate sharding dimension for a parameter.

        Args:
            param: The model parameter (variable) to check.

        Returns:
            Integer representing the axis to shard on. Defaults to 0.
        """
        if not self.tensor_parallel_config:
            return 0
        rule = self.tensor_parallel_config.state_rules.get(id(param))
        if rule:
            if hasattr(rule, "keywords") and "dim" in rule.keywords:
                return rule.keywords["dim"]
            return getattr(rule, "dim", 0)
        return 0

    def _transfer_state(self, local_opt, shard_idx, direction="to_local"):
        """Syncs data between the global sharded state and a specific local
           optimizer.

        This function handles the 'Gather/Scatter' logic for optimizer states.

        Args:
            local_opt: The optimizer instance local to a specific shard/device.
            shard_idx: The index of the shard currently being processed.
            direction: String, either 'to_local' (global -> local)
                or 'to_global' (local -> global).
        """
        for var in local_opt.variables:
            target_dtype = backend.standardize_dtype(var.dtype)

            if var is local_opt.iterations:
                if direction == "to_local":
                    val = self._sharded_states["iterations"][shard_idx]
                    var.assign(ops.cast(val, target_dtype))
                else:
                    self._sharded_states["iterations"][shard_idx] = (
                        ops.convert_to_numpy(var)
                    )
                continue

            param = self._state_var_to_param.get(var.path)
            slot = self._var_to_slot_name.get(var.path)

            if (
                param
                and slot in self._sharded_states
                and param.path in self._sharded_states[slot]
            ):
                if direction == "to_local":
                    val = self._sharded_states[slot][param.path][shard_idx]
                    if var.shape == val.shape:
                        var.assign(ops.cast(val, target_dtype))
                else:
                    self._sharded_states[slot][param.path][shard_idx] = (
                        ops.convert_to_numpy(var)
                    )
