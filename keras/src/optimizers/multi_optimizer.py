from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.optimizers import optimizer
from keras.src.saving import serialization_lib
from keras.src.utils import tracking


@keras_export("keras.optimizers.MultiOptimizer")
class MultiOptimizer(optimizer.Optimizer):
    """An optimizer that applies different optimizers to different variables.

    `MultiOptimizer` allows you to use different optimization strategies for
    different parts of your model. This is useful for discriminative layer
    training, where you might want to use different learning rates or
    optimization algorithms for different layers.

    Args:
        optimizers_and_variables: A list of tuples, where each tuple contains:
            - An optimizer instance (e.g., `Adam`, `SGD`)
            - A list of variables that this optimizer should update
        name: Optional name for the optimizer.

    Example:

    ```python
    import keras
    from keras import layers, optimizers

    # Build a simple model
    model = keras.Sequential([
        layers.Dense(64, activation='relu', name='layer1'),
        layers.Dense(32, activation='relu', name='layer2'),
        layers.Dense(10, activation='softmax', name='output')
    ])
    model.build(input_shape=(None, 20))

    # Create optimizers for different layer groups
    adam_opt = optimizers.Adam(learning_rate=0.001)
    sgd_opt = optimizers.SGD(learning_rate=0.01, momentum=0.9)

    # Get variables for each group
    layer1_vars = model.layers[0].trainable_variables
    other_vars = (
        model.layers[1].trainable_variables +
        model.layers[2].trainable_variables
    )

    # Create MultiOptimizer
    multi_opt = optimizers.MultiOptimizer([
        (adam_opt, layer1_vars),
        (sgd_opt, other_vars)
    ])

    # Compile and train
    model.compile(
        optimizer=multi_opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    ```

    Note:
        - Each variable should be assigned to exactly one optimizer.
        - If a variable is not assigned to any optimizer, it will not be
          updated during training.
        - The `learning_rate` property returns the learning rate of the
          first optimizer.
        - The `iterations` property returns the iteration count of the
          first optimizer (all optimizers share the same iteration count).
    """

    def __init__(
        self,
        optimizers_and_variables,
        name=None,
    ):
        super().__init__(learning_rate=0.0, name=name)

        if not optimizers_and_variables:
            raise ValueError(
                "`optimizers_and_variables` must be a non-empty list of "
                "(optimizer, variables) tuples. "
                f"Received: {optimizers_and_variables}"
            )

        self._inner_optimizers = []
        self._optimizer_variables_mapping = []

        for item in optimizers_and_variables:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(
                    "Each item in `optimizers_and_variables` must be a tuple "
                    "of (optimizer, variables). "
                    f"Received: {item}"
                )
            opt, variables = item
            if not isinstance(opt, optimizer.Optimizer):
                raise ValueError(
                    f"Expected an Optimizer instance, got: {type(opt)}"
                )
            if not isinstance(variables, (list, tuple)):
                raise ValueError(
                    "Expected a list or tuple of variables, "
                    f"got: {type(variables)}"
                )
            self._inner_optimizers.append(opt)
            # Store variable references (will be resolved during build)
            self._optimizer_variables_mapping.append(list(variables))

        # Disable loss scaling on inner optimizers to avoid double scaling
        for opt in self._inner_optimizers:
            opt.loss_scale_factor = None

    @tracking.no_automatic_dependency_tracking
    def build(self, var_list):
        # Create a mapping from variable id to optimizer index
        self._var_to_optimizer_idx = {}
        for opt_idx, variables in enumerate(self._optimizer_variables_mapping):
            for var in variables:
                var_key = self._var_key(var)
                if var_key in self._var_to_optimizer_idx:
                    raise ValueError(
                        f"Variable {var.name} is assigned to multiple "
                        "optimizers. Each variable should be assigned to "
                        "exactly one optimizer."
                    )
                self._var_to_optimizer_idx[var_key] = opt_idx

        # Build each inner optimizer with its assigned variables
        for opt_idx, (opt, variables) in enumerate(
            zip(self._inner_optimizers, self._optimizer_variables_mapping)
        ):
            if variables:
                with backend.name_scope(
                    f"{self.name}_optimizer_{opt_idx}", caller=self
                ):
                    opt.build(variables)

        super().build(var_list)

    @property
    def variables(self):
        all_variables = self._variables[:]
        for opt in self._inner_optimizers:
            all_variables.extend(opt.variables)
        return all_variables

    @property
    def learning_rate(self):
        """Returns the learning rate of the first optimizer."""
        if self._inner_optimizers:
            return self._inner_optimizers[0].learning_rate
        return 0.0

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        """Sets the learning rate on the first optimizer."""
        if self._inner_optimizers:
            self._inner_optimizers[0].learning_rate = learning_rate

    @property
    def iterations(self):
        """Returns the iteration count (shared across all optimizers)."""
        if self._inner_optimizers:
            return self._inner_optimizers[0].iterations
        return self._iterations

    def apply(self, grads, trainable_variables=None):
        """Apply gradients to variables using the appropriate optimizer.

        Each gradient is routed to the optimizer that was assigned to the
        corresponding variable during construction.
        """
        if len(grads) == 0:
            return

        if trainable_variables is None:
            if not self.built:
                raise ValueError(
                    "When passing `grads` without `variables`, the optimizer "
                    "must already be built on a list of variables. "
                    "Call `optimizer.build(trainable_variables)` first."
                )
            trainable_variables = self._trainable_variables
        else:
            trainable_variables = list(trainable_variables)
            if not self.built:
                with backend.name_scope(self.name, caller=self):
                    self.build(trainable_variables)
                self.built = True

        with backend.name_scope(self.name, caller=self):
            # Group gradients by optimizer
            optimizer_grads = [[] for _ in self._inner_optimizers]
            optimizer_vars = [[] for _ in self._inner_optimizers]

            for grad, var in zip(grads, trainable_variables):
                var_key = self._var_key(var)
                if var_key in self._var_to_optimizer_idx:
                    opt_idx = self._var_to_optimizer_idx[var_key]
                    optimizer_grads[opt_idx].append(grad)
                    optimizer_vars[opt_idx].append(var)
                # Variables not assigned to any optimizer are skipped

            # Apply gradients for each optimizer
            for opt_idx, (opt, grads_group, vars_group) in enumerate(
                zip(
                    self._inner_optimizers,
                    optimizer_grads,
                    optimizer_vars,
                )
            ):
                if grads_group:
                    opt.apply(grads_group, vars_group)

    def stateless_apply(self, optimizer_variables, grads, trainable_variables):
        """Stateless version of `apply` that returns modified variables.

        Args:
            optimizer_variables: List of tensors containing the current values
                for the optimizer variables.
            grads: List of gradients to apply.
            trainable_variables: List of tensors containing the current values
                for the model variables.

        Returns:
            A tuple containing two lists of tensors: the updated
            `trainable_variables` and the updated `optimizer_variables`.
        """
        if not self.built:
            raise ValueError(
                f"To call `stateless_apply`, {self.__class__.__name__} "
                "must be built (i.e. its variables must have been created). "
                "You can build it via `optimizer.build(trainable_variables)`."
            )

        # Split optimizer_variables by inner optimizer
        own_var_count = len(self._variables)
        own_variables = optimizer_variables[:own_var_count]
        remaining_variables = optimizer_variables[own_var_count:]

        inner_opt_variables = []
        offset = 0
        for opt in self._inner_optimizers:
            opt_var_count = len(opt.variables)
            inner_opt_variables.append(
                remaining_variables[offset : offset + opt_var_count]
            )
            offset += opt_var_count

        # Group gradients and trainable variables by optimizer
        optimizer_grads = [[] for _ in self._inner_optimizers]
        optimizer_train_vars = [[] for _ in self._inner_optimizers]
        optimizer_train_var_indices = [[] for _ in self._inner_optimizers]

        for i, (grad, var) in enumerate(
            zip(grads, self._trainable_variables)
        ):
            var_key = self._var_key(var)
            if var_key in self._var_to_optimizer_idx:
                opt_idx = self._var_to_optimizer_idx[var_key]
                optimizer_grads[opt_idx].append(grad)
                optimizer_train_vars[opt_idx].append(trainable_variables[i])
                optimizer_train_var_indices[opt_idx].append(i)

        # Apply stateless updates for each optimizer
        new_trainable_variables = list(trainable_variables)
        new_inner_opt_variables = []

        for opt_idx, opt in enumerate(self._inner_optimizers):
            grads_group = optimizer_grads[opt_idx]
            vars_group = optimizer_train_vars[opt_idx]
            indices = optimizer_train_var_indices[opt_idx]
            opt_vars = inner_opt_variables[opt_idx]

            if grads_group:
                updated_vars, updated_opt_vars = opt.stateless_apply(
                    opt_vars, grads_group, vars_group
                )
                # Update the trainable variables at the correct indices
                for j, idx in enumerate(indices):
                    new_trainable_variables[idx] = updated_vars[j]
                new_inner_opt_variables.extend(updated_opt_vars)
            else:
                new_inner_opt_variables.extend(opt_vars)

        new_optimizer_variables = list(own_variables) + new_inner_opt_variables
        return new_trainable_variables, new_optimizer_variables

    def scale_loss(self, loss):
        """Scale the loss before computing gradients.

        Uses the loss scale factor from this optimizer if set, otherwise
        returns the loss unchanged.
        """
        if self.loss_scale_factor is not None:
            return loss * self.loss_scale_factor
        return loss

    def finalize_variable_values(self, var_list):
        """Finalize variable values after training.

        Calls finalize_variable_values on each inner optimizer with its
        corresponding variables.
        """
        for opt, opt_vars in zip(
            self._inner_optimizers, self._optimizer_variables_mapping
        ):
            if opt_vars:
                opt.finalize_variable_values(opt_vars)

    def get_config(self):
        """Returns the optimizer configuration as a Python dict."""
        optimizers_config = []
        for opt, variables in zip(
            self._inner_optimizers, self._optimizer_variables_mapping
        ):
            # Serialize the optimizer
            opt_config = serialization_lib.serialize_keras_object(opt)
            # Store variable paths for reconstruction
            var_paths = []
            for var in variables:
                if hasattr(var, "path"):
                    var_paths.append(var.path)
                else:
                    var_paths.append(var.name)
            optimizers_config.append({
                "optimizer": opt_config,
                "variable_paths": var_paths,
            })

        return {
            "name": self.name,
            "optimizers_config": optimizers_config,
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Creates a MultiOptimizer from its config.

        Note: This method reconstructs the optimizer instances but cannot
        automatically reconstruct the variable mappings. You will need to
        re-assign variables after loading.

        For full model serialization including variable mappings, use
        `model.save()` and `keras.models.load_model()` instead.
        """
        optimizers_config = config.get("optimizers_config", [])
        optimizers_and_variables = []

        for opt_config in optimizers_config:
            opt = serialization_lib.deserialize_keras_object(
                opt_config["optimizer"],
                custom_objects=custom_objects,
            )
            # Variables cannot be automatically reconstructed from paths
            # They need to be re-assigned after loading
            optimizers_and_variables.append((opt, []))

        instance = cls(
            optimizers_and_variables=optimizers_and_variables,
            name=config.get("name"),
        )
        # Store paths for potential reconstruction
        instance._variable_paths = [
            opt_config.get("variable_paths", [])
            for opt_config in optimizers_config
        ]
        return instance

    def _get_optimizer_for_variable(self, variable):
        """Returns the optimizer assigned to the given variable."""
        var_key = self._var_key(variable)
        if var_key in self._var_to_optimizer_idx:
            return self._inner_optimizers[self._var_to_optimizer_idx[var_key]]
        return None

    def get_optimizer(self, index):
        """Returns the optimizer at the given index.

        Args:
            index: The index of the optimizer to retrieve.

        Returns:
            The optimizer instance at the given index.
        """
        if index < 0 or index >= len(self._inner_optimizers):
            raise IndexError(
                f"Optimizer index {index} out of range. "
                f"This MultiOptimizer has {len(self._inner_optimizers)} "
                "optimizers."
            )
        return self._inner_optimizers[index]

    @property
    def num_optimizers(self):
        """Returns the number of inner optimizers."""
        return len(self._inner_optimizers)
