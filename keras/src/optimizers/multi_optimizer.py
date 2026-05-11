import re

from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend.common.variables import Variable
from keras.src.optimizers import optimizer
from keras.src.saving import serialization_lib
from keras.src.utils import tracking


@keras_export("keras.optimizers.OptimizerMap")
class OptimizerMap:
    """A class mapping variables to optimizers.

    Args:
        optimizer: A list of optimizer instances.
        variable_identifier: A list of variable identifiers matching the
            optimizer list. Identifiers can be string regex patterns, Keras
            Variables, list/tuple of Keras Variables, or callables returning a
            boolean.
    """

    def __init__(self, optimizer, variable_identifier):
        self.optimizer = optimizer
        self.variable_identifier = variable_identifier

    def __call__(self, variable):
        """Evaluates a single variable and returns its assigned optimizer.

        Args:
            variable: A single Keras Variable.

        Returns:
            An Optimizer instance if matched, or None.

        Raises:
            ValueError: If multiple different optimizers are mapped to the same
                variable.
        """
        matched_optimizers = []
        for opt, identifier in zip(self.optimizer, self.variable_identifier):
            if self._match(variable, identifier):
                if opt not in matched_optimizers:
                    matched_optimizers.append(opt)

        if len(matched_optimizers) > 1:
            raise ValueError(
                f"Multiple optimizers assigned to variable "
                f"{variable.path or variable.name}: {matched_optimizers}"
            )
        elif len(matched_optimizers) == 1:
            return matched_optimizers[0]
        return None

    def _match(self, variable, identifier):
        # If callable, evaluate it
        if callable(identifier) and not isinstance(identifier, str):
            return bool(identifier(variable))
        # If string, treat it as a regex pattern
        if isinstance(identifier, str):
            path = getattr(variable, "path", "") or ""
            name = getattr(variable, "name", "") or ""
            return bool(
                re.match(identifier, path) or re.match(identifier, name)
            )
        # list/tuple/set
        if isinstance(identifier, (list, tuple, set)):
            return any(variable is item for item in identifier)
        # Direct identity comparison
        return variable is identifier


@keras_export("keras.optimizers.MultiOptimizer")
class MultiOptimizer(optimizer.Optimizer):
    """An optimizer wrapper that delegates variables to different optimizers.

    Example:

    ```python
    # 1. Define your sub-optimizers
    opt_adam = keras.optimizers.Adam(learning_rate=1e-3)
    opt_sgd = keras.optimizers.SGD(learning_rate=1e-2)
    default_opt = keras.optimizers.RMSprop(learning_rate=1e-4)

    # 2. Create mapping using different identifier types
    opt_map = keras.optimizers.OptimizerMap(
        optimizer=[opt_adam, opt_sgd, opt_adam, opt_sgd, opt_adam],
        variable_identifier=[
            "kernel",                            # variable name or path
            "^dense_1/.*",                       # regex pattern
            keras_var,                           # Keras Variable instance
            lambda var: "bias" in var.name,      # custom callable function
            [var1, var2]                         # list/tuple
        ]
    )

    # 3. Wrap and compile
    multi_opt = keras.optimizers.MultiOptimizer(
        obj_map=opt_map,
        default_optimizer=default_opt
    )
    model.compile(optimizer=multi_opt, loss="mse")
    ```

    Note: You can subclass `OptimizerMap` to define custom routing logic by
    overriding its `__call__(self, variable)` method.

    Args:
        obj_map: An `OptimizerMap` instance containing
               variable-to-optimizer rules.
        default_optimizer: Default Keras `Optimizer` for any unmapped variables.
        name: String. The name of this optimizer. Defaults to
              "multi_optimizer".
    """

    def __init__(self, obj_map, default_optimizer, name=None, **kwargs):
        if not isinstance(obj_map, OptimizerMap):
            raise ValueError(
                f"obj_map must be an instance "
                f"of OptimizerMap. Received: {obj_map}"
            )
        if not isinstance(default_optimizer, optimizer.Optimizer):
            raise ValueError(
                f"default_optimizer must be a Keras Optimizer instance. "
                f"Received: {default_optimizer}"
            )

        super().__init__(learning_rate=0.0, name=name)

        self.obj_map = obj_map
        self.default_optimizer = default_optimizer

        # Disable loss scaling for all inner optimizers to avoid double scaling
        self.default_optimizer.loss_scale_factor = None
        for opt in self.obj_map.optimizer:
            opt.loss_scale_factor = None

        # Unique inner optimizers list, keeping default optimizer at the end
        unique_opts = []
        for opt in self.obj_map.optimizer:
            if opt not in unique_opts:
                unique_opts.append(opt)
        if self.default_optimizer not in unique_opts:
            unique_opts.append(self.default_optimizer)
        else:
            unique_opts.remove(self.default_optimizer)
            unique_opts.append(self.default_optimizer)

        self._inner_optimizers = unique_opts

    def _var_key(self, variable):
        return id(variable)

    @tracking.no_automatic_dependency_tracking
    def build(self, var_list):
        if self.built:
            return

        self._var_to_optimizer_idx = {}
        self._optimizer_to_vars = {opt: [] for opt in self._inner_optimizers}

        for var in var_list:
            opt = self.obj_map(var)
            if opt is None:
                opt = self.default_optimizer

            idx = self._inner_optimizers.index(opt)
            self._var_to_optimizer_idx[self._var_key(var)] = idx
            self._optimizer_to_vars[opt].append(var)

        # Build all sub-optimizers
        for opt, variables in self._optimizer_to_vars.items():
            if variables:
                opt.build(variables)

        super().build(var_list)

    def _get_optimizer_for_variable(self, variable):
        idx = self._var_to_optimizer_idx.get(self._var_key(variable), None)
        if idx is None:
            raise ValueError(f"Variable {variable} not found in any optimizer.")
        return self._inner_optimizers[idx]

    def get_optimizer(self, index):
        return self._inner_optimizers[index]

    @property
    def num_optimizers(self):
        return len(self._inner_optimizers)

    @property
    def variables(self):
        # MultiOptimizer iterations + all sub-optimizer variables
        vars = self._variables[:]
        for opt in self._inner_optimizers:
            vars.extend(opt.variables)
        return vars

    @property
    def learning_rate(self):
        if self._inner_optimizers:
            return self._inner_optimizers[0].learning_rate
        return super().learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        if self._inner_optimizers:
            self._inner_optimizers[0].learning_rate = value
        else:
            super().learning_rate = value

    @property
    def iterations(self):
        # master and inner optimizer iterations will be in sync.
        return self._iterations

    def apply(self, grads, trainable_variables=None):
        if len(grads) == 0:
            return

        if trainable_variables is None:
            trainable_variables = self._trainable_variables

        if not self.built:
            self.build(trainable_variables)
            self.built = True

        # Group gradients and variables by sub-optimizer
        optimizer_to_grads_and_vars = {
            opt: [] for opt in self._inner_optimizers
        }
        for grad, var in zip(grads, trainable_variables):
            opt = self._get_optimizer_for_variable(var)
            optimizer_to_grads_and_vars[opt].append((grad, var))

        # Dispatch apply calls
        for opt, grads_and_vars in optimizer_to_grads_and_vars.items():
            if grads_and_vars:
                sub_grads, sub_vars = zip(*grads_and_vars)
                opt.apply(list(sub_grads), list(sub_vars))

        # Keep master iterations in sync
        self._iterations.assign_add(1)

    def stateless_apply(self, optimizer_variables, grads, trainable_variables):
        if len(grads) == 0:
            return trainable_variables, optimizer_variables

        if trainable_variables is None:
            trainable_variables = self._trainable_variables

        own_var_count = len(self._variables)
        own_variables = optimizer_variables[:own_var_count]
        remaining_opt_vars = optimizer_variables[own_var_count:]

        inner_opt_variables = []
        offset = 0
        for opt in self._inner_optimizers:
            opt_var_count = len(opt.variables)
            opt_vars = remaining_opt_vars[offset : offset + opt_var_count]
            inner_opt_variables.append(opt_vars)
            offset += opt_var_count

        optimizer_grads = [[] for _ in self._inner_optimizers]
        optimizer_train_vars = [[] for _ in self._inner_optimizers]
        optimizer_train_var_indices = [[] for _ in self._inner_optimizers]

        for i, (grad, var) in enumerate(zip(grads, trainable_variables)):
            # Map the incoming stateless tracer array
            # to the static Keras Variable via index alignment
            keras_var = self._trainable_variables[i]
            opt = self._get_optimizer_for_variable(keras_var)
            opt_idx = self._inner_optimizers.index(opt)

            optimizer_grads[opt_idx].append(grad)
            optimizer_train_vars[opt_idx].append(var)
            optimizer_train_var_indices[opt_idx].append(i)

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

        # Statelessly increment the master iterations counter
        new_own_variables = list(own_variables)
        if new_own_variables:
            new_own_variables[0] = ops.cast(
                new_own_variables[0] + 1, new_own_variables[0].dtype
            )

        new_optimizer_variables = new_own_variables + new_inner_opt_variables
        return new_trainable_variables, new_optimizer_variables

    def scale_loss(self, loss):
        if self.loss_scale_factor is not None:
            return loss * self.loss_scale_factor
        return loss

    def finalize_variable_values(self, var_list):
        optimizer_to_vars = {opt: [] for opt in self._inner_optimizers}
        for var in var_list:
            opt = self._get_optimizer_for_variable(var)
            optimizer_to_vars[opt].append(var)

        for opt, variables in optimizer_to_vars.items():
            if variables:
                opt.finalize_variable_values(variables)

    def set_weights(self, weights):
        if not self.built:
            raise ValueError(
                "You are calling `set_weights()` on an optimizer that has not "
                "yet been built. Please call "
                "`optimizer.build(trainable_variables)` to create the "
                "optimizer weights before calling `set_weights()`."
            )

        # Distribute own variables dynamically (iterations, learning_rate, etc.)
        own_var_count = len(self._variables)
        for i in range(own_var_count):
            self._variables[i].assign(weights[i])

        idx = own_var_count
        for opt in self._inner_optimizers:
            num_opt_vars = len(opt.variables)
            if num_opt_vars > 0:
                opt_weights = weights[idx : idx + num_opt_vars]
                opt.set_weights(opt_weights)
                idx += num_opt_vars

    def get_config(self):
        config = super().get_config()

        serialized_identifiers = []
        for identifier in self.obj_map.variable_identifier:
            if isinstance(identifier, Variable):
                serialized_identifiers.append(
                    f"^{re.escape(identifier.path or identifier.name)}$"
                )
            elif isinstance(identifier, (list, tuple, set)):
                if any(isinstance(item, Variable) for item in identifier):
                    escaped_paths = []
                    for item in identifier:
                        if isinstance(item, Variable):
                            escaped_paths.append(
                                re.escape(item.path or item.name)
                            )
                        else:
                            escaped_paths.append(re.escape(str(item)))
                    or_pattern = "|".join(escaped_paths)
                    serialized_identifiers.append(f"^({or_pattern})$")
                else:
                    serialized_identifiers.append(identifier)
            elif callable(identifier):
                serialized_identifiers.append(
                    serialization_lib.serialize_keras_object(identifier)
                )
            else:
                serialized_identifiers.append(identifier)

        serialized_map = {
            "class_name": self.obj_map.__class__.__name__,
            "config": {
                "optimizer": [
                    serialization_lib.serialize_keras_object(opt)
                    for opt in self.obj_map.optimizer
                ],
                "variable_identifier": serialized_identifiers,
            },
        }
        config.update(
            {
                "obj_map": serialized_map,
                "default_optimizer": serialization_lib.serialize_keras_object(
                    self.default_optimizer
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        obj_map_config = config.pop("obj_map")
        default_opt_config = config.pop("default_optimizer")

        default_optimizer = serialization_lib.deserialize_keras_object(
            default_opt_config, custom_objects=custom_objects
        )

        serialized_opts = obj_map_config["config"]["optimizer"]
        optimizers_list = [
            serialization_lib.deserialize_keras_object(
                opt_config, custom_objects=custom_objects
            )
            for opt_config in serialized_opts
        ]

        variable_identifiers = [
            serialization_lib.deserialize_keras_object(
                ident, custom_objects=custom_objects
            )
            for ident in obj_map_config["config"]["variable_identifier"]
        ]

        map_cls = cls._get_map_class(
            obj_map_config["class_name"], custom_objects
        )
        obj_map = map_cls(optimizers_list, variable_identifiers)

        return cls(
            obj_map=obj_map, default_optimizer=default_optimizer, **config
        )

    @classmethod
    def _get_map_class(cls, class_name, custom_objects=None):
        if custom_objects and class_name in custom_objects:
            return custom_objects[class_name]
        globals_dict = globals()
        if class_name in globals_dict:
            return globals_dict[class_name]
        raise ValueError(
            f"Could not find class {class_name} for "
            "OptimizerMap deserialization."
        )
