from keras.src import backend
from keras.src.api_export import keras_export
from keras.src import optimizers as keras_optimizers
from keras.src.optimizers import optimizer as base_optimizer
from keras.src.saving import serialization_lib
from keras.src.utils import tracking


@keras_export("keras.optimizers.DispatchOptimizer")
class DispatchOptimizer(base_optimizer.Optimizer):
    """Dispatches to a per-variable optimizers if applicable.

    Allows per-variable optimizer customizations by examining
    each variable's `variable.optimizer` property, and dispatching
    to the appropriate underlying optimizer if applicable.  If multiple
    variables share the same optimizer, the variables are grouped
    together for a single dispatched call.  Variables without a
    `variable.optimizer` are dispatched to the default optimizer.

    Args:
        default_optimizer: The `keras.optimizers.Optimizer` instance
            to use by default.
        {{base_optimizer_keyword_args}}
    """

    def __init__(
        self,
        default_optimizer="rmsprop",
        **kwargs,
    ):
        if "learning_rate" in kwargs:
            raise ValueError(
                "DispatchOptimizer does not support a learning rate. "
                "Instead, set the `learning_rate` directly on the "
                "`default_optimizer`."
            )
        super().__init__(learning_rate=0.0, **kwargs)
        self._default_optimizer = default_optimizer
        self._optimizers = [self._default_optimizer]
        self.built = False

    def _has_custom_optimizer(self, variable):
        return (
            hasattr(variable, "optimizer")
            and variable.optimizer is not None
            and variable.optimizer != self  # Prevent infinite recursion.
            and variable.optimizer != self._default_optimizer
        )

    def _separate_per_optimizer(self, var_list, value_list):
        """Separate a list of values into per-optimizer lists.

        Args:
            var_list: List of variables to use for determining the optimizer.
            value_list: List of values to separate per-optimizer.

        Returns:
            Nested lists of variables per optimizer.
        """
        lists = [[] for _ in range(len(self._optimizers))]
        for var, value in zip(var_list, value_list):
            odx = self._variable_to_optimizer_index[self._var_key(var)]
            lists[odx].append(value)

        return lists

    @tracking.no_automatic_dependency_tracking
    def build(self, var_list):
        self._default_optimizer = keras_optimizers.get(self._default_optimizer)
        # Extract optimizers and separate variables into groups.
        optimizers = [self._default_optimizer]
        optimizer_index = {id(self._default_optimizer): 0}
        # Map of training_variable -> optimizer, required for apply().
        variable_to_optimizer_index = {}
        # Map of training_variable index -> optimizer, required for
        # stateless_apply().
        variable_index_to_optimizer_index = []

        self._optimizers = optimizers
        self._trainable_variables = var_list[:]
        self._variable_to_optimizer_index = variable_to_optimizer_index
        self._variable_index_to_optimizer_index = (
            variable_index_to_optimizer_index
        )

        # First do a pass to check if we even need to dispatch
        # to per-variable optimizers.  If not, we can just build
        # the default optimizer.
        needs_dispatch = False
        for var in var_list:
            if self._has_custom_optimizer(var):
                needs_dispatch = True
                break

        if needs_dispatch:
            for var in var_list:
                optimizer_idx = 0
                if self._has_custom_optimizer(var):
                    optimizer = var.optimizer
                    optimizer_key = id(optimizer)
                    optimizer_idx = optimizer_index.get(optimizer_key, None)
                    if optimizer_idx is None:
                        optimizer_idx = len(optimizers)
                        optimizers.append(optimizer)
                        optimizer_index[optimizer_key] = optimizer_idx

                variable_to_optimizer_index[self._var_key(var)] = optimizer_idx
                variable_index_to_optimizer_index.append(optimizer_idx)

            # Build all optimizers.
            vars_per_optimizer_lists = self._separate_per_optimizer(
                var_list, var_list
            )
            for optimizer, optimizer_vars in zip(
                optimizers, vars_per_optimizer_lists
            ):
                optimizer.build(optimizer_vars)
        else:
            self._default_optimizer.build(var_list)

        # Separate optimizer variables for stateless_call.
        # Optimizer variables are simply stacked.  See self.variables.
        oidx = 0
        optimizer_variable_offsets = []
        for optimizer in optimizers:
            optimizer_variable_offsets.append(oidx)
            oidx += len(optimizer.variables)
        optimizer_variable_offsets.append(oidx)
        self._optimizer_variable_offsets = optimizer_variable_offsets

        self.built = True

    def set_weights(self, weights):
        raise ValueError(
            "DispatchOptimizer does not support adding weights. "
            "All weights must be set in the underlying optimizers."
        )

    @property
    def variables(self):
        if not self.built:
            return []

        if len(self._optimizers) == 1:
            return self._default_optimizer.variables

        # Stack all optimizer variables.
        variables = []
        for optimizer in self._optimizers:
            variables.extend(optimizer.variables)
        return variables

    def stateless_apply(self, optimizer_variables, grads, trainable_variables):
        if not self.built:
            raise ValueError(
                f"To call `stateless_apply`, {self.__class__.__name__} "
                "must be built (i.e. its variables must have been created). "
                "You can build it via `optimizer.build(trainable_variables)`."
            )
        if len(optimizer_variables) != self._optimizer_variable_offsets[-1]:
            raise ValueError(
                "Argument `optimizer_variables` must be a list of tensors "
                f"corresponding 1:1 to {self.__class__.__name__}().variables. "
                f"Received list with length {len(optimizer_variables)}, but "
                f"expected {self._optimizer_variable_offsets[-1]} variables."
            )
        if len(self._optimizers) > 1 and len(trainable_variables) != len(
            self._variable_index_to_optimizer_index
        ):
            raise ValueError(
                "Argument `trainable_variables` must be a list of tensors "
                "corresponding 1:1 to the trainable variables list that "
                "the optimizer was built with. Received "
                f"len(trainable_variables) == {len(trainable_variables)} "
                "whereas the optimizer was built with "
                f"{len(self._variable_index_to_optimizer_index)} variables."
            )

        num_optimizers = len(self._optimizers)
        if num_optimizers == 1:
            return self._default_optimizer.stateless_apply(
                optimizer_variables, grads, trainable_variables
            )

        # Separate into per-optimizer lists.
        optimizer_params = []
        for i in range(num_optimizers):
            optimizer_params.append(
                optimizer_variables[
                    self._optimizer_variable_offsets[
                        i
                    ] : self._optimizer_variable_offsets[i + 1]
                ]
            )

        per_optimizer_grads = [[] for _ in range(num_optimizers)]
        per_optimizer_variables = [[] for _ in range(num_optimizers)]
        reverse_map = [[] for _ in range(num_optimizers)]
        for i in range(len(trainable_variables)):
            oidx = self._variable_index_to_optimizer_index[i]
            per_optimizer_grads[oidx].append(grads[i])
            per_optimizer_variables[oidx].append(trainable_variables[i])
            reverse_map[oidx].append(i)

        # Apply and update lists.
        updated_optimizer_variables = []
        updated_trainable_variables = [None] * len(trainable_variables)
        for optimizer, ovars, tgrads, tvars, tidxs in zip(
            self._optimizers,
            optimizer_params,
            per_optimizer_grads,
            per_optimizer_variables,
            reverse_map,
        ):
            tvs, ovs = optimizer.stateless_apply(ovars, tgrads, tvars)
            updated_optimizer_variables.extend(ovs)
            # Scatter training vars into correct positions.
            for tv, idx in zip(tvs, tidxs):
                updated_trainable_variables[idx] = tv

        return updated_trainable_variables, updated_optimizer_variables

    def apply(self, grads, trainable_variables=None):
        # Optionally build optimizer.
        if not self.built:
            with backend.name_scope(self.name, caller=self):
                self.build(trainable_variables)

        if len(self._optimizers) == 1:
            return self._default_optimizer.apply(grads, trainable_variables)

        if trainable_variables is None:
            params = self._separate_per_optimizer(
                self._trainable_variables, grads
            )
            for optimizer, grads in zip(self._optimizers, params):
                optimizer.apply(grads)
        else:
            params = self._separate_per_optimizer(
                trainable_variables, zip(grads, trainable_variables)
            )
            for optimizer, apply_params in zip(self._optimizers, params):
                ograds, ovars = zip(*apply_params)
                optimizer.apply(ograds, ovars)

    @property
    def learning_rate(self):
        return self._default_optimizer.learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._default_optimizer.learning_rate = learning_rate

    @property
    def iterations(self):
        # It's possible the default optimizer has no trainable
        # variables, so it's iteration count is never incremented.
        # Take the maximum of all iteration counts.
        return max([optimizer.iterations for optimizer in self._optimizers])

    def finalize_variable_values(self, var_list):
        if self.built:
            if len(self._optimizers) == 1:
                self._default_optimizer.finalize_variable_values(var_list)
            else:
                vars_per_optimizer_lists = self._separate_per_optimizer(
                    var_list, var_list
                )
                for optimizer, optimizer_vars in zip(
                    self._optimizers, vars_per_optimizer_lists
                ):
                    optimizer.finalize_variable_values(optimizer_vars)

    def get_config(self):
        config = super().get_config()
        config["default_optimizer"] = serialization_lib.serialize_keras_object(
            self._default_optimizer
        )
        del config["learning_rate"]
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        default_optimizer = serialization_lib.deserialize_keras_object(
            config.pop("default_optimizer"),
            custom_objects=custom_objects,
        )
        return cls(default_optimizer, **config)


DispatchOptimizer.__doc__ = DispatchOptimizer.__doc__.replace(
    "{{base_optimizer_keyword_args}}",
    base_optimizer.base_optimizer_keyword_args,
)
