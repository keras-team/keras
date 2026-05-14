import re
from collections.abc import MutableMapping

from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.optimizers import optimizer
from keras.src.saving import serialization_lib
from keras.src.utils import tracking


@keras_export("keras.optimizers.OptimizerMap")
class OptimizerMap(MutableMapping):
    """A class mapping variables to optimizers.

    Args:
        default_optimizer: Default Keras `Optimizer`
            for any unmapped variables.
    """

    def __init__(self, default_optimizer):
        if not isinstance(default_optimizer, optimizer.Optimizer):
            raise TypeError(
                "default_optimizer must be a Keras Optimizer instance. "
                f"Received: {default_optimizer}"
            )
        self._default_optimizer = default_optimizer
        self._optimizer_map = dict()

    @property
    def default_optimizer(self):
        return self._default_optimizer

    def __getitem__(self, key):
        """Retrieves the corresponding `Optimizer` by the string key.

        This method first attempts an exact key match. If no exact match is
        found, it treats all keys in the map as regular expression patterns
        and uses `re.fullmatch` to find a policy.

        For example,
        to apply a optimizer to all sublayers of an `encoder` block,
        the key should be explicitly set to `"encoder/.*"`. A key of
        `"encoder"` will only match the layer with that exact path.

        Args:
            key: str. The key to query for an `Optimizer`.

        Returns:
            The corresponding `Optimizer`. If no match is found, this method
            returns `self.default_optimizer`.

        Raises:
            ValueError: If the `key` matches more than one regex pattern in the
            map.

        Example:

        ```python
        >>> from keras.src import optimizers
        >>> from keras.src.optimizers.multi_optimizer import OptimizerMap
        >>> opt_adam = optimizers.Adam()
        >>> opt_sgd = optimizers.SGD()
        >>> opt_rmsprop = optimizers.RMSprop()
        >>> optimizer_map = OptimizerMap(default_optimizer=opt_rmsprop)
        >>> optimizer_map["encoder/layer_0/dense"] = opt_adam
        >>> optimizer_map["encoder/.*"] = opt_sgd
        >>> optimizer_map["decoder"] = opt_adam

        >>> optimizer_map["decoder"].name
        'adam'

        >>> optimizer_map["encoder/layer_0/dense"].name
        'adam'

        >>> optimizer_map["encoder/layer_0/attention/query"].name
        'sgd'

        >>> optimizer_map["decoder/layer_0/dense"].name
        'rmsprop'
        ```
        """

        if key in self._optimizer_map:
            return self._optimizer_map[key]

        matching_keys = [
            pattern
            for pattern in self._optimizer_map.keys()
            if re.fullmatch(pattern, key)
        ]

        if len(matching_keys) > 1:
            raise ValueError(f"Multiple optimizers assigned to variable {key}.")
        elif len(matching_keys) == 1:
            return self._optimizer_map[matching_keys[0]]
        return self.default_optimizer

    def __setitem__(self, key, optim):
        """Set optimizer for a given variable.

        Args:
            key: string representing the variable path or regex pattern
            optim: Keras Optimizer instance
        """
        if isinstance(optim, MultiOptimizer):
            raise ValueError(
                f"{key} already exist in the OptimizerMap with "
                f"value {self._optimizer_map[key]}. Please make sure to "
                "not use duplicated keys."
            )
        if not isinstance(key, str):
            raise ValueError(f"key must be a string. Received: {key}")
        if not isinstance(optim, optimizer.Optimizer):
            raise ValueError(
                f"optim must be a Keras Optimizer instance. Received: {optim}"
            )
        self._optimizer_map[key] = optim

    def __delitem__(self, key):
        del self._optimizer_map[key]

    def __len__(self):
        return len(self._optimizer_map)

    def __iter__(self):
        return iter(self._optimizer_map)

    def __call__(self, variable):
        return self[variable.path]

    def get_config(self):
        return {
            "default_optimizer": serialization_lib.serialize_keras_object(
                self._default_optimizer
            ),
            "optimizer_map": {
                k: serialization_lib.serialize_keras_object(v)
                for k, v in self._optimizer_map.items()
            },
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        default_optimizer = serialization_lib.deserialize_keras_object(
            config["default_optimizer"], custom_objects=custom_objects
        )
        obj_map = cls(default_optimizer)
        for k, v in config["optimizer_map"].items():
            obj_map[k] = serialization_lib.deserialize_keras_object(
                v, custom_objects=custom_objects
            )
        return obj_map


@keras_export("keras.optimizers.MultiOptimizer")
class MultiOptimizer(optimizer.Optimizer):
    """An optimizer wrapper that delegates variables to different optimizers.

    Initialize the object with an OptimizerMap instance or a callable
    function that returns an optimizer for a given variable.

    Example:
        model.compile(
            optimizer=MultiOptimizer(
                OptimizerMap(default_optimizer=optimizers.Adam())
            ),
            loss="binary_crossentropy",
        )

        # Or using a callable
        def optimizer_fn(variable):
            if "encoder" in variable.path:
                return optimizers.Adam()
            else:
                return optimizers.SGD()

        model.compile(
            optimizer=MultiOptimizer(optimizer_fn),
            loss="binary_crossentropy",
        )

    To access the attributes of the sub-optimizers, iterate over the
    optimizers using `get_optimizer(i)`:

    For example:

    optimizer = MultiOptimizer(OptimizerMap(
        default_optimizer=optimizers.Adam()
    ))
    optimizer['.encoder'] = optimizers.SGD()

    for i in range(optimizer.num_optimizers):
        optim = optimizer.get_optimizer(i)
        print(optim.learning_rate)
        print(optim.iterations)
        print(optim.loss_scale_factor)
        ...

    The MultiOptimizer class instances will return the `learning_rate`
    and `iterations` attributes of the first optimizer in the
    OptimizerMap. In the case of callable function it will return the
    attributes of the first optimizer returned by the callable function.

    Note: Optimizer-specific callbacks are only supported for the
    first optimizer, irrespective of initialization method.
    For example: ReduceLROnPlateau will only update the learning rate
    of the first optimizer.

    """

    def __init__(self, optim_map, name="multi_optimizer"):
        super().__init__(learning_rate=0.0, name=name)
        self._optim_map = optim_map
        self._inner_optimizers = []
        self._optimizer_to_vars = {}

        if hasattr(self._optim_map, "values"):
            for opt in self._optim_map.values():
                if opt not in self._inner_optimizers:
                    self._inner_optimizers.append(opt)
                    self._optimizer_to_vars[opt] = []

        default_opt = getattr(self._optim_map, "default_optimizer", None)
        if (
            default_opt is not None
            and default_opt not in self._inner_optimizers
        ):
            self._inner_optimizers.append(default_opt)
            self._optimizer_to_vars[default_opt] = []

        # if len(self._inner_optimizers) > 0:
        #     # Use the first optimizer's attributes as defaults
        #     for attr in dir(self._inner_optimizers[0]):
        #         self.__dict__[attr] = getattr(self._inner_optimizers[0], attr)

    def _var_key(self, variable):
        return id(variable)

    @tracking.no_automatic_dependency_tracking
    def build(self, var_list):
        if self.built:
            return

        self._var_to_optimizer_idx = {}

        for var in var_list:
            opt = self._optim_map(var)
            if opt not in self._inner_optimizers:
                self._inner_optimizers.append(opt)
                self._optimizer_to_vars[opt] = []
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
        else:
            return super()._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        if self._inner_optimizers:
            self._inner_optimizers[0].learning_rate = value
        else:
            super()._learning_rate.assign(value)

    @property
    def iterations(self):
        if self._inner_optimizers:
            return self._inner_optimizers[0].iterations
        else:
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

        # Statelessly unscale gradients if loss_scale_factor is set
        if self.loss_scale_factor is not None:
            scale = self.loss_scale_factor
            grads = [g if g is None else g / scale for g in grads]

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
        if self._inner_optimizers:
            return self._inner_optimizers[0].scale_loss(loss)
        else:
            raise ValueError("MultiOptimizer has no inner optimizers.")

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
        return {
            "optim_map": serialization_lib.serialize_keras_object(
                self._optim_map
            ),
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(
            optim_map=serialization_lib.deserialize_keras_object(
                config.pop("optim_map"), custom_objects=custom_objects
            ),
            name=config.pop("name"),
        )
