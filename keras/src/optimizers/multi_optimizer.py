import re
from collections.abc import MutableMapping

from keras.src.api_export import keras_export
from keras.src.optimizers.optimizer import Optimizer
from keras.src.saving import serialization_lib


@keras_export("keras.optimizers.OptimizerMap")
class OptimizerMap(MutableMapping):
    """A class mapping variables to optimizers.

    Args:
        default_optimizer: Default Keras `Optimizer`
            for any unmapped variables.
    """

    def __init__(self, default_optimizer, optimizer_map=None):
        if not isinstance(default_optimizer, Optimizer):
            raise TypeError(
                "default_optimizer must be a Keras Optimizer instance. "
                f"Received: {default_optimizer}"
            )
        if optimizer_map is not None and not isinstance(optimizer_map, dict):
            raise TypeError(
                f"optimizer_map must be a dictionary. Received: {optimizer_map}"
            )
        self._default_optimizer = default_optimizer
        self._optimizer_map = optimizer_map or dict()

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

    def __setitem__(self, key, optimizer):
        """Set optimizer for a given variable.

        Args:
            key: string representing the variable path or regex pattern
            optimizer: Keras Optimizer instance
        """
        if key in self._optimizer_map:
            raise ValueError(
                f"'{key}' already exists in the OptimizerMap with "
                f"value {self._optimizer_map[key]}. Please make sure to "
                "not use duplicated keys."
            )
        if not isinstance(key, str):
            raise ValueError(f"key must be a string. Received: {key}")
        if not isinstance(optimizer, Optimizer):
            raise ValueError(
                f"optimizer must be a Keras Optimizer instance. "
                f"Received: {optimizer}"
            )
        if isinstance(optimizer, MultiOptimizer):
            raise ValueError(
                "MultiOptimizer cannot be nested inside an OptimizerMap."
            )
        self._optimizer_map[key] = optimizer

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
            "optimizer_map": serialization_lib.serialize_keras_object(
                self._optimizer_map
            ),
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        default_optimizer = serialization_lib.deserialize_keras_object(
            config["default_optimizer"], custom_objects=custom_objects
        )
        optimizer_map = serialization_lib.deserialize_keras_object(
            config["optimizer_map"], custom_objects=custom_objects
        )
        obj_map = cls(default_optimizer, optimizer_map)
        return obj_map


@keras_export("keras.optimizers.MultiOptimizer")
class MultiOptimizer(Optimizer):
    """An optimizer wrapper that delegates variables to different optimizers.

    Initialize the object with an OptimizerMap instance or a callable
    function that returns an optimizer for a given variable.

    Example:
        model.compile(
            optimizer=MultiOptimizer(
                OptimizerMap(default_optimizer=optimizers.SGD(),
                {"encoder/.*": optimizers.Adam()})
            ),
            loss="binary_crossentropy",
        )

        # Or using a callable
        def optimizer_selector(variable):
            if "encoder" in variable.path:
                return optimizers.Adam()
            else:
                return optimizers.SGD()

        model.compile(
            optimizer=MultiOptimizer(optimizer_selector),
            loss="binary_crossentropy",
        )

    To access the attributes of the sub-optimizers, iterate over the
    optimizers using `.optimizers`:

    For example:

    optimizer = MultiOptimizer(OptimizerMap(
        default_optimizer=optimizers.Adam()
    ))
    optimizer['.encoder'] = optimizers.SGD()

    for optim in optimizer.optimizers:
        print(optim.learning_rate)
        print(optim.iterations)
        print(optim.loss_scale_factor)
        ...

    The MultiOptimizer class instances will not expose `learning_rate`
    attribute and will raise an error if accessed. This is because the
    learning rate might be different for different sub-optimizers.

    Note: Optimizer-specific callbacks are not supported yet.

    """

    def __init__(self, optimizer_map, loss_scale_factor=None, name=None):
        """
        Initialize the MultiOptimizer.

        Args:
            optimizer_map: An OptimizerMap instance or a callable function that
                returns an optimizer for a given variable.
            loss_scale_factor: It overrides the loss_scale_factor passed
            to the sub-optimizers.
            name: The name of the optimizer.
        """
        super().__init__(
            learning_rate=0.0,
            loss_scale_factor=loss_scale_factor,
            name=name,
        )
        self._optimizer_map = optimizer_map
        self._inner_optimizers = []

        if hasattr(self._optimizer_map, "values"):
            for opt in self._optimizer_map.values():
                if opt not in self._inner_optimizers:
                    opt.loss_scale_factor = loss_scale_factor
                    self._inner_optimizers.append(opt)

        default_opt = getattr(self._optimizer_map, "default_optimizer", None)
        if (
            default_opt is not None
            and default_opt not in self._inner_optimizers
        ):
            default_opt.loss_scale_factor = loss_scale_factor
            self._inner_optimizers.append(default_opt)

    def build(self, var_list):
        if self.built:
            return

        self._var_to_optimizer_idx = {}
        self._trainable_variables_indices = {}
        optimizer_vars = [[] for _ in range(len(self._inner_optimizers))]

        for i, var in enumerate(var_list):
            opt = self._optimizer_map(var)
            if not isinstance(opt, Optimizer):
                raise ValueError(
                    f"Optimizer for variable {var} is not "
                    "an Optimizer instance."
                )
            if opt not in self._inner_optimizers:
                opt.loss_scale_factor = self.loss_scale_factor
                self._inner_optimizers.append(opt)
                optimizer_vars.append([])

            idx = self._inner_optimizers.index(opt)
            self._var_to_optimizer_idx[self._var_key(var)] = idx
            self._trainable_variables_indices[self._var_key(var)] = i
            optimizer_vars[idx].append(var)

        # Build all sub-optimizers
        for opt, variables in zip(self._inner_optimizers, optimizer_vars):
            if variables:
                opt.build(variables)

        super().build(var_list)

    @property
    def optimizers(self):
        return self._inner_optimizers

    @property
    def variables(self):
        # MultiOptimizer iterations + all sub-optimizer variables
        vars = self._variables[:]
        for opt in self._inner_optimizers:
            vars.extend(opt.variables)
        return vars

    @property
    def learning_rate(self):
        raise AttributeError(
            "Learning rate cannot be accessed on a MultiOptimizer. "
            "Access the learning rate on the individual sub-optimizers instead."
        )

    @learning_rate.setter
    def learning_rate(self, value):
        raise AttributeError(
            "Learning rate cannot be set on a MultiOptimizer. "
            "Set the learning rate on the individual sub-optimizers instead."
        )

    @property
    def iterations(self):
        return self._iterations

    def apply(self, grads, trainable_variables=None):
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

        if not self.built:
            self.build(trainable_variables)
            self.built = True

        if len(grads) != len(self._trainable_variables_indices):
            raise ValueError(
                "Gradients must match trainable variables one-to-one. "
                f"Received {len(grads)} gradients and "
                f"{len(self._trainable_variables_indices)} variables."
            )

        grads_and_vars = [[] for _ in range(len(self._inner_optimizers))]
        for grad, var in zip(grads, trainable_variables):
            idx = self._var_to_optimizer_idx[self._var_key(var)]
            grads_and_vars[idx].append((grad, var))

        # Dispatch apply calls
        for opt, sub_grads_and_vars in zip(
            self._inner_optimizers, grads_and_vars
        ):
            if sub_grads_and_vars:
                sub_grads, sub_vars = zip(*sub_grads_and_vars)
                opt.apply(list(sub_grads), list(sub_vars))

        self._iterations.assign_add(1)

    def finalize_variable_values(self, var_list):
        optimizer_vars = [[] for _ in range(len(self._inner_optimizers))]
        for var in var_list:
            idx = self._var_to_optimizer_idx[self._var_key(var)]
            optimizer_vars[idx].append(var)

        for opt, variables in zip(self._inner_optimizers, optimizer_vars):
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
            "optimizer_map": serialization_lib.serialize_keras_object(
                self._optimizer_map
            ),
            "loss_scale_factor": self.loss_scale_factor,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = config.copy()
        config["optimizer_map"] = serialization_lib.deserialize_keras_object(
            config["optimizer_map"], custom_objects=custom_objects
        )
        return cls(**config)
