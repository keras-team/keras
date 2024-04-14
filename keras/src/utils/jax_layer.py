import inspect

import numpy as np

from keras.src import backend
from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.saving import serialization_lib
from keras.src.utils import jax_utils
from keras.src.utils import tracking
from keras.src.utils.module_utils import jax


@keras_export("keras.layers.JaxLayer")
class JaxLayer(Layer):
    """Keras Layer that wraps a JAX model.

    This layer enables the use of JAX components within Keras when using JAX as
    the backend for Keras.

    ## Model function

    This layer accepts JAX models in the form of a function, `call_fn`, which
    must take the following arguments with these exact names:

    - `params`: trainable parameters of the model.
    - `state` (*optional*): non-trainable state of the model. Can be omitted if
        the model has no non-trainable state.
    - `rng` (*optional*): a `jax.random.PRNGKey` instance. Can be omitted if the
        model does not need RNGs, neither during training nor during inference.
    - `inputs`: inputs to the model, a JAX array or a `PyTree` of arrays.
    - `training` (*optional*): an argument specifying if we're in training mode
        or inference mode, `True` is passed in training mode. Can be omitted if
        the model behaves the same in training mode and inference mode.

    The `inputs` argument is mandatory. Inputs to the model must be provided via
    a single argument. If the JAX model takes multiple inputs as separate
    arguments, they must be combined into a single structure, for instance in a
    `tuple` or a `dict`.

    ## Model weights initialization

    The initialization of the `params` and `state` of the model can be handled
    by this layer, in which case the `init_fn` argument must be provided. This
    allows the model to be initialized dynamically with the right shape.
    Alternatively, and if the shape is known, the `params` argument and
    optionally the `state` argument can be used to create an already initialized
    model.

    The `init_fn` function, if provided, must take the following arguments with
    these exact names:

    - `rng`: a `jax.random.PRNGKey` instance.
    - `inputs`: a JAX array or a `PyTree` of arrays with placeholder values to
        provide the shape of the inputs.
    - `training` (*optional*): an argument specifying if we're in training mode
        or inference mode. `True` is always passed to `init_fn`. Can be omitted
        regardless of whether `call_fn` has a `training` argument.

    ## Models with non-trainable state

    For JAX models that have non-trainable state:

    - `call_fn` must have a `state` argument
    - `call_fn` must return a `tuple` containing the outputs of the model and
        the new non-trainable state of the model
    - `init_fn` must return a `tuple` containing the initial trainable params of
        the model and the initial non-trainable state of the model.

    This code shows a possible combination of `call_fn` and `init_fn` signatures
    for a model with non-trainable state. In this example, the model has a
    `training` argument and an `rng` argument in `call_fn`.

    ```python
    def stateful_call(params, state, rng, inputs, training):
        outputs = ...
        new_state = ...
        return outputs, new_state

    def stateful_init(rng, inputs):
        initial_params = ...
        initial_state = ...
        return initial_params, initial_state
    ```

    ## Models without non-trainable state

    For JAX models with no non-trainable state:

    - `call_fn` must not have a `state` argument
    - `call_fn` must return only the outputs of the model
    - `init_fn` must return only the initial trainable params of the model.

    This code shows a possible combination of `call_fn` and `init_fn` signatures
    for a model without non-trainable state. In this example, the model does not
    have a `training` argument and does not have an `rng` argument in `call_fn`.

    ```python
    def stateless_call(params, inputs):
        outputs = ...
        return outputs

    def stateless_init(rng, inputs):
        initial_params = ...
        return initial_params
    ```

    ## Conforming to the required signature

    If a model has a different signature than the one required by `JaxLayer`,
    one can easily write a wrapper method to adapt the arguments. This example
    shows a model that has multiple inputs as separate arguments, expects
    multiple RNGs in a `dict`, and has a `deterministic` argument with the
    opposite meaning of `training`. To conform, the inputs are combined in a
    single structure using a `tuple`, the RNG is split and used the populate the
    expected `dict`, and the Boolean flag is negated:

    ```python
    def my_model_fn(params, rngs, input1, input2, deterministic):
        ...
        if not deterministic:
            dropout_rng = rngs["dropout"]
            keep = jax.random.bernoulli(dropout_rng, dropout_rate, x.shape)
            x = jax.numpy.where(keep, x / dropout_rate, 0)
            ...
        ...
        return outputs

    def my_model_wrapper_fn(params, rng, inputs, training):
        input1, input2 = inputs
        rng1, rng2 = jax.random.split(rng)
        rngs = {"dropout": rng1, "preprocessing": rng2}
        deterministic = not training
        return my_model_fn(params, rngs, input1, input2, deterministic)

    keras_layer = JaxLayer(my_model_wrapper_fn, params=initial_params)
    ```

    ## Usage with Haiku modules

    `JaxLayer` enables the use of [Haiku](https://dm-haiku.readthedocs.io)
    components in the form of
    [`haiku.Module`](https://dm-haiku.readthedocs.io/en/latest/api.html#module).
    This is achieved by transforming the module per the Haiku pattern and then
    passing `module.apply` in the `call_fn` parameter and `module.init` in the
    `init_fn` parameter if needed.

    If the model has non-trainable state, it should be transformed with
    [`haiku.transform_with_state`](
      https://dm-haiku.readthedocs.io/en/latest/api.html#haiku.transform_with_state).
    If the model has no non-trainable state, it should be transformed with
    [`haiku.transform`](
      https://dm-haiku.readthedocs.io/en/latest/api.html#haiku.transform).
    Additionally, and optionally, if the module does not use RNGs in "apply", it
    can be transformed with
    [`haiku.without_apply_rng`](
      https://dm-haiku.readthedocs.io/en/latest/api.html#without-apply-rng).

    The following example shows how to create a `JaxLayer` from a Haiku module
    that uses random number generators via `hk.next_rng_key()` and takes a
    training positional argument:

    ```python
    class MyHaikuModule(hk.Module):
        def __call__(self, x, training):
            x = hk.Conv2D(32, (3, 3))(x)
            x = jax.nn.relu(x)
            x = hk.AvgPool((1, 2, 2, 1), (1, 2, 2, 1), "VALID")(x)
            x = hk.Flatten()(x)
            x = hk.Linear(200)(x)
            if training:
                x = hk.dropout(rng=hk.next_rng_key(), rate=0.3, x=x)
            x = jax.nn.relu(x)
            x = hk.Linear(10)(x)
            x = jax.nn.softmax(x)
            return x

    def my_haiku_module_fn(inputs, training):
        module = MyHaikuModule()
        return module(inputs, training)

    transformed_module = hk.transform(my_haiku_module_fn)

    keras_layer = JaxLayer(
        call_fn=transformed_module.apply,
        init_fn=transformed_module.init,
    )
    ```

    Args:
        call_fn: The function to call the model. See description above for the
            list of arguments it takes and the outputs it returns.
        init_fn: the function to call to initialize the model. See description
            above for the list of arguments it takes and the ouputs it returns.
            If `None`, then `params` and/or `state` must be provided.
      params: A `PyTree` containing all the model trainable parameters. This
            allows passing trained parameters or controlling the initialization.
            If both `params` and `state` are `None`, `init_fn` is called at
            build time to initialize the trainable parameters of the model.
      state: A `PyTree` containing all the model non-trainable state. This
            allows passing learned state or controlling the initialization. If
            both `params` and `state` are `None`, and `call_fn` takes a `state`
            argument, then `init_fn` is called at build time to initialize the
            non-trainable state of the model.
      seed: Seed for random number generator. Optional.
    """

    def __init__(
        self,
        call_fn,
        init_fn=None,
        params=None,
        state=None,
        seed=None,
        **kwargs,
    ):
        if backend.backend() != "jax":
            raise ValueError(
                "JaxLayer is only supported with the JAX backend. Current "
                f"backend: {backend.backend()}"
            )

        if init_fn is None and params is None and state is None:
            raise ValueError(
                "`init_fn`, `params` and `state` cannot all be `None`."
            )

        super().__init__(**kwargs)
        self.call_fn = call_fn
        self.init_fn = init_fn
        self.seed_generator = backend.random.SeedGenerator(seed)
        self.tracked_params = self._create_variables(params, trainable=True)
        self.tracked_state = self._create_variables(state, trainable=False)
        if self.params is not None or self.state is not None:
            self.built = True

        self.call_fn_arguments = self._validate_signature(
            call_fn,
            "call_fn",
            {"params", "state", "rng", "inputs", "training"},
            {"inputs"},
        )
        self.has_state = "state" in self.call_fn_arguments

        if init_fn:
            self.init_fn_arguments = self._validate_signature(
                init_fn, "init_fn", {"rng", "inputs", "training"}, {"inputs"}
            )

    def _validate_signature(self, fn, fn_name, allowed, required):
        fn_parameters = inspect.signature(fn).parameters
        for parameter_name in required:
            if parameter_name not in fn_parameters:
                raise ValueError(
                    f"Missing required argument in `{fn_name}`: "
                    f"`{parameter_name}`"
                )

        parameter_names = []
        for parameter in fn_parameters.values():
            if parameter.name not in allowed:
                raise ValueError(
                    f"Unsupported argument in `{fn_name}`: `{parameter.name}`, "
                    f"supported arguments are `{'`, `'.join(allowed)}`"
                )
            parameter_names.append(parameter.name)

        return parameter_names

    @tracking.no_automatic_dependency_tracking
    def _create_variables(self, values, trainable):
        """Create a structure of variables from a structure of JAX arrays.

        `values` is traversed via JAX's `tree_map`. When a leaf is a JAX array
        or a tensor-like object, a corresponding variable is created with it as
        the initial value. The resulting structure of variables is assigned to
        `self.params` or `self.state` depending on `trainable`. Then, a
        flattened version of the variables is returned for tracking.
        `self.params` or `self.state` are intentionally not tracked because
        structures like `TrackedList` interfere with `jax.tree_utils`.
        Note that leaf objects that are not JAX arrays and not tensor-like are
        left intact as they are assumed to be configuration used by the model.

        Args:
            values: the structure of values to traverse.
            trainable: whether to create trainable variables.

        Returns:
            flat list of variables initialized with `values` for tracking.
        """

        def create_variable(value):
            if backend.is_tensor(value) or isinstance(value, np.ndarray):
                variable = self.add_weight(
                    value.shape, initializer="zeros", trainable=trainable
                )
                variable.assign(value)
                return variable
            elif isinstance(value, (np.generic, int, float)):
                variable = self.add_weight(
                    (), initializer="zeros", trainable=trainable
                )
                variable.assign(value)
                return variable
            else:
                return value

        # Use JAX's tree_map as it understands registered classes.
        variables = jax.tree_util.tree_map(create_variable, values)

        if trainable:
            self.params = variables
        else:
            self.state = variables

        flat_variables, _ = jax.tree_util.tree_flatten(variables)
        return flat_variables

    def _get_init_rng(self):
        """
        Returns a JAX `PRNGKey` or structure of `PRNGKey`s to pass to `init_fn`.

        By default, this returns a single `PRNGKey` retrieved by calling
        `self.seed_generator.next()`. Override this to return a different
        structure.

        Returns:
            a JAX `PRNGKey` or structure of `PRNGKey`s that will be passed as
            the `rng` argument of `init_fn`.
        """
        return self.seed_generator.next()

    def _get_call_rng(self, training):
        """
        Returns a JAX `PRNGKey` or structure of `PRNGKey`s to pass to `call_fn`.

        By default, this returns a single `PRNGKey` retrieved by calling
        `self.seed_generator.next()` when `training` is `True`, and `None` when
        `training` is `False`. Override this to return a different structure or
        to pass RNGs in inference mode too.

        Returns:
            a JAX `PRNGKey` or structure of `PRNGKey`s that will be passed as
            the `rng` argument of `call_fn`.
        """
        if training:
            return self.seed_generator.next()
        else:
            return None

    def build(self, input_shape):
        if self.params is not None or self.state is not None:
            return

        if jax_utils.is_in_jax_tracing_scope():
            # This exception is not actually shown, it is caught and a detailed
            # warning about calling 'build' is printed.
            raise ValueError("'JaxLayer' cannot be built in tracing scope")

        # Initialize `params` and `state` if needed by calling `init_fn`.
        def create_input(shape):
            shape = [d if d is not None else 1 for d in shape]
            return jax.numpy.ones(shape)

        init_inputs = tree.map_shape_structure(create_input, input_shape)
        init_args = []
        for argument_name in self.init_fn_arguments:
            if argument_name == "rng":
                init_args.append(self._get_init_rng())
            elif argument_name == "inputs":
                init_args.append(init_inputs)
            elif argument_name == "training":
                init_args.append(True)

        init_result = self.init_fn(*init_args)
        if self.has_state:
            init_params, init_state = init_result
        else:
            init_params, init_state = init_result, None

        self.tracked_params = self._create_variables(
            init_params, trainable=True
        )
        self.tracked_state = self._create_variables(init_state, trainable=False)
        self.built = True

    def call(self, inputs, training=False):
        def unwrap_variable(variable):
            return None if variable is None else variable.value

        call_args = []
        for argument_name in self.call_fn_arguments:
            if argument_name == "params":
                call_args.append(
                    jax.tree_util.tree_map(unwrap_variable, self.params)
                )
            elif argument_name == "state":
                call_args.append(
                    jax.tree_util.tree_map(unwrap_variable, self.state)
                )
            elif argument_name == "rng":
                call_args.append(self._get_call_rng(training))
            elif argument_name == "inputs":
                call_args.append(inputs)
            elif argument_name == "training":
                call_args.append(training)

        def assign_state_to_variable(value, variable):
            # This exists only to make debugging this error case easier.
            if not hasattr(variable, "assign"):
                raise ValueError(
                    "Structure mismatch: the structure of the state returned "
                    "by `call` does not match the structure of the state at "
                    "initialization time."
                )
            variable.assign(value)

        if self.has_state:
            predictions, new_state = self.call_fn(*call_args)
            jax.tree_util.tree_map(
                assign_state_to_variable, new_state, self.state
            )
            return predictions
        else:
            return self.call_fn(*call_args)

    def get_config(self):
        config = {
            "call_fn": serialization_lib.serialize_keras_object(self.call_fn),
            "init_fn": serialization_lib.serialize_keras_object(self.init_fn),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        call_fn = serialization_lib.deserialize_keras_object(config["call_fn"])
        init_fn = serialization_lib.deserialize_keras_object(config["init_fn"])
        config["call_fn"] = call_fn
        config["init_fn"] = init_fn
        return super().from_config(config)


@keras_export("keras.layers.FlaxLayer")
class FlaxLayer(JaxLayer):
    """Keras Layer that wraps a [Flax](https://flax.readthedocs.io) module.

    This layer enables the use of Flax components in the form of
    [`flax.linen.Module`](
        https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html)
    instances within Keras when using JAX as the backend for Keras.

    The module method to use for the forward pass can be specified via the
    `method` argument and is `__call__` by default. This method must take the
    following arguments with these exact names:

    - `self` if the method is bound to the module, which is the case for the
        default of `__call__`, and `module` otherwise to pass the module.
    - `inputs`: the inputs to the model, a JAX array or a `PyTree` of arrays.
    - `training` *(optional)*: an argument specifying if we're in training mode
        or inference mode, `True` is passed in training mode.

    `FlaxLayer` handles the non-trainable state of your model and required RNGs
    automatically. Note that the `mutable` parameter of
    [`flax.linen.Module.apply()`](
        https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.apply)
    is set to `DenyList(["params"])`, therefore making the assumption that all
    the variables outside of the "params" collection are non-trainable weights.

    This example shows how to create a `FlaxLayer` from a Flax `Module` with
    the default `__call__` method and no training argument:

    ```python
    class MyFlaxModule(flax.linen.Module):
        @flax.linen.compact
        def __call__(self, inputs):
            x = inputs
            x = flax.linen.Conv(features=32, kernel_size=(3, 3))(x)
            x = flax.linen.relu(x)
            x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
            x = x.reshape((x.shape[0], -1))  # flatten
            x = flax.linen.Dense(features=200)(x)
            x = flax.linen.relu(x)
            x = flax.linen.Dense(features=10)(x)
            x = flax.linen.softmax(x)
            return x

    flax_module = MyFlaxModule()
    keras_layer = FlaxLayer(flax_module)
    ```

    This example shows how to wrap the module method to conform to the required
    signature. This allows having multiple input arguments and a training
    argument that has a different name and values. This additionally shows how
    to use a function that is not bound to the module.

    ```python
    class MyFlaxModule(flax.linen.Module):
        @flax.linen.compact
        def forward(self, input1, input2, deterministic):
            ...
            return outputs

    def my_flax_module_wrapper(module, inputs, training):
        input1, input2 = inputs
        return module.forward(input1, input2, not training)

    flax_module = MyFlaxModule()
    keras_layer = FlaxLayer(
        module=flax_module,
        method=my_flax_module_wrapper,
    )
    ```

    Args:
        module: An instance of `flax.linen.Module` or subclass.
        method: The method to call the model. This is generally a method in the
            `Module`. If not provided, the `__call__` method is used. `method`
            can also be a function not defined in the `Module`, in which case it
            must take the `Module` as the first argument. It is used for both
            `Module.init` and `Module.apply`. Details are documented in the
            `method` argument of [`flax.linen.Module.apply()`](
              https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.apply).
        variables: A `dict` containing all the variables of the module in the
            same format as what is returned by [`flax.linen.Module.init()`](
              https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.init).
            It should contain a "params" key and, if applicable, other keys for
            collections of variables for non-trainable state. This allows
            passing trained parameters and learned non-trainable state or
            controlling the initialization. If `None` is passed, the module's
            `init` function is called at build time to initialize the variables
            of the model.
    """

    def __init__(
        self,
        module,
        method=None,
        variables=None,
        **kwargs,
    ):
        # Late import to only require Flax when this is used.
        from flax.core import scope as flax_scope

        if backend.backend() != "jax":
            raise ValueError(
                "FlaxLayer is only supported with the JAX backend. Current "
                f"backend: {backend.backend()}"
            )

        self.module = module
        self.method = method

        apply_mutable = flax_scope.DenyList(["params"])

        def apply_with_training(params, state, rng, inputs, training):
            return self.module.apply(
                self._params_and_state_to_variables(params, state),
                inputs,
                rngs=rng,
                method=self.method,
                mutable=apply_mutable,
                training=training,
            )

        def apply_without_training(params, state, rng, inputs):
            return self.module.apply(
                self._params_and_state_to_variables(params, state),
                inputs,
                rngs=rng,
                method=self.method,
                mutable=apply_mutable,
            )

        def init_with_training(rng, inputs, training):
            return self._variables_to_params_and_state(
                self.module.init(
                    rng,
                    inputs,
                    method=self.method,
                    training=training,
                )
            )

        def init_without_training(rng, inputs):
            return self._variables_to_params_and_state(
                self.module.init(
                    rng,
                    inputs,
                    method=self.method,
                )
            )

        if (
            "training"
            in inspect.signature(method or module.__call__).parameters
        ):
            call_fn, init_fn = apply_with_training, init_with_training
        else:
            call_fn, init_fn = apply_without_training, init_without_training

        params, state = self._variables_to_params_and_state(variables)

        super().__init__(
            call_fn=call_fn,
            init_fn=init_fn,
            params=params,
            state=state,
            **kwargs,
        )

    def _params_and_state_to_variables(self, params, state):
        if params:
            if state:
                return {**params, **state}
            else:
                return params
        elif state:
            return state
        return {}

    def _variables_to_params_and_state(self, variables):
        # neither params nor state
        if variables is None:
            return None, None
        # state only
        if "params" not in variables:
            return {}, variables
        # params only
        if len(variables) == 1:
            return variables, {}
        # both, we need to split
        params = {"params": variables["params"]}
        state = {k: v for k, v in variables.items() if k != "params"}
        return params, state

    def _get_init_rng(self):
        return {
            "params": self.seed_generator.next(),
            "dropout": self.seed_generator.next(),
        }

    def _get_call_rng(self, training):
        if training:
            return {"dropout": self.seed_generator.next()}
        else:
            return {}

    def get_config(self):
        config_method = self.method
        if (
            hasattr(self.method, "__self__")
            and self.method.__self__ == self.module
        ):
            # A method bound to the module is serialized by name.
            config_method = self.method.__name__
        config = {
            "module": serialization_lib.serialize_keras_object(self.module),
            "method": serialization_lib.serialize_keras_object(config_method),
        }
        base_config = super().get_config()
        # call_fn and init_fn come from module, do not save them.
        base_config.pop("call_fn")
        base_config.pop("init_fn")
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        module = serialization_lib.deserialize_keras_object(config["module"])
        method = serialization_lib.deserialize_keras_object(config["method"])
        if isinstance(config["method"], str):
            # Deserialize bound method from the module.
            method = getattr(module, method)
        config["module"] = module
        config["method"] = method
        return cls(**config)
