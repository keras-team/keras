"""Layer is an Operation with state.

Takes care of:

- Weights / variables (and tracking thereof)
- deferred build
- trainable argument value inference
- masking
- autocasting

And some more magic:

- add_loss
- metric tracking
- RNG seed tracking
- activity regularization
"""
import collections
import inspect
import threading
import warnings

import numpy as np
from tensorflow import keras as tf_keras
from tensorflow import nest

from keras_core import backend
from keras_core import initializers
from keras_core import regularizers
from keras_core import utils
from keras_core.api_export import keras_core_export
from keras_core.backend import KerasTensor
from keras_core.layers import input_spec
from keras_core.metrics.metric import Metric
from keras_core.operations.operation import Operation
from keras_core.utils import summary_utils
from keras_core.utils.tracking import Tracker

# TODO: cache all call signature processing. See layer_utils.CallFunctionSpec() in Keras.


@keras_core_export(["keras_core.Layer", "keras_core.layers.Layer"])
class Layer(Operation):
    def __init__(
        self, activity_regularizer=None, trainable=True, dtype=None, name=None
    ):
        super().__init__(name=name)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self._trainable = trainable
        if dtype is None:
            dtype = backend.floatx()

        self.built = False
        self.dtype_policy = tf_keras.mixed_precision.Policy(dtype)
        self.input_spec = None

        self._layers = []
        self._metrics = []
        self._seed_generators = []
        self._losses = []
        self._variables = []
        self._supports_masking = not utils.is_default(self.compute_mask)
        self._build_shapes_dict = None
        self._call_signature_parameters = [
            p.name for p in inspect.signature(self.call).parameters.values()
        ]

        self._tracker = Tracker(
            {
                "variables": (
                    lambda x: isinstance(x, backend.Variable),
                    self._variables,
                ),
                "metrics": (lambda x: isinstance(x, Metric), self._metrics),
                "layers": (
                    lambda x: isinstance(x, Layer)
                    and not isinstance(x, Metric),
                    self._layers,
                ),
                "seed_generators": (
                    lambda x: isinstance(x, backend.random.SeedGenerator),
                    self._seed_generators,
                ),
            }
        )

    @utils.default
    def build(self, input_shape):
        self.built = True

    def get_build_config(self):
        """Returns a dictionary with the layer's input shape.

        This method returns a config dict that can be used by
        `build_from_config(config)` to create all states (e.g. Variables and
        Lookup tables) needed by the layer.

        By default, the config only contains the input shape that the layer
        was built with. If you're writing a custom layer that creates state in
        an unusual way, you should override this method to make sure this state
        is already created when Keras attempts to load its value upon model
        loading.

        Returns:
            A dict containing the input shape associated with the layer.
        """
        if self._build_shapes_dict is not None:
            if len(self._build_shapes_dict) == 1:
                return {
                    "input_shape": tuple(self._build_shapes_dict.values())[0],
                }
            else:
                return {"shapes_dict": self._build_shapes_dict}

    def build_from_config(self, config):
        """Builds the layer's states with the supplied config dict.

        By default, this method calls the `build(config["input_shape"])` method,
        which creates weights based on the layer's input shape in the supplied
        config. If your config contains other information needed to load the
        layer's state, you should override this method.

        Args:
            config: Dict containing the input shape associated with this layer.
        """
        if config:
            if "input_shape" in config:
                self.build(config["input_shape"])
                self._build_shapes_dict = config
            elif "shapes_dict" in config:
                self.build(**config["shapes_dict"])
                self._build_shapes_dict = config["shapes_dict"]
            self.built = True

    def add_variable(
        self,
        shape,
        initializer,
        dtype=None,
        trainable=True,
        regularizer=None,
        constraint=None,
        name=None,
    ):
        # TODO: handle constraint (in the optimizer)
        # TODO: handle layout
        self._check_super_called()
        initializer = initializers.get(initializer)
        variable = backend.Variable(
            initializer=initializer,
            shape=shape,
            dtype=dtype,
            trainable=trainable,
            name=name,
        )
        # Will be added to layer.losses
        variable.regularizer = regularizer
        variable.constraint = constraint
        self._variables.append(variable)
        # Prevent double-tracking
        self._tracker.stored_ids["variables"].add(id(variable))
        return variable

    def add_weight(self, *args, **kwargs):
        return self.add_variable(*args, **kwargs)

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        """Sets trainable attribute for the layer and its sublayers.

        When this value is changed during training (e.g. with a
        `Callback`) you need to call the parent
        `Model.make_train_function` with `force=True` in order to
        recompile the training graph.

        Args:
            value: Boolean with the desired state for the layer's trainable
                attribute.
        """
        for layer in self._layers():
            layer._trainable = value

    @property
    def variables(self):
        # Includes weights, seed generator state, and metric variables.
        variables = self.weights[:]
        for m in self._metrics:
            variables.extend(m.variables)
        for sg in self._seed_generators:
            variables.append(sg.state)
        return variables

    @property
    def trainable_variables(self):
        return [v for v in self.variables if v.trainable]

    @property
    def non_trainable_variables(self):
        return [v for v in self.variables if not v.trainable]

    @property
    def weights(self):
        # Return only "own weights" of all Layers, recursively
        weights = self._variables[:]
        for layer in self._layers:
            weights.extend(layer._variables)
        return weights

    @property
    def trainable_weights(self):
        return [v for v in self.weights if v.trainable]

    @property
    def non_trainable_weights(self):
        return [v for v in self.weights if not v.trainable]

    def get_weights(self):
        return [v.numpy() for v in self.weights]

    def set_weights(self, weights):
        layer_weights = self.weights
        if len(layer_weights) != len(weights):
            raise ValueError(
                f"You called `set_weights(weights)` on layer '{self.name}' "
                f"with a weight list of length {len(weights)}, but the layer was "
                f"expecting {len(layer_weights)} weights."
            )
        for variable, value in zip(layer_weights, weights):
            if variable.shape != value.shape:
                raise ValueError(
                    f"Layer {self.name} weight shape {variable.shape} "
                    "is not compatible with provided weight "
                    f"shape {value.shape}."
                )
            variable.assign(value)

    @property
    def dtype(self):
        """The dtype of the state (weights) of the layer."""
        return self.variable_dtype

    @property
    def compute_dtype(self):
        """The dtype of the computations performed by the layer."""
        return self.dtype_policy.compute_dtype

    @property
    def variable_dtype(self):
        """The dtype of the state (weights) of the layer."""
        return self.dtype_policy.compute_dtype

    @property
    def supports_masking(self):
        """Whether this layer supports computing a mask using `compute_mask`."""
        return self._supports_masking

    @supports_masking.setter
    def supports_masking(self, value):
        self._supports_masking = value

    @utils.default
    def compute_mask(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        self._check_super_called()

        ######################################
        # Argument validation and conversion. #
        # 1. Convert any array arguments to tensors of correct dtype.
        def maybe_convert(x):
            if isinstance(x, np.ndarray) or backend.is_tensor(x):
                return backend.convert_to_tensor(x, dtype=self.compute_dtype)
            # TODO: cast KerasTensor too
            return x

        args = nest.map_structure(maybe_convert, args)
        kwargs = nest.map_structure(maybe_convert, kwargs)

        # 3. Enforce that only tensors can be passed positionally.
        for arg in nest.flatten(args):
            if not isinstance(arg, KerasTensor) and not backend.is_tensor(arg):
                raise ValueError(
                    "Only input tensors may be passed as "
                    "positional arguments. The following argument value "
                    f"should be passed as a keyword argument: {arg} "
                    f"(of type {type(arg)})"
                )

        # 4. Check input spec for 1st positional arg.
        # TODO: consider extending this to all args and kwargs.
        self._assert_input_compatibility(*args)
        ######################################

        ###############
        # Call build. #
        self._maybe_build(*args, **kwargs)
        ###############

        # Maintains info about the `Layer.call` stack.
        call_context = self._get_call_context()

        # Infer training value
        # Training phase for `Layer.call` is set via (in order of priority):
        # (1) The `training` argument passed to this `Layer.call`, if it is not None
        # (2) The training argument of an outer `Layer.call`.
        # (4) Any non-None default value for `training` specified in the call signature
        # (5) False (treating the layer as if it's in inference)
        arguments_dict = get_arguments_dict(self.call, *args, **kwargs)
        training = arguments_dict.get("training", None)
        if training is None:
            training = call_context.training
            if training is None:
                training = self._get_default_training_value()
                if training is None:
                    training = False
        call_context.training = training
        if self._call_has_training_arg():
            kwargs["training"] = training

        # TODO: Populate mask argument(s)

        # Call the layer.
        with backend.name_scope(self.name):
            outputs = super().__call__(*args, **kwargs)
            # Record activity regularizer loss.
            if self.activity_regularizer is not None:
                self.add_loss(self.activity_regularizer(outputs))

        # TODO: Set masks on outputs
        # self._set_mask_metadata(inputs, outputs, previous_mask)

        # Destroy call context if we created it
        self._maybe_reset_call_context()
        return outputs

    def call(self, *args, **kwargs):
        raise NotImplementedError

    def stateless_call(
        self, trainable_variables, non_trainable_variables, *args, **kwargs
    ):
        # TODO: also handle losses

        self._check_super_called()

        if not self.built:
            raise ValueError(
                "To call stateless_call, {self.__class__.__name__} must be built "
                "(i.e. its variables must have been already created). "
                "You can build it by calling it on some data."
            )
        if len(trainable_variables) != len(self.trainable_variables):
            raise ValueError(
                "Argument `trainable_variables` must be a list of tensors "
                f"corresponding 1:1 to {self.__class__.__name__}().trainable_variables. "
                f"Received list with length {len(trainable_variables)}, but expected "
                f"{len(self.trainable_variables)} variables."
            )
        if len(non_trainable_variables) != len(self.non_trainable_variables):
            raise ValueError(
                "Argument `non_trainable_variables` must be a list of tensors "
                f"corresponding 1:1 to {self.__class__.__name__}().non_trainable_variables. "
                f"Received list with length {len(non_trainable_variables)}, but expected "
                f"{len(self.non_trainable_variables)} variables."
            )

        # Gather variable mapping
        trainable_mapping = zip(self.trainable_variables, trainable_variables)
        non_trainable_mapping = zip(
            self.non_trainable_variables, non_trainable_variables
        )
        mapping = list(trainable_mapping) + list(non_trainable_mapping)

        # Call in stateless scope
        with backend.StatelessScope(state_mapping=mapping) as scope:
            outputs = self.call(*args, **kwargs)

        # Gather updated non-trainable variables
        non_trainable_variables = []
        for v in self.non_trainable_variables:
            new_v = scope.get_current_value(v)
            if new_v is not None:
                non_trainable_variables.append(new_v)
            else:
                non_trainable_variables.append(v)
        return outputs, non_trainable_variables

    def compute_output_spec(self, *args, **kwargs):
        if utils.is_default(self.compute_output_shape):
            return super().compute_output_spec(*args, **kwargs)
        else:
            # Use compute_output_shape() to return the right output spec
            arguments_dict = get_arguments_dict(self.call, *args, **kwargs)
            shapes_dict = get_shapes_dict(arguments_dict)
            if len(shapes_dict) == 1:
                # Single arg: pass it positionally
                input_shape = tuple(shapes_dict.values())[0]
                output_shape = self.compute_output_shape(input_shape)
            else:
                # More than one shape: pass them by name.
                output_shape = self.compute_output_shape(**shapes_dict)
            if (
                isinstance(output_shape, tuple)
                and output_shape
                and isinstance(output_shape[0], (int, type(None)))
            ):
                return KerasTensor(output_shape, dtype=self.compute_dtype)
            return nest.map_structure(
                lambda s: KerasTensor(s, dtype=self.compute_dtype), output_shape
            )

    @utils.default
    def compute_output_shape(self, *args, **kwargs):
        return NotImplementedError

    def add_loss(self, loss):
        # Eager only.
        losses = nest.flatten(loss)
        for x in losses:
            if not backend.is_tensor(x):
                raise ValueError(
                    "`add_loss()` can only be called from inside `build()` or `call()`, "
                    f"on a tensor input. Received invalid value: {x}"
                )
        if backend.in_stateless_scope():
            scope = backend.get_stateless_scope()
            if scope.collect_losses:
                for x in losses:
                    scope.add_loss(loss)
        else:
            self._losses.extend(losses)

    @property
    def losses(self):
        losses = self._losses[:]
        for layer in self._layers:
            losses.extend(layer._losses)
        weight_regularization_losses = []
        for v in self.trainable_weights:
            regularizer = getattr(v, "regularizer", None)
            if regularizer:
                weight_regularization_losses.append(regularizer(v))
        losses.extend(weight_regularization_losses)
        return losses

    def save_own_variables(self, store):
        """Saves the state of the layer.

        You can override this method to take full control of how the state of
        the layer is saved upon calling `model.save()`.

        Args:
            store: Dict where the state of the model will be saved.
        """
        all_vars = self._variables
        for i, v in enumerate(all_vars):
            store[f"{i}"] = np.array(v)

    def load_own_variables(self, store):
        """Loads the state of the layer.

        You can override this method to take full control of how the state of
        the layer is loaded upon calling `keras.models.load_model()`.

        Args:
            store: Dict from which the state of the model will be loaded.
        """
        all_vars = self._variables
        if len(store.keys()) != len(all_vars):
            if len(all_vars) == 0 and not self.built:
                raise ValueError(
                    f"Layer '{self.name}' was never built "
                    "and thus it doesn't have any variables. "
                    f"However the weights file lists {len(store.keys())} "
                    "variables for this layer. In most cases, "
                    "this indicates that you need to implement the "
                    "`def build_from_config(self, config)` method "
                    "on the layer. "
                    "You might also want to implement the method "
                    "that generates the config at saving time, "
                    "`def get_build_config(self)`. "
                    "The method `build_from_config()` is meant "
                    "to create the state "
                    "of the layer (i.e. its variables) upon deserialization.",
                )
            raise ValueError(
                f"Layer '{self.name}' expected {len(all_vars)} variables, "
                "but received "
                f"{len(store.keys())} variables during loading. "
                f"Expected: {[v.name for v in all_vars]}"
            )
        for i, v in enumerate(all_vars):
            v.assign(store[f"{i}"])

    def _clear_losses(self):
        if backend.in_stateless_scope():
            scope = backend.get_stateless_scope()
            if scope.collect_losses:
                for x in scope.losses:
                    if x in self._losses:
                        scope.losses.remove(x)
        self._losses = []

    def add_metric(self):
        # Permanently disabled
        raise NotImplementedError

    def count_params(self):
        """Count the total number of scalars composing the weights.

        Returns:
            An integer count.
        """
        if not self.built:
            raise ValueError(
                "You tried to call `count_params` "
                f"on layer '{self.name}'"
                ", but the layer isn't built. "
                "You can build it manually via: "
                f"`layer.build(input_shape)`."
            )
        return summary_utils.count_params(self.weights)

    def _maybe_build(self, *args, **kwargs):
        if not self.built:
            arguments_dict = get_arguments_dict(self.call, *args, **kwargs)
            shapes_dict = get_shapes_dict(arguments_dict)
            self._build_shapes_dict = shapes_dict
            failure = False
            if len(shapes_dict) == 1:
                # Single arg: pass it positionally
                input_shape = tuple(shapes_dict.values())[0]
                with backend.name_scope(self.name):
                    if utils.is_default(self.build) and might_have_unbuilt_state(self):
                        status = self._build_by_run_for_single_pos_arg(input_shape)
                        if not status:
                            failure = True
                    else:
                        self.build(input_shape)
            else:
                # More than one shape: pass them by name,
                # and check that build() expects the right args.
                check_build_signature(self.build, shapes_dict)
                with backend.name_scope(self.name):
                    if utils.is_default(self.build) and might_have_unbuilt_state(self):
                        status = self._build_by_run_for_kwargs(shapes_dict)
                        if not status:
                            failure = True
                    else:
                        self.build(**shapes_dict)
            # if failure: # TODO: warn or raise
            self.built = True

            # Check input spec again (after build, since self.input_spec
            # may have been updated
            self._assert_input_compatibility(*args)

    def _build_by_run_for_single_pos_arg(self, input_shape):
        # Case: all inputs are in the first arg (possibly nested).
        if is_shape_tuple(input_shape):
            input_shape = tuple(input_shape)
        if isinstance(input_shape, list):
            input_tensors = [
                backend.KerasTensor(shape, record_history=False) for shape in input_shape
            ]
        elif isinstance(input_shape, dict):
            input_tensors = {
                k: backend.KerasTensor(shape, record_history=False)
                for k, shape in input_shape.items()
            }
        else:
            input_tensors = backend.KerasTensor(input_shape, record_history=False)
        try:
            self.compute_output_spec(input_tensors)
            return True
        except Exception as e:
            return False

    def _build_by_run_for_kwargs(self, shapes_dict):
        # Case: inputs were recorded as multiple keyword arguments.
        if all(is_shape_tuple(s) for s in shapes_dict.values()):
            # Case: all input keyword arguments were plain tensors.
            input_tensors = {
                k: backend.KerasTensor(v, record_history=False) for k, v in shapes_dict.items()
            }
            try:
                self.compute_output_spec(**input_tensors)
            except:
                return False
        else:
            # Not supported: nested input keyword arguments.
            return False

    def __repr__(self):
        # TODO: improve
        return f"<{self.__class__.__name__} name={self.name}>"

    def __str__(self):
        # TODO: improve
        return f"<{self.__class__.__name__} name={self.name}>"

    def __setattr__(self, name, value):
        # Track Variables, Layers, Metrics
        if hasattr(self, "_tracker"):
            value = self._tracker.track(value)
        return super().__setattr__(name, value)

    def _check_super_called(self):
        if not hasattr(self, "_tracker"):
            raise RuntimeError(
                f"In layer '{self.__class__.__name__}', you forgot to call "
                "`super().__init__()` in the `__init__()` method. "
                "Go add it!"
            )

    def _assert_input_compatibility(self, *args):
        if args and self.input_spec:
            input_spec.assert_input_compatibility(
                self.input_spec, args[0], layer_name=self.name
            )

    def _call_has_training_arg(self):
        return "training" in self._call_signature_parameters

    def _call_has_mask_arg(self):
        return "mask" in self._call_signature_parameters

    def _get_call_context(self):
        """Returns currently active `CallContext`."""
        global CALL_CTX
        call_ctx = getattr(CALL_CTX, "current", None)
        if call_ctx is None:
            # Enter new call context.
            call_ctx = CallContext(entry_layer=self)
            CALL_CTX.current = call_ctx
            self._clear_losses()
        return call_ctx

    def _maybe_reset_call_context(self):
        global CALL_CTX
        call_ctx = getattr(CALL_CTX, "current", None)
        if call_ctx is None and call_ctx.entry_layer == self:
            CALL_CTX.current = None

    def _get_default_training_value(self):
        signature = inspect.signature(self.call)
        kwargs = [
            p.name
            for p in signature.parameters.values()
            if p.default is not inspect.Parameter.empty
        ]
        if not kwargs:
            return None
        values = self.call.__defaults__
        mapping = dict(zip(kwargs, values))
        return mapping.get("training", None)

    def _flatten_layers(self, include_self=True, recursive=True):
        layers = []
        if include_self:
            layers.append(self)
        seen_object_ids = set()
        deque = collections.deque(self._layers)
        while deque:
            layer = deque.popleft()
            if id(layer) in seen_object_ids:
                continue
            seen_object_ids.add(id(layer))
            layers.append(layer)
            # Introspect recursively through sublayers.
            if recursive:
                deque.extendleft(layer._layers)
        return layers

    def _set_mask_metadata(self, inputs, outputs, previous_mask):
        # Many `Layer`s don't need to call `compute_mask`.
        # This method is optimized to do as little work as needed for the common
        # case.
        if not self._supports_masking:
            return

        flat_outputs = nest.flatten(outputs)

        mask_already_computed = all(
            getattr(x, "_keras_mask", None) is not None for x in flat_outputs
        )
        if mask_already_computed:
            return

        output_masks = self.compute_mask(inputs, previous_mask)
        if output_masks is None:
            return

        flat_masks = nest.flatten(output_masks)
        for tensor, mask in zip(flat_outputs, flat_masks):
            tensor._keras_mask = mask


def get_arguments_dict(fn, *args, **kwargs):
    """Return a dict mapping argument names to their values."""
    sig = inspect.signature(fn)
    bound_args = sig.bind(*args, **kwargs)
    arg_dict = {}
    for name, value in bound_args.arguments.items():
        arg_dict[name] = value
    return arg_dict


def get_shapes_dict(arguments_dict):
    """Convert the call() arguments dict into a dict of input shape arguments.

    Example:

    ```
    >>> get_shapes_dict({"input_a": KerasTensor(shape=(2, 3)), "training": False})
    {"input_a_shape": (2, 3)}
    ```
    """
    shapes_dict = {}
    for k, v in arguments_dict.items():
        if isinstance(v, KerasTensor) or backend.is_tensor(v):
            shapes_dict[f"{k}_shape"] = backend.standardize_shape(v.shape)
        elif nest.is_nested(v):
            flat = nest.flatten(v)
            if any(
                isinstance(x, KerasTensor) or backend.is_tensor(x) for x in flat
            ):
                if not all(
                    isinstance(x, KerasTensor) or backend.is_tensor(x)
                    for x in flat
                ):
                    raise ValueError(
                        "You cannot mix tensors and non-tensors in a nested argument. "
                        f"Invalid argument: {k}={v}"
                    )
            shapes_dict[f"{k}_shape"] = nest.map_structure(
                lambda x: backend.standardize_shape(x.shape), v
            )
    return shapes_dict


def check_build_signature(build_fn, shapes_dict):
    """Asserts that the argument names in build_fn match the entries in shapes_dict.

    For instance if call() has the signature `def call(self, a, b)`
    then we'll see `shapes_dict == {"a_shape": (...), "b_shape": (...)}
    and we expect build() to have signature `def build(self, a_shape, b_shape)`.

    When there is a single tensor argument, we pass it positionally and thus
    don't check names (if we did, it would force call() to always take
    `input` as its first argument, which is usually not the case).
    """
    if len(shapes_dict) == 1:
        return
    if utils.is_default(build_fn):
        return
    sig = inspect.signature(build_fn)
    expected_names = []
    for name, param in sig.parameters.items():
        if param.kind in (
            param.POSITIONAL_OR_KEYWORD,
            param.POSITIONAL_ONLY,
            param.KEYWORD_ONLY,
        ):
            expected_names.append(name)
    if set(expected_names) != set(shapes_dict.keys()):
        comma_separated = ", ".join(shapes_dict.keys())
        raise ValueError(
            "For a `call()` method with more than one tensor argument, "
            "the arguments of the `build()` method should match the "
            "tensor arguments of `call()` method. Here we expect the signature "
            f"`build(self, {comma_separated})`."
        )


CALL_CTX = threading.local()


class CallContext:
    def __init__(self, entry_layer):
        self.entry_layer = entry_layer
        self.training = None


def is_shape_tuple(s):
    return isinstance(s, (list, tuple)) and all(
        d is None or isinstance(d, int) for d in s
    )


def might_have_unbuilt_state(layer):
    return any(not lr.built for lr in layer._layers)
