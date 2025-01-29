import inspect
import types

from keras.src import backend
from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.saving import serialization_lib
from keras.src.utils import python_utils


@keras_export("keras.layers.Lambda")
class Lambda(Layer):
    """Wraps arbitrary expressions as a `Layer` object.

    The `Lambda` layer exists so that arbitrary expressions can be used
    as a `Layer` when constructing Sequential
    and Functional API models. `Lambda` layers are best suited for simple
    operations or quick experimentation. For more advanced use cases,
    prefer writing new subclasses of `Layer`.

    WARNING: `Lambda` layers have (de)serialization limitations!

    The main reason to subclass `Layer` instead of using a
    `Lambda` layer is saving and inspecting a model. `Lambda` layers
    are saved by serializing the Python bytecode, which is fundamentally
    non-portable and potentially unsafe.
    They should only be loaded in the same environment where
    they were saved. Subclassed layers can be saved in a more portable way
    by overriding their `get_config()` method. Models that rely on
    subclassed Layers are also often easier to visualize and reason about.

    Example:

    ```python
    # add a x -> x^2 layer
    model.add(Lambda(lambda x: x ** 2))
    ```

    Args:
        function: The function to be evaluated. Takes input tensor as first
            argument.
        output_shape: Expected output shape from function. This argument
            can usually be inferred if not explicitly provided.
            Can be a tuple or function. If a tuple, it only specifies
            the first dimension onward; sample dimension is assumed
            either the same as the input:
            `output_shape = (input_shape[0], ) + output_shape` or,
            the input is `None` and the sample dimension is also `None`:
            `output_shape = (None, ) + output_shape`.
            If a function, it specifies the
            entire shape as a function of the input shape:
            `output_shape = f(input_shape)`.
        mask: Either None (indicating no masking) or a callable with the same
            signature as the `compute_mask` layer method, or a tensor
            that will be returned as output mask regardless
            of what the input is.
        arguments: Optional dictionary of keyword arguments to be passed to the
            function.
    """

    def __init__(
        self, function, output_shape=None, mask=None, arguments=None, **kwargs
    ):
        super().__init__(**kwargs)

        self.arguments = arguments or {}
        self.function = function

        if mask is not None:
            self.supports_masking = True
        else:
            self.supports_masking = False
        self.mask = mask
        self._output_shape = output_shape

        # Warning on every invocation will be quite irksome in Eager mode.
        self._already_warned = False

        function_args = inspect.getfullargspec(function).args
        self._fn_expects_training_arg = "training" in function_args
        self._fn_expects_mask_arg = "mask" in function_args

    def compute_output_shape(self, input_shape):
        if self._output_shape is None:
            # Leverage backend shape inference
            try:
                inputs = tree.map_shape_structure(
                    lambda x: backend.KerasTensor(x, dtype=self.compute_dtype),
                    input_shape,
                )
                output_spec = backend.compute_output_spec(self.call, inputs)
                return tree.map_structure(lambda x: x.shape, output_spec)
            except:
                raise NotImplementedError(
                    "We could not automatically infer the shape of "
                    "the Lambda's output. Please specify the `output_shape` "
                    "argument for this Lambda layer."
                )

        if callable(self._output_shape):
            return self._output_shape(input_shape)

        # Output shapes are passed directly and don't include batch dimension.
        batch_size = tree.flatten(input_shape)[0]

        def _add_batch(shape):
            return (batch_size,) + shape

        return tree.map_shape_structure(_add_batch, self._output_shape)

    def call(self, inputs, mask=None, training=None):
        # We must copy for thread safety,
        # but it only needs to be a shallow copy.
        kwargs = {k: v for k, v in self.arguments.items()}
        if self._fn_expects_mask_arg:
            kwargs["mask"] = mask
        if self._fn_expects_training_arg:
            kwargs["training"] = training
        return self.function(inputs, **kwargs)

    def compute_mask(self, inputs, mask=None):
        if callable(self.mask):
            return self.mask(inputs, mask)
        return self.mask

    def get_config(self):
        config = {
            "function": self._serialize_function_to_config(self.function),
        }
        if self._output_shape is not None:
            if callable(self._output_shape):
                output_shape = self._serialize_function_to_config(
                    self._output_shape
                )
            else:
                output_shape = self._output_shape
            config["output_shape"] = output_shape
        if self.mask is not None:
            if callable(self.mask):
                mask = self._serialize_function_to_config(self.mask)
            else:
                mask = serialization_lib.serialize_keras_object(self.mask)
            config["mask"] = mask
        config["arguments"] = serialization_lib.serialize_keras_object(
            self.arguments
        )
        base_config = super().get_config()
        return {**base_config, **config}

    def _serialize_function_to_config(self, fn):
        if isinstance(fn, types.LambdaType) and fn.__name__ == "<lambda>":
            code, defaults, closure = python_utils.func_dump(fn)
            return {
                "class_name": "__lambda__",
                "config": {
                    "code": code,
                    "defaults": defaults,
                    "closure": closure,
                },
            }
        elif callable(fn):
            return serialization_lib.serialize_keras_object(fn)
        raise ValueError(
            "Invalid input type for serialization. "
            f"Received: {fn} of type {type(fn)}."
        )

    @staticmethod
    def _raise_for_lambda_deserialization(arg_name, safe_mode):
        if safe_mode:
            raise ValueError(
                "The `{arg_name}` of this `Lambda` layer is a Python lambda. "
                "Deserializing it is unsafe. If you trust the source of the "
                "config artifact, you can override this error "
                "by passing `safe_mode=False` "
                "to `from_config()`, or calling "
                "`keras.config.enable_unsafe_deserialization()."
            )

    @classmethod
    def from_config(cls, config, custom_objects=None, safe_mode=None):
        safe_mode = safe_mode or serialization_lib.in_safe_mode()
        fn_config = config["function"]
        if (
            isinstance(fn_config, dict)
            and "class_name" in fn_config
            and fn_config["class_name"] == "__lambda__"
        ):
            cls._raise_for_lambda_deserialization("function", safe_mode)
            inner_config = fn_config["config"]
            fn = python_utils.func_load(
                inner_config["code"],
                defaults=inner_config["defaults"],
                closure=inner_config["closure"],
            )
            config["function"] = fn
        else:
            config["function"] = serialization_lib.deserialize_keras_object(
                fn_config, custom_objects=custom_objects
            )
        if "output_shape" in config:
            fn_config = config["output_shape"]
            if (
                isinstance(fn_config, dict)
                and "class_name" in fn_config
                and fn_config["class_name"] == "__lambda__"
            ):
                cls._raise_for_lambda_deserialization("function", safe_mode)
                inner_config = fn_config["config"]
                fn = python_utils.func_load(
                    inner_config["code"],
                    defaults=inner_config["defaults"],
                    closure=inner_config["closure"],
                )
                config["output_shape"] = fn
            else:
                output_shape = serialization_lib.deserialize_keras_object(
                    fn_config, custom_objects=custom_objects
                )
                if isinstance(output_shape, list) and all(
                    isinstance(e, (int, type(None))) for e in output_shape
                ):
                    output_shape = tuple(output_shape)
                config["output_shape"] = output_shape

        if "arguments" in config:
            config["arguments"] = serialization_lib.deserialize_keras_object(
                config["arguments"], custom_objects=custom_objects
            )
        return cls(**config)
