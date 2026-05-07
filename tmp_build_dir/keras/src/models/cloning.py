from keras.src import backend
from keras.src import tree
from keras.src import utils
from keras.src.api_export import keras_export
from keras.src.layers import Input
from keras.src.layers import InputLayer
from keras.src.models.functional import Functional
from keras.src.models.functional import functional_like_constructor
from keras.src.models.sequential import Sequential
from keras.src.saving import serialization_lib


@keras_export("keras.models.clone_model")
def clone_model(
    model,
    input_tensors=None,
    clone_function=None,
    call_function=None,
    recursive=False,
    **kwargs,
):
    """Clone a Functional or Sequential `Model` instance.

    Model cloning is similar to calling a model on new inputs,
    except that it creates new layers (and thus new weights) instead
    of sharing the weights of the existing layers.

    Note that
    `clone_model` will not preserve the uniqueness of shared objects within the
    model (e.g. a single variable attached to two distinct layers will be
    restored as two separate variables).

    Args:
        model: Instance of `Model`
            (could be a Functional model or a Sequential model).
        input_tensors: optional list of input tensors or InputLayer objects
            to build the model upon. If not provided,
            new `Input` objects will be created.
        clone_function: Callable with signature `fn(layer)`
            to be used to clone each layer in the target
            model (except `Input` instances). It takes as argument the
            layer instance to be cloned, and returns the corresponding layer
            instance to be used in the model copy. If unspecified, this callable
            defaults to the following serialization/deserialization function:
            `lambda layer: layer.__class__.from_config(layer.get_config())`.
            By passing a custom callable, you can customize your copy of the
            model, e.g. by wrapping certain layers of interest (you might want
            to replace all `LSTM` instances with equivalent
            `Bidirectional(LSTM(...))` instances, for example).
            Defaults to `None`.
        call_function: Callable with signature
            `fn(layer, *args, **kwargs)` to be used to call each
            cloned layer and a set of inputs. It takes the layer instance,
            the call arguments and keyword arguments, and returns the
            call outputs. If unspecified, this callable defaults to
            the regular `__call__()` method:
            `def fn(layer, *args, **kwargs): return layer(*args, **kwargs)`.
            By passing a custom callable, you can insert new layers before or
            after a given layer. Note: this argument can only be used with
            Functional models.
        recursive: Boolean. Whether to recursively clone any Sequential
            or Functional models encountered in the original
            Sequential/Functional model. If `False`,
            then inner models are cloned by calling `clone_function()`.
            If `True`, then inner models are cloned by calling `clone_model()`
            with the same `clone_function`, `call_function`, and `recursive`
            arguments. Note that in this case, `call_function`
            will not be propagated to any Sequential model
            (since it is not applicable to Sequential models).

    Returns:
        An instance of `Model` reproducing the behavior
        of the original model, on top of new inputs tensors,
        using newly instantiated weights. The cloned model may behave
        differently from the original model if a custom `clone_function`
        or `call_function` modifies a layer or layer call.

    Example:

    ```python
    # Create a test Sequential model.
    model = keras.Sequential([
        keras.layers.Input(shape=(728,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid'),
    ])
    # Create a copy of the test model (with freshly initialized weights).
    new_model = clone_model(model)
    ```

    Using a `clone_function` to make a model deterministic by setting the
    random seed everywhere:

    ```python
    def clone_function(layer):
        config = layer.get_config()
        if "seed" in config:
            config["seed"] = 1337
        return layer.__class__.from_config(config)

    new_model = clone_model(model, clone_function=clone_function)
    ```

    Using a `call_function` to add a `Dropout` layer after each `Dense` layer
    (without recreating new layers):

    ```python
    def call_function(layer, *args, **kwargs):
        out = layer(*args, **kwargs)
        if isinstance(layer, keras.layers.Dense):
            out = keras.layers.Dropout(0.5)(out)
        return out

    new_model = clone_model(
        model,
        clone_function=lambda x: x,  # Reuse the same layers.
        call_function=call_function,
    )
    ```

    Note that subclassed models cannot be cloned by default,
    since their internal layer structure is not known.
    To achieve equivalent functionality
    as `clone_model` in the case of a subclassed model, simply make sure
    that the model class implements `get_config()`
    (and optionally `from_config()`), and call:

    ```python
    new_model = model.__class__.from_config(model.get_config())
    ```

    In the case of a subclassed model, you cannot using a custom
    `clone_function`.
    """
    cache = kwargs.pop("cache", None)
    if kwargs:
        raise ValueError(
            f"Unexpected keyword argument(s): {tuple(kwargs.keys())}"
        )

    if isinstance(model, Sequential):
        # Wrap clone_function to handle recursiveness and layer sharing.
        clone_function = _wrap_clone_function(
            clone_function,
            call_function=call_function,
            recursive=recursive,
            cache=cache,
        )
        if call_function is not None:
            raise ValueError(
                "`call_function` argument is not supported with Sequential "
                "models.  In a Sequential model, layers aren't called "
                "at model-construction time (they're merely listed). "
                "Use `call_function` with Functional models only. "
                "Received model of "
                f"type '{model.__class__.__name__}', with "
                f"call_function={clone_function}"
            )
        return _clone_sequential_model(
            model,
            clone_function=clone_function,
            input_tensors=input_tensors,
        )
    if isinstance(model, Functional):
        # Wrap clone_function to handle recursiveness and layer sharing.
        clone_function = _wrap_clone_function(
            clone_function,
            call_function=call_function,
            recursive=recursive,
            cache=cache,
        )

        # If the get_config() method is the same as a regular Functional
        # model, we're safe to use _clone_functional_model (which relies
        # on a Functional constructor). In the case where the get_config
        # is custom, this may not necessarily work, but if clone_function
        # or input_tensors are passed, we attempt it anyway
        # in order to preserve backwards compatibility.
        if utils.is_default(model.get_config) or (
            clone_function or input_tensors
        ):
            return _clone_functional_model(
                model,
                clone_function=clone_function,
                call_function=call_function,
                input_tensors=input_tensors,
            )

    # Case of a custom model class
    if clone_function or input_tensors:
        raise ValueError(
            "Arguments `clone_function` and `input_tensors` "
            "are only supported for Sequential models "
            "or Functional models. Received model of "
            f"type '{model.__class__.__name__}', with "
            f"clone_function={clone_function} and "
            f"input_tensors={input_tensors}"
        )
    if call_function is not None:
        raise ValueError(
            "Argument `call_function` is only supported "
            "for Functional models. Received model of "
            f"type '{model.__class__.__name__}', with "
            f"call_function={clone_function}"
        )
    config = serialization_lib.serialize_keras_object(model)
    return serialization_lib.deserialize_keras_object(
        config, custom_objects={model.__class__.__name__: model.__class__}
    )


def _wrap_clone_function(
    clone_function, call_function=None, recursive=False, cache=None
):
    """Wrapper to handle recursiveness and layer sharing."""
    if clone_function is None:

        def _clone_layer(layer):
            return layer.__class__.from_config(layer.get_config())

        clone_function = _clone_layer

    if cache is None:
        cache = {}

    def wrapped_clone_function(layer):
        if id(layer) in cache:
            return cache[id(layer)]
        if recursive:
            if isinstance(layer, Sequential):
                # Note: Sequential doesn't support call_function.
                clone = clone_model(
                    layer,
                    clone_function=clone_function,
                    cache=cache,
                )
                cache[id(layer)] = clone
                return clone
            elif isinstance(layer, Functional):
                clone = clone_model(
                    layer,
                    clone_function=clone_function,
                    call_function=call_function,
                    cache=cache,
                )
                cache[id(layer)] = clone
                return clone
        clone = clone_function(layer)
        cache[id(layer)] = clone
        return clone

    return wrapped_clone_function


def _clone_sequential_model(model, clone_function, input_tensors=None):
    """Clone a `Sequential` model instance.

    Model cloning is similar to calling a model on new inputs,
    except that it creates new layers (and thus new weights) instead
    of sharing the weights of the existing layers.

    Args:
        model: Instance of `Sequential`.
        input_tensors: optional list of input tensors
            to build the model upon. If not provided,
            placeholders will be created.
        clone_function: callable to be applied on non-input layers in the model.
            By default, it clones the layer (without copying the weights).

    Returns:
        An instance of `Sequential` reproducing the behavior
        of the original model, on top of new inputs tensors,
        using newly instantiated weights.
    """

    if not isinstance(model, Sequential):
        raise ValueError(
            "Expected `model` argument "
            "to be a `Sequential` model instance. "
            f"Received: model={model}"
        )

    if not callable(clone_function):
        raise ValueError(
            "Expected `clone_function` argument to be a callable. "
            f"Received: clone_function={clone_function}"
        )

    new_layers = [clone_function(layer) for layer in model.layers]

    if isinstance(model._layers[0], InputLayer):
        ref_input_layer = model._layers[0]
        input_name = ref_input_layer.name
        input_batch_shape = ref_input_layer.batch_shape
        input_dtype = ref_input_layer._dtype
    else:
        input_name = None
        input_dtype = None
        input_batch_shape = None

    if input_tensors is not None:
        if isinstance(input_tensors, (list, tuple)):
            if len(input_tensors) != 1:
                raise ValueError(
                    "Argument `input_tensors` must contain a single tensor."
                )
            input_tensors = input_tensors[0]
        if not isinstance(input_tensors, backend.KerasTensor):
            raise ValueError(
                "Argument `input_tensors` must be a KerasTensor. "
                f"Received invalid value: input_tensors={input_tensors}"
            )
        inputs = Input(
            tensor=input_tensors,
            name=input_name,
        )
        new_layers = [inputs] + new_layers
    else:
        if input_batch_shape is not None:
            inputs = Input(
                batch_shape=input_batch_shape,
                dtype=input_dtype,
                name=input_name,
            )
            new_layers = [inputs] + new_layers
    cloned_model = Sequential(
        new_layers, name=model.name, trainable=model.trainable
    )

    # If model compiled already then set same to cloned model
    if model.compiled:
        compiled_config = model.get_compile_config()
        cloned_model.compile_from_config(compiled_config)
    return cloned_model


def _clone_functional_model(
    model, clone_function, input_tensors=None, call_function=None
):
    """Clone a `Functional` model instance.

    Model cloning is similar to calling a model on new inputs,
    except that it creates new layers (and thus new weights) instead
    of sharing the weights of the existing layers.

    Input layers are always cloned.

    Args:
        model: Instance of `Functional`.
        input_tensors: optional list of input tensors
            to build the model upon. If not provided,
            placeholders will be created.
        clone_function: callable to be applied on non-input layers in the model.
            By default, it clones the layer (without copying the weights).

    Returns:
        An instance of `Functional` reproducing the behavior
        of the original model, on top of new inputs tensors,
        using newly instantiated weights.
    """

    if not callable(clone_function):
        raise ValueError(
            "Expected `clone_function` argument to be a callable. "
            f"Received: clone_function={clone_function}"
        )

    if not isinstance(model, Functional):
        raise ValueError(
            "Expected `model` argument "
            f"to be a Functional Model instance. Received: model={model}"
        )

    if input_tensors is not None:
        if not all(
            isinstance(x, backend.KerasTensor)
            for x in tree.flatten(input_tensors)
        ):
            raise ValueError(
                "All entries in `input_tensors` must be KerasTensors. "
                f"Received invalid values: inputs_tensors={input_tensors}"
            )
        try:
            tree.assert_same_structure(input_tensors, model.input)
        except ValueError as e:
            raise ValueError(
                "`input_tensors` must have the same structure as model.input"
                f"\nReference structure: {model.input}"
                f"\nReceived structure: {input_tensors}"
            ) from e
    else:
        input_tensors = tree.map_structure(
            lambda x: Input(batch_shape=x.shape, dtype=x.dtype, name=x.name),
            model.input,
        )

    def operation_fn(layer):
        new_layer = clone_function(layer)
        return new_layer

    output_tensors = model._run_through_graph(
        input_tensors,
        operation_fn=operation_fn,
        call_fn=call_function,
    )

    if functional_like_constructor(model.__class__):
        new_model = model.__class__(
            input_tensors, output_tensors, name=model.name
        )
    else:
        # This may be incorrect: the new model will end up having a different
        # class than the original. However various existing models rely
        # on this behavior, so we keep it.
        new_model = Functional(input_tensors, output_tensors, name=model.name)
    if model.compiled:
        compiled_config = model.get_compile_config()
        new_model.compile_from_config(compiled_config)
    return new_model
