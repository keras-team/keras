"""Utilities for converting models between different data formats."""

from keras.src.api_export import keras_export
from keras.src.layers import Input
from keras.src.layers import InputLayer
from keras.src.models.cloning import clone_model


def _permute_shape(shape, source_format, target_format):
    """Permute a shape from one data format to another.

    For spatial data (e.g., images), converts between:
    - channels_last: (..., spatial_dims..., channels)
    - channels_first: (..., channels, spatial_dims...)

    Args:
        shape: A tuple representing the shape (excluding batch dimension).
        source_format: The source data format ('channels_first' or
            'channels_last').
        target_format: The target data format ('channels_first' or
            'channels_last').

    Returns:
        The permuted shape tuple.
    """
    if source_format == target_format:
        return shape

    if shape is None or len(shape) < 2:
        return shape

    shape = tuple(shape)

    if source_format == "channels_last" and target_format == "channels_first":
        # Move channels from last to first position
        # (H, W, C) -> (C, H, W) or (D, H, W, C) -> (C, D, H, W)
        return (shape[-1],) + shape[:-1]
    elif source_format == "channels_first" and target_format == "channels_last":
        # Move channels from first to last position
        # (C, H, W) -> (H, W, C) or (C, D, H, W) -> (D, H, W, C)
        return shape[1:] + (shape[0],)
    else:
        raise ValueError(
            f"Invalid data format conversion: {source_format} -> "
            f"{target_format}. Supported formats are 'channels_first' and "
            "'channels_last'."
        )


def _convert_axis(axis, ndim, source_format, target_format):
    """Convert axis parameter from one data format to another.

    Used for layers like BatchNormalization that use an axis parameter
    to specify the channel dimension.

    Args:
        axis: The axis value to convert (can be negative).
        ndim: The number of dimensions of the tensor (including batch).
        source_format: The source data format.
        target_format: The target data format.

    Returns:
        The converted axis value.
    """
    if source_format == target_format:
        return axis

    # Normalize negative axis
    if axis < 0:
        axis = ndim + axis

    # For channels_last, channel axis is typically -1 (or ndim-1)
    # For channels_first, channel axis is typically 1
    if source_format == "channels_last" and target_format == "channels_first":
        if axis == ndim - 1:  # Was channel axis in channels_last
            return 1
        elif axis >= 1:  # Was spatial axis
            return axis + 1
    elif source_format == "channels_first" and target_format == "channels_last":
        if axis == 1:  # Was channel axis in channels_first
            return -1
        elif axis > 1:  # Was spatial axis
            return axis - 1

    return axis


def _create_clone_function(source_format, target_format, input_ndim=None):
    """Create a clone function that converts layer data formats.

    Args:
        source_format: The source data format.
        target_format: The target data format.
        input_ndim: Optional number of dimensions for input tensors.

    Returns:
        A function that can be used with clone_model.
    """

    def clone_function(layer):
        config = layer.get_config()
        layer_class = layer.__class__

        # Handle InputLayer: permute the shape
        if isinstance(layer, InputLayer):
            if "batch_shape" in config and config["batch_shape"] is not None:
                batch_shape = config["batch_shape"]
                # batch_shape includes batch dimension, so we permute [1:]
                new_shape = _permute_shape(
                    batch_shape[1:], source_format, target_format
                )
                config["batch_shape"] = (batch_shape[0],) + new_shape
            elif "shape" in config and config["shape"] is not None:
                config["shape"] = _permute_shape(
                    config["shape"], source_format, target_format
                )
            return layer_class.from_config(config)

        # Handle layers with data_format parameter
        if "data_format" in config:
            config["data_format"] = target_format

        # Handle BatchNormalization and similar layers with axis parameter
        # that depends on data format
        if "axis" in config and "data_format" not in config:
            # This is likely a normalization layer
            # Determine ndim from layer's input spec if possible
            ndim = 4  # Default assumption for 2D conv layers
            if hasattr(layer, "input_spec") and layer.input_spec is not None:
                if hasattr(layer.input_spec, "ndim"):
                    ndim = layer.input_spec.ndim
                elif hasattr(layer.input_spec, "min_ndim"):
                    ndim = layer.input_spec.min_ndim
            elif input_ndim is not None:
                ndim = input_ndim

            config["axis"] = _convert_axis(
                config["axis"], ndim, source_format, target_format
            )

        return layer_class.from_config(config)

    return clone_function


def _get_model_data_format(model):
    """Infer the data format of a model by examining its layers.

    Args:
        model: A Keras model.

    Returns:
        The inferred data format ('channels_first' or 'channels_last'),
        or None if it cannot be determined.
    """
    for layer in model.layers:
        if hasattr(layer, "data_format"):
            return layer.data_format
    return None


def _get_model_input_ndim(model):
    """Get the number of input dimensions for a model.

    Args:
        model: A Keras model.

    Returns:
        The number of dimensions including batch, or None.
    """
    if hasattr(model, "input_shape"):
        input_shape = model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        if input_shape is not None:
            return len(input_shape)
    return None


def _create_permuted_inputs(model, source_format, target_format):
    """Create new Input tensors with permuted shapes.

    Args:
        model: The source model.
        source_format: The source data format.
        target_format: The target data format.

    Returns:
        A single Input tensor or a list of Input tensors with permuted shapes.
    """
    if not hasattr(model, "input_shape"):
        return None

    input_shape = model.input_shape

    def create_input_from_shape(shape):
        if shape is None:
            return None
        # shape includes batch dimension
        shape_without_batch = shape[1:]
        new_shape = _permute_shape(
            shape_without_batch, source_format, target_format
        )
        return Input(shape=new_shape)

    # Handle single input
    if not isinstance(input_shape, list):
        return create_input_from_shape(input_shape)

    # Handle multiple inputs
    return [create_input_from_shape(s) for s in input_shape]


@keras_export("keras.utils.convert_data_format")
def convert_data_format(model, target_data_format, source_data_format=None):
    """Convert a model from one data format to another.

    This function creates a new model with the same architecture and weights
    as the input model, but with all layers converted to use the target
    data format. This is useful for converting models between
    `"channels_first"` and `"channels_last"` formats.

    The conversion handles:
    - Input shapes: Permuted to match the new format
    - Convolutional layers: `data_format` parameter updated
    - Pooling layers: `data_format` parameter updated
    - Normalization layers: `axis` parameter adjusted
    - All other layers with `data_format`: Parameter updated

    Note: The weights are copied directly without transposition, as Keras
    stores convolution kernels in a format-independent layout.

    Args:
        model: A Keras `Model` instance (Functional or Sequential).
        target_data_format: The target data format, either `"channels_first"`
            or `"channels_last"`.
        source_data_format: The source data format. If `None`, it will be
            inferred from the model's layers. Must be specified if the model
            has no layers with a `data_format` attribute.

    Returns:
        A new `Model` instance with the same weights but converted to use
        the target data format.

    Raises:
        ValueError: If the source data format cannot be inferred and is not
            provided, or if an invalid data format is specified.

    Example:

    ```python
    # Create a channels_last model
    model = keras.Sequential([
        keras.layers.Input(shape=(224, 224, 3)),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(10)
    ])

    # Convert to channels_first
    channels_first_model = keras.utils.convert_data_format(
        model, "channels_first"
    )

    # The new model expects input shape (3, 224, 224)
    # and has the same weights
    ```
    """
    valid_formats = ("channels_first", "channels_last")

    if target_data_format not in valid_formats:
        raise ValueError(
            f"Invalid target_data_format: {target_data_format}. "
            f"Must be one of {valid_formats}."
        )

    # Infer source format if not provided
    if source_data_format is None:
        source_data_format = _get_model_data_format(model)
        if source_data_format is None:
            raise ValueError(
                "Could not infer source data format from the model. "
                "Please provide the `source_data_format` argument."
            )

    if source_data_format not in valid_formats:
        raise ValueError(
            f"Invalid source_data_format: {source_data_format}. "
            f"Must be one of {valid_formats}."
        )

    # If source and target are the same, just clone the model
    if source_data_format == target_data_format:
        return clone_model(model)

    # Get input ndim for axis conversion
    input_ndim = _get_model_input_ndim(model)

    # Create clone function
    clone_fn = _create_clone_function(
        source_data_format, target_data_format, input_ndim
    )

    # Create new input tensors with permuted shapes
    # This is necessary because clone_model doesn't use clone_function for
    # InputLayers in Sequential models
    input_tensors = _create_permuted_inputs(
        model, source_data_format, target_data_format
    )

    # Clone the model with the new data format
    new_model = clone_model(
        model, input_tensors=input_tensors, clone_function=clone_fn
    )

    # Copy weights from original model to new model
    # Weights don't need transposition as Keras stores them format-independently
    for old_layer, new_layer in zip(model.layers, new_model.layers):
        if old_layer.weights:
            new_layer.set_weights(old_layer.get_weights())

    return new_model


@keras_export("keras.utils.convert_to_channels_first")
def convert_to_channels_first(model, source_data_format=None):
    """Convert a model to use channels_first data format.

    This is a convenience function that calls `convert_data_format` with
    `target_data_format="channels_first"`.

    Args:
        model: A Keras `Model` instance.
        source_data_format: The source data format. If `None`, it will be
            inferred from the model.

    Returns:
        A new model converted to channels_first format.

    Example:

    ```python
    # Create a channels_last model (default in Keras)
    model = keras.Sequential([
        keras.layers.Input(shape=(224, 224, 3)),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.Dense(10)
    ])

    # Convert to channels_first for PyTorch-style input
    channels_first_model = keras.utils.convert_to_channels_first(model)
    ```
    """
    return convert_data_format(
        model, "channels_first", source_data_format=source_data_format
    )


@keras_export("keras.utils.convert_to_channels_last")
def convert_to_channels_last(model, source_data_format=None):
    """Convert a model to use channels_last data format.

    This is a convenience function that calls `convert_data_format` with
    `target_data_format="channels_last"`.

    Args:
        model: A Keras `Model` instance.
        source_data_format: The source data format. If `None`, it will be
            inferred from the model.

    Returns:
        A new model converted to channels_last format.

    Example:

    ```python
    # Create a channels_first model
    model = keras.Sequential([
        keras.layers.Input(shape=(3, 224, 224)),
        keras.layers.Conv2D(
            32, 3, activation='relu', data_format='channels_first'
        ),
        keras.layers.Dense(10)
    ])

    # Convert to channels_last for TensorFlow-style input
    channels_last_model = keras.utils.convert_to_channels_last(model)
    ```
    """
    return convert_data_format(
        model, "channels_last", source_data_format=source_data_format
    )
