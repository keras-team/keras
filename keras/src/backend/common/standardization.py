from keras.src.backend import config
from keras.src.utils.module_utils import tensorflow as tf


def standardize_data_format(data_format):
    if data_format is None:
        # Avoid circular imports.
        from keras.src.backend import config

        return config.image_data_format()

    data_format = str(data_format).lower()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(
            "The `data_format` argument must be one of "
            "{'channels_first', 'channels_last'}. "
            f"Received: data_format={data_format}"
        )
    return data_format


def standardize_shape(shape):
    if not isinstance(shape, tuple):
        if shape is None:
            raise ValueError("Undefined shapes are not supported.")
        if not hasattr(shape, "__iter__"):
            raise ValueError(f"Cannot convert '{shape}' to a shape.")
        if tf.available and isinstance(shape, tf.TensorShape):
            if shape.rank is None:
                raise ValueError(
                    "Tensors with unknown rank are not supported. "
                    f"Received a tensor with shape: {shape}. This "
                    "commonly happens with `tf.numpy_function` or "
                    "`tf.py_function` in a `tf.data` pipeline, which "
                    "discard static shape information. Specify the "
                    "shape explicitly, e.g. with `tensor.set_shape(...)` "
                    "or `tf.ensure_shape(tensor, shape)`."
                )
            # `tf.TensorShape` may contain `Dimension` objects.
            # We need to convert the items in it to either int or `None`
            shape = shape.as_list()

    if config.backend() == "jax":
        # Replace `_DimExpr` (dimension expression) with None
        from jax import export as jax_export

        shape = tuple(
            None if jax_export.is_symbolic_dim(d) else d for d in shape
        )

    if config.backend() == "torch":
        # Replace symbolic dimensions with None to preserve dynamic shapes
        # during torch.export tracing
        import torch

        shape = tuple(None if isinstance(d, torch.SymInt) else d for d in shape)

    # Handle dimensions that are not ints and not None, verify they're >= 0.
    standardized_shape = []
    for d in shape:
        if d is None:
            standardized_shape.append(d)
            continue

        # Reject these even if they can be cast to int successfully.
        if isinstance(d, (str, float)):
            raise ValueError(
                f"Cannot convert '{shape}' to a shape. "
                f"Found invalid dimension '{d}' of type '{type(d)}'. "
            )

        try:
            # Cast numpy scalars, tf constant tensors, etc.
            d = int(d)
        except Exception as e:
            raise ValueError(
                f"Cannot convert '{shape}' to a shape. "
                f"Found invalid dimension '{d}' of type '{type(d)}'. "
            ) from e
        if d < 0:
            raise ValueError(
                f"Cannot convert '{shape}' to a shape. "
                "Negative dimensions are not allowed."
            )
        standardized_shape.append(d)

    # This also turns subclasses of `tuple` (e.g. `torch.Size`) to plain tuple.
    return tuple(standardized_shape)
