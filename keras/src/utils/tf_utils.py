from keras.src.utils.module_utils import tensorflow as tf


def get_tensor_spec(t, dynamic_batch=False, name=None):
    """Returns a `TensorSpec` given a single `Tensor` or `TensorSpec`."""
    if isinstance(t, tf.TypeSpec):
        spec = t
    elif isinstance(t, tf.__internal__.CompositeTensor):
        # Check for ExtensionTypes
        spec = t._type_spec
    elif hasattr(t, "shape") and hasattr(t, "dtype"):
        spec = tf.TensorSpec(shape=t.shape, dtype=t.dtype, name=name)
    else:
        return None  # Allow non-Tensors to pass through.

    if not dynamic_batch:
        return spec

    shape = spec.shape
    if shape.rank is None or shape.rank == 0:
        return spec

    shape_list = shape.as_list()
    shape_list[0] = None
    shape = tf.TensorShape(shape_list)
    spec._shape = shape
    return spec
