from keras.src.backend.common.tensor_attributes import get_tensor_attr
from keras.src.backend.common.tensor_attributes import set_tensor_attr


def set_keras_mask(x, mask):
    """Sets the Keras mask attribute for the given tensor in-place.

    Args:
        x: Input tensor.
        mask: The mask tensor to be set. If `None`, the `_keras_mask` attribute
            will be cleared.
    """
    set_tensor_attr(x, "_keras_mask", mask)


def get_keras_mask(x):
    """Gets the Keras mask attribute from the given tensor.

    Args:
        x: Input tensor.

    Returns:
        The mask tensor associated with the input tensor, or `None` if no mask
        has been set.
    """
    return get_tensor_attr(x, "_keras_mask")
