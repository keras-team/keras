from keras.src.backend.common.tensor_attributes import get_tensor_attr
from keras.src.backend.common.tensor_attributes import set_tensor_attr


def set_keras_mask(x, mask):
    return set_tensor_attr(x, "_keras_mask", mask)


def get_keras_mask(x):
    return get_tensor_attr(x, "_keras_mask")
