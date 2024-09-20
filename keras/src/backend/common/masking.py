import weakref

from keras.src.backend.common import global_state


def set_keras_mask(x, mask):
    try:
        x._keras_mask = mask
    except AttributeError:
        if mask is None:
            return
        mask_dict = global_state.get_global_attribute("keras_mask_dict")
        if mask_dict is None:
            mask_dict = weakref.WeakValueDictionary()
            global_state.set_global_attribute("keras_mask_dict", mask_dict)
        mask_dict[id(x)] = mask


def get_keras_mask(x):
    if not hasattr(x, "_keras_mask"):
        mask_dict = global_state.get_global_attribute("keras_mask_dict")
        if mask_dict is not None:
            return mask_dict.get(id(x), None)
    return getattr(x, "_keras_mask", None)
