import weakref

from keras.src.backend.common import global_state


def set_tensor_attr(tensor, attr, value):
    try:
        setattr(tensor, attr, value)
    except AttributeError:
        if value is None:
            return
        attr_dict = global_state.get_global_attribute(f"{attr}_dict")
        if attr_dict is None:
            attr_dict = weakref.WeakValueDictionary()
            global_state.set_global_attribute(f"{attr}_dict", attr_dict)
        attr_dict[id(tensor)] = value


def get_tensor_attr(tensor, attr):
    if not hasattr(tensor, attr):
        attr_dict = global_state.get_global_attribute(f"{attr}_dict")
        if attr_dict is not None:
            return attr_dict.get(id(tensor), None)
    return getattr(tensor, attr, None)
