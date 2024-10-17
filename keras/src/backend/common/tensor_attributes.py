import weakref

from keras.src.backend.common import global_state


def _clear_tensor_attr(tensor_id, attr):
    attr_dict = global_state.get_global_attribute(f"{attr}_dict")
    if attr_dict is not None and tensor_id in attr_dict:
        del attr_dict[tensor_id]


def set_tensor_attr(tensor, attr, value):
    try:
        setattr(tensor, attr, value)
    except AttributeError:
        attr_dict = global_state.get_global_attribute(f"{attr}_dict")
        if attr_dict is None:
            if value is None:
                return
            attr_dict = {}
            global_state.set_global_attribute(f"{attr}_dict", attr_dict)
        if value is not None:
            attr_dict[id(tensor)] = value
            weakref.finalize(tensor, _clear_tensor_attr, id(tensor), attr)
        elif id(tensor) in attr_dict:
            del attr_dict[id(tensor)]


def get_tensor_attr(tensor, attr):
    if not hasattr(tensor, attr):
        attr_dict = global_state.get_global_attribute(f"{attr}_dict")
        if attr_dict is not None:
            return attr_dict.get(id(tensor), None)
        else:
            return None
    return getattr(tensor, attr, None)
