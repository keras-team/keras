import weakref

from keras.src.backend.common import global_state

# Cache of `attr -> f"{attr}_dict"` global-state key strings. `attr` is
# always one of a small fixed set of literal attribute names (e.g.
# `_keras_mask`) shared by every tensor in the process, so the number of
# distinct keys is bounded and this cache avoids allocating a fresh f-string
# on every miss (`get_tensor_attr` is on the mask-lookup hot path, called
# once per positional call argument).
_DICT_NAME_CACHE = {}


def _dict_name(attr):
    name = _DICT_NAME_CACHE.get(attr)
    if name is None:
        name = f"{attr}_dict"
        _DICT_NAME_CACHE[attr] = name
    return name


def _clear_tensor_attr(tensor_id, attr):
    attr_dict = global_state.get_global_attribute(_dict_name(attr))
    if attr_dict is not None and tensor_id in attr_dict:
        del attr_dict[tensor_id]


def set_tensor_attr(tensor, attr, value):
    try:
        setattr(tensor, attr, value)
    except AttributeError:
        dict_name = _dict_name(attr)
        attr_dict = global_state.get_global_attribute(dict_name)
        if attr_dict is None:
            if value is None:
                return
            attr_dict = {}
            global_state.set_global_attribute(dict_name, attr_dict)
        if value is not None:
            attr_dict[id(tensor)] = value
            weakref.finalize(tensor, _clear_tensor_attr, id(tensor), attr)
        elif id(tensor) in attr_dict:
            del attr_dict[id(tensor)]


def get_tensor_attr(tensor, attr):
    if not hasattr(tensor, attr):
        attr_dict = global_state.get_global_attribute(_dict_name(attr))
        if attr_dict is not None:
            return attr_dict.get(id(tensor), None)
        else:
            return None
    return getattr(tensor, attr, None)
