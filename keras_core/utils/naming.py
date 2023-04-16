import collections
import re

OBJECT_NAME_UIDS = collections.defaultdict(int)


def auto_name(prefix):
    prefix = to_snake_case(prefix)
    return uniquify(prefix)


def uniquify(name):
    global OBJECT_NAME_UIDS
    if name in OBJECT_NAME_UIDS:
        unique_name = f"{name}_{OBJECT_NAME_UIDS[name]}"
    else:
        unique_name = name
    OBJECT_NAME_UIDS[name] += 1
    return unique_name


def to_snake_case(name):
    name = re.sub(r"\W+", "", name)
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub("([a-z])([A-Z])", r"\1_\2", name).lower()
    return name


def reset_uids():
    global OBJECT_NAME_UIDS
    OBJECT_NAME_UIDS = collections.defaultdict(int)


def get_object_name(obj):
    if hasattr(obj, "name"):  # Most Keras objects.
        return obj.name
    elif hasattr(obj, "__name__"):  # Function.
        return to_snake_case(obj.__name__)
    elif hasattr(obj, "__class__"):  # Class instance.
        return to_snake_case(obj.__class__.__name__)
    return to_snake_case(str(obj))
