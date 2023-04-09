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
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub("([a-z])([A-Z])", r"\1_\2", name).lower()
    return name


def reset_uids():
    global OBJECT_NAME_UIDS
    OBJECT_NAME_UIDS = collections.defaultdict(int)
