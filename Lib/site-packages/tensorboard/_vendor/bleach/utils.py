from collections import OrderedDict


def _attr_key(attr):
    """Returns appropriate key for sorting attribute names

    Attribute names are a tuple of ``(namespace, name)`` where namespace can be
    ``None`` or a string. These can't be compared in Python 3, so we conver the
    ``None`` to an empty string.

    """
    key = (attr[0][0] or ''), attr[0][1]
    return key


def alphabetize_attributes(attrs):
    """Takes a dict of attributes (or None) and returns them alphabetized"""
    if not attrs:
        return attrs

    return OrderedDict(
        [(k, v) for k, v in sorted(attrs.items(), key=_attr_key)]
    )
