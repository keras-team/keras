from collections import Counter

from .variable import variable


_LEARNING_PHASE = variable(0, dtype='uint8', name='keras_learning_phase')

_UID_PREFIXES = Counter()


def learning_phase():
    return _LEARNING_PHASE


def set_learning_phase(value):
    global _LEARNING_PHASE
    assert value in {0, 1}
    _LEARNING_PHASE = value


def get_uid(prefix=''):
    _UID_PREFIXES[prefix] += 1
    return _UID_PREFIXES[prefix]


def reset_uids():
    global _UID_PREFIXES
    _UID_PREFIXES = Counter()
