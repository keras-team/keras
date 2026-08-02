from collections import OrderedDict
from functools import wraps

from keras.src import tree
from keras.src.backend.common.global_state import get_global_attribute
from keras.src.backend.common.global_state import set_global_attribute
from keras.src.utils import python_utils


class DotNotTrackScope:
    def __enter__(self):
        self.original_value = is_tracking_enabled()
        set_global_attribute("tracking_on", False)

    def __exit__(self, *args, **kwargs):
        set_global_attribute("tracking_on", self.original_value)


def is_tracking_enabled():
    return get_global_attribute("tracking_on", True)


def no_automatic_dependency_tracking(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        with DotNotTrackScope():
            return fn(*args, **kwargs)

    return wrapper


class Tracker:
    """Attribute tracker, used for e.g. Variable tracking.

    Monitors certain attribute types and places matching
    objects into user provided tracking collections.

    Also passively tracks certain mutable collections
    (e.g. dict and list) ensuring that items added after
    initialization are still tracked. This is done by wrapping
    these collections in tracking-aware proxy objects.

    Example:

    ```python
    def __init__(self):
        self.tracker = Tracker(
            # Format: `name: (test_fn, store)`
            {
                "variables":
                    (lambda x: isinstance(x, Variable), self._variables),
                "metrics": (lambda x: isinstance(x, Metric), self._metrics),
                "layers": (lambda x: isinstance(x, Layer), self._layers),
            }
        )

    def __setattr__(self, name, value):
        if hasattr(self, "_tracker"):
            value = self._tracker.track(value)
        return super().__setattr__(name, value)
    ```
    """

    def __init__(self, config, exclusions=None):
        self.config = config
        self.stored_ids = {name: set() for name in self.config.keys()}
        self.locked = False
        self._lock_violation_msg = None
        self.exclusions = exclusions or {}

    def track(self, attr):
        if not is_tracking_enabled():
            return attr

        for store_name, (is_attr_type, _) in self.config.items():
            if is_attr_type(attr):
                if store_name in self.exclusions:
                    for excl in self.exclusions[store_name]:
                        if self.is_in_store(excl, attr):
                            return attr
                if not self.is_in_store(store_name, attr):
                    self.add_to_store(store_name, attr)
                return attr
        if isinstance(attr, tuple) and hasattr(attr, "_fields"):
            # Named tuple case.
            wrapped_attr = {}
            for name, e in attr._asdict().items():
                wrapped_attr[name] = self.track(e)
            return attr.__class__(**wrapped_attr)
        if isinstance(attr, tuple):
            wrapped_attr = []
            for e in attr:
                wrapped_attr.append(self.track(e))
            return attr.__class__(wrapped_attr)
        elif isinstance(attr, list):
            return TrackedList(attr, self)
        elif isinstance(attr, OrderedDict):
            return TrackedOrderedDict(attr, self)
        elif isinstance(attr, dict):
            return TrackedDict(attr, self)
        elif isinstance(attr, set):
            return TrackedSet(attr, self)
        return attr

    def untrack(self, value):
        for store_name in self.stored_ids.keys():
            if id(value) in self.stored_ids[store_name]:
                self.stored_ids[store_name].remove(id(value))
                python_utils.remove_by_id(self.config[store_name][1], value)

    def lock(self, msg=None):
        self.locked = True
        if msg is not None:
            self._lock_violation_msg = msg

    def unlock(self):
        self.locked = False

    def add_to_store(self, store_name, value):
        if self.locked:
            raise ValueError(self._lock_violation_msg)
        self.config[store_name][1].append(value)
        self.stored_ids[store_name].add(id(value))

    def is_in_store(self, store_name, value):
        return id(value) in self.stored_ids[store_name]

    def tracks(self, value):
        """Whether `value` is currently held in any of the stores."""
        return any(id(value) in ids for ids in self.stored_ids.values())

    def replace_tracked_value(self, store_name, old_value, new_value):
        if not self.is_in_store(store_name, old_value):
            raise ValueError(f"Unknown value: {old_value}")
        store_list = self.config[store_name][1]
        index = store_list.index(old_value)
        store_list[index] = new_value
        self.stored_ids[store_name].remove(id(old_value))
        self.stored_ids[store_name].add(id(new_value))
