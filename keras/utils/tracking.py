from functools import wraps

from keras.backend.common.global_state import get_global_attribute
from keras.backend.common.global_state import set_global_attribute


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

    Monitors certain attribute types
    and put them in appropriate lists in case of a match.

    Also passively tracks certain mutable collections
    (dict, list) so that items added to them later
    still get tracked. This is done by wrapping these
    collections into an equivalent, tracking-aware object.

    Usage:

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

    def __init__(self, config):
        self.config = config
        self.stored_ids = {name: set() for name in self.config.keys()}
        self.locked = False
        self._lock_violation_msg = None

    def track(self, attr):
        if not is_tracking_enabled():
            return attr

        for store_name, (is_attr_type, _) in self.config.items():
            if is_attr_type(attr):
                if id(attr) not in self.stored_ids[store_name]:
                    self.add_to_store(store_name, attr)
                return attr
        if isinstance(attr, tuple):
            wrapped_attr = []
            for e in attr:
                wrapped_attr.append(self.track(e))
            # This should cover tuples and nametuples
            return attr.__class__(wrapped_attr)
        elif isinstance(attr, list):
            return TrackedList(attr, self)
        elif isinstance(attr, dict):
            # TODO: OrderedDict?
            return TrackedDict(attr, self)
        elif isinstance(attr, set):
            return TrackedSet(attr, self)
        return attr

    def untrack(self, value):
        for store_name in self.stored_ids.keys():
            if id(value) in self.stored_ids[store_name]:
                self.config[store_name][1].remove(value)
                self.stored_ids[store_name].remove(id(value))

    def lock(self, msg):
        self.locked = True
        self._lock_violation_msg = msg

    def add_to_store(self, store_name, value):
        if self.locked:
            raise ValueError(self._lock_violation_msg)
        self.config[store_name][1].append(value)
        self.stored_ids[store_name].add(id(value))


class TrackedList(list):
    # TODO: override item removal methods?
    def __init__(self, values=None, tracker=None):
        self.tracker = tracker
        if tracker and values:
            values = [tracker.track(v) for v in values]
        super().__init__(values or [])

    def append(self, value):
        if self.tracker:
            self.tracker.track(value)
        super().append(value)

    def insert(self, value):
        if self.tracker:
            self.tracker.track(value)
        super().insert(value)

    def extend(self, values):
        if self.tracker:
            values = [self.tracker.track(v) for v in values]
        super().extend(values)

    def remove(self, value):
        if self.tracker:
            self.tracker.untrack(value)
        super().remove(value)


class TrackedDict(dict):
    # TODO: override item removal methods?
    def __init__(self, values=None, tracker=None):
        self.tracker = tracker
        if tracker and values:
            values = {k: tracker.track(v) for k, v in values.items()}
        super().__init__(values or [])

    def __setitem__(self, key, value):
        if self.tracker:
            self.tracker.track(value)
        super().__setitem__(key, value)

    def update(self, mapping):
        if self.tracker:
            mapping = {k: self.tracker.track(v) for k, v in mapping.items()}
        super().update(mapping)


class TrackedSet(set):
    # TODO: override item removal methods?
    def __init__(self, values=None, tracker=None):
        self.tracker = tracker
        if tracker and values:
            values = {tracker.track(v) for v in values}
        super().__init__(values or [])

    def add(self, value):
        if self.tracker:
            self.tracker.track(value)
        super().add(value)

    def update(self, values):
        if self.tracker:
            values = [self.tracker.track(v) for v in values]
        super().update(values)

    def remove(self, value):
        if self.tracker:
            self.tracker.untrack(value)
        super().remove(value)
