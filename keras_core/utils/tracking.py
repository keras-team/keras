import threading

GLOBAL_SCOPE_TRACKER = threading.local()


class DotNotTrackScope:
    def __enter__(self):
        self.original_value = is_tracking_enabled()
        GLOBAL_SCOPE_TRACKER.tracking_on = False

    def __exit__(self, *args, **kwargs):
        GLOBAL_SCOPE_TRACKER.tracking_on = self.original_value


def is_tracking_enabled():
    return getattr(GLOBAL_SCOPE_TRACKER, "tracking_on", True)


def no_automatic_dependency_tracking(fn):
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

    def track(self, attr):
        if not is_tracking_enabled():
            return attr

        for name, (is_attr_type, store) in self.config.items():
            if is_attr_type(attr):
                if id(attr) not in self.stored_ids[name]:
                    store.append(attr)
                    self.stored_ids[name].add(id(attr))
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
            # TODO: OrderedDict
            return TrackedDict(attr, self)
        elif isinstance(attr, set):
            return TrackedSet(attr, self)
        return attr


class TrackedList(list):
    # TODO(fchollet): override item removal methods?
    def __init__(self, values, tracker):
        self.tracker = tracker
        values = [tracker.track(v) for v in values]
        super().__init__(values)

    def append(self, value):
        self.tracker.track(value)
        super().append(value)

    def insert(self, value):
        self.tracker.track(value)
        super().insert(value)

    def extend(self, values):
        values = [self.tracker.track(v) for v in values]
        super().extend(values)


class TrackedDict(dict):
    # TODO(fchollet): override item removal methods?
    def __init__(self, values, tracker):
        self.tracker = tracker
        values = {k: tracker.track(v) for k, v in values.items()}
        super().__init__(values)

    def __setitem__(self, key, value):
        self.tracker.track(value)
        super().__setitem__(key, value)

    def update(self, mapping):
        mapping = {k: self.tracker.track(v) for k, v in mapping.items()}
        super().update(mapping)


class TrackedSet(set):
    # TODO(fchollet): override item removal methods?
    def __init__(self, values, tracker):
        self.tracker = tracker
        values = {tracker.track(v) for v in values}
        super().__init__(values)

    def add(self, value):
        self.tracker.track(value)
        super().add(value)

    def update(self, values):
        values = [self.tracker.track(v) for v in values]
        super().update(values)
