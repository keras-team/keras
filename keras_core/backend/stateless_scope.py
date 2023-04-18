import threading
from keras_core.backend.common import KerasVariable


### Stateless context manager

GLOBAL_SCOPE_TRACKER = threading.local()


class StatelessScope:
    def __init__(self, state_mapping=None, collect_losses=False):
        from keras_core import backend

        self.collect_losses = collect_losses
        self.losses = []
        self.state_mapping = {}
        state_mapping = state_mapping or {}
        for k, v in state_mapping:
            if not isinstance(k, KerasVariable):
                raise ValueError(
                    "Invalid reference variable in VariableSwapScope: "
                    "all keys in argument `mapping` must be KerasVariable "
                    f"instances. Received instead: {k}"
                )
            v = backend.convert_to_tensor(v, dtype=k.dtype)
            if k.shape != v.shape:
                raise ValueError(
                    "Invalid variable value in VariableSwapScope: "
                    "all values in argument `mapping` must be tensors with "
                    "a shape that matches the corresponding variable shape. "
                    f"For variable {k}, received invalid value {v} with shape {v.shape}."
                )
            self.state_mapping[id(k)] = v

    def __enter__(self):
        self.original_scope = get_stateless_scope()
        GLOBAL_SCOPE_TRACKER.stateless_scope = self
        return self

    def add_loss(self, loss):
        self.losses.append(loss)

    def add_update(self, update):
        variable, value = update
        self.state_mapping[id(variable)] = value

    def get_current_value(self, variable):
        return self.state_mapping.get(id(variable), None)

    def __exit__(self, *args, **kwargs):
        GLOBAL_SCOPE_TRACKER.stateless_scope = self.original_scope


def in_stateless_scope():
    return getattr(GLOBAL_SCOPE_TRACKER, "stateless_scope", None) is not None


def get_stateless_scope():
    return getattr(GLOBAL_SCOPE_TRACKER, "stateless_scope", None)
