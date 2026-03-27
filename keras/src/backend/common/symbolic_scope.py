from keras.src.api_export import keras_export
from keras.src.backend.common import global_state


@keras_export("keras.SymbolicScope")
class SymbolicScope:
    """Scope to indicate the symbolic stage."""

    def __enter__(self):
        self.original_scope = get_symbolic_scope()
        global_state.set_global_attribute("symbolic_scope", self)
        global_state._IN_SYMBOLIC_SCOPE = True
        return self

    def __exit__(self, *args, **kwargs):
        global_state.set_global_attribute("symbolic_scope", self.original_scope)
        global_state._IN_SYMBOLIC_SCOPE = self.original_scope is not None


def in_symbolic_scope():
    return global_state._IN_SYMBOLIC_SCOPE


def get_symbolic_scope():
    return global_state.get_global_attribute("symbolic_scope")
