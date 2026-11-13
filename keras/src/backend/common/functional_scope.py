from keras.src.api_export import keras_export
from keras.src.backend.common import global_state


@keras_export("keras.FunctionalScope")
class FunctionalScope:
    """Scope to force ops to use symbolic calls for functional model building.

    Example:
    ```python
    >>> b = 3
    >>> keras.ops.ones((b, b))
    Array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]], dtype=float32)
    >>> with keras.FunctionalScope():
    ...   keras.ops.ones((b, b))
    ...
    <KerasTensor shape=(3, 3), dtype=float32, name=keras_tensor>
    ```
    """

    def __enter__(self):
        self.original_scope = get_functional_scope()
        global_state.set_global_attribute("functional_scope", self)
        return self

    def __exit__(self, *args, **kwargs):
        global_state.set_global_attribute(
            "functional_scope", self.original_scope
        )


@keras_export("keras.in_functional_scope")
def in_functional_scope():
    return global_state.get_global_attribute("functional_scope") is not None


def get_functional_scope():
    return global_state.get_global_attribute("functional_scope")
