from keras.src.api_export import keras_export
from keras.src.backend.common import global_state


@keras_export("keras.StatelessScope")
class StatelessScope:
    """Scope to prevent any update to Keras Variables.

    The values of variables to be used inside the scope
    should be passed via the `state_mapping` argument, a
    list of tuples `(k, v)` where `k` is a `Variable`
    and `v` is the intended value for this variable
    (a backend tensor).

    Updated values can be collected on scope exit via
    `value = scope.get_current_value(variable)`. No updates
    will be applied in-place to any variables for the duration
    of the scope.

    Example:

    ```python
    state_mapping = [(k, ops.ones(k.shape, k.dtype)) for k in model.weights]
    with keras.StatelessScope(state_mapping) as scope:
        outputs = model.some_function(inputs)

    # All model variables remain unchanged. Their new values can be
    # collected via:
    for k in model.weights:
        new_value = scope.get_current_value(k)
        print(f"New value for {k}: {new_value})
    ```
    """

    def __init__(
        self,
        state_mapping=None,
        collect_losses=False,
        initialize_variables=True,
    ):
        from keras.src import backend
        from keras.src.backend.common.variables import Variable

        self.collect_losses = collect_losses
        self.initialize_variables = initialize_variables
        self.losses = []
        self.state_mapping = {}
        state_mapping = state_mapping or {}
        for k, v in state_mapping:
            if not isinstance(k, Variable):
                raise ValueError(
                    "Invalid reference variable in StatelessScope: "
                    "all keys in argument `mapping` must be Variable "
                    f"instances. Received instead: {k}"
                )
            if isinstance(v, Variable):
                v = backend.cast(v.value, dtype=k.dtype)
            else:
                v = backend.convert_to_tensor(v, dtype=k.dtype)
            if k.shape != v.shape:
                raise ValueError(
                    "Invalid variable value in StatelessScope: "
                    "all values in argument `mapping` must be tensors with "
                    "a shape that matches the corresponding variable shape. "
                    f"For variable {k}, received invalid value {v} with shape "
                    f"{v.shape}."
                )
            self.state_mapping[id(k)] = v

    def __enter__(self):
        self.original_scope = get_stateless_scope()
        global_state.set_global_attribute("stateless_scope", self)
        return self

    def add_loss(self, loss):
        self.losses.append(loss)

    def add_update(self, update):
        variable, value = update
        self.state_mapping[id(variable)] = value

    def get_current_value(self, variable):
        return self.state_mapping.get(id(variable), None)

    def __exit__(self, *args, **kwargs):
        global_state.set_global_attribute(
            "stateless_scope", self.original_scope
        )
        if self.original_scope is None and self.initialize_variables:
            # We're back in eager scope;
            # if any variables were created within the stateless
            # scope, we initialize them here.
            from keras.src.backend.common.variables import (
                initialize_all_variables,
            )

            initialize_all_variables()


def in_stateless_scope():
    return global_state.get_global_attribute("stateless_scope") is not None


def get_stateless_scope():
    return global_state.get_global_attribute("stateless_scope")
