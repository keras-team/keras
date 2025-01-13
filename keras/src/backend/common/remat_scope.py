from keras.src.backend.common import global_state


class RematScope:
    """
    A context manager for enabling rematerialization in Keras.

    Args:
        mode (str): Rematerialization mode to apply.
            Options:
            - "full": Apply rematerialization globally to all supported
                operations.
            - "activations": Apply rematerialization only to activation layers.
            - "larger_than": Apply rematerialization to layers with output
                sizes larger than a threshold.
            - None: Disable rematerialization.
    """

    def __init__(self, mode="full"):
        if mode not in {"full", "activations", "larger_than", None}:
            raise ValueError(
                f"Invalid mode '{mode}'. Supported modes are: "
                "'full', 'activations', 'larger_than', or None."
            )
        self.mode = mode
        self._pop_on_exit = False

    def __enter__(self):
        remat_scope_stack = global_state.get_global_attribute(
            "remat_scope_stack", default=[], set_to_default=True
        )
        remat_scope_stack.append(self)
        self._pop_on_exit = True
        return self

    def __exit__(self, *args, **kwargs):
        if self._pop_on_exit:
            remat_scope_stack = global_state.get_global_attribute(
                "remat_scope_stack"
            )
            remat_scope_stack.pop()


def get_current_remat_mode():
    """
    Get the current rematerialization mode from the active RematScope.

    Returns:
        str: The rematerialization mode or `None` if no scope is active.
    """
    remat_scope_stack = global_state.get_global_attribute("remat_scope_stack")
    if remat_scope_stack is None or not remat_scope_stack:
        return None
    return remat_scope_stack[-1].mode
