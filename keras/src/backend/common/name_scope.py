from keras.src.backend.common import global_state


def _get_name_scope_stack():
    """Get the name scope stack, creating it if needed.

    Uses direct getattr on GLOBAL_STATE_TRACKER for speed.
    """
    tracker = global_state.GLOBAL_STATE_TRACKER
    stack = getattr(tracker, "name_scope_stack", None)
    if stack is None:
        stack = []
        tracker.name_scope_stack = stack
    return stack


class name_scope:
    """Creates a sub-namespace for variable paths.

    Args:
        name: Name of the current scope (string).
        caller: Optional ID of a caller object (e.g. class instance).
        deduplicate: If `True`, if `caller` was passed,
            and the previous caller matches the current caller,
            and the previous name matches the current name,
            do not reenter a new namespace.
        override_parent: Can be used to provide an absolute path
            which would override any previously opened name scopes.
    """

    def __init__(
        self, name, caller=None, deduplicate=True, override_parent=None
    ):
        if not isinstance(name, str) or "/" in name:
            raise ValueError(
                "Argument `name` must be a string and "
                "cannot contain character `/`. "
                f"Received: name={name}"
            )
        self.name = name
        self.caller = caller
        self.deduplicate = deduplicate
        self.override_parent = override_parent
        if (
            override_parent is None
            and deduplicate
            and getattr(caller, "_parent_path", None) is not None
        ):
            self.override_parent = caller._parent_path
        self._pop_on_exit = False

    def __enter__(self):
        name_scope_stack = _get_name_scope_stack()
        if self.deduplicate and name_scope_stack:
            parent_caller = name_scope_stack[-1].caller
            parent_name = name_scope_stack[-1].name
            if (
                self.caller is not None
                and self.caller is parent_caller
                and self.name == parent_name
            ):
                return self
        name_scope_stack.append(self)
        self._pop_on_exit = True
        return self

    def __exit__(self, *args, **kwargs):
        if self._pop_on_exit:
            name_scope_stack = _get_name_scope_stack()
            if name_scope_stack:
                name_scope_stack.pop()


def current_path():
    name_scope_stack = getattr(
        global_state.GLOBAL_STATE_TRACKER, "name_scope_stack", None
    )
    if name_scope_stack is None:
        return ""
    parts = []
    for entry in name_scope_stack:
        if entry.override_parent is not None:
            parts = [p for p in entry.override_parent.split("/") if p]
        parts.append(entry.name)
    return "/".join(parts)
