def default(method):
    """Decorates a method to detect overrides in subclasses."""
    method._is_default = True
    return method


def is_default(method):
    """Check if a method is decorated with the `default` wrapper."""
    return getattr(method, "_is_default", False)
