import re


def should_quantize_layer(layer, filters):
    """Determines if a layer should be quantized based on filters.

    Args:
        layer: The layer to check.
        filters: A regex string, a list of regex strings, or a callable.
            If None, returns True.

    Returns:
        True if the layer should be quantized, False otherwise.
    """
    if filters is None:
        return True
    if isinstance(filters, str):
        return bool(re.search(filters, layer.name))
    if isinstance(filters, (list, tuple)):
        return any(re.search(pat, layer.name) for pat in filters)
    if callable(filters):
        return filters(layer)
    return True
