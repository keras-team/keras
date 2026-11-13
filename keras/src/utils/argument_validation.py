import numpy as np


def validate_int(name, value, allow_none=False, min_value=None):
    """Validates an integer argument and converts it to a Python `int`.

    A Python `int` or a NumPy integer, such as `np.int64(2)`, is accepted.
    A `bool` is not accepted.

    Args:
        name: string. The name of the argument, e.g. `"units"`. This is only
            used to format error messages.
        value: The value of the argument.
        allow_none: bool, defaults to `False`. If `True`, `None` is accepted
            and returned unchanged.
        min_value: int, optional. The smallest accepted value. For example,
            use `1` for a positive integer and `0` for a non-negative
            integer. Defaults to `None`, which accepts any integer.

    Returns:
        The value as a Python `int`, or `None` if `value` is `None` and
        `allow_none` is `True`.

    Raises:
        TypeError: If `value` is not an integer.
        ValueError: If `value` is smaller than `min_value`.
    """
    if value is None and allow_none:
        return None
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        expected = "an integer or `None`" if allow_none else "an integer"
        raise TypeError(
            f"Expected `{name}` to be {expected}. "
            f"Received: {name}={value!r} of type {type(value).__name__}"
        )
    value = int(value)
    if min_value is not None and value < min_value:
        raise ValueError(
            f"Expected `{name}` to be an integer >= {min_value}. "
            f"Received: {name}={value}"
        )
    return value


def validate_int_sequence(
    name, value, length=None, allow_none=False, min_value=None
):
    """Validates a list or a tuple of integers.

    Each element is validated with `validate_int()`.

    Args:
        name: string. The name of the argument, e.g. `"noise_shape"`. This is
            only used to format error messages.
        value: list or tuple. The value of the argument.
        length: int, optional. The required number of elements. Defaults to
            `None`, which accepts any number of elements.
        allow_none: bool, defaults to `False`. If `True`, an element can be
            `None`, e.g. for an unknown dimension in a shape.
        min_value: int, optional. The smallest accepted value of an element.
            Defaults to `None`, which accepts any integer.

    Returns:
        A list if `value` is a list, else a tuple, with each element as a
        Python `int` (or `None`).

    Raises:
        TypeError: If `value` is not a list or a tuple, or if an element is
            not an integer.
        ValueError: If `value` does not have `length` elements, or if an
            element is smaller than `min_value`.
    """
    if not isinstance(value, (list, tuple)):
        raise TypeError(
            f"Expected `{name}` to be a list or a tuple of integers. "
            f"Received: {name}={value!r} of type {type(value).__name__}"
        )
    if length is not None and len(value) != length:
        raise ValueError(
            f"Expected `{name}` to have {length} elements. "
            f"Received: {name}={value!r}"
        )
    result = [
        validate_int(
            f"{name}[{i}]", v, allow_none=allow_none, min_value=min_value
        )
        for i, v in enumerate(value)
    ]
    return result if isinstance(value, list) else tuple(result)


def standardize_tuple(value, n, name, allow_zero=False):
    """Transforms non-negative/positive integer/integers into an integer tuple.

    Args:
        value: int or iterable of ints. The value to validate and convert.
        n: int. The size of the tuple to be returned.
        name: string. The name of the argument being validated, e.g. "strides"
            or "kernel_size". This is only used to format error messages.
        allow_zero: bool, defaults to `False`. A `ValueError` will raised
            if zero is received and this argument is `False`.

    Returns:
        A tuple of n integers.
    """
    error_msg = (
        f"The `{name}` argument must be a tuple of {n} integers. "
        f"Received {name}={value}"
    )

    if isinstance(value, (int, np.integer)):
        value_tuple = (int(value),) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise ValueError(error_msg)
        if len(value_tuple) != n:
            raise ValueError(error_msg)
        converted_values = []
        for single_value in value_tuple:
            try:
                converted_values.append(int(single_value))
            except (ValueError, TypeError):
                error_msg += (
                    f"including element {single_value} of "
                    f"type {type(single_value)}"
                )
                raise ValueError(error_msg)
        value_tuple = tuple(converted_values)

    if allow_zero:
        unqualified_values = {v for v in value_tuple if v < 0}
        req_msg = ">= 0"
    else:
        unqualified_values = {v for v in value_tuple if v <= 0}
        req_msg = "> 0"

    if unqualified_values:
        error_msg += (
            f", including values {unqualified_values}"
            f" that do not satisfy `value {req_msg}`"
        )
        raise ValueError(error_msg)

    return value_tuple


def standardize_padding(value, allow_causal=False):
    if isinstance(value, (list, tuple)):
        return value
    padding = value.lower()
    if allow_causal:
        allowed_values = {"valid", "same", "causal"}
    else:
        allowed_values = {"valid", "same"}
    if padding not in allowed_values:
        raise ValueError(
            "The `padding` argument must be a list/tuple or one of "
            f"{allowed_values}. "
            f"Received: {padding}"
        )
    return padding


def validate_string_arg(
    value,
    allowable_strings,
    caller_name,
    arg_name,
    allow_none=False,
    allow_callables=False,
):
    """Validates the correctness of a string-based arg."""
    if allow_none and value is None:
        return
    elif allow_callables and callable(value):
        return
    elif isinstance(value, str) and value in allowable_strings:
        return
    raise ValueError(
        f"Unknown value for `{arg_name}` argument of {caller_name}. "
        f"Allowed values are: {allowable_strings}. Received: "
        f"{arg_name}={value}"
    )
