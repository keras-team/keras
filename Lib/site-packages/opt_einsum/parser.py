"""A functionally equivalent parser of the numpy.einsum input parser."""

import itertools
from typing import Any, Dict, Iterator, List, Sequence, Tuple

from opt_einsum.typing import ArrayType, TensorShapeType

__all__ = [
    "is_valid_einsum_char",
    "has_valid_einsum_chars_only",
    "get_symbol",
    "get_shape",
    "gen_unused_symbols",
    "convert_to_valid_einsum_chars",
    "alpha_canonicalize",
    "find_output_str",
    "find_output_shape",
    "possibly_convert_to_numpy",
    "parse_einsum_input",
]

_einsum_symbols_base = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def is_valid_einsum_char(x: str) -> bool:
    """Check if the character ``x`` is valid for numpy einsum.

    **Examples:**

    ```python
    is_valid_einsum_char("a")
    #> True

    is_valid_einsum_char("Ǵ")
    #> False
    ```
    """
    return (x in _einsum_symbols_base) or (x in ",->.")


def has_valid_einsum_chars_only(einsum_str: str) -> bool:
    """Check if ``einsum_str`` contains only valid characters for numpy einsum.

    **Examples:**

    ```python
    has_valid_einsum_chars_only("abAZ")
    #> True

    has_valid_einsum_chars_only("Över")
    #> False
    ```
    """
    return all(map(is_valid_einsum_char, einsum_str))


def get_symbol(i: int) -> str:
    """Get the symbol corresponding to int ``i`` - runs through the usual 52
    letters before resorting to unicode characters, starting at ``chr(192)`` and skipping surrogates.

    **Examples:**

    ```python
    get_symbol(2)
    #> 'c'

    get_symbol(200)
    #> 'Ŕ'

    get_symbol(20000)
    #> '京'
    ```
    """
    if i < 52:
        return _einsum_symbols_base[i]
    elif i >= 55296:
        # Skip chr(57343) - chr(55296) as surrogates
        return chr(i + 2048)
    else:
        return chr(i + 140)


def gen_unused_symbols(used: str, n: int) -> Iterator[str]:
    """Generate ``n`` symbols that are not already in ``used``.

    **Examples:**
    ```python
    list(oe.parser.gen_unused_symbols("abd", 2))
    #> ['c', 'e']
    ```
    """
    i = cnt = 0
    while cnt < n:
        s = get_symbol(i)
        i += 1
        if s in used:
            continue
        yield s
        cnt += 1


def convert_to_valid_einsum_chars(einsum_str: str) -> str:
    """Convert the str ``einsum_str`` to contain only the alphabetic characters
    valid for numpy einsum. If there are too many symbols, let the backend
    throw an error.

    Examples:
    --------
    >>> oe.parser.convert_to_valid_einsum_chars("Ĥěļļö")
    'cbdda'
    """
    symbols = sorted(set(einsum_str) - set(",->"))
    replacer = {x: get_symbol(i) for i, x in enumerate(symbols)}
    return "".join(replacer.get(x, x) for x in einsum_str)


def alpha_canonicalize(equation: str) -> str:
    """Alpha convert an equation in an order-independent canonical way.

    Examples:
    --------
    >>> oe.parser.alpha_canonicalize("dcba")
    'abcd'

    >>> oe.parser.alpha_canonicalize("Ĥěļļö")
    'abccd'
    """
    rename: Dict[str, str] = {}
    for name in equation:
        if name in ".,->":
            continue
        if name not in rename:
            rename[name] = get_symbol(len(rename))
    return "".join(rename.get(x, x) for x in equation)


def find_output_str(subscripts: str) -> str:
    """Find the output string for the inputs ``subscripts`` under canonical einstein summation rules.
    That is, repeated indices are summed over by default.

    Examples:
    --------
    >>> oe.parser.find_output_str("ab,bc")
    'ac'

    >>> oe.parser.find_output_str("a,b")
    'ab'

    >>> oe.parser.find_output_str("a,a,b,b")
    ''
    """
    tmp_subscripts = subscripts.replace(",", "")
    return "".join(s for s in sorted(set(tmp_subscripts)) if tmp_subscripts.count(s) == 1)


def find_output_shape(inputs: List[str], shapes: List[TensorShapeType], output: str) -> TensorShapeType:
    """Find the output shape for given inputs, shapes and output string, taking
    into account broadcasting.

    Examples:
    --------
    >>> oe.parser.find_output_shape(["ab", "bc"], [(2, 3), (3, 4)], "ac")
    (2, 4)

    # Broadcasting is accounted for
    >>> oe.parser.find_output_shape(["a", "a"], [(4, ), (1, )], "a")
    (4,)
    """
    return tuple(max(shape[loc] for shape, loc in zip(shapes, [x.find(c) for x in inputs]) if loc >= 0) for c in output)


_BaseTypes = (bool, int, float, complex, str, bytes)


def get_shape(x: Any) -> TensorShapeType:
    """Get the shape of the array-like object `x`. If `x` is not array-like, raise an error.

    Array-like objects are those that have a `shape` attribute, are sequences of BaseTypes, or are BaseTypes.
    BaseTypes are defined as `bool`, `int`, `float`, `complex`, `str`, and `bytes`.
    """
    if hasattr(x, "shape"):
        return x.shape
    elif isinstance(x, _BaseTypes):
        return ()
    elif isinstance(x, Sequence):
        shape = []
        while isinstance(x, Sequence) and not isinstance(x, _BaseTypes):
            shape.append(len(x))
            x = x[0]
        return tuple(shape)
    else:
        raise ValueError(f"Cannot determine the shape of {x}, can only determine the shape of array-like objects.")


def possibly_convert_to_numpy(x: Any) -> Any:
    """Convert things without a 'shape' to ndarrays, but leave everything else.

    Examples:
    --------
    >>> oe.parser.possibly_convert_to_numpy(5)
    array(5)

    >>> oe.parser.possibly_convert_to_numpy([5, 3])
    array([5, 3])

    >>> oe.parser.possibly_convert_to_numpy(np.array([5, 3]))
    array([5, 3])

    # Any class with a shape is passed through
    >>> class Shape:
    ...     def __init__(self, shape):
    ...         self.shape = shape
    ...

    >>> myshape = Shape((5, 5))
    >>> oe.parser.possibly_convert_to_numpy(myshape)
    <__main__.Shape object at 0x10f850710>
    """
    if not hasattr(x, "shape"):
        try:
            import numpy as np  # type: ignore
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "numpy is required to convert non-array objects to arrays. This function will be deprecated in the future."
            )

        return np.asanyarray(x)
    else:
        return x


def convert_subscripts(old_sub: List[Any], symbol_map: Dict[Any, Any]) -> str:
    """Convert user custom subscripts list to subscript string according to `symbol_map`.

    Examples:
    --------
    >>>  oe.parser.convert_subscripts(['abc', 'def'], {'abc':'a', 'def':'b'})
    'ab'
    >>> oe.parser.convert_subscripts([Ellipsis, object], {object:'a'})
    '...a'
    """
    new_sub = ""
    for s in old_sub:
        if s is Ellipsis:
            new_sub += "..."
        else:
            # no need to try/except here because symbol_map has already been checked
            new_sub += symbol_map[s]
    return new_sub


def convert_interleaved_input(operands: Sequence[Any]) -> Tuple[str, Tuple[Any, ...]]:
    """Convert 'interleaved' input to standard einsum input."""
    tmp_operands = list(operands)
    operand_list = []
    subscript_list = []
    for _ in range(len(operands) // 2):
        operand_list.append(tmp_operands.pop(0))
        subscript_list.append(tmp_operands.pop(0))

    output_list = tmp_operands[-1] if len(tmp_operands) else None

    # build a map from user symbols to single-character symbols based on `get_symbol`
    # The map retains the intrinsic order of user symbols
    try:
        # collect all user symbols
        symbol_set = set(itertools.chain.from_iterable(subscript_list))

        # remove Ellipsis because it can not be compared with other objects
        symbol_set.discard(Ellipsis)

        # build the map based on sorted user symbols, retaining the order we lost in the `set`
        symbol_map = {symbol: get_symbol(idx) for idx, symbol in enumerate(sorted(symbol_set))}

    except TypeError:  # unhashable or uncomparable object
        raise TypeError(
            "For this input type lists must contain either Ellipsis "
            "or hashable and comparable object (e.g. int, str)."
        )

    subscripts = ",".join(convert_subscripts(sub, symbol_map) for sub in subscript_list)
    if output_list is not None:
        subscripts += "->"
        subscripts += convert_subscripts(output_list, symbol_map)

    return subscripts, tuple(operand_list)


def parse_einsum_input(operands: Any, shapes: bool = False) -> Tuple[str, str, List[ArrayType]]:
    """A reproduction of einsum c side einsum parsing in python.

    Parameters:
        operands: Intakes the same inputs as `contract_path`, but NOT the keyword args. The only
            supported keyword argument is:
        shapes: Whether ``parse_einsum_input`` should assume arrays (the default) or
            array shapes have been supplied.

    Returns:
        input_strings: Parsed input strings
        output_string: Parsed output string
        operands: The operands to use in the numpy contraction

    Examples:
        The operand list is simplified to reduce printing:

        ```python
        >>> a = np.random.rand(4, 4)
        >>> b = np.random.rand(4, 4, 4)
        >>> parse_einsum_input(('...a,...a->...', a, b))
        ('za,xza', 'xz', [a, b])

        >>> parse_einsum_input((a, [Ellipsis, 0], b, [Ellipsis, 0]))
        ('za,xza', 'xz', [a, b])
        ```
    """
    if len(operands) == 0:
        raise ValueError("No input operands")

    if isinstance(operands[0], str):
        subscripts = operands[0].replace(" ", "")
        if shapes:
            if any(hasattr(o, "shape") for o in operands[1:]):
                raise ValueError(
                    "shapes is set to True but given at least one operand looks like an array"
                    " (at least one operand has a shape attribute). "
                )
        operands = operands[1:]
    else:
        subscripts, operands = convert_interleaved_input(operands)

    if shapes:
        operand_shapes = operands
    else:
        operand_shapes = [get_shape(o) for o in operands]

    # Check for proper "->"
    if ("-" in subscripts) or (">" in subscripts):
        invalid = (subscripts.count("-") > 1) or (subscripts.count(">") > 1)
        if invalid or (subscripts.count("->") != 1):
            raise ValueError("Subscripts can only contain one '->'.")

    # Parse ellipses
    if "." in subscripts:
        used = subscripts.replace(".", "").replace(",", "").replace("->", "")
        ellipse_inds = "".join(gen_unused_symbols(used, max(len(x) for x in operand_shapes)))
        longest = 0

        # Do we have an output to account for?
        if "->" in subscripts:
            input_tmp, output_sub = subscripts.split("->")
            split_subscripts = input_tmp.split(",")
            out_sub = True
        else:
            split_subscripts = subscripts.split(",")
            out_sub = False

        for num, sub in enumerate(split_subscripts):
            if "." in sub:
                if (sub.count(".") != 3) or (sub.count("...") != 1):
                    raise ValueError("Invalid Ellipses.")

                # Take into account numerical values
                if operand_shapes[num] == ():
                    ellipse_count = 0
                else:
                    ellipse_count = max(len(operand_shapes[num]), 1) - (len(sub) - 3)

                if ellipse_count > longest:
                    longest = ellipse_count

                if ellipse_count < 0:
                    raise ValueError("Ellipses lengths do not match.")
                elif ellipse_count == 0:
                    split_subscripts[num] = sub.replace("...", "")
                else:
                    split_subscripts[num] = sub.replace("...", ellipse_inds[-ellipse_count:])

        subscripts = ",".join(split_subscripts)

        # Figure out output ellipses
        if longest == 0:
            out_ellipse = ""
        else:
            out_ellipse = ellipse_inds[-longest:]

        if out_sub:
            subscripts += "->" + output_sub.replace("...", out_ellipse)
        else:
            # Special care for outputless ellipses
            output_subscript = find_output_str(subscripts)
            normal_inds = "".join(sorted(set(output_subscript) - set(out_ellipse)))

            subscripts += "->" + out_ellipse + normal_inds

    # Build output string if does not exist
    if "->" in subscripts:
        input_subscripts, output_subscript = subscripts.split("->")
    else:
        input_subscripts, output_subscript = subscripts, find_output_str(subscripts)

    # Make sure output subscripts are unique and in the input
    for char in output_subscript:
        if output_subscript.count(char) != 1:
            raise ValueError(f"Output character '{char}' appeared more than once in the output.")
        if char not in input_subscripts:
            raise ValueError(f"Output character '{char}' did not appear in the input")

    # Make sure number operands is equivalent to the number of terms
    if len(input_subscripts.split(",")) != len(operands):
        raise ValueError(
            f"Number of einsum subscripts, {len(input_subscripts.split(','))}, must be equal to the "
            f"number of operands, {len(operands)}."
        )

    return input_subscripts, output_subscript, operands
