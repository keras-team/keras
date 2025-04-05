"""
Directly tests various parser utility functions.
"""

from typing import Any, Tuple

import pytest

from opt_einsum.parser import get_shape, get_symbol, parse_einsum_input
from opt_einsum.testing import build_arrays_from_tuples


def test_get_symbol() -> None:
    assert get_symbol(2) == "c"
    assert get_symbol(200000) == "\U00031540"
    # Ensure we skip surrogates '[\uD800-\uDFFF]'
    assert get_symbol(55295) == "\ud88b"
    assert get_symbol(55296) == "\ue000"
    assert get_symbol(57343) == "\ue7ff"


def test_parse_einsum_input() -> None:
    eq = "ab,bc,cd"
    ops = build_arrays_from_tuples([(2, 3), (3, 4), (4, 5)])
    input_subscripts, output_subscript, operands = parse_einsum_input([eq, *ops])
    assert input_subscripts == eq
    assert output_subscript == "ad"
    assert operands == ops


def test_parse_einsum_input_shapes_error() -> None:
    eq = "ab,bc,cd"
    ops = build_arrays_from_tuples([(2, 3), (3, 4), (4, 5)])

    with pytest.raises(ValueError):
        _ = parse_einsum_input([eq, *ops], shapes=True)


def test_parse_einsum_input_shapes() -> None:
    eq = "ab,bc,cd"
    shapes = [(2, 3), (3, 4), (4, 5)]
    input_subscripts, output_subscript, operands = parse_einsum_input([eq, *shapes], shapes=True)
    assert input_subscripts == eq
    assert output_subscript == "ad"
    assert shapes == operands


def test_parse_with_ellisis() -> None:
    eq = "...a,ab"
    shapes = [(2, 3), (3, 4)]
    input_subscripts, output_subscript, operands = parse_einsum_input([eq, *shapes], shapes=True)
    assert input_subscripts == "da,ab"
    assert output_subscript == "db"
    assert shapes == operands


@pytest.mark.parametrize(
    "array, shape",
    [
        [[5], (1,)],
        [[5, 5], (2,)],
        [(5, 5), (2,)],
        [[[[[[5, 2]]]]], (1, 1, 1, 1, 2)],
        [[[[[["abcdef", "b"]]]]], (1, 1, 1, 1, 2)],
        ["A", ()],
        [b"A", ()],
        [True, ()],
        [5, ()],
        [5.0, ()],
        [5.0 + 0j, ()],
    ],
)
def test_get_shapes(array: Any, shape: Tuple[int]) -> None:
    assert get_shape(array) == shape
