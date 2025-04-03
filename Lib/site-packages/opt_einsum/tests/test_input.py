"""
Tests the input parsing for opt_einsum. Duplicates the np.einsum input tests.
"""

from typing import Any, List

import pytest

from opt_einsum import contract, contract_path
from opt_einsum.typing import ArrayType

np = pytest.importorskip("numpy")


def build_views(string: str) -> List[ArrayType]:
    """Builds random numpy arrays for testing by using a fixed size dictionary and an input string."""

    chars = "abcdefghij"
    sizes_array = np.array([2, 3, 4, 5, 4, 3, 2, 6, 5, 4])
    sizes = dict(zip(chars, sizes_array))

    views = []

    string = string.replace("...", "ij")

    terms = string.split("->")[0].split(",")
    for term in terms:
        dims = [sizes[x] for x in term]
        views.append(np.random.rand(*dims))
    return views


def test_type_errors() -> None:
    # subscripts must be a string
    with pytest.raises(TypeError):
        contract(0, 0)

    # out parameter must be an array
    with pytest.raises(TypeError):
        contract("", 0, out="test")

    # order parameter must be a valid order
    # changed in Numpy 1.19, see https://github.com/numpy/numpy/commit/35b0a051c19265f5643f6011ee11e31d30c8bc4c
    with pytest.raises((TypeError, ValueError)):
        contract("", 0, order="W")  # type: ignore

    # casting parameter must be a valid casting
    with pytest.raises(ValueError):
        contract("", 0, casting="blah")  # type: ignore

    # dtype parameter must be a valid dtype
    with pytest.raises(TypeError):
        contract("", 0, dtype="bad_data_type")

    # other keyword arguments are rejected
    with pytest.raises(TypeError):
        contract("", 0, bad_arg=0)

    # issue 4528 revealed a segfault with this call
    with pytest.raises(TypeError):
        contract(*(None,) * 63)

    # Cannot have two ->
    with pytest.raises(ValueError):
        contract("->,->", 0, 5)

    # Undefined symbol lhs
    with pytest.raises(ValueError):
        contract("&,a->", 0, 5)

    # Undefined symbol rhs
    with pytest.raises(ValueError):
        contract("a,a->&", 0, 5)

    with pytest.raises(ValueError):
        contract("a,a->&", 0, 5)

    # Catch ellipsis errors
    string = "...a->...a"
    views = build_views(string)

    # Subscript list must contain Ellipsis or (hashable && comparable) object
    with pytest.raises(TypeError):
        contract(views[0], [Ellipsis, 0], [Ellipsis, ["a"]])

    with pytest.raises(TypeError):
        contract(views[0], [Ellipsis, {}], [Ellipsis, "a"])


@pytest.mark.parametrize("contract_fn", [contract, contract_path])
def test_value_errors(contract_fn: Any) -> None:
    with pytest.raises(ValueError):
        contract_fn("")

    # subscripts must be a string
    with pytest.raises(TypeError):
        contract_fn(0, 0)

    # invalid subscript character
    with pytest.raises(ValueError):
        contract_fn("i%...", [0, 0])
    with pytest.raises(ValueError):
        contract_fn("...j$", [0, 0])
    with pytest.raises(ValueError):
        contract_fn("i->&", [0, 0])

    with pytest.raises(ValueError):
        contract_fn("")
    # number of operands must match count in subscripts string
    with pytest.raises(ValueError):
        contract_fn("", 0, 0)
    with pytest.raises(ValueError):
        contract_fn(",", 0, [0], [0])
    with pytest.raises(ValueError):
        contract_fn(",", [0])

    # can't have more subscripts than dimensions in the operand
    with pytest.raises(ValueError):
        contract_fn("i", 0)
    with pytest.raises(ValueError):
        contract_fn("ij", [0, 0])
    with pytest.raises(ValueError):
        contract_fn("...i", 0)
    with pytest.raises(ValueError):
        contract_fn("i...j", [0, 0])
    with pytest.raises(ValueError):
        contract_fn("i...", 0)
    with pytest.raises(ValueError):
        contract_fn("ij...", [0, 0])

    # invalid ellipsis
    with pytest.raises(ValueError):
        contract_fn("i..", [0, 0])
    with pytest.raises(ValueError):
        contract_fn(".i...", [0, 0])
    with pytest.raises(ValueError):
        contract_fn("j->..j", [0, 0])
    with pytest.raises(ValueError):
        contract_fn("j->.j...", [0, 0])

    # invalid subscript character
    with pytest.raises(ValueError):
        contract_fn("i%...", [0, 0])
    with pytest.raises(ValueError):
        contract_fn("...j$", [0, 0])
    with pytest.raises(ValueError):
        contract_fn("i->&", [0, 0])

    # output subscripts must appear in input
    with pytest.raises(ValueError):
        contract_fn("i->ij", [0, 0])

    # output subscripts may only be specified once
    with pytest.raises(ValueError):
        contract_fn("ij->jij", [[0, 0], [0, 0]])

    # dimensions much match when being collapsed
    with pytest.raises(ValueError):
        contract_fn("ii", np.arange(6).reshape(2, 3))
    with pytest.raises(ValueError):
        contract_fn("ii->i", np.arange(6).reshape(2, 3))

    # broadcasting to new dimensions must be enabled explicitly
    with pytest.raises(ValueError):
        contract_fn("i", np.arange(6).reshape(2, 3))

    with pytest.raises(TypeError):
        contract_fn("ij->ij", [[0, 1], [0, 1]], bad_kwarg=True)


@pytest.mark.parametrize(
    "string",
    [
        # Ellipse
        "...a->...",
        "a...->...",
        "a...a->...a",
        "...,...",
        "a,b",
        "...a,...b",
    ],
)
def test_compare(string: str) -> None:
    views = build_views(string)

    ein = contract(string, *views, optimize=False)
    opt = contract(string, *views)
    assert np.allclose(ein, opt)

    opt = contract(string, *views, optimize="optimal")
    assert np.allclose(ein, opt)


def test_ellipse_input1() -> None:
    string = "...a->..."
    views = build_views(string)

    ein = contract(string, *views, optimize=False)
    opt = contract(views[0], [Ellipsis, 0], [Ellipsis])
    assert np.allclose(ein, opt)


def test_ellipse_input2() -> None:
    string = "...a"
    views = build_views(string)

    ein = contract(string, *views, optimize=False)
    opt = contract(views[0], [Ellipsis, 0])
    assert np.allclose(ein, opt)


def test_ellipse_input3() -> None:
    string = "...a->...a"
    views = build_views(string)

    ein = contract(string, *views, optimize=False)
    opt = contract(views[0], [Ellipsis, 0], [Ellipsis, 0])
    assert np.allclose(ein, opt)


def test_ellipse_input4() -> None:
    string = "...b,...a->..."
    views = build_views(string)

    ein = contract(string, *views, optimize=False)
    opt = contract(views[0], [Ellipsis, 1], views[1], [Ellipsis, 0], [Ellipsis])
    assert np.allclose(ein, opt)


def test_singleton_dimension_broadcast() -> None:
    # singleton dimensions broadcast (gh-10343)
    p = np.ones((10, 2))
    q = np.ones((1, 2))

    ein = contract("ij,ij->j", p, q, optimize=False)
    opt = contract("ij,ij->j", p, q, optimize=True)
    assert np.allclose(ein, opt)
    assert np.allclose(opt, [10.0, 10.0])

    p = np.ones((1, 5))
    q = np.ones((5, 5))

    for optimize in (True, False):
        res1 = (contract("...ij,...jk->...ik", p, p, optimize=optimize),)
        res2 = contract("...ij,...jk->...ik", p, q, optimize=optimize)
        assert np.allclose(res1, res2)
        assert np.allclose(res2, np.full((1, 5), 5))


def test_large_int_input_format() -> None:
    string = "ab,bc,cd"
    x, y, z = build_views(string)
    string_output = contract(string, x, y, z)
    int_output = contract(x, (1000, 1001), y, (1001, 1002), z, (1002, 1003))
    assert np.allclose(string_output, int_output)
    for i in range(10):
        transpose_output = contract(x, (i + 1, i))
        assert np.allclose(transpose_output, x.T)


def test_hashable_object_input_format() -> None:
    string = "ab,bc,cd"
    x, y, z = build_views(string)
    string_output = contract(string, x, y, z)
    hash_output1 = contract(x, ("left", "bond1"), y, ("bond1", "bond2"), z, ("bond2", "right"))
    hash_output2 = contract(
        x,
        ("left", "bond1"),
        y,
        ("bond1", "bond2"),
        z,
        ("bond2", "right"),
        ("left", "right"),
    )
    assert np.allclose(string_output, hash_output1)
    assert np.allclose(hash_output1, hash_output2)
    for i in range(1, 10):
        transpose_output = contract(x, ("b" * i, "a" * i))
        assert np.allclose(transpose_output, x.T)
