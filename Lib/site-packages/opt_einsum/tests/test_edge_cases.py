"""
Tets a series of opt_einsum contraction paths to ensure the results are the same for different paths
"""

from typing import Any, Tuple

import pytest

from opt_einsum import contract, contract_expression, contract_path
from opt_einsum.typing import PathType

# NumPy is required for the majority of this file
np = pytest.importorskip("numpy")


def test_contract_expression_checks() -> None:
    # check optimize needed
    with pytest.raises(ValueError):
        contract_expression("ab,bc->ac", (2, 3), (3, 4), optimize=False)

    # check sizes are still checked
    with pytest.raises(ValueError):
        contract_expression("ab,bc->ac", (2, 3), (3, 4), (42, 42))

    # check if out given
    out = np.empty((2, 4))
    with pytest.raises(ValueError):
        contract_expression("ab,bc->ac", (2, 3), (3, 4), out=out)

    # check still get errors when wrong ranks supplied to expression
    expr = contract_expression("ab,bc->ac", (2, 3), (3, 4))

    # too few arguments
    with pytest.raises(ValueError) as err:
        expr(np.random.rand(2, 3))
    assert "`ContractExpression` takes exactly 2" in str(err.value)

    # too many arguments
    with pytest.raises(ValueError) as err:
        expr(np.random.rand(2, 3), np.random.rand(2, 3), np.random.rand(2, 3))
    assert "`ContractExpression` takes exactly 2" in str(err.value)

    # wrong shapes
    with pytest.raises(ValueError) as err:
        expr(np.random.rand(2, 3, 4), np.random.rand(3, 4))
    assert "Internal error while evaluating `ContractExpression`" in str(err.value)
    with pytest.raises(ValueError) as err:
        expr(np.random.rand(2, 4), np.random.rand(3, 4, 5))
    assert "Internal error while evaluating `ContractExpression`" in str(err.value)
    with pytest.raises(ValueError) as err:
        expr(np.random.rand(2, 3), np.random.rand(3, 4), out=np.random.rand(2, 4, 6))
    assert "Internal error while evaluating `ContractExpression`" in str(err.value)

    # should only be able to specify out
    with pytest.raises(TypeError) as err_type:
        expr(np.random.rand(2, 3), np.random.rand(3, 4), order="F")  # type: ignore
    assert "got an unexpected keyword" in str(err_type.value)


def test_broadcasting_contraction() -> None:
    a = np.random.rand(1, 5, 4)
    b = np.random.rand(4, 6)
    c = np.random.rand(5, 6)
    d = np.random.rand(10)

    ein_scalar = contract("ijk,kl,jl", a, b, c, optimize=False)
    opt_scalar = contract("ijk,kl,jl", a, b, c, optimize=True)
    assert np.allclose(ein_scalar, opt_scalar)

    result = ein_scalar * d

    ein = contract("ijk,kl,jl,i->i", a, b, c, d, optimize=False)
    opt = contract("ijk,kl,jl,i->i", a, b, c, d, optimize=True)

    assert np.allclose(ein, result)
    assert np.allclose(opt, result)


def test_broadcasting_contraction2() -> None:
    a = np.random.rand(1, 1, 5, 4)
    b = np.random.rand(4, 6)
    c = np.random.rand(5, 6)
    d = np.random.rand(7, 7)

    ein_scalar = contract("abjk,kl,jl", a, b, c, optimize=False)
    opt_scalar = contract("abjk,kl,jl", a, b, c, optimize=True)
    assert np.allclose(ein_scalar, opt_scalar)

    result = ein_scalar * d

    ein = contract("abjk,kl,jl,ab->ab", a, b, c, d, optimize=False)
    opt = contract("abjk,kl,jl,ab->ab", a, b, c, d, optimize=True)

    assert np.allclose(ein, result)
    assert np.allclose(opt, result)


def test_broadcasting_contraction3() -> None:
    a = np.random.rand(1, 5, 4)
    b = np.random.rand(4, 1, 6)
    c = np.random.rand(5, 6)
    d = np.random.rand(7, 7)

    ein = contract("ajk,kbl,jl,ab->ab", a, b, c, d, optimize=False)
    opt = contract("ajk,kbl,jl,ab->ab", a, b, c, d, optimize=True)

    assert np.allclose(ein, opt)


def test_broadcasting_contraction4() -> None:
    a = np.arange(64).reshape(2, 4, 8)
    ein = contract("obk,ijk->ioj", a, a, optimize=False)
    opt = contract("obk,ijk->ioj", a, a, optimize=True)

    assert np.allclose(ein, opt)


def test_can_blas_on_healed_broadcast_dimensions() -> None:
    expr = contract_expression("ab,bc,bd->acd", (5, 4), (1, 5), (4, 20))
    # first contraction involves broadcasting
    assert expr.contraction_list[0][2] == "bc,ab->bca"
    assert expr.contraction_list[0][-1] is False
    # but then is healed GEMM is usable
    assert expr.contraction_list[1][2] == "bca,bd->acd"
    assert expr.contraction_list[1][-1] == "GEMM"


def test_pathinfo_for_empty_contraction() -> None:
    eq = "->"
    arrays = (1.0,)
    path: PathType = []
    _, info = contract_path(eq, *arrays, optimize=path)
    # some info is built lazily, so check repr
    assert repr(info)
    assert info.largest_intermediate == 1


@pytest.mark.parametrize(
    "expression, operands",
    [
        [",,->", (5, 5.0, 2.0j)],
        ["ab,->", ([[5, 5], [2.0, 1]], 2.0j)],
        ["ab,bc->ac", ([[5, 5], [2.0, 1]], [[2.0, 1], [3.0, 4]])],
        ["ab,->", ([[5, 5], [2.0, 1]], True)],
    ],
)
def test_contract_with_assumed_shapes(expression: str, operands: Tuple[Any]) -> None:
    """Test that we can contract with assumed shapes, and that the output is correct. This is required as we need to infer intermediate shape sizes."""

    benchmark = np.einsum(expression, *operands)
    result = contract(expression, *operands, optimize=True)
    assert np.allclose(benchmark, result)
