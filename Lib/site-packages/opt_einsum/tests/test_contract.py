"""
Tets a series of opt_einsum contraction paths to ensure the results are the same for different paths
"""

from typing import Any, List

import pytest

from opt_einsum import contract, contract_expression, contract_path
from opt_einsum.paths import _PATH_OPTIONS, linear_to_ssa, ssa_to_linear
from opt_einsum.testing import build_views, rand_equation
from opt_einsum.typing import OptimizeKind

# NumPy is required for the majority of this file
np = pytest.importorskip("numpy")


tests = [
    # Test scalar-like operations
    "a,->a",
    "ab,->ab",
    ",ab,->ab",
    ",,->",
    # Test hadamard-like products
    "a,ab,abc->abc",
    "a,b,ab->ab",
    # Test index-transformations
    "ea,fb,gc,hd,abcd->efgh",
    "ea,fb,abcd,gc,hd->efgh",
    "abcd,ea,fb,gc,hd->efgh",
    # Test complex contractions
    "acdf,jbje,gihb,hfac,gfac,gifabc,hfac",
    "acdf,jbje,gihb,hfac,gfac,gifabc,hfac",
    "cd,bdhe,aidb,hgca,gc,hgibcd,hgac",
    "abhe,hidj,jgba,hiab,gab",
    "bde,cdh,agdb,hica,ibd,hgicd,hiac",
    "chd,bde,agbc,hiad,hgc,hgi,hiad",
    "chd,bde,agbc,hiad,bdi,cgh,agdb",
    "bdhe,acad,hiab,agac,hibd",
    # Test collapse
    "ab,ab,c->",
    "ab,ab,c->c",
    "ab,ab,cd,cd->",
    "ab,ab,cd,cd->ac",
    "ab,ab,cd,cd->cd",
    "ab,ab,cd,cd,ef,ef->",
    # Test outer prodcuts
    "ab,cd,ef->abcdef",
    "ab,cd,ef->acdf",
    "ab,cd,de->abcde",
    "ab,cd,de->be",
    "ab,bcd,cd->abcd",
    "ab,bcd,cd->abd",
    # Random test cases that have previously failed
    "eb,cb,fb->cef",
    "dd,fb,be,cdb->cef",
    "bca,cdb,dbf,afc->",
    "dcc,fce,ea,dbf->ab",
    "fdf,cdd,ccd,afe->ae",
    "abcd,ad",
    "ed,fcd,ff,bcf->be",
    "baa,dcf,af,cde->be",
    "bd,db,eac->ace",
    "fff,fae,bef,def->abd",
    "efc,dbc,acf,fd->abe",
    # Inner products
    "ab,ab",
    "ab,ba",
    "abc,abc",
    "abc,bac",
    "abc,cba",
    # GEMM test cases
    "ab,bc",
    "ab,cb",
    "ba,bc",
    "ba,cb",
    "abcd,cd",
    "abcd,ab",
    "abcd,cdef",
    "abcd,cdef->feba",
    "abcd,efdc",
    # Inner than dot
    "aab,bc->ac",
    "ab,bcc->ac",
    "aab,bcc->ac",
    "baa,bcc->ac",
    "aab,ccb->ac",
    # Randomly build test caes
    "aab,fa,df,ecc->bde",
    "ecb,fef,bad,ed->ac",
    "bcf,bbb,fbf,fc->",
    "bb,ff,be->e",
    "bcb,bb,fc,fff->",
    "fbb,dfd,fc,fc->",
    "afd,ba,cc,dc->bf",
    "adb,bc,fa,cfc->d",
    "bbd,bda,fc,db->acf",
    "dba,ead,cad->bce",
    "aef,fbc,dca->bde",
]


@pytest.mark.parametrize("optimize", (True, False, None))
def test_contract_plain_types(optimize: OptimizeKind) -> None:
    expr = "ij,jk,kl->il"
    ops = [np.random.rand(2, 2), np.random.rand(2, 2), np.random.rand(2, 2)]

    path = contract_path(expr, *ops, optimize=optimize)
    assert len(path) == 2

    result = contract(expr, *ops, optimize=optimize)
    assert result.shape == (2, 2)


@pytest.mark.parametrize("string", tests)
@pytest.mark.parametrize("optimize", _PATH_OPTIONS)
def test_compare(optimize: OptimizeKind, string: str) -> None:
    views = build_views(string)

    ein = contract(string, *views, optimize=False, use_blas=False)
    opt = contract(string, *views, optimize=optimize, use_blas=False)
    assert np.allclose(ein, opt)


@pytest.mark.parametrize("string", tests)
def test_drop_in_replacement(string: str) -> None:
    views = build_views(string)
    opt = contract(string, *views)
    assert np.allclose(opt, np.einsum(string, *views))


@pytest.mark.parametrize("string", tests)
@pytest.mark.parametrize("optimize", _PATH_OPTIONS)
def test_compare_greek(optimize: OptimizeKind, string: str) -> None:
    views = build_views(string)

    ein = contract(string, *views, optimize=False, use_blas=False)

    # convert to greek
    string = "".join(chr(ord(c) + 848) if c not in ",->." else c for c in string)

    opt = contract(string, *views, optimize=optimize, use_blas=False)
    assert np.allclose(ein, opt)


@pytest.mark.parametrize("string", tests)
@pytest.mark.parametrize("optimize", _PATH_OPTIONS)
def test_compare_blas(optimize: OptimizeKind, string: str) -> None:
    views = build_views(string)

    ein = contract(string, *views, optimize=False)
    opt = contract(string, *views, optimize=optimize)
    assert np.allclose(ein, opt)


@pytest.mark.parametrize("string", tests)
@pytest.mark.parametrize("optimize", _PATH_OPTIONS)
def test_compare_blas_greek(optimize: OptimizeKind, string: str) -> None:
    views = build_views(string)

    ein = contract(string, *views, optimize=False)

    # convert to greek
    string = "".join(chr(ord(c) + 848) if c not in ",->." else c for c in string)

    opt = contract(string, *views, optimize=optimize)
    assert np.allclose(ein, opt)


def test_some_non_alphabet_maintains_order() -> None:
    # 'c beta a' should automatically go to -> 'a c beta'
    string = "c" + chr(ord("b") + 848) + "a"
    # but beta will be temporarily replaced with 'b' for which 'cba->abc'
    # so check manual output kicks in:
    x = np.random.rand(2, 3, 4)
    assert np.allclose(contract(string, x), contract("cxa", x))


def test_printing():
    string = "bbd,bda,fc,db->acf"
    views = build_views(string)

    ein = contract_path(string, *views)
    assert len(str(ein[1])) == 728


@pytest.mark.parametrize("string", tests)
@pytest.mark.parametrize("optimize", _PATH_OPTIONS)
@pytest.mark.parametrize("use_blas", [False, True])
@pytest.mark.parametrize("out_spec", [False, True])
def test_contract_expressions(string: str, optimize: OptimizeKind, use_blas: bool, out_spec: bool) -> None:
    views = build_views(string)
    shapes = [view.shape if hasattr(view, "shape") else () for view in views]
    expected = contract(string, *views, optimize=False, use_blas=False)

    expr = contract_expression(string, *shapes, optimize=optimize, use_blas=use_blas)

    if out_spec and ("->" in string) and (string[-2:] != "->"):
        (out,) = build_views(string.split("->")[1])
        expr(*views, out=out)
    else:
        out = expr(*views)

    assert np.allclose(out, expected)

    # check representations
    assert string in expr.__repr__()
    assert string in expr.__str__()


def test_contract_expression_interleaved_input() -> None:
    x, y, z = (np.random.randn(2, 2) for _ in "xyz")
    expected = np.einsum(x, [0, 1], y, [1, 2], z, [2, 3], [3, 0])
    xshp, yshp, zshp = ((2, 2) for _ in "xyz")
    expr = contract_expression(xshp, [0, 1], yshp, [1, 2], zshp, [2, 3], [3, 0])
    out = expr(x, y, z)
    assert np.allclose(out, expected)


@pytest.mark.parametrize(
    "string,constants",
    [
        ("hbc,bdef,cdkj,ji,ikeh,lfo", [1, 2, 3, 4]),
        ("bdef,cdkj,ji,ikeh,hbc,lfo", [0, 1, 2, 3]),
        ("hbc,bdef,cdkj,ji,ikeh,lfo", [1, 2, 3, 4]),
        ("hbc,bdef,cdkj,ji,ikeh,lfo", [1, 2, 3, 4]),
        ("ijab,acd,bce,df,ef->ji", [1, 2, 3, 4]),
        ("ab,cd,ad,cb", [1, 3]),
        ("ab,bc,cd", [0, 1]),
    ],
)
def test_contract_expression_with_constants(string: str, constants: List[int]) -> None:
    views = build_views(string)
    expected = contract(string, *views, optimize=False, use_blas=False)

    shapes = [view.shape if hasattr(view, "shape") else () for view in views]

    expr_args: List[Any] = []
    ctrc_args = []
    for i, (shape, view) in enumerate(zip(shapes, views)):
        if i in constants:
            expr_args.append(view)
        else:
            expr_args.append(shape)
            ctrc_args.append(view)

    expr = contract_expression(string, *expr_args, constants=constants)
    out = expr(*ctrc_args)
    assert np.allclose(expected, out)


@pytest.mark.parametrize("optimize", ["greedy", "optimal"])
@pytest.mark.parametrize("n", [4, 5])
@pytest.mark.parametrize("reg", [2, 3])
@pytest.mark.parametrize("n_out", [0, 2, 4])
@pytest.mark.parametrize("global_dim", [False, True])
def test_rand_equation(optimize: OptimizeKind, n: int, reg: int, n_out: int, global_dim: bool) -> None:
    eq, _, size_dict = rand_equation(n, reg, n_out, d_min=2, d_max=5, seed=42, return_size_dict=True)
    views = build_views(eq, size_dict)

    expected = contract(eq, *views, optimize=False)
    actual = contract(eq, *views, optimize=optimize)

    assert np.allclose(expected, actual)


@pytest.mark.parametrize("equation", tests)
def test_linear_vs_ssa(equation: str) -> None:
    views = build_views(equation)
    linear_path, _ = contract_path(equation, *views)
    ssa_path = linear_to_ssa(linear_path)
    linear_path2 = ssa_to_linear(ssa_path)
    assert linear_path2 == linear_path


def test_contract_path_supply_shapes() -> None:
    eq = "ab,bc,cd"
    shps = [(2, 3), (3, 4), (4, 5)]
    contract_path(eq, *shps, shapes=True)
