"""
Tests the BLAS capability for the opt_einsum module.
"""

from typing import Any

import pytest

from opt_einsum import blas, contract

blas_tests = [
    # DOT
    ((["k", "k"], "", set("k")), "DOT"),  # DDOT
    ((["ijk", "ijk"], "", set("ijk")), "DOT"),  # DDOT
    # GEMV?
    # GEMM
    ((["ij", "jk"], "ik", set("j")), "GEMM"),  # GEMM N N
    ((["ijl", "jlk"], "ik", set("jl")), "GEMM"),  # GEMM N N Tensor
    ((["ij", "kj"], "ik", set("j")), "GEMM"),  # GEMM N T
    ((["ijl", "kjl"], "ik", set("jl")), "GEMM"),  # GEMM N T Tensor
    ((["ji", "jk"], "ik", set("j")), "GEMM"),  # GEMM T N
    ((["jli", "jlk"], "ik", set("jl")), "GEMM"),  # GEMM T N Tensor
    ((["ji", "kj"], "ik", set("j")), "GEMM"),  # GEMM T T
    ((["jli", "kjl"], "ik", set("jl")), "GEMM"),  # GEMM T T Tensor
    # GEMM with final transpose
    ((["ij", "jk"], "ki", set("j")), "GEMM"),  # GEMM N N
    ((["ijl", "jlk"], "ki", set("jl")), "GEMM"),  # GEMM N N Tensor
    ((["ij", "kj"], "ki", set("j")), "GEMM"),  # GEMM N T
    ((["ijl", "kjl"], "ki", set("jl")), "GEMM"),  # GEMM N T Tensor
    ((["ji", "jk"], "ki", set("j")), "GEMM"),  # GEMM T N
    ((["jli", "jlk"], "ki", set("jl")), "GEMM"),  # GEMM T N Tensor
    ((["ji", "kj"], "ki", set("j")), "GEMM"),  # GEMM T T
    ((["jli", "kjl"], "ki", set("jl")), "GEMM"),  # GEMM T T Tensor
    # Tensor Dot (requires copy), lets not deal with this for now
    ((["ilj", "jlk"], "ik", set("jl")), "TDOT"),  # FT GEMM N N Tensor
    ((["ijl", "ljk"], "ik", set("jl")), "TDOT"),  # ST GEMM N N Tensor
    ((["ilj", "kjl"], "ik", set("jl")), "TDOT"),  # FT GEMM N T Tensor
    ((["ijl", "klj"], "ik", set("jl")), "TDOT"),  # ST GEMM N T Tensor
    ((["lji", "jlk"], "ik", set("jl")), "TDOT"),  # FT GEMM T N Tensor
    ((["jli", "ljk"], "ik", set("jl")), "TDOT"),  # ST GEMM T N Tensor
    ((["lji", "jlk"], "ik", set("jl")), "TDOT"),  # FT GEMM T N Tensor
    ((["jli", "ljk"], "ik", set("jl")), "TDOT"),  # ST GEMM T N Tensor
    # Tensor Dot (requires copy), lets not deal with this for now with transpose
    ((["ilj", "jlk"], "ik", set("lj")), "TDOT"),  # FT GEMM N N Tensor
    ((["ijl", "ljk"], "ik", set("lj")), "TDOT"),  # ST GEMM N N Tensor
    ((["ilj", "kjl"], "ik", set("lj")), "TDOT"),  # FT GEMM N T Tensor
    ((["ijl", "klj"], "ik", set("lj")), "TDOT"),  # ST GEMM N T Tensor
    ((["lji", "jlk"], "ik", set("lj")), "TDOT"),  # FT GEMM T N Tensor
    ((["jli", "ljk"], "ik", set("lj")), "TDOT"),  # ST GEMM T N Tensor
    ((["lji", "jlk"], "ik", set("lj")), "TDOT"),  # FT GEMM T N Tensor
    ((["jli", "ljk"], "ik", set("lj")), "TDOT"),  # ST GEMM T N Tensor
    # Other
    ((["ijk", "ikj"], "", set("ijk")), "DOT/EINSUM"),  # Transpose DOT
    ((["i", "j"], "ij", set()), "OUTER/EINSUM"),  # Outer
    ((["ijk", "ik"], "j", set("ik")), "GEMV/EINSUM"),  # Matrix-vector
    ((["ijj", "jk"], "ik", set("j")), False),  # Double index
    ((["ijk", "j"], "ij", set()), False),  # Index sum 1
    ((["ij", "ij"], "ij", set()), False),  # Index sum 2
]


@pytest.mark.parametrize("inp,benchmark", blas_tests)
def test_can_blas(inp: Any, benchmark: bool) -> None:
    result = blas.can_blas(*inp)
    assert result == benchmark


def test_blas_out() -> None:
    np = pytest.importorskip("numpy")

    a = np.random.rand(4, 4)
    b = np.random.rand(4, 4)
    c = np.random.rand(4, 4)
    d = np.empty((4, 4))

    contract("ij,jk->ik", a, b, out=d)
    np.testing.assert_allclose(d, np.dot(a, b))
    assert np.allclose(d, np.dot(a, b))

    contract("ij,jk,kl->il", a, b, c, out=d)
    np.testing.assert_allclose(d, np.dot(a, b).dot(c))
