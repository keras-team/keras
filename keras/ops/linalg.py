from keras import backend
from keras.api_export import keras_export
from keras.backend import KerasTensor
from keras.backend import any_symbolic_tensors
from keras.ops.operation import Operation


class LinalgError(ValueError):
    """Generic exception raised by linalg operations.

    Raised when a linear algebra-related condition prevents the correct
    execution of the operation.
    """


class Cholesky(Operation):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return _cholesky(x)

    def compute_output_spec(self, x):
        _assert_2d(x)
        _assert_square(x)
        return KerasTensor(x.shape, x.dtype)


@keras_export("keras.ops.linalg.cholesky")
def cholesky(x):
    """Computes the Cholesky decomposition of a positive semi-definite matrix.

    Args:
        x: A tensor or variable.

    Returns:
        A tensor.

    """
    if any_symbolic_tensors((x,)):
        return Cholesky().symbolic_call(x)
    return _cholesky(x)


def _cholesky(x):
    _assert_2d(x)
    _assert_square(x)
    return backend.linalg.cholesky(x)


class Det(Operation):

    def __init__(self):
        super().__init__()

    def call(self, x):
        return _det(x)

    def compute_output_spec(self, x):
        _assert_2d(x)
        _assert_square(x)
        return KerasTensor(x.shape[:-2], x.dtype)


@keras_export("keras.ops.linalg.det")
def det(x):
    """Computes the determinant of a square tensor.

    Args:
        x: Input tensor of shape (..., M, M)

    Returns:
        A tensor of shape (...,) as the determinant of `x`.

    """
    if any_symbolic_tensors((x,)):
        return Det().symbolic_call(x)
    return _det(x)


def _det(x):
    _assert_2d(x)
    _assert_square(x)
    return backend.linalg.det(x)


class Eig(Operation):

    def __init__(self):
        super().__init__()

    def call(self, x):
        return _eig(x)

    def compute_output_spec(self, x):
        _assert_2d(x)
        _assert_square(x)
        return (
            KerasTensor(x.shape[:-1], x.dtype),
            KerasTensor(x.shape, x.dtype),
        )


@keras_export("keras.ops.linalg.eig")
def eig(x):
    """Computes the eigenvalues and eigenvectors of a square matrix.

    Args:
        x: A tensor of shape (..., M, M).

    Returns:
        A tuple of two tensors: a tensor of shape (..., M) containing the eigenvalues
        and a tensor of shape (..., M, M) containing the eigenvectors.

    """
    if any_symbolic_tensors((x,)):
        return Eig().symbolic_call(x)
    return _eig(x)


def _eig(x):
    _assert_2d(x)
    _assert_square(x)
    return backend.linalg.eig(x)


class Inv(Operation):

    def __init__(self):
        super().__init__()

    def call(self, x):
        return _inv(x)

    def compute_output_spec(self, x):
        _assert_2d(x)
        _assert_square(x)
        return KerasTensor(x.shape, x.dtype)


@keras_export("keras.ops.linalg.inv")
def inv(x):
    """Computes the inverse of a square tensor.

    Args:
        x: Input tensor of shape (..., M, M).

    Returns:
        A tensor of shape (..., M, M) representing the inverse of `x`.

    """
    if any_symbolic_tensors((x,)):
        return Inv().symbolic_call(x)
    return _inv(x)


def _inv(x):
    _assert_2d(x)
    _assert_square(x)
    return backend.linalg.inv(x)


class LU(Operation):

    def __init__(self):
        super().__init__()

    def call(self, x):
        return _lu(x)

    def compute_output_spec(self, x):
        _assert_2d(x)
        _assert_square(x)
        return (
            KerasTensor(x.shape, x.dtype),
            KerasTensor(x.shape, x.dtype),
            KerasTensor(x.shape[:-1], x.dtype),
        )
    

@keras_export("keras.ops.linalg.lu")
def lu(x):
    """Computes the LU decomposition of a square matrix.

    Args:
        x: A tensor of shape (..., M, M).

    Returns:
        A tuple of three tensors: a tensor of shape (..., M, M) containing the
        lower triangular matrix, a tensor of shape (..., M, M) containing the
        upper triangular matrix and a tensor of shape (..., M) containing the
        permutation indices.

    """
    if any_symbolic_tensors((x,)):
        return LU().symbolic_call(x)
    return _lu(x)


def _lu(x):
    _assert_2d(x)
    _assert_square(x)
    return backend.linalg.lu(x)


class Solve(Operation):

    def __init__(self):
        super().__init__()

    def call(self, a, b):
        return _solve(a, b)

    def compute_output_spec(self, a, b):
        _assert_2d(a)
        _assert_square(a)
        _assert_1d(b)
        _assert_a_b_compat(a, b)
        return KerasTensor(b.shape, b.dtype)


@keras_export("keras.ops.linalg.solve")
def solve(a, b):
    """Solves a linear system of equations given by `a x = b`.

    Args:
        a: A tensor of shape (..., M, M) representing the coefficients matrix.
        b: A tensor of shape (..., M) or (..., M, K) represeting the right-hand side or "dependent variable" matrix.

    Returns:
        A tensor of shape (..., M) or (..., M, K) representing the solution of the
        linear system. Returned shape is identical to `b`.

    """
    if any_symbolic_tensors((a, b)):
        return Solve().symbolic_call(a, b)
    return _solve(a, b)


def _solve(a, b):
    _assert_2d(a)
    _assert_square(a)
    _assert_1d(b)
    _assert_a_b_compat(a, b)
    return backend.linalg.solve(a, b)


class SVD(Operation):
    
    def __init__(self):
        super().__init__()

    def call(self, x):
        return _svd(x)

    def compute_output_spec(self, x):
        _assert_2d(x)
        return (
            KerasTensor(x.shape, x.dtype),
            KerasTensor(x.shape, x.dtype),
            KerasTensor(x.shape, x.dtype),
        )
    

@keras_export("keras.ops.linalg.svd")
def svd(x):
    """Computes the singular value decomposition of a matrix.

    Args:
        x: A tensor of shape (..., M, N).

    Returns:
        A tuple of three tensors: a tensor of shape (..., M, M) containing the
        left singular vectors, a tensor of shape (..., M, N) containing the
        singular values and a tensor of shape (..., N, N) containing the
        right singular vectors.

    """
    if any_symbolic_tensors((x,)):
        return SVD().symbolic_call(x)
    return _svd(x)


def _svd(x):
    _assert_2d(x)
    return backend.linalg.svd(x)


def _assert_1d(*arrays):
    for a in arrays:
        if a.ndim < 1:
            raise LinalgError(
                f"{a.ndim}-dimensional array given. Array must be "
                "at least one-dimensional"
            )


def _assert_2d(*arrays):
    for a in arrays:
        if a.ndim < 2:
            raise LinalgError(
                f"{a.ndim}-dimensional array given. Array must be "
                "at least two-dimensional"
            )


def _assert_square(*arrays):
    for a in arrays:
        m, n = a.shape[-2:]
        if m != n:
            raise LinalgError("Last 2 dimensions of the array must be square")


def _assert_a_b_compat(a, b):
    if a.ndim == b.ndim:
        if a.shape[-2] != b.shape[-2]:
            raise LinalgError(
                f"Incompatible shapes between `a` {a.shape} and `b` {b.shape}"
            )
    elif a.ndim == b.ndim - 1:
        if a.shape[-1] != b.shape[-1]:
            raise LinalgError(
                f"Incompatible shapes between `a` {a.shape} and `b` {b.shape}"
            )
