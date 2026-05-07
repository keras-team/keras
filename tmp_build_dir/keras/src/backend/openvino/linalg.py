def cholesky(a, upper=False):
    raise NotImplementedError(
        "`cholesky` is not supported with openvino backend."
    )


def cholesky_inverse(a, upper=False):
    raise NotImplementedError(
        "`cholesky_inverse` is not supported with openvino backend."
    )


def det(a):
    raise NotImplementedError("`det` is not supported with openvino backend")


def eig(a):
    raise NotImplementedError("`eig` is not supported with openvino backend")


def eigh(a):
    raise NotImplementedError("`eigh` is not supported with openvino backend")


def inv(a):
    raise NotImplementedError("`inv` is not supported with openvino backend")


def lu_factor(a):
    raise NotImplementedError(
        "`lu_factor` is not supported with openvino backend"
    )


def norm(x, ord=None, axis=None, keepdims=False):
    raise NotImplementedError("`norm` is not supported with openvino backend")


def qr(x, mode="reduced"):
    raise NotImplementedError("`qr` is not supported with openvino backend")


def solve(a, b):
    raise NotImplementedError("`solve` is not supported with openvino backend")


def solve_triangular(a, b, lower=False):
    raise NotImplementedError(
        "`solve_triangular` is not supported with openvino backend"
    )


def svd(x, full_matrices=True, compute_uv=True):
    raise NotImplementedError("`svd` is not supported with openvino backend")


def lstsq(a, b, rcond=None):
    raise NotImplementedError("`lstsq` is not supported with openvino backend")


def jvp(fun, primals, tangents, has_aux=False):
    raise NotImplementedError("`jvp` is not supported with openvino backend")
