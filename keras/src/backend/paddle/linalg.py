import paddle

from keras.src.backend.paddle.core import convert_to_tensor


def cholesky(a):
    return paddle.linalg.cholesky(convert_to_tensor(a))


def cholesky_inverse(x, upper=False):
    x = convert_to_tensor(x)
    if upper:
        x = x.transpose(*range(x.ndim - 2), x.ndim - 1, x.ndim - 2)
    return paddle.linalg.cholesky_inverse(x)


def det(a):
    return paddle.linalg.det(convert_to_tensor(a))


def eig(a):
    return paddle.linalg.eig(convert_to_tensor(a))


def eigh(a):
    w, v = paddle.linalg.eigh(convert_to_tensor(a))
    return w, v


def inv(a):
    return paddle.linalg.inv(convert_to_tensor(a))


def lu_factor(x):
    x = convert_to_tensor(x)
    LU, pivots = paddle.linalg.lu(x)
    return LU, pivots


def matrix_rank(x, tol=None):
    x = convert_to_tensor(x)
    if tol is not None:
        return paddle.linalg.matrix_rank(x, tol=tol)
    return paddle.linalg.matrix_rank(x)


def norm(x, ord=None, axis=None, keepdims=False):
    return paddle.linalg.norm(
        convert_to_tensor(x), p=ord, axis=axis, keepdim=keepdims
    )


def pinv(x, rcond=None):
    x = convert_to_tensor(x)
    if rcond is not None:
        return paddle.linalg.pinv(x, rcond=rcond)
    return paddle.linalg.pinv(x)


def qr(x, mode="reduced"):
    if mode not in {"reduced", "complete"}:
        raise ValueError(
            "`mode` argument value not supported. "
            "Expected one of {'reduced', 'complete'}. "
            f"Received: mode={mode}"
        )
    return paddle.linalg.qr(convert_to_tensor(x), mode=mode)


def solve(a, b):
    return paddle.linalg.solve(convert_to_tensor(a), convert_to_tensor(b))


def solve_triangular(a, b, lower=False):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    if b.ndim == a.ndim - 1:
        b = paddle.unsqueeze(b, axis=-1)
        return paddle.linalg.triangular_solve(a, b, upper=not lower).squeeze(
            axis=-1
        )
    return paddle.linalg.triangular_solve(a, b, upper=not lower)


def svd(x, full_matrices=True, compute_uv=True):
    x = convert_to_tensor(x)
    if not compute_uv:
        return paddle.linalg.svdvals(x)
    return paddle.linalg.svd(x, full_matrices=full_matrices)


def lstsq(a, b, rcond=None):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    result = paddle.linalg.lstsq(a, b, rcond=rcond)
    return result[0]


def jvp(fun, primals, tangents, has_aux=False):
    eps = 1e-5
    if has_aux:
        out_pos, aux = fun(*[p + eps * t for p, t in zip(primals, tangents)])
        out_neg, _ = fun(*[p - eps * t for p, t in zip(primals, tangents)])
        jvp_out = (out_pos - out_neg) / (2 * eps)
        return jvp_out, aux
    out_pos = fun(*[p + eps * t for p, t in zip(primals, tangents)])
    out_neg = fun(*[p - eps * t for p, t in zip(primals, tangents)])
    return (out_pos - out_neg) / (2 * eps)
