import openvino.opset15 as ov_opset
from openvino import Type

from keras.src.backend import config
from keras.src.backend import standardize_dtype
from keras.src.backend.common import dtypes
from keras.src.backend.openvino.core import OpenVINOKerasTensor
from keras.src.backend.openvino.core import cast
from keras.src.backend.openvino.core import convert_to_tensor
from keras.src.backend.openvino.core import get_ov_output


def cholesky(a, upper=False):
    raise NotImplementedError(
        "`cholesky` is not supported with openvino backend."
    )


def cholesky_inverse(a, upper=False):
    a = convert_to_tensor(a)
    a_ov = get_ov_output(a)
    if upper:
        # Reconstruct A = U^T @ U, then invert
        reconstructed_matrix = ov_opset.matmul(a_ov, a_ov, True, False).output(
            0
        )
    else:
        # Reconstruct A = L @ L^T, then invert
        reconstructed_matrix = ov_opset.matmul(a_ov, a_ov, False, True).output(
            0
        )
    result = ov_opset.inverse(reconstructed_matrix, adjoint=False).output(0)
    return OpenVINOKerasTensor(result)


def det(a):
    raise NotImplementedError("`det` is not supported with openvino backend")


def eig(a):
    raise NotImplementedError("`eig` is not supported with openvino backend")


def eigh(a):
    raise NotImplementedError("`eigh` is not supported with openvino backend")


def inv(a):
    a = convert_to_tensor(a)
    a_ov = get_ov_output(a)
    result = ov_opset.inverse(a_ov, adjoint=False).output(0)
    return OpenVINOKerasTensor(result)


def lu_factor(a):
    raise NotImplementedError(
        "`lu_factor` is not supported with openvino backend"
    )


def norm(x, ord=None, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    x_shape = tuple(x.shape)
    ndim = len(x_shape)

    if axis is None:
        axis = tuple(range(ndim))
    elif isinstance(axis, int):
        axis = (axis,)
    if any(a < -ndim or a >= ndim for a in axis):
        raise ValueError(
            "All `axis` values must be in the range [-ndim, ndim). "
            f"Received inputs with ndim={ndim}, while axis={axis}"
        )
    axis = axis[0] if len(axis) == 1 else axis
    num_axes = 1 if isinstance(axis, int) else len(axis)

    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)

    x_ov = get_ov_output(x)

    # Ref: jax.numpy.linalg.norm
    if num_axes == 1:
        if ord is None or ord == 2:
            # L2 norm: sqrt(sum(x * conj(x)))
            x_conj = x_ov
            x_sq = ov_opset.multiply(x_conj, x_conj).output(0)
            axis_for_const = list(axis) if isinstance(axis, tuple) else axis
            axis_const = ov_opset.constant(axis_for_const, Type.i32).output(0)
            norm_result = ov_opset.reduce_sum(
                x_sq, axis_const, keepdims
            ).output(0)
            norm_result = ov_opset.sqrt(norm_result).output(0)
        elif ord == float("inf"):
            axis_for_const = list(axis) if isinstance(axis, tuple) else axis
            axis_const = ov_opset.constant(axis_for_const, Type.i32).output(0)
            x_abs = ov_opset.abs(x_ov).output(0)
            norm_result = ov_opset.reduce_max(
                x_abs, axis_const, keepdims
            ).output(0)
        elif ord == float("-inf"):
            axis_for_const = list(axis) if isinstance(axis, tuple) else axis
            axis_const = ov_opset.constant(axis_for_const, Type.i32).output(0)
            x_abs = ov_opset.abs(x_ov).output(0)
            norm_result = ov_opset.reduce_min(
                x_abs, axis_const, keepdims
            ).output(0)
        elif ord == 0:
            # Count non-zero elements
            axis_for_const = list(axis) if isinstance(axis, tuple) else axis
            axis_const = ov_opset.constant(axis_for_const, Type.i32).output(0)
            zero = ov_opset.constant(0.0, Type.f32).output(0)
            not_equal = ov_opset.not_equal(x_ov, zero).output(0)
            not_equal_float = ov_opset.convert(not_equal, Type.f32).output(0)
            norm_result = ov_opset.reduce_sum(
                not_equal_float, axis_const, keepdims
            ).output(0)
        elif ord == 1:
            # L1 norm: sum(|x|)
            axis_for_const = list(axis) if isinstance(axis, tuple) else axis
            axis_const = ov_opset.constant(axis_for_const, Type.i32).output(0)
            x_abs = ov_opset.abs(x_ov).output(0)
            norm_result = ov_opset.reduce_sum(
                x_abs, axis_const, keepdims
            ).output(0)
        elif isinstance(ord, str):
            raise ValueError(
                f"Invalid `ord` argument for vector norm. Received: ord={ord}"
            )
        else:
            # p-norm: (sum(|x|^p))^(1/p)
            ord_tensor = convert_to_tensor(ord, dtype=dtype)
            ord_ov = get_ov_output(ord_tensor)
            axis_for_const = list(axis) if isinstance(axis, tuple) else axis
            axis_const = ov_opset.constant(axis_for_const, Type.i32).output(0)
            x_abs = ov_opset.abs(x_ov).output(0)
            x_pow = ov_opset.power(x_abs, ord_ov).output(0)
            sum_pow = ov_opset.reduce_sum(x_pow, axis_const, keepdims).output(0)
            one = convert_to_tensor(1.0, dtype=dtype)
            one_ov = get_ov_output(one)
            inv_ord = ov_opset.divide(one_ov, ord_ov).output(0)
            norm_result = ov_opset.power(sum_pow, inv_ord).output(0)

    elif num_axes == 2:
        row_axis, col_axis = axis[0], axis[1]
        row_axis = row_axis + ndim if row_axis < 0 else row_axis
        col_axis = col_axis + ndim if col_axis < 0 else col_axis

        if ord is None or ord == "fro":
            # Frobenius norm: sqrt(sum(x * conj(x)))
            x_sq = ov_opset.multiply(x_ov, x_ov).output(0)
            axis_for_const = list(axis) if isinstance(axis, tuple) else axis
            axis_const = ov_opset.constant(axis_for_const, Type.i32).output(0)
            sum_sq = ov_opset.reduce_sum(x_sq, axis_const, keepdims).output(0)
            norm_result = ov_opset.sqrt(sum_sq).output(0)
        elif ord == 1:
            # Maximum absolute column sum
            if not keepdims and col_axis > row_axis:
                col_axis -= 1
            row_axis_const = ov_opset.constant(row_axis, Type.i32).output(0)
            col_axis_const = ov_opset.constant(col_axis, Type.i32).output(0)

            x_abs = ov_opset.abs(x_ov).output(0)
            col_sum = ov_opset.reduce_sum(
                x_abs, row_axis_const, keep_dims=keepdims
            ).output(0)
            norm_result = ov_opset.reduce_max(
                col_sum, col_axis_const, keep_dims=keepdims
            ).output(0)
        elif ord == -1:
            # Minimum absolute column sum
            if not keepdims and col_axis > row_axis:
                col_axis -= 1
            row_axis_const = ov_opset.constant(row_axis, Type.i32).output(0)
            col_axis_const = ov_opset.constant(col_axis, Type.i32).output(0)

            x_abs = ov_opset.abs(x_ov).output(0)
            col_sum = ov_opset.reduce_sum(
                x_abs, row_axis_const, keep_dims=keepdims
            ).output(0)
            norm_result = ov_opset.reduce_min(
                col_sum, col_axis_const, keep_dims=keepdims
            ).output(0)
        elif ord == float("inf"):
            # Maximum absolute row sum
            if not keepdims and row_axis > col_axis:
                row_axis -= 1
            col_axis_const = ov_opset.constant(col_axis, Type.i32).output(0)
            row_axis_const = ov_opset.constant(row_axis, Type.i32).output(0)

            x_abs = ov_opset.abs(x_ov).output(0)
            row_sum = ov_opset.reduce_sum(
                x_abs, col_axis_const, keep_dims=keepdims
            ).output(0)
            norm_result = ov_opset.reduce_max(
                row_sum, row_axis_const, keep_dims=keepdims
            ).output(0)
        elif ord == float("-inf"):
            # Minimum absolute row sum
            if not keepdims and row_axis > col_axis:
                row_axis -= 1
            col_axis_const = ov_opset.constant(col_axis, Type.i32).output(0)
            row_axis_const = ov_opset.constant(row_axis, Type.i32).output(0)

            x_abs = ov_opset.abs(x_ov).output(0)
            row_sum = ov_opset.reduce_sum(
                x_abs, col_axis_const, keep_dims=keepdims
            ).output(0)
            norm_result = ov_opset.reduce_min(
                row_sum, row_axis_const, keep_dims=keepdims
            ).output(0)
        elif ord in ("nuc", 2, -2):
            # Nuclear norm, spectral norm, and minimum singular value
            # These require SVD which is not supported in OpenVINO backend
            raise NotImplementedError(
                f"`norm` with ord={ord} for matrix norms requires SVD "
                "which is not supported with openvino backend"
            )
        else:
            raise ValueError(
                f"Invalid `ord` argument for matrix norm. Received: ord={ord}"
            )
    else:
        raise ValueError(f"Invalid axis values. Received: axis={axis}")

    return OpenVINOKerasTensor(norm_result)


def qr(x, mode="reduced"):
    if mode not in {"reduced", "complete"}:
        raise ValueError(
            "`mode` argument value not supported. "
            "Expected one of {'reduced', 'complete'}. "
            f"Received: mode={mode}"
        )

    x = convert_to_tensor(x)
    x_ov = get_ov_output(x)

    shape = ov_opset.shape_of(x_ov).output(0)
    dtype = x_ov.get_element_type()
    zero_const = ov_opset.constant(0, Type.i64).output(0)
    one_const = ov_opset.constant(1, Type.i64).output(0)
    neg_1 = ov_opset.constant(-1, Type.i64).output(0)
    neg_2 = ov_opset.constant(-2, Type.i64).output(0)

    M = ov_opset.gather(shape, neg_2, zero_const).output(0)
    N = ov_opset.gather(shape, neg_1, zero_const).output(0)

    shape_len = ov_opset.shape_of(shape).output(0)
    batch_dims_len = ov_opset.subtract(
        shape_len, ov_opset.constant([2], Type.i64).output(0)
    ).output(0)
    start = ov_opset.constant([0], Type.i64).output(0)
    end = batch_dims_len
    step = ov_opset.constant([1], Type.i64).output(0)
    axes = ov_opset.constant([0], Type.i64).output(0)
    batch_dims = ov_opset.slice(shape, start, end, step, axes).output(0)

    # Calculate batch size carefully. If batch_dims is empty, it returns 1.
    batch_size = ov_opset.reduce_prod(batch_dims, zero_const, False).output(0)

    # To avoid OpenVINO CPU plugin bug with f64 in loop body, cast to f32 before loop
    loop_dtype = Type.f32 if dtype == Type.f64 else dtype
    x_ov_loop = ov_opset.convert(x_ov, loop_dtype).output(0) if dtype == Type.f64 else x_ov

    flat_shape = ov_opset.concat(
        [
            ov_opset.unsqueeze(batch_size, zero_const).output(0),
            ov_opset.unsqueeze(M, zero_const).output(0),
            ov_opset.unsqueeze(N, zero_const).output(0),
        ],
        0,
    ).output(0)
    A_flat = ov_opset.reshape(x_ov_loop, flat_shape, False).output(0)

    # Simplified eye creation logic
    range_m = ov_opset.range(
        zero_const, M, ov_opset.constant(1, Type.i64).output(0), Type.i64
    ).output(0)
    range_m_reshaped_h = ov_opset.unsqueeze(
        range_m, ov_opset.constant(1, Type.i64).output(0)
    ).output(0)
    range_m_reshaped_w = ov_opset.unsqueeze(
        range_m, ov_opset.constant(0, Type.i64).output(0)
    ).output(0)
    eq = ov_opset.equal(range_m_reshaped_h, range_m_reshaped_w).output(0)
    eye_mat = ov_opset.convert(eq, loop_dtype).output(0)

    eye_reshaped = ov_opset.unsqueeze(eye_mat, zero_const).output(0)
    Q_flat_shape = ov_opset.concat(
        [
            ov_opset.unsqueeze(batch_size, zero_const).output(0),
            ov_opset.unsqueeze(M, zero_const).output(0),
            ov_opset.unsqueeze(M, zero_const).output(0),
        ],
        0,
    ).output(0)
    Q_flat = ov_opset.broadcast(eye_reshaped, Q_flat_shape).output(0)

    K = ov_opset.minimum(M, N).output(0)

    loop = ov_opset.loop(K, ov_opset.constant(True, Type.boolean).output(0))
    k_body = ov_opset.parameter([], Type.i64)
    r_body = ov_opset.parameter([-1, -1, -1], loop_dtype)
    q_body = ov_opset.parameter([-1, -1, -1], loop_dtype)
    M_body = ov_opset.parameter([], Type.i64)

    k_idx = ov_opset.unsqueeze(k_body, zero_const).output(0)

    col = ov_opset.gather(
        r_body, k_idx, ov_opset.constant(2, Type.i64).output(0)
    ).output(0)
    col = ov_opset.squeeze(col, ov_opset.constant([2], Type.i64).output(0)).output(0)

    range_m_body = ov_opset.range(zero_const, M_body, one_const, Type.i64).output(0)
    mask = ov_opset.greater_equal(range_m_body, k_idx).output(0)
    mask_float = ov_opset.unsqueeze(
        ov_opset.convert(mask, loop_dtype).output(0), zero_const
    ).output(0)

    x_mask = ov_opset.multiply(col, mask_float).output(0)

    one_hot_k_float = ov_opset.unsqueeze(
        ov_opset.convert(ov_opset.equal(range_m_body, k_idx).output(0), loop_dtype).output(0),
        zero_const,
    ).output(0)

    x_k = ov_opset.reduce_sum(
        ov_opset.multiply(x_mask, one_hot_k_float).output(0),
        ov_opset.constant([-1], Type.i64).output(0),
        True,
    ).output(0)

    norm_x = ov_opset.sqrt(
        ov_opset.reduce_sum(
            ov_opset.multiply(x_mask, x_mask).output(0),
            ov_opset.constant([-1], Type.i64).output(0),
            True,
        ).output(0)
    ).output(0)

    s = ov_opset.sign(x_k).output(0)
    s = ov_opset.select(
        ov_opset.equal(s, ov_opset.constant(0, loop_dtype).output(0)).output(0),
        ov_opset.constant(1, loop_dtype).output(0),
        s,
    ).output(0)

    add_term = ov_opset.multiply(
        one_hot_k_float, ov_opset.multiply(s, norm_x).output(0)
    ).output(0)
    v = ov_opset.add(x_mask, add_term).output(0)

    norm_v = ov_opset.sqrt(
        ov_opset.reduce_sum(
            ov_opset.multiply(v, v).output(0),
            ov_opset.constant([-1], Type.i64).output(0),
            True,
        ).output(0)
    ).output(0)
    safe_norm_v = ov_opset.select(
        ov_opset.equal(norm_v, ov_opset.constant(0, loop_dtype).output(0)).output(0),
        ov_opset.constant(1, loop_dtype).output(0),
        norm_v,
    ).output(0)

    v = ov_opset.divide(v, safe_norm_v).output(0)
    v = ov_opset.unsqueeze(v, ov_opset.constant([-1], Type.i64).output(0)).output(0)
    v_t = ov_opset.transpose(
        v, ov_opset.constant([0, 2, 1], Type.i64).output(0)
    ).output(0)

    v_t_R = ov_opset.matmul(v_t, r_body, False, False).output(0)
    two_v_v_t_R = ov_opset.multiply(
        ov_opset.constant(2, loop_dtype).output(0),
        ov_opset.matmul(v, v_t_R, False, False).output(0),
    ).output(0)
    new_r = ov_opset.subtract(r_body, two_v_v_t_R).output(0)

    Q_v = ov_opset.matmul(q_body, v, False, False).output(0)
    two_Q_v_v_t = ov_opset.multiply(
        ov_opset.constant(2, loop_dtype).output(0),
        ov_opset.matmul(Q_v, v_t, False, False).output(0),
    ).output(0)
    new_q = ov_opset.subtract(q_body, two_Q_v_v_t).output(0)

    cond_out = ov_opset.constant(True, Type.boolean).output(0)
    from openvino import Model
    body = Model([cond_out, new_r, new_q], [k_body, r_body, q_body, M_body])
    loop.set_function(body)
    loop.set_special_body_ports([0, 0])

    loop.set_merged_input(r_body, A_flat, new_r)
    loop.set_merged_input(q_body, Q_flat, new_q)
    loop.set_invariant_input(M_body, M)

    res_r = loop.get_iter_value(new_r, -1)
    res_q = loop.get_iter_value(new_q, -1)

    if dtype == Type.f64:
        res_q = ov_opset.convert(res_q, Type.f64).output(0)
        res_r = ov_opset.convert(res_r, Type.f64).output(0)

    if mode == "reduced":
        q_target_shape = ov_opset.concat(
            [
                batch_dims,
                ov_opset.unsqueeze(M, zero_const).output(0),
                ov_opset.unsqueeze(K, zero_const).output(0),
            ],
            0,
        ).output(0)
        q_slice = ov_opset.slice(
            res_q,
            ov_opset.constant([0], Type.i64).output(0),
            ov_opset.unsqueeze(K, zero_const).output(0),
            ov_opset.constant([1], Type.i64).output(0),
            ov_opset.constant([-1], Type.i64).output(0),
        ).output(0)
        res_q = ov_opset.reshape(q_slice, q_target_shape, False).output(0)

        r_target_shape = ov_opset.concat(
            [
                batch_dims,
                ov_opset.unsqueeze(K, zero_const).output(0),
                ov_opset.unsqueeze(N, zero_const).output(0),
            ],
            0,
        ).output(0)
        r_slice = ov_opset.slice(
            res_r,
            ov_opset.constant([0], Type.i64).output(0),
            ov_opset.unsqueeze(K, zero_const).output(0),
            ov_opset.constant([1], Type.i64).output(0),
            ov_opset.constant([-2], Type.i64).output(0),
        ).output(0)
        res_r = ov_opset.reshape(r_slice, r_target_shape, False).output(0)
    else:
        q_target_shape = ov_opset.concat(
            [
                batch_dims,
                ov_opset.unsqueeze(M, zero_const).output(0),
                ov_opset.unsqueeze(M, zero_const).output(0),
            ],
            0,
        ).output(0)
        res_q = ov_opset.reshape(res_q, q_target_shape, False).output(0)

        r_target_shape = ov_opset.concat(
            [
                batch_dims,
                ov_opset.unsqueeze(M, zero_const).output(0),
                ov_opset.unsqueeze(N, zero_const).output(0),
            ],
            0,
        ).output(0)
        res_r = ov_opset.reshape(res_r, r_target_shape, False).output(0)

    return OpenVINOKerasTensor(res_q), OpenVINOKerasTensor(res_r)


def solve(a, b):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    a_ov = get_ov_output(a)
    b_ov = get_ov_output(b)
    squeeze = b.ndim == a.ndim - 1
    if squeeze:
        minus_one = ov_opset.constant([-1], Type.i32).output(0)
        b_ov = ov_opset.unsqueeze(b_ov, minus_one).output(0)
    a_inv = ov_opset.inverse(a_ov, adjoint=False).output(0)
    result = ov_opset.matmul(a_inv, b_ov, False, False).output(0)
    if squeeze:
        minus_one = ov_opset.constant([-1], Type.i32).output(0)
        result = ov_opset.squeeze(result, minus_one).output(0)
    return OpenVINOKerasTensor(result)


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
