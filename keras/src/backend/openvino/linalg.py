import numpy as np
import openvino.opset15 as ov_opset
from openvino import Type, Model

from keras.src.backend import config
from keras.src.backend import standardize_dtype
from keras.src.backend.common import dtypes
from keras.src.backend.openvino.core import OPENVINO_DTYPES
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
    a = convert_to_tensor(a)
    a_ov = get_ov_output(a)
    a_ov_type = a_ov.get_element_type()
    if not a_ov_type.is_real():
        
        a_ov = ov_opset.convert(a_ov, Type.f32).output(0)
        out_ov_type = Type.f32
    else:
        out_ov_type = a_ov_type
    zero_const = ov_opset.constant(0, Type.i32).output(0)
    one_const = ov_opset.constant(1, Type.i32).output(0)
    minus_one_const = ov_opset.constant(-1, Type.i32).output(0)
    minus_two_const = ov_opset.constant(-2, Type.i32).output(0)
    a_shape = ov_opset.shape_of(a_ov, Type.i32).output(0)
    rank = a_ov.get_partial_shape().rank.get_length()
    if rank == 2:
        n = ov_opset.gather(a_shape, ov_opset.constant(0, Type.i32), zero_const).output(0)
        n_int = n
        batch_size_prod = ov_opset.constant(1, Type.i32).output(0)
    else:
        n = ov_opset.gather(a_shape, minus_one_const, zero_const).output(0)
        n_int = n
        batch_shape = ov_opset.slice(
            a_shape,
            ov_opset.constant([0], Type.i32),
            ov_opset.constant([-2], Type.i32),
            ov_opset.constant([1], Type.i32),
            ov_opset.constant([0], Type.i32)
        ).output(0)
        batch_size_prod = ov_opset.reduce_prod(batch_shape, zero_const, False).output(0)
    a_flat_shape = ov_opset.concat([
        ov_opset.unsqueeze(batch_size_prod, zero_const).output(0),
        ov_opset.unsqueeze(n, zero_const).output(0),
        ov_opset.unsqueeze(n, zero_const).output(0)
    ], axis=0).output(0)
    A_flat = ov_opset.reshape(a_ov, a_flat_shape, False).output(0)
    n_scalar_shape = ov_opset.unsqueeze(n, zero_const).output(0)
    range_n = ov_opset.range(
        zero_const, n, one_const, output_type=Type.i32
    ).output(0)
    eye_n = ov_opset.one_hot(
        range_n, n,
        ov_opset.constant(1.0, out_ov_type),
        ov_opset.constant(0.0, out_ov_type),
        axis=-1
    ).output(0)
    V_flat = ov_opset.broadcast(
        eye_n, 
        a_flat_shape
    ).output(0)
    n_minus_one = ov_opset.subtract(n_int, one_const).output(0)
    n_squared_minus_n = ov_opset.multiply(n_int, n_minus_one).output(0)
    sweep_iters = ov_opset.divide(n_squared_minus_n, ov_opset.constant(2, Type.i32)).output(0)
    max_iter = ov_opset.multiply(ov_opset.constant(15, Type.i32), sweep_iters).output(0)
    trip_count = max_iter
    execution_cond = ov_opset.constant(True, Type.boolean).output(0)
    loop = ov_opset.loop(trip_count, execution_cond)
    A_param = ov_opset.parameter(A_flat.get_partial_shape(), A_flat.get_element_type())
    V_param = ov_opset.parameter(V_flat.get_partial_shape(), V_flat.get_element_type())
    A_curr = A_param.output(0)
    V_curr = V_param.output(0)
    A_curr_shape = ov_opset.shape_of(A_curr, Type.i32).output(0)
    l_batch_size_prod = ov_opset.gather(A_curr_shape, zero_const, zero_const).output(0)
    l_n = ov_opset.gather(A_curr_shape, minus_one_const, zero_const).output(0)
    l_flat_shape = A_curr_shape
    l_range_n = ov_opset.range(zero_const, l_n, one_const, output_type=Type.i32).output(0)
    l_eye_n = ov_opset.one_hot(l_range_n, l_n, ov_opset.constant(1.0, out_ov_type), ov_opset.constant(0.0, out_ov_type), axis=-1).output(0)
    mask = ov_opset.subtract(
        ov_opset.constant(1.0, out_ov_type),
        l_eye_n
    ).output(0)
    mask_b = ov_opset.broadcast(mask, l_flat_shape).output(0)
    A_off = ov_opset.multiply(A_curr, mask_b).output(0)
    A_off_abs = ov_opset.abs(A_off).output(0)
    flat_n2 = ov_opset.concat([
        ov_opset.unsqueeze(l_batch_size_prod, zero_const).output(0),
        ov_opset.unsqueeze(ov_opset.multiply(l_n, l_n), zero_const).output(0)
    ], axis=0).output(0)
    A_off_abs_flat = ov_opset.reshape(A_off_abs, flat_n2, False).output(0)
    max_val = ov_opset.reduce_max(A_off_abs_flat, one_const, False).output(0)
    epsilon = ov_opset.constant(1e-6, out_ov_type).output(0)
    continue_cond = ov_opset.reduce_logical_or(
        ov_opset.greater(max_val, epsilon),
        zero_const, False
    ).output(0)
    topk = ov_opset.topk(A_off_abs_flat, ov_opset.constant(1, Type.i32), 1, "max", "value")
    argmax_flat = topk.output(1) # shape [B, 1]
    argmax_flat_sq = ov_opset.squeeze(argmax_flat, one_const).output(0) # shape [B]
    p = ov_opset.divide(argmax_flat_sq, l_n).output(0) # shape [B]
    q = ov_opset.mod(argmax_flat_sq, l_n).output(0)    # shape [B]
    p_unsqueezed = ov_opset.unsqueeze(p, one_const).output(0)
    q_unsqueezed = ov_opset.unsqueeze(q, one_const).output(0)
    b_indices = ov_opset.range(zero_const, l_batch_size_prod, one_const, output_type=Type.i32).output(0)
    b_unsqueezed = ov_opset.unsqueeze(b_indices, one_const).output(0)
    pp_indices = ov_opset.concat([b_unsqueezed, p_unsqueezed, p_unsqueezed], axis=1).output(0)
    qq_indices = ov_opset.concat([b_unsqueezed, q_unsqueezed, q_unsqueezed], axis=1).output(0)
    pq_indices = ov_opset.concat([b_unsqueezed, p_unsqueezed, q_unsqueezed], axis=1).output(0)
    App = ov_opset.gather_nd(A_curr, pp_indices).output(0) # shape [B]
    Aqq = ov_opset.gather_nd(A_curr, qq_indices).output(0)
    Apq = ov_opset.gather_nd(A_curr, pq_indices).output(0)
    zero_out = ov_opset.constant(0.0, out_ov_type).output(0)
    is_p_eq_q = ov_opset.equal(p, q).output(0)
    is_apq_zero = ov_opset.logical_or(ov_opset.equal(Apq, zero_out).output(0), is_p_eq_q).output(0)
    safe_Apq = ov_opset.select(is_apq_zero, ov_opset.constant(1.0, out_ov_type), Apq).output(0)
    theta = ov_opset.divide(
        ov_opset.subtract(Aqq, App),
        ov_opset.multiply(ov_opset.constant(2.0, out_ov_type), safe_Apq)
    ).output(0)
    theta_abs = ov_opset.abs(theta).output(0)
    theta_sign = ov_opset.sign(theta).output(0)
    theta_sign = ov_opset.select(ov_opset.equal(theta, zero_out), ov_opset.constant(1.0, out_ov_type), theta_sign).output(0)
    sqrt_term = ov_opset.sqrt(
        ov_opset.add(
            ov_opset.multiply(theta, theta),
            ov_opset.constant(1.0, out_ov_type)
        )
    ).output(0)
    t = ov_opset.divide(
        theta_sign,
        ov_opset.add(theta_abs, sqrt_term)
    ).output(0)
    t = ov_opset.select(is_apq_zero, zero_out, t).output(0)
    c = ov_opset.divide(
        ov_opset.constant(1.0, out_ov_type),
        ov_opset.sqrt(
            ov_opset.add(
                ov_opset.multiply(t, t),
                ov_opset.constant(1.0, out_ov_type)
            )
        )
    ).output(0)
    s = ov_opset.multiply(c, t).output(0)
    c_unsqueezed = ov_opset.unsqueeze(c, one_const).output(0)
    s_unsqueezed = ov_opset.unsqueeze(s, one_const).output(0)
    R = ov_opset.broadcast(l_eye_n, l_flat_shape).output(0)
    c_safe = ov_opset.select(is_p_eq_q, ov_opset.constant(1.0, out_ov_type), c).output(0)
    s_safe = ov_opset.select(is_p_eq_q, zero_out, s).output(0)
    c_updates = c_safe
    s_updates = s_safe
    neg_s_updates = ov_opset.negative(s_safe).output(0)
    p_safe = ov_opset.select(is_p_eq_q, ov_opset.constant(0, Type.i32), p).output(0)
    q_safe = ov_opset.select(is_p_eq_q, ov_opset.constant(1, Type.i32), q).output(0)
    p_safe_unsqueezed = ov_opset.unsqueeze(p_safe, one_const).output(0)
    q_safe_unsqueezed = ov_opset.unsqueeze(q_safe, one_const).output(0)
    pp_safe_indices = ov_opset.concat([b_unsqueezed, p_safe_unsqueezed, p_safe_unsqueezed], axis=1).output(0)
    qq_safe_indices = ov_opset.concat([b_unsqueezed, q_safe_unsqueezed, q_safe_unsqueezed], axis=1).output(0)
    pq_safe_indices = ov_opset.concat([b_unsqueezed, p_safe_unsqueezed, q_safe_unsqueezed], axis=1).output(0)
    qp_safe_indices = ov_opset.concat([b_unsqueezed, q_safe_unsqueezed, p_safe_unsqueezed], axis=1).output(0)

    R = ov_opset.scatter_nd_update(R, pp_safe_indices, c_updates).output(0)
    R = ov_opset.scatter_nd_update(R, qq_safe_indices, c_updates).output(0)
    R = ov_opset.scatter_nd_update(R, pq_safe_indices, s_updates).output(0)
    R = ov_opset.scatter_nd_update(R, qp_safe_indices, neg_s_updates).output(0)
    
    # Transpose R to R^T: swap last two dims
    RT = ov_opset.transpose(R, ov_opset.constant([0, 2, 1], Type.i32)).output(0)
    
    A_next = ov_opset.matmul(RT, ov_opset.matmul(A_curr, R, False, False), False, False).output(0)
    V_next = ov_opset.matmul(V_curr, R, False, False).output(0)
    
    # If not continue_cond, just return A_curr and V_curr
    A_next = ov_opset.select(
        ov_opset.unsqueeze(ov_opset.unsqueeze(continue_cond, zero_const), zero_const).output(0),
        A_next, A_curr
    ).output(0)
    
    V_next = ov_opset.select(
        ov_opset.unsqueeze(ov_opset.unsqueeze(continue_cond, zero_const), zero_const).output(0),
        V_next, V_curr
    ).output(0)
    body = Model([continue_cond, A_next, V_next], [A_param, V_param], "jacobi_loop")
    loop.set_function(body)
    loop.set_special_body_ports([-1, 0])
    loop.set_merged_input(A_param, A_flat, A_next)
    loop.set_merged_input(V_param, V_flat, V_next)
    A_out = loop.get_iter_value(A_next)
    V_out = loop.get_iter_value(V_next)
    eigenvalues_flat = ov_opset.reduce_sum(
        ov_opset.multiply(A_out, eye_n), minus_one_const, False
    ).output(0)
    neg_eigenvalues = ov_opset.negative(eigenvalues_flat).output(0)
    topk_sort = ov_opset.topk(neg_eigenvalues, n, -1, "max", "value")
    w_flat = ov_opset.negative(topk_sort.output(0)).output(0)
    sort_indices = topk_sort.output(1)
    v_indices = ov_opset.broadcast(
        ov_opset.unsqueeze(sort_indices, one_const).output(0),
        a_flat_shape
    ).output(0)
    v_flat = ov_opset.gather_elements(V_out, v_indices, -1).output(0)
    if rank == 2:
        w_final = ov_opset.squeeze(w_flat, zero_const).output(0)
        v_final = ov_opset.squeeze(v_flat, zero_const).output(0)
    else:
        w_shape_final = ov_opset.concat([batch_shape, ov_opset.unsqueeze(n, zero_const).output(0)], axis=0).output(0)
        w_final = ov_opset.reshape(w_flat, w_shape_final, False).output(0)
        v_final = ov_opset.reshape(v_flat, a_shape, False).output(0)
    if out_ov_type == Type.f64:
        w_final = ov_opset.convert(w_final, Type.f64).output(0)
        v_final = ov_opset.convert(v_final, Type.f64).output(0)

    return (
        OpenVINOKerasTensor(w_final),
        OpenVINOKerasTensor(v_final),
    )

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
    raise NotImplementedError("`qr` is not supported with openvino backend")


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
