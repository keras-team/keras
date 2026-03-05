import openvino.opset15 as ov_opset
import openvino as ov
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
    a = convert_to_tensor(a)
    a_ov = get_ov_output(a)
    original_type = a_ov.get_element_type()
    
    # Avoid constant folding bug for f64 in OpenVINO CPU Loop evaluate
    if original_type == Type.f64:
        a_ov = ov_opset.convert(a_ov, Type.f32).output(0)
    
    a_shape = ov_opset.shape_of(a_ov, output_type="i32").output(0)
    
    rank = a_ov.get_partial_shape().rank.get_length()
    
    minus_1 = ov_opset.constant([-1], Type.i32).output(0)
    minus_2 = ov_opset.constant([-2], Type.i32).output(0)
    
    N_node_1d = ov_opset.gather(
        a_shape, minus_1, ov_opset.constant(0, Type.i32).output(0)
    ).output(0)
    N_node_scalar = ov_opset.squeeze(
        N_node_1d, ov_opset.constant([0], Type.i32).output(0)
    ).output(0)
    
    num_batch_dims = rank - 2
    if num_batch_dims > 0:
        batch_dims_shape = ov_opset.broadcast(
            ov_opset.constant([1], Type.i32).output(0),
            ov_opset.constant([num_batch_dims], Type.i32).output(0)
        ).output(0)
        eye_shape = ov_opset.concat(
            [batch_dims_shape, N_node_1d, N_node_1d], 0
        ).output(0)
    else:
        eye_shape = ov_opset.concat([N_node_1d, N_node_1d], 0).output(0)
        
    eye = ov_opset.eye(
        N_node_scalar, 
        N_node_scalar, 
        ov_opset.constant(0, Type.i32).output(0), 
        a_ov.get_element_type()
    ).output(0)
    eye_reshaped = ov_opset.reshape(eye, eye_shape, False).output(0)
    
    trip_count = N_node_scalar
    loop = ov_opset.loop(
        trip_count, ov_opset.constant(True, Type.boolean).output(0)
    )
    
    M_param = ov_opset.parameter(
        [-1] * rank, a_ov.get_element_type(), "M"
    )
    k_param = ov_opset.parameter([], Type.i32, "k")
    A_body_param = ov_opset.parameter(
        [-1] * rank, a_ov.get_element_type(), "A_body"
    )
    eye_body_param = ov_opset.parameter(
        [-1] * rank, a_ov.get_element_type(), "eye_body"
    )
    
    k_next = ov_opset.add(
        k_param.output(0), ov_opset.constant(1, Type.i32).output(0)
    ).output(0)
    k_f32 = ov_opset.convert(k_next, a_ov.get_element_type()).output(0)
    
    M_diag = ov_opset.multiply(
        M_param.output(0), eye_body_param.output(0)
    ).output(0)
    trace_axes = ov_opset.concat([minus_2, minus_1], 0).output(0)
    trace = ov_opset.reduce_sum(M_diag, trace_axes, keep_dims=True).output(0)
    
    minus_one = ov_opset.constant(-1.0, a_ov.get_element_type()).output(0)
    c_k_factor = ov_opset.divide(minus_one, k_f32).output(0)
    c_k = ov_opset.multiply(c_k_factor, trace).output(0)
    
    c_k_I = ov_opset.multiply(c_k, eye_body_param.output(0)).output(0)
    M_plus_c_k_I = ov_opset.add(M_param.output(0), c_k_I).output(0)
    
    M_next = ov_opset.matmul(
        A_body_param.output(0), M_plus_c_k_I, False, False
    ).output(0)
    
    cond_next = ov_opset.constant(True, Type.boolean).output(0)
    
    body = ov.Model(
        [M_next, k_next, c_k, cond_next], 
        [M_param, k_param, A_body_param, eye_body_param]
    )
    loop.set_function(body)
    loop.set_special_body_ports([-1, 3])
    
    loop.set_merged_input(M_param, a_ov, M_next)
    loop.set_merged_input(
        k_param, ov_opset.constant(0, Type.i32).output(0), k_next
    )
    loop.set_invariant_input(A_body_param, a_ov)
    loop.set_invariant_input(eye_body_param, eye_reshaped)
    
    out_c_k = loop.get_iter_value(c_k, -1)
    
    det_c_k = ov_opset.squeeze(out_c_k, trace_axes).output(0)
    
    N_mod_2 = ov_opset.mod(
        N_node_scalar, ov_opset.constant(2, Type.i32).output(0)
    ).output(0)
    N_mod_2_f32 = ov_opset.convert(N_mod_2, a_ov.get_element_type()).output(0)
    one = ov_opset.constant(1.0, a_ov.get_element_type()).output(0)
    two = ov_opset.constant(2.0, a_ov.get_element_type()).output(0)
    sign = ov_opset.subtract(
        one, ov_opset.multiply(two, N_mod_2_f32).output(0)
    ).output(0)
    
    det = ov_opset.multiply(det_c_k, sign).output(0)
    
    if original_type == Type.f64:
        det = ov_opset.convert(det, Type.f64).output(0)
        
    return OpenVINOKerasTensor(det)


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
