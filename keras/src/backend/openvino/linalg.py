import openvino.opset15 as ov_opset
from openvino import Type

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
    a = convert_to_tensor(a)
    a_ov = get_ov_output(a)
    a_ov_type = a_ov.get_element_type()

    if a_ov_type.is_integral() or a_ov_type == Type.boolean:
        float_type = OPENVINO_DTYPES[config.floatx()]
        a_ov = ov_opset.convert(a_ov, float_type).output(0)
        a_ov_type = a_ov.get_element_type()

    a_shape = a_ov.get_partial_shape()
    a_rank = len(a_shape)
    n = a_shape[-1].get_length()

    flat_shape = ov_opset.constant([-1, n, n], Type.i32).output(0)
    a_batched = ov_opset.reshape(a_ov, flat_shape, False).output(0)

    batch_shape = ov_opset.shape_of(a_batched, Type.i32).output(0)
    batch_size = ov_opset.gather(
        batch_shape,
        ov_opset.constant([0], Type.i32).output(0),
        ov_opset.constant(0, Type.i32).output(0),
    ).output(0)

    one = ov_opset.constant(1.0, a_ov_type).output(0)
    two = ov_opset.constant(2.0, a_ov_type).output(0)

    det_val = ov_opset.broadcast(one, batch_size).output(0)

    row_axis = ov_opset.constant(1, Type.i32).output(0)
    col_axis = ov_opset.constant(2, Type.i32).output(0)

    for k in range(n):
        col_k = ov_opset.gather(
            a_batched, ov_opset.constant(k, Type.i32).output(0), col_axis
        ).output(0)
        abs_col_k = ov_opset.absolute(col_k).output(0)

        abs_col_k_sub = ov_opset.slice(
            abs_col_k,
            ov_opset.constant([0, k], Type.i32).output(0),
            ov_opset.constant([2**30, n], Type.i32).output(0),
            ov_opset.constant([1, 1], Type.i32).output(0),
            ov_opset.constant([0, 1], Type.i32).output(0),
        ).output(0)

        topk_result = ov_opset.topk(
            abs_col_k_sub,
            ov_opset.constant(1, Type.i32).output(0),
            axis=1,
            mode="max",
            sort="none",
        )
        local_max_idx = ov_opset.squeeze(
            ov_opset.convert(topk_result.output(1), Type.i32).output(0),
            ov_opset.constant([1], Type.i32).output(0),
        ).output(0)

        pivot_row = ov_opset.add(
            local_max_idx, ov_opset.constant(k, Type.i32).output(0)
        ).output(0)

        swap_needed = ov_opset.not_equal(
            pivot_row, ov_opset.constant(k, Type.i32).output(0)
        ).output(0)
        swap_needed_f = ov_opset.convert(swap_needed, a_ov_type).output(0)
        sign_flip = ov_opset.subtract(
            ov_opset.broadcast(one, batch_size).output(0),
            ov_opset.multiply(two, swap_needed_f).output(0),
        ).output(0)
        det_val = ov_opset.multiply(det_val, sign_flip).output(0)

        row_k = ov_opset.gather(
            a_batched, ov_opset.constant([k], Type.i32).output(0), row_axis
        ).output(0)
        pivot_row_2d = ov_opset.unsqueeze(
            pivot_row, ov_opset.constant([1], Type.i32).output(0)
        ).output(0)
        pivot_row_data = ov_opset.gather(
            a_batched, pivot_row_2d, row_axis, batch_dims=1
        ).output(0)

        a_batched = ov_opset.scatter_update(
            a_batched,
            ov_opset.constant([k], Type.i32).output(0),
            pivot_row_data,
            row_axis,
        ).output(0)

        all_row_indices = ov_opset.unsqueeze(
            ov_opset.range(
                ov_opset.constant(0, Type.i32).output(0),
                ov_opset.constant(n, Type.i32).output(0),
                ov_opset.constant(1, Type.i32).output(0),
                output_type=Type.i32,
            ).output(0),
            ov_opset.constant([0, 2], Type.i32).output(0),
        ).output(0)

        pivot_row_3d = ov_opset.unsqueeze(
            pivot_row_2d, ov_opset.constant([2], Type.i32).output(0)
        ).output(0)
        swap_mask = ov_opset.equal(all_row_indices, pivot_row_3d).output(0)
        row_k_tiled = ov_opset.broadcast(
            row_k, ov_opset.shape_of(a_batched, Type.i32).output(0)
        ).output(0)
        a_batched = ov_opset.select(swap_mask, row_k_tiled, a_batched).output(0)

        k_idx = ov_opset.constant([k], Type.i32).output(0)
        pivot_row_cur = ov_opset.gather(a_batched, k_idx, row_axis).output(0)
        pivot_elem = ov_opset.gather(pivot_row_cur, k_idx, col_axis).output(0)
        pivot_scalar = ov_opset.squeeze(
            pivot_elem, ov_opset.constant([1, 2], Type.i32).output(0)
        ).output(0)

        det_val = ov_opset.multiply(det_val, pivot_scalar).output(0)

        safe_pivot = ov_opset.select(
            ov_opset.equal(
                pivot_elem, ov_opset.constant(0.0, a_ov_type).output(0)
            ).output(0),
            ov_opset.constant(1.0, a_ov_type).output(0),
            pivot_elem,
        ).output(0)

        for i in range(k + 1, n):
            i_idx = ov_opset.constant([i], Type.i32).output(0)
            row_i = ov_opset.gather(a_batched, i_idx, row_axis).output(0)
            elem_ik = ov_opset.gather(row_i, k_idx, col_axis).output(0)
            multiplier = ov_opset.divide(elem_ik, safe_pivot).output(0)
            row_i_new = ov_opset.subtract(
                row_i, ov_opset.multiply(multiplier, pivot_row_cur).output(0)
            ).output(0)
            a_batched = ov_opset.scatter_update(
                a_batched, i_idx, row_i_new, row_axis
            ).output(0)

    if a_rank > 2:
        batch_dims = [a_shape[i].get_length() for i in range(a_rank - 2)]
        out_shape = ov_opset.constant(batch_dims, Type.i32).output(0)
    else:
        out_shape = ov_opset.constant([], Type.i32).output(0)

    det_result = ov_opset.reshape(det_val, out_shape, False).output(0)

    return OpenVINOKerasTensor(det_result)


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
