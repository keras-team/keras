import warnings

import openvino as ov
import openvino.opset15 as ov_opset
from openvino import Model
from openvino import Type

from keras.src.backend import config
from keras.src.backend import standardize_dtype
from keras.src.backend.common import dtypes
from keras.src.backend.openvino.core import OpenVINOKerasTensor
from keras.src.backend.openvino.core import cast
from keras.src.backend.openvino.core import convert_to_tensor
from keras.src.backend.openvino.core import get_ov_output


def cholesky(a, upper=False):
    warnings.warn(
        "OpenVINO backend does not check for positive definiteness. "
        "Non-positive-definite inputs will silently produce NaN outputs."
    )
    a = convert_to_tensor(a)
    a_ov = get_ov_output(a)
    original_type = a_ov.get_element_type()

    # avoid OV CPU constant-folding bug for f64 inside Loop
    if original_type == Type.f64:
        a_ov = ov_opset.convert(a_ov, Type.f32).output(0)
    work_type = a_ov.get_element_type()

    rank = a_ov.get_partial_shape().rank.get_length()
    a_shape = ov_opset.shape_of(a_ov, output_type="i32").output(0)

    zero_s = ov_opset.constant(0, Type.i32).output(0)
    one_s = ov_opset.constant(1, Type.i32).output(0)

    N_node = ov_opset.squeeze(
        ov_opset.gather(
            a_shape,
            ov_opset.constant([-1], Type.i32).output(0),
            zero_s,
        ).output(0),
        ov_opset.constant([0], Type.i32).output(0),
    ).output(0)

    if rank == 2:
        B_node = ov_opset.constant(1, Type.i32).output(0)
        batch_shape = ov_opset.constant([], Type.i32).output(0)
    else:
        batch_shape = ov_opset.slice(
            a_shape,
            ov_opset.constant([0], Type.i32).output(0),
            ov_opset.constant([-2], Type.i32).output(0),
            ov_opset.constant([1], Type.i32).output(0),
            ov_opset.constant([0], Type.i32).output(0),
        ).output(0)
        B_node = ov_opset.reduce_prod(batch_shape, zero_s, False).output(0)

    flat_shape = ov_opset.concat(
        [
            ov_opset.unsqueeze(B_node, zero_s).output(0),
            ov_opset.unsqueeze(N_node, zero_s).output(0),
            ov_opset.unsqueeze(N_node, zero_s).output(0),
        ],
        0,
    ).output(0)
    A_flat = ov_opset.reshape(a_ov, flat_shape, False).output(0)

    L_init = ov_opset.broadcast(
        ov_opset.constant(0.0, work_type).output(0), flat_shape
    ).output(0)

    loop = ov_opset.loop(
        N_node, ov_opset.constant(True, Type.boolean).output(0)
    )

    A_param = ov_opset.parameter(
        A_flat.get_partial_shape(), work_type, name="A"
    )
    L_param = ov_opset.parameter(
        L_init.get_partial_shape(), work_type, name="L"
    )
    k_param = ov_opset.parameter([], Type.i32, name="k")
    B_param = ov_opset.parameter([], Type.i32, name="B")
    N_param = ov_opset.parameter([], Type.i32, name="N")

    A_body = A_param.output(0)
    L_body = L_param.output(0)
    k = k_param.output(0)
    N_body = N_param.output(0)
    k_p1 = ov_opset.add(k, one_s).output(0)

    # Cholesky-Banachiewicz: at step k, compute full column k of L in one pass.
    # prev = L[:, :, :k]  already-computed columns, shape [B, N, k]
    prev = ov_opset.slice(
        L_body,
        ov_opset.constant([0], Type.i32).output(0),
        ov_opset.unsqueeze(k, zero_s).output(0),
        ov_opset.constant([1], Type.i32).output(0),
        ov_opset.constant([2], Type.i32).output(0),
    ).output(0)

    a_col = ov_opset.slice(
        A_body,
        ov_opset.unsqueeze(k, zero_s).output(0),
        ov_opset.unsqueeze(k_p1, zero_s).output(0),
        ov_opset.constant([1], Type.i32).output(0),
        ov_opset.constant([2], Type.i32).output(0),
    ).output(0)

    row_k_prev = ov_opset.slice(
        prev,
        ov_opset.unsqueeze(k, zero_s).output(0),
        ov_opset.unsqueeze(k_p1, zero_s).output(0),
        ov_opset.constant([1], Type.i32).output(0),
        ov_opset.constant([1], Type.i32).output(0),
    ).output(0)

    # acc[b,i] = sum_j L[b,i,j]*L[b,k,j]  via matmul([B,N,k], [B,k,1])
    acc = ov_opset.matmul(prev, row_k_prev, False, True).output(0)

    raw_col = ov_opset.subtract(a_col, acc).output(0)

    diag_val = ov_opset.slice(
        raw_col,
        ov_opset.unsqueeze(k, zero_s).output(0),
        ov_opset.unsqueeze(k_p1, zero_s).output(0),
        ov_opset.constant([1], Type.i32).output(0),
        ov_opset.constant([1], Type.i32).output(0),
    ).output(0)

    L_kk = ov_opset.sqrt(diag_val).output(0)
    new_col = ov_opset.divide(raw_col, L_kk).output(0)

    # Zero out rows above the diagonal (rows 0..k-1)
    row_range = ov_opset.range(
        zero_s, N_body, one_s, output_type=Type.i32
    ).output(0)
    row_mask_1d = ov_opset.greater_equal(row_range, k).output(0)
    row_mask_3d = ov_opset.unsqueeze(
        ov_opset.unsqueeze(row_mask_1d, zero_s).output(0),
        ov_opset.constant(2, Type.i32).output(0),
    ).output(0)
    zero_col = ov_opset.broadcast(
        ov_opset.constant(0.0, work_type).output(0),
        ov_opset.shape_of(new_col, output_type="i32").output(0),
    ).output(0)
    new_col_masked = ov_opset.select(row_mask_3d, new_col, zero_col).output(0)

    # Pad [B,N,1] -> [B,N,N], placing the column at index k
    pads_begin = ov_opset.concat(
        [
            ov_opset.constant([0], Type.i32).output(0),
            ov_opset.constant([0], Type.i32).output(0),
            ov_opset.unsqueeze(k, zero_s).output(0),
        ],
        0,
    ).output(0)
    pads_end = ov_opset.concat(
        [
            ov_opset.constant([0], Type.i32).output(0),
            ov_opset.constant([0], Type.i32).output(0),
            ov_opset.unsqueeze(
                ov_opset.subtract(
                    ov_opset.subtract(N_body, k).output(0), one_s
                ).output(0),
                zero_s,
            ).output(0),
        ],
        0,
    ).output(0)
    col_padded = ov_opset.pad(
        new_col_masked, pads_begin, pads_end, "constant"
    ).output(0)

    col_range = ov_opset.range(
        zero_s, N_body, one_s, output_type=Type.i32
    ).output(0)
    col_mask_1d = ov_opset.equal(col_range, k).output(0)
    L_shape = ov_opset.shape_of(L_body, output_type="i32").output(0)
    col_mask_3d = ov_opset.broadcast(
        ov_opset.unsqueeze(
            ov_opset.unsqueeze(col_mask_1d, zero_s).output(0),
            ov_opset.constant(1, Type.i32).output(0),
        ).output(0),
        L_shape,
    ).output(0)
    row_mask_full = ov_opset.broadcast(
        ov_opset.unsqueeze(
            ov_opset.unsqueeze(row_mask_1d, zero_s).output(0),
            ov_opset.constant(2, Type.i32).output(0),
        ).output(0),
        L_shape,
    ).output(0)
    update_mask = ov_opset.logical_and(col_mask_3d, row_mask_full).output(0)

    L_next = ov_opset.select(update_mask, col_padded, L_body).output(0)
    k_next = ov_opset.add(k, one_s).output(0)
    cond = ov_opset.constant(True, Type.boolean).output(0)

    body = Model(
        [L_next, k_next, cond],
        [A_param, L_param, k_param, B_param, N_param],
    )
    loop.set_function(body)
    loop.set_special_body_ports([-1, 2])

    loop.set_invariant_input(A_param, A_flat)
    loop.set_merged_input(L_param, L_init, L_next)
    loop.set_merged_input(k_param, zero_s, k_next)
    loop.set_invariant_input(B_param, B_node)
    loop.set_invariant_input(N_param, N_node)

    L_out = loop.get_iter_value(L_next, -1)

    if rank == 2:
        L_final = ov_opset.squeeze(L_out, zero_s).output(0)
    else:
        out_shape = ov_opset.concat(
            [
                batch_shape,
                ov_opset.unsqueeze(N_node, zero_s).output(0),
                ov_opset.unsqueeze(N_node, zero_s).output(0),
            ],
            0,
        ).output(0)
        L_final = ov_opset.reshape(L_out, out_shape, False).output(0)

    if upper:
        order = list(range(rank))
        order[-2], order[-1] = order[-1], order[-2]
        perm = ov_opset.constant(order, Type.i32).output(0)
        L_final = ov_opset.transpose(L_final, perm).output(0)

    if original_type == Type.f64:
        L_final = ov_opset.convert(L_final, Type.f64).output(0)

    return OpenVINOKerasTensor(L_final)


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
            ov_opset.constant([num_batch_dims], Type.i32).output(0),
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
        a_ov.get_element_type(),
    ).output(0)
    eye_reshaped = ov_opset.reshape(eye, eye_shape, False).output(0)

    trip_count = N_node_scalar
    loop = ov_opset.loop(
        trip_count, ov_opset.constant(True, Type.boolean).output(0)
    )

    M_param = ov_opset.parameter([-1] * rank, a_ov.get_element_type(), "M")
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
        [M_param, k_param, A_body_param, eye_body_param],
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
    a_shape = ov_opset.shape_of(a_ov, Type.i32).output(0)
    rank = a_ov.get_partial_shape().rank.get_length()
    if rank == 2:
        n = ov_opset.gather(
            a_shape, ov_opset.constant(0, Type.i32), zero_const
        ).output(0)
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
            ov_opset.constant([0], Type.i32),
        ).output(0)
        batch_size_prod = ov_opset.reduce_prod(
            batch_shape, zero_const, False
        ).output(0)
    a_flat_shape = ov_opset.concat(
        [
            ov_opset.unsqueeze(batch_size_prod, zero_const).output(0),
            ov_opset.unsqueeze(n, zero_const).output(0),
            ov_opset.unsqueeze(n, zero_const).output(0),
        ],
        axis=0,
    ).output(0)
    A_flat = ov_opset.reshape(a_ov, a_flat_shape, False).output(0)
    range_n = ov_opset.range(
        zero_const, n, one_const, output_type=Type.i32
    ).output(0)
    eye_n = ov_opset.one_hot(
        range_n,
        n,
        ov_opset.constant(1.0, out_ov_type),
        ov_opset.constant(0.0, out_ov_type),
        axis=-1,
    ).output(0)
    V_flat = ov_opset.broadcast(eye_n, a_flat_shape).output(0)
    n_minus_one = ov_opset.subtract(n_int, one_const).output(0)
    n_squared_minus_n = ov_opset.multiply(n_int, n_minus_one).output(0)
    sweep_iters = ov_opset.divide(
        n_squared_minus_n, ov_opset.constant(2, Type.i32)
    ).output(0)
    max_iter = ov_opset.multiply(
        ov_opset.constant(15, Type.i32), sweep_iters
    ).output(0)
    trip_count = max_iter
    execution_cond = ov_opset.constant(True, Type.boolean).output(0)
    loop = ov_opset.loop(trip_count, execution_cond)
    A_param = ov_opset.parameter(
        A_flat.get_partial_shape(), A_flat.get_element_type()
    )
    V_param = ov_opset.parameter(
        V_flat.get_partial_shape(), V_flat.get_element_type()
    )
    A_curr = A_param.output(0)
    V_curr = V_param.output(0)
    A_curr_shape = ov_opset.shape_of(A_curr, Type.i32).output(0)
    l_batch_size_prod = ov_opset.gather(
        A_curr_shape, zero_const, zero_const
    ).output(0)
    l_n = ov_opset.gather(A_curr_shape, minus_one_const, zero_const).output(0)
    l_flat_shape = A_curr_shape
    l_range_n = ov_opset.range(
        zero_const, l_n, one_const, output_type=Type.i32
    ).output(0)
    l_eye_n = ov_opset.one_hot(
        l_range_n,
        l_n,
        ov_opset.constant(1.0, out_ov_type),
        ov_opset.constant(0.0, out_ov_type),
        axis=-1,
    ).output(0)
    mask = ov_opset.subtract(
        ov_opset.constant(1.0, out_ov_type), l_eye_n
    ).output(0)
    mask_b = ov_opset.broadcast(mask, l_flat_shape).output(0)
    A_off = ov_opset.multiply(A_curr, mask_b).output(0)
    A_off_abs = ov_opset.abs(A_off).output(0)
    flat_n2 = ov_opset.concat(
        [
            ov_opset.unsqueeze(l_batch_size_prod, zero_const).output(0),
            ov_opset.unsqueeze(ov_opset.multiply(l_n, l_n), zero_const).output(
                0
            ),
        ],
        axis=0,
    ).output(0)
    A_off_abs_flat = ov_opset.reshape(A_off_abs, flat_n2, False).output(0)
    max_val = ov_opset.reduce_max(A_off_abs_flat, one_const, False).output(0)
    epsilon = ov_opset.constant(1e-6, out_ov_type).output(0)
    continue_cond = ov_opset.reduce_logical_or(
        ov_opset.greater(max_val, epsilon), zero_const, False
    ).output(0)
    topk = ov_opset.topk(
        A_off_abs_flat, ov_opset.constant(1, Type.i32), 1, "max", "value"
    )
    argmax_flat = topk.output(1)  # shape [B, 1]
    argmax_flat_sq = ov_opset.squeeze(argmax_flat, one_const).output(
        0
    )  # shape [B]
    p = ov_opset.divide(argmax_flat_sq, l_n).output(0)  # shape [B]
    q = ov_opset.mod(argmax_flat_sq, l_n).output(0)  # shape [B]
    p_unsqueezed = ov_opset.unsqueeze(p, one_const).output(0)
    q_unsqueezed = ov_opset.unsqueeze(q, one_const).output(0)
    b_indices = ov_opset.range(
        zero_const, l_batch_size_prod, one_const, output_type=Type.i32
    ).output(0)
    b_unsqueezed = ov_opset.unsqueeze(b_indices, one_const).output(0)
    pp_indices = ov_opset.concat(
        [b_unsqueezed, p_unsqueezed, p_unsqueezed], axis=1
    ).output(0)
    qq_indices = ov_opset.concat(
        [b_unsqueezed, q_unsqueezed, q_unsqueezed], axis=1
    ).output(0)
    pq_indices = ov_opset.concat(
        [b_unsqueezed, p_unsqueezed, q_unsqueezed], axis=1
    ).output(0)
    App = ov_opset.gather_nd(A_curr, pp_indices).output(0)  # shape [B]
    Aqq = ov_opset.gather_nd(A_curr, qq_indices).output(0)
    Apq = ov_opset.gather_nd(A_curr, pq_indices).output(0)
    zero_out = ov_opset.constant(0.0, out_ov_type).output(0)
    is_p_eq_q = ov_opset.equal(p, q).output(0)
    is_apq_zero = ov_opset.logical_or(
        ov_opset.equal(Apq, zero_out).output(0), is_p_eq_q
    ).output(0)
    safe_Apq = ov_opset.select(
        is_apq_zero, ov_opset.constant(1.0, out_ov_type), Apq
    ).output(0)
    theta = ov_opset.divide(
        ov_opset.subtract(Aqq, App),
        ov_opset.multiply(ov_opset.constant(2.0, out_ov_type), safe_Apq),
    ).output(0)
    theta_abs = ov_opset.abs(theta).output(0)
    theta_sign = ov_opset.sign(theta).output(0)
    theta_sign = ov_opset.select(
        ov_opset.equal(theta, zero_out),
        ov_opset.constant(1.0, out_ov_type),
        theta_sign,
    ).output(0)
    sqrt_term = ov_opset.sqrt(
        ov_opset.add(
            ov_opset.multiply(theta, theta), ov_opset.constant(1.0, out_ov_type)
        )
    ).output(0)
    t = ov_opset.divide(theta_sign, ov_opset.add(theta_abs, sqrt_term)).output(
        0
    )
    t = ov_opset.select(is_apq_zero, zero_out, t).output(0)
    c = ov_opset.divide(
        ov_opset.constant(1.0, out_ov_type),
        ov_opset.sqrt(
            ov_opset.add(
                ov_opset.multiply(t, t), ov_opset.constant(1.0, out_ov_type)
            )
        ),
    ).output(0)
    s = ov_opset.multiply(c, t).output(0)
    R = ov_opset.broadcast(l_eye_n, l_flat_shape).output(0)
    c_safe = ov_opset.select(
        is_p_eq_q, ov_opset.constant(1.0, out_ov_type), c
    ).output(0)
    s_safe = ov_opset.select(is_p_eq_q, zero_out, s).output(0)
    c_updates = c_safe
    s_updates = s_safe
    neg_s_updates = ov_opset.negative(s_safe).output(0)
    p_safe = ov_opset.select(
        is_p_eq_q, ov_opset.constant(0, Type.i32), p
    ).output(0)
    q_safe = ov_opset.select(
        is_p_eq_q, ov_opset.constant(1, Type.i32), q
    ).output(0)
    p_safe_unsqueezed = ov_opset.unsqueeze(p_safe, one_const).output(0)
    q_safe_unsqueezed = ov_opset.unsqueeze(q_safe, one_const).output(0)
    pp_safe_indices = ov_opset.concat(
        [b_unsqueezed, p_safe_unsqueezed, p_safe_unsqueezed], axis=1
    ).output(0)
    qq_safe_indices = ov_opset.concat(
        [b_unsqueezed, q_safe_unsqueezed, q_safe_unsqueezed], axis=1
    ).output(0)
    pq_safe_indices = ov_opset.concat(
        [b_unsqueezed, p_safe_unsqueezed, q_safe_unsqueezed], axis=1
    ).output(0)
    qp_safe_indices = ov_opset.concat(
        [b_unsqueezed, q_safe_unsqueezed, p_safe_unsqueezed], axis=1
    ).output(0)

    R = ov_opset.scatter_nd_update(R, pp_safe_indices, c_updates).output(0)
    R = ov_opset.scatter_nd_update(R, qq_safe_indices, c_updates).output(0)
    R = ov_opset.scatter_nd_update(R, pq_safe_indices, s_updates).output(0)
    R = ov_opset.scatter_nd_update(R, qp_safe_indices, neg_s_updates).output(0)

    # Transpose R to R^T: swap last two dims
    RT = ov_opset.transpose(R, ov_opset.constant([0, 2, 1], Type.i32)).output(0)

    A_next = ov_opset.matmul(
        RT, ov_opset.matmul(A_curr, R, False, False), False, False
    ).output(0)
    V_next = ov_opset.matmul(V_curr, R, False, False).output(0)

    # If not continue_cond, just return A_curr and V_curr
    A_next = ov_opset.select(
        ov_opset.unsqueeze(
            ov_opset.unsqueeze(continue_cond, zero_const), zero_const
        ).output(0),
        A_next,
        A_curr,
    ).output(0)

    V_next = ov_opset.select(
        ov_opset.unsqueeze(
            ov_opset.unsqueeze(continue_cond, zero_const), zero_const
        ).output(0),
        V_next,
        V_curr,
    ).output(0)
    body = Model(
        [continue_cond, A_next, V_next], [A_param, V_param], "jacobi_loop"
    )
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
        ov_opset.unsqueeze(sort_indices, one_const).output(0), a_flat_shape
    ).output(0)
    v_flat = ov_opset.gather_elements(V_out, v_indices, -1).output(0)
    if rank == 2:
        w_final = ov_opset.squeeze(w_flat, zero_const).output(0)
        v_final = ov_opset.squeeze(v_flat, zero_const).output(0)
    else:
        w_shape_final = ov_opset.concat(
            [batch_shape, ov_opset.unsqueeze(n, zero_const).output(0)], axis=0
        ).output(0)
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
    a = convert_to_tensor(a)
    a_ov = get_ov_output(a)
    a_ov_type = a_ov.get_element_type()

    if not a_ov_type.is_real():
        a_ov = ov_opset.convert(a_ov, Type.f32).output(0)
        out_ov_type = Type.f32
    else:
        out_ov_type = a_ov_type

    original_type = out_ov_type
    if original_type == Type.f64:
        a_ov = ov_opset.convert(a_ov, Type.f32).output(0)
        out_ov_type = Type.f32

    zero_const = ov_opset.constant(0, Type.i32).output(0)
    one_const = ov_opset.constant(1, Type.i32).output(0)
    minus_one_const = ov_opset.constant(-1, Type.i32).output(0)
    minus_2 = ov_opset.constant(-2, Type.i32).output(0)
    a_shape = ov_opset.shape_of(a_ov, Type.i32).output(0)
    rank = a_ov.get_partial_shape().rank.get_length()

    M_node = ov_opset.gather(a_shape, minus_2, zero_const).output(0)
    N_node = ov_opset.gather(a_shape, minus_one_const, zero_const).output(0)
    K_node = ov_opset.minimum(M_node, N_node).output(0)

    if rank == 2:
        B_node = ov_opset.constant(1, Type.i32).output(0)
        batch_shape = ov_opset.constant([], Type.i32).output(0)
    else:
        batch_shape = ov_opset.slice(
            a_shape,
            ov_opset.constant([0], Type.i32).output(0),
            ov_opset.constant([-2], Type.i32).output(0),
            ov_opset.constant([1], Type.i32).output(0),
            ov_opset.constant([0], Type.i32).output(0),
        ).output(0)
        B_node = ov_opset.reduce_prod(batch_shape, zero_const, False).output(0)

    flat_shape = ov_opset.concat(
        [
            ov_opset.unsqueeze(B_node, zero_const).output(0),
            ov_opset.unsqueeze(M_node, zero_const).output(0),
            ov_opset.unsqueeze(N_node, zero_const).output(0),
        ],
        0,
    ).output(0)
    A_flat = ov_opset.reshape(a_ov, flat_shape, False).output(0)

    range_M = ov_opset.range(
        zero_const, M_node, one_const, output_type=Type.i32
    ).output(0)
    P_init = ov_opset.broadcast(
        ov_opset.unsqueeze(range_M, zero_const).output(0),
        ov_opset.concat(
            [
                ov_opset.unsqueeze(B_node, zero_const).output(0),
                ov_opset.unsqueeze(M_node, zero_const).output(0),
            ],
            axis=0,
        ).output(0),
    ).output(0)

    loop = ov_opset.loop(
        K_node, ov_opset.constant(True, Type.boolean).output(0)
    )

    A_body_param = ov_opset.parameter(
        A_flat.get_partial_shape(), out_ov_type, name="A_curr"
    )
    P_body_param = ov_opset.parameter(
        P_init.get_partial_shape(), Type.i32, name="P_curr"
    )
    k_param = ov_opset.parameter([], Type.i32, name="k")
    B_body_param = ov_opset.parameter([], Type.i32, name="B_node_body")
    M_body_param = ov_opset.parameter([], Type.i32, name="M_node_body")
    N_body_param = ov_opset.parameter([], Type.i32, name="N_node_body")

    A_curr = A_body_param.output(0)
    P_curr = P_body_param.output(0)
    k = k_param.output(0)
    k_plus_1 = ov_opset.add(k, one_const).output(0)
    B_body = B_body_param.output(0)
    M_body = M_body_param.output(0)
    N_body = N_body_param.output(0)

    A_k_col_slice1 = ov_opset.slice(
        A_curr,
        ov_opset.unsqueeze(k, zero_const).output(0),
        ov_opset.unsqueeze(M_body, zero_const).output(0),
        ov_opset.constant([1], Type.i32).output(0),
        ov_opset.constant([1], Type.i32).output(0),
    ).output(0)

    A_k_col_elem = ov_opset.slice(
        A_k_col_slice1,
        ov_opset.unsqueeze(k, zero_const).output(0),
        ov_opset.unsqueeze(k_plus_1, zero_const).output(0),
        ov_opset.constant([1], Type.i32).output(0),
        ov_opset.constant([2], Type.i32).output(0),
    ).output(0)

    A_k_col_abs = ov_opset.abs(A_k_col_elem).output(0)

    topk = ov_opset.topk(
        A_k_col_abs, ov_opset.constant(1, Type.i32), 1, "max", "value"
    )
    p_rel = topk.output(1)
    p_rel_sq = ov_opset.squeeze(
        p_rel, ov_opset.constant([1, 2], Type.i32).output(0)
    ).output(0)
    p = ov_opset.add(p_rel_sq, k).output(0)

    indices = ov_opset.broadcast(
        ov_opset.unsqueeze(
            ov_opset.range(
                zero_const, M_body, one_const, output_type=Type.i32
            ).output(0),
            zero_const,
        ).output(0),
        ov_opset.concat(
            [
                ov_opset.unsqueeze(B_body, zero_const).output(0),
                ov_opset.unsqueeze(M_body, zero_const).output(0),
            ],
            axis=0,
        ).output(0),
    ).output(0)

    B_range = ov_opset.range(
        zero_const, B_body, one_const, output_type=Type.i32
    ).output(0)
    b_unsqueezed = ov_opset.unsqueeze(B_range, one_const).output(0)
    k_b = ov_opset.broadcast(
        ov_opset.unsqueeze(k, zero_const).output(0),
        ov_opset.unsqueeze(B_body, zero_const).output(0),
    ).output(0)
    k_b_unsqueezed = ov_opset.unsqueeze(k_b, one_const).output(0)
    p_unsqueezed = ov_opset.unsqueeze(p, one_const).output(0)

    upd_k_indices = ov_opset.concat([b_unsqueezed, k_b_unsqueezed], 1).output(0)
    upd_p_indices = ov_opset.concat([b_unsqueezed, p_unsqueezed], 1).output(0)

    indices = ov_opset.scatter_nd_update(indices, upd_k_indices, p).output(0)
    indices = ov_opset.scatter_nd_update(indices, upd_p_indices, k_b).output(0)

    A_curr_flat = ov_opset.reshape(
        A_curr, ov_opset.constant([-1], Type.i32).output(0), False
    ).output(0)

    b_coords = ov_opset.broadcast(
        ov_opset.unsqueeze(
            ov_opset.unsqueeze(B_range, one_const).output(0),
            ov_opset.constant(2, Type.i32).output(0),
        ).output(0),
        ov_opset.concat(
            [
                ov_opset.unsqueeze(B_body, zero_const).output(0),
                ov_opset.unsqueeze(M_body, zero_const).output(0),
                ov_opset.unsqueeze(N_body, zero_const).output(0),
            ],
            0,
        ).output(0),
    ).output(0)

    r_coords = ov_opset.broadcast(
        ov_opset.unsqueeze(
            indices, ov_opset.constant(2, Type.i32).output(0)
        ).output(0),
        ov_opset.concat(
            [
                ov_opset.unsqueeze(B_body, zero_const).output(0),
                ov_opset.unsqueeze(M_body, zero_const).output(0),
                ov_opset.unsqueeze(N_body, zero_const).output(0),
            ],
            0,
        ).output(0),
    ).output(0)

    c_coords = ov_opset.broadcast(
        ov_opset.unsqueeze(
            ov_opset.unsqueeze(
                ov_opset.range(
                    zero_const, N_body, one_const, output_type=Type.i32
                ).output(0),
                zero_const,
            ).output(0),
            zero_const,
        ).output(0),
        ov_opset.concat(
            [
                ov_opset.unsqueeze(B_body, zero_const).output(0),
                ov_opset.unsqueeze(M_body, zero_const).output(0),
                ov_opset.unsqueeze(N_body, zero_const).output(0),
            ],
            0,
        ).output(0),
    ).output(0)

    slice_stride = ov_opset.multiply(M_body, N_body).output(0)

    flat_indices = ov_opset.add(
        ov_opset.add(
            ov_opset.multiply(b_coords, slice_stride).output(0),
            ov_opset.multiply(r_coords, N_body).output(0),
        ).output(0),
        c_coords,
    ).output(0)

    flat_indices_1d = ov_opset.reshape(
        flat_indices, ov_opset.constant([-1], Type.i32).output(0), False
    ).output(0)

    A_swapped_flat = ov_opset.gather(
        A_curr_flat, flat_indices_1d, zero_const
    ).output(0)
    A_swapped = ov_opset.reshape(
        A_swapped_flat,
        ov_opset.concat(
            [
                ov_opset.unsqueeze(B_body, zero_const).output(0),
                ov_opset.unsqueeze(M_body, zero_const).output(0),
                ov_opset.unsqueeze(N_body, zero_const).output(0),
            ],
            0,
        ).output(0),
        False,
    ).output(0)

    A_k_k = ov_opset.slice(
        A_swapped,
        ov_opset.unsqueeze(k, zero_const).output(0),
        ov_opset.unsqueeze(k_plus_1, zero_const).output(0),
        ov_opset.constant([1], Type.i32).output(0),
        ov_opset.constant([1], Type.i32).output(0),
    ).output(0)
    A_k_k = ov_opset.slice(
        A_k_k,
        ov_opset.unsqueeze(k, zero_const).output(0),
        ov_opset.unsqueeze(k_plus_1, zero_const).output(0),
        ov_opset.constant([1], Type.i32).output(0),
        ov_opset.constant([2], Type.i32).output(0),
    ).output(0)

    A_k1_end_k = ov_opset.slice(
        A_swapped,
        ov_opset.unsqueeze(k_plus_1, zero_const).output(0),
        ov_opset.unsqueeze(M_body, zero_const).output(0),
        ov_opset.constant([1], Type.i32).output(0),
        ov_opset.constant([1], Type.i32).output(0),
    ).output(0)
    A_k1_end_k = ov_opset.slice(
        A_k1_end_k,
        ov_opset.unsqueeze(k, zero_const).output(0),
        ov_opset.unsqueeze(k_plus_1, zero_const).output(0),
        ov_opset.constant([1], Type.i32).output(0),
        ov_opset.constant([2], Type.i32).output(0),
    ).output(0)

    zero_f32 = ov_opset.constant(0.0, out_ov_type).output(0)
    is_zero = ov_opset.equal(A_k_k, zero_f32).output(0)
    A_k_k_safe = ov_opset.select(
        is_zero, ov_opset.constant(1e-10, out_ov_type).output(0), A_k_k
    ).output(0)

    L_col = ov_opset.divide(A_k1_end_k, A_k_k_safe).output(0)

    pads_begin = ov_opset.concat(
        [
            ov_opset.constant([0], Type.i32).output(0),
            ov_opset.unsqueeze(k_plus_1, zero_const).output(0),
            ov_opset.unsqueeze(k, zero_const).output(0),
        ],
        0,
    ).output(0)
    pads_end = ov_opset.concat(
        [
            ov_opset.constant([0], Type.i32).output(0),
            ov_opset.constant([0], Type.i32).output(0),
            ov_opset.unsqueeze(
                ov_opset.subtract(N_body, k_plus_1).output(0), zero_const
            ).output(0),
        ],
        0,
    ).output(0)

    L_col_padded = ov_opset.pad(L_col, pads_begin, pads_end, "constant").output(
        0
    )

    row_range_body = ov_opset.range(
        zero_const, M_body, one_const, output_type=Type.i32
    ).output(0)
    col_range_body = ov_opset.range(
        zero_const, N_body, one_const, output_type=Type.i32
    ).output(0)
    row_mask = ov_opset.greater(row_range_body, k).output(0)
    col_mask = ov_opset.equal(col_range_body, k).output(0)

    row_mask_3d = ov_opset.broadcast(
        ov_opset.unsqueeze(
            ov_opset.unsqueeze(row_mask, zero_const).output(0),
            ov_opset.constant(2, Type.i32).output(0),
        ).output(0),
        ov_opset.shape_of(A_curr, Type.i32).output(0),
    ).output(0)
    col_mask_3d = ov_opset.broadcast(
        ov_opset.unsqueeze(
            ov_opset.unsqueeze(col_mask, zero_const).output(0),
            ov_opset.constant(1, Type.i32).output(0),
        ).output(0),
        ov_opset.shape_of(A_curr, Type.i32).output(0),
    ).output(0)
    update_mask = ov_opset.logical_and(row_mask_3d, col_mask_3d).output(0)

    A_masked = ov_opset.select(update_mask, L_col_padded, A_swapped).output(0)

    A_k_k1_end = ov_opset.slice(
        A_swapped,
        ov_opset.unsqueeze(k, zero_const).output(0),
        ov_opset.unsqueeze(k_plus_1, zero_const).output(0),
        ov_opset.constant([1], Type.i32).output(0),
        ov_opset.constant([1], Type.i32).output(0),
    ).output(0)
    A_k_k1_end = ov_opset.slice(
        A_k_k1_end,
        ov_opset.unsqueeze(k_plus_1, zero_const).output(0),
        ov_opset.unsqueeze(N_body, zero_const).output(0),
        ov_opset.constant([1], Type.i32).output(0),
        ov_opset.constant([2], Type.i32).output(0),
    ).output(0)

    outer_prod = ov_opset.matmul(L_col, A_k_k1_end, False, False).output(0)

    outer_prod_padded = ov_opset.pad(
        outer_prod,
        ov_opset.concat(
            [
                ov_opset.constant([0], Type.i32).output(0),
                ov_opset.unsqueeze(k_plus_1, zero_const).output(0),
                ov_opset.unsqueeze(k_plus_1, zero_const).output(0),
            ],
            0,
        ).output(0),
        ov_opset.constant([0, 0, 0], Type.i32).output(0),
        "constant",
    ).output(0)

    A_next = ov_opset.subtract(A_masked, outer_prod_padded).output(0)

    P_next = ov_opset.scatter_nd_update(P_curr, upd_k_indices, p).output(0)

    k_next = ov_opset.add(k, one_const).output(0)
    cond = ov_opset.constant(True, Type.boolean).output(0)

    body = Model(
        [A_next, P_next, k_next, cond],
        [
            A_body_param,
            P_body_param,
            k_param,
            B_body_param,
            M_body_param,
            N_body_param,
        ],
    )
    loop.set_function(body)
    loop.set_special_body_ports([-1, 3])

    loop.set_merged_input(A_body_param, A_flat, A_next)
    loop.set_merged_input(P_body_param, P_init, P_next)
    loop.set_merged_input(k_param, zero_const, k_next)
    loop.set_invariant_input(B_body_param, B_node)
    loop.set_invariant_input(M_body_param, M_node)
    loop.set_invariant_input(N_body_param, N_node)

    A_out = loop.get_iter_value(A_next, -1)
    P_out = loop.get_iter_value(P_next, -1)

    P_out = ov_opset.slice(
        P_out,
        ov_opset.constant([0], Type.i32).output(0),
        ov_opset.unsqueeze(K_node, zero_const).output(0),
        ov_opset.constant([1], Type.i32).output(0),
        ov_opset.constant([1], Type.i32).output(0),
    ).output(0)

    if rank == 2:
        A_out_final = ov_opset.squeeze(A_out, zero_const).output(0)
        P_out_final = ov_opset.squeeze(P_out, zero_const).output(0)
    else:
        A_out_shape = ov_opset.concat(
            [
                batch_shape,
                ov_opset.unsqueeze(M_node, zero_const).output(0),
                ov_opset.unsqueeze(N_node, zero_const).output(0),
            ],
            0,
        ).output(0)
        P_out_shape = ov_opset.concat(
            [batch_shape, ov_opset.unsqueeze(K_node, zero_const).output(0)], 0
        ).output(0)
        A_out_final = ov_opset.reshape(A_out, A_out_shape, False).output(0)
        P_out_final = ov_opset.reshape(P_out, P_out_shape, False).output(0)

    if original_type == Type.f64:
        A_out_final = ov_opset.convert(A_out_final, Type.f64).output(0)

    return (
        OpenVINOKerasTensor(A_out_final),
        OpenVINOKerasTensor(P_out_final),
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
            axis_for_const = list(axis) if isinstance(axis, tuple) else axis
            axis_const = ov_opset.constant(axis_for_const, Type.i32).output(0)
            norm_result = ov_opset.reduce_l2(x_ov, axis_const, keepdims).output(
                0
            )
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
            axis_for_const = list(axis) if isinstance(axis, tuple) else axis
            axis_const = ov_opset.constant(axis_for_const, Type.i32).output(0)
            norm_result = ov_opset.reduce_l1(x_ov, axis_const, keepdims).output(
                0
            )
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
    orig_type = x_ov.get_element_type()

    # Work in f32:
    #   f64 — constant-folding bug in OpenVINO CPU Loop evaluate (same as det())
    #   f16/bf16 — upcast to f32 for numerical stability in iterative
    #              Householder
    #   complex/other — not supported for QR; convert best-effort to f32
    if orig_type != Type.f32:
        x_ov = ov_opset.convert(x_ov, Type.f32).output(0)
    work_type = Type.f32

    rank = x_ov.get_partial_shape().rank.get_length()

    # Scalar and 1-D integer constants
    SLICE_END = 2**30  # large sentinel for "slice to end of dimension"

    zero_s = ov_opset.constant(0, Type.i32).output(0)
    one_s = ov_opset.constant(1, Type.i32).output(0)
    zero_1d = ov_opset.constant([0], Type.i32).output(0)
    one_1d = ov_opset.constant([1], Type.i32).output(0)
    large_1d = ov_opset.constant([SLICE_END], Type.i32).output(0)
    axes012 = ov_opset.constant([0, 1, 2], Type.i32).output(0)
    step111 = ov_opset.constant([1, 1, 1], Type.i32).output(0)

    x_shape = ov_opset.shape_of(x_ov, output_type="i32").output(0)
    m_1d = ov_opset.gather(
        x_shape, ov_opset.constant([-2], Type.i32), zero_s
    ).output(0)
    n_1d = ov_opset.gather(
        x_shape, ov_opset.constant([-1], Type.i32), zero_s
    ).output(0)
    m_s = ov_opset.squeeze(m_1d, zero_s).output(0)
    n_s = ov_opset.squeeze(n_1d, zero_s).output(0)

    # Flatten batch dims → [B, M, N]
    if rank == 2:
        batch_1d = ov_opset.constant([1], Type.i32).output(0)
    else:
        batch_shape = ov_opset.slice(
            x_shape,
            zero_1d,
            ov_opset.constant([-2], Type.i32),
            one_1d,
            zero_1d,
        ).output(0)
        batch_s = ov_opset.reduce_prod(batch_shape, zero_s, False).output(0)
        batch_1d = ov_opset.unsqueeze(batch_s, zero_s).output(0)

    flat_shape = ov_opset.concat([batch_1d, m_1d, n_1d], 0).output(0)
    x_flat = ov_opset.reshape(x_ov, flat_shape, False).output(0)

    # K = min(M, N) — number of Householder steps
    k_s = ov_opset.minimum(m_s, n_s).output(0)

    # Q = eye(M) broadcast to [B, M, M],  R = x_flat [B, M, N]
    range_m = ov_opset.range(zero_s, m_s, one_s, output_type=Type.i32).output(0)
    eye_m = ov_opset.one_hot(
        range_m,
        m_s,
        ov_opset.constant(1.0, work_type),
        ov_opset.constant(0.0, work_type),
        axis=-1,
    ).output(0)  # [M, M]
    Q_init = ov_opset.broadcast(
        eye_m, ov_opset.concat([batch_1d, m_1d, m_1d], 0).output(0)
    ).output(0)  # [B, M, M]

    # ---- Householder loop ----
    loop = ov_opset.loop(k_s, ov_opset.constant(True, Type.boolean).output(0))

    R_param = ov_opset.parameter(x_flat.get_partial_shape(), work_type, "R")
    Q_param = ov_opset.parameter(Q_init.get_partial_shape(), work_type, "Q")
    k_param = ov_opset.parameter([], Type.i32, "k")

    R_body = R_param.output(0)
    Q_body = Q_param.output(0)
    k_body = k_param.output(0)
    k_1d = ov_opset.unsqueeze(k_body, zero_s).output(0)  # scalar → [1]

    # sub_R = R[:, k:, k:]  →  [B, sub_m, sub_n]
    sub_R = ov_opset.slice(
        R_body,
        ov_opset.concat([zero_1d, k_1d, k_1d], 0).output(0),
        ov_opset.concat([large_1d, large_1d, large_1d], 0).output(0),
        step111,
        axes012,
    ).output(0)

    # x_col = sub_R[:, :, 0]  →  [B, sub_m]
    x_col = ov_opset.gather(
        sub_R, ov_opset.constant(0, Type.i32), ov_opset.constant(2, Type.i32)
    ).output(0)

    # alpha = -sign(x_col[:, 0]) * ||x_col||
    x0 = ov_opset.gather(x_col, ov_opset.constant(0, Type.i32), one_s).output(0)
    sign_x0 = ov_opset.sign(x0).output(0)
    zero_f = ov_opset.constant(0.0, work_type).output(0)
    one_f = ov_opset.constant(1.0, work_type).output(0)
    sign_x0 = ov_opset.select(
        ov_opset.equal(sign_x0, zero_f), one_f, sign_x0
    ).output(0)
    norm_x = ov_opset.sqrt(
        ov_opset.reduce_sum(
            ov_opset.multiply(x_col, x_col).output(0), one_s, keep_dims=False
        ).output(0)
    ).output(0)
    alpha = ov_opset.negative(
        ov_opset.multiply(sign_x0, norm_x).output(0)
    ).output(0)  # [B]

    # Householder vector: v = x_col - alpha * e_0  (one-hot at position 0)
    # Use one_hot so v keeps the same shape as x_col → shape inference stays
    # clean.
    x_col_sh = ov_opset.shape_of(x_col, output_type="i32").output(0)
    sub_m_from_col = ov_opset.gather(x_col_sh, one_s, zero_s).output(0)
    e0 = ov_opset.one_hot(
        ov_opset.constant([0], Type.i32).output(0),
        sub_m_from_col,
        one_f,
        zero_f,
        axis=-1,
    ).output(0)  # [1, sub_m]
    alpha_2d = ov_opset.unsqueeze(alpha, one_s).output(0)  # [B, 1]
    v = ov_opset.subtract(
        x_col, ov_opset.multiply(alpha_2d, e0).output(0)
    ).output(0)  # [B, sub_m]

    # v_hat = v / ||v||
    norm_v = ov_opset.sqrt(
        ov_opset.reduce_sum(
            ov_opset.multiply(v, v).output(0), one_s, keep_dims=True
        ).output(0)
    ).output(0)  # [B, 1]
    eps = ov_opset.constant(1e-12, work_type).output(0)
    v_hat = ov_opset.divide(v, ov_opset.maximum(norm_v, eps).output(0)).output(
        0
    )

    # When sub_m == 1 the LAPACK convention is tau=0 (identity reflector).
    # Zero out v_hat so the Householder step is a no-op for that iteration.
    is_trivial = ov_opset.equal(
        sub_m_from_col, ov_opset.constant(1, Type.i32).output(0)
    ).output(0)  # bool scalar
    scale = ov_opset.subtract(
        one_f,
        ov_opset.convert(is_trivial, work_type).output(0),
    ).output(0)  # 1.0 normally, 0.0 when sub_m==1
    v_hat = ov_opset.multiply(v_hat, scale).output(0)

    v_col = ov_opset.unsqueeze(v_hat, ov_opset.constant(2, Type.i32)).output(
        0
    )  # [B, sub_m, 1]
    v_row = ov_opset.unsqueeze(v_hat, one_s).output(0)  # [B, 1, sub_m]

    two_f = ov_opset.constant(2.0, work_type).output(0)

    # Apply H to sub_R: sub_R -= 2 * v_col @ (v_row @ sub_R)
    vTR = ov_opset.matmul(v_row, sub_R, False, False).output(0)  # [B, 1, sub_n]
    sub_R_new = ov_opset.subtract(
        sub_R,
        ov_opset.multiply(
            two_f, ov_opset.matmul(v_col, vTR, False, False).output(0)
        ).output(0),
    ).output(0)  # [B, sub_m, sub_n]

    # Apply H to Q columns k..: Q[:, :, k:] -= 2 * (Q[:, :, k:] @ v_col) @ v_row
    Q_sub = ov_opset.slice(
        Q_body,
        ov_opset.concat([zero_1d, zero_1d, k_1d], 0).output(0),
        ov_opset.concat([large_1d, large_1d, large_1d], 0).output(0),
        step111,
        axes012,
    ).output(0)  # [B, M, sub_m]
    Qv = ov_opset.matmul(Q_sub, v_col, False, False).output(0)  # [B, M, 1]
    Q_sub_new = ov_opset.subtract(
        Q_sub,
        ov_opset.multiply(
            two_f, ov_opset.matmul(Qv, v_row, False, False).output(0)
        ).output(0),
    ).output(0)  # [B, M, sub_m]

    # Reconstruct R_next: keep top rows and left cols, replace bottom-right
    # block
    top_R = ov_opset.slice(
        R_body,
        ov_opset.concat([zero_1d, zero_1d, zero_1d], 0).output(0),
        ov_opset.concat([large_1d, k_1d, large_1d], 0).output(0),
        step111,
        axes012,
    ).output(0)  # [B, k, N]
    left_bot_R = ov_opset.slice(
        R_body,
        ov_opset.concat([zero_1d, k_1d, zero_1d], 0).output(0),
        ov_opset.concat([large_1d, large_1d, k_1d], 0).output(0),
        step111,
        axes012,
    ).output(0)  # [B, sub_m, k]
    R_next = ov_opset.concat(
        [top_R, ov_opset.concat([left_bot_R, sub_R_new], 2).output(0)],
        1,
    ).output(0)  # [B, M, N]

    # Reconstruct Q_next: keep left cols, replace right cols
    left_Q = ov_opset.slice(
        Q_body,
        ov_opset.concat([zero_1d, zero_1d, zero_1d], 0).output(0),
        ov_opset.concat([large_1d, large_1d, k_1d], 0).output(0),
        step111,
        axes012,
    ).output(0)  # [B, M, k]
    Q_next = ov_opset.concat([left_Q, Q_sub_new], 2).output(0)  # [B, M, M]

    cond_next = ov_opset.constant(True, Type.boolean).output(0)
    k_next = ov_opset.add(k_body, one_s).output(0)

    body_model = ov.Model(
        [cond_next, R_next, Q_next, k_next],
        [R_param, Q_param, k_param],
        "householder_qr",
    )
    loop.set_function(body_model)
    loop.set_special_body_ports([-1, 0])

    loop.set_merged_input(R_param, x_flat, R_next)
    loop.set_merged_input(Q_param, Q_init, Q_next)
    loop.set_merged_input(
        k_param, ov_opset.constant(0, Type.i32).output(0), k_next
    )

    R_out = loop.get_iter_value(R_next, -1)
    Q_out = loop.get_iter_value(Q_next, -1)

    # Reshape immediately after the loop to restore concrete shape information
    # (loop body slices with dynamic k cause output shape to become dynamic).
    Q_out = ov_opset.reshape(
        Q_out, ov_opset.concat([batch_1d, m_1d, m_1d], 0).output(0), False
    ).output(0)  # [B, M, M]
    R_out = ov_opset.reshape(
        R_out, ov_opset.concat([batch_1d, m_1d, n_1d], 0).output(0), False
    ).output(0)  # [B, M, N]

    k_1d_out = ov_opset.unsqueeze(k_s, zero_s).output(0)

    # Trim to requested mode: Q [B,M,K], R [B,K,N]
    # (complete keeps [B,M,M],[B,M,N])
    if mode == "reduced":
        Q_out = ov_opset.slice(
            Q_out,
            ov_opset.constant([0, 0, 0], Type.i32).output(0),
            ov_opset.concat([large_1d, large_1d, k_1d_out], 0).output(0),
            step111,
            axes012,
        ).output(0)  # [B, M, K]
        R_out = ov_opset.slice(
            R_out,
            ov_opset.constant([0, 0, 0], Type.i32).output(0),
            ov_opset.concat([large_1d, k_1d_out, large_1d], 0).output(0),
            step111,
            axes012,
        ).output(0)  # [B, K, N]

    # Restore original batch shape using reshape (not squeeze) to keep
    # concrete dims
    if rank == 2:
        if mode == "reduced":
            q_shape_2d = ov_opset.concat([m_1d, k_1d_out], 0).output(0)
            r_shape_2d = ov_opset.concat([k_1d_out, n_1d], 0).output(0)
        else:
            q_shape_2d = ov_opset.concat([m_1d, m_1d], 0).output(0)
            r_shape_2d = ov_opset.concat([m_1d, n_1d], 0).output(0)
        Q_out = ov_opset.reshape(Q_out, q_shape_2d, False).output(0)
        R_out = ov_opset.reshape(R_out, r_shape_2d, False).output(0)
    elif rank > 2:
        batch_shape_node = ov_opset.slice(
            ov_opset.shape_of(x_ov, output_type="i32").output(0),
            zero_1d,
            ov_opset.constant([-2], Type.i32),
            one_1d,
            zero_1d,
        ).output(0)
        q_last = ov_opset.gather(
            ov_opset.shape_of(Q_out, output_type="i32").output(0),
            ov_opset.constant([-1], Type.i32),
            zero_s,
        ).output(0)
        r_second_last = ov_opset.gather(
            ov_opset.shape_of(R_out, output_type="i32").output(0),
            ov_opset.constant([-2], Type.i32),
            zero_s,
        ).output(0)
        Q_out = ov_opset.reshape(
            Q_out,
            ov_opset.concat([batch_shape_node, m_1d, q_last], 0).output(0),
            False,
        ).output(0)
        R_out = ov_opset.reshape(
            R_out,
            ov_opset.concat([batch_shape_node, r_second_last, n_1d], 0).output(
                0
            ),
            False,
        ).output(0)

    if orig_type != work_type:
        Q_out = ov_opset.convert(Q_out, orig_type).output(0)
        R_out = ov_opset.convert(R_out, orig_type).output(0)

    return OpenVINOKerasTensor(Q_out), OpenVINOKerasTensor(R_out)


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
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    a_ov = get_ov_output(a)
    b_ov = get_ov_output(b)

    orig_type = a_ov.get_element_type()
    if orig_type == Type.f64:
        a_ov = ov_opset.convert(a_ov, Type.f32).output(0)
        b_ov = ov_opset.convert(b_ov, Type.f32).output(0)
    work_type = a_ov.get_element_type()

    squeeze = (
        b_ov.get_partial_shape().rank.get_length()
        == a_ov.get_partial_shape().rank.get_length() - 1
    )
    if squeeze:
        b_ov = ov_opset.unsqueeze(
            b_ov, ov_opset.constant([-1], Type.i32).output(0)
        ).output(0)

    zero_s = ov_opset.constant(0, Type.i32).output(0)
    one_s = ov_opset.constant(1, Type.i32).output(0)
    a_shape = ov_opset.shape_of(a_ov, output_type="i32").output(0)
    n_s = ov_opset.squeeze(
        ov_opset.gather(
            a_shape, ov_opset.constant([-1], Type.i32), zero_s
        ).output(0),
        zero_s,
    ).output(0)

    range_n = ov_opset.range(zero_s, n_s, one_s, output_type=Type.i32).output(0)
    row_idx = ov_opset.unsqueeze(
        range_n, ov_opset.constant(1, Type.i32)
    ).output(0)
    col_idx = ov_opset.unsqueeze(
        range_n, ov_opset.constant(0, Type.i32)
    ).output(0)

    if lower:
        mask = ov_opset.greater_equal(row_idx, col_idx).output(0)
    else:
        mask = ov_opset.less_equal(row_idx, col_idx).output(0)

    mask_f = ov_opset.convert(mask, work_type).output(0)
    a_ov = ov_opset.multiply(a_ov, mask_f).output(0)

    a_inv = ov_opset.inverse(a_ov, adjoint=False).output(0)
    result = ov_opset.matmul(a_inv, b_ov, False, False).output(0)

    if squeeze:
        result = ov_opset.squeeze(
            result, ov_opset.constant([-1], Type.i32).output(0)
        ).output(0)

    if orig_type == Type.f64:
        result = ov_opset.convert(result, Type.f64).output(0)

    return OpenVINOKerasTensor(result)


def svd(x, full_matrices=True, compute_uv=True):
    raise NotImplementedError("`svd` is not supported with openvino backend")


def lstsq(a, b, rcond=None):
    raise NotImplementedError("`lstsq` is not supported with openvino backend")


def matrix_rank(x, tol=None):
    raise NotImplementedError(
        "`matrix_rank` is not supported with openvino backend"
    )


def pinv(x, rcond=None):
    raise NotImplementedError("`pinv` is not supported with openvino backend")


def jvp(fun, primals, tangents, has_aux=False):
    raise NotImplementedError("`jvp` is not supported with openvino backend")
