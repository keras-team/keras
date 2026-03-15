import openvino.opset15 as ov_opset
from openvino import Model
from openvino import Type

from keras.src import tree
from keras.src.backend.openvino.core import OpenVINOKerasTensor
from keras.src.backend.openvino.core import get_ov_output


def rnn(
    step_function,
    inputs,
    initial_states,
    go_backwards=False,
    mask=None,
    constants=None,
    unroll=False,
    input_length=None,
    time_major=False,
    zero_output_for_mask=False,
    return_all_outputs=True,
):
    def swap_batch_timestep(input_t):
        axes = list(range(len(input_t.shape)))
        axes[0], axes[1] = 1, 0
        perm_const = ov_opset.constant(axes, dtype=Type.i32).output(0)
        input_ov = get_ov_output(input_t)
        return OpenVINOKerasTensor(
            ov_opset.transpose(input_ov, perm_const).output(0)
        )

    if not time_major:
        inputs = tree.map_structure(swap_batch_timestep, inputs)
        if mask is not None:
            mask = swap_batch_timestep(mask)
    flattened_inputs = tree.flatten(inputs)
    flattened_states = tree.flatten(initial_states)
    flattened_constants = tree.flatten(constants) if constants else []
    input_0 = flattened_inputs[0]
    input_0_ov = get_ov_output(input_0)
    input_shape = ov_opset.shape_of(input_0_ov, Type.i32).output(0)
    time_steps = ov_opset.gather(
        input_shape,
        ov_opset.constant([0], dtype=Type.i32).output(0),
        ov_opset.constant(0, dtype=Type.i32).output(0),
    ).output(0)
    time_steps = ov_opset.squeeze(
        time_steps, ov_opset.constant([0], dtype=Type.i32).output(0)
    ).output(0)
    if mask is None and input_length is not None:
        input_len_ov = get_ov_output(input_length)
        if input_len_ov.get_partial_shape().rank.get_length() == 1:
            indices = ov_opset.range(
                ov_opset.constant(0, dtype=Type.i32).output(0),
                time_steps,
                ov_opset.constant(1, dtype=Type.i32).output(0),
                output_type=Type.i32,
            ).output(0)
            indices = ov_opset.unsqueeze(
                indices, ov_opset.constant([1], dtype=Type.i32).output(0)
            ).output(0)
            input_len_casted = ov_opset.convert(input_len_ov, Type.i32).output(
                0
            )
            input_len_expanded = ov_opset.unsqueeze(
                input_len_casted,
                ov_opset.constant([0], dtype=Type.i32).output(0),
            ).output(0)
            mask_bool = ov_opset.less(indices, input_len_expanded).output(0)
            mask = OpenVINOKerasTensor(mask_bool)
    if mask is not None:
        mask_ov = get_ov_output(mask)
        if mask_ov.get_element_type() != Type.boolean:
            mask_ov = ov_opset.convert(mask_ov, Type.boolean).output(0)
        pshape = mask_ov.get_partial_shape()
        rank = pshape.rank.get_length()
        if rank == 2:
            mask_ov = ov_opset.unsqueeze(
                mask_ov, ov_opset.constant([-1], dtype=Type.i32).output(0)
            ).output(0)
        mask = OpenVINOKerasTensor(mask_ov)
    if go_backwards:

        def reverse_time(x):
            x_ov = get_ov_output(x)
            start = ov_opset.constant([0], dtype=Type.i32).output(0)
            idx = ov_opset.range(
                ov_opset.subtract(
                    time_steps, ov_opset.constant(1, dtype=Type.i32).output(0)
                ).output(0),
                ov_opset.constant(-1, dtype=Type.i32).output(0),
                ov_opset.constant(-1, dtype=Type.i32).output(0),
                output_type=Type.i32,
            ).output(0)
            return OpenVINOKerasTensor(
                ov_opset.gather(x_ov, idx, start).output(0)
            )

        inputs = tree.map_structure(reverse_time, inputs)
        if mask is not None:
            mask = reverse_time(mask)
        flattened_inputs = tree.flatten(inputs)

    def _slice_at_0(x):
        x_ov = get_ov_output(x)
        slice_0 = ov_opset.gather(
            x_ov,
            ov_opset.constant(0, dtype=Type.i32).output(0),
            ov_opset.constant(0, dtype=Type.i32).output(0),
        ).output(0)
        return OpenVINOKerasTensor(slice_0)

    inputs_0 = tree.map_structure(_slice_at_0, inputs)
    output_0, _ = step_function(
        inputs_0, tuple(initial_states) + tuple(constants or [])
    )
    flattened_output_0 = tree.flatten(output_0)
    last_output_states = []
    for out in flattened_output_0:
        out_ov = get_ov_output(out)
        shape = ov_opset.shape_of(out_ov, Type.i32).output(0)
        dtype = out_ov.get_element_type()
        zeros = ov_opset.broadcast(
            ov_opset.constant(0, dtype).output(0), shape
        ).output(0)
        last_output_states.append(OpenVINOKerasTensor(zeros))
    params = []
    sliced_inputs_params = []
    for inp in flattened_inputs:
        inp_ov = get_ov_output(inp)
        pshape = inp_ov.get_partial_shape()
        if pshape.rank.is_static:
            new_shape = [1] + list(pshape)[1:]
        else:
            new_shape = None
        param = ov_opset.parameter(new_shape, inp_ov.get_element_type())
        sliced_inputs_params.append(param)
        params.append(param)
    sliced_mask_params = []
    if mask is not None:
        mask_ov = get_ov_output(mask)
        pshape = mask_ov.get_partial_shape()
        new_shape = [1] + list(pshape)[1:] if pshape.rank.is_static else None
        param = ov_opset.parameter(new_shape, mask_ov.get_element_type())
        sliced_mask_params.append(param)
        params.append(param)
    merged_states_params = []
    for st in flattened_states:
        st_ov = get_ov_output(st)
        param = ov_opset.parameter(
            st_ov.get_partial_shape(), st_ov.get_element_type()
        )
        merged_states_params.append(param)
        params.append(param)
    last_output_params = []
    for lo in last_output_states:
        lo_ov = get_ov_output(lo)
        param = ov_opset.parameter(
            lo_ov.get_partial_shape(), lo_ov.get_element_type()
        )
        last_output_params.append(param)
        params.append(param)
    constants_params = []
    for c in flattened_constants:
        c_ov = get_ov_output(c)
        param = ov_opset.parameter(
            c_ov.get_partial_shape(), c_ov.get_element_type()
        )
        constants_params.append(param)
        params.append(param)
    sliced_inputs_t = [
        OpenVINOKerasTensor(
            ov_opset.squeeze(
                p.output(0),
                ov_opset.constant([0], dtype=Type.i32).output(0),
            ).output(0)
        )
        for p in sliced_inputs_params
    ]
    merged_states_t = [
        OpenVINOKerasTensor(p.output(0)) for p in merged_states_params
    ]
    constants_t = [OpenVINOKerasTensor(p.output(0)) for p in constants_params]

    packed_inputs = tree.pack_sequence_as(inputs, sliced_inputs_t)
    packed_states = tree.pack_sequence_as(initial_states, merged_states_t)
    step_output, step_new_states = step_function(
        packed_inputs, tuple(packed_states) + tuple(constants_t)
    )
    flat_step_output = tree.flatten(step_output)
    flat_step_new_states = tree.flatten(step_new_states)
    final_output_list = []
    final_states_list = []
    final_last_output_list = []
    if mask is not None:
        mask_t = ov_opset.squeeze(
            sliced_mask_params[0].output(0),
            ov_opset.constant([0], dtype=Type.i32).output(0),
        ).output(0)
        for i, (new_st, old_st) in enumerate(
            zip(flat_step_new_states, merged_states_t)
        ):
            new_st_ov = get_ov_output(new_st)
            old_st_ov = get_ov_output(old_st)
            res = ov_opset.select(mask_t, new_st_ov, old_st_ov).output(0)
            final_states_list.append(res)
        for i, (new_out, old_last_out) in enumerate(
            zip(flat_step_output, last_output_params)
        ):
            new_out_ov = get_ov_output(new_out)
            old_last_out_ov = old_last_out.output(0)
            last_out_res = ov_opset.select(
                mask_t, new_out_ov, old_last_out_ov
            ).output(0)
            final_last_output_list.append(last_out_res)
            if zero_output_for_mask:
                zero = ov_opset.broadcast(
                    ov_opset.constant(0, new_out_ov.get_element_type()).output(
                        0
                    ),
                    ov_opset.shape_of(new_out_ov, Type.i32).output(0),
                ).output(0)
                seq_out_res = ov_opset.select(mask_t, new_out_ov, zero).output(
                    0
                )
            else:
                seq_out_res = last_out_res
            final_output_list.append(seq_out_res)
    else:
        final_states_list = [get_ov_output(x) for x in flat_step_new_states]
        final_output_list = [get_ov_output(x) for x in flat_step_output]
        final_last_output_list = [get_ov_output(x) for x in flat_step_output]
    unsq_ax = ov_opset.constant([0], dtype=Type.i32).output(0)
    final_output_list = [
        ov_opset.unsqueeze(x, unsq_ax).output(0) for x in final_output_list
    ]
    cond_const = ov_opset.constant(True, Type.boolean).output(0)
    results = (
        [cond_const]
        + final_states_list
        + final_last_output_list
        + final_output_list
    )
    body_model = Model(results, params)
    exec_cond_in = ov_opset.constant(True, Type.boolean).output(0)
    loop = ov_opset.loop(time_steps, exec_cond_in)
    loop.set_function(body_model)
    loop.set_special_body_ports([-1, 0])
    for param, inp in zip(sliced_inputs_params, flattened_inputs):
        loop.set_sliced_input(param, get_ov_output(inp), 0, 1, 1, -1, 0)
    if mask is not None:
        mask_ov = get_ov_output(mask)
        loop.set_sliced_input(sliced_mask_params[0], mask_ov, 0, 1, 1, -1, 0)
    current_res_idx = 1
    for param, init in zip(merged_states_params, flattened_states):
        loop.set_merged_input(
            param, get_ov_output(init), results[current_res_idx]
        )
        current_res_idx += 1
    final_last_outputs_res = []
    for param, init in zip(last_output_params, last_output_states):
        loop.set_merged_input(
            param, get_ov_output(init), results[current_res_idx]
        )
        final_last_outputs_res.append(results[current_res_idx])
        current_res_idx += 1
    for param, val in zip(constants_params, flattened_constants):
        loop.set_invariant_input(param, get_ov_output(val))
    loop_outputs = []
    for _ in final_output_list:
        out = loop.get_concatenated_slices(
            results[current_res_idx], 0, 1, 1, -1, 0
        )
        loop_outputs.append(OpenVINOKerasTensor(out))
        current_res_idx += 1
    loop_final_states = []
    st_res_idx = 1
    for _ in flattened_states:
        out = loop.get_iter_value(results[st_res_idx], -1)
        loop_final_states.append(OpenVINOKerasTensor(out))
        st_res_idx += 1
    lo_res_idx = st_res_idx
    loop_final_last_outputs = []
    for _ in last_output_states:
        out = loop.get_iter_value(results[lo_res_idx], -1)
        loop_final_last_outputs.append(OpenVINOKerasTensor(out))
        lo_res_idx += 1
    outputs = tree.pack_sequence_as(output_0, loop_outputs)
    new_states = tree.pack_sequence_as(initial_states, loop_final_states)
    last_output = tree.pack_sequence_as(output_0, loop_final_last_outputs)
    if not time_major:
        outputs = tree.map_structure(swap_batch_timestep, outputs)
    return last_output, outputs, new_states


def _reorder_gates(x_ov, from_order, to_order, axis):
    """Reorder gate slices of a tensor along `axis`.

    `from_order` and `to_order` are lists of single-char gate names, e.g.
    from_order=['i','f','c','o'], to_order=['f','i','c','o'].
    The tensor dimension along `axis` must be divisible by len(from_order).
    """
    n_gates = len(from_order)
    axis_const = ov_opset.constant(axis, dtype=Type.i32).output(0)
    chunks = ov_opset.split(x_ov, axis_const, n_gates).outputs()
    gate_map = {g: chunks[i] for i, g in enumerate(from_order)}
    reordered = [gate_map[g] for g in to_order]
    return ov_opset.concat(reordered, axis=axis).output(0)


def _seq_lengths(inputs_ov):
    """Return int32 sequence-length tensor [batch] equal to full time steps."""
    input_shape = ov_opset.shape_of(inputs_ov, Type.i32).output(0)
    batch_size = ov_opset.gather(
        input_shape,
        ov_opset.constant([0], dtype=Type.i32).output(0),
        ov_opset.constant(0, dtype=Type.i32).output(0),
    ).output(0)
    time_steps = ov_opset.gather(
        input_shape,
        ov_opset.constant([1], dtype=Type.i32).output(0),
        ov_opset.constant(0, dtype=Type.i32).output(0),
    ).output(0)
    return ov_opset.broadcast(time_steps, batch_size).output(0)


def lstm(
    inputs,
    initial_h,
    initial_c,
    mask,
    kernel,
    recurrent_kernel,
    bias,
    activation,
    recurrent_activation,
    return_sequences=False,
    go_backwards=False,
    unroll=False,
):
    act_name = getattr(activation, "__name__", None)
    rec_act_name = getattr(recurrent_activation, "__name__", None)
    if not (
        act_name == "tanh"
        and rec_act_name == "sigmoid"
        and not unroll
        and bias is not None
        and mask is None
    ):
        raise NotImplementedError

    inputs_ov = get_ov_output(inputs)
    initial_h_ov = get_ov_output(initial_h)
    initial_c_ov = get_ov_output(initial_c)
    kernel_ov = get_ov_output(kernel)
    recurrent_kernel_ov = get_ov_output(recurrent_kernel)
    bias_ov = get_ov_output(bias)

    hidden_size = recurrent_kernel_ov.get_partial_shape()[0].get_length()

    kt = ov_opset.transpose(
        kernel_ov,
        ov_opset.constant([1, 0], dtype=Type.i32).output(0),
    ).output(0)
    w = _reorder_gates(kt, ["i", "f", "c", "o"], ["f", "i", "c", "o"], axis=0)
    w = ov_opset.unsqueeze(
        w, ov_opset.constant([0], dtype=Type.i32).output(0)
    ).output(0)

    rt = ov_opset.transpose(
        recurrent_kernel_ov,
        ov_opset.constant([1, 0], dtype=Type.i32).output(0),
    ).output(0)
    r = _reorder_gates(rt, ["i", "f", "c", "o"], ["f", "i", "c", "o"], axis=0)
    r = ov_opset.unsqueeze(
        r, ov_opset.constant([0], dtype=Type.i32).output(0)
    ).output(0)

    b = _reorder_gates(
        bias_ov, ["i", "f", "c", "o"], ["f", "i", "c", "o"], axis=0
    )
    b = ov_opset.unsqueeze(
        b, ov_opset.constant([0], dtype=Type.i32).output(0)
    ).output(0)

    h0 = ov_opset.unsqueeze(
        initial_h_ov, ov_opset.constant([1], dtype=Type.i32).output(0)
    ).output(0)
    c0 = ov_opset.unsqueeze(
        initial_c_ov, ov_opset.constant([1], dtype=Type.i32).output(0)
    ).output(0)

    seq_lens = _seq_lengths(inputs_ov)
    direction = "reverse" if go_backwards else "forward"

    lstm_out = ov_opset.lstm_sequence(
        inputs_ov, h0, c0, seq_lens, w, r, b, hidden_size, direction
    )
    dir_axis = ov_opset.constant([1], dtype=Type.i32).output(0)
    all_outputs = ov_opset.squeeze(lstm_out.output(0), dir_axis).output(0)
    h_n = ov_opset.squeeze(lstm_out.output(1), dir_axis).output(0)
    c_n = ov_opset.squeeze(lstm_out.output(2), dir_axis).output(0)

    if go_backwards:
        shape = ov_opset.shape_of(all_outputs, Type.i32).output(0)
        time_len = ov_opset.gather(
            shape,
            ov_opset.constant(1, dtype=Type.i32).output(0),
            ov_opset.constant(0, dtype=Type.i32).output(0),
        ).output(0)
        idx = ov_opset.range(
            ov_opset.subtract(
                time_len, ov_opset.constant(1, dtype=Type.i32).output(0)
            ).output(0),
            ov_opset.constant(-1, dtype=Type.i32).output(0),
            ov_opset.constant(-1, dtype=Type.i32).output(0),
            output_type=Type.i32,
        ).output(0)
        all_outputs = ov_opset.gather(
            all_outputs,
            idx,
            ov_opset.constant(1, dtype=Type.i32).output(0),
        ).output(0)

    return (
        OpenVINOKerasTensor(h_n),
        OpenVINOKerasTensor(all_outputs),
        [OpenVINOKerasTensor(h_n), OpenVINOKerasTensor(c_n)],
    )


def gru(
    inputs,
    initial_state,
    mask,
    kernel,
    recurrent_kernel,
    bias,
    activation,
    recurrent_activation,
    return_sequences=False,
    go_backwards=False,
    unroll=False,
    reset_after=True,
):
    act_name = getattr(activation, "__name__", None)
    rec_act_name = getattr(recurrent_activation, "__name__", None)
    if not (
        act_name == "tanh"
        and rec_act_name == "sigmoid"
        and not unroll
        and bias is not None
        and reset_after
        and mask is None
    ):
        raise NotImplementedError

    inputs_ov = get_ov_output(inputs)
    initial_state_ov = get_ov_output(initial_state)
    kernel_ov = get_ov_output(kernel)
    recurrent_kernel_ov = get_ov_output(recurrent_kernel)
    bias_ov = get_ov_output(bias)

    hidden_size = recurrent_kernel_ov.get_partial_shape()[0].get_length()

    w = ov_opset.transpose(
        kernel_ov,
        ov_opset.constant([1, 0], dtype=Type.i32).output(0),
    ).output(0)
    w = ov_opset.unsqueeze(
        w, ov_opset.constant([0], dtype=Type.i32).output(0)
    ).output(0)

    r = ov_opset.transpose(
        recurrent_kernel_ov,
        ov_opset.constant([1, 0], dtype=Type.i32).output(0),
    ).output(0)
    r = ov_opset.unsqueeze(
        r, ov_opset.constant([0], dtype=Type.i32).output(0)
    ).output(0)

    # Keras bias [2, 3*units]: row 0 = input biases [b_z, b_r, b_h],
    # row 1 = recurrent biases [rb_z, rb_r, rb_h].
    # OV gru_sequence (linear_before_reset=True) wants B [1, 4*units]:
    # [b_z+rb_z, b_r+rb_r, b_h, rb_h]
    ax = ov_opset.constant(0, dtype=Type.i32).output(0)
    b_input = ov_opset.gather(
        bias_ov, ov_opset.constant(0, dtype=Type.i32).output(0), ax
    ).output(0)
    b_recur = ov_opset.gather(
        bias_ov, ov_opset.constant(1, dtype=Type.i32).output(0), ax
    ).output(0)
    split_ax = ov_opset.constant(0, dtype=Type.i32).output(0)
    b_in_parts = ov_opset.split(b_input, split_ax, 3).outputs()
    b_rc_parts = ov_opset.split(b_recur, split_ax, 3).outputs()
    b_z = ov_opset.add(b_in_parts[0], b_rc_parts[0]).output(0)
    b_r = ov_opset.add(b_in_parts[1], b_rc_parts[1]).output(0)
    b_h = b_in_parts[2]
    rb_h = b_rc_parts[2]
    b = ov_opset.concat([b_z, b_r, b_h, rb_h], axis=0).output(0)
    b = ov_opset.unsqueeze(
        b, ov_opset.constant([0], dtype=Type.i32).output(0)
    ).output(0)

    h0 = ov_opset.unsqueeze(
        initial_state_ov, ov_opset.constant([1], dtype=Type.i32).output(0)
    ).output(0)

    seq_lens = _seq_lengths(inputs_ov)
    direction = "reverse" if go_backwards else "forward"

    gru_out = ov_opset.gru_sequence(
        inputs_ov,
        h0,
        seq_lens,
        w,
        r,
        b,
        hidden_size,
        direction,
        linear_before_reset=True,
    )
    dir_axis = ov_opset.constant([1], dtype=Type.i32).output(0)
    all_outputs = ov_opset.squeeze(gru_out.output(0), dir_axis).output(0)
    h_n = ov_opset.squeeze(gru_out.output(1), dir_axis).output(0)

    if go_backwards:
        # OV direction="reverse" outputs Y in original time order
        # (Y[0]=fully-accumulated state). Keras go_backwards expects
        # Y[0]=state after first reversed step. Flip time axis to match.
        shape = ov_opset.shape_of(all_outputs, Type.i32).output(0)
        time_len = ov_opset.gather(
            shape,
            ov_opset.constant(1, dtype=Type.i32).output(0),
            ov_opset.constant(0, dtype=Type.i32).output(0),
        ).output(0)
        idx = ov_opset.range(
            ov_opset.subtract(
                time_len, ov_opset.constant(1, dtype=Type.i32).output(0)
            ).output(0),
            ov_opset.constant(-1, dtype=Type.i32).output(0),
            ov_opset.constant(-1, dtype=Type.i32).output(0),
            output_type=Type.i32,
        ).output(0)
        all_outputs = ov_opset.gather(
            all_outputs,
            idx,
            ov_opset.constant(1, dtype=Type.i32).output(0),
        ).output(0)

    return (
        OpenVINOKerasTensor(h_n),
        OpenVINOKerasTensor(all_outputs),
        [OpenVINOKerasTensor(h_n)],
    )


def cudnn_ok(*args, **kwargs):
    return False
