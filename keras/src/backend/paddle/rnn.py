import paddle

from keras.src import tree
from keras.src.backend.paddle.core import convert_to_tensor


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
    input_length = input_length or inputs.shape[1]

    def swap_batch_timestep(input_t):
        axes = list(range(len(input_t.shape)))
        axes[0], axes[1] = 1, 0
        return paddle.transpose(input_t, axes)

    if not time_major:
        inputs = tree.map_structure(swap_batch_timestep, inputs)

    flattened_inputs = tree.flatten(inputs)
    time_steps = flattened_inputs[0].shape[0]
    time_steps_t = time_steps

    if mask is not None:
        if mask.dtype != paddle.bool:
            mask = mask.cast(paddle.bool)
        if len(mask.shape) == 2:
            mask = paddle.unsqueeze(mask, -1)
        if not time_major:
            mask = swap_batch_timestep(mask)

    if constants is None:
        constants = []

    def _expand_mask(mask_t, input_t, fixed_dim=1):
        rank_diff = len(input_t.shape) - len(mask_t.shape)
        for _ in range(rank_diff):
            mask_t = paddle.unsqueeze(mask_t, -1)
        multiples = [1] * fixed_dim + list(input_t.shape[fixed_dim:])
        return paddle.tile(mask_t, multiples)

    if unroll:
        if not time_steps:
            raise ValueError("Unrolling requires a fixed number of timesteps.")
        states = tuple(initial_states)
        successive_states = []
        successive_outputs = []

        def _process_single_input_t(input_t):
            input_t = paddle.unbind(input_t)
            if go_backwards:
                input_t = input_t[::-1]
            return input_t

        if tree.is_nested(inputs):
            processed_input = tree.map_structure(
                _process_single_input_t, inputs
            )
        else:
            processed_input = (_process_single_input_t(inputs),)

        def _get_input_tensor(time):
            inp = [t_[time] for t_ in processed_input]
            return tree.pack_sequence_as(inputs, inp)

        if mask is not None:
            mask_list = paddle.unbind(mask)
            if go_backwards:
                mask_list = mask_list[::-1]

            for i in range(time_steps):
                inp = _get_input_tensor(i)
                mask_t = mask_list[i]
                output, new_states = step_function(
                    inp, tuple(states) + tuple(constants)
                )
                tiled_mask_t = _expand_mask(mask_t, output)

                if not successive_outputs:
                    prev_output = paddle.zeros_like(output)
                else:
                    prev_output = successive_outputs[-1]

                output = paddle.where(tiled_mask_t, output, prev_output)

                flat_states = tree.flatten(states)
                flat_new_states = tree.flatten(new_states)
                tiled_mask_t = tuple(
                    _expand_mask(mask_t, s) for s in flat_states
                )
                flat_final_states = tuple(
                    paddle.where(m, s, ps)
                    for m, s, ps in zip(
                        tiled_mask_t, flat_new_states, flat_states
                    )
                )
                states = tree.pack_sequence_as(states, flat_final_states)

                if return_all_outputs:
                    successive_outputs.append(output)
                    successive_states.append(states)
                else:
                    successive_outputs = [output]
                    successive_states = [states]
            last_output = successive_outputs[-1]
            new_states = successive_states[-1]
            outputs = paddle.stack(successive_outputs)

            if zero_output_for_mask:
                last_output = paddle.where(
                    _expand_mask(mask_list[-1], last_output),
                    last_output,
                    paddle.zeros_like(last_output),
                )
                outputs = paddle.where(
                    _expand_mask(mask, outputs, fixed_dim=2),
                    outputs,
                    paddle.zeros_like(outputs),
                )

        else:
            for i in range(time_steps):
                inp = _get_input_tensor(i)
                output, states = step_function(
                    inp, tuple(states) + tuple(constants)
                )
                if return_all_outputs:
                    successive_outputs.append(output)
                    successive_states.append(states)
                else:
                    successive_outputs = [output]
                    successive_states = [states]
            last_output = successive_outputs[-1]
            new_states = successive_states[-1]
            outputs = paddle.stack(successive_outputs)

    else:
        states = tuple(initial_states)

        input_ta = tuple(
            (
                list(paddle.unbind(input_))
                if not go_backwards
                else list(paddle.unbind(paddle.flip(input_, [0])))
            )
            for input_ in flattened_inputs
        )

        input_time_zero = tree.pack_sequence_as(
            inputs, [inp[0] for inp in flattened_inputs]
        )
        output_time_zero, _ = step_function(
            input_time_zero, tuple(initial_states) + tuple(constants)
        )

        output_ta_size = time_steps_t if return_all_outputs else 1
        output_ta = []
        for out in tree.flatten(output_time_zero):
            out_list = list(out)
            if len(out) < output_ta_size:
                out_list.extend([[]] * (output_ta_size - len(out)))
            output_ta.append(out_list)

        time = 0

        if input_length is None:
            max_iterations = time_steps_t
        else:
            if hasattr(input_length, "__len__"):
                input_length = convert_to_tensor(input_length)
                max_iterations = int(paddle.max(input_length).item())
            else:
                max_iterations = input_length

        if mask is not None:
            if go_backwards:
                mask = paddle.flip(mask, [0])

            mask_ta = list(paddle.unbind(mask))

            def masking_fn(t):
                return mask_ta[t]

            def compute_masked_output(mask_t, flat_out, flat_mask):
                tiled_mask_t = tuple(
                    _expand_mask(mask_t, o, fixed_dim=len(mask_t.shape))
                    for o in flat_out
                )
                return tuple(
                    paddle.where(m, o, fm)
                    for m, o, fm in zip(tiled_mask_t, flat_out, flat_mask)
                )

        elif isinstance(input_length, paddle.Tensor):
            if go_backwards:
                max_len = paddle.max(input_length)
                rev_input_length = max_len - 1 - input_length

                def masking_fn(t):
                    return rev_input_length < t
            else:

                def masking_fn(t):
                    return input_length > t

            def compute_masked_output(mask_t, flat_out, flat_mask):
                return tuple(
                    paddle.where(mask_t, o, zo)
                    for o, zo in zip(flat_out, flat_mask)
                )
        else:
            masking_fn = None

        if masking_fn is not None:
            flat_zero_output = tuple(
                paddle.zeros_like(o) for o in tree.flatten(output_time_zero)
            )

            def _step(t, output_ta_t, prev_output, *states):
                current_input = tuple(ta[t] for ta in input_ta)
                current_input = tree.pack_sequence_as(inputs, current_input)
                mask_t = masking_fn(t)
                output, new_states = step_function(
                    current_input, tuple(states) + tuple(constants)
                )
                flat_output = tree.flatten(output)
                flat_mask_output = (
                    flat_zero_output
                    if zero_output_for_mask
                    else tree.flatten(prev_output)
                )
                flat_new_output = compute_masked_output(
                    mask_t, flat_output, flat_mask_output
                )

                flat_state = tree.flatten(states)
                flat_new_state = tree.flatten(new_states)
                flat_final_state = compute_masked_output(
                    mask_t, flat_new_state, flat_state
                )
                new_states = tree.pack_sequence_as(new_states, flat_final_state)

                ta_index_to_write = t if return_all_outputs else 0
                for ta, out in zip(output_ta_t, flat_new_output):
                    ta[ta_index_to_write] = out

                return (t + 1, output_ta_t, tuple(flat_new_output)) + tuple(
                    new_states
                )

            it = 0
            output_ta_t, new_states, prev_output = (
                output_ta,
                states,
                flat_zero_output,
            )
            while time < time_steps_t and it < max_iterations:
                final_outputs = _step(
                    time, output_ta_t, prev_output, *new_states
                )
                time, output_ta_t, prev_output = final_outputs[:3]
                new_states = final_outputs[3:]
                it += 1

        else:

            def _step(t, output_ta_t, *states):
                current_input = tuple(ta[t] for ta in input_ta)
                current_input = tree.pack_sequence_as(inputs, current_input)
                output, new_states = step_function(
                    current_input, tuple(states) + tuple(constants)
                )
                flat_new_state = tree.flatten(new_states)

                flat_output = tree.flatten(output)
                ta_index_to_write = t if return_all_outputs else 0
                for ta, out in zip(output_ta_t, flat_output):
                    ta[ta_index_to_write] = out

                new_states = tree.pack_sequence_as(
                    initial_states, flat_new_state
                )
                return (t + 1, output_ta_t) + tuple(new_states)

            it = 0
            output_ta_t = output_ta
            new_states = states
            while time < time_steps_t and it < max_iterations:
                final_outputs = _step(time, output_ta_t, *new_states)
                time, output_ta_t = final_outputs[:2]
                new_states = final_outputs[2:]
                it += 1

        def _stack(tensor_list):
            max_ndims = max([t.ndim for t in tensor_list])
            max_list = []
            for i, t in enumerate(tensor_list):
                if t.ndim == max_ndims:
                    max_list.append(t)
            return paddle.stack(max_list)

        output_ta = final_outputs[1]

        outputs = tuple(_stack(o) for o in output_ta)
        last_output = tuple(o[-1] for o in outputs)

        outputs = tree.pack_sequence_as(output_time_zero, outputs)
        last_output = tree.pack_sequence_as(output_time_zero, last_output)

    if not time_major:
        outputs = tree.map_structure(swap_batch_timestep, outputs)

    return last_output, outputs, new_states


def cudnn_ok(
    activation,
    recurrent_activation,
    unroll,
    use_bias=True,
):
    return False


def lstm(
    inputs,
    initial_state_h,
    initial_state_c,
    mask,
    kernel,
    recurrent_kernel,
    bias,
    activation,
    recurrent_activation,
    return_sequences=False,
    go_backwards=False,
    unroll=False,
    batch_first=True,
):
    if mask is not None:
        raise NotImplementedError

    kernel = convert_to_tensor(kernel)
    recurrent_kernel = convert_to_tensor(recurrent_kernel)
    if bias is not None:
        bias = convert_to_tensor(bias)

    compute_dtype = kernel.dtype
    inputs = convert_to_tensor(inputs).cast(compute_dtype)
    initial_state_h = convert_to_tensor(initial_state_h).cast(compute_dtype)
    initial_state_c = convert_to_tensor(initial_state_c).cast(compute_dtype)

    if go_backwards:
        seq_dim = 1 if batch_first else 0
        inputs = paddle.flip(inputs, axis=[seq_dim])

    return _fallback_lstm(
        inputs,
        initial_state_h,
        initial_state_c,
        kernel,
        recurrent_kernel,
        bias,
        activation,
        recurrent_activation,
        return_sequences,
        batch_first,
    )


def _fallback_lstm(
    inputs,
    initial_state_h,
    initial_state_c,
    kernel,
    recurrent_kernel,
    bias,
    activation,
    recurrent_activation,
    return_sequences,
    batch_first,
):
    if batch_first:
        inputs = paddle.transpose(inputs, [1, 0, 2])

    x_proj = paddle.matmul(inputs, kernel)
    if bias is not None:
        x_proj = x_proj + bias

    time_steps = inputs.shape[0]
    h = initial_state_h
    c = initial_state_c
    outputs = []

    for t in range(time_steps):
        z = x_proj[t] + paddle.matmul(h, recurrent_kernel)
        z_i, z_f, z_c, z_o = paddle.split(z, 4, axis=1)

        new_c = recurrent_activation(z_f) * c + recurrent_activation(
            z_i
        ) * activation(z_c)
        new_h = recurrent_activation(z_o) * activation(new_c)

        h = new_h
        c = new_c
        outputs.append(h)

    outputs = paddle.stack(outputs, axis=0)

    if batch_first:
        outputs = paddle.transpose(outputs, [1, 0, 2])

    last_output = h

    if not return_sequences:
        outputs = paddle.unsqueeze(last_output, 1 if batch_first else 0)

    return last_output, outputs, [h, c]


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
    if mask is not None:
        raise NotImplementedError

    kernel = convert_to_tensor(kernel)
    recurrent_kernel = convert_to_tensor(recurrent_kernel)
    if bias is not None:
        bias = convert_to_tensor(bias)

    compute_dtype = kernel.dtype
    inputs = convert_to_tensor(inputs).cast(compute_dtype)
    initial_state = convert_to_tensor(initial_state).cast(compute_dtype)

    if go_backwards:
        inputs = paddle.flip(inputs, axis=[1])

    return _fallback_gru(
        inputs,
        initial_state,
        kernel,
        recurrent_kernel,
        bias,
        activation,
        recurrent_activation,
        return_sequences,
        reset_after,
    )


def _fallback_gru(
    inputs,
    initial_state,
    kernel,
    recurrent_kernel,
    bias,
    activation,
    recurrent_activation,
    return_sequences,
    reset_after=True,
):
    inputs = paddle.transpose(inputs, [1, 0, 2])

    x_proj = paddle.matmul(inputs, kernel)

    time_steps = inputs.shape[0]
    h = initial_state
    outputs = []

    for t in range(time_steps):
        h_proj = paddle.matmul(h, recurrent_kernel)

        if reset_after and bias is not None:
            x_z, x_r, x_h = paddle.split(x_proj[t] + bias[0], 3, axis=1)
            h_z, h_r, h_h = paddle.split(h_proj + bias[1], 3, axis=1)
        else:
            x_input = x_proj[t]
            if bias is not None:
                x_input = x_input + bias
            x_z, x_r, x_h = paddle.split(x_input, 3, axis=1)
            h_z, h_r, h_h = paddle.split(h_proj, 3, axis=1)

        z = recurrent_activation(x_z + h_z)
        r = recurrent_activation(x_r + h_r)
        hh = activation(x_h + r * h_h)
        h = z * h + (1.0 - z) * hh

        outputs.append(h)

    outputs = paddle.stack(outputs, axis=0)
    outputs = paddle.transpose(outputs, [1, 0, 2])

    last_output = h
    if not return_sequences:
        outputs = paddle.unsqueeze(last_output, 1)

    return last_output, outputs, [h]


def bidirectional_lstm(
    inputs,
    fwd_initial_state_h,
    fwd_initial_state_c,
    bwd_initial_state_h,
    bwd_initial_state_c,
    mask,
    fwd_kernel,
    fwd_recurrent_kernel,
    fwd_bias,
    bwd_kernel,
    bwd_recurrent_kernel,
    bwd_bias,
    activation,
    recurrent_activation,
    return_sequences=False,
    unroll=False,
):
    fwd_last, fwd_outputs, fwd_states = lstm(
        inputs,
        fwd_initial_state_h,
        fwd_initial_state_c,
        mask,
        fwd_kernel,
        fwd_recurrent_kernel,
        fwd_bias,
        activation,
        recurrent_activation,
        return_sequences=return_sequences,
        go_backwards=False,
        unroll=unroll,
        batch_first=True,
    )
    bwd_last, bwd_outputs, bwd_states = lstm(
        inputs,
        bwd_initial_state_h,
        bwd_initial_state_c,
        mask,
        bwd_kernel,
        bwd_recurrent_kernel,
        bwd_bias,
        activation,
        recurrent_activation,
        return_sequences=return_sequences,
        go_backwards=True,
        unroll=unroll,
        batch_first=True,
    )

    if return_sequences:
        bwd_outputs = paddle.flip(bwd_outputs, axis=[1])

    return (
        (fwd_last, fwd_outputs, fwd_states),
        (bwd_last, bwd_outputs, bwd_states),
    )


def bidirectional_gru(
    inputs,
    fwd_initial_state,
    bwd_initial_state,
    mask,
    fwd_kernel,
    fwd_recurrent_kernel,
    fwd_bias,
    bwd_kernel,
    bwd_recurrent_kernel,
    bwd_bias,
    activation,
    recurrent_activation,
    return_sequences=False,
    unroll=False,
    reset_after=True,
):
    fwd_last, fwd_outputs, fwd_states = gru(
        inputs,
        fwd_initial_state,
        mask,
        fwd_kernel,
        fwd_recurrent_kernel,
        fwd_bias,
        activation,
        recurrent_activation,
        return_sequences=return_sequences,
        go_backwards=False,
        unroll=unroll,
        reset_after=reset_after,
    )
    bwd_last, bwd_outputs, bwd_states = gru(
        inputs,
        bwd_initial_state,
        mask,
        bwd_kernel,
        bwd_recurrent_kernel,
        bwd_bias,
        activation,
        recurrent_activation,
        return_sequences=return_sequences,
        go_backwards=True,
        unroll=unroll,
        reset_after=reset_after,
    )

    if return_sequences:
        bwd_outputs = paddle.flip(bwd_outputs, axis=[1])

    return (
        (fwd_last, fwd_outputs, fwd_states),
        (bwd_last, bwd_outputs, bwd_states),
    )


def numpy_scan(f, init, xs, reverse=False, mask=None):
    init = convert_to_tensor(init)
    xs = convert_to_tensor(xs)

    if reverse:
        xs = paddle.flip(xs, axis=[0])

    results = []
    carry = init
    for i in range(xs.shape[0]):
        carry, y = f(carry, xs[i])
        if y is not None:
            results.append(y)

    if results:
        stacked = paddle.stack(results, axis=0)
        if reverse:
            stacked = paddle.flip(stacked, axis=[0])
    else:
        stacked = None

    return carry, stacked
