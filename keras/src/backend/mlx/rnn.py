import contextlib

import mlx.core as mx

from keras.src import tree
from keras.src.backend.common import stateless_scope
from keras.src.backend.mlx.core import convert_to_tensor
from keras.src.backend.mlx.core import reverse_sequence
from keras.src.backend.mlx.core import scan
from keras.src.backend.mlx.core import unstack


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
        # Swap the batch and timestep dim for the incoming tensor.
        axes = list(range(len(input_t.shape)))
        axes[0], axes[1] = 1, 0
        return mx.transpose(input_t, axes)

    if not time_major:
        inputs = tree.map_structure(swap_batch_timestep, inputs)

    flattened_inputs = tree.flatten(inputs)
    time_steps = flattened_inputs[0].shape[0]

    if mask is not None:
        if mask.dtype != mx.bool_:
            mask = mask.astype(mx.bool_)
        if len(mask.shape) == 2:
            mask = mx.expand_dims(mask, axis=-1)
        if not time_major:
            mask = swap_batch_timestep(mask)

    if constants is None:
        constants = []

    def _expand_mask(mask_t, input_t, fixed_dim=1):
        if tree.is_nested(mask_t):
            raise ValueError(
                f"mask_t is expected to be tensor, but got {mask_t}"
            )
        if tree.is_nested(input_t):
            raise ValueError(
                f"input_t is expected to be tensor, but got {input_t}"
            )
        rank_diff = len(input_t.shape) - len(mask_t.shape)
        for _ in range(rank_diff):
            mask_t = mx.expand_dims(mask_t, axis=-1)
        multiples = [1] * fixed_dim + list(input_t.shape[fixed_dim:])
        return mx.tile(mask_t, multiples)

    if unroll:
        if not time_steps:
            raise ValueError("Unrolling requires a fixed number of timesteps.")
        states = tuple(initial_states)
        successive_states = []
        successive_outputs = []

        # Process the input tensors. The input tensor need to be split on the
        # time_step dim, and reverse if go_backwards is True. In the case of
        # nested input, the input is flattened and then transformed
        # individually.  The result of this will be a tuple of lists, each of
        # the item in tuple is list of the tensor with shape (batch, feature)
        def _process_single_input_t(input_t):
            input_t = unstack(input_t)  # unstack for time_step dim
            if go_backwards:
                input_t.reverse()
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
            mask_list = unstack(mask)
            if go_backwards:
                mask_list.reverse()

            for i in range(time_steps):
                inp = _get_input_tensor(i)
                mask_t = mask_list[i]
                output, new_states = step_function(
                    inp, tuple(states) + tuple(constants)
                )
                tiled_mask_t = _expand_mask(mask_t, output)

                if not successive_outputs:
                    prev_output = mx.zeros_like(output)
                else:
                    prev_output = successive_outputs[-1]

                output = mx.where(tiled_mask_t, output, prev_output)

                flat_states = tree.flatten(states)
                flat_new_states = tree.flatten(new_states)
                tiled_mask_t = tuple(
                    _expand_mask(mask_t, s) for s in flat_states
                )
                flat_final_states = tuple(
                    mx.where(m, s, ps)
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
            outputs = mx.stack(successive_outputs)

        else:  # mask is None
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
            outputs = mx.stack(successive_outputs)

    else:  # Unroll == False
        if mask is not None:

            def _step(states, current_input):
                current_input, current_mask = current_input
                is_masked = mx.all(
                    mx.logical_not(current_mask), axis=-1, keepdims=True
                )

                output_t, new_states = step_function(current_input, states)

                if zero_output_for_mask:
                    masked_outs = mx.where(
                        is_masked, mx.zeros_like(output_t), output_t
                    )
                else:
                    # Assume the first state is the previous output.
                    output_tm1 = states[0]
                    if tree.is_nested(output_tm1):
                        # Stacked RNN case: assume first state of last cell.
                        output_tm1 = states[-1][0]
                    masked_outs = mx.where(is_masked, output_tm1, output_t)

                new_states = tree.map_structure(
                    lambda s, ns: mx.where(is_masked, s, ns),
                    states,
                    new_states,
                )
                return (new_states, masked_outs)

            scan_xs = (inputs, mask)

        else:

            def _step(states, current_input):
                output_t, new_states = step_function(current_input, states)
                return new_states, output_t

            scan_xs = inputs
        if stateless_scope.in_stateless_scope():
            # Reuse the existing parent stateless scope.
            scope = contextlib.nullcontext()
        else:
            scope = stateless_scope.StatelessScope()
        with scope:
            new_states, outputs = scan(
                f=_step,
                init=initial_states,
                xs=scan_xs,
                reverse=go_backwards,
            )

        if go_backwards:
            outputs = reverse_sequence(outputs)

        last_output = outputs[-1]

    if not time_major:
        outputs = tree.map_structure(swap_batch_timestep, outputs)

    return last_output, outputs, new_states


def cudnn_ok(*args, **kwargs):
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
    time_major=False,
):
    """LSTM implementation for MLX backend.

    Args:
        inputs: Input tensor of shape (batch, timesteps, features) or
            (timesteps, batch, features) if time_major=True.
        initial_state_h: Initial hidden state of shape (batch, units).
        initial_state_c: Initial cell state of shape (batch, units).
        mask: Optional mask tensor of shape (batch, timesteps).
        kernel: Input kernel weights of shape (features, units * 4).
        recurrent_kernel: Recurrent kernel weights of shape (units, units * 4).
        bias: Bias weights of shape (units * 4,) or None.
        activation: Activation function for cell state (typically tanh).
        recurrent_activation: Activation function for gates (typically sigmoid).
        return_sequences: Whether to return all outputs or just the last.
        go_backwards: Whether to process the sequence backwards.
        unroll: Whether to unroll the loop (unused, for API compatibility).
        time_major: Whether inputs are time-major (timesteps, batch, features).

    Returns:
        Tuple of (last_output, outputs, [final_h, final_c]).
    """
    # Masking requires special handling (zero_output_for_mask, etc.) that
    # the generic rnn() function handles. Fall back to generic implementation.
    if mask is not None:
        raise NotImplementedError(
            "MLX LSTM does not support masking. "
            "Use the generic RNN implementation instead."
        )

    # Convert inputs and weights to MLX tensors
    inputs = convert_to_tensor(inputs)
    initial_state_h = convert_to_tensor(initial_state_h)
    initial_state_c = convert_to_tensor(initial_state_c)
    kernel = convert_to_tensor(kernel)
    recurrent_kernel = convert_to_tensor(recurrent_kernel)
    if bias is not None:
        bias = convert_to_tensor(bias)

    # Handle time_major format
    if not time_major:
        # Convert from (batch, time, features) to (time, batch, features)
        inputs = mx.transpose(inputs, (1, 0, 2))

    seq_len = inputs.shape[0]
    hidden = initial_state_h
    cell = initial_state_c

    # Precompute input projections: (time, batch, 4*units)
    # kernel is (features, 4*units)
    x_proj = mx.matmul(inputs, kernel)
    if bias is not None:
        x_proj = x_proj + bias

    # Process sequence
    all_outputs = []
    indices = range(seq_len)
    if go_backwards:
        indices = reversed(indices)

    for t in indices:
        x_t = x_proj[t]  # (batch, 4*units)

        # Compute gates: x_t + h @ recurrent_kernel
        gates = x_t + mx.matmul(hidden, recurrent_kernel)

        # Split into i, f, c, o gates
        units = hidden.shape[-1]
        i = gates[:, :units]
        f = gates[:, units : units * 2]
        c_candidate = gates[:, units * 2 : units * 3]
        o = gates[:, units * 3 :]

        # Apply activations
        i = recurrent_activation(i)
        f = recurrent_activation(f)
        c_candidate = activation(c_candidate)
        o = recurrent_activation(o)

        # Update cell and hidden states
        cell = f * cell + i * c_candidate
        hidden = o * activation(cell)

        all_outputs.append(hidden)

    # Stack outputs
    if go_backwards:
        all_outputs = all_outputs[::-1]
    outputs = mx.stack(all_outputs, axis=0)  # (time, batch, units)

    # Convert back to batch-major if needed
    if not time_major:
        outputs = mx.transpose(outputs, (1, 0, 2))  # (batch, time, units)

    # Get last output
    if go_backwards:
        last_output = outputs[:, 0, :] if not time_major else outputs[0]
    else:
        last_output = outputs[:, -1, :] if not time_major else outputs[-1]

    if not return_sequences:
        outputs = last_output

    return last_output, outputs, [hidden, cell]


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
    time_major=False,
):
    """GRU implementation for MLX backend.

    Args:
        inputs: Input tensor of shape (batch, timesteps, features) or
            (timesteps, batch, features) if time_major=True.
        initial_state: Initial hidden state of shape (batch, units).
        mask: Optional mask tensor of shape (batch, timesteps).
        kernel: Input kernel weights of shape (features, units * 3).
        recurrent_kernel: Recurrent kernel weights of shape (units, units * 3).
        bias: Bias weights. If reset_after=True, shape is (2, units * 3).
            If reset_after=False, shape is (units * 3,).
        activation: Activation function for output (typically tanh).
        recurrent_activation: Activation function for gates (typically sigmoid).
        return_sequences: Whether to return all outputs or just the last.
        go_backwards: Whether to process the sequence backwards.
        unroll: Whether to unroll the loop (unused, for API compatibility).
        reset_after: GRU convention. True applies reset gate after matrix
            multiplication (cuDNN compatible). False applies before.
        time_major: Whether inputs are time-major (timesteps, batch, features).

    Returns:
        Tuple of (last_output, outputs, [final_hidden]).
    """
    # Masking requires special handling (zero_output_for_mask, etc.) that
    # the generic rnn() function handles. Fall back to generic implementation.
    if mask is not None:
        raise NotImplementedError(
            "MLX GRU does not support masking. "
            "Use the generic RNN implementation instead."
        )

    # Convert inputs and weights to MLX tensors
    inputs = convert_to_tensor(inputs)
    initial_state = convert_to_tensor(initial_state)
    kernel = convert_to_tensor(kernel)
    recurrent_kernel = convert_to_tensor(recurrent_kernel)
    if bias is not None:
        bias = convert_to_tensor(bias)

    # Handle time_major format
    if not time_major:
        # Convert from (batch, time, features) to (time, batch, features)
        inputs = mx.transpose(inputs, (1, 0, 2))

    seq_len = inputs.shape[0]
    hidden = initial_state
    units = hidden.shape[-1]

    # Handle bias
    input_bias = None
    recurrent_bias = None
    if bias is not None:
        if reset_after:
            # bias shape is (2, 3*units)
            input_bias = bias[0]  # (3*units,)
            recurrent_bias = bias[1]  # (3*units,)
        else:
            input_bias = bias

    # Precompute input projections: (time, batch, 3*units)
    x_proj = mx.matmul(inputs, kernel)
    if input_bias is not None:
        x_proj = x_proj + input_bias

    # Process sequence
    all_outputs = []
    indices = range(seq_len)
    if go_backwards:
        indices = reversed(indices)

    for t in indices:
        x_t = x_proj[t]  # (batch, 3*units)

        # Split input projection into z, r, h gates
        x_z = x_t[:, :units]
        x_r = x_t[:, units : units * 2]
        x_h = x_t[:, units * 2 :]

        # Compute recurrent projection
        h_proj = mx.matmul(hidden, recurrent_kernel)
        if reset_after and recurrent_bias is not None:
            h_proj = h_proj + recurrent_bias

        h_z = h_proj[:, :units]
        h_r = h_proj[:, units : units * 2]
        h_h = h_proj[:, units * 2 :]

        # Compute gates
        z = recurrent_activation(x_z + h_z)  # update gate
        r = recurrent_activation(x_r + h_r)  # reset gate

        # Compute candidate hidden state
        if reset_after:
            # Apply reset gate after matrix multiplication
            h_candidate = activation(x_h + r * h_h)
        else:
            # Apply reset gate before matrix multiplication
            recurrent_h = mx.matmul(
                r * hidden, recurrent_kernel[:, units * 2 :]
            )
            h_candidate = activation(x_h + recurrent_h)

        # Update hidden state
        hidden = (1 - z) * h_candidate + z * hidden

        all_outputs.append(hidden)

    # Stack outputs
    if go_backwards:
        all_outputs = all_outputs[::-1]
    outputs = mx.stack(all_outputs, axis=0)  # (time, batch, units)

    # Convert back to batch-major if needed
    if not time_major:
        outputs = mx.transpose(outputs, (1, 0, 2))  # (batch, time, units)

    # Get last output
    if go_backwards:
        last_output = outputs[:, 0, :] if not time_major else outputs[0]
    else:
        last_output = outputs[:, -1, :] if not time_major else outputs[-1]

    if not return_sequences:
        outputs = last_output

    return last_output, outputs, [hidden]
