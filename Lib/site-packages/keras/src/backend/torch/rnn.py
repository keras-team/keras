import torch

from keras.src import tree
from keras.src.backend.torch.core import convert_to_tensor


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
        # Swap the batch and timestep dim for the incoming tensor.
        axes = list(range(len(input_t.shape)))
        axes[0], axes[1] = 1, 0
        return torch.permute(input_t, axes)

    if not time_major:
        inputs = tree.map_structure(swap_batch_timestep, inputs)

    flattened_inputs = tree.flatten(inputs)
    time_steps = flattened_inputs[0].shape[0]
    time_steps_t = time_steps

    if mask is not None:
        if mask.dtype != torch.bool:
            mask = mask.type(torch.bool)
        if len(mask.shape) == 2:
            mask = torch.unsqueeze(mask, -1)
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
            mask_t = torch.unsqueeze(mask_t, -1)
        multiples = [1] * fixed_dim + list(input_t.shape[fixed_dim:])
        return torch.tile(mask_t, multiples)

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
            input_t = torch.unbind(input_t)  # unstack for time_step dim
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
            mask_list = torch.unbind(mask)
            if go_backwards:
                mask_list = torch.flip(mask_list, dims=mask_list.shape)

            for i in range(time_steps):
                inp = _get_input_tensor(i)
                mask_t = mask_list[i]
                output, new_states = step_function(
                    inp, tuple(states) + tuple(constants)
                )
                tiled_mask_t = _expand_mask(mask_t, output)

                if not successive_outputs:
                    prev_output = torch.zeros_like(output)
                else:
                    prev_output = successive_outputs[-1]

                output = torch.where(tiled_mask_t, output, prev_output)

                flat_states = tree.flatten(states)
                flat_new_states = tree.flatten(new_states)
                tiled_mask_t = tuple(
                    _expand_mask(mask_t, s) for s in flat_states
                )
                flat_final_states = tuple(
                    torch.where(m, s, ps)
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
            outputs = torch.stack(successive_outputs)

            if zero_output_for_mask:
                last_output = torch.where(
                    _expand_mask(mask_list[-1], last_output),
                    last_output,
                    torch.zeros_like(last_output),
                )
                outputs = torch.where(
                    _expand_mask(mask, outputs, fixed_dim=2),
                    outputs,
                    torch.zeros_like(outputs),
                )

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
            outputs = torch.stack(successive_outputs)

    else:  # Unroll == False
        states = tuple(initial_states)

        # Create input tensor array, if the inputs is nested tensors, then it
        # will be flattened first, and tensor array will be created one per
        # flattened tensor.

        input_ta = tuple(
            (
                list(torch.unbind(input_))
                if not go_backwards
                else list(torch.unbind(torch.flip(input_, [0])))
            )
            for input_ in flattened_inputs
        )

        # Get the time(0) input and compute the output for that.
        input_time_zero = tree.pack_sequence_as(
            inputs, [inp[0] for inp in flattened_inputs]
        )
        # output_time_zero is used to determine the cell output shape.
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

        time = torch.tensor(0, dtype=torch.int32)

        if input_length is None:
            max_iterations = time_steps_t
        else:
            if hasattr(input_length, "__len__"):
                input_length = convert_to_tensor(input_length)
                max_iterations = torch.max(input_length)
            else:
                max_iterations = input_length

        if mask is not None:
            if go_backwards:
                mask = torch.flip(mask, [0])

            mask_ta = list(torch.unbind(mask))

            def masking_fn(time):
                return mask_ta[time]

            def compute_masked_output(mask_t, flat_out, flat_mask):
                tiled_mask_t = tuple(
                    _expand_mask(mask_t, o, fixed_dim=len(mask_t.shape))
                    for o in flat_out
                )
                return tuple(
                    torch.where(m, o, fm)
                    for m, o, fm in zip(tiled_mask_t, flat_out, flat_mask)
                )

        elif isinstance(input_length, torch.Tensor):
            if go_backwards:
                max_len = torch.max(input_length, dim=0)
                rev_input_length = torch.subtract(max_len - 1, input_length)

                def masking_fn(time):
                    return torch.less(rev_input_length, time)

            else:

                def masking_fn(time):
                    return torch.greater(input_length, time)

            def compute_masked_output(mask_t, flat_out, flat_mask):
                return tuple(
                    torch.where(mask_t, o, zo)
                    for (o, zo) in zip(flat_out, flat_mask)
                )

        else:
            masking_fn = None

        if masking_fn is not None:
            # Mask for the T output will be base on the output of T - 1. In the
            # case T = 0, a zero filled tensor will be used.
            flat_zero_output = tuple(
                torch.zeros_like(o) for o in tree.flatten(output_time_zero)
            )

            def _step(time, output_ta_t, prev_output, *states):
                """RNN step function.

                Args:
                    time: Current timestep value.
                    output_ta_t: TensorArray.
                    prev_output: tuple of outputs from time - 1.
                    *states: List of states.

                Returns:
                    Tuple: `(time + 1, output_ta_t, output) + tuple(new_states)`
                """
                current_input = tuple(ta[time] for ta in input_ta)
                # maybe set shape.
                current_input = tree.pack_sequence_as(inputs, current_input)
                mask_t = masking_fn(time)
                output, new_states = step_function(
                    current_input, tuple(states) + tuple(constants)
                )
                # mask output
                flat_output = tree.flatten(output)
                flat_mask_output = (
                    flat_zero_output
                    if zero_output_for_mask
                    else tree.flatten(prev_output)
                )
                flat_new_output = compute_masked_output(
                    mask_t, flat_output, flat_mask_output
                )

                # mask states
                flat_state = tree.flatten(states)
                flat_new_state = tree.flatten(new_states)
                flat_final_state = compute_masked_output(
                    mask_t, flat_new_state, flat_state
                )
                new_states = tree.pack_sequence_as(new_states, flat_final_state)

                ta_index_to_write = time if return_all_outputs else 0
                for ta, out in zip(output_ta_t, flat_new_output):
                    ta[ta_index_to_write] = out

                return (time + 1, output_ta_t, tuple(flat_new_output)) + tuple(
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

            def _step(time, output_ta_t, *states):
                """RNN step function.

                Args:
                    time: Current timestep value.
                    output_ta_t: TensorArray.
                    *states: List of states.

                Returns:
                    Tuple: `(time + 1,output_ta_t) + tuple(new_states)`
                """
                current_input = tuple(ta[time] for ta in input_ta)
                current_input = tree.pack_sequence_as(inputs, current_input)
                output, new_states = step_function(
                    current_input, tuple(states) + tuple(constants)
                )
                flat_new_state = tree.flatten(new_states)

                flat_output = tree.flatten(output)
                ta_index_to_write = time if return_all_outputs else 0
                for ta, out in zip(output_ta_t, flat_output):
                    ta[ta_index_to_write] = out

                new_states = tree.pack_sequence_as(
                    initial_states, flat_new_state
                )
                return (time + 1, output_ta_t) + tuple(new_states)

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
            return torch.stack(max_list)

        output_ta = final_outputs[1]

        outputs = tuple(_stack(o) for o in output_ta)
        last_output = tuple(o[-1] for o in outputs)

        outputs = tree.pack_sequence_as(output_time_zero, outputs)
        last_output = tree.pack_sequence_as(output_time_zero, last_output)

    if not time_major:
        outputs = tree.map_structure(swap_batch_timestep, outputs)

    return last_output, outputs, new_states


def cudnn_ok(*args, **kwargs):
    return False


def lstm(*args, **kwargs):
    raise NotImplementedError


def gru(*args, **kwargs):
    raise NotImplementedError
