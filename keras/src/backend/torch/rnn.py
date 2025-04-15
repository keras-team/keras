import numpy as np
import torch

from keras.src import tree
from keras.src.backend.torch.core import convert_to_tensor
from keras.src.backend.torch.core import get_device


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
                f"mask_t is expected to be tensor,\
                  but got {mask_t}"
            )
        if tree.is_nested(input_t):
            raise ValueError(
                f"input_t is expected to be tensor,\
                  but got {input_t}"
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
            )  # noqa: E501
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
                )  # noqa: E501
                flat_final_states = tuple(
                    torch.where(m, s, ps)
                    for m, s, ps in zip(
                        tiled_mask_t, flat_new_states, flat_states
                    )  # noqa: E501
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
                )  # noqa: E501
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
                    for (o, zo) in zip(flat_out, flat_mask)  # noqa: E501
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
                new_states = tree.pack_sequence_as(new_states, flat_final_state)  # noqa: E501

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
                )  # noqa: E501
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
                )  # noqa: E501
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


def _is_sequence_right_padded(mask):
    """Check the mask tensor and see if it right padded.

    cuDNN uses the sequence length param to skip the tailing
    timestep. If the data is left padded, or not a strict right padding (has
    masked value in the middle of the sequence), then cuDNN won't work
    properly in those cases.

    Left padded data: [[False, False, True, True, True]].
    Right padded data: [[True, True, True, False, False]].
    Mixture of mask/unmasked data: [[True, False, True, False, False]].

    Note that for the mixed data example above, the actually data RNN should see
    are those 2 Trues (index 0 and 2), the index 1 False should be ignored and
    not pollute the internal states.

    Args:
        mask: the Boolean tensor with shape [batch, timestep]

    Returns:
        boolean scalar tensor, whether the mask is strictly right padded.
    """
    # Get max sequence length
    max_seq_length = mask.shape[1]
    # Count True values in each sequence
    count_of_true = torch.sum(mask, dim=1)
    # Create right padded mask
    batch_size = mask.shape[0]
    indices = torch.arange(max_seq_length, device=mask.device).repeat(
        batch_size, 1
    )  # noqa: E501
    right_padded_mask = indices < count_of_true.unsqueeze(1)
    return torch.all(mask == right_padded_mask)


def _has_fully_masked_sequence(mask):
    # Cudnn kernel will error out if the input sequence contains any
    # fully masked data. We walk around this issue by rerouting the computation
    # to standard kernel, until the issue on cudnn side has been fixed.  For a
    # fully masked sequence, it will contain all Falses. To make it easy to
    # check, we inverse the boolean, check if any of the sequence has all True.
    return torch.any(torch.all(~mask, dim=1))


def _assert_valid_mask(mask):
    # Check if mask is valid for cuDNN
    no_fully_masked = ~_has_fully_masked_sequence(mask)
    is_right_padded = _is_sequence_right_padded(mask)
    valid = no_fully_masked & is_right_padded

    if not valid.item():
        error_message = (
            "You are passing a RNN mask that does not correspond to "
            "right-padded sequences, while using cuDNN, which is not "
            "supported. With cuDNN, RNN masks can only be used for "
            "right-padding, e.g. `[[True, True, False, False]]` would "
            "be a valid mask, but any mask that isn't just contiguous "
            "`True`'s on the left and contiguous `False`'s on the right "
            "would be invalid. You can pass `use_cudnn=False` to your "
            "RNN layer to stop using cuDNN (this may be slower)."
        )
        raise ValueError(error_message)


def _compute_sequence_length_from_mask(mask, batch_first):
    """Calculate the sequence length tensor (1-D) based on the masking tensor.

    The masking tensor is a 2D boolean tensor with shape [batch, timestep]. For
    any timestep that should be masked, the corresponding field will be False.
    Consider the following example:
      a = [[True, True, False, False]
           [True, True, True, False]]
    It is a (2, 4) tensor, and the corresponding sequence length result should
    be 1D tensor with value [2, 3]. Note that the masking tensor must be right
    padded that could be checked by, e.g., `is_sequence_right_padded()`.

    Args:
        mask: Boolean tensor with shape [batch, timestep] or [timestep, batch]
            if time_major=True.
        time_major: Boolean, which indicates whether the mask is time major or
            batch major.

    Returns:
        sequence_length: 1D int32 tensor.
    """
    timestep_index = 0 if not batch_first else 1
    return torch.sum(mask.int(), dim=timestep_index)


def prepare_lstm_weights(lstm, kernel, recurrent_kernel, bias, device):
    """Copies kernel and recurrent kernel weights in the Pytorch format
    We split the kernel and recurrent kernel weights, create associated
    torch tensors adapted to be in line with the Cudnn optimization.
    After we have copied the weights, we ensure the paramters are on
    the same device and memory layout is optimized for Cudnn.

    """

    lstm = lstm.to(device)
    hidden_size = lstm.hidden_size

    # Convert gates from Keras [i,f,c,o] to PyTorch [i,f,g,o]
    i_k, f_k, c_k, o_k = np.split(kernel, 4, axis=1)
    weight_ih_data = np.concatenate([i_k, f_k, c_k, o_k], axis=1).T

    i_r, f_r, c_r, o_r = np.split(recurrent_kernel, 4, axis=1)
    weight_hh_data = np.concatenate([i_r, f_r, c_r, o_r], axis=1).T

    if bias is not None:
        # Split Keras combined bias into input and hidden biases
        bias_ih_data = convert_to_tensor(bias, dtype="float32")
        bias_hh_data = torch.zeros_like(bias_ih_data)

    else:
        bias_ih_data = torch.zeros(4 * hidden_size, device=device)
        bias_hh_data = torch.zeros(4 * hidden_size, device=device)

    # Create PyTorch tensors for weights
    weight_ih = convert_to_tensor(weight_ih_data, dtype="float32").contiguous()
    weight_hh = convert_to_tensor(weight_hh_data, dtype="float32").contiguous()
    bias_ih = convert_to_tensor(bias_ih_data, dtype="float32").contiguous()
    bias_hh = convert_to_tensor(bias_hh_data, dtype="float32").contiguous()

    # Ensure the weights are all on the same device
    weight_ih = weight_ih.to(device)
    weight_hh = weight_hh.to(device)
    bias_ih = bias_ih.to(device)
    bias_hh = bias_hh.to(device)

    # Copy Keras weights into Torch's flat weights
    with torch.no_grad():
        lstm.weight_ih_l0.copy_(weight_ih)
        lstm.weight_hh_l0.copy_(weight_hh)
        lstm.bias_ih_l0.copy_(bias_ih)
        lstm.bias_hh_l0.copy_(bias_hh)

    # Optimize the layout
    lstm.flatten_parameters()

    # After prepare_lstm_weights:
    # Force all LSTM parameters to be on the correct device
    for param in lstm.parameters():
        if param.device != device:
            param.data = param.data.to(device)


def _is_cuda_cudnn_available():
    # We check if the cuda device and drivers are available
    return torch.cuda.is_available() and torch.backends.cudnn.is_available()


def cudnn_ok(
    activation,
    recurrent_activation,
    unroll,
    use_bias=True,
):
    from keras.src import activations
    from keras.src import ops

    return (
        activation in (activations.tanh, torch.tanh, ops.tanh)
        and recurrent_activation
        in (activations.sigmoid, torch.sigmoid, ops.sigmoid)  # noqa: E501
        and not unroll
        and use_bias
        and _is_cuda_cudnn_available()
    )


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
    cudnn_supported = cudnn_ok(
        activation,
        recurrent_activation,
        unroll,
        use_bias=bias is not None,
    )

    if not cudnn_supported:
        raise NotImplementedError

    # Get device from inputs
    device = get_device()

    from keras.src.backend.torch import Variable

    if isinstance(kernel, Variable):
        kernel = kernel.value
    if isinstance(recurrent_kernel, Variable):
        recurrent_kernel = recurrent_kernel.value
    if isinstance(bias, Variable):
        bias = bias.value

    # Convert to torch tensors
    inputs = convert_to_tensor(inputs, dtype="float32")
    initial_state_h = convert_to_tensor(initial_state_h, dtype="float32")
    initial_state_c = convert_to_tensor(initial_state_c, dtype="float32")
    if mask is not None:
        mask = convert_to_tensor(mask, dtype="bool")

    # Preprocess for go_backwards by flipping the sequence
    if go_backwards:
        seq_dim = 1 if batch_first else 0
        inputs = torch.flip(inputs, dims=[seq_dim])
        if mask is not None:
            mask = torch.flip(mask, dims=[seq_dim])

    # Move all tensors to the same device
    inputs = inputs.to(device)
    initial_state_h = initial_state_h.to(device)
    initial_state_c = initial_state_c.to(device)
    if mask is not None:
        mask = mask.to(device)

    try:
        return _cudnn_lstm(
            inputs,
            initial_state_h,
            initial_state_c,
            kernel,
            recurrent_kernel,
            bias,
            mask,
            batch_first,
            go_backwards,
            return_sequences,
            device,
        )
    except Exception:
        raise NotImplementedError


def _cudnn_lstm(
    inputs,
    initial_state_h,
    initial_state_c,
    kernel,
    recurrent_kernel,
    bias,
    mask,
    batch_first,
    go_backwards,
    return_sequences,
    device,
):
    if mask is not None:
        _assert_valid_mask(mask)
        sequence_lengths = _compute_sequence_length_from_mask(mask, batch_first)

    # Ensure inputs are in batch_first format for consistency
    if not batch_first:
        inputs = inputs.permute(1, 0, 2)

    seq_axis, batch_axis = (0, 1) if not batch_first else (1, 0)

    # If shape is [batch, hidden]; Make [1, batch, hidden]
    if initial_state_h.dim() == 2:
        initial_state_h = initial_state_h.unsqueeze(0)
        initial_state_c = initial_state_c.unsqueeze(0)
    # If shape is [batch, 1, hidden]
    elif initial_state_h.dim() == 3 and initial_state_h.shape[1] == 1:
        initial_state_h = initial_state_h.permute(1, 0, 2)
        initial_state_c = initial_state_c.permute(1, 0, 2)

    input_size = kernel.shape[0]
    hidden_size = recurrent_kernel.shape[0]

    # Configure LSTM with the provided parameters
    lstm = torch.nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        batch_first=batch_first,
        bidirectional=False,
    )

    prepare_lstm_weights(lstm, kernel, recurrent_kernel, bias, device)

    if mask is not None:
        # Sort and pack
        sorted_lengths, sorted_indices = torch.sort(
            sequence_lengths, descending=True
        )  # noqa: E501
        sorted_inputs = inputs[sorted_indices]
        sorted_initial_h = initial_state_h[:, sorted_indices]
        sorted_initial_c = initial_state_c[:, sorted_indices]

        # Create the packed sequence
        packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(
            sorted_inputs, sorted_lengths.cpu(), batch_first
        )

        # Process with LSTM (which handles the packed sequence correctly)
        packed_outputs, (h_n, c_n) = lstm(
            packed_inputs, (sorted_initial_h, sorted_initial_c)
        )

        # Unpack back to padded tensor
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_outputs, batch_first
        )  # noqa: E501

    else:
        # Run LSTM without packing for fixed-length sequences
        outputs, (h_n, c_n) = lstm(inputs, (initial_state_h, initial_state_c))

    outputs = outputs.detach().clone().cpu()
    h_n = h_n.detach().clone().cpu()
    c_n = c_n.detach().clone().cpu()
    # Reshape hidden states for return
    h_n = h_n.squeeze(batch_axis)
    c_n = c_n.squeeze(batch_axis)

    # Return appropriate outputs based on return_sequences flag

    if mask is not None:
        last_output = h_n
    else:
        last_output = outputs[:, -1] if batch_first else outputs[-1]

    if not return_sequences:
        outputs = (
            last_output.unsqueeze(1)
            if batch_first
            else last_output.unsqueeze(0)
        )  # noqa: E501

    if go_backwards and return_sequences:
        outputs = torch.flip(outputs, dims=[seq_axis])

    return last_output, outputs, [h_n, c_n]


def gru(*args, **kwargs):
    raise NotImplementedError
