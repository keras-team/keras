import tensorflow as tf

from keras.src import tree


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
    """Iterates over the time dimension of a tensor.

    Args:
        step_function: RNN step function.
            Args;
                `input`; Tensor with shape `(samples, ...)` (no time dimension),
                    representing input for the batch of samples at a certain
                    time step.
                `states`; List of tensors.
            Returns;
                `output`; Tensor with shape `(samples, output_dim)`
                    (no time dimension).
                `new_states`; List of tensors, same length and shapes
                    as 'states'. The first state in the list must be the
                    output tensor at the previous timestep.
        inputs: Tensor of temporal data of shape `(samples, time, ...)`
            (at least 3D), or nested tensors, and each of which has shape
            `(samples, time, ...)`.
        initial_states: Tensor with shape `(samples, state_size)`
            (no time dimension), containing the initial values for the states
            used in the step function. In the case that state_size is in a
            nested shape, the shape of initial_states will also follow the
            nested structure.
        go_backwards: Boolean. If `True`, do the iteration over the time
            dimension in reverse order and return the reversed sequence.
        mask: Binary tensor with shape `(samples, time, 1)`,
            with a zero for every element that is masked.
        constants: List of constant values passed at each step.
        unroll: Whether to unroll the RNN or to use a symbolic `while_loop`.
        input_length: An integer or a 1-D Tensor, depending on whether
            the time dimension is fixed-length or not. In case of variable
            length input, it is used for masking in case there's no mask
            specified.
        time_major: Boolean. If `True`, the inputs and outputs will be in shape
            `(timesteps, batch, ...)`, whereas in the False case, it will be
            `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
            efficient because it avoids transposes at the beginning and end of
            the RNN calculation. However, most TensorFlow data is batch-major,
            so by default this function accepts input and emits output in
            batch-major form.
        zero_output_for_mask: Boolean. If `True`, the output for masked timestep
            will be zeros, whereas in the `False` case, output from previous
            timestep is returned.
        return_all_outputs: Boolean. If `True`, return the recurrent outputs for
            all timesteps in the sequence. If `False`, only return the output
            for the last timestep (which consumes less memory).

    Returns:
        A tuple, `(last_output, outputs, new_states)`.
            - `last_output`: the latest output of the rnn,
                with shape `(samples, ...)`.
            - `outputs`:
                - If `return_all_outputs=True`: a tensor with shape
                  `(samples, time, ...)` where each entry `outputs[s, t]` is the
                  output of the step function at time `t` for sample `s`
                - Else, a tensor equal to `last_output` with shape
                  `(samples, 1, ...)`
            - `new_states`: list of tensors, latest states returned by
                the step function, of shape `(samples, ...)`.
    """
    input_length = input_length or inputs.shape[1]

    def swap_batch_timestep(input_t):
        # Swap the batch and timestep dim for the incoming tensor.
        axes = list(range(len(input_t.shape)))
        axes[0], axes[1] = 1, 0
        return tf.transpose(input_t, axes)

    if not time_major:
        inputs = tree.map_structure(swap_batch_timestep, inputs)

    flattened_inputs = tree.flatten(inputs)
    time_steps = flattened_inputs[0].shape[0]
    time_steps_t = (
        tf.shape(flattened_inputs[0])[0] if time_steps is None else time_steps
    )

    for input_ in flattened_inputs:
        input_.shape.with_rank_at_least(3)

    if mask is not None:
        if mask.dtype != tf.bool:
            mask = tf.cast(mask, tf.bool)
        if len(mask.shape) == 2:
            mask = tf.expand_dims(mask, axis=-1)
        if not time_major:
            mask = swap_batch_timestep(mask)

    if constants is None:
        constants = []

    # tf.where needs its condition tensor to be the same shape as its two
    # result tensors, but in our case the condition (mask) tensor is
    # (nsamples, 1), and inputs are (nsamples, ndimensions) or even more.
    # So we need to broadcast the mask to match the shape of inputs.
    # That's what the tile call does, it just repeats the mask along its
    # second dimension n times.
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
            mask_t = tf.expand_dims(mask_t, -1)
        multiples = [1] * fixed_dim + input_t.shape.as_list()[fixed_dim:]
        return tf.tile(mask_t, multiples)

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
            input_t = tf.unstack(input_t)  # unstack for time_step dim
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
            mask_list = tf.unstack(mask)
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
                    prev_output = tf.zeros_like(output)
                else:
                    prev_output = successive_outputs[-1]

                output = tf.where(tiled_mask_t, output, prev_output)

                flat_states = tree.flatten(states)
                flat_new_states = tree.flatten(new_states)
                tiled_mask_t = tuple(
                    _expand_mask(mask_t, s) for s in flat_states
                )
                flat_final_states = tuple(
                    tf.where(m, s, ps)
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
            outputs = tf.stack(successive_outputs)

            if zero_output_for_mask:
                last_output = tf.where(
                    _expand_mask(mask_list[-1], last_output),
                    last_output,
                    tf.zeros_like(last_output),
                )
                outputs = tf.where(
                    _expand_mask(mask, outputs, fixed_dim=2),
                    outputs,
                    tf.zeros_like(outputs),
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
            outputs = tf.stack(successive_outputs)

    else:  # Unroll == False
        states = tuple(initial_states)

        # Create input tensor array, if the inputs is nested tensors, then it
        # will be flattened first, and tensor array will be created one per
        # flattened tensor.
        input_ta = tuple(
            tf.TensorArray(
                dtype=inp.dtype,
                size=time_steps_t,
                tensor_array_name=f"input_ta_{i}",
            )
            for i, inp in enumerate(flattened_inputs)
        )
        input_ta = tuple(
            (
                ta.unstack(input_)
                if not go_backwards
                else ta.unstack(tf.reverse(input_, [0]))
            )
            for ta, input_ in zip(input_ta, flattened_inputs)
        )

        # Get the time(0) input and compute the output for that, the output will
        # be used to determine the dtype of output tensor array. Don't read from
        # input_ta due to TensorArray clear_after_read default to True.
        input_time_zero = tree.pack_sequence_as(
            inputs, [inp[0] for inp in flattened_inputs]
        )
        # output_time_zero is used to determine the cell output shape and its
        # dtype.  the value is discarded.
        output_time_zero, _ = step_function(
            input_time_zero, tuple(initial_states) + tuple(constants)
        )

        output_ta_size = time_steps_t if return_all_outputs else 1
        output_ta = tuple(
            tf.TensorArray(
                dtype=out.dtype,
                size=output_ta_size,
                element_shape=out.shape,
                tensor_array_name=f"output_ta_{i}",
            )
            for i, out in enumerate(tree.flatten(output_time_zero))
        )

        time = tf.constant(0, dtype="int32", name="time")

        if input_length is None:
            max_iterations = time_steps_t
        else:
            max_iterations = tf.reduce_max(input_length)

        while_loop_kwargs = {
            "cond": lambda time, *_: time < time_steps_t,
            "maximum_iterations": max_iterations,
            "parallel_iterations": 32,
            "swap_memory": True,
        }
        if mask is not None:
            if go_backwards:
                mask = tf.reverse(mask, [0])

            mask_ta = tf.TensorArray(
                dtype=tf.bool, size=time_steps_t, tensor_array_name="mask_ta"
            )
            mask_ta = mask_ta.unstack(mask)

            def masking_fn(time):
                return mask_ta.read(time)

            def compute_masked_output(mask_t, flat_out, flat_mask):
                tiled_mask_t = tuple(
                    _expand_mask(mask_t, o, fixed_dim=len(mask_t.shape))
                    for o in flat_out
                )
                return tuple(
                    tf.where(m, o, fm)
                    for m, o, fm in zip(tiled_mask_t, flat_out, flat_mask)
                )

        elif isinstance(input_length, tf.Tensor):
            if go_backwards:
                max_len = tf.reduce_max(input_length, axis=0)
                rev_input_length = tf.subtract(max_len - 1, input_length)

                def masking_fn(time):
                    return tf.less(rev_input_length, time)

            else:

                def masking_fn(time):
                    return tf.greater(input_length, time)

            def compute_masked_output(mask_t, flat_out, flat_mask):
                return tuple(
                    tf.where(mask_t, o, zo)
                    for (o, zo) in zip(flat_out, flat_mask)
                )

        else:
            masking_fn = None

        if masking_fn is not None:
            # Mask for the T output will be base on the output of T - 1. In the
            # case T = 0, a zero filled tensor will be used.
            flat_zero_output = tuple(
                tf.zeros_like(o) for o in tree.flatten(output_time_zero)
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
                current_input = tuple(ta.read(time) for ta in input_ta)
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
                output_ta_t = tuple(
                    ta.write(ta_index_to_write, out)
                    for ta, out in zip(output_ta_t, flat_new_output)
                )

                return (time + 1, output_ta_t, tuple(flat_new_output)) + tuple(
                    new_states
                )

            final_outputs = tf.while_loop(
                body=_step,
                loop_vars=(time, output_ta, flat_zero_output) + states,
                **while_loop_kwargs,
            )
            # Skip final_outputs[2] which is the output for final timestep.
            new_states = final_outputs[3:]
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
                current_input = tuple(ta.read(time) for ta in input_ta)
                current_input = tree.pack_sequence_as(inputs, current_input)
                output, new_states = step_function(
                    current_input, tuple(states) + tuple(constants)
                )
                flat_new_state = tree.flatten(new_states)

                flat_output = tree.flatten(output)
                ta_index_to_write = time if return_all_outputs else 0
                output_ta_t = tuple(
                    ta.write(ta_index_to_write, out)
                    for ta, out in zip(output_ta_t, flat_output)
                )

                new_states = tree.pack_sequence_as(
                    initial_states, flat_new_state
                )
                return (time + 1, output_ta_t) + tuple(new_states)

            final_outputs = tf.while_loop(
                body=_step,
                loop_vars=(time, output_ta) + states,
                **while_loop_kwargs,
            )
            new_states = final_outputs[2:]

        output_ta = final_outputs[1]

        outputs = tuple(o.stack() for o in output_ta)
        last_output = tuple(o[-1] for o in outputs)

        outputs = tree.pack_sequence_as(output_time_zero, outputs)
        last_output = tree.pack_sequence_as(output_time_zero, last_output)

    if not time_major:
        outputs = tree.map_structure(swap_batch_timestep, outputs)

    return last_output, outputs, new_states


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
    time_major=False,
    reset_after=True,
):
    cudnn_supported = cudnn_ok(
        activation,
        recurrent_activation,
        unroll,
        use_bias=bias is not None,
        reset_after=reset_after,
    )
    if not cudnn_supported:
        raise NotImplementedError

    from keras.src.backend.tensorflow import Variable

    if isinstance(kernel, Variable):
        kernel = kernel.value
    if isinstance(recurrent_kernel, Variable):
        recurrent_kernel = recurrent_kernel.value
    if isinstance(bias, Variable):
        bias = bias.value

    try:
        return _cudnn_gru(
            inputs,
            initial_state,
            kernel,
            recurrent_kernel,
            bias,
            mask,
            time_major,
            go_backwards,
            return_sequences,
        )
    except tf.errors.InvalidArgumentError:
        # cuDNN op not found.
        raise NotImplementedError
    except tf.errors.NotFoundError:
        # alternative error: device not found for op
        raise NotImplementedError


def _do_gru_arguments_support_cudnn(
    activation,
    recurrent_activation,
    unroll,
    use_bias,
    reset_after,
):
    from keras.src import activations
    from keras.src import ops

    return (
        activation in (activations.tanh, tf.tanh, ops.tanh)
        and recurrent_activation
        in (activations.sigmoid, tf.sigmoid, ops.sigmoid)
        and not unroll
        and use_bias
        and reset_after
    )


def _do_lstm_arguments_support_cudnn(
    activation,
    recurrent_activation,
    unroll,
    use_bias,
):
    from keras.src import activations
    from keras.src import ops

    return (
        activation in (activations.tanh, tf.tanh, ops.tanh)
        and recurrent_activation
        in (activations.sigmoid, tf.sigmoid, ops.sigmoid)
        and not unroll
        and use_bias
    )


def _has_fully_masked_sequence(mask):
    # Cudnn kernel will error out if the input sequence contains any
    # fully masked data. We walk around this issue by rerouting the computation
    # to standard kernel, until the issue on cudnn side has been fixed.  For a
    # fully masked sequence, it will contain all Falses. To make it easy to
    # check, we inverse the boolean, check if any of the sequence has all True.
    return tf.reduce_any(
        tf.reduce_all(tf.logical_not(tf.cast(mask, dtype="bool")), axis=1)
    )


def _assert_valid_mask(mask):
    valid = tf.logical_and(
        tf.logical_not(_has_fully_masked_sequence(mask)),
        _is_sequence_right_padded(mask),
    )
    tf.Assert(
        valid,
        [
            (
                "You are passing a RNN mask that does not correspond to "
                "right-padded sequences, while using cuDNN, which is not "
                "supported. With cuDNN, RNN masks can only be used for "
                "right-padding, e.g. `[[True, True, False, False]]` would "
                "be a valid mask, but any mask that isn't just contiguous "
                "`True`'s on the left and contiguous `False`'s on the right "
                "would be invalid. You can pass `use_cudnn=False` to your "
                "RNN layer to stop using cuDNN (this may be slower)."
            )
        ],
    )


def _standardize_cudnn_weights(weights, biases, shape, transpose_weights=False):
    """Utility function convert variable to cuDNN compatible parameter.

    Note that Keras weights for kernels are different from the cuDNN format.
    Eg.:

    ```
      Keras                 cuDNN
      [[0, 1, 2],  <--->  [[0, 2, 4],
       [3, 4, 5]]          [1, 3, 5]]
    ```

    If the input weights need to be in a unified format, then set
    `transpose_weights=True` to convert the weights.

    Args:
        weights: list of weights for the kernels and recurrent kernels.
        biases: list of biases for individual gate.
        shape: the shape for the converted variables that will be feed to cuDNN.
        transpose_weights: boolean, whether to transpose the weights.

    Returns:
        The converted weights that can be feed to cuDNN ops as param.
    """

    def convert(w):
        return tf.transpose(w) if transpose_weights else w

    weights = [tf.reshape(convert(x), shape) for x in weights]
    biases = [tf.reshape(x, shape) for x in biases]
    return tf.concat(weights + biases, axis=0)


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
    max_seq_length = tf.shape(mask)[1]
    count_of_true = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
    right_padded_mask = tf.sequence_mask(count_of_true, maxlen=max_seq_length)
    return tf.reduce_all(
        tf.equal(
            tf.cast(mask, dtype="bool"),
            tf.cast(right_padded_mask, dtype="bool"),
        )
    )


def _compute_sequence_length_from_mask(mask, time_major):
    """Calculate the sequence length tensor (1-D) based on the masking tensor.

    The masking tensor is a 2D boolean tensor with shape [batch, timestep]. For
    any timestep that should be masked, the corresponding field will be False.
    Consider the following example:
      a = [[True, True, False, False],
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
    timestep_index = 0 if time_major else 1
    return tf.reduce_sum(tf.cast(mask, tf.int32), axis=timestep_index)


def _is_gpu_available():
    return bool(tf.config.list_logical_devices("GPU"))


def _cudnn_gru(
    inputs,
    initial_state,
    kernel,
    recurrent_kernel,
    bias,
    mask,
    time_major,
    go_backwards,
    return_sequences,
):
    """GRU with cuDNN implementation which is only available for GPU."""
    if mask is not None:
        _assert_valid_mask(mask)
        sequence_lengths = _compute_sequence_length_from_mask(mask, time_major)
    else:
        if time_major:
            batch_dim = tf.shape(inputs)[1]
            max_sequence_length = tf.shape(inputs)[0]
        else:
            batch_dim = tf.shape(inputs)[0]
            max_sequence_length = tf.shape(inputs)[1]
        sequence_lengths = tf.fill([batch_dim], max_sequence_length)

    if not time_major and sequence_lengths is None:
        inputs = tf.transpose(inputs, perm=(1, 0, 2))
        seq_axis, batch_axis = (0, 1)
    else:
        seq_axis, batch_axis = (0, 1) if time_major else (1, 0)

    # For init_h, cuDNN expects one more dim of num_layers before or after batch
    # dim for time major or batch major inputs respectively
    init_h = tf.expand_dims(initial_state, axis=seq_axis)

    weights = tf.split(kernel, 3, axis=1)
    weights += tf.split(recurrent_kernel, 3, axis=1)
    # Note that the bias was initialized as shape (2, 3 * units), flatten it to
    # (6 * units)
    bias = tf.split(tf.reshape(bias, [-1]), 6)

    if tf.sysconfig.get_build_info()["is_cuda_build"]:
        # Note that the gate order for cuDNN is different from the canonical
        # format.  canonical format is [z, r, h], whereas cuDNN is [r, z, h].
        # The swap need to be done for kernel, recurrent_kernel, input_bias,
        # recurrent_bias.
        # z is update gate weights.
        # r is reset gate weights.
        # h is output gate weights.
        weights[0], weights[1] = weights[1], weights[0]
        weights[3], weights[4] = weights[4], weights[3]
        bias[0], bias[1] = bias[1], bias[0]
        bias[3], bias[4] = bias[4], bias[3]

    params = _standardize_cudnn_weights(
        weights=weights,
        biases=bias,
        shape=tf.constant([-1]),
        transpose_weights=True,
    )

    if go_backwards:
        # Three reversals are required. E.g.,
        # normal input = [1, 2, 3, 0, 0]  # where 0 need to be masked
        # reversed_input_to_cudnn = [3, 2, 1, 0, 0]
        # output_from_cudnn = [6, 5, 4, 0, 0]
        # expected_output = [0, 0, 6, 5 ,4]
        inputs = tf.reverse_sequence(
            inputs,
            sequence_lengths,
            seq_axis=seq_axis,
            batch_axis=batch_axis,
        )
    outputs, h, _, _, _ = tf.raw_ops.CudnnRNNV3(
        input=inputs,
        input_h=init_h,
        input_c=0,
        params=params,
        is_training=True,
        rnn_mode="gru",
        sequence_lengths=sequence_lengths,
        time_major=time_major,
    )
    if go_backwards:
        outputs = tf.reverse_sequence(
            outputs,
            sequence_lengths,
            seq_axis=seq_axis,
            batch_axis=batch_axis,
        )
        outputs = tf.reverse(outputs, axis=[seq_axis])

    last_output = outputs[-1]
    if not time_major and sequence_lengths is None and return_sequences:
        outputs = tf.transpose(outputs, perm=[1, 0, 2])
    state = tf.squeeze(h, axis=seq_axis)

    # In the case of variable length input, the cudnn kernel will fill zeros for
    # the output, whereas the default keras behavior is to bring over the
    # previous output for t-1, so that in the return_sequence=False case, user
    # can quickly get the final effect output instead just 0s at the last
    # timestep.  In order to mimic the default keras behavior, we copy the final
    # h state as the last_output, since it is numerically same as the output.
    if sequence_lengths is not None:
        last_output = state

    # Match CPU return format
    if not return_sequences:
        outputs = tf.expand_dims(last_output, axis=0 if time_major else 1)

    return (
        last_output,
        outputs,
        [state],
    )


def cudnn_ok(
    activation,
    recurrent_activation,
    unroll,
    use_bias,
    reset_after=None,
):
    if reset_after is None:
        args_supported = _do_lstm_arguments_support_cudnn(
            activation=activation,
            recurrent_activation=recurrent_activation,
            unroll=unroll,
            use_bias=use_bias,
        )
    else:
        args_supported = _do_gru_arguments_support_cudnn(
            activation=activation,
            recurrent_activation=recurrent_activation,
            unroll=unroll,
            use_bias=use_bias,
            reset_after=reset_after,
        )
    return args_supported and _is_gpu_available()


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
    cudnn_supported = cudnn_ok(
        activation, recurrent_activation, unroll, use_bias=bias is not None
    )
    if not cudnn_supported:
        raise NotImplementedError

    from keras.src.backend.tensorflow import Variable

    if isinstance(kernel, Variable):
        kernel = kernel.value
    if isinstance(recurrent_kernel, Variable):
        recurrent_kernel = recurrent_kernel.value
    if isinstance(bias, Variable):
        bias = bias.value

    try:
        return _cudnn_lstm(
            inputs,
            initial_state_h,
            initial_state_c,
            kernel,
            recurrent_kernel,
            bias,
            mask,
            time_major,
            go_backwards,
            return_sequences,
        )
    except tf.errors.InvalidArgumentError:
        # cuDNN op not found.
        raise NotImplementedError
    except tf.errors.NotFoundError:
        # alternative error: device not found for op
        raise NotImplementedError


def _cudnn_lstm(
    inputs,
    initial_state_h,
    initial_state_c,
    kernel,
    recurrent_kernel,
    bias,
    mask,
    time_major,
    go_backwards,
    return_sequences,
):
    if mask is not None:
        _assert_valid_mask(mask)
        sequence_lengths = _compute_sequence_length_from_mask(mask, time_major)
    else:
        if time_major:
            batch_dim = tf.shape(inputs)[1]
            max_sequence_length = tf.shape(inputs)[0]
        else:
            batch_dim = tf.shape(inputs)[0]
            max_sequence_length = tf.shape(inputs)[1]
        sequence_lengths = tf.fill([batch_dim], max_sequence_length)

    if not time_major and sequence_lengths is None:
        inputs = tf.transpose(inputs, perm=(1, 0, 2))
        seq_axis, batch_axis = (0, 1)
    else:
        seq_axis, batch_axis = (0, 1) if time_major else (1, 0)
    # For init_h and init_c, cuDNN expects one more dim of num_layers before or
    # after batch dim for time major or batch major inputs respectively
    init_h = tf.expand_dims(initial_state_h, axis=seq_axis)
    init_c = tf.expand_dims(initial_state_c, axis=seq_axis)

    weights = tf.split(kernel, 4, axis=1)
    weights += tf.split(recurrent_kernel, 4, axis=1)
    # cuDNN has an extra set of bias for inputs, we disable them (setting to 0),
    # so that mathematically it is same as the canonical LSTM implementation.
    full_bias = tf.concat((tf.zeros_like(bias), bias), 0)

    if tf.sysconfig.get_build_info()["is_rocm_build"]:
        # ROCm MIOpen's weight sequence for LSTM is different from both
        # canonical and Cudnn format
        # MIOpen: [i, f, o, c] Cudnn/Canonical: [i, f, c, o]
        # i is input gate weights.
        # f is forget gate weights.
        # o is output gate weights.
        # c is cell gate weights.
        weights = [weights[x] for x in (0, 1, 3, 2, 4, 5, 7, 6)]
        # full_bias is a tensor of shape (8*n,)
        full_bias = tf.split(full_bias, 8, axis=0)
        full_bias = [full_bias[x] for x in (0, 1, 3, 2, 4, 5, 7, 6)]

    params = _standardize_cudnn_weights(
        weights=weights,
        biases=tf.split(full_bias, 8),
        shape=tf.constant([-1]),
        transpose_weights=True,
    )

    if go_backwards:
        # Three reversals are required. E.g.,
        # normal input = [1, 2, 3, 0, 0]  # where 0 need to be masked
        # reversed_input_to_cudnn = [3, 2, 1, 0, 0]
        # output_from_cudnn = [6, 5, 4, 0, 0]
        # expected_output = [0, 0, 6, 5 ,4]
        inputs = tf.reverse_sequence(
            inputs,
            sequence_lengths,
            seq_axis=seq_axis,
            batch_axis=batch_axis,
        )
    outputs, h, c, _, _ = tf.raw_ops.CudnnRNNV3(
        input=inputs,
        input_h=init_h,
        input_c=init_c,
        params=params,
        is_training=True,
        rnn_mode="lstm",
        sequence_lengths=sequence_lengths,
        time_major=time_major,
    )
    if go_backwards:
        outputs = tf.reverse_sequence(
            outputs,
            sequence_lengths,
            seq_axis=seq_axis,
            batch_axis=batch_axis,
        )
        outputs = tf.reverse(outputs, axis=[seq_axis])

    last_output = outputs[-1]
    if not time_major and sequence_lengths is None and return_sequences:
        outputs = tf.transpose(outputs, perm=[1, 0, 2])
    h = tf.squeeze(h, axis=seq_axis)
    c = tf.squeeze(c, axis=seq_axis)

    # In the case of variable length input, the cudnn kernel will fill zeros for
    # the output, whereas the default keras behavior is to bring over the
    # previous output for t-1, so that in the return_sequence=False case, user
    # can quickly get the final effect output instead just 0s at the last
    # timestep.  In order to mimic the default keras behavior, we copy the final
    # h state as the last_output, since it is numerically same as the output.
    if sequence_lengths is not None:
        last_output = h

    # Match CPU return format
    if not return_sequences:
        outputs = tf.expand_dims(last_output, axis=0 if time_major else 1)

    return (last_output, outputs, [h, c])
