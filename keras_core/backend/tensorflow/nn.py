import warnings

import tensorflow as tf

from keras_core.backend.common.backend_utils import (
    compute_conv_transpose_output_shape,
)
from keras_core.backend.config import epsilon


def relu(x):
    return tf.nn.relu(x)


def relu6(x):
    return tf.nn.relu6(x)


def sigmoid(x):
    logits = x
    output = tf.nn.sigmoid(x)
    output._keras_logits = logits
    return output


def tanh(x):
    return tf.nn.tanh(x)


def softplus(x):
    return tf.math.softplus(x)


def softsign(x):
    return tf.nn.softsign(x)


def silu(x, beta=1.0):
    return tf.nn.silu(x, beta=beta)


def swish(x):
    return x * sigmoid(x)


def log_sigmoid(x):
    return tf.math.log_sigmoid(x)


def leaky_relu(x, negative_slope=0.2):
    return tf.nn.leaky_relu(x, alpha=negative_slope)


def hard_sigmoid(x):
    x = x / 6.0 + 0.5
    return tf.clip_by_value(x, 0.0, 1.0)


def elu(x):
    return tf.nn.elu(x)


def selu(x):
    return tf.nn.selu(x)


def gelu(x, approximate=True):
    return tf.nn.gelu(x, approximate)


def softmax(x, axis=None):
    logits = x
    output = tf.nn.softmax(x, axis=axis)
    output._keras_logits = logits
    return output


def log_softmax(x, axis=None):
    return tf.nn.log_softmax(x, axis=axis)


def _transpose_spatial_inputs(inputs):
    num_spatial_dims = len(inputs.shape) - 2
    # Tensorflow pooling does not support `channels_first` format, so
    # we need to transpose to `channels_last` format.
    if num_spatial_dims == 1:
        inputs = tf.transpose(inputs, (0, 2, 1))
    elif num_spatial_dims == 2:
        inputs = tf.transpose(inputs, (0, 2, 3, 1))
    elif num_spatial_dims == 3:
        inputs = tf.transpose(inputs, (0, 2, 3, 4, 1))
    else:
        raise ValueError(
            "Pooling inputs's shape must be 3, 4 or 5, corresponding to 1D, 2D "
            f"and 3D inputs. But received shape: {inputs.shape}."
        )
    return inputs


def _transpose_spatial_outputs(outputs):
    # Undo the tranpose in `_transpose_spatial_inputs`.
    num_spatial_dims = len(outputs.shape) - 2
    if num_spatial_dims == 1:
        outputs = tf.transpose(outputs, (0, 2, 1))
    elif num_spatial_dims == 2:
        outputs = tf.transpose(outputs, (0, 3, 1, 2))
    elif num_spatial_dims == 3:
        outputs = tf.transpose(outputs, (0, 4, 1, 2, 3))
    return outputs


def max_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format="channels_last",
):
    strides = pool_size if strides is None else strides
    padding = padding.upper()
    tf_data_format = _convert_data_format("channels_last", len(inputs.shape))
    if data_format == "channels_first":
        # Tensorflow pooling does not support `channels_first` format, so
        # we need to transpose to `channels_last` format.
        inputs = _transpose_spatial_inputs(inputs)

    outputs = tf.nn.max_pool(
        inputs,
        pool_size,
        strides,
        padding,
        tf_data_format,
    )
    if data_format == "channels_first":
        outputs = _transpose_spatial_outputs(outputs)
    return outputs


def average_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format="channels_last",
):
    strides = pool_size if strides is None else strides
    padding = padding.upper()
    tf_data_format = _convert_data_format("channels_last", len(inputs.shape))
    if data_format == "channels_first":
        # Tensorflow pooling does not support `channels_first` format, so
        # we need to transpose to `channels_last` format.
        inputs = _transpose_spatial_inputs(inputs)

    outputs = tf.nn.avg_pool(
        inputs,
        pool_size,
        strides,
        padding,
        tf_data_format,
    )
    if data_format == "channels_first":
        outputs = _transpose_spatial_outputs(outputs)
    return outputs


def _convert_data_format(data_format, ndim):
    if data_format == "channels_last":
        if ndim == 3:
            return "NWC"
        elif ndim == 4:
            return "NHWC"
        elif ndim == 5:
            return "NDHWC"
        else:
            raise ValueError(
                f"Input rank not supported: {ndim}. "
                "Expected values are [3, 4, 5]"
            )
    elif data_format == "channels_first":
        if ndim == 3:
            return "NCW"
        elif ndim == 4:
            return "NCHW"
        elif ndim == 5:
            return "NCDHW"
        else:
            raise ValueError(
                f"Input rank not supported: {ndim}. "
                "Expected values are [3, 4, 5]"
            )
    else:
        raise ValueError(
            f"Invalid data_format: {data_format}. "
            'Expected values are ["channels_first", "channels_last"]'
        )


def conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format="channel_last",
    dilation_rate=1,
):
    def _conv():
        tf_data_format = _convert_data_format(data_format, len(inputs.shape))
        return tf.nn.convolution(
            inputs,
            kernel,
            strides,
            padding.upper(),
            data_format=tf_data_format,
            dilations=dilation_rate,
        )

    # Reason for making this function is in Tensorflow, `groups > 1` does not
    # work on CPU for `tf.nn.convolution`, but wrapping it by XLA works.
    @tf.function(jit_compile=True)
    def _conv_xla():
        return _conv()

    if data_format == "channels_last":
        channels = inputs.shape[-1]
    else:
        channels = inputs.shape[1]
    if channels != kernel.shape[-2]:
        # If kernel's in_channel does not match input's channels,  it indicates
        # convolution is broken down into groups.
        return _conv_xla()
    return _conv()


def depthwise_conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format="channels_last",
    dilation_rate=1,
):
    num_spatial_dims = len(inputs.shape) - 2
    if num_spatial_dims > 2:
        raise ValueError(
            "`inputs` rank must be 3 (1D conv) or 4 (2D conv). Received: "
            "{inputs.ndim}."
        )
    tf_data_format = _convert_data_format(data_format, len(inputs.shape))
    padding = padding.upper()
    if isinstance(strides, int):
        strides = (strides,) * num_spatial_dims
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate,) * num_spatial_dims
    if num_spatial_dims == 1:
        # 1D depthwise conv.
        if data_format == "channels_last":
            strides = (1,) + strides * 2 + (1,)
            spatial_start_dim = 1
        else:
            strides = (1, 1) + strides * 2
            spatial_start_dim = 2
        inputs = tf.expand_dims(inputs, spatial_start_dim)
        kernel = tf.expand_dims(kernel, axis=0)

        dilation_rate = None if dilation_rate is None else (1,) + dilation_rate

        outputs = tf.nn.depthwise_conv2d(
            inputs,
            kernel,
            strides,
            padding,
            data_format=tf_data_format,
            dilations=dilation_rate,
        )
        return tf.squeeze(outputs, [spatial_start_dim])

    if data_format == "channels_last":
        strides = (1,) + strides + (1,)
        spatial_start_dim = 1
    else:
        strides = (1, 1) + strides
        spatial_start_dim = 2
    return tf.nn.depthwise_conv2d(
        inputs,
        kernel,
        strides,
        padding,
        data_format=tf_data_format,
        dilations=dilation_rate,
    )


def separable_conv(
    inputs,
    depthwise_kernel,
    pointwise_kernel,
    strides=1,
    padding="valid",
    data_format="channels_last",
    dilation_rate=1,
):
    num_spatial_dims = len(inputs.shape) - 2
    if num_spatial_dims > 2:
        raise ValueError(
            "`num_spatial_dims` must be 1 or 2. Received: "
            f"num_spatial_dims={num_spatial_dims}."
        )
    tf_data_format = _convert_data_format(data_format, len(inputs.shape))
    padding = padding.upper()
    if isinstance(strides, int):
        strides = (strides,) * num_spatial_dims
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate,) * num_spatial_dims
    if num_spatial_dims == 1:
        # 1D depthwise conv.
        if data_format == "channels_last":
            strides = (1,) + strides * 2 + (1,)
            spatial_start_dim = 1
        else:
            strides = (1, 1) + strides * 2
            spatial_start_dim = 2
        inputs = tf.expand_dims(inputs, spatial_start_dim)
        depthwise_kernel = tf.expand_dims(depthwise_kernel, axis=0)
        pointwise_kernel = tf.expand_dims(pointwise_kernel, axis=0)
        dilation_rate = None if dilation_rate is None else (1,) + dilation_rate

        outputs = tf.nn.separable_conv2d(
            inputs,
            depthwise_kernel,
            pointwise_kernel,
            strides,
            padding,
            data_format=tf_data_format,
            dilations=dilation_rate,
        )
        return tf.squeeze(outputs, [spatial_start_dim])

    if data_format == "channels_last":
        strides = (1,) + strides + (1,)
    else:
        strides = (1, 1) + strides
    return tf.nn.separable_conv2d(
        inputs,
        depthwise_kernel,
        pointwise_kernel,
        strides,
        padding,
        data_format=tf_data_format,
        dilations=dilation_rate,
    )


def conv_transpose(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    output_padding=None,
    data_format="channels_last",
    dilation_rate=1,
):
    tf_data_format = _convert_data_format(data_format, len(inputs.shape))
    output_shape = compute_conv_transpose_output_shape(
        inputs,
        kernel,
        strides,
        padding,
        output_padding,
        data_format,
        dilation_rate,
    )

    return tf.nn.conv_transpose(
        inputs,
        kernel,
        output_shape,
        strides,
        padding=padding.upper(),
        data_format=tf_data_format,
        dilations=dilation_rate,
    )


def one_hot(x, num_classes, axis=-1):
    return tf.one_hot(x, num_classes, axis=axis)


def _get_logits(output, from_logits, op_type, fn_name):
    """Retrieves logits tensor from maybe-softmax or maybe-sigmoid tensor."""
    output_ = output
    from_logits_ = from_logits

    has_keras_logits = hasattr(output, "_keras_logits")
    if has_keras_logits:
        output_ = output._keras_logits
        from_logits_ = True

    from_expected_op_type = (
        not isinstance(output, (tf.__internal__.EagerTensor, tf.Variable))
        and output.op.type == op_type
    ) and not has_keras_logits

    if from_expected_op_type:
        # When softmax activation function is used for output operation, we
        # use logits from the softmax function directly to compute loss in order
        # to prevent collapsing zero when training.
        assert len(output.op.inputs) == 1
        output_ = output.op.inputs[0]
        from_logits_ = True

    if from_logits and (has_keras_logits or from_expected_op_type):
        warnings.warn(
            f'"`{fn_name}` received `from_logits=True`, but '
            f"the `output` argument was produced by a {op_type} "
            "activation and thus does not represent logits. "
            "Was this intended?",
            stacklevel=2,
        )
    return output_, from_logits_


def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    """Categorical crossentropy between an output tensor and a target tensor.

    Args:
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is `True`, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
        axis: Int specifying the channels axis. `axis=-1` corresponds to data
            format `channels_last`, and `axis=1` corresponds to data format
            `channels_first`.

    Returns:
        Output tensor.

    Example:

    >>> a = tf.constant([1., 0., 0., 0., 1., 0., 0., 0., 1.], shape=[3,3])
    >>> print(a)
    tf.Tensor(
      [[1. 0. 0.]
       [0. 1. 0.]
       [0. 0. 1.]], shape=(3, 3), dtype=float32)
    >>> b = tf.constant([.9, .05, .05, .05, .89, .06, .05, .01, .94],
    ...                 shape=[3, 3])
    >>> print(b)
    tf.Tensor(
      [[0.9  0.05 0.05]
       [0.05 0.89 0.06]
       [0.05 0.01 0.94]], shape=(3, 3), dtype=float32)
    >>> loss = categorical_crossentropy(a, b)
    >>> print(np.around(loss, 5))
    [0.10536 0.11653 0.06188]
    >>> loss = categorical_crossentropy(a, a)
    >>> print(np.around(loss, 5))
    [0. 0. 0.]
    """
    target = tf.convert_to_tensor(target)
    output = tf.convert_to_tensor(output)

    if target.shape != output.shape:
        raise ValueError(
            "Arguments `target` and `output` must have the same shape. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )
    if len(target.shape) < 1:
        raise ValueError(
            "Arguments `target` and `output` must be at least rank 1. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )

    output, from_logits = _get_logits(
        output, from_logits, "Softmax", "categorical_crossentropy"
    )
    if from_logits:
        return tf.nn.softmax_cross_entropy_with_logits(
            labels=target, logits=output, axis=axis
        )

    # Adjust the predictions so that the probability of
    # each class for every sample adds up to 1
    # This is needed to ensure that the cross entropy is
    # computed correctly.
    output = output / tf.reduce_sum(output, axis, keepdims=True)

    # Compute cross entropy from probabilities.
    output = tf.clip_by_value(output, epsilon(), 1.0 - epsilon())
    return -tf.reduce_sum(target * tf.math.log(output), axis)


def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    """Categorical crossentropy with integer targets.

    Args:
        target: An integer tensor.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
        axis: Int specifying the channels axis. `axis=-1` corresponds to data
            format `channels_last`, and `axis=1` corresponds to data format
            `channels_first`.

    Returns:
        Output tensor.
    """
    if axis != -1 and axis != len(output.shape) - 1:
        raise ValueError(
            f"Only axis=-1 is currently supported. Received: axis={axis}"
        )

    target = tf.convert_to_tensor(target)
    target = tf.cast(target, dtype="int64")
    output = tf.convert_to_tensor(output)
    if len(target.shape) == len(output.shape) and target.shape[-1] == 1:
        target = tf.squeeze(target, axis=-1)

    if len(output.shape) < 1:
        raise ValueError(
            "Argument `output` must be at least rank 1. "
            "Received: "
            f"output.shape={output.shape}"
        )
    if target.shape != output.shape[:-1]:
        raise ValueError(
            "Arguments `target` and `output` must have the same shape "
            "up until the last dimension: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )

    output, from_logits = _get_logits(
        output, from_logits, "Softmax", "sparse_categorical_crossentropy"
    )
    if not from_logits:
        output = tf.clip_by_value(output, epsilon(), 1 - epsilon())
        output = tf.math.log(output)

    result = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target, logits=output
    )
    return result


def binary_crossentropy(target, output, from_logits=False):
    """Binary crossentropy between an output tensor and a target tensor.

    Args:
        target: A tensor with the same shape as `output`.
        output: A tensor.
        from_logits: Whether `output` is expected to be a logits tensor.
            By default, we consider that `output`
            encodes a probability distribution.

    Returns:
        A tensor.
    """
    target = tf.convert_to_tensor(target)
    output = tf.convert_to_tensor(output)

    if target.shape != output.shape:
        raise ValueError(
            "Arguments `target` and `output` must have the same shape. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )

    output, from_logits = _get_logits(
        output, from_logits, "Sigmoid", "binary_crossentropy"
    )
    if from_logits:
        return tf.nn.sigmoid_cross_entropy_with_logits(
            labels=target, logits=output
        )

    # Compute cross entropy from probabilities.
    output = tf.clip_by_value(output, epsilon(), 1.0 - epsilon())
    bce = target * tf.math.log(output)
    bce += (1 - target) * tf.math.log(1 - output)
    return -bce


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

    def swap_batch_timestep(input_t):
        # Swap the batch and timestep dim for the incoming tensor.
        axes = list(range(len(input_t.shape)))
        axes[0], axes[1] = 1, 0
        return tf.transpose(input_t, axes)

    if not time_major:
        inputs = tf.nest.map_structure(swap_batch_timestep, inputs)

    flatted_inputs = tf.nest.flatten(inputs)
    time_steps = flatted_inputs[0].shape[0]
    batch = flatted_inputs[0].shape[1]
    time_steps_t = tf.shape(flatted_inputs[0])[0]

    for input_ in flatted_inputs:
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
        if tf.nest.is_nested(mask_t):
            raise ValueError(
                f"mask_t is expected to be tensor, but got {mask_t}"
            )
        if tf.nest.is_nested(input_t):
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

        if tf.nest.is_nested(inputs):
            processed_input = tf.nest.map_structure(
                _process_single_input_t, inputs
            )
        else:
            processed_input = (_process_single_input_t(inputs),)

        def _get_input_tensor(time):
            inp = [t_[time] for t_ in processed_input]
            return tf.nest.pack_sequence_as(inputs, inp)

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

                flat_states = tf.nest.flatten(states)
                flat_new_states = tf.nest.flatten(new_states)
                tiled_mask_t = tuple(
                    _expand_mask(mask_t, s) for s in flat_states
                )
                flat_final_states = tuple(
                    tf.where(m, s, ps)
                    for m, s, ps in zip(
                        tiled_mask_t, flat_new_states, flat_states
                    )
                )
                states = tf.nest.pack_sequence_as(states, flat_final_states)

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
            for i, inp in enumerate(flatted_inputs)
        )
        input_ta = tuple(
            ta.unstack(input_)
            if not go_backwards
            else ta.unstack(tf.reverse(input_, [0]))
            for ta, input_ in zip(input_ta, flatted_inputs)
        )

        # Get the time(0) input and compute the output for that, the output will
        # be used to determine the dtype of output tensor array. Don't read from
        # input_ta due to TensorArray clear_after_read default to True.
        input_time_zero = tf.nest.pack_sequence_as(
            inputs, [inp[0] for inp in flatted_inputs]
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
            for i, out in enumerate(tf.nest.flatten(output_time_zero))
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
                tf.zeros_like(o) for o in tf.nest.flatten(output_time_zero)
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
                current_input = tf.nest.pack_sequence_as(inputs, current_input)
                mask_t = masking_fn(time)
                output, new_states = step_function(
                    current_input, tuple(states) + tuple(constants)
                )
                # mask output
                flat_output = tf.nest.flatten(output)
                flat_mask_output = (
                    flat_zero_output
                    if zero_output_for_mask
                    else tf.nest.flatten(prev_output)
                )
                flat_new_output = compute_masked_output(
                    mask_t, flat_output, flat_mask_output
                )

                # mask states
                flat_state = tf.nest.flatten(states)
                flat_new_state = tf.nest.flatten(new_states)
                flat_final_state = compute_masked_output(
                    mask_t, flat_new_state, flat_state
                )
                new_states = tf.nest.pack_sequence_as(
                    new_states, flat_final_state
                )

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
                current_input = tf.nest.pack_sequence_as(inputs, current_input)
                output, new_states = step_function(
                    current_input, tuple(states) + tuple(constants)
                )
                flat_new_state = tf.nest.flatten(new_states)

                flat_output = tf.nest.flatten(output)
                ta_index_to_write = time if return_all_outputs else 0
                output_ta_t = tuple(
                    ta.write(ta_index_to_write, out)
                    for ta, out in zip(output_ta_t, flat_output)
                )

                new_states = tf.nest.pack_sequence_as(
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

        outputs = tf.nest.pack_sequence_as(output_time_zero, outputs)
        last_output = tf.nest.pack_sequence_as(output_time_zero, last_output)

    # static shape inference
    def set_shape(output_):
        if isinstance(output_, tf.Tensor):
            shape = output_.shape.as_list()
            if return_all_outputs:
                shape[0] = time_steps
            else:
                shape[0] = 1
            shape[1] = batch
            output_.set_shape(shape)
        return output_

    outputs = tf.nest.map_structure(set_shape, outputs)

    if not time_major:
        outputs = tf.nest.map_structure(swap_batch_timestep, outputs)

    return last_output, outputs, new_states
