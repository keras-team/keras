from keras.src.api_export import keras_export
from keras.src.layers.rnn.conv_lstm import ConvLSTM


@keras_export("keras.layers.ConvLSTM2D")
class ConvLSTM2D(ConvLSTM):
    """2D Convolutional LSTM.

    Similar to an LSTM layer, but the input transformations
    and recurrent transformations are both convolutional.

    Args:
        filters: int, the dimension of the output space (the number of filters
            in the convolution).
        kernel_size: int or tuple/list of 2 integers, specifying the size of the
            convolution window.
        strides: int or tuple/list of 2 integers, specifying the stride length
            of the convolution. `strides > 1` is incompatible with
            `dilation_rate > 1`.
        padding: string, `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input such that output has the same
            height/width dimension as the input.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, steps, features)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, features, steps)`. It defaults to the `image_data_format`
            value found in your Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `"channels_last"`.
        dilation_rate: int or tuple/list of 2 integers, specifying the dilation
            rate to use for dilated convolution.
        activation: Activation function to use. By default hyperbolic tangent
            activation function is applied (`tanh(x)`).
        recurrent_activation: Activation function to use for the recurrent step.
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel` weights
            matrix, used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        unit_forget_bias: Boolean. If `True`, add 1 to the bias of the forget
            gate at initialization.
            Use in combination with `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al., 2015](
            http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix.
        recurrent_regularizer: Regularizer function applied to the
            `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix.
        recurrent_constraint: Constraint function applied to the
            `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        dropout: Float between 0 and 1. Fraction of the units to drop for the
            linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
            for the linear transformation of the recurrent state.
        seed: Random seed for dropout.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence. Default: `False`.
        return_state: Boolean. Whether to return the last state in addition
            to the output. Default: `False`.
        go_backwards: Boolean (default: `False`).
            If `True`, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If `True`, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default: `False`).
            If `True`, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.


    Call arguments:
        inputs: A 5D tensor.
        mask: Binary tensor of shape `(samples, timesteps)` indicating whether a
            given timestep should be masked.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode.
            This is only relevant if `dropout` or `recurrent_dropout` are set.
        initial_state: List of initial state tensors to be passed to the first
            call of the cell.

    Input shape:

    - If `data_format='channels_first'`:
        5D tensor with shape: `(samples, time, channels, rows, cols)`
    - If `data_format='channels_last'`:
        5D tensor with shape: `(samples, time, rows, cols, channels)`

    Output shape:

    - If `return_state`: a list of tensors. The first tensor is the output.
        The remaining tensors are the last states,
        each 4D tensor with shape: `(samples, filters, new_rows, new_cols)` if
        `data_format='channels_first'`
        or shape: `(samples, new_rows, new_cols, filters)` if
        `data_format='channels_last'`. `rows` and `cols` values might have
        changed due to padding.
    - If `return_sequences`: 5D tensor with shape: `(samples, timesteps,
        filters, new_rows, new_cols)` if data_format='channels_first'
        or shape: `(samples, timesteps, new_rows, new_cols, filters)` if
        `data_format='channels_last'`.
    - Else, 4D tensor with shape: `(samples, filters, new_rows, new_cols)` if
        `data_format='channels_first'`
        or shape: `(samples, new_rows, new_cols, filters)` if
        `data_format='channels_last'`.

    References:

    - [Shi et al., 2015](http://arxiv.org/abs/1506.04214v1)
        (the current implementation does not include the feedback loop on the
        cells output).
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        activation="tanh",
        recurrent_activation="sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        unit_forget_bias=True,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        seed=None,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        **kwargs,
    ):
        super().__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            unit_forget_bias=unit_forget_bias,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            seed=seed,
            **kwargs,
        )
