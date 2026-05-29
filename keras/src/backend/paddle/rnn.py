import paddle

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
    raise NotImplementedError("`rnn` is not supported with paddle backend")


def lstm(*args, **kwargs):
    raise NotImplementedError("`lstm` is not supported with paddle backend")


def gru(*args, **kwargs):
    raise NotImplementedError("`gru` is not supported with paddle backend")


def bidirectional_lstm(*args, **kwargs):
    raise NotImplementedError(
        "`bidirectional_lstm` is not supported with paddle backend"
    )


def bidirectional_gru(*args, **kwargs):
    raise NotImplementedError(
        "`bidirectional_gru` is not supported with paddle backend"
    )


def cudnn_ok(*args, **kwargs):
    return False


def numpy_scan(f, init, xs, reverse=False, mask=None):
    raise NotImplementedError(
        "`numpy_scan` is not supported with paddle backend"
    )
