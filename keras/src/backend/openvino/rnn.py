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
    raise NotImplementedError("`rnn` is not supported with openvino backend")


def lstm(*args, **kwargs):
    raise NotImplementedError("`lstm` is not supported with openvino backend")


def gru(*args, **kwargs):
    raise NotImplementedError("`gru` is not supported with openvino backend")


def unstack(x, axis=0):
    raise NotImplementedError(
        "`unstack` is not supported with openvino backend"
    )


def numpy_scan(f, init, xs, reverse=False, mask=None):
    raise NotImplementedError(
        "`numpy_scan` is not supported with openvino backend"
    )


def cudnn_ok(*args, **kwargs):
    return False
