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
    raise NotImplementedError("rnn not yet implemented in mlx")


def cudnn_ok(*args, **kwargs):
    return False


def lstm(*args, **kwargs):
    raise NotImplementedError("lstm not yet implemented in mlx")


def gru(*args, **kwargs):
    raise NotImplementedError("gru not yet implemented in mlx")
