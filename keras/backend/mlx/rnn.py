import contextlib

import tree
import mlx.core as mx

from keras.backend import standardize_dtype
from keras.backend.common import stateless_scope
from keras.backend.mlx.core import to_mlx_dtype
from keras.utils.nest import pack_sequence_as


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


def unstack(x, axis=0):
    slices = (slice(None),) * axis
    return [x[*slices, i] for i in range(x.shape[axis])]

