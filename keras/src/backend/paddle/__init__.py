"""PaddlePaddle backend APIs."""

from keras.src.backend.common.name_scope import name_scope
from keras.src.backend.paddle import core
from keras.src.backend.paddle import image
from keras.src.backend.paddle import linalg
from keras.src.backend.paddle import math
from keras.src.backend.paddle import nn
from keras.src.backend.paddle import numpy
from keras.src.backend.paddle import random
from keras.src.backend.paddle import rnn as rnn_module
from keras.src.backend.paddle.core import IS_THREAD_SAFE
from keras.src.backend.paddle.core import SUPPORTS_COMPLEX_DTYPES
from keras.src.backend.paddle.core import SUPPORTS_RAGGED_TENSORS
from keras.src.backend.paddle.core import SUPPORTS_SPARSE_TENSORS
from keras.src.backend.paddle.core import Variable
from keras.src.backend.paddle.core import cast
from keras.src.backend.paddle.core import compute_output_spec
from keras.src.backend.paddle.core import cond
from keras.src.backend.paddle.core import convert_to_numpy
from keras.src.backend.paddle.core import convert_to_tensor
from keras.src.backend.paddle.core import device_scope
from keras.src.backend.paddle.core import is_tensor
from keras.src.backend.paddle.core import random_seed_dtype
from keras.src.backend.paddle.core import shape
from keras.src.backend.paddle.core import stop_gradient
from keras.src.backend.paddle.core import vectorized_map
from keras.src.backend.paddle.rnn import bidirectional_gru
from keras.src.backend.paddle.rnn import bidirectional_lstm
from keras.src.backend.paddle.rnn import cudnn_ok
from keras.src.backend.paddle.rnn import gru
from keras.src.backend.paddle.rnn import lstm
from keras.src.backend.paddle.rnn import rnn
