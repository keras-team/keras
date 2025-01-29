"""MLX backend APIs."""

from keras.src.backend.common.name_scope import name_scope
from keras.src.backend.mlx import core
from keras.src.backend.mlx import image
from keras.src.backend.mlx import linalg
from keras.src.backend.mlx import math
from keras.src.backend.mlx import nn
from keras.src.backend.mlx import numpy
from keras.src.backend.mlx import random
from keras.src.backend.mlx.core import SUPPORTS_SPARSE_TENSORS
from keras.src.backend.mlx.core import Variable
from keras.src.backend.mlx.core import cast
from keras.src.backend.mlx.core import compute_output_spec
from keras.src.backend.mlx.core import cond
from keras.src.backend.mlx.core import convert_to_numpy
from keras.src.backend.mlx.core import convert_to_tensor
from keras.src.backend.mlx.core import is_tensor
from keras.src.backend.mlx.core import scatter
from keras.src.backend.mlx.core import shape
from keras.src.backend.mlx.core import stop_gradient
from keras.src.backend.mlx.core import to_mlx_dtype
from keras.src.backend.mlx.core import vectorized_map
from keras.src.backend.mlx.rnn import cudnn_ok
from keras.src.backend.mlx.rnn import gru
from keras.src.backend.mlx.rnn import lstm
from keras.src.backend.mlx.rnn import rnn
