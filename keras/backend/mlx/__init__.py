"""MLX backend APIs."""

from keras.backend.mlx import core
from keras.backend.mlx import numpy
from keras.backend.mlx import random
from keras.backend.mlx.core import SUPPORTS_SPARSE_TENSORS
from keras.backend.mlx.core import Variable
from keras.backend.mlx.core import cast
from keras.backend.mlx.core import compute_output_spec
from keras.backend.mlx.core import cond
from keras.backend.mlx.core import convert_to_numpy
from keras.backend.mlx.core import convert_to_tensor
from keras.backend.mlx.core import is_tensor
from keras.backend.mlx.core import scatter
from keras.backend.mlx.core import shape
from keras.backend.mlx.core import stop_gradient
from keras.backend.mlx.core import to_mlx_dtype
from keras.backend.mlx.core import vectorized_map
