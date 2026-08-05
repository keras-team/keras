from keras.src.backend.jax import distribution_lib
from keras.src.backend.jax import ops
from keras.src.backend.jax import random
from keras.src.backend.jax import rnn
from keras.src.backend.jax import tensorboard
from keras.src.backend.jax.ops.core import Variable
from keras.src.backend.jax.ops.core import compute_output_spec
from keras.src.backend.jax.ops.core import device_scope
from keras.src.backend.jax.ops.core import name_scope

SUPPORTS_SPARSE_TENSORS = True
SUPPORTS_RAGGED_TENSORS = False
SUPPORTS_COMPLEX_DTYPES = True
IS_THREAD_SAFE = True
