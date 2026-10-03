from keras.src.backend.common.name_scope import name_scope
from keras.src.backend.numpy import ops
from keras.src.backend.numpy import random
from keras.src.backend.numpy import rnn
from keras.src.backend.numpy.ops.core import Variable
from keras.src.backend.numpy.ops.core import compute_output_spec
from keras.src.backend.numpy.ops.core import device_scope

IS_THREAD_SAFE = True
SUPPORTS_COMPLEX_DTYPES = True
SUPPORTS_GRADIENT = False
SUPPORTS_RAGGED_TENSORS = False
SUPPORTS_SPARSE_TENSORS = False

distribution_lib = None
