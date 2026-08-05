from keras.src.backend.tensorflow import distribution_lib
from keras.src.backend.tensorflow import ops
from keras.src.backend.tensorflow import random
from keras.src.backend.tensorflow import rnn
from keras.src.backend.tensorflow import tensorboard
from keras.src.backend.tensorflow.ops.core import Variable
from keras.src.backend.tensorflow.ops.core import compute_output_spec
from keras.src.backend.tensorflow.ops.core import device_scope
from keras.src.backend.tensorflow.ops.core import name_scope

SUPPORTS_SPARSE_TENSORS = True
SUPPORTS_RAGGED_TENSORS = True
SUPPORTS_COMPLEX_DTYPES = True
# https://github.com/tensorflow/tensorflow/issues/78338
IS_THREAD_SAFE = False
