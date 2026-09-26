from keras.src.backend.common.name_scope import name_scope
from keras.src.backend.openvino import ops
from keras.src.backend.openvino import random
from keras.src.backend.openvino import rnn
from keras.src.backend.openvino.ops.core import Variable
from keras.src.backend.openvino.ops.core import compute_output_spec
from keras.src.backend.openvino.ops.core import device_scope

IS_THREAD_SAFE = True
SUPPORTS_COMPLEX_DTYPES = False
SUPPORTS_GRADIENT = False
SUPPORTS_RAGGED_TENSORS = False
SUPPORTS_SPARSE_TENSORS = False

distribution_lib = None
