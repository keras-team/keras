from keras.src.backend.common.name_scope import name_scope
from keras.src.backend.torch import distribution_lib
from keras.src.backend.torch import ops
from keras.src.backend.torch import random
from keras.src.backend.torch import rnn
from keras.src.backend.torch.ops.core import Variable
from keras.src.backend.torch.ops.core import compute_output_spec
from keras.src.backend.torch.ops.core import device_scope

SUPPORTS_SPARSE_TENSORS = False
SUPPORTS_RAGGED_TENSORS = False
SUPPORTS_COMPLEX_DTYPES = True
IS_THREAD_SAFE = True
