"""Torch backend APIs.

# Note on device placement

Torch has a different device placement style compared to TF and JAX.
In short, variables/tensors are not created on GPU by default,
and the GPU cannot directly communicate with the CPU.
To bring Torch behavior in line with TF and JAX automated device placement,
we are doing the following to automate device placement if a GPU is available:

- Variables are created on GPU.
- Input data will be placed on GPU at the first `keras.layers.Layer` call.
- Tensor creation happens on GPU, e.g., `zeros()` will create a tensor on GPU.
- `convert_to_numpy` will bring the tensor to CPU before converting it to NumPy.
"""

from keras.src.backend.common.name_scope import name_scope
from keras.src.backend.torch import core
from keras.src.backend.torch import image
from keras.src.backend.torch import linalg
from keras.src.backend.torch import math
from keras.src.backend.torch import nn
from keras.src.backend.torch import numpy
from keras.src.backend.torch import random
from keras.src.backend.torch.core import IS_THREAD_SAFE
from keras.src.backend.torch.core import SUPPORTS_RAGGED_TENSORS
from keras.src.backend.torch.core import SUPPORTS_SPARSE_TENSORS
from keras.src.backend.torch.core import Variable
from keras.src.backend.torch.core import cast
from keras.src.backend.torch.core import compute_output_spec
from keras.src.backend.torch.core import cond
from keras.src.backend.torch.core import convert_to_numpy
from keras.src.backend.torch.core import convert_to_tensor
from keras.src.backend.torch.core import device_scope
from keras.src.backend.torch.core import is_tensor
from keras.src.backend.torch.core import random_seed_dtype
from keras.src.backend.torch.core import scatter
from keras.src.backend.torch.core import shape
from keras.src.backend.torch.core import stop_gradient
from keras.src.backend.torch.core import to_torch_dtype
from keras.src.backend.torch.core import vectorized_map
from keras.src.backend.torch.rnn import cudnn_ok
from keras.src.backend.torch.rnn import gru
from keras.src.backend.torch.rnn import lstm
from keras.src.backend.torch.rnn import rnn
