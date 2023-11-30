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

from keras.backend.torch import core
from keras.backend.torch import image
from keras.backend.torch import math
from keras.backend.torch import nn
from keras.backend.torch import numpy
from keras.backend.torch import random
from keras.backend.torch.core import SUPPORTS_SPARSE_TENSORS
from keras.backend.torch.core import Variable
from keras.backend.torch.core import cast
from keras.backend.torch.core import compute_output_spec
from keras.backend.torch.core import cond
from keras.backend.torch.core import convert_to_numpy
from keras.backend.torch.core import convert_to_tensor
from keras.backend.torch.core import device_scope
from keras.backend.torch.core import is_tensor
from keras.backend.torch.core import scatter
from keras.backend.torch.core import shape
from keras.backend.torch.core import stop_gradient
from keras.backend.torch.core import to_torch_dtype
from keras.backend.torch.core import vectorized_map
from keras.backend.torch.rnn import cudnn_ok
from keras.backend.torch.rnn import gru
from keras.backend.torch.rnn import lstm
from keras.backend.torch.rnn import rnn
