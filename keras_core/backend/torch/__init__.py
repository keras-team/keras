"""Torch backend APIs.

# Note on device placement

Torch has a different device placement style compared to TF and JAX.
In short, variables/tensors are not created on GPU by default,
and the GPU cannot directly communicate with the CPU.
To bring Torch behavior in line with TF and JAX automated device placement,
we are doing the following to automate device placement if a GPU is available:

- Variables are created on GPU.
- Input data will be placed on GPU at the first `keras_core.layers.Layer` call.
- Tensor creation happens on GPU, e.g., `zeros()` will create a tensor on GPU.
- `convert_to_numpy` will bring the tensor to CPU before converting it to NumPy.
"""

from keras_core.backend.torch import core
from keras_core.backend.torch import image
from keras_core.backend.torch import math
from keras_core.backend.torch import nn
from keras_core.backend.torch import numpy
from keras_core.backend.torch import random
from keras_core.backend.torch.core import DYNAMIC_SHAPES_OK
from keras_core.backend.torch.core import Variable
from keras_core.backend.torch.core import cast
from keras_core.backend.torch.core import compute_output_spec
from keras_core.backend.torch.core import cond
from keras_core.backend.torch.core import convert_to_numpy
from keras_core.backend.torch.core import convert_to_tensor
from keras_core.backend.torch.core import is_tensor
from keras_core.backend.torch.core import scatter
from keras_core.backend.torch.core import shape
from keras_core.backend.torch.core import stop_gradient
from keras_core.backend.torch.core import to_torch_dtype
from keras_core.backend.torch.core import vectorized_map
from keras_core.backend.torch.rnn import cudnn_ok
from keras_core.backend.torch.rnn import gru
from keras_core.backend.torch.rnn import lstm
from keras_core.backend.torch.rnn import rnn
