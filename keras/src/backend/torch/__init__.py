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
from keras.src.backend.torch import (core, image, linalg, math, nn, numpy,
                                     random)
from keras.src.backend.torch.core import (IS_THREAD_SAFE,
                                          SUPPORTS_SPARSE_TENSORS, Variable,
                                          cast, compute_output_spec, cond,
                                          convert_to_numpy, convert_to_tensor,
                                          device_scope, is_tensor,
                                          random_seed_dtype, scatter, shape,
                                          stop_gradient, to_torch_dtype,
                                          vectorized_map)
from keras.src.backend.torch.rnn import cudnn_ok, gru, lstm, rnn
