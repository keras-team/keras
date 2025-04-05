# Copyright 2021 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import operator

from jax import dtypes
from jax import vmap
from jax import random
from jax.util import split_list
import jax.numpy as jnp
from jax.experimental import sparse


def random_bcoo(key, shape, *, dtype=jnp.float_, indices_dtype=None,
                nse=0.2, n_batch=0, n_dense=0, unique_indices=True,
                sorted_indices=False, generator=random.uniform, **kwds):
  """Generate a random BCOO matrix.

  Args:
    key : PRNG key to be passed to ``generator`` function.
    shape : tuple specifying the shape of the array to be generated.
    dtype : dtype of the array to be generated.
    indices_dtype: dtype of the BCOO indices.
    nse : number of specified elements in the matrix, or if 0 < nse < 1, a
      fraction of sparse dimensions to be specified (default: 0.2).
    n_batch : number of batch dimensions. must satisfy ``n_batch >= 0`` and
      ``n_batch + n_dense <= len(shape)``.
    n_dense : number of batch dimensions. must satisfy ``n_dense >= 0`` and
      ``n_batch + n_dense <= len(shape)``.
    unique_indices : boolean specifying whether indices should be unique
      (default: True).
    sorted_indices : boolean specifying whether indices should be row-sorted in
      lexicographical order (default: False).
    generator : function for generating random values accepting a key, shape,
      and dtype. It defaults to :func:`jax.random.uniform`, and may be any
      function with a similar signature.
    **kwds : additional keyword arguments to pass to ``generator``.

  Returns:
    arr : a sparse.BCOO array with the specified properties.
  """
  shape = tuple(map(operator.index, shape))
  n_batch = operator.index(n_batch)
  n_dense = operator.index(n_dense)
  if n_batch < 0 or n_dense < 0 or n_batch + n_dense > len(shape):
    raise ValueError(f"Invalid {n_batch=}, {n_dense=} for {shape=}")
  n_sparse = len(shape) - n_batch - n_dense
  batch_shape, sparse_shape, dense_shape = map(tuple, split_list(shape, [n_batch, n_sparse]))
  batch_size = math.prod(batch_shape)
  sparse_size = math.prod(sparse_shape)
  if not 0 <= nse < sparse_size:
    raise ValueError(f"got {nse=}, expected to be between 0 and {sparse_size}")
  if 0 < nse < 1:
    nse = int(math.ceil(nse * sparse_size))
  nse = operator.index(nse)

  data_shape = batch_shape + (nse,) + dense_shape
  indices_shape = batch_shape + (nse, n_sparse)
  if indices_dtype is None:
    indices_dtype = dtypes.canonicalize_dtype(jnp.int_)
  if sparse_size > jnp.iinfo(indices_dtype).max:
    raise ValueError(f"{indices_dtype=} does not have enough range to generate "
                     f"sparse indices of size {sparse_size}.")
  @vmap
  def _indices(key):
    if not sparse_shape:
      return jnp.empty((nse, n_sparse), dtype=indices_dtype)
    flat_ind = random.choice(key, sparse_size, shape=(nse,),
                             replace=not unique_indices).astype(indices_dtype)
    return jnp.column_stack(jnp.unravel_index(flat_ind, sparse_shape))

  keys = random.split(key, batch_size + 1)
  data_key, index_keys = keys[0], keys[1:]
  data = generator(data_key, shape=data_shape, dtype=dtype, **kwds)
  indices = _indices(index_keys).reshape(indices_shape)
  mat = sparse.BCOO((data, indices), shape=shape)
  return mat.sort_indices() if sorted_indices else mat
