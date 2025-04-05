# Copyright 2018 The JAX Authors.
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

import numpy as np

from jax import lax
import jax.numpy as jnp

from jax._src.lax import lax as lax_internal
from jax._src import dtypes
from jax._src.tree_util import tree_flatten, tree_unflatten
from jax._src.util import safe_zip, unzip2, HashablePartial

zip = safe_zip


def ravel_pytree(pytree):
  """Ravel (flatten) a pytree of arrays down to a 1D array.

  Args:
    pytree: a pytree of arrays and scalars to ravel.

  Returns:
    A pair where the first element is a 1D array representing the flattened and
    concatenated leaf values, with dtype determined by promoting the dtypes of
    leaf values, and the second element is a callable for unflattening a 1D
    vector of the same length back to a pytree of the same structure as the
    input ``pytree``. If the input pytree is empty (i.e. has no leaves) then as
    a convention a 1D empty array of dtype float32 is returned in the first
    component of the output.

  For details on dtype promotion, see
  https://jax.readthedocs.io/en/latest/type_promotion.html.

  """
  leaves, treedef = tree_flatten(pytree)
  flat, unravel_list = _ravel_list(leaves)
  return flat, HashablePartial(unravel_pytree, treedef, unravel_list)

def unravel_pytree(treedef, unravel_list, flat):
  return tree_unflatten(treedef, unravel_list(flat))

def _ravel_list(lst):
  if not lst: return jnp.array([], jnp.float32), lambda _: []
  from_dtypes = tuple(dtypes.dtype(l) for l in lst)
  to_dtype = dtypes.result_type(*from_dtypes)
  sizes, shapes = unzip2((jnp.size(x), jnp.shape(x)) for x in lst)
  indices = tuple(np.cumsum(sizes))

  if all(dt == to_dtype for dt in from_dtypes):
    # Skip any dtype conversion, resulting in a dtype-polymorphic `unravel`.
    # See https://github.com/jax-ml/jax/issues/7809.
    del from_dtypes, to_dtype
    raveled = jnp.concatenate([jnp.ravel(e) for e in lst])
    return raveled, HashablePartial(_unravel_list_single_dtype, indices, shapes)

  # When there is more than one distinct input dtype, we perform type
  # conversions and produce a dtype-specific unravel function.
  ravel = lambda e: jnp.ravel(lax.convert_element_type(e, to_dtype))
  raveled = jnp.concatenate([ravel(e) for e in lst])
  unrav = HashablePartial(_unravel_list, indices, shapes, from_dtypes, to_dtype)
  return raveled, unrav

def _unravel_list_single_dtype(indices, shapes, arr):
  chunks = jnp.split(arr, indices[:-1])
  return [chunk.reshape(shape) for chunk, shape in zip(chunks, shapes)]

def _unravel_list(indices, shapes, from_dtypes, to_dtype, arr):
  arr_dtype = dtypes.dtype(arr)
  if arr_dtype != to_dtype:
    raise TypeError(f"unravel function given array of dtype {arr_dtype}, "
                    f"but expected dtype {to_dtype}")
  chunks = jnp.split(arr, indices[:-1])
  return [
    lax_internal._convert_element_type(chunk.reshape(shape), dtype,
                                       warn_on_complex_to_real_cast=False)
    for chunk, shape, dtype in zip(chunks, shapes, from_dtypes)
  ]
