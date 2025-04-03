# Copyright 2023 The JAX Authors.
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

"""
The JAX typing module is where JAX-specific static type annotations live.
This submodule is a work in progress; to see the proposal behind the types exported
here, see https://jax.readthedocs.io/en/latest/jep/12049-type-annotations.html.

The currently-available types are:

- :class:`jax.Array`: annotation for any JAX array or tracer (i.e. representations of arrays
  within JAX transforms).
- :obj:`jax.typing.ArrayLike`: annotation for any value that is safe to implicitly cast to
  a JAX array; this includes :class:`jax.Array`, :class:`numpy.ndarray`, as well as Python
  builtin numeric values (e.g. :class:`int`, :class:`float`, etc.) and numpy scalar values
  (e.g. :class:`numpy.int32`, :class:`numpy.float64`, etc.)
- :obj:`jax.typing.DTypeLike`: annotation for any value that can be cast to a JAX-compatible
  dtype; this includes strings (e.g. `'float32'`, `'int32'`), scalar types (e.g. `float`,
  `np.float32`), dtypes (e.g. `np.dtype('float32')`), or objects with a dtype attribute
  (e.g. `jnp.float32`, `jnp.int32`).

We may add additional types here in future releases.

JAX Typing Best Practices
-------------------------
When annotating JAX arrays in public API functions, we recommend using :class:`~jax.typing.ArrayLike`
for array inputs, and :class:`~jax.Array` for array outputs.

For example, your function might look like this::

    import numpy as np
    import jax.numpy as jnp
    from jax import Array
    from jax.typing import ArrayLike

    def my_function(x: ArrayLike) -> Array:
      # Runtime type validation, Python 3.10 or newer:
      if not isinstance(x, ArrayLike):
        raise TypeError(f"Expected arraylike input; got {x}")
      # Runtime type validation, any Python version:
      if not (isinstance(x, (np.ndarray, Array)) or np.isscalar(x)):
        raise TypeError(f"Expected arraylike input; got {x}")

      # Convert input to jax.Array:
      x_arr = jnp.asarray(x)

      # ... do some computation; JAX functions will return Array types:
      result = x_arr.sum(0) / x_arr.shape[0]

      # return an Array
      return result

Most of JAX's public APIs follow this pattern. Note in particular that we recommend JAX functions
to not accept sequences such as :class:`list` or :class:`tuple` in place of arrays, as this can
cause extra overhead in JAX transforms like :func:`~jax.jit` and can behave in unexpected ways with
batch-wise transforms like :func:`~jax.vmap` or :func:`jax.pmap`. For more information on this,
see `Non-array inputs NumPy vs JAX`_

.. _Non-array inputs NumPy vs JAX: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#non-array-inputs-numpy-vs-jax
"""
from jax._src.typing import (
    ArrayLike as ArrayLike,
    DTypeLike as DTypeLike,
)
