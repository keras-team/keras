# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""APIs for handling dtypes in Linen Modules."""

from typing import Any
from flax.typing import Dtype
from jax import numpy as jnp


def canonicalize_dtype(
  *args, dtype: Dtype | None = None, inexact: bool = True
) -> Dtype:
  """Canonicalize an optional dtype to the definitive dtype.

  If the ``dtype`` is None this function will infer the dtype. If it is not
  None it will be returned unmodified or an exceptions is raised if the dtype
  is invalid.
  from the input arguments using ``jnp.result_type``.

  Args:
    *args: JAX array compatible values. None values
      are ignored.
    dtype: Optional dtype override. If specified the arguments are cast to
      the specified dtype instead and dtype inference is disabled.
    inexact: When True, the output dtype must be a subdtype
    of `jnp.inexact`. Inexact dtypes are real or complex floating points. This
    is useful when you want to apply operations that don't work directly on
    integers like taking a mean for example.
  Returns:
    The dtype that *args should be cast to.
  """
  if dtype is None:
    args_filtered = [jnp.asarray(x) for x in args if x is not None]
    dtype = jnp.result_type(*args_filtered)
    if inexact and not jnp.issubdtype(dtype, jnp.inexact):
      dtype = jnp.promote_types(jnp.float32, dtype)
  if inexact and not jnp.issubdtype(dtype, jnp.inexact):
    raise ValueError(f'Dtype must be inexact: {dtype}')
  return dtype


def promote_dtype(*args, dtype=None, inexact=True) -> list[Any]:
  """ "Promotes input arguments to a specified or inferred dtype.

  All args are cast to the same dtype. See ``canonicalize_dtype`` for how
  this dtype is determined.

  The behavior of promote_dtype is mostly a convinience wrapper around
  ``jax.numpy.promote_types``. The differences being that it automatically casts
  all input to the inferred dtypes, allows inference to be overridden by a
  forced dtype, and has an optional check to garantuee the resulting dtype is
  inexact.

  Args:
    *args: JAX array compatible values. None values are returned as is.
    dtype: Optional dtype override. If specified the arguments are cast to the
      specified dtype instead and dtype inference is disabled.
    inexact: When True, the output dtype must be a subdtype of `jnp.inexact`.
      Inexact dtypes are real or complex floating points. This is useful when
      you want to apply operations that don't work directly on integers like
      taking a mean for example.

  Returns:
    The arguments cast to arrays of the same dtype.
  """
  dtype = canonicalize_dtype(*args, dtype=dtype, inexact=inexact)
  return [jnp.asarray(x, dtype) if x is not None else None for x in args]
