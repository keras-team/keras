# Copyright 2024 The Flax Authors.
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

import functools
from typing import Any

import jax
from jax import core
from jax.extend import linear_util as lu
from jax.interpreters import partial_eval as pe

from flax import errors


def _maybe_unknown(x: Any) -> pe.PartialVal:
  if isinstance(x, jax.ShapeDtypeStruct):
    return pe.PartialVal.unknown(core.ShapedArray(x.shape, x.dtype))
  else:
    return pe.PartialVal.known(x)


def lazy_init(fn):
  """Lazily evaluates a function by using the shapes of the inputs.

  The returned function accepts a combination of JAX values and
  ``jax.ShapeDtypeStruct`` instances for the inputs for which we
  don't need concrete values (only the shape and dtype).

  This API is used by ``core.lazy_init`` or ``Module.lazy_init``
  to initialize variables without doing any actual computation on the
  inputs.

  Args:
    fn: the function to be lazily evaluated.
  Returns:
    A new function that accepts a mix of concrete values and
    ``jax.ShapeDtypeStruct`` instances.
  """

  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    # TODO(mattjj,jheek): use a public JAX API
    # flatten fn and prepare for internal JAX transform
    inputs_flat, in_tree = jax.tree_util.tree_flatten((args, kwargs))
    debug_info = jax.api_util.debug_info("lazy_init", fn, (in_tree,), {})
    f_flat, out_tree = jax.api_util.flatten_fun(
      lu.wrap_init(fn, debug_info=debug_info), in_tree)
    # map inputs to PartialVal known/unknown
    # only the computations depending on knowns will be executed
    in_pvals = [_maybe_unknown(x) for x in inputs_flat]
    _, out_pvals, _ = pe.trace_to_jaxpr_nounits(f_flat, in_pvals)
    # all outputs should be knowns. If this fails
    # the user is creating variables that depend on a
    # argument that was passed as a ShapeDtypeStruct.
    out_flat = []
    for pv, const in out_pvals:
      if pv is None:
        # const is the actual value of the known output
        out_flat.append(const)
      else:
        raise errors.LazyInitError(pv)
    return jax.tree_util.tree_unflatten(out_tree(), out_flat)

  return wrapper
