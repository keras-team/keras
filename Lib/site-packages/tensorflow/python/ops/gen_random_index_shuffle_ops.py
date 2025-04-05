"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
"""

import collections

from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes

from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export

from typing import TypeVar, List, Any
from typing_extensions import Annotated

TV_RandomIndexShuffle_dtype = TypeVar("TV_RandomIndexShuffle_dtype", _atypes.Int32, _atypes.Int64, _atypes.UInt32, _atypes.UInt64)
TV_RandomIndexShuffle_Tseed = TypeVar("TV_RandomIndexShuffle_Tseed", _atypes.Int32, _atypes.Int64, _atypes.UInt32, _atypes.UInt64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('random_index_shuffle')
def random_index_shuffle(index: Annotated[Any, TV_RandomIndexShuffle_dtype], seed: Annotated[Any, TV_RandomIndexShuffle_Tseed], max_index: Annotated[Any, TV_RandomIndexShuffle_dtype], rounds:int=4, name=None) -> Annotated[Any, TV_RandomIndexShuffle_dtype]:
  r"""Outputs the position of `value` in a permutation of [0, ..., max_index].

  Output values are a bijection of the `index` for any combination and `seed` and `max_index`.

  If multiple inputs are vectors (matrix in case of seed) then the size of the
  first dimension must match.

  The outputs are deterministic.

  Args:
    index: A `Tensor`. Must be one of the following types: `int32`, `uint32`, `int64`, `uint64`.
      A scalar tensor or a vector of dtype `dtype`. The index (or indices) to be shuffled. Must be within [0, max_index].
    seed: A `Tensor`. Must be one of the following types: `int32`, `uint32`, `int64`, `uint64`.
      A tensor of dtype `Tseed` and shape [3] or [n, 3]. The random seed.
    max_index: A `Tensor`. Must have the same type as `index`.
      A scalar tensor or vector of dtype `dtype`. The upper bound(s) of the interval (inclusive).
    rounds: An optional `int`. Defaults to `4`.
      The number of rounds to use the in block cipher.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `index`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RandomIndexShuffle", name, index, seed, max_index, "rounds",
        rounds)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_random_index_shuffle(
          (index, seed, max_index, rounds, name,), None)
      if _result is not NotImplemented:
        return _result
      return random_index_shuffle_eager_fallback(
          index, seed, max_index, rounds=rounds, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            random_index_shuffle, (), dict(index=index, seed=seed,
                                           max_index=max_index, rounds=rounds,
                                           name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_random_index_shuffle(
        (index, seed, max_index, rounds, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if rounds is None:
    rounds = 4
  rounds = _execute.make_int(rounds, "rounds")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RandomIndexShuffle", index=index, seed=seed, max_index=max_index,
                              rounds=rounds, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          random_index_shuffle, (), dict(index=index, seed=seed,
                                         max_index=max_index, rounds=rounds,
                                         name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("rounds", _op._get_attr_int("rounds"), "dtype",
              _op._get_attr_type("dtype"), "Tseed",
              _op._get_attr_type("Tseed"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RandomIndexShuffle", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RandomIndexShuffle = tf_export("raw_ops.RandomIndexShuffle")(_ops.to_raw_op(random_index_shuffle))
_dispatcher_for_random_index_shuffle = random_index_shuffle._tf_type_based_dispatcher.Dispatch


def random_index_shuffle_eager_fallback(index: Annotated[Any, TV_RandomIndexShuffle_dtype], seed: Annotated[Any, TV_RandomIndexShuffle_Tseed], max_index: Annotated[Any, TV_RandomIndexShuffle_dtype], rounds: int, name, ctx) -> Annotated[Any, TV_RandomIndexShuffle_dtype]:
  if rounds is None:
    rounds = 4
  rounds = _execute.make_int(rounds, "rounds")
  _attr_dtype, _inputs_dtype = _execute.args_to_matching_eager([index, max_index], ctx, [_dtypes.int32, _dtypes.uint32, _dtypes.int64, _dtypes.uint64, ])
  (index, max_index) = _inputs_dtype
  _attr_Tseed, (seed,) = _execute.args_to_matching_eager([seed], ctx, [_dtypes.int32, _dtypes.uint32, _dtypes.int64, _dtypes.uint64, ])
  _inputs_flat = [index, seed, max_index]
  _attrs = ("rounds", rounds, "dtype", _attr_dtype, "Tseed", _attr_Tseed)
  _result = _execute.execute(b"RandomIndexShuffle", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RandomIndexShuffle", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

