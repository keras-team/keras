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

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('check_preemption')
def check_preemption(preemption_key:str="TF_DEFAULT_PREEMPTION_NOTICE_KEY", name=None):
  r"""Check if a preemption notice has been received in coordination service.

  Args:
    preemption_key: An optional `string`. Defaults to `"TF_DEFAULT_PREEMPTION_NOTICE_KEY"`.
      Key for preemption check in coordination service.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CheckPreemption", name, "preemption_key", preemption_key)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_check_preemption(
          (preemption_key, name,), None)
      if _result is not NotImplemented:
        return _result
      return check_preemption_eager_fallback(
          preemption_key=preemption_key, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            check_preemption, (), dict(preemption_key=preemption_key,
                                       name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_check_preemption(
        (preemption_key, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if preemption_key is None:
    preemption_key = "TF_DEFAULT_PREEMPTION_NOTICE_KEY"
  preemption_key = _execute.make_str(preemption_key, "preemption_key")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CheckPreemption", preemption_key=preemption_key, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          check_preemption, (), dict(preemption_key=preemption_key, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
CheckPreemption = tf_export("raw_ops.CheckPreemption")(_ops.to_raw_op(check_preemption))
_dispatcher_for_check_preemption = check_preemption._tf_type_based_dispatcher.Dispatch


def check_preemption_eager_fallback(preemption_key: str, name, ctx):
  if preemption_key is None:
    preemption_key = "TF_DEFAULT_PREEMPTION_NOTICE_KEY"
  preemption_key = _execute.make_str(preemption_key, "preemption_key")
  _inputs_flat = []
  _attrs = ("preemption_key", preemption_key)
  _result = _execute.execute(b"CheckPreemption", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result

