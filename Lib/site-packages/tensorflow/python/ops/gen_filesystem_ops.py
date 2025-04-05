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

def file_system_set_configuration(scheme: Annotated[Any, _atypes.String], key: Annotated[Any, _atypes.String], value: Annotated[Any, _atypes.String], name=None):
  r"""Set configuration of the file system.

  Args:
    scheme: A `Tensor` of type `string`. File system scheme.
    key: A `Tensor` of type `string`. The name of the configuration option.
    value: A `Tensor` of type `string`. The value of the configuration option.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FileSystemSetConfiguration", name, scheme, key, value)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return file_system_set_configuration_eager_fallback(
          scheme, key, value, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FileSystemSetConfiguration", scheme=scheme, key=key, value=value,
                                      name=name)
  return _op
FileSystemSetConfiguration = tf_export("raw_ops.FileSystemSetConfiguration")(_ops.to_raw_op(file_system_set_configuration))


def file_system_set_configuration_eager_fallback(scheme: Annotated[Any, _atypes.String], key: Annotated[Any, _atypes.String], value: Annotated[Any, _atypes.String], name, ctx):
  scheme = _ops.convert_to_tensor(scheme, _dtypes.string)
  key = _ops.convert_to_tensor(key, _dtypes.string)
  value = _ops.convert_to_tensor(value, _dtypes.string)
  _inputs_flat = [scheme, key, value]
  _attrs = None
  _result = _execute.execute(b"FileSystemSetConfiguration", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result

