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

def case(branch_index: Annotated[Any, _atypes.Int32], input, Tout, branches, output_shapes=[], name=None):
  r"""An n-way switch statement which calls a single branch function.

      An n-way switch statement, implementing the following:
      ```
      switch (branch_index) {
        case 0:
          output = branches[0](input);
          break;
        case 1:
          output = branches[1](input);
          break;
        ...
        case [[nbranches-1]]:
        default:
          output = branches[nbranches-1](input);
          break;
      }
      ```

  Args:
    branch_index: A `Tensor` of type `int32`.
      The branch selector, an int32 Tensor.
    input: A list of `Tensor` objects.
      A list of input tensors passed to the branch function.
    Tout: A list of `tf.DTypes`. A list of output types.
    branches: A list of functions decorated with @Defun that has length `>= 1`.
            A list of functions each of which takes 'inputs' and returns a list of
            tensors, whose types are the same as what every other branch returns.
    output_shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Case", name, branch_index, input, "Tout", Tout, "branches",
        branches, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return case_eager_fallback(
          branch_index, input, Tout=Tout, branches=branches,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'case' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  if not isinstance(branches, (list, tuple)):
    raise TypeError(
        "Expected list for 'branches' argument to "
        "'case' Op, not %r." % branches)
  if output_shapes is None:
    output_shapes = []
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'case' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Case", branch_index=branch_index, input=input, Tout=Tout,
                branches=branches, output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("Tin", _op.get_attr("Tin"), "Tout", _op.get_attr("Tout"),
              "branches", _op.get_attr("branches"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Case", _inputs_flat, _attrs, _result)
  return _result

Case = tf_export("raw_ops.Case")(_ops.to_raw_op(case))


def case_eager_fallback(branch_index: Annotated[Any, _atypes.Int32], input, Tout, branches, output_shapes, name, ctx):
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'case' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  if not isinstance(branches, (list, tuple)):
    raise TypeError(
        "Expected list for 'branches' argument to "
        "'case' Op, not %r." % branches)
  if output_shapes is None:
    output_shapes = []
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'case' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _attr_Tin, input = _execute.convert_to_mixed_eager_tensors(input, ctx)
  branch_index = _ops.convert_to_tensor(branch_index, _dtypes.int32)
  _inputs_flat = [branch_index] + list(input)
  _attrs = ("Tin", _attr_Tin, "Tout", Tout, "branches", branches,
  "output_shapes", output_shapes)
  _result = _execute.execute(b"Case", len(Tout), inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Case", _inputs_flat, _attrs, _result)
  return _result


def device_index(device_names, name=None) -> Annotated[Any, _atypes.Int32]:
  r"""Return the index of device the op runs.

  Given a list of device names, this operation returns the index of the device
  this op runs. The length of the list is returned in two cases:
  (1) Device does not exist in the given device list.
  (2) It is in XLA compilation.

  Args:
    device_names: A list of `strings`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DeviceIndex", name, "device_names", device_names)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return device_index_eager_fallback(
          device_names=device_names, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(device_names, (list, tuple)):
    raise TypeError(
        "Expected list for 'device_names' argument to "
        "'device_index' Op, not %r." % device_names)
  device_names = [_execute.make_str(_s, "device_names") for _s in device_names]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DeviceIndex", device_names=device_names, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("device_names", _op.get_attr("device_names"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DeviceIndex", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DeviceIndex = tf_export("raw_ops.DeviceIndex")(_ops.to_raw_op(device_index))


def device_index_eager_fallback(device_names, name, ctx) -> Annotated[Any, _atypes.Int32]:
  if not isinstance(device_names, (list, tuple)):
    raise TypeError(
        "Expected list for 'device_names' argument to "
        "'device_index' Op, not %r." % device_names)
  device_names = [_execute.make_str(_s, "device_names") for _s in device_names]
  _inputs_flat = []
  _attrs = ("device_names", device_names)
  _result = _execute.execute(b"DeviceIndex", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DeviceIndex", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_FakeParam_dtype = TypeVar("TV_FakeParam_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def fake_param(dtype: TV_FakeParam_dtype, shape, name=None) -> Annotated[Any, TV_FakeParam_dtype]:
  r"""  This op is used as a placeholder in If branch functions. It doesn't provide a
  valid output when run, so must either be removed (e.g. replaced with a
  function input) or guaranteed not to be used (e.g. if mirroring an
  intermediate output needed for the gradient computation of the other branch).

  Args:
    dtype: A `tf.DType`. The type of the output.
    shape: A `tf.TensorShape` or list of `ints`.
          The purported shape of the output. This is only used for shape inference;
          the output will not necessarily have this shape. Can be a partial shape.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FakeParam", name, "dtype", dtype, "shape", shape)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return fake_param_eager_fallback(
          dtype=dtype, shape=shape, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  shape = _execute.make_shape(shape, "shape")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FakeParam", dtype=dtype, shape=shape, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "shape",
              _op.get_attr("shape"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FakeParam", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

FakeParam = tf_export("raw_ops.FakeParam")(_ops.to_raw_op(fake_param))


def fake_param_eager_fallback(dtype: TV_FakeParam_dtype, shape, name, ctx) -> Annotated[Any, TV_FakeParam_dtype]:
  dtype = _execute.make_type(dtype, "dtype")
  shape = _execute.make_shape(shape, "shape")
  _inputs_flat = []
  _attrs = ("dtype", dtype, "shape", shape)
  _result = _execute.execute(b"FakeParam", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FakeParam", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def _for(start: Annotated[Any, _atypes.Int32], limit: Annotated[Any, _atypes.Int32], delta: Annotated[Any, _atypes.Int32], input, body, name=None):
  r"""Applies a for loop.

    ```python
     output = input;
     for i in range(start, limit, delta)
       output = body(i, output);
    ```

  Args:
    start: A `Tensor` of type `int32`. The lower bound. An int32
    limit: A `Tensor` of type `int32`. The upper bound. An int32
    delta: A `Tensor` of type `int32`. The increment. An int32
    input: A list of `Tensor` objects.
      A list of input tensors whose types are T.
    body: A function decorated with @Defun.
          A function that takes a list of tensors (int32, T) and returns another
          list of tensors (T).
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "For", name, start, limit, delta, input, "body", body)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return _for_eager_fallback(
          start, limit, delta, input, body=body, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "For", start=start, limit=limit, delta=delta, input=input, body=body,
               name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op.get_attr("T"), "body", _op.get_attr("body"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "For", _inputs_flat, _attrs, _result)
  return _result

For = tf_export("raw_ops.For")(_ops.to_raw_op(_for))


def _for_eager_fallback(start: Annotated[Any, _atypes.Int32], limit: Annotated[Any, _atypes.Int32], delta: Annotated[Any, _atypes.Int32], input, body, name, ctx):
  _attr_T, input = _execute.convert_to_mixed_eager_tensors(input, ctx)
  start = _ops.convert_to_tensor(start, _dtypes.int32)
  limit = _ops.convert_to_tensor(limit, _dtypes.int32)
  delta = _ops.convert_to_tensor(delta, _dtypes.int32)
  _inputs_flat = [start, limit, delta] + list(input)
  _attrs = ("T", _attr_T, "body", body)
  _result = _execute.execute(b"For", len(input), inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "For", _inputs_flat, _attrs, _result)
  return _result


TV_If_Tcond = TypeVar("TV_If_Tcond", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def _if(cond: Annotated[Any, TV_If_Tcond], input, Tout, then_branch, else_branch, output_shapes=[], name=None):
  r"""output = cond ? then_branch(input) : else_branch(input)

  Args:
    cond: A `Tensor`.
            A Tensor. If the tensor is a scalar of non-boolean type, the
            scalar is converted to a boolean according to the
            following rule: if the scalar is a numerical value, non-zero means
            `True` and zero means False; if the scalar is a string, non-empty
            means `True` and empty means `False`. If the tensor is not a scalar,
            being empty means False and being non-empty means True.
    input: A list of `Tensor` objects. A list of input tensors.
    Tout: A list of `tf.DTypes`. A list of output types.
    then_branch: A function decorated with @Defun.
            A function that takes 'inputs' and returns a list of tensors, whose
            types are the same as what else_branch returns.
    else_branch: A function decorated with @Defun.
          A function that takes 'inputs' and returns a list of tensors, whose
          types are the same as what then_branch returns.
    output_shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "If", name, cond, input, "Tout", Tout, "then_branch",
        then_branch, "else_branch", else_branch, "output_shapes",
        output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return _if_eager_fallback(
          cond, input, Tout=Tout, then_branch=then_branch,
          else_branch=else_branch, output_shapes=output_shapes, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'if' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  if output_shapes is None:
    output_shapes = []
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'if' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "If", cond=cond, input=input, Tout=Tout, then_branch=then_branch,
              else_branch=else_branch, output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("Tcond", _op._get_attr_type("Tcond"), "Tin",
              _op.get_attr("Tin"), "Tout", _op.get_attr("Tout"),
              "then_branch", _op.get_attr("then_branch"), "else_branch",
              _op.get_attr("else_branch"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "If", _inputs_flat, _attrs, _result)
  return _result

If = tf_export("raw_ops.If")(_ops.to_raw_op(_if))


def _if_eager_fallback(cond: Annotated[Any, TV_If_Tcond], input, Tout, then_branch, else_branch, output_shapes, name, ctx):
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'if' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  if output_shapes is None:
    output_shapes = []
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'if' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _attr_Tcond, (cond,) = _execute.args_to_matching_eager([cond], ctx, [])
  _attr_Tin, input = _execute.convert_to_mixed_eager_tensors(input, ctx)
  _inputs_flat = [cond] + list(input)
  _attrs = ("Tcond", _attr_Tcond, "Tin", _attr_Tin, "Tout", Tout,
  "then_branch", then_branch, "else_branch", else_branch, "output_shapes",
  output_shapes)
  _result = _execute.execute(b"If", len(Tout), inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "If", _inputs_flat, _attrs, _result)
  return _result


def partitioned_call(args, Tout, f, config:str="", config_proto:str="", executor_type:str="", name=None):
  r"""returns `f(inputs)`, where `f`'s body is placed and partitioned.

  Asynchronously executes a function, potentially across multiple devices but
  within a single process. The kernel places and partitions a given function's
  underlying graph, and executes each of the partitioned subgraphs as a function.

  Args:
    args: A list of `Tensor` objects. A list of input tensors.
    Tout: A list of `tf.DTypes`. A list of output types.
    f: A function decorated with @Defun.
            A function that takes 'args', a list of tensors, and returns 'output',
            another list of tensors. Input and output types are specified by 'Tin'
            and 'Tout'. The function body of f will be placed and partitioned across
            devices, setting this op apart from the regular Call op.
    config: An optional `string`. Defaults to `""`.
    config_proto: An optional `string`. Defaults to `""`.
    executor_type: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "PartitionedCall", name, args, "Tout", Tout, "f", f, "config",
        config, "config_proto", config_proto, "executor_type", executor_type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return partitioned_call_eager_fallback(
          args, Tout=Tout, f=f, config=config, config_proto=config_proto,
          executor_type=executor_type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'partitioned_call' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  if config_proto is None:
    config_proto = ""
  config_proto = _execute.make_str(config_proto, "config_proto")
  if executor_type is None:
    executor_type = ""
  executor_type = _execute.make_str(executor_type, "executor_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "PartitionedCall", args=args, Tout=Tout, f=f, config=config,
                           config_proto=config_proto,
                           executor_type=executor_type, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tin", _op.get_attr("Tin"), "Tout", _op.get_attr("Tout"), "f",
              _op.get_attr("f"), "config", _op.get_attr("config"),
              "config_proto", _op.get_attr("config_proto"), "executor_type",
              _op.get_attr("executor_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "PartitionedCall", _inputs_flat, _attrs, _result)
  return _result

PartitionedCall = tf_export("raw_ops.PartitionedCall")(_ops.to_raw_op(partitioned_call))


def partitioned_call_eager_fallback(args, Tout, f, config: str, config_proto: str, executor_type: str, name, ctx):
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'partitioned_call' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  if config_proto is None:
    config_proto = ""
  config_proto = _execute.make_str(config_proto, "config_proto")
  if executor_type is None:
    executor_type = ""
  executor_type = _execute.make_str(executor_type, "executor_type")
  _attr_Tin, args = _execute.convert_to_mixed_eager_tensors(args, ctx)
  _inputs_flat = list(args)
  _attrs = ("Tin", _attr_Tin, "Tout", Tout, "f", f, "config", config,
  "config_proto", config_proto, "executor_type", executor_type)
  _result = _execute.execute(b"PartitionedCall", len(Tout),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "PartitionedCall", _inputs_flat, _attrs, _result)
  return _result


def remote_call(target: Annotated[Any, _atypes.String], args, Tout, f, name=None):
  r"""Runs function `f` on a remote device indicated by `target`.

  Args:
    target: A `Tensor` of type `string`.
      A fully specified device name where we want to run the function.
    args: A list of `Tensor` objects. A list of arguments for the function.
    Tout: A list of `tf.DTypes` that has length `>= 1`.
      The type list for the return values.
    f: A function decorated with @Defun. The function to run remotely.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RemoteCall", name, target, args, "Tout", Tout, "f", f)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return remote_call_eager_fallback(
          target, args, Tout=Tout, f=f, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'remote_call' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RemoteCall", target=target, args=args, Tout=Tout, f=f, name=name)
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("Tin", _op.get_attr("Tin"), "Tout", _op.get_attr("Tout"), "f",
              _op.get_attr("f"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RemoteCall", _inputs_flat, _attrs, _result)
  return _result

RemoteCall = tf_export("raw_ops.RemoteCall")(_ops.to_raw_op(remote_call))


def remote_call_eager_fallback(target: Annotated[Any, _atypes.String], args, Tout, f, name, ctx):
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'remote_call' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  _attr_Tin, args = _execute.convert_to_mixed_eager_tensors(args, ctx)
  target = _ops.convert_to_tensor(target, _dtypes.string)
  _inputs_flat = [target] + list(args)
  _attrs = ("Tin", _attr_Tin, "Tout", Tout, "f", f)
  _result = _execute.execute(b"RemoteCall", len(Tout), inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RemoteCall", _inputs_flat, _attrs, _result)
  return _result


def stateful_partitioned_call(args, Tout, f, config:str="", config_proto:str="", executor_type:str="", name=None):
  r"""returns `f(inputs)`, where `f`'s body is placed and partitioned.

  Args:
    args: A list of `Tensor` objects. A list of input tensors.
    Tout: A list of `tf.DTypes`. A list of output types.
    f: A function decorated with @Defun.
            A function that takes 'args', a list of tensors, and returns 'output',
            another list of tensors. Input and output types are specified by 'Tin'
            and 'Tout'. The function body of f will be placed and partitioned across
            devices, setting this op apart from the regular Call op. This op is
            stateful.
    config: An optional `string`. Defaults to `""`.
    config_proto: An optional `string`. Defaults to `""`.
    executor_type: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StatefulPartitionedCall", name, args, "Tout", Tout, "f", f,
        "config", config, "config_proto", config_proto, "executor_type",
        executor_type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateful_partitioned_call_eager_fallback(
          args, Tout=Tout, f=f, config=config, config_proto=config_proto,
          executor_type=executor_type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'stateful_partitioned_call' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  if config_proto is None:
    config_proto = ""
  config_proto = _execute.make_str(config_proto, "config_proto")
  if executor_type is None:
    executor_type = ""
  executor_type = _execute.make_str(executor_type, "executor_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatefulPartitionedCall", args=args, Tout=Tout, f=f, config=config,
                                   config_proto=config_proto,
                                   executor_type=executor_type, name=name)
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("Tin", _op.get_attr("Tin"), "Tout", _op.get_attr("Tout"), "f",
              _op.get_attr("f"), "config", _op.get_attr("config"),
              "config_proto", _op.get_attr("config_proto"), "executor_type",
              _op.get_attr("executor_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatefulPartitionedCall", _inputs_flat, _attrs, _result)
  return _result

StatefulPartitionedCall = tf_export("raw_ops.StatefulPartitionedCall")(_ops.to_raw_op(stateful_partitioned_call))


def stateful_partitioned_call_eager_fallback(args, Tout, f, config: str, config_proto: str, executor_type: str, name, ctx):
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'stateful_partitioned_call' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  if config_proto is None:
    config_proto = ""
  config_proto = _execute.make_str(config_proto, "config_proto")
  if executor_type is None:
    executor_type = ""
  executor_type = _execute.make_str(executor_type, "executor_type")
  _attr_Tin, args = _execute.convert_to_mixed_eager_tensors(args, ctx)
  _inputs_flat = list(args)
  _attrs = ("Tin", _attr_Tin, "Tout", Tout, "f", f, "config", config,
  "config_proto", config_proto, "executor_type", executor_type)
  _result = _execute.execute(b"StatefulPartitionedCall", len(Tout),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatefulPartitionedCall", _inputs_flat, _attrs, _result)
  return _result


def stateless_case(branch_index: Annotated[Any, _atypes.Int32], input, Tout, branches, output_shapes=[], name=None):
  r"""An n-way switch statement which calls a single branch function.

      An n-way switch statement, implementing the following:
      ```
      switch (branch_index) {
        case 0:
          output = branches[0](input);
          break;
        case 1:
          output = branches[1](input);
          break;
        ...
        case [[nbranches-1]]:
        default:
          output = branches[nbranches-1](input);
          break;
      }
      ```

      This should only be used when the none of branches has stateful ops.

  Args:
    branch_index: A `Tensor` of type `int32`.
      The branch selector, an int32 Tensor.
    input: A list of `Tensor` objects.
      A list of input tensors passed to the branch function.
    Tout: A list of `tf.DTypes`. A list of output types.
    branches: A list of functions decorated with @Defun that has length `>= 1`.
            A list of functions each of which takes 'inputs' and returns a list of
            tensors, whose types are the same as what every other branch returns.
    output_shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StatelessCase", name, branch_index, input, "Tout", Tout,
        "branches", branches, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateless_case_eager_fallback(
          branch_index, input, Tout=Tout, branches=branches,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'stateless_case' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  if not isinstance(branches, (list, tuple)):
    raise TypeError(
        "Expected list for 'branches' argument to "
        "'stateless_case' Op, not %r." % branches)
  if output_shapes is None:
    output_shapes = []
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'stateless_case' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatelessCase", branch_index=branch_index, input=input, Tout=Tout,
                         branches=branches, output_shapes=output_shapes,
                         name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tin", _op.get_attr("Tin"), "Tout", _op.get_attr("Tout"),
              "branches", _op.get_attr("branches"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatelessCase", _inputs_flat, _attrs, _result)
  return _result

StatelessCase = tf_export("raw_ops.StatelessCase")(_ops.to_raw_op(stateless_case))


def stateless_case_eager_fallback(branch_index: Annotated[Any, _atypes.Int32], input, Tout, branches, output_shapes, name, ctx):
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'stateless_case' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  if not isinstance(branches, (list, tuple)):
    raise TypeError(
        "Expected list for 'branches' argument to "
        "'stateless_case' Op, not %r." % branches)
  if output_shapes is None:
    output_shapes = []
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'stateless_case' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _attr_Tin, input = _execute.convert_to_mixed_eager_tensors(input, ctx)
  branch_index = _ops.convert_to_tensor(branch_index, _dtypes.int32)
  _inputs_flat = [branch_index] + list(input)
  _attrs = ("Tin", _attr_Tin, "Tout", Tout, "branches", branches,
  "output_shapes", output_shapes)
  _result = _execute.execute(b"StatelessCase", len(Tout), inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatelessCase", _inputs_flat, _attrs, _result)
  return _result


TV_StatelessIf_Tcond = TypeVar("TV_StatelessIf_Tcond", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def stateless_if(cond: Annotated[Any, TV_StatelessIf_Tcond], input, Tout, then_branch, else_branch, output_shapes=[], name=None):
  r"""output = cond ? then_branch(input) : else_branch(input)

  Args:
    cond: A `Tensor`.
            A Tensor. If the tensor is a scalar of non-boolean type, the
            scalar is converted to a boolean according to the
            following rule: if the scalar is a numerical value, non-zero means
            `True` and zero means False; if the scalar is a string, non-empty
            means `True` and empty means `False`. If the tensor is not a scalar,
            being empty means False and being non-empty means True.

            This should only be used when the if then/else body functions do not
            have stateful ops.
    input: A list of `Tensor` objects. A list of input tensors.
    Tout: A list of `tf.DTypes`. A list of output types.
    then_branch: A function decorated with @Defun.
            A function that takes 'inputs' and returns a list of tensors, whose
            types are the same as what else_branch returns.
    else_branch: A function decorated with @Defun.
          A function that takes 'inputs' and returns a list of tensors, whose
          types are the same as what then_branch returns.
    output_shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StatelessIf", name, cond, input, "Tout", Tout, "then_branch",
        then_branch, "else_branch", else_branch, "output_shapes",
        output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateless_if_eager_fallback(
          cond, input, Tout=Tout, then_branch=then_branch,
          else_branch=else_branch, output_shapes=output_shapes, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'stateless_if' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  if output_shapes is None:
    output_shapes = []
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'stateless_if' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatelessIf", cond=cond, input=input, Tout=Tout,
                       then_branch=then_branch, else_branch=else_branch,
                       output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tcond", _op._get_attr_type("Tcond"), "Tin",
              _op.get_attr("Tin"), "Tout", _op.get_attr("Tout"),
              "then_branch", _op.get_attr("then_branch"), "else_branch",
              _op.get_attr("else_branch"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatelessIf", _inputs_flat, _attrs, _result)
  return _result

StatelessIf = tf_export("raw_ops.StatelessIf")(_ops.to_raw_op(stateless_if))


def stateless_if_eager_fallback(cond: Annotated[Any, TV_StatelessIf_Tcond], input, Tout, then_branch, else_branch, output_shapes, name, ctx):
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'stateless_if' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  if output_shapes is None:
    output_shapes = []
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'stateless_if' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _attr_Tcond, (cond,) = _execute.args_to_matching_eager([cond], ctx, [])
  _attr_Tin, input = _execute.convert_to_mixed_eager_tensors(input, ctx)
  _inputs_flat = [cond] + list(input)
  _attrs = ("Tcond", _attr_Tcond, "Tin", _attr_Tin, "Tout", Tout,
  "then_branch", then_branch, "else_branch", else_branch, "output_shapes",
  output_shapes)
  _result = _execute.execute(b"StatelessIf", len(Tout), inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatelessIf", _inputs_flat, _attrs, _result)
  return _result


def stateless_while(input, cond, body, output_shapes=[], parallel_iterations:int=10, name=None):
  r"""output = input; While (Cond(output)) { output = Body(output) }

  Args:
    input: A list of `Tensor` objects.
      A list of input tensors whose types are T.
    cond: A function decorated with @Defun.
            A function takes 'input' and returns a tensor.  If the tensor is
            a scalar of non-boolean, the scalar is converted to a boolean
            according to the following rule: if the scalar is a numerical
            value, non-zero means True and zero means False; if the scalar is
            a string, non-empty means True and empty means False. If the
            tensor is not a scalar, non-emptiness means True and False
            otherwise.

            This should only be used when the while condition and body functions
            do not have stateful ops.
    body: A function decorated with @Defun.
            A function that takes a list of tensors and returns another
            list of tensors. Both lists have the same types as specified
            by T.
    output_shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
    parallel_iterations: An optional `int`. Defaults to `10`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StatelessWhile", name, input, "cond", cond, "body", body,
        "output_shapes", output_shapes, "parallel_iterations",
        parallel_iterations)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateless_while_eager_fallback(
          input, cond=cond, body=body, output_shapes=output_shapes,
          parallel_iterations=parallel_iterations, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if output_shapes is None:
    output_shapes = []
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'stateless_while' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if parallel_iterations is None:
    parallel_iterations = 10
  parallel_iterations = _execute.make_int(parallel_iterations, "parallel_iterations")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatelessWhile", input=input, cond=cond, body=body,
                          output_shapes=output_shapes,
                          parallel_iterations=parallel_iterations, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op.get_attr("T"), "cond", _op.get_attr("cond"), "body",
              _op.get_attr("body"), "output_shapes",
              _op.get_attr("output_shapes"), "parallel_iterations",
              _op._get_attr_int("parallel_iterations"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatelessWhile", _inputs_flat, _attrs, _result)
  return _result

StatelessWhile = tf_export("raw_ops.StatelessWhile")(_ops.to_raw_op(stateless_while))


def stateless_while_eager_fallback(input, cond, body, output_shapes, parallel_iterations: int, name, ctx):
  if output_shapes is None:
    output_shapes = []
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'stateless_while' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if parallel_iterations is None:
    parallel_iterations = 10
  parallel_iterations = _execute.make_int(parallel_iterations, "parallel_iterations")
  _attr_T, input = _execute.convert_to_mixed_eager_tensors(input, ctx)
  _inputs_flat = list(input)
  _attrs = ("T", _attr_T, "cond", cond, "body", body, "output_shapes",
  output_shapes, "parallel_iterations", parallel_iterations)
  _result = _execute.execute(b"StatelessWhile", len(input),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatelessWhile", _inputs_flat, _attrs, _result)
  return _result


def symbolic_gradient(input, Tout, f, name=None):
  r"""Computes the gradient function for function f via backpropagation.

  Args:
    input: A list of `Tensor` objects. a list of input tensors of size N + M;
    Tout: A list of `tf.DTypes` that has length `>= 1`.
      the type list for the input list.
    f: A function decorated with @Defun.
      The function we want to compute the gradient for.

      The function 'f' must be a numerical function which takes N inputs and
      produces M outputs. Its gradient function 'g', which is computed by
      this SymbolicGradient op is a function taking N + M inputs and
      produces N outputs.

      I.e. if we have
         (y1, y2, ..., y_M) = f(x1, x2, ..., x_N),
      then, g is
         (dL/dx1, dL/dx2, ..., dL/dx_N) = g(x1, x2, ..., x_N,
                                           dL/dy1, dL/dy2, ..., dL/dy_M),

      where L is a scalar-value function of (x1, x2, ..., xN) (e.g., the
      loss function). dL/dx_i is the partial derivative of L with respect
      to x_i.

      (Needs some math expert to say the comment above better.)
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SymbolicGradient", name, input, "Tout", Tout, "f", f)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return symbolic_gradient_eager_fallback(
          input, Tout=Tout, f=f, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'symbolic_gradient' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SymbolicGradient", input=input, Tout=Tout, f=f, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tin", _op.get_attr("Tin"), "Tout", _op.get_attr("Tout"), "f",
              _op.get_attr("f"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SymbolicGradient", _inputs_flat, _attrs, _result)
  return _result

SymbolicGradient = tf_export("raw_ops.SymbolicGradient")(_ops.to_raw_op(symbolic_gradient))


def symbolic_gradient_eager_fallback(input, Tout, f, name, ctx):
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'symbolic_gradient' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  _attr_Tin, input = _execute.convert_to_mixed_eager_tensors(input, ctx)
  _inputs_flat = list(input)
  _attrs = ("Tin", _attr_Tin, "Tout", Tout, "f", f)
  _result = _execute.execute(b"SymbolicGradient", len(Tout),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SymbolicGradient", _inputs_flat, _attrs, _result)
  return _result


TV_ToBool_T = TypeVar("TV_ToBool_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def to_bool(input: Annotated[Any, TV_ToBool_T], name=None) -> Annotated[Any, _atypes.Bool]:
  r"""Converts a tensor to a scalar predicate.

  Converts a tensor to a scalar predicate with the following rules:

  - For 0D tensors, truthiness is determined by comparing against a "zero"
    value. For numerical types it is the obvious zero. For strings it is the
    empty string.

  - For >0D tensors, truthiness is determined by looking at the number of
    elements. If has zero elements, then the result is false. Otherwise the
    result is true.

  This matches the behavior of If and While for determining if a tensor counts
  as true/false for a branch condition.

  Args:
    input: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ToBool", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return to_bool_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ToBool", input=input, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ToBool", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ToBool = tf_export("raw_ops.ToBool")(_ops.to_raw_op(to_bool))


def to_bool_eager_fallback(input: Annotated[Any, TV_ToBool_T], name, ctx) -> Annotated[Any, _atypes.Bool]:
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"ToBool", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ToBool", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def _while(input, cond, body, output_shapes=[], parallel_iterations:int=10, name=None):
  r"""output = input; While (Cond(output)) { output = Body(output) }

  Args:
    input: A list of `Tensor` objects.
      A list of input tensors whose types are T.
    cond: A function decorated with @Defun.
            A function takes 'input' and returns a tensor.  If the tensor is
            a scalar of non-boolean, the scalar is converted to a boolean
            according to the following rule: if the scalar is a numerical
            value, non-zero means True and zero means False; if the scalar is
            a string, non-empty means True and empty means False. If the
            tensor is not a scalar, non-emptiness means True and False
            otherwise.
    body: A function decorated with @Defun.
            A function that takes a list of tensors and returns another
            list of tensors. Both lists have the same types as specified
            by T.
    output_shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
    parallel_iterations: An optional `int`. Defaults to `10`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "While", name, input, "cond", cond, "body", body,
        "output_shapes", output_shapes, "parallel_iterations",
        parallel_iterations)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return _while_eager_fallback(
          input, cond=cond, body=body, output_shapes=output_shapes,
          parallel_iterations=parallel_iterations, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if output_shapes is None:
    output_shapes = []
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'while' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if parallel_iterations is None:
    parallel_iterations = 10
  parallel_iterations = _execute.make_int(parallel_iterations, "parallel_iterations")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "While", input=input, cond=cond, body=body,
                 output_shapes=output_shapes,
                 parallel_iterations=parallel_iterations, name=name)
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("T", _op.get_attr("T"), "cond", _op.get_attr("cond"), "body",
              _op.get_attr("body"), "output_shapes",
              _op.get_attr("output_shapes"), "parallel_iterations",
              _op._get_attr_int("parallel_iterations"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "While", _inputs_flat, _attrs, _result)
  return _result

While = tf_export("raw_ops.While")(_ops.to_raw_op(_while))


def _while_eager_fallback(input, cond, body, output_shapes, parallel_iterations: int, name, ctx):
  if output_shapes is None:
    output_shapes = []
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'while' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if parallel_iterations is None:
    parallel_iterations = 10
  parallel_iterations = _execute.make_int(parallel_iterations, "parallel_iterations")
  _attr_T, input = _execute.convert_to_mixed_eager_tensors(input, ctx)
  _inputs_flat = list(input)
  _attrs = ("T", _attr_T, "cond", cond, "body", body, "output_shapes",
  output_shapes, "parallel_iterations", parallel_iterations)
  _result = _execute.execute(b"While", len(input), inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "While", _inputs_flat, _attrs, _result)
  return _result

