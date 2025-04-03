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
@tf_export('configure_and_initialize_global_tpu')
def configure_and_initialize_global_tpu(use_tfrt_host_runtime:bool=True, name=None) -> Annotated[Any, _atypes.Int32]:
  r"""TODO: add doc.

  Args:
    use_tfrt_host_runtime: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ConfigureAndInitializeGlobalTPU", name,
        "use_tfrt_host_runtime", use_tfrt_host_runtime)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_configure_and_initialize_global_tpu(
          (use_tfrt_host_runtime, name,), None)
      if _result is not NotImplemented:
        return _result
      return configure_and_initialize_global_tpu_eager_fallback(
          use_tfrt_host_runtime=use_tfrt_host_runtime, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            configure_and_initialize_global_tpu, (), dict(use_tfrt_host_runtime=use_tfrt_host_runtime,
                                                          name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_configure_and_initialize_global_tpu(
        (use_tfrt_host_runtime, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if use_tfrt_host_runtime is None:
    use_tfrt_host_runtime = True
  use_tfrt_host_runtime = _execute.make_bool(use_tfrt_host_runtime, "use_tfrt_host_runtime")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ConfigureAndInitializeGlobalTPU", use_tfrt_host_runtime=use_tfrt_host_runtime,
                                           name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          configure_and_initialize_global_tpu, (), dict(use_tfrt_host_runtime=use_tfrt_host_runtime,
                                                        name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("use_tfrt_host_runtime",
              _op._get_attr_bool("use_tfrt_host_runtime"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ConfigureAndInitializeGlobalTPU", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ConfigureAndInitializeGlobalTPU = tf_export("raw_ops.ConfigureAndInitializeGlobalTPU")(_ops.to_raw_op(configure_and_initialize_global_tpu))
_dispatcher_for_configure_and_initialize_global_tpu = configure_and_initialize_global_tpu._tf_type_based_dispatcher.Dispatch


def configure_and_initialize_global_tpu_eager_fallback(use_tfrt_host_runtime: bool, name, ctx) -> Annotated[Any, _atypes.Int32]:
  if use_tfrt_host_runtime is None:
    use_tfrt_host_runtime = True
  use_tfrt_host_runtime = _execute.make_bool(use_tfrt_host_runtime, "use_tfrt_host_runtime")
  _inputs_flat = []
  _attrs = ("use_tfrt_host_runtime", use_tfrt_host_runtime)
  _result = _execute.execute(b"ConfigureAndInitializeGlobalTPU", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ConfigureAndInitializeGlobalTPU", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_CopyToMesh_T = TypeVar("TV_CopyToMesh_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('copy_to_mesh')
def copy_to_mesh(input: Annotated[Any, TV_CopyToMesh_T], mesh: str, name=None) -> Annotated[Any, TV_CopyToMesh_T]:
  r"""TODO: add doc.

  Args:
    input: A `Tensor`.
    mesh: A `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CopyToMesh", name, input, "mesh", mesh)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_copy_to_mesh(
          (input, mesh, name,), None)
      if _result is not NotImplemented:
        return _result
      return copy_to_mesh_eager_fallback(
          input, mesh=mesh, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            copy_to_mesh, (), dict(input=input, mesh=mesh, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_copy_to_mesh(
        (input, mesh, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  mesh = _execute.make_str(mesh, "mesh")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CopyToMesh", input=input, mesh=mesh, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          copy_to_mesh, (), dict(input=input, mesh=mesh, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("mesh", _op.get_attr("mesh"), "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CopyToMesh", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CopyToMesh = tf_export("raw_ops.CopyToMesh")(_ops.to_raw_op(copy_to_mesh))
_dispatcher_for_copy_to_mesh = copy_to_mesh._tf_type_based_dispatcher.Dispatch


def copy_to_mesh_eager_fallback(input: Annotated[Any, TV_CopyToMesh_T], mesh: str, name, ctx) -> Annotated[Any, TV_CopyToMesh_T]:
  mesh = _execute.make_str(mesh, "mesh")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("mesh", mesh, "T", _attr_T)
  _result = _execute.execute(b"CopyToMesh", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CopyToMesh", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_CopyToMeshGrad_T = TypeVar("TV_CopyToMeshGrad_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('copy_to_mesh_grad')
def copy_to_mesh_grad(input: Annotated[Any, TV_CopyToMeshGrad_T], forward_input: Annotated[Any, TV_CopyToMeshGrad_T], name=None) -> Annotated[Any, TV_CopyToMeshGrad_T]:
  r"""TODO: add doc.

  Args:
    input: A `Tensor`.
    forward_input: A `Tensor`. Must have the same type as `input`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CopyToMeshGrad", name, input, forward_input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_copy_to_mesh_grad(
          (input, forward_input, name,), None)
      if _result is not NotImplemented:
        return _result
      return copy_to_mesh_grad_eager_fallback(
          input, forward_input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            copy_to_mesh_grad, (), dict(input=input,
                                        forward_input=forward_input,
                                        name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_copy_to_mesh_grad(
        (input, forward_input, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CopyToMeshGrad", input=input, forward_input=forward_input, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          copy_to_mesh_grad, (), dict(input=input,
                                      forward_input=forward_input, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CopyToMeshGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CopyToMeshGrad = tf_export("raw_ops.CopyToMeshGrad")(_ops.to_raw_op(copy_to_mesh_grad))
_dispatcher_for_copy_to_mesh_grad = copy_to_mesh_grad._tf_type_based_dispatcher.Dispatch


def copy_to_mesh_grad_eager_fallback(input: Annotated[Any, TV_CopyToMeshGrad_T], forward_input: Annotated[Any, TV_CopyToMeshGrad_T], name, ctx) -> Annotated[Any, TV_CopyToMeshGrad_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, forward_input], ctx, [])
  (input, forward_input) = _inputs_T
  _inputs_flat = [input, forward_input]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"CopyToMeshGrad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CopyToMeshGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('d_tensor_restore_v2')
def d_tensor_restore_v2(prefix: Annotated[Any, _atypes.String], tensor_names: Annotated[Any, _atypes.String], shape_and_slices: Annotated[Any, _atypes.String], input_shapes, input_layouts, dtypes, name=None):
  r"""TODO: add doc.

  Args:
    prefix: A `Tensor` of type `string`.
    tensor_names: A `Tensor` of type `string`.
    shape_and_slices: A `Tensor` of type `string`.
    input_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
    input_layouts: A list of `strings`.
    dtypes: A list of `tf.DTypes` that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `dtypes`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DTensorRestoreV2", name, prefix, tensor_names,
        shape_and_slices, "input_shapes", input_shapes, "input_layouts",
        input_layouts, "dtypes", dtypes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_d_tensor_restore_v2(
          (prefix, tensor_names, shape_and_slices, input_shapes,
          input_layouts, dtypes, name,), None)
      if _result is not NotImplemented:
        return _result
      return d_tensor_restore_v2_eager_fallback(
          prefix, tensor_names, shape_and_slices, input_shapes=input_shapes,
          input_layouts=input_layouts, dtypes=dtypes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            d_tensor_restore_v2, (), dict(prefix=prefix,
                                          tensor_names=tensor_names,
                                          shape_and_slices=shape_and_slices,
                                          input_shapes=input_shapes,
                                          input_layouts=input_layouts,
                                          dtypes=dtypes, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_d_tensor_restore_v2(
        (prefix, tensor_names, shape_and_slices, input_shapes, input_layouts,
        dtypes, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(input_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'input_shapes' argument to "
        "'d_tensor_restore_v2' Op, not %r." % input_shapes)
  input_shapes = [_execute.make_shape(_s, "input_shapes") for _s in input_shapes]
  if not isinstance(input_layouts, (list, tuple)):
    raise TypeError(
        "Expected list for 'input_layouts' argument to "
        "'d_tensor_restore_v2' Op, not %r." % input_layouts)
  input_layouts = [_execute.make_str(_s, "input_layouts") for _s in input_layouts]
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'d_tensor_restore_v2' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DTensorRestoreV2", prefix=prefix, tensor_names=tensor_names,
                            shape_and_slices=shape_and_slices,
                            input_shapes=input_shapes,
                            input_layouts=input_layouts, dtypes=dtypes,
                            name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          d_tensor_restore_v2, (), dict(prefix=prefix,
                                        tensor_names=tensor_names,
                                        shape_and_slices=shape_and_slices,
                                        input_shapes=input_shapes,
                                        input_layouts=input_layouts,
                                        dtypes=dtypes, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("input_shapes", _op.get_attr("input_shapes"), "input_layouts",
              _op.get_attr("input_layouts"), "dtypes", _op.get_attr("dtypes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DTensorRestoreV2", _inputs_flat, _attrs, _result)
  return _result

DTensorRestoreV2 = tf_export("raw_ops.DTensorRestoreV2")(_ops.to_raw_op(d_tensor_restore_v2))
_dispatcher_for_d_tensor_restore_v2 = d_tensor_restore_v2._tf_type_based_dispatcher.Dispatch


def d_tensor_restore_v2_eager_fallback(prefix: Annotated[Any, _atypes.String], tensor_names: Annotated[Any, _atypes.String], shape_and_slices: Annotated[Any, _atypes.String], input_shapes, input_layouts, dtypes, name, ctx):
  if not isinstance(input_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'input_shapes' argument to "
        "'d_tensor_restore_v2' Op, not %r." % input_shapes)
  input_shapes = [_execute.make_shape(_s, "input_shapes") for _s in input_shapes]
  if not isinstance(input_layouts, (list, tuple)):
    raise TypeError(
        "Expected list for 'input_layouts' argument to "
        "'d_tensor_restore_v2' Op, not %r." % input_layouts)
  input_layouts = [_execute.make_str(_s, "input_layouts") for _s in input_layouts]
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'d_tensor_restore_v2' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  prefix = _ops.convert_to_tensor(prefix, _dtypes.string)
  tensor_names = _ops.convert_to_tensor(tensor_names, _dtypes.string)
  shape_and_slices = _ops.convert_to_tensor(shape_and_slices, _dtypes.string)
  _inputs_flat = [prefix, tensor_names, shape_and_slices]
  _attrs = ("input_shapes", input_shapes, "input_layouts", input_layouts,
  "dtypes", dtypes)
  _result = _execute.execute(b"DTensorRestoreV2", len(dtypes),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DTensorRestoreV2", _inputs_flat, _attrs, _result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('d_tensor_set_global_tpu_array')
def d_tensor_set_global_tpu_array(topology: Annotated[Any, _atypes.String], name=None):
  r"""TODO: add doc.

  Args:
    topology: A `Tensor` of type `string`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DTensorSetGlobalTPUArray", name, topology)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_d_tensor_set_global_tpu_array(
          (topology, name,), None)
      if _result is not NotImplemented:
        return _result
      return d_tensor_set_global_tpu_array_eager_fallback(
          topology, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            d_tensor_set_global_tpu_array, (), dict(topology=topology,
                                                    name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_d_tensor_set_global_tpu_array(
        (topology, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DTensorSetGlobalTPUArray", topology=topology, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          d_tensor_set_global_tpu_array, (), dict(topology=topology,
                                                  name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
DTensorSetGlobalTPUArray = tf_export("raw_ops.DTensorSetGlobalTPUArray")(_ops.to_raw_op(d_tensor_set_global_tpu_array))
_dispatcher_for_d_tensor_set_global_tpu_array = d_tensor_set_global_tpu_array._tf_type_based_dispatcher.Dispatch


def d_tensor_set_global_tpu_array_eager_fallback(topology: Annotated[Any, _atypes.String], name, ctx):
  topology = _ops.convert_to_tensor(topology, _dtypes.string)
  _inputs_flat = [topology]
  _attrs = None
  _result = _execute.execute(b"DTensorSetGlobalTPUArray", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_Relayout_T = TypeVar("TV_Relayout_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('relayout')
def relayout(input: Annotated[Any, TV_Relayout_T], layout: str, name=None) -> Annotated[Any, TV_Relayout_T]:
  r"""TODO: add doc.

  Args:
    input: A `Tensor`.
    layout: A `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Relayout", name, input, "layout", layout)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_relayout(
          (input, layout, name,), None)
      if _result is not NotImplemented:
        return _result
      return relayout_eager_fallback(
          input, layout=layout, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            relayout, (), dict(input=input, layout=layout, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_relayout(
        (input, layout, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  layout = _execute.make_str(layout, "layout")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Relayout", input=input, layout=layout, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          relayout, (), dict(input=input, layout=layout, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("layout", _op.get_attr("layout"), "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Relayout", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Relayout = tf_export("raw_ops.Relayout")(_ops.to_raw_op(relayout))
_dispatcher_for_relayout = relayout._tf_type_based_dispatcher.Dispatch


def relayout_eager_fallback(input: Annotated[Any, TV_Relayout_T], layout: str, name, ctx) -> Annotated[Any, TV_Relayout_T]:
  layout = _execute.make_str(layout, "layout")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("layout", layout, "T", _attr_T)
  _result = _execute.execute(b"Relayout", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Relayout", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_RelayoutLike_T = TypeVar("TV_RelayoutLike_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_RelayoutLike_U = TypeVar("TV_RelayoutLike_U", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('relayout_like')
def relayout_like(input: Annotated[Any, TV_RelayoutLike_T], layout_input: Annotated[Any, TV_RelayoutLike_U], name=None) -> Annotated[Any, TV_RelayoutLike_T]:
  r"""TODO: add doc.

  Args:
    input: A `Tensor`.
    layout_input: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RelayoutLike", name, input, layout_input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_relayout_like(
          (input, layout_input, name,), None)
      if _result is not NotImplemented:
        return _result
      return relayout_like_eager_fallback(
          input, layout_input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            relayout_like, (), dict(input=input, layout_input=layout_input,
                                    name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_relayout_like(
        (input, layout_input, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RelayoutLike", input=input, layout_input=layout_input, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          relayout_like, (), dict(input=input, layout_input=layout_input,
                                  name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "U", _op._get_attr_type("U"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RelayoutLike", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RelayoutLike = tf_export("raw_ops.RelayoutLike")(_ops.to_raw_op(relayout_like))
_dispatcher_for_relayout_like = relayout_like._tf_type_based_dispatcher.Dispatch


def relayout_like_eager_fallback(input: Annotated[Any, TV_RelayoutLike_T], layout_input: Annotated[Any, TV_RelayoutLike_U], name, ctx) -> Annotated[Any, TV_RelayoutLike_T]:
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _attr_U, (layout_input,) = _execute.args_to_matching_eager([layout_input], ctx, [])
  _inputs_flat = [input, layout_input]
  _attrs = ("T", _attr_T, "U", _attr_U)
  _result = _execute.execute(b"RelayoutLike", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RelayoutLike", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('shutdown_tpu_system')
def shutdown_tpu_system(name=None) -> Annotated[Any, _atypes.Bool]:
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ShutdownTPUSystem", name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_shutdown_tpu_system(
          (name,), None)
      if _result is not NotImplemented:
        return _result
      return shutdown_tpu_system_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            shutdown_tpu_system, (), dict(name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_shutdown_tpu_system(
        (name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ShutdownTPUSystem", name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          shutdown_tpu_system, (), dict(name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ShutdownTPUSystem", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ShutdownTPUSystem = tf_export("raw_ops.ShutdownTPUSystem")(_ops.to_raw_op(shutdown_tpu_system))
_dispatcher_for_shutdown_tpu_system = shutdown_tpu_system._tf_type_based_dispatcher.Dispatch


def shutdown_tpu_system_eager_fallback(name, ctx) -> Annotated[Any, _atypes.Bool]:
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"ShutdownTPUSystem", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ShutdownTPUSystem", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

