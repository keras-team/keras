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

def anonymous_iterator(output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Resource]:
  r"""A container for an iterator resource.

  Args:
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AnonymousIterator", name, "output_types", output_types,
        "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return anonymous_iterator_eager_fallback(
          output_types=output_types, output_shapes=output_shapes, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'anonymous_iterator' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'anonymous_iterator' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AnonymousIterator", output_types=output_types,
                             output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AnonymousIterator", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AnonymousIterator = tf_export("raw_ops.AnonymousIterator")(_ops.to_raw_op(anonymous_iterator))


def anonymous_iterator_eager_fallback(output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Resource]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'anonymous_iterator' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'anonymous_iterator' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _inputs_flat = []
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"AnonymousIterator", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AnonymousIterator", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_AnonymousIteratorV2Output = collections.namedtuple(
    "AnonymousIteratorV2",
    ["handle", "deleter"])


def anonymous_iterator_v2(output_types, output_shapes, name=None):
  r"""A container for an iterator resource.

  Args:
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (handle, deleter).

    handle: A `Tensor` of type `resource`.
    deleter: A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AnonymousIteratorV2", name, "output_types", output_types,
        "output_shapes", output_shapes)
      _result = _AnonymousIteratorV2Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return anonymous_iterator_v2_eager_fallback(
          output_types=output_types, output_shapes=output_shapes, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'anonymous_iterator_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'anonymous_iterator_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AnonymousIteratorV2", output_types=output_types,
                               output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AnonymousIteratorV2", _inputs_flat, _attrs, _result)
  _result = _AnonymousIteratorV2Output._make(_result)
  return _result

AnonymousIteratorV2 = tf_export("raw_ops.AnonymousIteratorV2")(_ops.to_raw_op(anonymous_iterator_v2))


def anonymous_iterator_v2_eager_fallback(output_types, output_shapes, name, ctx):
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'anonymous_iterator_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'anonymous_iterator_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _inputs_flat = []
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"AnonymousIteratorV2", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AnonymousIteratorV2", _inputs_flat, _attrs, _result)
  _result = _AnonymousIteratorV2Output._make(_result)
  return _result


def anonymous_iterator_v3(output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Resource]:
  r"""A container for an iterator resource.

  Args:
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AnonymousIteratorV3", name, "output_types", output_types,
        "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return anonymous_iterator_v3_eager_fallback(
          output_types=output_types, output_shapes=output_shapes, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'anonymous_iterator_v3' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'anonymous_iterator_v3' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AnonymousIteratorV3", output_types=output_types,
                               output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AnonymousIteratorV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AnonymousIteratorV3 = tf_export("raw_ops.AnonymousIteratorV3")(_ops.to_raw_op(anonymous_iterator_v3))


def anonymous_iterator_v3_eager_fallback(output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Resource]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'anonymous_iterator_v3' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'anonymous_iterator_v3' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _inputs_flat = []
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"AnonymousIteratorV3", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AnonymousIteratorV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_AnonymousMemoryCacheOutput = collections.namedtuple(
    "AnonymousMemoryCache",
    ["handle", "deleter"])


def anonymous_memory_cache(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (handle, deleter).

    handle: A `Tensor` of type `resource`.
    deleter: A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AnonymousMemoryCache", name)
      _result = _AnonymousMemoryCacheOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return anonymous_memory_cache_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AnonymousMemoryCache", name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AnonymousMemoryCache", _inputs_flat, _attrs, _result)
  _result = _AnonymousMemoryCacheOutput._make(_result)
  return _result

AnonymousMemoryCache = tf_export("raw_ops.AnonymousMemoryCache")(_ops.to_raw_op(anonymous_memory_cache))


def anonymous_memory_cache_eager_fallback(name, ctx):
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"AnonymousMemoryCache", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AnonymousMemoryCache", _inputs_flat, _attrs, _result)
  _result = _AnonymousMemoryCacheOutput._make(_result)
  return _result

_AnonymousMultiDeviceIteratorOutput = collections.namedtuple(
    "AnonymousMultiDeviceIterator",
    ["handle", "deleter"])


def anonymous_multi_device_iterator(devices, output_types, output_shapes, name=None):
  r"""A container for a multi device iterator resource.

  Args:
    devices: A list of `strings` that has length `>= 1`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (handle, deleter).

    handle: A `Tensor` of type `resource`.
    deleter: A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AnonymousMultiDeviceIterator", name, "devices", devices,
        "output_types", output_types, "output_shapes", output_shapes)
      _result = _AnonymousMultiDeviceIteratorOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return anonymous_multi_device_iterator_eager_fallback(
          devices=devices, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(devices, (list, tuple)):
    raise TypeError(
        "Expected list for 'devices' argument to "
        "'anonymous_multi_device_iterator' Op, not %r." % devices)
  devices = [_execute.make_str(_s, "devices") for _s in devices]
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'anonymous_multi_device_iterator' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'anonymous_multi_device_iterator' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AnonymousMultiDeviceIterator", devices=devices,
                                        output_types=output_types,
                                        output_shapes=output_shapes,
                                        name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("devices", _op.get_attr("devices"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AnonymousMultiDeviceIterator", _inputs_flat, _attrs, _result)
  _result = _AnonymousMultiDeviceIteratorOutput._make(_result)
  return _result

AnonymousMultiDeviceIterator = tf_export("raw_ops.AnonymousMultiDeviceIterator")(_ops.to_raw_op(anonymous_multi_device_iterator))


def anonymous_multi_device_iterator_eager_fallback(devices, output_types, output_shapes, name, ctx):
  if not isinstance(devices, (list, tuple)):
    raise TypeError(
        "Expected list for 'devices' argument to "
        "'anonymous_multi_device_iterator' Op, not %r." % devices)
  devices = [_execute.make_str(_s, "devices") for _s in devices]
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'anonymous_multi_device_iterator' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'anonymous_multi_device_iterator' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _inputs_flat = []
  _attrs = ("devices", devices, "output_types", output_types, "output_shapes",
  output_shapes)
  _result = _execute.execute(b"AnonymousMultiDeviceIterator", 2,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AnonymousMultiDeviceIterator", _inputs_flat, _attrs, _result)
  _result = _AnonymousMultiDeviceIteratorOutput._make(_result)
  return _result


def anonymous_multi_device_iterator_v3(devices, output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Resource]:
  r"""A container for a multi device iterator resource.

  Args:
    devices: A list of `strings` that has length `>= 1`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AnonymousMultiDeviceIteratorV3", name, "devices", devices,
        "output_types", output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return anonymous_multi_device_iterator_v3_eager_fallback(
          devices=devices, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(devices, (list, tuple)):
    raise TypeError(
        "Expected list for 'devices' argument to "
        "'anonymous_multi_device_iterator_v3' Op, not %r." % devices)
  devices = [_execute.make_str(_s, "devices") for _s in devices]
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'anonymous_multi_device_iterator_v3' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'anonymous_multi_device_iterator_v3' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AnonymousMultiDeviceIteratorV3", devices=devices,
                                          output_types=output_types,
                                          output_shapes=output_shapes,
                                          name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("devices", _op.get_attr("devices"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AnonymousMultiDeviceIteratorV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AnonymousMultiDeviceIteratorV3 = tf_export("raw_ops.AnonymousMultiDeviceIteratorV3")(_ops.to_raw_op(anonymous_multi_device_iterator_v3))


def anonymous_multi_device_iterator_v3_eager_fallback(devices, output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Resource]:
  if not isinstance(devices, (list, tuple)):
    raise TypeError(
        "Expected list for 'devices' argument to "
        "'anonymous_multi_device_iterator_v3' Op, not %r." % devices)
  devices = [_execute.make_str(_s, "devices") for _s in devices]
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'anonymous_multi_device_iterator_v3' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'anonymous_multi_device_iterator_v3' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _inputs_flat = []
  _attrs = ("devices", devices, "output_types", output_types, "output_shapes",
  output_shapes)
  _result = _execute.execute(b"AnonymousMultiDeviceIteratorV3", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AnonymousMultiDeviceIteratorV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_AnonymousRandomSeedGeneratorOutput = collections.namedtuple(
    "AnonymousRandomSeedGenerator",
    ["handle", "deleter"])


def anonymous_random_seed_generator(seed: Annotated[Any, _atypes.Int64], seed2: Annotated[Any, _atypes.Int64], name=None):
  r"""TODO: add doc.

  Args:
    seed: A `Tensor` of type `int64`.
    seed2: A `Tensor` of type `int64`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (handle, deleter).

    handle: A `Tensor` of type `resource`.
    deleter: A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AnonymousRandomSeedGenerator", name, seed, seed2)
      _result = _AnonymousRandomSeedGeneratorOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return anonymous_random_seed_generator_eager_fallback(
          seed, seed2, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AnonymousRandomSeedGenerator", seed=seed, seed2=seed2, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AnonymousRandomSeedGenerator", _inputs_flat, _attrs, _result)
  _result = _AnonymousRandomSeedGeneratorOutput._make(_result)
  return _result

AnonymousRandomSeedGenerator = tf_export("raw_ops.AnonymousRandomSeedGenerator")(_ops.to_raw_op(anonymous_random_seed_generator))


def anonymous_random_seed_generator_eager_fallback(seed: Annotated[Any, _atypes.Int64], seed2: Annotated[Any, _atypes.Int64], name, ctx):
  seed = _ops.convert_to_tensor(seed, _dtypes.int64)
  seed2 = _ops.convert_to_tensor(seed2, _dtypes.int64)
  _inputs_flat = [seed, seed2]
  _attrs = None
  _result = _execute.execute(b"AnonymousRandomSeedGenerator", 2,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AnonymousRandomSeedGenerator", _inputs_flat, _attrs, _result)
  _result = _AnonymousRandomSeedGeneratorOutput._make(_result)
  return _result

_AnonymousSeedGeneratorOutput = collections.namedtuple(
    "AnonymousSeedGenerator",
    ["handle", "deleter"])


def anonymous_seed_generator(seed: Annotated[Any, _atypes.Int64], seed2: Annotated[Any, _atypes.Int64], reshuffle: Annotated[Any, _atypes.Bool], name=None):
  r"""TODO: add doc.

  Args:
    seed: A `Tensor` of type `int64`.
    seed2: A `Tensor` of type `int64`.
    reshuffle: A `Tensor` of type `bool`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (handle, deleter).

    handle: A `Tensor` of type `resource`.
    deleter: A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AnonymousSeedGenerator", name, seed, seed2, reshuffle)
      _result = _AnonymousSeedGeneratorOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return anonymous_seed_generator_eager_fallback(
          seed, seed2, reshuffle, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AnonymousSeedGenerator", seed=seed, seed2=seed2, reshuffle=reshuffle,
                                  name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AnonymousSeedGenerator", _inputs_flat, _attrs, _result)
  _result = _AnonymousSeedGeneratorOutput._make(_result)
  return _result

AnonymousSeedGenerator = tf_export("raw_ops.AnonymousSeedGenerator")(_ops.to_raw_op(anonymous_seed_generator))


def anonymous_seed_generator_eager_fallback(seed: Annotated[Any, _atypes.Int64], seed2: Annotated[Any, _atypes.Int64], reshuffle: Annotated[Any, _atypes.Bool], name, ctx):
  seed = _ops.convert_to_tensor(seed, _dtypes.int64)
  seed2 = _ops.convert_to_tensor(seed2, _dtypes.int64)
  reshuffle = _ops.convert_to_tensor(reshuffle, _dtypes.bool)
  _inputs_flat = [seed, seed2, reshuffle]
  _attrs = None
  _result = _execute.execute(b"AnonymousSeedGenerator", 2,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AnonymousSeedGenerator", _inputs_flat, _attrs, _result)
  _result = _AnonymousSeedGeneratorOutput._make(_result)
  return _result


def batch_dataset(input_dataset: Annotated[Any, _atypes.Variant], batch_size: Annotated[Any, _atypes.Int64], output_types, output_shapes, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that batches `batch_size` elements from `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    batch_size: A `Tensor` of type `int64`.
      A scalar representing the number of elements to accumulate in a
      batch.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BatchDataset", name, input_dataset, batch_size, "output_types",
        output_types, "output_shapes", output_shapes, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return batch_dataset_eager_fallback(
          input_dataset, batch_size, output_types=output_types,
          output_shapes=output_shapes, metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'batch_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'batch_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BatchDataset", input_dataset=input_dataset, batch_size=batch_size,
                        output_types=output_types,
                        output_shapes=output_shapes, metadata=metadata,
                        name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BatchDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BatchDataset = tf_export("raw_ops.BatchDataset")(_ops.to_raw_op(batch_dataset))


def batch_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], batch_size: Annotated[Any, _atypes.Int64], output_types, output_shapes, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'batch_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'batch_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  batch_size = _ops.convert_to_tensor(batch_size, _dtypes.int64)
  _inputs_flat = [input_dataset, batch_size]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes,
  "metadata", metadata)
  _result = _execute.execute(b"BatchDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BatchDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def batch_dataset_v2(input_dataset: Annotated[Any, _atypes.Variant], batch_size: Annotated[Any, _atypes.Int64], drop_remainder: Annotated[Any, _atypes.Bool], output_types, output_shapes, parallel_copy:bool=False, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that batches `batch_size` elements from `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    batch_size: A `Tensor` of type `int64`.
      A scalar representing the number of elements to accumulate in a batch.
    drop_remainder: A `Tensor` of type `bool`.
      A scalar representing whether the last batch should be dropped in case its size
      is smaller than desired.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    parallel_copy: An optional `bool`. Defaults to `False`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BatchDatasetV2", name, input_dataset, batch_size,
        drop_remainder, "parallel_copy", parallel_copy, "output_types",
        output_types, "output_shapes", output_shapes, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return batch_dataset_v2_eager_fallback(
          input_dataset, batch_size, drop_remainder,
          parallel_copy=parallel_copy, output_types=output_types,
          output_shapes=output_shapes, metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'batch_dataset_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'batch_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if parallel_copy is None:
    parallel_copy = False
  parallel_copy = _execute.make_bool(parallel_copy, "parallel_copy")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BatchDatasetV2", input_dataset=input_dataset, batch_size=batch_size,
                          drop_remainder=drop_remainder,
                          output_types=output_types,
                          output_shapes=output_shapes,
                          parallel_copy=parallel_copy, metadata=metadata,
                          name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("parallel_copy", _op._get_attr_bool("parallel_copy"),
              "output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BatchDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BatchDatasetV2 = tf_export("raw_ops.BatchDatasetV2")(_ops.to_raw_op(batch_dataset_v2))


def batch_dataset_v2_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], batch_size: Annotated[Any, _atypes.Int64], drop_remainder: Annotated[Any, _atypes.Bool], output_types, output_shapes, parallel_copy: bool, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'batch_dataset_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'batch_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if parallel_copy is None:
    parallel_copy = False
  parallel_copy = _execute.make_bool(parallel_copy, "parallel_copy")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  batch_size = _ops.convert_to_tensor(batch_size, _dtypes.int64)
  drop_remainder = _ops.convert_to_tensor(drop_remainder, _dtypes.bool)
  _inputs_flat = [input_dataset, batch_size, drop_remainder]
  _attrs = ("parallel_copy", parallel_copy, "output_types", output_types,
  "output_shapes", output_shapes, "metadata", metadata)
  _result = _execute.execute(b"BatchDatasetV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BatchDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def cache_dataset(input_dataset: Annotated[Any, _atypes.Variant], filename: Annotated[Any, _atypes.String], output_types, output_shapes, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that caches elements from `input_dataset`.

  A CacheDataset will iterate over the input_dataset, and store tensors. If the
  cache already exists, the cache will be used. If the cache is inappropriate
  (e.g. cannot be opened, contains tensors of the wrong shape / size), an error
  will the returned when used.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    filename: A `Tensor` of type `string`.
      A path on the filesystem where we should cache the dataset. Note: this
      will be a directory.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CacheDataset", name, input_dataset, filename, "output_types",
        output_types, "output_shapes", output_shapes, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return cache_dataset_eager_fallback(
          input_dataset, filename, output_types=output_types,
          output_shapes=output_shapes, metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'cache_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'cache_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CacheDataset", input_dataset=input_dataset, filename=filename,
                        output_types=output_types,
                        output_shapes=output_shapes, metadata=metadata,
                        name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CacheDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CacheDataset = tf_export("raw_ops.CacheDataset")(_ops.to_raw_op(cache_dataset))


def cache_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], filename: Annotated[Any, _atypes.String], output_types, output_shapes, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'cache_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'cache_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  filename = _ops.convert_to_tensor(filename, _dtypes.string)
  _inputs_flat = [input_dataset, filename]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes,
  "metadata", metadata)
  _result = _execute.execute(b"CacheDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CacheDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def cache_dataset_v2(input_dataset: Annotated[Any, _atypes.Variant], filename: Annotated[Any, _atypes.String], cache: Annotated[Any, _atypes.Resource], output_types, output_shapes, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    filename: A `Tensor` of type `string`.
    cache: A `Tensor` of type `resource`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CacheDatasetV2", name, input_dataset, filename, cache,
        "output_types", output_types, "output_shapes", output_shapes,
        "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return cache_dataset_v2_eager_fallback(
          input_dataset, filename, cache, output_types=output_types,
          output_shapes=output_shapes, metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'cache_dataset_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'cache_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CacheDatasetV2", input_dataset=input_dataset, filename=filename,
                          cache=cache, output_types=output_types,
                          output_shapes=output_shapes, metadata=metadata,
                          name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CacheDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CacheDatasetV2 = tf_export("raw_ops.CacheDatasetV2")(_ops.to_raw_op(cache_dataset_v2))


def cache_dataset_v2_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], filename: Annotated[Any, _atypes.String], cache: Annotated[Any, _atypes.Resource], output_types, output_shapes, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'cache_dataset_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'cache_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  filename = _ops.convert_to_tensor(filename, _dtypes.string)
  cache = _ops.convert_to_tensor(cache, _dtypes.resource)
  _inputs_flat = [input_dataset, filename, cache]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes,
  "metadata", metadata)
  _result = _execute.execute(b"CacheDatasetV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CacheDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def concatenate_dataset(input_dataset: Annotated[Any, _atypes.Variant], another_dataset: Annotated[Any, _atypes.Variant], output_types, output_shapes, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that concatenates `input_dataset` with `another_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    another_dataset: A `Tensor` of type `variant`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ConcatenateDataset", name, input_dataset, another_dataset,
        "output_types", output_types, "output_shapes", output_shapes,
        "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return concatenate_dataset_eager_fallback(
          input_dataset, another_dataset, output_types=output_types,
          output_shapes=output_shapes, metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'concatenate_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'concatenate_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ConcatenateDataset", input_dataset=input_dataset,
                              another_dataset=another_dataset,
                              output_types=output_types,
                              output_shapes=output_shapes, metadata=metadata,
                              name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ConcatenateDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ConcatenateDataset = tf_export("raw_ops.ConcatenateDataset")(_ops.to_raw_op(concatenate_dataset))


def concatenate_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], another_dataset: Annotated[Any, _atypes.Variant], output_types, output_shapes, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'concatenate_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'concatenate_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  another_dataset = _ops.convert_to_tensor(another_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset, another_dataset]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes,
  "metadata", metadata)
  _result = _execute.execute(b"ConcatenateDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ConcatenateDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def dataset_cardinality(input_dataset: Annotated[Any, _atypes.Variant], cardinality_options:str="", name=None) -> Annotated[Any, _atypes.Int64]:
  r"""Returns the cardinality of `input_dataset`.

  Returns the cardinality of `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the dataset to return cardinality for.
    cardinality_options: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DatasetCardinality", name, input_dataset,
        "cardinality_options", cardinality_options)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return dataset_cardinality_eager_fallback(
          input_dataset, cardinality_options=cardinality_options, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if cardinality_options is None:
    cardinality_options = ""
  cardinality_options = _execute.make_str(cardinality_options, "cardinality_options")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DatasetCardinality", input_dataset=input_dataset,
                              cardinality_options=cardinality_options,
                              name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("cardinality_options", _op.get_attr("cardinality_options"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DatasetCardinality", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DatasetCardinality = tf_export("raw_ops.DatasetCardinality")(_ops.to_raw_op(dataset_cardinality))


def dataset_cardinality_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], cardinality_options: str, name, ctx) -> Annotated[Any, _atypes.Int64]:
  if cardinality_options is None:
    cardinality_options = ""
  cardinality_options = _execute.make_str(cardinality_options, "cardinality_options")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset]
  _attrs = ("cardinality_options", cardinality_options)
  _result = _execute.execute(b"DatasetCardinality", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DatasetCardinality", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def dataset_fingerprint(input_dataset: Annotated[Any, _atypes.Variant], name=None) -> Annotated[Any, _atypes.UInt64]:
  r"""Returns the fingerprint of `input_dataset`.

  Returns the fingerprint of `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the dataset to return fingerprint for.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `uint64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DatasetFingerprint", name, input_dataset)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return dataset_fingerprint_eager_fallback(
          input_dataset, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DatasetFingerprint", input_dataset=input_dataset, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DatasetFingerprint", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DatasetFingerprint = tf_export("raw_ops.DatasetFingerprint")(_ops.to_raw_op(dataset_fingerprint))


def dataset_fingerprint_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], name, ctx) -> Annotated[Any, _atypes.UInt64]:
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset]
  _attrs = None
  _result = _execute.execute(b"DatasetFingerprint", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DatasetFingerprint", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def dataset_to_graph(input_dataset: Annotated[Any, _atypes.Variant], stateful_whitelist=[], allow_stateful:bool=False, strip_device_assignment:bool=False, name=None) -> Annotated[Any, _atypes.String]:
  r"""Returns a serialized GraphDef representing `input_dataset`.

  Returns a graph representation for `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the dataset to return the graph representation for.
    stateful_whitelist: An optional list of `strings`. Defaults to `[]`.
    allow_stateful: An optional `bool`. Defaults to `False`.
    strip_device_assignment: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DatasetToGraph", name, input_dataset, "stateful_whitelist",
        stateful_whitelist, "allow_stateful", allow_stateful,
        "strip_device_assignment", strip_device_assignment)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return dataset_to_graph_eager_fallback(
          input_dataset, stateful_whitelist=stateful_whitelist,
          allow_stateful=allow_stateful,
          strip_device_assignment=strip_device_assignment, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if stateful_whitelist is None:
    stateful_whitelist = []
  if not isinstance(stateful_whitelist, (list, tuple)):
    raise TypeError(
        "Expected list for 'stateful_whitelist' argument to "
        "'dataset_to_graph' Op, not %r." % stateful_whitelist)
  stateful_whitelist = [_execute.make_str(_s, "stateful_whitelist") for _s in stateful_whitelist]
  if allow_stateful is None:
    allow_stateful = False
  allow_stateful = _execute.make_bool(allow_stateful, "allow_stateful")
  if strip_device_assignment is None:
    strip_device_assignment = False
  strip_device_assignment = _execute.make_bool(strip_device_assignment, "strip_device_assignment")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DatasetToGraph", input_dataset=input_dataset,
                          stateful_whitelist=stateful_whitelist,
                          allow_stateful=allow_stateful,
                          strip_device_assignment=strip_device_assignment,
                          name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("stateful_whitelist", _op.get_attr("stateful_whitelist"),
              "allow_stateful", _op._get_attr_bool("allow_stateful"),
              "strip_device_assignment",
              _op._get_attr_bool("strip_device_assignment"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DatasetToGraph", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DatasetToGraph = tf_export("raw_ops.DatasetToGraph")(_ops.to_raw_op(dataset_to_graph))


def dataset_to_graph_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], stateful_whitelist, allow_stateful: bool, strip_device_assignment: bool, name, ctx) -> Annotated[Any, _atypes.String]:
  if stateful_whitelist is None:
    stateful_whitelist = []
  if not isinstance(stateful_whitelist, (list, tuple)):
    raise TypeError(
        "Expected list for 'stateful_whitelist' argument to "
        "'dataset_to_graph' Op, not %r." % stateful_whitelist)
  stateful_whitelist = [_execute.make_str(_s, "stateful_whitelist") for _s in stateful_whitelist]
  if allow_stateful is None:
    allow_stateful = False
  allow_stateful = _execute.make_bool(allow_stateful, "allow_stateful")
  if strip_device_assignment is None:
    strip_device_assignment = False
  strip_device_assignment = _execute.make_bool(strip_device_assignment, "strip_device_assignment")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset]
  _attrs = ("stateful_whitelist", stateful_whitelist, "allow_stateful",
  allow_stateful, "strip_device_assignment", strip_device_assignment)
  _result = _execute.execute(b"DatasetToGraph", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DatasetToGraph", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def dataset_to_graph_v2(input_dataset: Annotated[Any, _atypes.Variant], external_state_policy:int=0, strip_device_assignment:bool=False, name=None) -> Annotated[Any, _atypes.String]:
  r"""Returns a serialized GraphDef representing `input_dataset`.

  Returns a graph representation for `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the dataset to return the graph representation for.
    external_state_policy: An optional `int`. Defaults to `0`.
    strip_device_assignment: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DatasetToGraphV2", name, input_dataset,
        "external_state_policy", external_state_policy,
        "strip_device_assignment", strip_device_assignment)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return dataset_to_graph_v2_eager_fallback(
          input_dataset, external_state_policy=external_state_policy,
          strip_device_assignment=strip_device_assignment, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if external_state_policy is None:
    external_state_policy = 0
  external_state_policy = _execute.make_int(external_state_policy, "external_state_policy")
  if strip_device_assignment is None:
    strip_device_assignment = False
  strip_device_assignment = _execute.make_bool(strip_device_assignment, "strip_device_assignment")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DatasetToGraphV2", input_dataset=input_dataset,
                            external_state_policy=external_state_policy,
                            strip_device_assignment=strip_device_assignment,
                            name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("external_state_policy",
              _op._get_attr_int("external_state_policy"),
              "strip_device_assignment",
              _op._get_attr_bool("strip_device_assignment"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DatasetToGraphV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DatasetToGraphV2 = tf_export("raw_ops.DatasetToGraphV2")(_ops.to_raw_op(dataset_to_graph_v2))


def dataset_to_graph_v2_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], external_state_policy: int, strip_device_assignment: bool, name, ctx) -> Annotated[Any, _atypes.String]:
  if external_state_policy is None:
    external_state_policy = 0
  external_state_policy = _execute.make_int(external_state_policy, "external_state_policy")
  if strip_device_assignment is None:
    strip_device_assignment = False
  strip_device_assignment = _execute.make_bool(strip_device_assignment, "strip_device_assignment")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset]
  _attrs = ("external_state_policy", external_state_policy,
  "strip_device_assignment", strip_device_assignment)
  _result = _execute.execute(b"DatasetToGraphV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DatasetToGraphV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def dataset_to_single_element(dataset: Annotated[Any, _atypes.Variant], output_types, output_shapes, metadata:str="", name=None):
  r"""Outputs the single element from the given dataset.

  Args:
    dataset: A `Tensor` of type `variant`.
      A handle to a dataset that contains a single element.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `output_types`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DatasetToSingleElement", name, dataset, "output_types",
        output_types, "output_shapes", output_shapes, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return dataset_to_single_element_eager_fallback(
          dataset, output_types=output_types, output_shapes=output_shapes,
          metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'dataset_to_single_element' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'dataset_to_single_element' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DatasetToSingleElement", dataset=dataset, output_types=output_types,
                                  output_shapes=output_shapes,
                                  metadata=metadata, name=name)
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DatasetToSingleElement", _inputs_flat, _attrs, _result)
  return _result

DatasetToSingleElement = tf_export("raw_ops.DatasetToSingleElement")(_ops.to_raw_op(dataset_to_single_element))


def dataset_to_single_element_eager_fallback(dataset: Annotated[Any, _atypes.Variant], output_types, output_shapes, metadata: str, name, ctx):
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'dataset_to_single_element' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'dataset_to_single_element' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  dataset = _ops.convert_to_tensor(dataset, _dtypes.variant)
  _inputs_flat = [dataset]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes,
  "metadata", metadata)
  _result = _execute.execute(b"DatasetToSingleElement", len(output_types),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DatasetToSingleElement", _inputs_flat, _attrs, _result)
  return _result


def delete_iterator(handle: Annotated[Any, _atypes.Resource], deleter: Annotated[Any, _atypes.Variant], name=None):
  r"""A container for an iterator resource.

  Args:
    handle: A `Tensor` of type `resource`. A handle to the iterator to delete.
    deleter: A `Tensor` of type `variant`. A variant deleter.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DeleteIterator", name, handle, deleter)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return delete_iterator_eager_fallback(
          handle, deleter, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DeleteIterator", handle=handle, deleter=deleter, name=name)
  return _op
DeleteIterator = tf_export("raw_ops.DeleteIterator")(_ops.to_raw_op(delete_iterator))


def delete_iterator_eager_fallback(handle: Annotated[Any, _atypes.Resource], deleter: Annotated[Any, _atypes.Variant], name, ctx):
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  deleter = _ops.convert_to_tensor(deleter, _dtypes.variant)
  _inputs_flat = [handle, deleter]
  _attrs = None
  _result = _execute.execute(b"DeleteIterator", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def delete_memory_cache(handle: Annotated[Any, _atypes.Resource], deleter: Annotated[Any, _atypes.Variant], name=None):
  r"""TODO: add doc.

  Args:
    handle: A `Tensor` of type `resource`.
    deleter: A `Tensor` of type `variant`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DeleteMemoryCache", name, handle, deleter)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return delete_memory_cache_eager_fallback(
          handle, deleter, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DeleteMemoryCache", handle=handle, deleter=deleter, name=name)
  return _op
DeleteMemoryCache = tf_export("raw_ops.DeleteMemoryCache")(_ops.to_raw_op(delete_memory_cache))


def delete_memory_cache_eager_fallback(handle: Annotated[Any, _atypes.Resource], deleter: Annotated[Any, _atypes.Variant], name, ctx):
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  deleter = _ops.convert_to_tensor(deleter, _dtypes.variant)
  _inputs_flat = [handle, deleter]
  _attrs = None
  _result = _execute.execute(b"DeleteMemoryCache", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def delete_multi_device_iterator(multi_device_iterator: Annotated[Any, _atypes.Resource], iterators: Annotated[List[Any], _atypes.Resource], deleter: Annotated[Any, _atypes.Variant], name=None):
  r"""A container for an iterator resource.

  Args:
    multi_device_iterator: A `Tensor` of type `resource`.
      A handle to the multi device iterator to delete.
    iterators: A list of `Tensor` objects with type `resource`.
      A list of iterator handles (unused). This is added so that automatic control dependencies get added during function tracing that ensure this op runs after all the dependent iterators are deleted.
    deleter: A `Tensor` of type `variant`. A variant deleter.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DeleteMultiDeviceIterator", name, multi_device_iterator,
        iterators, deleter)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return delete_multi_device_iterator_eager_fallback(
          multi_device_iterator, iterators, deleter, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(iterators, (list, tuple)):
    raise TypeError(
        "Expected list for 'iterators' argument to "
        "'delete_multi_device_iterator' Op, not %r." % iterators)
  _attr_N = len(iterators)
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DeleteMultiDeviceIterator", multi_device_iterator=multi_device_iterator,
                                     iterators=iterators, deleter=deleter,
                                     name=name)
  return _op
DeleteMultiDeviceIterator = tf_export("raw_ops.DeleteMultiDeviceIterator")(_ops.to_raw_op(delete_multi_device_iterator))


def delete_multi_device_iterator_eager_fallback(multi_device_iterator: Annotated[Any, _atypes.Resource], iterators: Annotated[List[Any], _atypes.Resource], deleter: Annotated[Any, _atypes.Variant], name, ctx):
  if not isinstance(iterators, (list, tuple)):
    raise TypeError(
        "Expected list for 'iterators' argument to "
        "'delete_multi_device_iterator' Op, not %r." % iterators)
  _attr_N = len(iterators)
  multi_device_iterator = _ops.convert_to_tensor(multi_device_iterator, _dtypes.resource)
  iterators = _ops.convert_n_to_tensor(iterators, _dtypes.resource)
  deleter = _ops.convert_to_tensor(deleter, _dtypes.variant)
  _inputs_flat = [multi_device_iterator] + list(iterators) + [deleter]
  _attrs = ("N", _attr_N)
  _result = _execute.execute(b"DeleteMultiDeviceIterator", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def delete_random_seed_generator(handle: Annotated[Any, _atypes.Resource], deleter: Annotated[Any, _atypes.Variant], name=None):
  r"""TODO: add doc.

  Args:
    handle: A `Tensor` of type `resource`.
    deleter: A `Tensor` of type `variant`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DeleteRandomSeedGenerator", name, handle, deleter)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return delete_random_seed_generator_eager_fallback(
          handle, deleter, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DeleteRandomSeedGenerator", handle=handle, deleter=deleter,
                                     name=name)
  return _op
DeleteRandomSeedGenerator = tf_export("raw_ops.DeleteRandomSeedGenerator")(_ops.to_raw_op(delete_random_seed_generator))


def delete_random_seed_generator_eager_fallback(handle: Annotated[Any, _atypes.Resource], deleter: Annotated[Any, _atypes.Variant], name, ctx):
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  deleter = _ops.convert_to_tensor(deleter, _dtypes.variant)
  _inputs_flat = [handle, deleter]
  _attrs = None
  _result = _execute.execute(b"DeleteRandomSeedGenerator", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def delete_seed_generator(handle: Annotated[Any, _atypes.Resource], deleter: Annotated[Any, _atypes.Variant], name=None):
  r"""TODO: add doc.

  Args:
    handle: A `Tensor` of type `resource`.
    deleter: A `Tensor` of type `variant`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DeleteSeedGenerator", name, handle, deleter)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return delete_seed_generator_eager_fallback(
          handle, deleter, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DeleteSeedGenerator", handle=handle, deleter=deleter, name=name)
  return _op
DeleteSeedGenerator = tf_export("raw_ops.DeleteSeedGenerator")(_ops.to_raw_op(delete_seed_generator))


def delete_seed_generator_eager_fallback(handle: Annotated[Any, _atypes.Resource], deleter: Annotated[Any, _atypes.Variant], name, ctx):
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  deleter = _ops.convert_to_tensor(deleter, _dtypes.variant)
  _inputs_flat = [handle, deleter]
  _attrs = None
  _result = _execute.execute(b"DeleteSeedGenerator", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def deserialize_iterator(resource_handle: Annotated[Any, _atypes.Resource], serialized: Annotated[Any, _atypes.Variant], name=None):
  r"""Converts the given variant tensor to an iterator and stores it in the given resource.

  Args:
    resource_handle: A `Tensor` of type `resource`.
      A handle to an iterator resource.
    serialized: A `Tensor` of type `variant`.
      A variant tensor storing the state of the iterator contained in the
      resource.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DeserializeIterator", name, resource_handle, serialized)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return deserialize_iterator_eager_fallback(
          resource_handle, serialized, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DeserializeIterator", resource_handle=resource_handle,
                               serialized=serialized, name=name)
  return _op
DeserializeIterator = tf_export("raw_ops.DeserializeIterator")(_ops.to_raw_op(deserialize_iterator))


def deserialize_iterator_eager_fallback(resource_handle: Annotated[Any, _atypes.Resource], serialized: Annotated[Any, _atypes.Variant], name, ctx):
  resource_handle = _ops.convert_to_tensor(resource_handle, _dtypes.resource)
  serialized = _ops.convert_to_tensor(serialized, _dtypes.variant)
  _inputs_flat = [resource_handle, serialized]
  _attrs = None
  _result = _execute.execute(b"DeserializeIterator", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def dummy_memory_cache(name=None) -> Annotated[Any, _atypes.Resource]:
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DummyMemoryCache", name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return dummy_memory_cache_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DummyMemoryCache", name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DummyMemoryCache", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DummyMemoryCache = tf_export("raw_ops.DummyMemoryCache")(_ops.to_raw_op(dummy_memory_cache))


def dummy_memory_cache_eager_fallback(name, ctx) -> Annotated[Any, _atypes.Resource]:
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"DummyMemoryCache", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DummyMemoryCache", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def dummy_seed_generator(name=None) -> Annotated[Any, _atypes.Resource]:
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DummySeedGenerator", name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return dummy_seed_generator_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DummySeedGenerator", name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DummySeedGenerator", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DummySeedGenerator = tf_export("raw_ops.DummySeedGenerator")(_ops.to_raw_op(dummy_seed_generator))


def dummy_seed_generator_eager_fallback(name, ctx) -> Annotated[Any, _atypes.Resource]:
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"DummySeedGenerator", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DummySeedGenerator", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def filter_by_last_component_dataset(input_dataset: Annotated[Any, _atypes.Variant], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset containing elements of first component of `input_dataset` having true in the last component.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FilterByLastComponentDataset", name, input_dataset,
        "output_types", output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return filter_by_last_component_dataset_eager_fallback(
          input_dataset, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'filter_by_last_component_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'filter_by_last_component_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FilterByLastComponentDataset", input_dataset=input_dataset,
                                        output_types=output_types,
                                        output_shapes=output_shapes,
                                        name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FilterByLastComponentDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

FilterByLastComponentDataset = tf_export("raw_ops.FilterByLastComponentDataset")(_ops.to_raw_op(filter_by_last_component_dataset))


def filter_by_last_component_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'filter_by_last_component_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'filter_by_last_component_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"FilterByLastComponentDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FilterByLastComponentDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def filter_dataset(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, predicate, output_types, output_shapes, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset containing elements of `input_dataset` matching `predicate`.

  The `predicate` function must return a scalar boolean and accept the
  following arguments:

  * One tensor for each component of an element of `input_dataset`.
  * One tensor for each value in `other_arguments`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    other_arguments: A list of `Tensor` objects.
      A list of tensors, typically values that were captured when
      building a closure for `predicate`.
    predicate: A function decorated with @Defun.
      A function returning a scalar boolean.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FilterDataset", name, input_dataset, other_arguments,
        "predicate", predicate, "output_types", output_types, "output_shapes",
        output_shapes, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return filter_dataset_eager_fallback(
          input_dataset, other_arguments, predicate=predicate,
          output_types=output_types, output_shapes=output_shapes,
          metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'filter_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'filter_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FilterDataset", input_dataset=input_dataset,
                         other_arguments=other_arguments, predicate=predicate,
                         output_types=output_types,
                         output_shapes=output_shapes, metadata=metadata,
                         name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("predicate", _op.get_attr("predicate"), "Targuments",
              _op.get_attr("Targuments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FilterDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

FilterDataset = tf_export("raw_ops.FilterDataset")(_ops.to_raw_op(filter_dataset))


def filter_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, predicate, output_types, output_shapes, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'filter_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'filter_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset] + list(other_arguments)
  _attrs = ("predicate", predicate, "Targuments", _attr_Targuments,
  "output_types", output_types, "output_shapes", output_shapes, "metadata",
  metadata)
  _result = _execute.execute(b"FilterDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FilterDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def finalize_dataset(input_dataset: Annotated[Any, _atypes.Variant], output_types, output_shapes, has_captured_ref:bool=False, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset by applying `tf.data.Options` to `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the input dataset.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    has_captured_ref: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FinalizeDataset", name, input_dataset, "has_captured_ref",
        has_captured_ref, "output_types", output_types, "output_shapes",
        output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return finalize_dataset_eager_fallback(
          input_dataset, has_captured_ref=has_captured_ref,
          output_types=output_types, output_shapes=output_shapes, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'finalize_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'finalize_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if has_captured_ref is None:
    has_captured_ref = False
  has_captured_ref = _execute.make_bool(has_captured_ref, "has_captured_ref")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FinalizeDataset", input_dataset=input_dataset,
                           output_types=output_types,
                           output_shapes=output_shapes,
                           has_captured_ref=has_captured_ref, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("has_captured_ref", _op._get_attr_bool("has_captured_ref"),
              "output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FinalizeDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

FinalizeDataset = tf_export("raw_ops.FinalizeDataset")(_ops.to_raw_op(finalize_dataset))


def finalize_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], output_types, output_shapes, has_captured_ref: bool, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'finalize_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'finalize_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if has_captured_ref is None:
    has_captured_ref = False
  has_captured_ref = _execute.make_bool(has_captured_ref, "has_captured_ref")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset]
  _attrs = ("has_captured_ref", has_captured_ref, "output_types",
  output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"FinalizeDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FinalizeDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def fixed_length_record_dataset(filenames: Annotated[Any, _atypes.String], header_bytes: Annotated[Any, _atypes.Int64], record_bytes: Annotated[Any, _atypes.Int64], footer_bytes: Annotated[Any, _atypes.Int64], buffer_size: Annotated[Any, _atypes.Int64], metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that emits the records from one or more binary files.

  Args:
    filenames: A `Tensor` of type `string`.
      A scalar or a vector containing the name(s) of the file(s) to be
      read.
    header_bytes: A `Tensor` of type `int64`.
      A scalar representing the number of bytes to skip at the
      beginning of a file.
    record_bytes: A `Tensor` of type `int64`.
      A scalar representing the number of bytes in each record.
    footer_bytes: A `Tensor` of type `int64`.
      A scalar representing the number of bytes to skip at the end
      of a file.
    buffer_size: A `Tensor` of type `int64`.
      A scalar representing the number of bytes to buffer. Must be > 0.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FixedLengthRecordDataset", name, filenames, header_bytes,
        record_bytes, footer_bytes, buffer_size, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return fixed_length_record_dataset_eager_fallback(
          filenames, header_bytes, record_bytes, footer_bytes, buffer_size,
          metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FixedLengthRecordDataset", filenames=filenames,
                                    header_bytes=header_bytes,
                                    record_bytes=record_bytes,
                                    footer_bytes=footer_bytes,
                                    buffer_size=buffer_size,
                                    metadata=metadata, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("metadata", _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FixedLengthRecordDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

FixedLengthRecordDataset = tf_export("raw_ops.FixedLengthRecordDataset")(_ops.to_raw_op(fixed_length_record_dataset))


def fixed_length_record_dataset_eager_fallback(filenames: Annotated[Any, _atypes.String], header_bytes: Annotated[Any, _atypes.Int64], record_bytes: Annotated[Any, _atypes.Int64], footer_bytes: Annotated[Any, _atypes.Int64], buffer_size: Annotated[Any, _atypes.Int64], metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  filenames = _ops.convert_to_tensor(filenames, _dtypes.string)
  header_bytes = _ops.convert_to_tensor(header_bytes, _dtypes.int64)
  record_bytes = _ops.convert_to_tensor(record_bytes, _dtypes.int64)
  footer_bytes = _ops.convert_to_tensor(footer_bytes, _dtypes.int64)
  buffer_size = _ops.convert_to_tensor(buffer_size, _dtypes.int64)
  _inputs_flat = [filenames, header_bytes, record_bytes, footer_bytes, buffer_size]
  _attrs = ("metadata", metadata)
  _result = _execute.execute(b"FixedLengthRecordDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FixedLengthRecordDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def fixed_length_record_dataset_v2(filenames: Annotated[Any, _atypes.String], header_bytes: Annotated[Any, _atypes.Int64], record_bytes: Annotated[Any, _atypes.Int64], footer_bytes: Annotated[Any, _atypes.Int64], buffer_size: Annotated[Any, _atypes.Int64], compression_type: Annotated[Any, _atypes.String], metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    filenames: A `Tensor` of type `string`.
    header_bytes: A `Tensor` of type `int64`.
    record_bytes: A `Tensor` of type `int64`.
    footer_bytes: A `Tensor` of type `int64`.
    buffer_size: A `Tensor` of type `int64`.
    compression_type: A `Tensor` of type `string`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FixedLengthRecordDatasetV2", name, filenames, header_bytes,
        record_bytes, footer_bytes, buffer_size, compression_type, "metadata",
        metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return fixed_length_record_dataset_v2_eager_fallback(
          filenames, header_bytes, record_bytes, footer_bytes, buffer_size,
          compression_type, metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FixedLengthRecordDatasetV2", filenames=filenames,
                                      header_bytes=header_bytes,
                                      record_bytes=record_bytes,
                                      footer_bytes=footer_bytes,
                                      buffer_size=buffer_size,
                                      compression_type=compression_type,
                                      metadata=metadata, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("metadata", _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FixedLengthRecordDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

FixedLengthRecordDatasetV2 = tf_export("raw_ops.FixedLengthRecordDatasetV2")(_ops.to_raw_op(fixed_length_record_dataset_v2))


def fixed_length_record_dataset_v2_eager_fallback(filenames: Annotated[Any, _atypes.String], header_bytes: Annotated[Any, _atypes.Int64], record_bytes: Annotated[Any, _atypes.Int64], footer_bytes: Annotated[Any, _atypes.Int64], buffer_size: Annotated[Any, _atypes.Int64], compression_type: Annotated[Any, _atypes.String], metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  filenames = _ops.convert_to_tensor(filenames, _dtypes.string)
  header_bytes = _ops.convert_to_tensor(header_bytes, _dtypes.int64)
  record_bytes = _ops.convert_to_tensor(record_bytes, _dtypes.int64)
  footer_bytes = _ops.convert_to_tensor(footer_bytes, _dtypes.int64)
  buffer_size = _ops.convert_to_tensor(buffer_size, _dtypes.int64)
  compression_type = _ops.convert_to_tensor(compression_type, _dtypes.string)
  _inputs_flat = [filenames, header_bytes, record_bytes, footer_bytes, buffer_size, compression_type]
  _attrs = ("metadata", metadata)
  _result = _execute.execute(b"FixedLengthRecordDatasetV2", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FixedLengthRecordDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def flat_map_dataset(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, f, output_types, output_shapes, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that applies `f` to the outputs of `input_dataset`.

  Unlike MapDataset, the `f` in FlatMapDataset is expected to return a
  Dataset variant, and FlatMapDataset will flatten successive results
  into a single Dataset.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    other_arguments: A list of `Tensor` objects.
    f: A function decorated with @Defun.
      A function mapping elements of `input_dataset`, concatenated with
      `other_arguments`, to a Dataset variant that contains elements matching
      `output_types` and `output_shapes`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FlatMapDataset", name, input_dataset, other_arguments, "f", f,
        "output_types", output_types, "output_shapes", output_shapes,
        "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return flat_map_dataset_eager_fallback(
          input_dataset, other_arguments, f=f, output_types=output_types,
          output_shapes=output_shapes, metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'flat_map_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'flat_map_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FlatMapDataset", input_dataset=input_dataset,
                          other_arguments=other_arguments, f=f,
                          output_types=output_types,
                          output_shapes=output_shapes, metadata=metadata,
                          name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("f", _op.get_attr("f"), "Targuments",
              _op.get_attr("Targuments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FlatMapDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

FlatMapDataset = tf_export("raw_ops.FlatMapDataset")(_ops.to_raw_op(flat_map_dataset))


def flat_map_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, f, output_types, output_shapes, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'flat_map_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'flat_map_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset] + list(other_arguments)
  _attrs = ("f", f, "Targuments", _attr_Targuments, "output_types",
  output_types, "output_shapes", output_shapes, "metadata", metadata)
  _result = _execute.execute(b"FlatMapDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FlatMapDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def generator_dataset(init_func_other_args, next_func_other_args, finalize_func_other_args, init_func, next_func, finalize_func, output_types, output_shapes, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that invokes a function to generate elements.

  Args:
    init_func_other_args: A list of `Tensor` objects.
    next_func_other_args: A list of `Tensor` objects.
    finalize_func_other_args: A list of `Tensor` objects.
    init_func: A function decorated with @Defun.
    next_func: A function decorated with @Defun.
    finalize_func: A function decorated with @Defun.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "GeneratorDataset", name, init_func_other_args,
        next_func_other_args, finalize_func_other_args, "init_func",
        init_func, "next_func", next_func, "finalize_func", finalize_func,
        "output_types", output_types, "output_shapes", output_shapes,
        "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return generator_dataset_eager_fallback(
          init_func_other_args, next_func_other_args,
          finalize_func_other_args, init_func=init_func, next_func=next_func,
          finalize_func=finalize_func, output_types=output_types,
          output_shapes=output_shapes, metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'generator_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'generator_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "GeneratorDataset", init_func_other_args=init_func_other_args,
                            next_func_other_args=next_func_other_args,
                            finalize_func_other_args=finalize_func_other_args,
                            init_func=init_func, next_func=next_func,
                            finalize_func=finalize_func,
                            output_types=output_types,
                            output_shapes=output_shapes, metadata=metadata,
                            name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("init_func", _op.get_attr("init_func"), "next_func",
              _op.get_attr("next_func"), "finalize_func",
              _op.get_attr("finalize_func"), "Tinit_func_args",
              _op.get_attr("Tinit_func_args"), "Tnext_func_args",
              _op.get_attr("Tnext_func_args"), "Tfinalize_func_args",
              _op.get_attr("Tfinalize_func_args"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "GeneratorDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

GeneratorDataset = tf_export("raw_ops.GeneratorDataset")(_ops.to_raw_op(generator_dataset))


def generator_dataset_eager_fallback(init_func_other_args, next_func_other_args, finalize_func_other_args, init_func, next_func, finalize_func, output_types, output_shapes, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'generator_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'generator_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _attr_Tinit_func_args, init_func_other_args = _execute.convert_to_mixed_eager_tensors(init_func_other_args, ctx)
  _attr_Tnext_func_args, next_func_other_args = _execute.convert_to_mixed_eager_tensors(next_func_other_args, ctx)
  _attr_Tfinalize_func_args, finalize_func_other_args = _execute.convert_to_mixed_eager_tensors(finalize_func_other_args, ctx)
  _inputs_flat = list(init_func_other_args) + list(next_func_other_args) + list(finalize_func_other_args)
  _attrs = ("init_func", init_func, "next_func", next_func, "finalize_func",
  finalize_func, "Tinit_func_args", _attr_Tinit_func_args, "Tnext_func_args",
  _attr_Tnext_func_args, "Tfinalize_func_args", _attr_Tfinalize_func_args,
  "output_types", output_types, "output_shapes", output_shapes, "metadata",
  metadata)
  _result = _execute.execute(b"GeneratorDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "GeneratorDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def get_options(input_dataset: Annotated[Any, _atypes.Variant], name=None) -> Annotated[Any, _atypes.String]:
  r"""Returns the `tf.data.Options` attached to `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the input dataset.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "GetOptions", name, input_dataset)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return get_options_eager_fallback(
          input_dataset, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "GetOptions", input_dataset=input_dataset, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "GetOptions", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

GetOptions = tf_export("raw_ops.GetOptions")(_ops.to_raw_op(get_options))


def get_options_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], name, ctx) -> Annotated[Any, _atypes.String]:
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset]
  _attrs = None
  _result = _execute.execute(b"GetOptions", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "GetOptions", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def interleave_dataset(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, cycle_length: Annotated[Any, _atypes.Int64], block_length: Annotated[Any, _atypes.Int64], f, output_types, output_shapes, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that applies `f` to the outputs of `input_dataset`.

  Unlike MapDataset, the `f` in InterleaveDataset is expected to return
  a Dataset variant, and InterleaveDataset will flatten successive
  results into a single Dataset. Unlike FlatMapDataset,
  InterleaveDataset will interleave sequences of up to `block_length`
  consecutive elements from `cycle_length` input elements.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    other_arguments: A list of `Tensor` objects.
    cycle_length: A `Tensor` of type `int64`.
    block_length: A `Tensor` of type `int64`.
    f: A function decorated with @Defun.
      A function mapping elements of `input_dataset`, concatenated with
      `other_arguments`, to a Dataset variant that contains elements matching
      `output_types` and `output_shapes`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "InterleaveDataset", name, input_dataset, other_arguments,
        cycle_length, block_length, "f", f, "output_types", output_types,
        "output_shapes", output_shapes, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return interleave_dataset_eager_fallback(
          input_dataset, other_arguments, cycle_length, block_length, f=f,
          output_types=output_types, output_shapes=output_shapes,
          metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'interleave_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'interleave_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "InterleaveDataset", input_dataset=input_dataset,
                             other_arguments=other_arguments,
                             cycle_length=cycle_length,
                             block_length=block_length, f=f,
                             output_types=output_types,
                             output_shapes=output_shapes, metadata=metadata,
                             name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("f", _op.get_attr("f"), "Targuments",
              _op.get_attr("Targuments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "InterleaveDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

InterleaveDataset = tf_export("raw_ops.InterleaveDataset")(_ops.to_raw_op(interleave_dataset))


def interleave_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, cycle_length: Annotated[Any, _atypes.Int64], block_length: Annotated[Any, _atypes.Int64], f, output_types, output_shapes, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'interleave_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'interleave_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  cycle_length = _ops.convert_to_tensor(cycle_length, _dtypes.int64)
  block_length = _ops.convert_to_tensor(block_length, _dtypes.int64)
  _inputs_flat = [input_dataset] + list(other_arguments) + [cycle_length, block_length]
  _attrs = ("f", f, "Targuments", _attr_Targuments, "output_types",
  output_types, "output_shapes", output_shapes, "metadata", metadata)
  _result = _execute.execute(b"InterleaveDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "InterleaveDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def iterator(shared_name: str, container: str, output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Resource]:
  r"""A container for an iterator resource.

  Args:
    shared_name: A `string`.
    container: A `string`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Iterator", name, "shared_name", shared_name, "container",
        container, "output_types", output_types, "output_shapes",
        output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return iterator_eager_fallback(
          shared_name=shared_name, container=container,
          output_types=output_types, output_shapes=output_shapes, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  shared_name = _execute.make_str(shared_name, "shared_name")
  container = _execute.make_str(container, "container")
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'iterator' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'iterator' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Iterator", shared_name=shared_name, container=container,
                    output_types=output_types, output_shapes=output_shapes,
                    name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("shared_name", _op.get_attr("shared_name"), "container",
              _op.get_attr("container"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Iterator", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Iterator = tf_export("raw_ops.Iterator")(_ops.to_raw_op(iterator))


def iterator_eager_fallback(shared_name: str, container: str, output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Resource]:
  shared_name = _execute.make_str(shared_name, "shared_name")
  container = _execute.make_str(container, "container")
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'iterator' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'iterator' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _inputs_flat = []
  _attrs = ("shared_name", shared_name, "container", container,
  "output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"Iterator", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Iterator", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def iterator_from_string_handle(string_handle: Annotated[Any, _atypes.String], output_types=[], output_shapes=[], name=None) -> Annotated[Any, _atypes.Resource]:
  r"""Converts the given string representing a handle to an iterator to a resource.

  Args:
    string_handle: A `Tensor` of type `string`.
      A string representation of the given handle.
    output_types: An optional list of `tf.DTypes`. Defaults to `[]`.
      If specified, defines the type of each tuple component in an
      element produced by the resulting iterator.
    output_shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
      If specified, defines the shape of each tuple component in an
      element produced by the resulting iterator.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IteratorFromStringHandle", name, string_handle, "output_types",
        output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return iterator_from_string_handle_eager_fallback(
          string_handle, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if output_types is None:
    output_types = []
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'iterator_from_string_handle' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if output_shapes is None:
    output_shapes = []
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'iterator_from_string_handle' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IteratorFromStringHandle", string_handle=string_handle,
                                    output_types=output_types,
                                    output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IteratorFromStringHandle", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IteratorFromStringHandle = tf_export("raw_ops.IteratorFromStringHandle")(_ops.to_raw_op(iterator_from_string_handle))


def iterator_from_string_handle_eager_fallback(string_handle: Annotated[Any, _atypes.String], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Resource]:
  if output_types is None:
    output_types = []
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'iterator_from_string_handle' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if output_shapes is None:
    output_shapes = []
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'iterator_from_string_handle' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  string_handle = _ops.convert_to_tensor(string_handle, _dtypes.string)
  _inputs_flat = [string_handle]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"IteratorFromStringHandle", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IteratorFromStringHandle", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def iterator_from_string_handle_v2(string_handle: Annotated[Any, _atypes.String], output_types=[], output_shapes=[], name=None) -> Annotated[Any, _atypes.Resource]:
  r"""TODO: add doc.

  Args:
    string_handle: A `Tensor` of type `string`.
    output_types: An optional list of `tf.DTypes`. Defaults to `[]`.
    output_shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IteratorFromStringHandleV2", name, string_handle,
        "output_types", output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return iterator_from_string_handle_v2_eager_fallback(
          string_handle, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if output_types is None:
    output_types = []
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'iterator_from_string_handle_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if output_shapes is None:
    output_shapes = []
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'iterator_from_string_handle_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IteratorFromStringHandleV2", string_handle=string_handle,
                                      output_types=output_types,
                                      output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IteratorFromStringHandleV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IteratorFromStringHandleV2 = tf_export("raw_ops.IteratorFromStringHandleV2")(_ops.to_raw_op(iterator_from_string_handle_v2))


def iterator_from_string_handle_v2_eager_fallback(string_handle: Annotated[Any, _atypes.String], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Resource]:
  if output_types is None:
    output_types = []
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'iterator_from_string_handle_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if output_shapes is None:
    output_shapes = []
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'iterator_from_string_handle_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  string_handle = _ops.convert_to_tensor(string_handle, _dtypes.string)
  _inputs_flat = [string_handle]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"IteratorFromStringHandleV2", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IteratorFromStringHandleV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def iterator_get_next(iterator: Annotated[Any, _atypes.Resource], output_types, output_shapes, name=None):
  r"""Gets the next output from the given iterator .

  Args:
    iterator: A `Tensor` of type `resource`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `output_types`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IteratorGetNext", name, iterator, "output_types", output_types,
        "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return iterator_get_next_eager_fallback(
          iterator, output_types=output_types, output_shapes=output_shapes,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'iterator_get_next' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'iterator_get_next' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IteratorGetNext", iterator=iterator, output_types=output_types,
                           output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IteratorGetNext", _inputs_flat, _attrs, _result)
  return _result

IteratorGetNext = tf_export("raw_ops.IteratorGetNext")(_ops.to_raw_op(iterator_get_next))


def iterator_get_next_eager_fallback(iterator: Annotated[Any, _atypes.Resource], output_types, output_shapes, name, ctx):
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'iterator_get_next' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'iterator_get_next' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  iterator = _ops.convert_to_tensor(iterator, _dtypes.resource)
  _inputs_flat = [iterator]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"IteratorGetNext", len(output_types),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IteratorGetNext", _inputs_flat, _attrs, _result)
  return _result


def iterator_get_next_as_optional(iterator: Annotated[Any, _atypes.Resource], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Gets the next output from the given iterator as an Optional variant.

  Args:
    iterator: A `Tensor` of type `resource`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IteratorGetNextAsOptional", name, iterator, "output_types",
        output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return iterator_get_next_as_optional_eager_fallback(
          iterator, output_types=output_types, output_shapes=output_shapes,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'iterator_get_next_as_optional' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'iterator_get_next_as_optional' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IteratorGetNextAsOptional", iterator=iterator,
                                     output_types=output_types,
                                     output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IteratorGetNextAsOptional", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IteratorGetNextAsOptional = tf_export("raw_ops.IteratorGetNextAsOptional")(_ops.to_raw_op(iterator_get_next_as_optional))


def iterator_get_next_as_optional_eager_fallback(iterator: Annotated[Any, _atypes.Resource], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'iterator_get_next_as_optional' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'iterator_get_next_as_optional' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  iterator = _ops.convert_to_tensor(iterator, _dtypes.resource)
  _inputs_flat = [iterator]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"IteratorGetNextAsOptional", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IteratorGetNextAsOptional", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def iterator_get_next_sync(iterator: Annotated[Any, _atypes.Resource], output_types, output_shapes, name=None):
  r"""Gets the next output from the given iterator.

  This operation is a synchronous version IteratorGetNext. It should only be used
  in situations where the iterator does not block the calling thread, or where
  the calling thread is not a member of the thread pool used to execute parallel
  operations (e.g. in eager mode).

  Args:
    iterator: A `Tensor` of type `resource`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `output_types`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IteratorGetNextSync", name, iterator, "output_types",
        output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return iterator_get_next_sync_eager_fallback(
          iterator, output_types=output_types, output_shapes=output_shapes,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'iterator_get_next_sync' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'iterator_get_next_sync' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IteratorGetNextSync", iterator=iterator, output_types=output_types,
                               output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IteratorGetNextSync", _inputs_flat, _attrs, _result)
  return _result

IteratorGetNextSync = tf_export("raw_ops.IteratorGetNextSync")(_ops.to_raw_op(iterator_get_next_sync))


def iterator_get_next_sync_eager_fallback(iterator: Annotated[Any, _atypes.Resource], output_types, output_shapes, name, ctx):
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'iterator_get_next_sync' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'iterator_get_next_sync' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  iterator = _ops.convert_to_tensor(iterator, _dtypes.resource)
  _inputs_flat = [iterator]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"IteratorGetNextSync", len(output_types),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IteratorGetNextSync", _inputs_flat, _attrs, _result)
  return _result


def iterator_to_string_handle(resource_handle: Annotated[Any, _atypes.Resource], name=None) -> Annotated[Any, _atypes.String]:
  r"""Converts the given `resource_handle` representing an iterator to a string.

  Args:
    resource_handle: A `Tensor` of type `resource`.
      A handle to an iterator resource.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IteratorToStringHandle", name, resource_handle)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return iterator_to_string_handle_eager_fallback(
          resource_handle, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IteratorToStringHandle", resource_handle=resource_handle, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IteratorToStringHandle", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IteratorToStringHandle = tf_export("raw_ops.IteratorToStringHandle")(_ops.to_raw_op(iterator_to_string_handle))


def iterator_to_string_handle_eager_fallback(resource_handle: Annotated[Any, _atypes.Resource], name, ctx) -> Annotated[Any, _atypes.String]:
  resource_handle = _ops.convert_to_tensor(resource_handle, _dtypes.resource)
  _inputs_flat = [resource_handle]
  _attrs = None
  _result = _execute.execute(b"IteratorToStringHandle", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IteratorToStringHandle", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def iterator_v2(shared_name: str, container: str, output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Resource]:
  r"""TODO: add doc.

  Args:
    shared_name: A `string`.
    container: A `string`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IteratorV2", name, "shared_name", shared_name, "container",
        container, "output_types", output_types, "output_shapes",
        output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return iterator_v2_eager_fallback(
          shared_name=shared_name, container=container,
          output_types=output_types, output_shapes=output_shapes, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  shared_name = _execute.make_str(shared_name, "shared_name")
  container = _execute.make_str(container, "container")
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'iterator_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'iterator_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IteratorV2", shared_name=shared_name, container=container,
                      output_types=output_types, output_shapes=output_shapes,
                      name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("shared_name", _op.get_attr("shared_name"), "container",
              _op.get_attr("container"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IteratorV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IteratorV2 = tf_export("raw_ops.IteratorV2")(_ops.to_raw_op(iterator_v2))


def iterator_v2_eager_fallback(shared_name: str, container: str, output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Resource]:
  shared_name = _execute.make_str(shared_name, "shared_name")
  container = _execute.make_str(container, "container")
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'iterator_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'iterator_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _inputs_flat = []
  _attrs = ("shared_name", shared_name, "container", container,
  "output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"IteratorV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IteratorV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def make_iterator(dataset: Annotated[Any, _atypes.Variant], iterator: Annotated[Any, _atypes.Resource], name=None):
  r"""Makes a new iterator from the given `dataset` and stores it in `iterator`.

  This operation may be executed multiple times. Each execution will reset the
  iterator in `iterator` to the first element of `dataset`.

  Args:
    dataset: A `Tensor` of type `variant`.
    iterator: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MakeIterator", name, dataset, iterator)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return make_iterator_eager_fallback(
          dataset, iterator, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MakeIterator", dataset=dataset, iterator=iterator, name=name)
  return _op
MakeIterator = tf_export("raw_ops.MakeIterator")(_ops.to_raw_op(make_iterator))


def make_iterator_eager_fallback(dataset: Annotated[Any, _atypes.Variant], iterator: Annotated[Any, _atypes.Resource], name, ctx):
  dataset = _ops.convert_to_tensor(dataset, _dtypes.variant)
  iterator = _ops.convert_to_tensor(iterator, _dtypes.resource)
  _inputs_flat = [dataset, iterator]
  _attrs = None
  _result = _execute.execute(b"MakeIterator", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def map_dataset(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, f, output_types, output_shapes, use_inter_op_parallelism:bool=True, preserve_cardinality:bool=False, force_synchronous:bool=False, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that applies `f` to the outputs of `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    other_arguments: A list of `Tensor` objects.
    f: A function decorated with @Defun.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    use_inter_op_parallelism: An optional `bool`. Defaults to `True`.
    preserve_cardinality: An optional `bool`. Defaults to `False`.
    force_synchronous: An optional `bool`. Defaults to `False`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MapDataset", name, input_dataset, other_arguments, "f", f,
        "output_types", output_types, "output_shapes", output_shapes,
        "use_inter_op_parallelism", use_inter_op_parallelism,
        "preserve_cardinality", preserve_cardinality, "force_synchronous",
        force_synchronous, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return map_dataset_eager_fallback(
          input_dataset, other_arguments, f=f, output_types=output_types,
          output_shapes=output_shapes,
          use_inter_op_parallelism=use_inter_op_parallelism,
          preserve_cardinality=preserve_cardinality,
          force_synchronous=force_synchronous, metadata=metadata, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'map_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'map_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if use_inter_op_parallelism is None:
    use_inter_op_parallelism = True
  use_inter_op_parallelism = _execute.make_bool(use_inter_op_parallelism, "use_inter_op_parallelism")
  if preserve_cardinality is None:
    preserve_cardinality = False
  preserve_cardinality = _execute.make_bool(preserve_cardinality, "preserve_cardinality")
  if force_synchronous is None:
    force_synchronous = False
  force_synchronous = _execute.make_bool(force_synchronous, "force_synchronous")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MapDataset", input_dataset=input_dataset,
                      other_arguments=other_arguments, f=f,
                      output_types=output_types, output_shapes=output_shapes,
                      use_inter_op_parallelism=use_inter_op_parallelism,
                      preserve_cardinality=preserve_cardinality,
                      force_synchronous=force_synchronous, metadata=metadata,
                      name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("f", _op.get_attr("f"), "Targuments",
              _op.get_attr("Targuments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "use_inter_op_parallelism",
              _op._get_attr_bool("use_inter_op_parallelism"),
              "preserve_cardinality",
              _op._get_attr_bool("preserve_cardinality"), "force_synchronous",
              _op._get_attr_bool("force_synchronous"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MapDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MapDataset = tf_export("raw_ops.MapDataset")(_ops.to_raw_op(map_dataset))


def map_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, f, output_types, output_shapes, use_inter_op_parallelism: bool, preserve_cardinality: bool, force_synchronous: bool, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'map_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'map_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if use_inter_op_parallelism is None:
    use_inter_op_parallelism = True
  use_inter_op_parallelism = _execute.make_bool(use_inter_op_parallelism, "use_inter_op_parallelism")
  if preserve_cardinality is None:
    preserve_cardinality = False
  preserve_cardinality = _execute.make_bool(preserve_cardinality, "preserve_cardinality")
  if force_synchronous is None:
    force_synchronous = False
  force_synchronous = _execute.make_bool(force_synchronous, "force_synchronous")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset] + list(other_arguments)
  _attrs = ("f", f, "Targuments", _attr_Targuments, "output_types",
  output_types, "output_shapes", output_shapes, "use_inter_op_parallelism",
  use_inter_op_parallelism, "preserve_cardinality", preserve_cardinality,
  "force_synchronous", force_synchronous, "metadata", metadata)
  _result = _execute.execute(b"MapDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MapDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def map_defun(arguments, captured_inputs, output_types, output_shapes, f, max_intra_op_parallelism:int=1, name=None):
  r"""  Maps a function on the list of tensors unpacked from arguments on dimension 0.
  The function given by `f` is assumed to be stateless, and is executed
  concurrently on all the slices; up to batch_size (i.e. the size of the 0th
  dimension of each argument) functions will be scheduled at once.

  The `max_intra_op_parallelism` attr, which defaults to 1, can be used to
  limit the intra op parallelism. To limit inter-op parallelism, a user can
  set a private threadpool on the dataset using `tf.data.Options`'s
  `ThreadingOptions`.

  Note that this op is not exposed to users directly, but is invoked in tf.data
  rewrites.

  Args:
    arguments: A list of `Tensor` objects.
          A list of tensors whose types are `Targuments`, corresponding to the inputs
          the function should be mapped over.
    captured_inputs: A list of `Tensor` objects.
          A list of tensors whose types are `Tcaptured`, corresponding to the captured
          inputs of the defun.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
      A list of types.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
      A list of shapes.
    f: A function decorated with @Defun.
    max_intra_op_parallelism: An optional `int`. Defaults to `1`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `output_types`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MapDefun", name, arguments, captured_inputs, "output_types",
        output_types, "output_shapes", output_shapes, "f", f,
        "max_intra_op_parallelism", max_intra_op_parallelism)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return map_defun_eager_fallback(
          arguments, captured_inputs, output_types=output_types,
          output_shapes=output_shapes, f=f,
          max_intra_op_parallelism=max_intra_op_parallelism, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'map_defun' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'map_defun' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if max_intra_op_parallelism is None:
    max_intra_op_parallelism = 1
  max_intra_op_parallelism = _execute.make_int(max_intra_op_parallelism, "max_intra_op_parallelism")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MapDefun", arguments=arguments, captured_inputs=captured_inputs,
                    output_types=output_types, output_shapes=output_shapes,
                    f=f, max_intra_op_parallelism=max_intra_op_parallelism,
                    name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Targuments", _op.get_attr("Targuments"), "Tcaptured",
              _op.get_attr("Tcaptured"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "f", _op.get_attr("f"),
              "max_intra_op_parallelism",
              _op._get_attr_int("max_intra_op_parallelism"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MapDefun", _inputs_flat, _attrs, _result)
  return _result

MapDefun = tf_export("raw_ops.MapDefun")(_ops.to_raw_op(map_defun))


def map_defun_eager_fallback(arguments, captured_inputs, output_types, output_shapes, f, max_intra_op_parallelism: int, name, ctx):
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'map_defun' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'map_defun' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if max_intra_op_parallelism is None:
    max_intra_op_parallelism = 1
  max_intra_op_parallelism = _execute.make_int(max_intra_op_parallelism, "max_intra_op_parallelism")
  _attr_Targuments, arguments = _execute.convert_to_mixed_eager_tensors(arguments, ctx)
  _attr_Tcaptured, captured_inputs = _execute.convert_to_mixed_eager_tensors(captured_inputs, ctx)
  _inputs_flat = list(arguments) + list(captured_inputs)
  _attrs = ("Targuments", _attr_Targuments, "Tcaptured", _attr_Tcaptured,
  "output_types", output_types, "output_shapes", output_shapes, "f", f,
  "max_intra_op_parallelism", max_intra_op_parallelism)
  _result = _execute.execute(b"MapDefun", len(output_types),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MapDefun", _inputs_flat, _attrs, _result)
  return _result


def model_dataset(input_dataset: Annotated[Any, _atypes.Variant], output_types, output_shapes, algorithm:int=0, cpu_budget:int=0, ram_budget:int=0, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Identity transformation that models performance.

  Identity transformation that models performance.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the input dataset.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    algorithm: An optional `int`. Defaults to `0`.
    cpu_budget: An optional `int`. Defaults to `0`.
    ram_budget: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ModelDataset", name, input_dataset, "algorithm", algorithm,
        "cpu_budget", cpu_budget, "ram_budget", ram_budget, "output_types",
        output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return model_dataset_eager_fallback(
          input_dataset, algorithm=algorithm, cpu_budget=cpu_budget,
          ram_budget=ram_budget, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'model_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'model_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if algorithm is None:
    algorithm = 0
  algorithm = _execute.make_int(algorithm, "algorithm")
  if cpu_budget is None:
    cpu_budget = 0
  cpu_budget = _execute.make_int(cpu_budget, "cpu_budget")
  if ram_budget is None:
    ram_budget = 0
  ram_budget = _execute.make_int(ram_budget, "ram_budget")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ModelDataset", input_dataset=input_dataset,
                        output_types=output_types,
                        output_shapes=output_shapes, algorithm=algorithm,
                        cpu_budget=cpu_budget, ram_budget=ram_budget,
                        name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("algorithm", _op._get_attr_int("algorithm"), "cpu_budget",
              _op._get_attr_int("cpu_budget"), "ram_budget",
              _op._get_attr_int("ram_budget"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ModelDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ModelDataset = tf_export("raw_ops.ModelDataset")(_ops.to_raw_op(model_dataset))


def model_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], output_types, output_shapes, algorithm: int, cpu_budget: int, ram_budget: int, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'model_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'model_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if algorithm is None:
    algorithm = 0
  algorithm = _execute.make_int(algorithm, "algorithm")
  if cpu_budget is None:
    cpu_budget = 0
  cpu_budget = _execute.make_int(cpu_budget, "cpu_budget")
  if ram_budget is None:
    ram_budget = 0
  ram_budget = _execute.make_int(ram_budget, "ram_budget")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset]
  _attrs = ("algorithm", algorithm, "cpu_budget", cpu_budget, "ram_budget",
  ram_budget, "output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"ModelDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ModelDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def multi_device_iterator(devices, shared_name: str, container: str, output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Resource]:
  r"""Creates a MultiDeviceIterator resource.

  Args:
    devices: A list of `strings` that has length `>= 1`.
      A list of devices the iterator works across.
    shared_name: A `string`.
      If non-empty, this resource will be shared under the given name
      across multiple sessions.
    container: A `string`.
      If non-empty, this resource is placed in the given container.
      Otherwise, a default container is used.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
      The type list for the return values.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
      The list of shapes being produced.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MultiDeviceIterator", name, "devices", devices, "shared_name",
        shared_name, "container", container, "output_types", output_types,
        "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return multi_device_iterator_eager_fallback(
          devices=devices, shared_name=shared_name, container=container,
          output_types=output_types, output_shapes=output_shapes, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(devices, (list, tuple)):
    raise TypeError(
        "Expected list for 'devices' argument to "
        "'multi_device_iterator' Op, not %r." % devices)
  devices = [_execute.make_str(_s, "devices") for _s in devices]
  shared_name = _execute.make_str(shared_name, "shared_name")
  container = _execute.make_str(container, "container")
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'multi_device_iterator' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'multi_device_iterator' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MultiDeviceIterator", devices=devices, shared_name=shared_name,
                               container=container, output_types=output_types,
                               output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("devices", _op.get_attr("devices"), "shared_name",
              _op.get_attr("shared_name"), "container",
              _op.get_attr("container"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MultiDeviceIterator", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MultiDeviceIterator = tf_export("raw_ops.MultiDeviceIterator")(_ops.to_raw_op(multi_device_iterator))


def multi_device_iterator_eager_fallback(devices, shared_name: str, container: str, output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Resource]:
  if not isinstance(devices, (list, tuple)):
    raise TypeError(
        "Expected list for 'devices' argument to "
        "'multi_device_iterator' Op, not %r." % devices)
  devices = [_execute.make_str(_s, "devices") for _s in devices]
  shared_name = _execute.make_str(shared_name, "shared_name")
  container = _execute.make_str(container, "container")
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'multi_device_iterator' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'multi_device_iterator' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _inputs_flat = []
  _attrs = ("devices", devices, "shared_name", shared_name, "container",
  container, "output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"MultiDeviceIterator", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MultiDeviceIterator", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def multi_device_iterator_from_string_handle(string_handle: Annotated[Any, _atypes.String], output_types=[], output_shapes=[], name=None) -> Annotated[Any, _atypes.Resource]:
  r"""Generates a MultiDeviceIterator resource from its provided string handle.

  Args:
    string_handle: A `Tensor` of type `string`.
      String representing the resource.
    output_types: An optional list of `tf.DTypes`. Defaults to `[]`.
      The type list for the return values.
    output_shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
      The list of shapes being produced.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MultiDeviceIteratorFromStringHandle", name, string_handle,
        "output_types", output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return multi_device_iterator_from_string_handle_eager_fallback(
          string_handle, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if output_types is None:
    output_types = []
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'multi_device_iterator_from_string_handle' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if output_shapes is None:
    output_shapes = []
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'multi_device_iterator_from_string_handle' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MultiDeviceIteratorFromStringHandle", string_handle=string_handle,
                                               output_types=output_types,
                                               output_shapes=output_shapes,
                                               name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MultiDeviceIteratorFromStringHandle", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MultiDeviceIteratorFromStringHandle = tf_export("raw_ops.MultiDeviceIteratorFromStringHandle")(_ops.to_raw_op(multi_device_iterator_from_string_handle))


def multi_device_iterator_from_string_handle_eager_fallback(string_handle: Annotated[Any, _atypes.String], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Resource]:
  if output_types is None:
    output_types = []
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'multi_device_iterator_from_string_handle' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if output_shapes is None:
    output_shapes = []
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'multi_device_iterator_from_string_handle' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  string_handle = _ops.convert_to_tensor(string_handle, _dtypes.string)
  _inputs_flat = [string_handle]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"MultiDeviceIteratorFromStringHandle", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MultiDeviceIteratorFromStringHandle", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def multi_device_iterator_get_next_from_shard(multi_device_iterator: Annotated[Any, _atypes.Resource], shard_num: Annotated[Any, _atypes.Int32], incarnation_id: Annotated[Any, _atypes.Int64], output_types, output_shapes, name=None):
  r"""Gets next element for the provided shard number.

  Args:
    multi_device_iterator: A `Tensor` of type `resource`.
      A MultiDeviceIterator resource.
    shard_num: A `Tensor` of type `int32`.
      Integer representing which shard to fetch data for.
    incarnation_id: A `Tensor` of type `int64`.
      Which incarnation of the MultiDeviceIterator is running.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
      The type list for the return values.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
      The list of shapes being produced.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `output_types`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MultiDeviceIteratorGetNextFromShard", name,
        multi_device_iterator, shard_num, incarnation_id, "output_types",
        output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return multi_device_iterator_get_next_from_shard_eager_fallback(
          multi_device_iterator, shard_num, incarnation_id,
          output_types=output_types, output_shapes=output_shapes, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'multi_device_iterator_get_next_from_shard' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'multi_device_iterator_get_next_from_shard' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MultiDeviceIteratorGetNextFromShard", multi_device_iterator=multi_device_iterator,
                                               shard_num=shard_num,
                                               incarnation_id=incarnation_id,
                                               output_types=output_types,
                                               output_shapes=output_shapes,
                                               name=name)
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MultiDeviceIteratorGetNextFromShard", _inputs_flat, _attrs, _result)
  return _result

MultiDeviceIteratorGetNextFromShard = tf_export("raw_ops.MultiDeviceIteratorGetNextFromShard")(_ops.to_raw_op(multi_device_iterator_get_next_from_shard))


def multi_device_iterator_get_next_from_shard_eager_fallback(multi_device_iterator: Annotated[Any, _atypes.Resource], shard_num: Annotated[Any, _atypes.Int32], incarnation_id: Annotated[Any, _atypes.Int64], output_types, output_shapes, name, ctx):
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'multi_device_iterator_get_next_from_shard' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'multi_device_iterator_get_next_from_shard' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  multi_device_iterator = _ops.convert_to_tensor(multi_device_iterator, _dtypes.resource)
  shard_num = _ops.convert_to_tensor(shard_num, _dtypes.int32)
  incarnation_id = _ops.convert_to_tensor(incarnation_id, _dtypes.int64)
  _inputs_flat = [multi_device_iterator, shard_num, incarnation_id]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"MultiDeviceIteratorGetNextFromShard",
                             len(output_types), inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MultiDeviceIteratorGetNextFromShard", _inputs_flat, _attrs, _result)
  return _result


def multi_device_iterator_init(dataset: Annotated[Any, _atypes.Variant], multi_device_iterator: Annotated[Any, _atypes.Resource], max_buffer_size: Annotated[Any, _atypes.Int64], name=None) -> Annotated[Any, _atypes.Int64]:
  r"""Initializes the multi device iterator with the given dataset.

  Args:
    dataset: A `Tensor` of type `variant`. Dataset to be iterated upon.
    multi_device_iterator: A `Tensor` of type `resource`.
      A MultiDeviceIteratorResource.
    max_buffer_size: A `Tensor` of type `int64`.
      The maximum size of the host side per device buffer to keep.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MultiDeviceIteratorInit", name, dataset, multi_device_iterator,
        max_buffer_size)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return multi_device_iterator_init_eager_fallback(
          dataset, multi_device_iterator, max_buffer_size, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MultiDeviceIteratorInit", dataset=dataset,
                                   multi_device_iterator=multi_device_iterator,
                                   max_buffer_size=max_buffer_size, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MultiDeviceIteratorInit", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MultiDeviceIteratorInit = tf_export("raw_ops.MultiDeviceIteratorInit")(_ops.to_raw_op(multi_device_iterator_init))


def multi_device_iterator_init_eager_fallback(dataset: Annotated[Any, _atypes.Variant], multi_device_iterator: Annotated[Any, _atypes.Resource], max_buffer_size: Annotated[Any, _atypes.Int64], name, ctx) -> Annotated[Any, _atypes.Int64]:
  dataset = _ops.convert_to_tensor(dataset, _dtypes.variant)
  multi_device_iterator = _ops.convert_to_tensor(multi_device_iterator, _dtypes.resource)
  max_buffer_size = _ops.convert_to_tensor(max_buffer_size, _dtypes.int64)
  _inputs_flat = [dataset, multi_device_iterator, max_buffer_size]
  _attrs = None
  _result = _execute.execute(b"MultiDeviceIteratorInit", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MultiDeviceIteratorInit", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def multi_device_iterator_to_string_handle(multi_device_iterator: Annotated[Any, _atypes.Resource], name=None) -> Annotated[Any, _atypes.String]:
  r"""Produces a string handle for the given MultiDeviceIterator.

  Args:
    multi_device_iterator: A `Tensor` of type `resource`.
      A MultiDeviceIterator resource.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MultiDeviceIteratorToStringHandle", name,
        multi_device_iterator)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return multi_device_iterator_to_string_handle_eager_fallback(
          multi_device_iterator, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MultiDeviceIteratorToStringHandle", multi_device_iterator=multi_device_iterator,
                                             name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MultiDeviceIteratorToStringHandle", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MultiDeviceIteratorToStringHandle = tf_export("raw_ops.MultiDeviceIteratorToStringHandle")(_ops.to_raw_op(multi_device_iterator_to_string_handle))


def multi_device_iterator_to_string_handle_eager_fallback(multi_device_iterator: Annotated[Any, _atypes.Resource], name, ctx) -> Annotated[Any, _atypes.String]:
  multi_device_iterator = _ops.convert_to_tensor(multi_device_iterator, _dtypes.resource)
  _inputs_flat = [multi_device_iterator]
  _attrs = None
  _result = _execute.execute(b"MultiDeviceIteratorToStringHandle", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MultiDeviceIteratorToStringHandle", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def one_shot_iterator(dataset_factory, output_types, output_shapes, container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.Resource]:
  r"""Makes a "one-shot" iterator that can be iterated only once.

  A one-shot iterator bundles the logic for defining the dataset and
  the state of the iterator in a single op, which allows simple input
  pipelines to be defined without an additional initialization
  ("MakeIterator") step.

  One-shot iterators have the following limitations:

  * They do not support parameterization: all logic for creating the underlying
    dataset must be bundled in the `dataset_factory` function.
  * They are not resettable. Once a one-shot iterator reaches the end of its
    underlying dataset, subsequent "IteratorGetNext" operations on that
    iterator will always produce an `OutOfRange` error.

  For greater flexibility, use "Iterator" and "MakeIterator" to define
  an iterator using an arbitrary subgraph, which may capture tensors
  (including fed values) as parameters, and which may be reset multiple
  times by rerunning "MakeIterator".

  Args:
    dataset_factory: A function decorated with @Defun.
      A function of type `() -> DT_VARIANT`, where the returned
      DT_VARIANT is a dataset.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "OneShotIterator", name, "dataset_factory", dataset_factory,
        "output_types", output_types, "output_shapes", output_shapes,
        "container", container, "shared_name", shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return one_shot_iterator_eager_fallback(
          dataset_factory=dataset_factory, output_types=output_types,
          output_shapes=output_shapes, container=container,
          shared_name=shared_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'one_shot_iterator' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'one_shot_iterator' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "OneShotIterator", dataset_factory=dataset_factory,
                           output_types=output_types,
                           output_shapes=output_shapes, container=container,
                           shared_name=shared_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dataset_factory", _op.get_attr("dataset_factory"),
              "output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "container",
              _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "OneShotIterator", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

OneShotIterator = tf_export("raw_ops.OneShotIterator")(_ops.to_raw_op(one_shot_iterator))


def one_shot_iterator_eager_fallback(dataset_factory, output_types, output_shapes, container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.Resource]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'one_shot_iterator' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'one_shot_iterator' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _inputs_flat = []
  _attrs = ("dataset_factory", dataset_factory, "output_types", output_types,
  "output_shapes", output_shapes, "container", container, "shared_name",
  shared_name)
  _result = _execute.execute(b"OneShotIterator", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "OneShotIterator", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def optimize_dataset(input_dataset: Annotated[Any, _atypes.Variant], optimizations: Annotated[Any, _atypes.String], output_types, output_shapes, optimization_configs=[], name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset by applying optimizations to `input_dataset`.

  Creates a dataset by applying optimizations to `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the input dataset.
    optimizations: A `Tensor` of type `string`.
      A `tf.string` vector `tf.Tensor` identifying optimizations to use.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    optimization_configs: An optional list of `strings`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "OptimizeDataset", name, input_dataset, optimizations,
        "output_types", output_types, "output_shapes", output_shapes,
        "optimization_configs", optimization_configs)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return optimize_dataset_eager_fallback(
          input_dataset, optimizations, output_types=output_types,
          output_shapes=output_shapes,
          optimization_configs=optimization_configs, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'optimize_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'optimize_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if optimization_configs is None:
    optimization_configs = []
  if not isinstance(optimization_configs, (list, tuple)):
    raise TypeError(
        "Expected list for 'optimization_configs' argument to "
        "'optimize_dataset' Op, not %r." % optimization_configs)
  optimization_configs = [_execute.make_str(_s, "optimization_configs") for _s in optimization_configs]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "OptimizeDataset", input_dataset=input_dataset,
                           optimizations=optimizations,
                           output_types=output_types,
                           output_shapes=output_shapes,
                           optimization_configs=optimization_configs,
                           name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "optimization_configs",
              _op.get_attr("optimization_configs"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "OptimizeDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

OptimizeDataset = tf_export("raw_ops.OptimizeDataset")(_ops.to_raw_op(optimize_dataset))


def optimize_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], optimizations: Annotated[Any, _atypes.String], output_types, output_shapes, optimization_configs, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'optimize_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'optimize_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if optimization_configs is None:
    optimization_configs = []
  if not isinstance(optimization_configs, (list, tuple)):
    raise TypeError(
        "Expected list for 'optimization_configs' argument to "
        "'optimize_dataset' Op, not %r." % optimization_configs)
  optimization_configs = [_execute.make_str(_s, "optimization_configs") for _s in optimization_configs]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  optimizations = _ops.convert_to_tensor(optimizations, _dtypes.string)
  _inputs_flat = [input_dataset, optimizations]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes,
  "optimization_configs", optimization_configs)
  _result = _execute.execute(b"OptimizeDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "OptimizeDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def optimize_dataset_v2(input_dataset: Annotated[Any, _atypes.Variant], optimizations_enabled: Annotated[Any, _atypes.String], optimizations_disabled: Annotated[Any, _atypes.String], optimizations_default: Annotated[Any, _atypes.String], output_types, output_shapes, optimization_configs=[], name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset by applying related optimizations to `input_dataset`.

  Creates a dataset by applying related optimizations to `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the input dataset.
    optimizations_enabled: A `Tensor` of type `string`.
      A `tf.string` vector `tf.Tensor` identifying user enabled optimizations.
    optimizations_disabled: A `Tensor` of type `string`.
      A `tf.string` vector `tf.Tensor` identifying user disabled optimizations.
    optimizations_default: A `Tensor` of type `string`.
      A `tf.string` vector `tf.Tensor` identifying optimizations by default.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    optimization_configs: An optional list of `strings`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "OptimizeDatasetV2", name, input_dataset, optimizations_enabled,
        optimizations_disabled, optimizations_default, "output_types",
        output_types, "output_shapes", output_shapes, "optimization_configs",
        optimization_configs)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return optimize_dataset_v2_eager_fallback(
          input_dataset, optimizations_enabled, optimizations_disabled,
          optimizations_default, output_types=output_types,
          output_shapes=output_shapes,
          optimization_configs=optimization_configs, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'optimize_dataset_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'optimize_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if optimization_configs is None:
    optimization_configs = []
  if not isinstance(optimization_configs, (list, tuple)):
    raise TypeError(
        "Expected list for 'optimization_configs' argument to "
        "'optimize_dataset_v2' Op, not %r." % optimization_configs)
  optimization_configs = [_execute.make_str(_s, "optimization_configs") for _s in optimization_configs]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "OptimizeDatasetV2", input_dataset=input_dataset,
                             optimizations_enabled=optimizations_enabled,
                             optimizations_disabled=optimizations_disabled,
                             optimizations_default=optimizations_default,
                             output_types=output_types,
                             output_shapes=output_shapes,
                             optimization_configs=optimization_configs,
                             name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "optimization_configs",
              _op.get_attr("optimization_configs"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "OptimizeDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

OptimizeDatasetV2 = tf_export("raw_ops.OptimizeDatasetV2")(_ops.to_raw_op(optimize_dataset_v2))


def optimize_dataset_v2_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], optimizations_enabled: Annotated[Any, _atypes.String], optimizations_disabled: Annotated[Any, _atypes.String], optimizations_default: Annotated[Any, _atypes.String], output_types, output_shapes, optimization_configs, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'optimize_dataset_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'optimize_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if optimization_configs is None:
    optimization_configs = []
  if not isinstance(optimization_configs, (list, tuple)):
    raise TypeError(
        "Expected list for 'optimization_configs' argument to "
        "'optimize_dataset_v2' Op, not %r." % optimization_configs)
  optimization_configs = [_execute.make_str(_s, "optimization_configs") for _s in optimization_configs]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  optimizations_enabled = _ops.convert_to_tensor(optimizations_enabled, _dtypes.string)
  optimizations_disabled = _ops.convert_to_tensor(optimizations_disabled, _dtypes.string)
  optimizations_default = _ops.convert_to_tensor(optimizations_default, _dtypes.string)
  _inputs_flat = [input_dataset, optimizations_enabled, optimizations_disabled, optimizations_default]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes,
  "optimization_configs", optimization_configs)
  _result = _execute.execute(b"OptimizeDatasetV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "OptimizeDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def options_dataset(input_dataset: Annotated[Any, _atypes.Variant], serialized_options: str, output_types, output_shapes, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset by attaching tf.data.Options to `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the input dataset.
    serialized_options: A `string`.
      A `tf.string` scalar `tf.Tensor` of serialized `tf.data.Options` protocol buffer.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "OptionsDataset", name, input_dataset, "serialized_options",
        serialized_options, "output_types", output_types, "output_shapes",
        output_shapes, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return options_dataset_eager_fallback(
          input_dataset, serialized_options=serialized_options,
          output_types=output_types, output_shapes=output_shapes,
          metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  serialized_options = _execute.make_str(serialized_options, "serialized_options")
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'options_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'options_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "OptionsDataset", input_dataset=input_dataset,
                          serialized_options=serialized_options,
                          output_types=output_types,
                          output_shapes=output_shapes, metadata=metadata,
                          name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("serialized_options", _op.get_attr("serialized_options"),
              "output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "OptionsDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

OptionsDataset = tf_export("raw_ops.OptionsDataset")(_ops.to_raw_op(options_dataset))


def options_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], serialized_options: str, output_types, output_shapes, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  serialized_options = _execute.make_str(serialized_options, "serialized_options")
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'options_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'options_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset]
  _attrs = ("serialized_options", serialized_options, "output_types",
  output_types, "output_shapes", output_shapes, "metadata", metadata)
  _result = _execute.execute(b"OptionsDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "OptionsDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def padded_batch_dataset(input_dataset: Annotated[Any, _atypes.Variant], batch_size: Annotated[Any, _atypes.Int64], padded_shapes: Annotated[List[Any], _atypes.Int64], padding_values, output_shapes, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that batches and pads `batch_size` elements from the input.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    batch_size: A `Tensor` of type `int64`.
      A scalar representing the number of elements to accumulate in a
      batch.
    padded_shapes: A list of at least 1 `Tensor` objects with type `int64`.
      A list of int64 tensors representing the desired padded shapes
      of the corresponding output components. These shapes may be partially
      specified, using `-1` to indicate that a particular dimension should be
      padded to the maximum size of all batch elements.
    padding_values: A list of `Tensor` objects.
      A list of scalars containing the padding value to use for
      each of the outputs.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "PaddedBatchDataset", name, input_dataset, batch_size,
        padded_shapes, padding_values, "output_shapes", output_shapes,
        "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return padded_batch_dataset_eager_fallback(
          input_dataset, batch_size, padded_shapes, padding_values,
          output_shapes=output_shapes, metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(padded_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'padded_shapes' argument to "
        "'padded_batch_dataset' Op, not %r." % padded_shapes)
  _attr_N = len(padded_shapes)
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'padded_batch_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "PaddedBatchDataset", input_dataset=input_dataset,
                              batch_size=batch_size,
                              padded_shapes=padded_shapes,
                              padding_values=padding_values,
                              output_shapes=output_shapes, metadata=metadata,
                              name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Toutput_types", _op.get_attr("Toutput_types"), "output_shapes",
              _op.get_attr("output_shapes"), "N", _op._get_attr_int("N"),
              "metadata", _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "PaddedBatchDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

PaddedBatchDataset = tf_export("raw_ops.PaddedBatchDataset")(_ops.to_raw_op(padded_batch_dataset))


def padded_batch_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], batch_size: Annotated[Any, _atypes.Int64], padded_shapes: Annotated[List[Any], _atypes.Int64], padding_values, output_shapes, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(padded_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'padded_shapes' argument to "
        "'padded_batch_dataset' Op, not %r." % padded_shapes)
  _attr_N = len(padded_shapes)
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'padded_batch_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _attr_Toutput_types, padding_values = _execute.convert_to_mixed_eager_tensors(padding_values, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  batch_size = _ops.convert_to_tensor(batch_size, _dtypes.int64)
  padded_shapes = _ops.convert_n_to_tensor(padded_shapes, _dtypes.int64)
  _inputs_flat = [input_dataset, batch_size] + list(padded_shapes) + list(padding_values)
  _attrs = ("Toutput_types", _attr_Toutput_types, "output_shapes",
  output_shapes, "N", _attr_N, "metadata", metadata)
  _result = _execute.execute(b"PaddedBatchDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "PaddedBatchDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def padded_batch_dataset_v2(input_dataset: Annotated[Any, _atypes.Variant], batch_size: Annotated[Any, _atypes.Int64], padded_shapes: Annotated[List[Any], _atypes.Int64], padding_values, drop_remainder: Annotated[Any, _atypes.Bool], output_shapes, parallel_copy:bool=False, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that batches and pads `batch_size` elements from the input.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    batch_size: A `Tensor` of type `int64`.
      A scalar representing the number of elements to accumulate in a
      batch.
    padded_shapes: A list of at least 1 `Tensor` objects with type `int64`.
      A list of int64 tensors representing the desired padded shapes
      of the corresponding output components. These shapes may be partially
      specified, using `-1` to indicate that a particular dimension should be
      padded to the maximum size of all batch elements.
    padding_values: A list of `Tensor` objects.
      A list of scalars containing the padding value to use for
      each of the outputs.
    drop_remainder: A `Tensor` of type `bool`.
      A scalar representing whether the last batch should be dropped in case its size
      is smaller than desired.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    parallel_copy: An optional `bool`. Defaults to `False`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "PaddedBatchDatasetV2", name, input_dataset, batch_size,
        padded_shapes, padding_values, drop_remainder, "parallel_copy",
        parallel_copy, "output_shapes", output_shapes, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return padded_batch_dataset_v2_eager_fallback(
          input_dataset, batch_size, padded_shapes, padding_values,
          drop_remainder, parallel_copy=parallel_copy,
          output_shapes=output_shapes, metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(padded_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'padded_shapes' argument to "
        "'padded_batch_dataset_v2' Op, not %r." % padded_shapes)
  _attr_N = len(padded_shapes)
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'padded_batch_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if parallel_copy is None:
    parallel_copy = False
  parallel_copy = _execute.make_bool(parallel_copy, "parallel_copy")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "PaddedBatchDatasetV2", input_dataset=input_dataset,
                                batch_size=batch_size,
                                padded_shapes=padded_shapes,
                                padding_values=padding_values,
                                drop_remainder=drop_remainder,
                                output_shapes=output_shapes,
                                parallel_copy=parallel_copy,
                                metadata=metadata, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("parallel_copy", _op._get_attr_bool("parallel_copy"),
              "Toutput_types", _op.get_attr("Toutput_types"), "output_shapes",
              _op.get_attr("output_shapes"), "N", _op._get_attr_int("N"),
              "metadata", _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "PaddedBatchDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

PaddedBatchDatasetV2 = tf_export("raw_ops.PaddedBatchDatasetV2")(_ops.to_raw_op(padded_batch_dataset_v2))


def padded_batch_dataset_v2_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], batch_size: Annotated[Any, _atypes.Int64], padded_shapes: Annotated[List[Any], _atypes.Int64], padding_values, drop_remainder: Annotated[Any, _atypes.Bool], output_shapes, parallel_copy: bool, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(padded_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'padded_shapes' argument to "
        "'padded_batch_dataset_v2' Op, not %r." % padded_shapes)
  _attr_N = len(padded_shapes)
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'padded_batch_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if parallel_copy is None:
    parallel_copy = False
  parallel_copy = _execute.make_bool(parallel_copy, "parallel_copy")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _attr_Toutput_types, padding_values = _execute.convert_to_mixed_eager_tensors(padding_values, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  batch_size = _ops.convert_to_tensor(batch_size, _dtypes.int64)
  padded_shapes = _ops.convert_n_to_tensor(padded_shapes, _dtypes.int64)
  drop_remainder = _ops.convert_to_tensor(drop_remainder, _dtypes.bool)
  _inputs_flat = [input_dataset, batch_size] + list(padded_shapes) + list(padding_values) + [drop_remainder]
  _attrs = ("parallel_copy", parallel_copy, "Toutput_types",
  _attr_Toutput_types, "output_shapes", output_shapes, "N", _attr_N,
  "metadata", metadata)
  _result = _execute.execute(b"PaddedBatchDatasetV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "PaddedBatchDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def parallel_batch_dataset(input_dataset: Annotated[Any, _atypes.Variant], batch_size: Annotated[Any, _atypes.Int64], num_parallel_calls: Annotated[Any, _atypes.Int64], drop_remainder: Annotated[Any, _atypes.Bool], output_types, output_shapes, parallel_copy:bool=False, deterministic:str="default", metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    batch_size: A `Tensor` of type `int64`.
    num_parallel_calls: A `Tensor` of type `int64`.
    drop_remainder: A `Tensor` of type `bool`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    parallel_copy: An optional `bool`. Defaults to `False`.
    deterministic: An optional `string`. Defaults to `"default"`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ParallelBatchDataset", name, input_dataset, batch_size,
        num_parallel_calls, drop_remainder, "parallel_copy", parallel_copy,
        "output_types", output_types, "output_shapes", output_shapes,
        "deterministic", deterministic, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return parallel_batch_dataset_eager_fallback(
          input_dataset, batch_size, num_parallel_calls, drop_remainder,
          parallel_copy=parallel_copy, output_types=output_types,
          output_shapes=output_shapes, deterministic=deterministic,
          metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'parallel_batch_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'parallel_batch_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if parallel_copy is None:
    parallel_copy = False
  parallel_copy = _execute.make_bool(parallel_copy, "parallel_copy")
  if deterministic is None:
    deterministic = "default"
  deterministic = _execute.make_str(deterministic, "deterministic")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ParallelBatchDataset", input_dataset=input_dataset,
                                batch_size=batch_size,
                                num_parallel_calls=num_parallel_calls,
                                drop_remainder=drop_remainder,
                                output_types=output_types,
                                output_shapes=output_shapes,
                                parallel_copy=parallel_copy,
                                deterministic=deterministic,
                                metadata=metadata, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("parallel_copy", _op._get_attr_bool("parallel_copy"),
              "output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "deterministic",
              _op.get_attr("deterministic"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ParallelBatchDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ParallelBatchDataset = tf_export("raw_ops.ParallelBatchDataset")(_ops.to_raw_op(parallel_batch_dataset))


def parallel_batch_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], batch_size: Annotated[Any, _atypes.Int64], num_parallel_calls: Annotated[Any, _atypes.Int64], drop_remainder: Annotated[Any, _atypes.Bool], output_types, output_shapes, parallel_copy: bool, deterministic: str, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'parallel_batch_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'parallel_batch_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if parallel_copy is None:
    parallel_copy = False
  parallel_copy = _execute.make_bool(parallel_copy, "parallel_copy")
  if deterministic is None:
    deterministic = "default"
  deterministic = _execute.make_str(deterministic, "deterministic")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  batch_size = _ops.convert_to_tensor(batch_size, _dtypes.int64)
  num_parallel_calls = _ops.convert_to_tensor(num_parallel_calls, _dtypes.int64)
  drop_remainder = _ops.convert_to_tensor(drop_remainder, _dtypes.bool)
  _inputs_flat = [input_dataset, batch_size, num_parallel_calls, drop_remainder]
  _attrs = ("parallel_copy", parallel_copy, "output_types", output_types,
  "output_shapes", output_shapes, "deterministic", deterministic, "metadata",
  metadata)
  _result = _execute.execute(b"ParallelBatchDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ParallelBatchDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def parallel_filter_dataset(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, num_parallel_calls: Annotated[Any, _atypes.Int64], predicate, output_types, output_shapes, deterministic:str="default", metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset containing elements of `input_dataset` matching `predicate`.

  The `predicate` function must return a scalar boolean and accept the
  following arguments:

  * One tensor for each component of an element of `input_dataset`.
  * One tensor for each value in `other_arguments`.

  Unlike a "FilterDataset", which applies `predicate` sequentially, this dataset
  invokes up to `num_parallel_calls` copies of `predicate` in parallel.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    other_arguments: A list of `Tensor` objects.
      A list of tensors, typically values that were captured when
      building a closure for `predicate`.
    num_parallel_calls: A `Tensor` of type `int64`.
      The number of concurrent invocations of `predicate` that process
      elements from `input_dataset` in parallel.
    predicate: A function decorated with @Defun.
      A function returning a scalar boolean.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    deterministic: An optional `string`. Defaults to `"default"`.
      A string indicating the op-level determinism to use. Deterministic controls
      whether the interleave is allowed to return elements out of order if the next
      element to be returned isn't available, but a later element is. Options are
      "true", "false", and "default". "default" indicates that determinism should be
      decided by the `experimental_deterministic` parameter of `tf.data.Options`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ParallelFilterDataset", name, input_dataset, other_arguments,
        num_parallel_calls, "predicate", predicate, "deterministic",
        deterministic, "output_types", output_types, "output_shapes",
        output_shapes, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return parallel_filter_dataset_eager_fallback(
          input_dataset, other_arguments, num_parallel_calls,
          predicate=predicate, deterministic=deterministic,
          output_types=output_types, output_shapes=output_shapes,
          metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'parallel_filter_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'parallel_filter_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if deterministic is None:
    deterministic = "default"
  deterministic = _execute.make_str(deterministic, "deterministic")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ParallelFilterDataset", input_dataset=input_dataset,
                                 other_arguments=other_arguments,
                                 num_parallel_calls=num_parallel_calls,
                                 predicate=predicate,
                                 output_types=output_types,
                                 output_shapes=output_shapes,
                                 deterministic=deterministic,
                                 metadata=metadata, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("predicate", _op.get_attr("predicate"), "deterministic",
              _op.get_attr("deterministic"), "Targuments",
              _op.get_attr("Targuments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ParallelFilterDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ParallelFilterDataset = tf_export("raw_ops.ParallelFilterDataset")(_ops.to_raw_op(parallel_filter_dataset))


def parallel_filter_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, num_parallel_calls: Annotated[Any, _atypes.Int64], predicate, output_types, output_shapes, deterministic: str, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'parallel_filter_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'parallel_filter_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if deterministic is None:
    deterministic = "default"
  deterministic = _execute.make_str(deterministic, "deterministic")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  num_parallel_calls = _ops.convert_to_tensor(num_parallel_calls, _dtypes.int64)
  _inputs_flat = [input_dataset] + list(other_arguments) + [num_parallel_calls]
  _attrs = ("predicate", predicate, "deterministic", deterministic,
  "Targuments", _attr_Targuments, "output_types", output_types,
  "output_shapes", output_shapes, "metadata", metadata)
  _result = _execute.execute(b"ParallelFilterDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ParallelFilterDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def parallel_interleave_dataset_v2(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, cycle_length: Annotated[Any, _atypes.Int64], block_length: Annotated[Any, _atypes.Int64], num_parallel_calls: Annotated[Any, _atypes.Int64], f, output_types, output_shapes, sloppy:bool=False, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that applies `f` to the outputs of `input_dataset`.

  The resulting dataset is similar to the `InterleaveDataset`, except that the
  dataset will fetch records from the interleaved datasets in parallel.

  The `tf.data` Python API creates instances of this op from
  `Dataset.interleave()` when the `num_parallel_calls` parameter of that method
  is set to any value other than `None`.

  By default, the output of this dataset will be deterministic, which may result
  in the dataset blocking if the next data item to be returned isn't available.
  In order to avoid head-of-line blocking, one can set the
  `experimental_deterministic` parameter of `tf.data.Options` to `False`,
  which can improve performance at the expense of non-determinism.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      Dataset that produces a stream of arguments for the function `f`.
    other_arguments: A list of `Tensor` objects.
      Additional arguments to pass to `f` beyond those produced by `input_dataset`.
      Evaluated once when the dataset is instantiated.
    cycle_length: A `Tensor` of type `int64`.
      Number of datasets (each created by applying `f` to the elements of
      `input_dataset`) among which the `ParallelInterleaveDatasetV2` will cycle in a
      round-robin fashion.
    block_length: A `Tensor` of type `int64`.
      Number of elements at a time to produce from each interleaved invocation of a
      dataset returned by `f`.
    num_parallel_calls: A `Tensor` of type `int64`.
      Determines the number of threads that should be used for fetching data from
      input datasets in parallel. The Python API `tf.data.experimental.AUTOTUNE`
      constant can be used to indicate that the level of parallelism should be autotuned.
    f: A function decorated with @Defun.
      A function mapping elements of `input_dataset`, concatenated with
      `other_arguments`, to a Dataset variant that contains elements matching
      `output_types` and `output_shapes`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    sloppy: An optional `bool`. Defaults to `False`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ParallelInterleaveDatasetV2", name, input_dataset,
        other_arguments, cycle_length, block_length, num_parallel_calls, "f",
        f, "output_types", output_types, "output_shapes", output_shapes,
        "sloppy", sloppy, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return parallel_interleave_dataset_v2_eager_fallback(
          input_dataset, other_arguments, cycle_length, block_length,
          num_parallel_calls, f=f, output_types=output_types,
          output_shapes=output_shapes, sloppy=sloppy, metadata=metadata,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'parallel_interleave_dataset_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'parallel_interleave_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if sloppy is None:
    sloppy = False
  sloppy = _execute.make_bool(sloppy, "sloppy")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ParallelInterleaveDatasetV2", input_dataset=input_dataset,
                                       other_arguments=other_arguments,
                                       cycle_length=cycle_length,
                                       block_length=block_length,
                                       num_parallel_calls=num_parallel_calls,
                                       f=f, output_types=output_types,
                                       output_shapes=output_shapes,
                                       sloppy=sloppy, metadata=metadata,
                                       name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("f", _op.get_attr("f"), "Targuments",
              _op.get_attr("Targuments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "sloppy",
              _op._get_attr_bool("sloppy"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ParallelInterleaveDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ParallelInterleaveDatasetV2 = tf_export("raw_ops.ParallelInterleaveDatasetV2")(_ops.to_raw_op(parallel_interleave_dataset_v2))


def parallel_interleave_dataset_v2_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, cycle_length: Annotated[Any, _atypes.Int64], block_length: Annotated[Any, _atypes.Int64], num_parallel_calls: Annotated[Any, _atypes.Int64], f, output_types, output_shapes, sloppy: bool, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'parallel_interleave_dataset_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'parallel_interleave_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if sloppy is None:
    sloppy = False
  sloppy = _execute.make_bool(sloppy, "sloppy")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  cycle_length = _ops.convert_to_tensor(cycle_length, _dtypes.int64)
  block_length = _ops.convert_to_tensor(block_length, _dtypes.int64)
  num_parallel_calls = _ops.convert_to_tensor(num_parallel_calls, _dtypes.int64)
  _inputs_flat = [input_dataset] + list(other_arguments) + [cycle_length, block_length, num_parallel_calls]
  _attrs = ("f", f, "Targuments", _attr_Targuments, "output_types",
  output_types, "output_shapes", output_shapes, "sloppy", sloppy, "metadata",
  metadata)
  _result = _execute.execute(b"ParallelInterleaveDatasetV2", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ParallelInterleaveDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def parallel_interleave_dataset_v3(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, cycle_length: Annotated[Any, _atypes.Int64], block_length: Annotated[Any, _atypes.Int64], num_parallel_calls: Annotated[Any, _atypes.Int64], f, output_types, output_shapes, deterministic:str="default", metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that applies `f` to the outputs of `input_dataset`.

  The resulting dataset is similar to the `InterleaveDataset`, except that the
  dataset will fetch records from the interleaved datasets in parallel.

  The `tf.data` Python API creates instances of this op from
  `Dataset.interleave()` when the `num_parallel_calls` parameter of that method
  is set to any value other than `None`.

  By default, the output of this dataset will be deterministic, which may result
  in the dataset blocking if the next data item to be returned isn't available.
  In order to avoid head-of-line blocking, one can either set the `deterministic`
  attribute to "false", or leave it as "default" and set the
  `experimental_deterministic` parameter of `tf.data.Options` to `False`.
  This can improve performance at the expense of non-determinism.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      Dataset that produces a stream of arguments for the function `f`.
    other_arguments: A list of `Tensor` objects.
      Additional arguments to pass to `f` beyond those produced by `input_dataset`.
      Evaluated once when the dataset is instantiated.
    cycle_length: A `Tensor` of type `int64`.
      Number of datasets (each created by applying `f` to the elements of
      `input_dataset`) among which the `ParallelInterleaveDatasetV2` will cycle in a
      round-robin fashion.
    block_length: A `Tensor` of type `int64`.
      Number of elements at a time to produce from each interleaved invocation of a
      dataset returned by `f`.
    num_parallel_calls: A `Tensor` of type `int64`.
      Determines the number of threads that should be used for fetching data from
      input datasets in parallel. The Python API `tf.data.experimental.AUTOTUNE`
      constant can be used to indicate that the level of parallelism should be autotuned.
    f: A function decorated with @Defun.
      A function mapping elements of `input_dataset`, concatenated with
      `other_arguments`, to a Dataset variant that contains elements matching
      `output_types` and `output_shapes`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    deterministic: An optional `string`. Defaults to `"default"`.
      A string indicating the op-level determinism to use. Deterministic controls
      whether the interleave is allowed to return elements out of order if the next
      element to be returned isn't available, but a later element is. Options are
      "true", "false", and "default". "default" indicates that determinism should be
      decided by the `experimental_deterministic` parameter of `tf.data.Options`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ParallelInterleaveDatasetV3", name, input_dataset,
        other_arguments, cycle_length, block_length, num_parallel_calls, "f",
        f, "deterministic", deterministic, "output_types", output_types,
        "output_shapes", output_shapes, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return parallel_interleave_dataset_v3_eager_fallback(
          input_dataset, other_arguments, cycle_length, block_length,
          num_parallel_calls, f=f, deterministic=deterministic,
          output_types=output_types, output_shapes=output_shapes,
          metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'parallel_interleave_dataset_v3' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'parallel_interleave_dataset_v3' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if deterministic is None:
    deterministic = "default"
  deterministic = _execute.make_str(deterministic, "deterministic")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ParallelInterleaveDatasetV3", input_dataset=input_dataset,
                                       other_arguments=other_arguments,
                                       cycle_length=cycle_length,
                                       block_length=block_length,
                                       num_parallel_calls=num_parallel_calls,
                                       f=f, output_types=output_types,
                                       output_shapes=output_shapes,
                                       deterministic=deterministic,
                                       metadata=metadata, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("f", _op.get_attr("f"), "deterministic",
              _op.get_attr("deterministic"), "Targuments",
              _op.get_attr("Targuments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ParallelInterleaveDatasetV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ParallelInterleaveDatasetV3 = tf_export("raw_ops.ParallelInterleaveDatasetV3")(_ops.to_raw_op(parallel_interleave_dataset_v3))


def parallel_interleave_dataset_v3_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, cycle_length: Annotated[Any, _atypes.Int64], block_length: Annotated[Any, _atypes.Int64], num_parallel_calls: Annotated[Any, _atypes.Int64], f, output_types, output_shapes, deterministic: str, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'parallel_interleave_dataset_v3' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'parallel_interleave_dataset_v3' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if deterministic is None:
    deterministic = "default"
  deterministic = _execute.make_str(deterministic, "deterministic")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  cycle_length = _ops.convert_to_tensor(cycle_length, _dtypes.int64)
  block_length = _ops.convert_to_tensor(block_length, _dtypes.int64)
  num_parallel_calls = _ops.convert_to_tensor(num_parallel_calls, _dtypes.int64)
  _inputs_flat = [input_dataset] + list(other_arguments) + [cycle_length, block_length, num_parallel_calls]
  _attrs = ("f", f, "deterministic", deterministic, "Targuments",
  _attr_Targuments, "output_types", output_types, "output_shapes",
  output_shapes, "metadata", metadata)
  _result = _execute.execute(b"ParallelInterleaveDatasetV3", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ParallelInterleaveDatasetV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def parallel_interleave_dataset_v4(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, cycle_length: Annotated[Any, _atypes.Int64], block_length: Annotated[Any, _atypes.Int64], buffer_output_elements: Annotated[Any, _atypes.Int64], prefetch_input_elements: Annotated[Any, _atypes.Int64], num_parallel_calls: Annotated[Any, _atypes.Int64], f, output_types, output_shapes, deterministic:str="default", metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that applies `f` to the outputs of `input_dataset`.

  The resulting dataset is similar to the `InterleaveDataset`, except that the
  dataset will fetch records from the interleaved datasets in parallel.

  The `tf.data` Python API creates instances of this op from
  `Dataset.interleave()` when the `num_parallel_calls` parameter of that method
  is set to any value other than `None`.

  By default, the output of this dataset will be deterministic, which may result
  in the dataset blocking if the next data item to be returned isn't available.
  In order to avoid head-of-line blocking, one can either set the `deterministic`
  attribute to "false", or leave it as "default" and set the
  `experimental_deterministic` parameter of `tf.data.Options` to `False`.
  This can improve performance at the expense of non-determinism.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      Dataset that produces a stream of arguments for the function `f`.
    other_arguments: A list of `Tensor` objects.
      Additional arguments to pass to `f` beyond those produced by `input_dataset`.
      Evaluated once when the dataset is instantiated.
    cycle_length: A `Tensor` of type `int64`.
      Number of datasets (each created by applying `f` to the elements of
      `input_dataset`) among which the `ParallelInterleaveDatasetV2` will cycle in a
      round-robin fashion.
    block_length: A `Tensor` of type `int64`.
      Number of elements at a time to produce from each interleaved invocation of a
      dataset returned by `f`.
    buffer_output_elements: A `Tensor` of type `int64`.
      The number of elements each iterator being interleaved should buffer (similar
      to the `.prefetch()` transformation for each interleaved iterator).
    prefetch_input_elements: A `Tensor` of type `int64`.
      Determines the number of iterators to prefetch, allowing buffers to warm up and
      data to be pre-fetched without blocking the main thread.
    num_parallel_calls: A `Tensor` of type `int64`.
      Determines the number of threads that should be used for fetching data from
      input datasets in parallel. The Python API `tf.data.experimental.AUTOTUNE`
      constant can be used to indicate that the level of parallelism should be autotuned.
    f: A function decorated with @Defun.
      A function mapping elements of `input_dataset`, concatenated with
      `other_arguments`, to a Dataset variant that contains elements matching
      `output_types` and `output_shapes`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    deterministic: An optional `string`. Defaults to `"default"`.
      A string indicating the op-level determinism to use. Deterministic controls
      whether the interleave is allowed to return elements out of order if the next
      element to be returned isn't available, but a later element is. Options are
      "true", "false", and "default". "default" indicates that determinism should be
      decided by the `experimental_deterministic` parameter of `tf.data.Options`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ParallelInterleaveDatasetV4", name, input_dataset,
        other_arguments, cycle_length, block_length, buffer_output_elements,
        prefetch_input_elements, num_parallel_calls, "f", f, "deterministic",
        deterministic, "output_types", output_types, "output_shapes",
        output_shapes, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return parallel_interleave_dataset_v4_eager_fallback(
          input_dataset, other_arguments, cycle_length, block_length,
          buffer_output_elements, prefetch_input_elements, num_parallel_calls,
          f=f, deterministic=deterministic, output_types=output_types,
          output_shapes=output_shapes, metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'parallel_interleave_dataset_v4' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'parallel_interleave_dataset_v4' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if deterministic is None:
    deterministic = "default"
  deterministic = _execute.make_str(deterministic, "deterministic")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ParallelInterleaveDatasetV4", input_dataset=input_dataset,
                                       other_arguments=other_arguments,
                                       cycle_length=cycle_length,
                                       block_length=block_length,
                                       buffer_output_elements=buffer_output_elements,
                                       prefetch_input_elements=prefetch_input_elements,
                                       num_parallel_calls=num_parallel_calls,
                                       f=f, output_types=output_types,
                                       output_shapes=output_shapes,
                                       deterministic=deterministic,
                                       metadata=metadata, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("f", _op.get_attr("f"), "deterministic",
              _op.get_attr("deterministic"), "Targuments",
              _op.get_attr("Targuments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ParallelInterleaveDatasetV4", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ParallelInterleaveDatasetV4 = tf_export("raw_ops.ParallelInterleaveDatasetV4")(_ops.to_raw_op(parallel_interleave_dataset_v4))


def parallel_interleave_dataset_v4_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, cycle_length: Annotated[Any, _atypes.Int64], block_length: Annotated[Any, _atypes.Int64], buffer_output_elements: Annotated[Any, _atypes.Int64], prefetch_input_elements: Annotated[Any, _atypes.Int64], num_parallel_calls: Annotated[Any, _atypes.Int64], f, output_types, output_shapes, deterministic: str, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'parallel_interleave_dataset_v4' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'parallel_interleave_dataset_v4' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if deterministic is None:
    deterministic = "default"
  deterministic = _execute.make_str(deterministic, "deterministic")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  cycle_length = _ops.convert_to_tensor(cycle_length, _dtypes.int64)
  block_length = _ops.convert_to_tensor(block_length, _dtypes.int64)
  buffer_output_elements = _ops.convert_to_tensor(buffer_output_elements, _dtypes.int64)
  prefetch_input_elements = _ops.convert_to_tensor(prefetch_input_elements, _dtypes.int64)
  num_parallel_calls = _ops.convert_to_tensor(num_parallel_calls, _dtypes.int64)
  _inputs_flat = [input_dataset] + list(other_arguments) + [cycle_length, block_length, buffer_output_elements, prefetch_input_elements, num_parallel_calls]
  _attrs = ("f", f, "deterministic", deterministic, "Targuments",
  _attr_Targuments, "output_types", output_types, "output_shapes",
  output_shapes, "metadata", metadata)
  _result = _execute.execute(b"ParallelInterleaveDatasetV4", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ParallelInterleaveDatasetV4", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def parallel_map_dataset(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, num_parallel_calls: Annotated[Any, _atypes.Int32], f, output_types, output_shapes, use_inter_op_parallelism:bool=True, sloppy:bool=False, preserve_cardinality:bool=False, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that applies `f` to the outputs of `input_dataset`.

  Unlike a "MapDataset", which applies `f` sequentially, this dataset invokes up
  to `num_parallel_calls` copies of `f` in parallel.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    other_arguments: A list of `Tensor` objects.
    num_parallel_calls: A `Tensor` of type `int32`.
      The number of concurrent invocations of `f` that process
      elements from `input_dataset` in parallel.
    f: A function decorated with @Defun.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    use_inter_op_parallelism: An optional `bool`. Defaults to `True`.
    sloppy: An optional `bool`. Defaults to `False`.
    preserve_cardinality: An optional `bool`. Defaults to `False`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ParallelMapDataset", name, input_dataset, other_arguments,
        num_parallel_calls, "f", f, "output_types", output_types,
        "output_shapes", output_shapes, "use_inter_op_parallelism",
        use_inter_op_parallelism, "sloppy", sloppy, "preserve_cardinality",
        preserve_cardinality, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return parallel_map_dataset_eager_fallback(
          input_dataset, other_arguments, num_parallel_calls, f=f,
          output_types=output_types, output_shapes=output_shapes,
          use_inter_op_parallelism=use_inter_op_parallelism, sloppy=sloppy,
          preserve_cardinality=preserve_cardinality, metadata=metadata,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'parallel_map_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'parallel_map_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if use_inter_op_parallelism is None:
    use_inter_op_parallelism = True
  use_inter_op_parallelism = _execute.make_bool(use_inter_op_parallelism, "use_inter_op_parallelism")
  if sloppy is None:
    sloppy = False
  sloppy = _execute.make_bool(sloppy, "sloppy")
  if preserve_cardinality is None:
    preserve_cardinality = False
  preserve_cardinality = _execute.make_bool(preserve_cardinality, "preserve_cardinality")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ParallelMapDataset", input_dataset=input_dataset,
                              other_arguments=other_arguments,
                              num_parallel_calls=num_parallel_calls, f=f,
                              output_types=output_types,
                              output_shapes=output_shapes,
                              use_inter_op_parallelism=use_inter_op_parallelism,
                              sloppy=sloppy,
                              preserve_cardinality=preserve_cardinality,
                              metadata=metadata, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("f", _op.get_attr("f"), "Targuments",
              _op.get_attr("Targuments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "use_inter_op_parallelism",
              _op._get_attr_bool("use_inter_op_parallelism"), "sloppy",
              _op._get_attr_bool("sloppy"), "preserve_cardinality",
              _op._get_attr_bool("preserve_cardinality"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ParallelMapDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ParallelMapDataset = tf_export("raw_ops.ParallelMapDataset")(_ops.to_raw_op(parallel_map_dataset))


def parallel_map_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, num_parallel_calls: Annotated[Any, _atypes.Int32], f, output_types, output_shapes, use_inter_op_parallelism: bool, sloppy: bool, preserve_cardinality: bool, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'parallel_map_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'parallel_map_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if use_inter_op_parallelism is None:
    use_inter_op_parallelism = True
  use_inter_op_parallelism = _execute.make_bool(use_inter_op_parallelism, "use_inter_op_parallelism")
  if sloppy is None:
    sloppy = False
  sloppy = _execute.make_bool(sloppy, "sloppy")
  if preserve_cardinality is None:
    preserve_cardinality = False
  preserve_cardinality = _execute.make_bool(preserve_cardinality, "preserve_cardinality")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  num_parallel_calls = _ops.convert_to_tensor(num_parallel_calls, _dtypes.int32)
  _inputs_flat = [input_dataset] + list(other_arguments) + [num_parallel_calls]
  _attrs = ("f", f, "Targuments", _attr_Targuments, "output_types",
  output_types, "output_shapes", output_shapes, "use_inter_op_parallelism",
  use_inter_op_parallelism, "sloppy", sloppy, "preserve_cardinality",
  preserve_cardinality, "metadata", metadata)
  _result = _execute.execute(b"ParallelMapDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ParallelMapDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def parallel_map_dataset_v2(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, num_parallel_calls: Annotated[Any, _atypes.Int64], f, output_types, output_shapes, use_inter_op_parallelism:bool=True, deterministic:str="default", preserve_cardinality:bool=False, use_unbounded_threadpool:bool=False, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that applies `f` to the outputs of `input_dataset`.

  Unlike a "MapDataset", which applies `f` sequentially, this dataset invokes up
  to `num_parallel_calls` copies of `f` in parallel.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    other_arguments: A list of `Tensor` objects.
    num_parallel_calls: A `Tensor` of type `int64`.
      The number of concurrent invocations of `f` that process
      elements from `input_dataset` in parallel.
    f: A function decorated with @Defun.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    use_inter_op_parallelism: An optional `bool`. Defaults to `True`.
    deterministic: An optional `string`. Defaults to `"default"`.
    preserve_cardinality: An optional `bool`. Defaults to `False`.
    use_unbounded_threadpool: An optional `bool`. Defaults to `False`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ParallelMapDatasetV2", name, input_dataset, other_arguments,
        num_parallel_calls, "f", f, "output_types", output_types,
        "output_shapes", output_shapes, "use_inter_op_parallelism",
        use_inter_op_parallelism, "deterministic", deterministic,
        "preserve_cardinality", preserve_cardinality,
        "use_unbounded_threadpool", use_unbounded_threadpool, "metadata",
        metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return parallel_map_dataset_v2_eager_fallback(
          input_dataset, other_arguments, num_parallel_calls, f=f,
          output_types=output_types, output_shapes=output_shapes,
          use_inter_op_parallelism=use_inter_op_parallelism,
          deterministic=deterministic,
          preserve_cardinality=preserve_cardinality,
          use_unbounded_threadpool=use_unbounded_threadpool,
          metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'parallel_map_dataset_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'parallel_map_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if use_inter_op_parallelism is None:
    use_inter_op_parallelism = True
  use_inter_op_parallelism = _execute.make_bool(use_inter_op_parallelism, "use_inter_op_parallelism")
  if deterministic is None:
    deterministic = "default"
  deterministic = _execute.make_str(deterministic, "deterministic")
  if preserve_cardinality is None:
    preserve_cardinality = False
  preserve_cardinality = _execute.make_bool(preserve_cardinality, "preserve_cardinality")
  if use_unbounded_threadpool is None:
    use_unbounded_threadpool = False
  use_unbounded_threadpool = _execute.make_bool(use_unbounded_threadpool, "use_unbounded_threadpool")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ParallelMapDatasetV2", input_dataset=input_dataset,
                                other_arguments=other_arguments,
                                num_parallel_calls=num_parallel_calls, f=f,
                                output_types=output_types,
                                output_shapes=output_shapes,
                                use_inter_op_parallelism=use_inter_op_parallelism,
                                deterministic=deterministic,
                                preserve_cardinality=preserve_cardinality,
                                use_unbounded_threadpool=use_unbounded_threadpool,
                                metadata=metadata, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("f", _op.get_attr("f"), "Targuments",
              _op.get_attr("Targuments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "use_inter_op_parallelism",
              _op._get_attr_bool("use_inter_op_parallelism"), "deterministic",
              _op.get_attr("deterministic"), "preserve_cardinality",
              _op._get_attr_bool("preserve_cardinality"),
              "use_unbounded_threadpool",
              _op._get_attr_bool("use_unbounded_threadpool"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ParallelMapDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ParallelMapDatasetV2 = tf_export("raw_ops.ParallelMapDatasetV2")(_ops.to_raw_op(parallel_map_dataset_v2))


def parallel_map_dataset_v2_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], other_arguments, num_parallel_calls: Annotated[Any, _atypes.Int64], f, output_types, output_shapes, use_inter_op_parallelism: bool, deterministic: str, preserve_cardinality: bool, use_unbounded_threadpool: bool, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'parallel_map_dataset_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'parallel_map_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if use_inter_op_parallelism is None:
    use_inter_op_parallelism = True
  use_inter_op_parallelism = _execute.make_bool(use_inter_op_parallelism, "use_inter_op_parallelism")
  if deterministic is None:
    deterministic = "default"
  deterministic = _execute.make_str(deterministic, "deterministic")
  if preserve_cardinality is None:
    preserve_cardinality = False
  preserve_cardinality = _execute.make_bool(preserve_cardinality, "preserve_cardinality")
  if use_unbounded_threadpool is None:
    use_unbounded_threadpool = False
  use_unbounded_threadpool = _execute.make_bool(use_unbounded_threadpool, "use_unbounded_threadpool")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  num_parallel_calls = _ops.convert_to_tensor(num_parallel_calls, _dtypes.int64)
  _inputs_flat = [input_dataset] + list(other_arguments) + [num_parallel_calls]
  _attrs = ("f", f, "Targuments", _attr_Targuments, "output_types",
  output_types, "output_shapes", output_shapes, "use_inter_op_parallelism",
  use_inter_op_parallelism, "deterministic", deterministic,
  "preserve_cardinality", preserve_cardinality, "use_unbounded_threadpool",
  use_unbounded_threadpool, "metadata", metadata)
  _result = _execute.execute(b"ParallelMapDatasetV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ParallelMapDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def prefetch_dataset(input_dataset: Annotated[Any, _atypes.Variant], buffer_size: Annotated[Any, _atypes.Int64], output_types, output_shapes, slack_period:int=0, legacy_autotune:bool=True, buffer_size_min:int=0, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that asynchronously prefetches elements from `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    buffer_size: A `Tensor` of type `int64`.
      The maximum number of elements to buffer in an iterator over
      this dataset.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    slack_period: An optional `int`. Defaults to `0`.
    legacy_autotune: An optional `bool`. Defaults to `True`.
    buffer_size_min: An optional `int`. Defaults to `0`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "PrefetchDataset", name, input_dataset, buffer_size,
        "output_types", output_types, "output_shapes", output_shapes,
        "slack_period", slack_period, "legacy_autotune", legacy_autotune,
        "buffer_size_min", buffer_size_min, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return prefetch_dataset_eager_fallback(
          input_dataset, buffer_size, output_types=output_types,
          output_shapes=output_shapes, slack_period=slack_period,
          legacy_autotune=legacy_autotune, buffer_size_min=buffer_size_min,
          metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'prefetch_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'prefetch_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if slack_period is None:
    slack_period = 0
  slack_period = _execute.make_int(slack_period, "slack_period")
  if legacy_autotune is None:
    legacy_autotune = True
  legacy_autotune = _execute.make_bool(legacy_autotune, "legacy_autotune")
  if buffer_size_min is None:
    buffer_size_min = 0
  buffer_size_min = _execute.make_int(buffer_size_min, "buffer_size_min")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "PrefetchDataset", input_dataset=input_dataset,
                           buffer_size=buffer_size, output_types=output_types,
                           output_shapes=output_shapes,
                           slack_period=slack_period,
                           legacy_autotune=legacy_autotune,
                           buffer_size_min=buffer_size_min, metadata=metadata,
                           name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "slack_period",
              _op._get_attr_int("slack_period"), "legacy_autotune",
              _op._get_attr_bool("legacy_autotune"), "buffer_size_min",
              _op._get_attr_int("buffer_size_min"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "PrefetchDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

PrefetchDataset = tf_export("raw_ops.PrefetchDataset")(_ops.to_raw_op(prefetch_dataset))


def prefetch_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], buffer_size: Annotated[Any, _atypes.Int64], output_types, output_shapes, slack_period: int, legacy_autotune: bool, buffer_size_min: int, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'prefetch_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'prefetch_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if slack_period is None:
    slack_period = 0
  slack_period = _execute.make_int(slack_period, "slack_period")
  if legacy_autotune is None:
    legacy_autotune = True
  legacy_autotune = _execute.make_bool(legacy_autotune, "legacy_autotune")
  if buffer_size_min is None:
    buffer_size_min = 0
  buffer_size_min = _execute.make_int(buffer_size_min, "buffer_size_min")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  buffer_size = _ops.convert_to_tensor(buffer_size, _dtypes.int64)
  _inputs_flat = [input_dataset, buffer_size]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes,
  "slack_period", slack_period, "legacy_autotune", legacy_autotune,
  "buffer_size_min", buffer_size_min, "metadata", metadata)
  _result = _execute.execute(b"PrefetchDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "PrefetchDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def range_dataset(start: Annotated[Any, _atypes.Int64], stop: Annotated[Any, _atypes.Int64], step: Annotated[Any, _atypes.Int64], output_types, output_shapes, metadata:str="", replicate_on_split:bool=False, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset with a range of values. Corresponds to python's xrange.

  Args:
    start: A `Tensor` of type `int64`.
      corresponds to start in python's xrange().
    stop: A `Tensor` of type `int64`.
      corresponds to stop in python's xrange().
    step: A `Tensor` of type `int64`.
      corresponds to step in python's xrange().
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    metadata: An optional `string`. Defaults to `""`.
    replicate_on_split: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RangeDataset", name, start, stop, step, "output_types",
        output_types, "output_shapes", output_shapes, "metadata", metadata,
        "replicate_on_split", replicate_on_split)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return range_dataset_eager_fallback(
          start, stop, step, output_types=output_types,
          output_shapes=output_shapes, metadata=metadata,
          replicate_on_split=replicate_on_split, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'range_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'range_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  if replicate_on_split is None:
    replicate_on_split = False
  replicate_on_split = _execute.make_bool(replicate_on_split, "replicate_on_split")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RangeDataset", start=start, stop=stop, step=step,
                        output_types=output_types,
                        output_shapes=output_shapes, metadata=metadata,
                        replicate_on_split=replicate_on_split, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"), "replicate_on_split",
              _op._get_attr_bool("replicate_on_split"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RangeDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RangeDataset = tf_export("raw_ops.RangeDataset")(_ops.to_raw_op(range_dataset))


def range_dataset_eager_fallback(start: Annotated[Any, _atypes.Int64], stop: Annotated[Any, _atypes.Int64], step: Annotated[Any, _atypes.Int64], output_types, output_shapes, metadata: str, replicate_on_split: bool, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'range_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'range_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  if replicate_on_split is None:
    replicate_on_split = False
  replicate_on_split = _execute.make_bool(replicate_on_split, "replicate_on_split")
  start = _ops.convert_to_tensor(start, _dtypes.int64)
  stop = _ops.convert_to_tensor(stop, _dtypes.int64)
  step = _ops.convert_to_tensor(step, _dtypes.int64)
  _inputs_flat = [start, stop, step]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes,
  "metadata", metadata, "replicate_on_split", replicate_on_split)
  _result = _execute.execute(b"RangeDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RangeDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def reduce_dataset(input_dataset: Annotated[Any, _atypes.Variant], initial_state, other_arguments, f, output_types, output_shapes, use_inter_op_parallelism:bool=True, metadata:str="", name=None):
  r"""Reduces the input dataset to a singleton using a reduce function.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the input dataset.
    initial_state: A list of `Tensor` objects.
      A nested structure of tensors, representing the initial state of the
      transformation.
    other_arguments: A list of `Tensor` objects.
    f: A function decorated with @Defun.
      A function that maps `(old_state, input_element)` to `new_state`. It must take
      two arguments and return a nested structures of tensors. The structure of
      `new_state` must match the structure of `initial_state`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    use_inter_op_parallelism: An optional `bool`. Defaults to `True`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `output_types`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ReduceDataset", name, input_dataset, initial_state,
        other_arguments, "f", f, "output_types", output_types,
        "output_shapes", output_shapes, "use_inter_op_parallelism",
        use_inter_op_parallelism, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return reduce_dataset_eager_fallback(
          input_dataset, initial_state, other_arguments, f=f,
          output_types=output_types, output_shapes=output_shapes,
          use_inter_op_parallelism=use_inter_op_parallelism,
          metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'reduce_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'reduce_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if use_inter_op_parallelism is None:
    use_inter_op_parallelism = True
  use_inter_op_parallelism = _execute.make_bool(use_inter_op_parallelism, "use_inter_op_parallelism")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ReduceDataset", input_dataset=input_dataset,
                         initial_state=initial_state,
                         other_arguments=other_arguments, f=f,
                         output_types=output_types,
                         output_shapes=output_shapes,
                         use_inter_op_parallelism=use_inter_op_parallelism,
                         metadata=metadata, name=name)
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("f", _op.get_attr("f"), "Tstate", _op.get_attr("Tstate"),
              "Targuments", _op.get_attr("Targuments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "use_inter_op_parallelism",
              _op._get_attr_bool("use_inter_op_parallelism"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ReduceDataset", _inputs_flat, _attrs, _result)
  return _result

ReduceDataset = tf_export("raw_ops.ReduceDataset")(_ops.to_raw_op(reduce_dataset))


def reduce_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], initial_state, other_arguments, f, output_types, output_shapes, use_inter_op_parallelism: bool, metadata: str, name, ctx):
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'reduce_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'reduce_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if use_inter_op_parallelism is None:
    use_inter_op_parallelism = True
  use_inter_op_parallelism = _execute.make_bool(use_inter_op_parallelism, "use_inter_op_parallelism")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _attr_Tstate, initial_state = _execute.convert_to_mixed_eager_tensors(initial_state, ctx)
  _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset] + list(initial_state) + list(other_arguments)
  _attrs = ("f", f, "Tstate", _attr_Tstate, "Targuments", _attr_Targuments,
  "output_types", output_types, "output_shapes", output_shapes,
  "use_inter_op_parallelism", use_inter_op_parallelism, "metadata", metadata)
  _result = _execute.execute(b"ReduceDataset", len(output_types),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ReduceDataset", _inputs_flat, _attrs, _result)
  return _result


def repeat_dataset(input_dataset: Annotated[Any, _atypes.Variant], count: Annotated[Any, _atypes.Int64], output_types, output_shapes, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that emits the outputs of `input_dataset` `count` times.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    count: A `Tensor` of type `int64`.
      A scalar representing the number of times that `input_dataset` should
      be repeated. A value of `-1` indicates that it should be repeated infinitely.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RepeatDataset", name, input_dataset, count, "output_types",
        output_types, "output_shapes", output_shapes, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return repeat_dataset_eager_fallback(
          input_dataset, count, output_types=output_types,
          output_shapes=output_shapes, metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'repeat_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'repeat_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RepeatDataset", input_dataset=input_dataset, count=count,
                         output_types=output_types,
                         output_shapes=output_shapes, metadata=metadata,
                         name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RepeatDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RepeatDataset = tf_export("raw_ops.RepeatDataset")(_ops.to_raw_op(repeat_dataset))


def repeat_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], count: Annotated[Any, _atypes.Int64], output_types, output_shapes, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'repeat_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'repeat_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  count = _ops.convert_to_tensor(count, _dtypes.int64)
  _inputs_flat = [input_dataset, count]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes,
  "metadata", metadata)
  _result = _execute.execute(b"RepeatDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RepeatDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def rewrite_dataset(input_dataset: Annotated[Any, _atypes.Variant], rewrite_name: Annotated[Any, _atypes.String], output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    rewrite_name: A `Tensor` of type `string`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RewriteDataset", name, input_dataset, rewrite_name,
        "output_types", output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return rewrite_dataset_eager_fallback(
          input_dataset, rewrite_name, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'rewrite_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'rewrite_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RewriteDataset", input_dataset=input_dataset,
                          rewrite_name=rewrite_name,
                          output_types=output_types,
                          output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RewriteDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RewriteDataset = tf_export("raw_ops.RewriteDataset")(_ops.to_raw_op(rewrite_dataset))


def rewrite_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], rewrite_name: Annotated[Any, _atypes.String], output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'rewrite_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'rewrite_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  rewrite_name = _ops.convert_to_tensor(rewrite_name, _dtypes.string)
  _inputs_flat = [input_dataset, rewrite_name]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"RewriteDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RewriteDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def serialize_iterator(resource_handle: Annotated[Any, _atypes.Resource], external_state_policy:int=0, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Converts the given `resource_handle` representing an iterator to a variant tensor.

  Args:
    resource_handle: A `Tensor` of type `resource`.
      A handle to an iterator resource.
    external_state_policy: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SerializeIterator", name, resource_handle,
        "external_state_policy", external_state_policy)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return serialize_iterator_eager_fallback(
          resource_handle, external_state_policy=external_state_policy,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if external_state_policy is None:
    external_state_policy = 0
  external_state_policy = _execute.make_int(external_state_policy, "external_state_policy")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SerializeIterator", resource_handle=resource_handle,
                             external_state_policy=external_state_policy,
                             name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("external_state_policy",
              _op._get_attr_int("external_state_policy"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SerializeIterator", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SerializeIterator = tf_export("raw_ops.SerializeIterator")(_ops.to_raw_op(serialize_iterator))


def serialize_iterator_eager_fallback(resource_handle: Annotated[Any, _atypes.Resource], external_state_policy: int, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if external_state_policy is None:
    external_state_policy = 0
  external_state_policy = _execute.make_int(external_state_policy, "external_state_policy")
  resource_handle = _ops.convert_to_tensor(resource_handle, _dtypes.resource)
  _inputs_flat = [resource_handle]
  _attrs = ("external_state_policy", external_state_policy)
  _result = _execute.execute(b"SerializeIterator", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SerializeIterator", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def shard_dataset(input_dataset: Annotated[Any, _atypes.Variant], num_shards: Annotated[Any, _atypes.Int64], index: Annotated[Any, _atypes.Int64], output_types, output_shapes, require_non_empty:bool=False, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a `Dataset` that includes only 1/`num_shards` of this dataset.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    num_shards: A `Tensor` of type `int64`.
      An integer representing the number of shards operating in parallel.
    index: A `Tensor` of type `int64`.
      An integer representing the current worker index.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    require_non_empty: An optional `bool`. Defaults to `False`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ShardDataset", name, input_dataset, num_shards, index,
        "require_non_empty", require_non_empty, "output_types", output_types,
        "output_shapes", output_shapes, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return shard_dataset_eager_fallback(
          input_dataset, num_shards, index,
          require_non_empty=require_non_empty, output_types=output_types,
          output_shapes=output_shapes, metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'shard_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'shard_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if require_non_empty is None:
    require_non_empty = False
  require_non_empty = _execute.make_bool(require_non_empty, "require_non_empty")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ShardDataset", input_dataset=input_dataset, num_shards=num_shards,
                        index=index, output_types=output_types,
                        output_shapes=output_shapes,
                        require_non_empty=require_non_empty,
                        metadata=metadata, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("require_non_empty", _op._get_attr_bool("require_non_empty"),
              "output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ShardDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ShardDataset = tf_export("raw_ops.ShardDataset")(_ops.to_raw_op(shard_dataset))


def shard_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], num_shards: Annotated[Any, _atypes.Int64], index: Annotated[Any, _atypes.Int64], output_types, output_shapes, require_non_empty: bool, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'shard_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'shard_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if require_non_empty is None:
    require_non_empty = False
  require_non_empty = _execute.make_bool(require_non_empty, "require_non_empty")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  num_shards = _ops.convert_to_tensor(num_shards, _dtypes.int64)
  index = _ops.convert_to_tensor(index, _dtypes.int64)
  _inputs_flat = [input_dataset, num_shards, index]
  _attrs = ("require_non_empty", require_non_empty, "output_types",
  output_types, "output_shapes", output_shapes, "metadata", metadata)
  _result = _execute.execute(b"ShardDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ShardDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def shuffle_and_repeat_dataset(input_dataset: Annotated[Any, _atypes.Variant], buffer_size: Annotated[Any, _atypes.Int64], seed: Annotated[Any, _atypes.Int64], seed2: Annotated[Any, _atypes.Int64], count: Annotated[Any, _atypes.Int64], output_types, output_shapes, reshuffle_each_iteration:bool=True, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that shuffles and repeats elements from `input_dataset`

  pseudorandomly.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    buffer_size: A `Tensor` of type `int64`.
      The number of output elements to buffer in an iterator over
      this dataset. Compare with the `min_after_dequeue` attr when creating a
      `RandomShuffleQueue`.
    seed: A `Tensor` of type `int64`.
      A scalar seed for the random number generator. If either `seed` or
      `seed2` is set to be non-zero, the random number generator is seeded
      by the given seed.  Otherwise, a random seed is used.
    seed2: A `Tensor` of type `int64`.
      A second scalar seed to avoid seed collision.
    count: A `Tensor` of type `int64`.
      A scalar representing the number of times the underlying dataset
      should be repeated. The default is `-1`, which results in infinite repetition.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    reshuffle_each_iteration: An optional `bool`. Defaults to `True`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ShuffleAndRepeatDataset", name, input_dataset, buffer_size,
        seed, seed2, count, "output_types", output_types, "output_shapes",
        output_shapes, "reshuffle_each_iteration", reshuffle_each_iteration,
        "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return shuffle_and_repeat_dataset_eager_fallback(
          input_dataset, buffer_size, seed, seed2, count,
          output_types=output_types, output_shapes=output_shapes,
          reshuffle_each_iteration=reshuffle_each_iteration,
          metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'shuffle_and_repeat_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'shuffle_and_repeat_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if reshuffle_each_iteration is None:
    reshuffle_each_iteration = True
  reshuffle_each_iteration = _execute.make_bool(reshuffle_each_iteration, "reshuffle_each_iteration")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ShuffleAndRepeatDataset", input_dataset=input_dataset,
                                   buffer_size=buffer_size, seed=seed,
                                   seed2=seed2, count=count,
                                   output_types=output_types,
                                   output_shapes=output_shapes,
                                   reshuffle_each_iteration=reshuffle_each_iteration,
                                   metadata=metadata, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "reshuffle_each_iteration",
              _op._get_attr_bool("reshuffle_each_iteration"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ShuffleAndRepeatDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ShuffleAndRepeatDataset = tf_export("raw_ops.ShuffleAndRepeatDataset")(_ops.to_raw_op(shuffle_and_repeat_dataset))


def shuffle_and_repeat_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], buffer_size: Annotated[Any, _atypes.Int64], seed: Annotated[Any, _atypes.Int64], seed2: Annotated[Any, _atypes.Int64], count: Annotated[Any, _atypes.Int64], output_types, output_shapes, reshuffle_each_iteration: bool, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'shuffle_and_repeat_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'shuffle_and_repeat_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if reshuffle_each_iteration is None:
    reshuffle_each_iteration = True
  reshuffle_each_iteration = _execute.make_bool(reshuffle_each_iteration, "reshuffle_each_iteration")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  buffer_size = _ops.convert_to_tensor(buffer_size, _dtypes.int64)
  seed = _ops.convert_to_tensor(seed, _dtypes.int64)
  seed2 = _ops.convert_to_tensor(seed2, _dtypes.int64)
  count = _ops.convert_to_tensor(count, _dtypes.int64)
  _inputs_flat = [input_dataset, buffer_size, seed, seed2, count]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes,
  "reshuffle_each_iteration", reshuffle_each_iteration, "metadata", metadata)
  _result = _execute.execute(b"ShuffleAndRepeatDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ShuffleAndRepeatDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def shuffle_and_repeat_dataset_v2(input_dataset: Annotated[Any, _atypes.Variant], buffer_size: Annotated[Any, _atypes.Int64], seed: Annotated[Any, _atypes.Int64], seed2: Annotated[Any, _atypes.Int64], count: Annotated[Any, _atypes.Int64], seed_generator: Annotated[Any, _atypes.Resource], output_types, output_shapes, reshuffle_each_iteration:bool=True, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    buffer_size: A `Tensor` of type `int64`.
    seed: A `Tensor` of type `int64`.
    seed2: A `Tensor` of type `int64`.
    count: A `Tensor` of type `int64`.
    seed_generator: A `Tensor` of type `resource`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    reshuffle_each_iteration: An optional `bool`. Defaults to `True`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ShuffleAndRepeatDatasetV2", name, input_dataset, buffer_size,
        seed, seed2, count, seed_generator, "reshuffle_each_iteration",
        reshuffle_each_iteration, "output_types", output_types,
        "output_shapes", output_shapes, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return shuffle_and_repeat_dataset_v2_eager_fallback(
          input_dataset, buffer_size, seed, seed2, count, seed_generator,
          reshuffle_each_iteration=reshuffle_each_iteration,
          output_types=output_types, output_shapes=output_shapes,
          metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'shuffle_and_repeat_dataset_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'shuffle_and_repeat_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if reshuffle_each_iteration is None:
    reshuffle_each_iteration = True
  reshuffle_each_iteration = _execute.make_bool(reshuffle_each_iteration, "reshuffle_each_iteration")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ShuffleAndRepeatDatasetV2", input_dataset=input_dataset,
                                     buffer_size=buffer_size, seed=seed,
                                     seed2=seed2, count=count,
                                     seed_generator=seed_generator,
                                     output_types=output_types,
                                     output_shapes=output_shapes,
                                     reshuffle_each_iteration=reshuffle_each_iteration,
                                     metadata=metadata, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("reshuffle_each_iteration",
              _op._get_attr_bool("reshuffle_each_iteration"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ShuffleAndRepeatDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ShuffleAndRepeatDatasetV2 = tf_export("raw_ops.ShuffleAndRepeatDatasetV2")(_ops.to_raw_op(shuffle_and_repeat_dataset_v2))


def shuffle_and_repeat_dataset_v2_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], buffer_size: Annotated[Any, _atypes.Int64], seed: Annotated[Any, _atypes.Int64], seed2: Annotated[Any, _atypes.Int64], count: Annotated[Any, _atypes.Int64], seed_generator: Annotated[Any, _atypes.Resource], output_types, output_shapes, reshuffle_each_iteration: bool, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'shuffle_and_repeat_dataset_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'shuffle_and_repeat_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if reshuffle_each_iteration is None:
    reshuffle_each_iteration = True
  reshuffle_each_iteration = _execute.make_bool(reshuffle_each_iteration, "reshuffle_each_iteration")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  buffer_size = _ops.convert_to_tensor(buffer_size, _dtypes.int64)
  seed = _ops.convert_to_tensor(seed, _dtypes.int64)
  seed2 = _ops.convert_to_tensor(seed2, _dtypes.int64)
  count = _ops.convert_to_tensor(count, _dtypes.int64)
  seed_generator = _ops.convert_to_tensor(seed_generator, _dtypes.resource)
  _inputs_flat = [input_dataset, buffer_size, seed, seed2, count, seed_generator]
  _attrs = ("reshuffle_each_iteration", reshuffle_each_iteration,
  "output_types", output_types, "output_shapes", output_shapes, "metadata",
  metadata)
  _result = _execute.execute(b"ShuffleAndRepeatDatasetV2", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ShuffleAndRepeatDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def shuffle_dataset(input_dataset: Annotated[Any, _atypes.Variant], buffer_size: Annotated[Any, _atypes.Int64], seed: Annotated[Any, _atypes.Int64], seed2: Annotated[Any, _atypes.Int64], output_types, output_shapes, reshuffle_each_iteration:bool=True, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that shuffles elements from `input_dataset` pseudorandomly.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    buffer_size: A `Tensor` of type `int64`.
      The number of output elements to buffer in an iterator over
      this dataset. Compare with the `min_after_dequeue` attr when creating a
      `RandomShuffleQueue`.
    seed: A `Tensor` of type `int64`.
      A scalar seed for the random number generator. If either `seed` or
      `seed2` is set to be non-zero, the random number generator is seeded
      by the given seed.  Otherwise, a random seed is used.
    seed2: A `Tensor` of type `int64`.
      A second scalar seed to avoid seed collision.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    reshuffle_each_iteration: An optional `bool`. Defaults to `True`.
      If true, each iterator over this dataset will be given
      a different pseudorandomly generated seed, based on a sequence seeded by the
      `seed` and `seed2` inputs. If false, each iterator will be given the same
      seed, and repeated iteration over this dataset will yield the exact same
      sequence of results.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ShuffleDataset", name, input_dataset, buffer_size, seed, seed2,
        "reshuffle_each_iteration", reshuffle_each_iteration, "output_types",
        output_types, "output_shapes", output_shapes, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return shuffle_dataset_eager_fallback(
          input_dataset, buffer_size, seed, seed2,
          reshuffle_each_iteration=reshuffle_each_iteration,
          output_types=output_types, output_shapes=output_shapes,
          metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'shuffle_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'shuffle_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if reshuffle_each_iteration is None:
    reshuffle_each_iteration = True
  reshuffle_each_iteration = _execute.make_bool(reshuffle_each_iteration, "reshuffle_each_iteration")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ShuffleDataset", input_dataset=input_dataset,
                          buffer_size=buffer_size, seed=seed, seed2=seed2,
                          output_types=output_types,
                          output_shapes=output_shapes,
                          reshuffle_each_iteration=reshuffle_each_iteration,
                          metadata=metadata, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("reshuffle_each_iteration",
              _op._get_attr_bool("reshuffle_each_iteration"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ShuffleDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ShuffleDataset = tf_export("raw_ops.ShuffleDataset")(_ops.to_raw_op(shuffle_dataset))


def shuffle_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], buffer_size: Annotated[Any, _atypes.Int64], seed: Annotated[Any, _atypes.Int64], seed2: Annotated[Any, _atypes.Int64], output_types, output_shapes, reshuffle_each_iteration: bool, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'shuffle_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'shuffle_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if reshuffle_each_iteration is None:
    reshuffle_each_iteration = True
  reshuffle_each_iteration = _execute.make_bool(reshuffle_each_iteration, "reshuffle_each_iteration")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  buffer_size = _ops.convert_to_tensor(buffer_size, _dtypes.int64)
  seed = _ops.convert_to_tensor(seed, _dtypes.int64)
  seed2 = _ops.convert_to_tensor(seed2, _dtypes.int64)
  _inputs_flat = [input_dataset, buffer_size, seed, seed2]
  _attrs = ("reshuffle_each_iteration", reshuffle_each_iteration,
  "output_types", output_types, "output_shapes", output_shapes, "metadata",
  metadata)
  _result = _execute.execute(b"ShuffleDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ShuffleDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def shuffle_dataset_v2(input_dataset: Annotated[Any, _atypes.Variant], buffer_size: Annotated[Any, _atypes.Int64], seed_generator: Annotated[Any, _atypes.Resource], output_types, output_shapes, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    buffer_size: A `Tensor` of type `int64`.
    seed_generator: A `Tensor` of type `resource`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ShuffleDatasetV2", name, input_dataset, buffer_size,
        seed_generator, "output_types", output_types, "output_shapes",
        output_shapes, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return shuffle_dataset_v2_eager_fallback(
          input_dataset, buffer_size, seed_generator,
          output_types=output_types, output_shapes=output_shapes,
          metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'shuffle_dataset_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'shuffle_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ShuffleDatasetV2", input_dataset=input_dataset,
                            buffer_size=buffer_size,
                            seed_generator=seed_generator,
                            output_types=output_types,
                            output_shapes=output_shapes, metadata=metadata,
                            name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ShuffleDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ShuffleDatasetV2 = tf_export("raw_ops.ShuffleDatasetV2")(_ops.to_raw_op(shuffle_dataset_v2))


def shuffle_dataset_v2_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], buffer_size: Annotated[Any, _atypes.Int64], seed_generator: Annotated[Any, _atypes.Resource], output_types, output_shapes, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'shuffle_dataset_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'shuffle_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  buffer_size = _ops.convert_to_tensor(buffer_size, _dtypes.int64)
  seed_generator = _ops.convert_to_tensor(seed_generator, _dtypes.resource)
  _inputs_flat = [input_dataset, buffer_size, seed_generator]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes,
  "metadata", metadata)
  _result = _execute.execute(b"ShuffleDatasetV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ShuffleDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def shuffle_dataset_v3(input_dataset: Annotated[Any, _atypes.Variant], buffer_size: Annotated[Any, _atypes.Int64], seed: Annotated[Any, _atypes.Int64], seed2: Annotated[Any, _atypes.Int64], seed_generator: Annotated[Any, _atypes.Resource], output_types, output_shapes, reshuffle_each_iteration:bool=True, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    buffer_size: A `Tensor` of type `int64`.
    seed: A `Tensor` of type `int64`.
    seed2: A `Tensor` of type `int64`.
    seed_generator: A `Tensor` of type `resource`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    reshuffle_each_iteration: An optional `bool`. Defaults to `True`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ShuffleDatasetV3", name, input_dataset, buffer_size, seed,
        seed2, seed_generator, "reshuffle_each_iteration",
        reshuffle_each_iteration, "output_types", output_types,
        "output_shapes", output_shapes, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return shuffle_dataset_v3_eager_fallback(
          input_dataset, buffer_size, seed, seed2, seed_generator,
          reshuffle_each_iteration=reshuffle_each_iteration,
          output_types=output_types, output_shapes=output_shapes,
          metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'shuffle_dataset_v3' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'shuffle_dataset_v3' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if reshuffle_each_iteration is None:
    reshuffle_each_iteration = True
  reshuffle_each_iteration = _execute.make_bool(reshuffle_each_iteration, "reshuffle_each_iteration")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ShuffleDatasetV3", input_dataset=input_dataset,
                            buffer_size=buffer_size, seed=seed, seed2=seed2,
                            seed_generator=seed_generator,
                            output_types=output_types,
                            output_shapes=output_shapes,
                            reshuffle_each_iteration=reshuffle_each_iteration,
                            metadata=metadata, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("reshuffle_each_iteration",
              _op._get_attr_bool("reshuffle_each_iteration"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ShuffleDatasetV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ShuffleDatasetV3 = tf_export("raw_ops.ShuffleDatasetV3")(_ops.to_raw_op(shuffle_dataset_v3))


def shuffle_dataset_v3_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], buffer_size: Annotated[Any, _atypes.Int64], seed: Annotated[Any, _atypes.Int64], seed2: Annotated[Any, _atypes.Int64], seed_generator: Annotated[Any, _atypes.Resource], output_types, output_shapes, reshuffle_each_iteration: bool, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'shuffle_dataset_v3' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'shuffle_dataset_v3' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if reshuffle_each_iteration is None:
    reshuffle_each_iteration = True
  reshuffle_each_iteration = _execute.make_bool(reshuffle_each_iteration, "reshuffle_each_iteration")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  buffer_size = _ops.convert_to_tensor(buffer_size, _dtypes.int64)
  seed = _ops.convert_to_tensor(seed, _dtypes.int64)
  seed2 = _ops.convert_to_tensor(seed2, _dtypes.int64)
  seed_generator = _ops.convert_to_tensor(seed_generator, _dtypes.resource)
  _inputs_flat = [input_dataset, buffer_size, seed, seed2, seed_generator]
  _attrs = ("reshuffle_each_iteration", reshuffle_each_iteration,
  "output_types", output_types, "output_shapes", output_shapes, "metadata",
  metadata)
  _result = _execute.execute(b"ShuffleDatasetV3", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ShuffleDatasetV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def skip_dataset(input_dataset: Annotated[Any, _atypes.Variant], count: Annotated[Any, _atypes.Int64], output_types, output_shapes, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that skips `count` elements from the `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    count: A `Tensor` of type `int64`.
      A scalar representing the number of elements from the `input_dataset`
      that should be skipped.  If count is -1, skips everything.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SkipDataset", name, input_dataset, count, "output_types",
        output_types, "output_shapes", output_shapes, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return skip_dataset_eager_fallback(
          input_dataset, count, output_types=output_types,
          output_shapes=output_shapes, metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'skip_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'skip_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SkipDataset", input_dataset=input_dataset, count=count,
                       output_types=output_types, output_shapes=output_shapes,
                       metadata=metadata, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SkipDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SkipDataset = tf_export("raw_ops.SkipDataset")(_ops.to_raw_op(skip_dataset))


def skip_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], count: Annotated[Any, _atypes.Int64], output_types, output_shapes, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'skip_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'skip_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  count = _ops.convert_to_tensor(count, _dtypes.int64)
  _inputs_flat = [input_dataset, count]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes,
  "metadata", metadata)
  _result = _execute.execute(b"SkipDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SkipDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SparseTensorSliceDataset_Tvalues = TypeVar("TV_SparseTensorSliceDataset_Tvalues", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def sparse_tensor_slice_dataset(indices: Annotated[Any, _atypes.Int64], values: Annotated[Any, TV_SparseTensorSliceDataset_Tvalues], dense_shape: Annotated[Any, _atypes.Int64], name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that splits a SparseTensor into elements row-wise.

  Args:
    indices: A `Tensor` of type `int64`.
    values: A `Tensor`.
    dense_shape: A `Tensor` of type `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseTensorSliceDataset", name, indices, values, dense_shape)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_tensor_slice_dataset_eager_fallback(
          indices, values, dense_shape, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseTensorSliceDataset", indices=indices, values=values,
                                    dense_shape=dense_shape, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tvalues", _op._get_attr_type("Tvalues"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseTensorSliceDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseTensorSliceDataset = tf_export("raw_ops.SparseTensorSliceDataset")(_ops.to_raw_op(sparse_tensor_slice_dataset))


def sparse_tensor_slice_dataset_eager_fallback(indices: Annotated[Any, _atypes.Int64], values: Annotated[Any, TV_SparseTensorSliceDataset_Tvalues], dense_shape: Annotated[Any, _atypes.Int64], name, ctx) -> Annotated[Any, _atypes.Variant]:
  _attr_Tvalues, (values,) = _execute.args_to_matching_eager([values], ctx, [])
  indices = _ops.convert_to_tensor(indices, _dtypes.int64)
  dense_shape = _ops.convert_to_tensor(dense_shape, _dtypes.int64)
  _inputs_flat = [indices, values, dense_shape]
  _attrs = ("Tvalues", _attr_Tvalues)
  _result = _execute.execute(b"SparseTensorSliceDataset", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseTensorSliceDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def tf_record_dataset(filenames: Annotated[Any, _atypes.String], compression_type: Annotated[Any, _atypes.String], buffer_size: Annotated[Any, _atypes.Int64], metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that emits the records from one or more TFRecord files.

  Args:
    filenames: A `Tensor` of type `string`.
      A scalar or vector containing the name(s) of the file(s) to be
      read.
    compression_type: A `Tensor` of type `string`.
      A scalar containing either (i) the empty string (no
      compression), (ii) "ZLIB", or (iii) "GZIP".
    buffer_size: A `Tensor` of type `int64`.
      A scalar representing the number of bytes to buffer. A value of
      0 means no buffering will be performed.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TFRecordDataset", name, filenames, compression_type,
        buffer_size, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tf_record_dataset_eager_fallback(
          filenames, compression_type, buffer_size, metadata=metadata,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TFRecordDataset", filenames=filenames,
                           compression_type=compression_type,
                           buffer_size=buffer_size, metadata=metadata,
                           name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("metadata", _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TFRecordDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TFRecordDataset = tf_export("raw_ops.TFRecordDataset")(_ops.to_raw_op(tf_record_dataset))


def tf_record_dataset_eager_fallback(filenames: Annotated[Any, _atypes.String], compression_type: Annotated[Any, _atypes.String], buffer_size: Annotated[Any, _atypes.Int64], metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  filenames = _ops.convert_to_tensor(filenames, _dtypes.string)
  compression_type = _ops.convert_to_tensor(compression_type, _dtypes.string)
  buffer_size = _ops.convert_to_tensor(buffer_size, _dtypes.int64)
  _inputs_flat = [filenames, compression_type, buffer_size]
  _attrs = ("metadata", metadata)
  _result = _execute.execute(b"TFRecordDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TFRecordDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def tf_record_dataset_v2(filenames: Annotated[Any, _atypes.String], compression_type: Annotated[Any, _atypes.String], buffer_size: Annotated[Any, _atypes.Int64], byte_offsets: Annotated[Any, _atypes.Int64], metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that emits the records from one or more TFRecord files.

  Args:
    filenames: A `Tensor` of type `string`.
      A scalar or vector containing the name(s) of the file(s) to be
      read.
    compression_type: A `Tensor` of type `string`.
      A scalar containing either (i) the empty string (no
      compression), (ii) "ZLIB", or (iii) "GZIP".
    buffer_size: A `Tensor` of type `int64`.
      A scalar representing the number of bytes to buffer. A value of
      0 means no buffering will be performed.
    byte_offsets: A `Tensor` of type `int64`.
      A scalar or vector containing the number of bytes for each file
      that will be skipped prior to reading.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TFRecordDatasetV2", name, filenames, compression_type,
        buffer_size, byte_offsets, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tf_record_dataset_v2_eager_fallback(
          filenames, compression_type, buffer_size, byte_offsets,
          metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TFRecordDatasetV2", filenames=filenames,
                             compression_type=compression_type,
                             buffer_size=buffer_size,
                             byte_offsets=byte_offsets, metadata=metadata,
                             name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("metadata", _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TFRecordDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TFRecordDatasetV2 = tf_export("raw_ops.TFRecordDatasetV2")(_ops.to_raw_op(tf_record_dataset_v2))


def tf_record_dataset_v2_eager_fallback(filenames: Annotated[Any, _atypes.String], compression_type: Annotated[Any, _atypes.String], buffer_size: Annotated[Any, _atypes.Int64], byte_offsets: Annotated[Any, _atypes.Int64], metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  filenames = _ops.convert_to_tensor(filenames, _dtypes.string)
  compression_type = _ops.convert_to_tensor(compression_type, _dtypes.string)
  buffer_size = _ops.convert_to_tensor(buffer_size, _dtypes.int64)
  byte_offsets = _ops.convert_to_tensor(byte_offsets, _dtypes.int64)
  _inputs_flat = [filenames, compression_type, buffer_size, byte_offsets]
  _attrs = ("metadata", metadata)
  _result = _execute.execute(b"TFRecordDatasetV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TFRecordDatasetV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def take_dataset(input_dataset: Annotated[Any, _atypes.Variant], count: Annotated[Any, _atypes.Int64], output_types, output_shapes, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that contains `count` elements from the `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    count: A `Tensor` of type `int64`.
      A scalar representing the number of elements from the `input_dataset`
      that should be taken. A value of `-1` indicates that all of `input_dataset`
      is taken.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TakeDataset", name, input_dataset, count, "output_types",
        output_types, "output_shapes", output_shapes, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return take_dataset_eager_fallback(
          input_dataset, count, output_types=output_types,
          output_shapes=output_shapes, metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'take_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'take_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TakeDataset", input_dataset=input_dataset, count=count,
                       output_types=output_types, output_shapes=output_shapes,
                       metadata=metadata, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TakeDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TakeDataset = tf_export("raw_ops.TakeDataset")(_ops.to_raw_op(take_dataset))


def take_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], count: Annotated[Any, _atypes.Int64], output_types, output_shapes, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'take_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'take_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  count = _ops.convert_to_tensor(count, _dtypes.int64)
  _inputs_flat = [input_dataset, count]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes,
  "metadata", metadata)
  _result = _execute.execute(b"TakeDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TakeDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def tensor_dataset(components, output_shapes, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that emits `components` as a tuple of tensors once.

  Args:
    components: A list of `Tensor` objects.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorDataset", name, components, "output_shapes",
        output_shapes, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_dataset_eager_fallback(
          components, output_shapes=output_shapes, metadata=metadata,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'tensor_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorDataset", components=components, output_shapes=output_shapes,
                         metadata=metadata, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Toutput_types", _op.get_attr("Toutput_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorDataset = tf_export("raw_ops.TensorDataset")(_ops.to_raw_op(tensor_dataset))


def tensor_dataset_eager_fallback(components, output_shapes, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'tensor_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _attr_Toutput_types, components = _execute.convert_to_mixed_eager_tensors(components, ctx)
  _inputs_flat = list(components)
  _attrs = ("Toutput_types", _attr_Toutput_types, "output_shapes",
  output_shapes, "metadata", metadata)
  _result = _execute.execute(b"TensorDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def tensor_slice_dataset(components, output_shapes, is_files:bool=False, metadata:str="", replicate_on_split:bool=False, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that emits each dim-0 slice of `components` once.

  Args:
    components: A list of `Tensor` objects.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    is_files: An optional `bool`. Defaults to `False`.
    metadata: An optional `string`. Defaults to `""`.
    replicate_on_split: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorSliceDataset", name, components, "output_shapes",
        output_shapes, "is_files", is_files, "metadata", metadata,
        "replicate_on_split", replicate_on_split)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_slice_dataset_eager_fallback(
          components, output_shapes=output_shapes, is_files=is_files,
          metadata=metadata, replicate_on_split=replicate_on_split, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'tensor_slice_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if is_files is None:
    is_files = False
  is_files = _execute.make_bool(is_files, "is_files")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  if replicate_on_split is None:
    replicate_on_split = False
  replicate_on_split = _execute.make_bool(replicate_on_split, "replicate_on_split")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorSliceDataset", components=components,
                              output_shapes=output_shapes, is_files=is_files,
                              metadata=metadata,
                              replicate_on_split=replicate_on_split,
                              name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Toutput_types", _op.get_attr("Toutput_types"), "output_shapes",
              _op.get_attr("output_shapes"), "is_files",
              _op._get_attr_bool("is_files"), "metadata",
              _op.get_attr("metadata"), "replicate_on_split",
              _op._get_attr_bool("replicate_on_split"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorSliceDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorSliceDataset = tf_export("raw_ops.TensorSliceDataset")(_ops.to_raw_op(tensor_slice_dataset))


def tensor_slice_dataset_eager_fallback(components, output_shapes, is_files: bool, metadata: str, replicate_on_split: bool, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'tensor_slice_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if is_files is None:
    is_files = False
  is_files = _execute.make_bool(is_files, "is_files")
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  if replicate_on_split is None:
    replicate_on_split = False
  replicate_on_split = _execute.make_bool(replicate_on_split, "replicate_on_split")
  _attr_Toutput_types, components = _execute.convert_to_mixed_eager_tensors(components, ctx)
  _inputs_flat = list(components)
  _attrs = ("Toutput_types", _attr_Toutput_types, "output_shapes",
  output_shapes, "is_files", is_files, "metadata", metadata,
  "replicate_on_split", replicate_on_split)
  _result = _execute.execute(b"TensorSliceDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorSliceDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def text_line_dataset(filenames: Annotated[Any, _atypes.String], compression_type: Annotated[Any, _atypes.String], buffer_size: Annotated[Any, _atypes.Int64], metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that emits the lines of one or more text files.

  Args:
    filenames: A `Tensor` of type `string`.
      A scalar or a vector containing the name(s) of the file(s) to be
      read.
    compression_type: A `Tensor` of type `string`.
      A scalar containing either (i) the empty string (no
      compression), (ii) "ZLIB", or (iii) "GZIP".
    buffer_size: A `Tensor` of type `int64`.
      A scalar containing the number of bytes to buffer.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TextLineDataset", name, filenames, compression_type,
        buffer_size, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return text_line_dataset_eager_fallback(
          filenames, compression_type, buffer_size, metadata=metadata,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TextLineDataset", filenames=filenames,
                           compression_type=compression_type,
                           buffer_size=buffer_size, metadata=metadata,
                           name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("metadata", _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TextLineDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TextLineDataset = tf_export("raw_ops.TextLineDataset")(_ops.to_raw_op(text_line_dataset))


def text_line_dataset_eager_fallback(filenames: Annotated[Any, _atypes.String], compression_type: Annotated[Any, _atypes.String], buffer_size: Annotated[Any, _atypes.Int64], metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  filenames = _ops.convert_to_tensor(filenames, _dtypes.string)
  compression_type = _ops.convert_to_tensor(compression_type, _dtypes.string)
  buffer_size = _ops.convert_to_tensor(buffer_size, _dtypes.int64)
  _inputs_flat = [filenames, compression_type, buffer_size]
  _attrs = ("metadata", metadata)
  _result = _execute.execute(b"TextLineDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TextLineDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def unwrap_dataset_variant(input_handle: Annotated[Any, _atypes.Variant], name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    input_handle: A `Tensor` of type `variant`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UnwrapDatasetVariant", name, input_handle)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return unwrap_dataset_variant_eager_fallback(
          input_handle, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UnwrapDatasetVariant", input_handle=input_handle, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UnwrapDatasetVariant", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

UnwrapDatasetVariant = tf_export("raw_ops.UnwrapDatasetVariant")(_ops.to_raw_op(unwrap_dataset_variant))


def unwrap_dataset_variant_eager_fallback(input_handle: Annotated[Any, _atypes.Variant], name, ctx) -> Annotated[Any, _atypes.Variant]:
  input_handle = _ops.convert_to_tensor(input_handle, _dtypes.variant)
  _inputs_flat = [input_handle]
  _attrs = None
  _result = _execute.execute(b"UnwrapDatasetVariant", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UnwrapDatasetVariant", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def window_dataset(input_dataset: Annotated[Any, _atypes.Variant], size: Annotated[Any, _atypes.Int64], shift: Annotated[Any, _atypes.Int64], stride: Annotated[Any, _atypes.Int64], drop_remainder: Annotated[Any, _atypes.Bool], output_types, output_shapes, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""  Combines (nests of) input elements into a dataset of (nests of) windows.

  A "window" is a finite dataset of flat elements of size `size` (or possibly
  fewer if there are not enough input elements to fill the window and
  `drop_remainder` evaluates to false).

  The `shift` argument determines the number of input elements by which
  the window moves on each iteration.  The first element in the `k`th window
  will be element

  ```
  1 + (k-1) * shift
  ```

  of the input dataset. In particular, the first element of the first window
  will always be the first element of the input dataset.  

  If the `stride` parameter is greater than 1, then each window will skip
  `(stride - 1)` input elements between each element that appears in the
  window. Output windows will still contain `size` elements regardless of
  the value of `stride`.

  The `stride` argument determines the stride of the input elements, and the
  `shift` argument determines the shift of the window.

  For example, letting `{...}` to represent a Dataset:

  - `tf.data.Dataset.range(7).window(2)` produces
    `{{0, 1}, {2, 3}, {4, 5}, {6}}`
  - `tf.data.Dataset.range(7).window(3, 2, 1, True)` produces
    `{{0, 1, 2}, {2, 3, 4}, {4, 5, 6}}`
  - `tf.data.Dataset.range(7).window(3, 1, 2, True)` produces
    `{{0, 2, 4}, {1, 3, 5}, {2, 4, 6}}`

  Note that when the `window` transformation is applied to a dataset of
  nested elements, it produces a dataset of nested windows.

  For example:

  - `tf.data.Dataset.from_tensor_slices((range(4), range(4))).window(2)`
    produces `{({0, 1}, {0, 1}), ({2, 3}, {2, 3})}`
  - `tf.data.Dataset.from_tensor_slices({"a": range(4)}).window(2)`
    produces `{{"a": {0, 1}}, {"a": {2, 3}}}`

  Args:
    input_dataset: A `Tensor` of type `variant`.
    size: A `Tensor` of type `int64`.
      An integer scalar, representing the number of elements
      of the input dataset to combine into a window. Must be positive.
    shift: A `Tensor` of type `int64`.
      An integer scalar, representing the number of input elements
      by which the window moves in each iteration.  Defaults to `size`.
      Must be positive.
    stride: A `Tensor` of type `int64`.
      An integer scalar, representing the stride of the input elements
      in the sliding window. Must be positive. The default value of 1 means
      "retain every input element".
    drop_remainder: A `Tensor` of type `bool`.
      A Boolean scalar, representing whether the last window should be
      dropped if its size is smaller than `window_size`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "WindowDataset", name, input_dataset, size, shift, stride,
        drop_remainder, "output_types", output_types, "output_shapes",
        output_shapes, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return window_dataset_eager_fallback(
          input_dataset, size, shift, stride, drop_remainder,
          output_types=output_types, output_shapes=output_shapes,
          metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'window_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'window_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "WindowDataset", input_dataset=input_dataset, size=size, shift=shift,
                         stride=stride, drop_remainder=drop_remainder,
                         output_types=output_types,
                         output_shapes=output_shapes, metadata=metadata,
                         name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "metadata",
              _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "WindowDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

WindowDataset = tf_export("raw_ops.WindowDataset")(_ops.to_raw_op(window_dataset))


def window_dataset_eager_fallback(input_dataset: Annotated[Any, _atypes.Variant], size: Annotated[Any, _atypes.Int64], shift: Annotated[Any, _atypes.Int64], stride: Annotated[Any, _atypes.Int64], drop_remainder: Annotated[Any, _atypes.Bool], output_types, output_shapes, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'window_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'window_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  size = _ops.convert_to_tensor(size, _dtypes.int64)
  shift = _ops.convert_to_tensor(shift, _dtypes.int64)
  stride = _ops.convert_to_tensor(stride, _dtypes.int64)
  drop_remainder = _ops.convert_to_tensor(drop_remainder, _dtypes.bool)
  _inputs_flat = [input_dataset, size, shift, stride, drop_remainder]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes,
  "metadata", metadata)
  _result = _execute.execute(b"WindowDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "WindowDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def window_op(inputs, output_types, output_shapes, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    inputs: A list of `Tensor` objects.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "WindowOp", name, inputs, "output_types", output_types,
        "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return window_op_eager_fallback(
          inputs, output_types=output_types, output_shapes=output_shapes,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'window_op' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'window_op' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "WindowOp", inputs=inputs, output_types=output_types,
                    output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "Tinputs",
              _op.get_attr("Tinputs"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "WindowOp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

WindowOp = tf_export("raw_ops.WindowOp")(_ops.to_raw_op(window_op))


def window_op_eager_fallback(inputs, output_types, output_shapes, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'window_op' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'window_op' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _attr_Tinputs, inputs = _execute.convert_to_mixed_eager_tensors(inputs, ctx)
  _inputs_flat = list(inputs)
  _attrs = ("output_types", output_types, "output_shapes", output_shapes,
  "Tinputs", _attr_Tinputs)
  _result = _execute.execute(b"WindowOp", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "WindowOp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def wrap_dataset_variant(input_handle: Annotated[Any, _atypes.Variant], name=None) -> Annotated[Any, _atypes.Variant]:
  r"""TODO: add doc.

  Args:
    input_handle: A `Tensor` of type `variant`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "WrapDatasetVariant", name, input_handle)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return wrap_dataset_variant_eager_fallback(
          input_handle, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "WrapDatasetVariant", input_handle=input_handle, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "WrapDatasetVariant", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

WrapDatasetVariant = tf_export("raw_ops.WrapDatasetVariant")(_ops.to_raw_op(wrap_dataset_variant))


def wrap_dataset_variant_eager_fallback(input_handle: Annotated[Any, _atypes.Variant], name, ctx) -> Annotated[Any, _atypes.Variant]:
  input_handle = _ops.convert_to_tensor(input_handle, _dtypes.variant)
  _inputs_flat = [input_handle]
  _attrs = None
  _result = _execute.execute(b"WrapDatasetVariant", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "WrapDatasetVariant", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def zip_dataset(input_datasets: Annotated[List[Any], _atypes.Variant], output_types, output_shapes, metadata:str="", name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates a dataset that zips together `input_datasets`.

  The elements of the resulting dataset are created by zipping corresponding
  elements from each of the input datasets.

  The size of the resulting dataset will match the size of the smallest input
  dataset, and no error will be raised if input datasets have different sizes.

  Args:
    input_datasets: A list of at least 1 `Tensor` objects with type `variant`.
      List of `N` variant Tensors representing datasets to be zipped together.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ZipDataset", name, input_datasets, "output_types",
        output_types, "output_shapes", output_shapes, "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return zip_dataset_eager_fallback(
          input_datasets, output_types=output_types,
          output_shapes=output_shapes, metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(input_datasets, (list, tuple)):
    raise TypeError(
        "Expected list for 'input_datasets' argument to "
        "'zip_dataset' Op, not %r." % input_datasets)
  _attr_N = len(input_datasets)
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'zip_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'zip_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ZipDataset", input_datasets=input_datasets,
                      output_types=output_types, output_shapes=output_shapes,
                      metadata=metadata, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "N", _op._get_attr_int("N"),
              "metadata", _op.get_attr("metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ZipDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ZipDataset = tf_export("raw_ops.ZipDataset")(_ops.to_raw_op(zip_dataset))


def zip_dataset_eager_fallback(input_datasets: Annotated[List[Any], _atypes.Variant], output_types, output_shapes, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  if not isinstance(input_datasets, (list, tuple)):
    raise TypeError(
        "Expected list for 'input_datasets' argument to "
        "'zip_dataset' Op, not %r." % input_datasets)
  _attr_N = len(input_datasets)
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'zip_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'zip_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if metadata is None:
    metadata = ""
  metadata = _execute.make_str(metadata, "metadata")
  input_datasets = _ops.convert_n_to_tensor(input_datasets, _dtypes.variant)
  _inputs_flat = list(input_datasets)
  _attrs = ("output_types", output_types, "output_shapes", output_shapes, "N",
  _attr_N, "metadata", metadata)
  _result = _execute.execute(b"ZipDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ZipDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

