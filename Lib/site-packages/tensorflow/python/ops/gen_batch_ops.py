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
_BatchOutput = collections.namedtuple(
    "Batch",
    ["batched_tensors", "batch_index", "id"])


def batch(in_tensors, num_batch_threads: int, max_batch_size: int, batch_timeout_micros: int, grad_timeout_micros: int, max_enqueued_batches:int=10, allowed_batch_sizes=[], container:str="", shared_name:str="", batching_queue:str="", name=None):
  r"""Batches all input tensors nondeterministically.

  When many instances of this Op are being run concurrently with the same
  container/shared_name in the same device, some will output zero-shaped Tensors
  and others will output Tensors of size up to max_batch_size.

  All Tensors in in_tensors are batched together (so, for example, labels and
  features should be batched with a single instance of this operation.

  Each invocation of batch emits an `id` scalar which will be used to identify
  this particular invocation when doing unbatch or its gradient.

  Each op which emits a non-empty batch will also emit a non-empty batch_index
  Tensor, which, is a [K, 3] matrix where each row contains the invocation's id,
  start, and length of elements of each set of Tensors present in batched_tensors.

  Batched tensors are concatenated along the first dimension, and all tensors in
  in_tensors must have the first dimension of the same size.

  in_tensors: The tensors to be batched.
  num_batch_threads: Number of scheduling threads for processing batches of work.
   Determines the number of batches processed in parallel.
  max_batch_size: Batch sizes will never be bigger than this.
  batch_timeout_micros: Maximum number of microseconds to wait before outputting
   an incomplete batch.
  allowed_batch_sizes: Optional list of allowed batch sizes. If left empty, does
   nothing. Otherwise, supplies a list of batch sizes, causing the op to pad
   batches up to one of those sizes. The entries must increase monotonically, and
   the final entry must equal max_batch_size.
  grad_timeout_micros: The timeout to use for the gradient. See Unbatch.
  batched_tensors: Either empty tensors or a batch of concatenated Tensors.
  batch_index: If out_tensors is non-empty, has information to invert it.
  container: Controls the scope of sharing of this batch.
  id: always contains a scalar with a unique ID for this invocation of Batch.
  shared_name: Concurrently running instances of batch in the same device with the
   same container and shared_name will batch their elements together. If left
   empty, the op name will be used as the shared name.
  T: the types of tensors to be batched.

  Args:
    in_tensors: A list of `Tensor` objects.
    num_batch_threads: An `int`.
    max_batch_size: An `int`.
    batch_timeout_micros: An `int`.
    grad_timeout_micros: An `int`.
    max_enqueued_batches: An optional `int`. Defaults to `10`.
    allowed_batch_sizes: An optional list of `ints`. Defaults to `[]`.
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    batching_queue: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (batched_tensors, batch_index, id).

    batched_tensors: A list of `Tensor` objects. Has the same type as `in_tensors`.
    batch_index: A `Tensor` of type `int64`.
    id: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Batch", name, in_tensors, "num_batch_threads",
        num_batch_threads, "max_batch_size", max_batch_size,
        "max_enqueued_batches", max_enqueued_batches, "batch_timeout_micros",
        batch_timeout_micros, "allowed_batch_sizes", allowed_batch_sizes,
        "grad_timeout_micros", grad_timeout_micros, "container", container,
        "shared_name", shared_name, "batching_queue", batching_queue)
      _result = _BatchOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return batch_eager_fallback(
          in_tensors, num_batch_threads=num_batch_threads,
          max_batch_size=max_batch_size,
          max_enqueued_batches=max_enqueued_batches,
          batch_timeout_micros=batch_timeout_micros,
          allowed_batch_sizes=allowed_batch_sizes,
          grad_timeout_micros=grad_timeout_micros, container=container,
          shared_name=shared_name, batching_queue=batching_queue, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_batch_threads = _execute.make_int(num_batch_threads, "num_batch_threads")
  max_batch_size = _execute.make_int(max_batch_size, "max_batch_size")
  batch_timeout_micros = _execute.make_int(batch_timeout_micros, "batch_timeout_micros")
  grad_timeout_micros = _execute.make_int(grad_timeout_micros, "grad_timeout_micros")
  if max_enqueued_batches is None:
    max_enqueued_batches = 10
  max_enqueued_batches = _execute.make_int(max_enqueued_batches, "max_enqueued_batches")
  if allowed_batch_sizes is None:
    allowed_batch_sizes = []
  if not isinstance(allowed_batch_sizes, (list, tuple)):
    raise TypeError(
        "Expected list for 'allowed_batch_sizes' argument to "
        "'batch' Op, not %r." % allowed_batch_sizes)
  allowed_batch_sizes = [_execute.make_int(_i, "allowed_batch_sizes") for _i in allowed_batch_sizes]
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if batching_queue is None:
    batching_queue = ""
  batching_queue = _execute.make_str(batching_queue, "batching_queue")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Batch", in_tensors=in_tensors, num_batch_threads=num_batch_threads,
                 max_batch_size=max_batch_size,
                 batch_timeout_micros=batch_timeout_micros,
                 grad_timeout_micros=grad_timeout_micros,
                 max_enqueued_batches=max_enqueued_batches,
                 allowed_batch_sizes=allowed_batch_sizes, container=container,
                 shared_name=shared_name, batching_queue=batching_queue,
                 name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("num_batch_threads", _op._get_attr_int("num_batch_threads"),
              "max_batch_size", _op._get_attr_int("max_batch_size"),
              "max_enqueued_batches",
              _op._get_attr_int("max_enqueued_batches"),
              "batch_timeout_micros",
              _op._get_attr_int("batch_timeout_micros"),
              "allowed_batch_sizes", _op.get_attr("allowed_batch_sizes"),
              "grad_timeout_micros", _op._get_attr_int("grad_timeout_micros"),
              "container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"), "batching_queue",
              _op.get_attr("batching_queue"), "T", _op.get_attr("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Batch", _inputs_flat, _attrs, _result)
  _result = [_result[:len(in_tensors)]] + _result[len(in_tensors):]
  _result = _BatchOutput._make(_result)
  return _result

Batch = tf_export("raw_ops.Batch")(_ops.to_raw_op(batch))


def batch_eager_fallback(in_tensors, num_batch_threads: int, max_batch_size: int, batch_timeout_micros: int, grad_timeout_micros: int, max_enqueued_batches: int, allowed_batch_sizes, container: str, shared_name: str, batching_queue: str, name, ctx):
  num_batch_threads = _execute.make_int(num_batch_threads, "num_batch_threads")
  max_batch_size = _execute.make_int(max_batch_size, "max_batch_size")
  batch_timeout_micros = _execute.make_int(batch_timeout_micros, "batch_timeout_micros")
  grad_timeout_micros = _execute.make_int(grad_timeout_micros, "grad_timeout_micros")
  if max_enqueued_batches is None:
    max_enqueued_batches = 10
  max_enqueued_batches = _execute.make_int(max_enqueued_batches, "max_enqueued_batches")
  if allowed_batch_sizes is None:
    allowed_batch_sizes = []
  if not isinstance(allowed_batch_sizes, (list, tuple)):
    raise TypeError(
        "Expected list for 'allowed_batch_sizes' argument to "
        "'batch' Op, not %r." % allowed_batch_sizes)
  allowed_batch_sizes = [_execute.make_int(_i, "allowed_batch_sizes") for _i in allowed_batch_sizes]
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if batching_queue is None:
    batching_queue = ""
  batching_queue = _execute.make_str(batching_queue, "batching_queue")
  _attr_T, in_tensors = _execute.convert_to_mixed_eager_tensors(in_tensors, ctx)
  _inputs_flat = list(in_tensors)
  _attrs = ("num_batch_threads", num_batch_threads, "max_batch_size",
  max_batch_size, "max_enqueued_batches", max_enqueued_batches,
  "batch_timeout_micros", batch_timeout_micros, "allowed_batch_sizes",
  allowed_batch_sizes, "grad_timeout_micros", grad_timeout_micros,
  "container", container, "shared_name", shared_name, "batching_queue",
  batching_queue, "T", _attr_T)
  _result = _execute.execute(b"Batch", len(in_tensors) + 2,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Batch", _inputs_flat, _attrs, _result)
  _result = [_result[:len(in_tensors)]] + _result[len(in_tensors):]
  _result = _BatchOutput._make(_result)
  return _result


def batch_function(in_tensors, captured_tensors, f, num_batch_threads: int, max_batch_size: int, batch_timeout_micros: int, Tout, max_enqueued_batches:int=10, allowed_batch_sizes=[], container:str="", shared_name:str="", batching_queue:str="", low_priority_max_batch_size:int=0, low_priority_batch_timeout_micros:int=0, low_priority_allowed_batch_sizes=[], low_priority_max_enqueued_batches:int=0, mixed_priority_policy:str="low_priority_padding_with_max_batch_size", batch_padding_policy:str="PAD_UP", enable_large_batch_splitting:bool=False, name=None):
  r"""Batches all the inputs tensors to the computation done by the function.

  So, for example, in the following code

    ```python

    # This input will be captured.
    y = tf.placeholder_with_default(1.0, shape=[])

    @tf.Defun(tf.float32)
    def computation(a):
      return tf.matmul(a, a) + y

    b = gen_batch_ops.batch_function(
            f=computation
            in_tensors=[a],
            captured_tensors=computation.captured_inputs,
            Tout=[o.type for o in computation.definition.signature.output_arg],
            num_batch_threads=1,
            max_batch_size=10,
            batch_timeout_micros=100000,  # 100ms
            allowed_batch_sizes=[3, 10],
            batching_queue="")
    ```

  If more than one session.run call is simultaneously trying to compute `b`
  the values of `a` will be gathered, non-deterministically concatenated
  along the first axis, and only one thread will run the computation.

  Assumes that all arguments of the function are Tensors which will be batched
  along their first dimension.

  Arguments that are captured, are not batched. The session.run call which does
  the concatenation, will use the values of the captured tensors available to it.
  Therefore, typical uses of captured tensors should involve values which remain
  unchanged across session.run calls. Inference is a good example of this.

  SparseTensor is not supported. The return value of the decorated function
  must be a Tensor or a list/tuple of Tensors.

  Args:
    in_tensors: A list of `Tensor` objects. The tensors to be batched.
    captured_tensors: A list of `Tensor` objects.
      The tensors which are captured in the function, and don't need
      to be batched.
    f: A function decorated with @Defun.
    num_batch_threads: An `int`.
      Number of scheduling threads for processing batches of work.
      Determines the number of batches processed in parallel.
    max_batch_size: An `int`. Batch sizes will never be bigger than this.
    batch_timeout_micros: An `int`.
      Maximum number of microseconds to wait before outputting
      an incomplete batch.
    Tout: A list of `tf.DTypes` that has length `>= 1`.
      the types of the output tensors.
    max_enqueued_batches: An optional `int`. Defaults to `10`.
      Maximum number of batches enqueued. Default: 10.
    allowed_batch_sizes: An optional list of `ints`. Defaults to `[]`.
      Optional list of allowed batch sizes. If left empty, does
      nothing. Otherwise, supplies a list of batch sizes, causing the op to pad
      batches up to one of those sizes. The entries must increase monotonically.
      If enable_large_batch_splitting is false (i.e., large-input-split is not
      enabled) the final entry must equal max_batch_size.
    container: An optional `string`. Defaults to `""`.
      Controls the scope of sharing of this batch.
    shared_name: An optional `string`. Defaults to `""`.
      Concurrently running instances of batch in the same device with the
      same container and shared_name will batch their elements together. If left
      empty, the op name will be used as the shared name.
    batching_queue: An optional `string`. Defaults to `""`.
    low_priority_max_batch_size: An optional `int`. Defaults to `0`.
    low_priority_batch_timeout_micros: An optional `int`. Defaults to `0`.
    low_priority_allowed_batch_sizes: An optional list of `ints`. Defaults to `[]`.
    low_priority_max_enqueued_batches: An optional `int`. Defaults to `0`.
    mixed_priority_policy: An optional `string` from: `"low_priority_padding_with_max_batch_size", "low_priority_padding_with_next_allowed_batch_size", "priority_isolation"`. Defaults to `"low_priority_padding_with_max_batch_size"`.
    batch_padding_policy: An optional `string` from: `"PAD_UP", "BATCH_DOWN", "MINIMIZE_TPU_COST_PER_REQUEST"`. Defaults to `"PAD_UP"`.
    enable_large_batch_splitting: An optional `bool`. Defaults to `False`.
      input with a large size (i.e., larger than the largest value of
      `allowed_batch_sizes`) will be splitted into multiple batches with batch size.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BatchFunction", name, in_tensors, captured_tensors, "f", f,
        "num_batch_threads", num_batch_threads, "max_batch_size",
        max_batch_size, "batch_timeout_micros", batch_timeout_micros,
        "max_enqueued_batches", max_enqueued_batches, "allowed_batch_sizes",
        allowed_batch_sizes, "container", container, "shared_name",
        shared_name, "batching_queue", batching_queue,
        "low_priority_max_batch_size", low_priority_max_batch_size,
        "low_priority_batch_timeout_micros",
        low_priority_batch_timeout_micros, "low_priority_allowed_batch_sizes",
        low_priority_allowed_batch_sizes, "low_priority_max_enqueued_batches",
        low_priority_max_enqueued_batches, "mixed_priority_policy",
        mixed_priority_policy, "batch_padding_policy", batch_padding_policy,
        "Tout", Tout, "enable_large_batch_splitting",
        enable_large_batch_splitting)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return batch_function_eager_fallback(
          in_tensors, captured_tensors, f=f,
          num_batch_threads=num_batch_threads, max_batch_size=max_batch_size,
          batch_timeout_micros=batch_timeout_micros,
          max_enqueued_batches=max_enqueued_batches,
          allowed_batch_sizes=allowed_batch_sizes, container=container,
          shared_name=shared_name, batching_queue=batching_queue,
          low_priority_max_batch_size=low_priority_max_batch_size,
          low_priority_batch_timeout_micros=low_priority_batch_timeout_micros,
          low_priority_allowed_batch_sizes=low_priority_allowed_batch_sizes,
          low_priority_max_enqueued_batches=low_priority_max_enqueued_batches,
          mixed_priority_policy=mixed_priority_policy,
          batch_padding_policy=batch_padding_policy, Tout=Tout,
          enable_large_batch_splitting=enable_large_batch_splitting,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_batch_threads = _execute.make_int(num_batch_threads, "num_batch_threads")
  max_batch_size = _execute.make_int(max_batch_size, "max_batch_size")
  batch_timeout_micros = _execute.make_int(batch_timeout_micros, "batch_timeout_micros")
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'batch_function' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  if max_enqueued_batches is None:
    max_enqueued_batches = 10
  max_enqueued_batches = _execute.make_int(max_enqueued_batches, "max_enqueued_batches")
  if allowed_batch_sizes is None:
    allowed_batch_sizes = []
  if not isinstance(allowed_batch_sizes, (list, tuple)):
    raise TypeError(
        "Expected list for 'allowed_batch_sizes' argument to "
        "'batch_function' Op, not %r." % allowed_batch_sizes)
  allowed_batch_sizes = [_execute.make_int(_i, "allowed_batch_sizes") for _i in allowed_batch_sizes]
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if batching_queue is None:
    batching_queue = ""
  batching_queue = _execute.make_str(batching_queue, "batching_queue")
  if low_priority_max_batch_size is None:
    low_priority_max_batch_size = 0
  low_priority_max_batch_size = _execute.make_int(low_priority_max_batch_size, "low_priority_max_batch_size")
  if low_priority_batch_timeout_micros is None:
    low_priority_batch_timeout_micros = 0
  low_priority_batch_timeout_micros = _execute.make_int(low_priority_batch_timeout_micros, "low_priority_batch_timeout_micros")
  if low_priority_allowed_batch_sizes is None:
    low_priority_allowed_batch_sizes = []
  if not isinstance(low_priority_allowed_batch_sizes, (list, tuple)):
    raise TypeError(
        "Expected list for 'low_priority_allowed_batch_sizes' argument to "
        "'batch_function' Op, not %r." % low_priority_allowed_batch_sizes)
  low_priority_allowed_batch_sizes = [_execute.make_int(_i, "low_priority_allowed_batch_sizes") for _i in low_priority_allowed_batch_sizes]
  if low_priority_max_enqueued_batches is None:
    low_priority_max_enqueued_batches = 0
  low_priority_max_enqueued_batches = _execute.make_int(low_priority_max_enqueued_batches, "low_priority_max_enqueued_batches")
  if mixed_priority_policy is None:
    mixed_priority_policy = "low_priority_padding_with_max_batch_size"
  mixed_priority_policy = _execute.make_str(mixed_priority_policy, "mixed_priority_policy")
  if batch_padding_policy is None:
    batch_padding_policy = "PAD_UP"
  batch_padding_policy = _execute.make_str(batch_padding_policy, "batch_padding_policy")
  if enable_large_batch_splitting is None:
    enable_large_batch_splitting = False
  enable_large_batch_splitting = _execute.make_bool(enable_large_batch_splitting, "enable_large_batch_splitting")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BatchFunction", in_tensors=in_tensors,
                         captured_tensors=captured_tensors, f=f,
                         num_batch_threads=num_batch_threads,
                         max_batch_size=max_batch_size,
                         batch_timeout_micros=batch_timeout_micros, Tout=Tout,
                         max_enqueued_batches=max_enqueued_batches,
                         allowed_batch_sizes=allowed_batch_sizes,
                         container=container, shared_name=shared_name,
                         batching_queue=batching_queue,
                         low_priority_max_batch_size=low_priority_max_batch_size,
                         low_priority_batch_timeout_micros=low_priority_batch_timeout_micros,
                         low_priority_allowed_batch_sizes=low_priority_allowed_batch_sizes,
                         low_priority_max_enqueued_batches=low_priority_max_enqueued_batches,
                         mixed_priority_policy=mixed_priority_policy,
                         batch_padding_policy=batch_padding_policy,
                         enable_large_batch_splitting=enable_large_batch_splitting,
                         name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("f", _op.get_attr("f"), "num_batch_threads",
              _op._get_attr_int("num_batch_threads"), "max_batch_size",
              _op._get_attr_int("max_batch_size"), "batch_timeout_micros",
              _op._get_attr_int("batch_timeout_micros"),
              "max_enqueued_batches",
              _op._get_attr_int("max_enqueued_batches"),
              "allowed_batch_sizes", _op.get_attr("allowed_batch_sizes"),
              "container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"), "batching_queue",
              _op.get_attr("batching_queue"), "low_priority_max_batch_size",
              _op._get_attr_int("low_priority_max_batch_size"),
              "low_priority_batch_timeout_micros",
              _op._get_attr_int("low_priority_batch_timeout_micros"),
              "low_priority_allowed_batch_sizes",
              _op.get_attr("low_priority_allowed_batch_sizes"),
              "low_priority_max_enqueued_batches",
              _op._get_attr_int("low_priority_max_enqueued_batches"),
              "mixed_priority_policy", _op.get_attr("mixed_priority_policy"),
              "batch_padding_policy", _op.get_attr("batch_padding_policy"),
              "Tin", _op.get_attr("Tin"), "Tcaptured",
              _op.get_attr("Tcaptured"), "Tout", _op.get_attr("Tout"),
              "enable_large_batch_splitting",
              _op._get_attr_bool("enable_large_batch_splitting"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BatchFunction", _inputs_flat, _attrs, _result)
  return _result

BatchFunction = tf_export("raw_ops.BatchFunction")(_ops.to_raw_op(batch_function))


def batch_function_eager_fallback(in_tensors, captured_tensors, f, num_batch_threads: int, max_batch_size: int, batch_timeout_micros: int, Tout, max_enqueued_batches: int, allowed_batch_sizes, container: str, shared_name: str, batching_queue: str, low_priority_max_batch_size: int, low_priority_batch_timeout_micros: int, low_priority_allowed_batch_sizes, low_priority_max_enqueued_batches: int, mixed_priority_policy: str, batch_padding_policy: str, enable_large_batch_splitting: bool, name, ctx):
  num_batch_threads = _execute.make_int(num_batch_threads, "num_batch_threads")
  max_batch_size = _execute.make_int(max_batch_size, "max_batch_size")
  batch_timeout_micros = _execute.make_int(batch_timeout_micros, "batch_timeout_micros")
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'batch_function' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  if max_enqueued_batches is None:
    max_enqueued_batches = 10
  max_enqueued_batches = _execute.make_int(max_enqueued_batches, "max_enqueued_batches")
  if allowed_batch_sizes is None:
    allowed_batch_sizes = []
  if not isinstance(allowed_batch_sizes, (list, tuple)):
    raise TypeError(
        "Expected list for 'allowed_batch_sizes' argument to "
        "'batch_function' Op, not %r." % allowed_batch_sizes)
  allowed_batch_sizes = [_execute.make_int(_i, "allowed_batch_sizes") for _i in allowed_batch_sizes]
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if batching_queue is None:
    batching_queue = ""
  batching_queue = _execute.make_str(batching_queue, "batching_queue")
  if low_priority_max_batch_size is None:
    low_priority_max_batch_size = 0
  low_priority_max_batch_size = _execute.make_int(low_priority_max_batch_size, "low_priority_max_batch_size")
  if low_priority_batch_timeout_micros is None:
    low_priority_batch_timeout_micros = 0
  low_priority_batch_timeout_micros = _execute.make_int(low_priority_batch_timeout_micros, "low_priority_batch_timeout_micros")
  if low_priority_allowed_batch_sizes is None:
    low_priority_allowed_batch_sizes = []
  if not isinstance(low_priority_allowed_batch_sizes, (list, tuple)):
    raise TypeError(
        "Expected list for 'low_priority_allowed_batch_sizes' argument to "
        "'batch_function' Op, not %r." % low_priority_allowed_batch_sizes)
  low_priority_allowed_batch_sizes = [_execute.make_int(_i, "low_priority_allowed_batch_sizes") for _i in low_priority_allowed_batch_sizes]
  if low_priority_max_enqueued_batches is None:
    low_priority_max_enqueued_batches = 0
  low_priority_max_enqueued_batches = _execute.make_int(low_priority_max_enqueued_batches, "low_priority_max_enqueued_batches")
  if mixed_priority_policy is None:
    mixed_priority_policy = "low_priority_padding_with_max_batch_size"
  mixed_priority_policy = _execute.make_str(mixed_priority_policy, "mixed_priority_policy")
  if batch_padding_policy is None:
    batch_padding_policy = "PAD_UP"
  batch_padding_policy = _execute.make_str(batch_padding_policy, "batch_padding_policy")
  if enable_large_batch_splitting is None:
    enable_large_batch_splitting = False
  enable_large_batch_splitting = _execute.make_bool(enable_large_batch_splitting, "enable_large_batch_splitting")
  _attr_Tin, in_tensors = _execute.convert_to_mixed_eager_tensors(in_tensors, ctx)
  _attr_Tcaptured, captured_tensors = _execute.convert_to_mixed_eager_tensors(captured_tensors, ctx)
  _inputs_flat = list(in_tensors) + list(captured_tensors)
  _attrs = ("f", f, "num_batch_threads", num_batch_threads, "max_batch_size",
  max_batch_size, "batch_timeout_micros", batch_timeout_micros,
  "max_enqueued_batches", max_enqueued_batches, "allowed_batch_sizes",
  allowed_batch_sizes, "container", container, "shared_name", shared_name,
  "batching_queue", batching_queue, "low_priority_max_batch_size",
  low_priority_max_batch_size, "low_priority_batch_timeout_micros",
  low_priority_batch_timeout_micros, "low_priority_allowed_batch_sizes",
  low_priority_allowed_batch_sizes, "low_priority_max_enqueued_batches",
  low_priority_max_enqueued_batches, "mixed_priority_policy",
  mixed_priority_policy, "batch_padding_policy", batch_padding_policy, "Tin",
  _attr_Tin, "Tcaptured", _attr_Tcaptured, "Tout", Tout,
  "enable_large_batch_splitting", enable_large_batch_splitting)
  _result = _execute.execute(b"BatchFunction", len(Tout), inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BatchFunction", _inputs_flat, _attrs, _result)
  return _result


TV_Unbatch_T = TypeVar("TV_Unbatch_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def unbatch(batched_tensor: Annotated[Any, TV_Unbatch_T], batch_index: Annotated[Any, _atypes.Int64], id: Annotated[Any, _atypes.Int64], timeout_micros: int, container:str="", shared_name:str="", name=None) -> Annotated[Any, TV_Unbatch_T]:
  r"""Reverses the operation of Batch for a single output Tensor.

  An instance of Unbatch either receives an empty batched_tensor, in which case it
  asynchronously waits until the values become available from a concurrently
  running instance of Unbatch with the same container and shared_name, or receives
  a non-empty batched_tensor in which case it finalizes all other concurrently
  running instances and outputs its own element from the batch.

  batched_tensor: The possibly transformed output of Batch. The size of the first
   dimension should remain unchanged by the transformations for the operation to
   work.
  batch_index: The matching batch_index obtained from Batch.
  id: The id scalar emitted by Batch.
  unbatched_tensor: The Tensor corresponding to this execution.
  timeout_micros: Maximum amount of time (in microseconds) to wait to receive the
   batched input tensor associated with a given invocation of the op.
  container: Container to control resource sharing.
  shared_name: Instances of Unbatch with the same container and shared_name are
   assumed to possibly belong to the same batch. If left empty, the op name will
   be used as the shared name.

  Args:
    batched_tensor: A `Tensor`.
    batch_index: A `Tensor` of type `int64`.
    id: A `Tensor` of type `int64`.
    timeout_micros: An `int`.
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `batched_tensor`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Unbatch", name, batched_tensor, batch_index, id,
        "timeout_micros", timeout_micros, "container", container,
        "shared_name", shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return unbatch_eager_fallback(
          batched_tensor, batch_index, id, timeout_micros=timeout_micros,
          container=container, shared_name=shared_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  timeout_micros = _execute.make_int(timeout_micros, "timeout_micros")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Unbatch", batched_tensor=batched_tensor, batch_index=batch_index,
                   id=id, timeout_micros=timeout_micros, container=container,
                   shared_name=shared_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("timeout_micros", _op._get_attr_int("timeout_micros"),
              "container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"), "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Unbatch", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Unbatch = tf_export("raw_ops.Unbatch")(_ops.to_raw_op(unbatch))


def unbatch_eager_fallback(batched_tensor: Annotated[Any, TV_Unbatch_T], batch_index: Annotated[Any, _atypes.Int64], id: Annotated[Any, _atypes.Int64], timeout_micros: int, container: str, shared_name: str, name, ctx) -> Annotated[Any, TV_Unbatch_T]:
  timeout_micros = _execute.make_int(timeout_micros, "timeout_micros")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _attr_T, (batched_tensor,) = _execute.args_to_matching_eager([batched_tensor], ctx, [])
  batch_index = _ops.convert_to_tensor(batch_index, _dtypes.int64)
  id = _ops.convert_to_tensor(id, _dtypes.int64)
  _inputs_flat = [batched_tensor, batch_index, id]
  _attrs = ("timeout_micros", timeout_micros, "container", container,
  "shared_name", shared_name, "T", _attr_T)
  _result = _execute.execute(b"Unbatch", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Unbatch", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_UnbatchGrad_T = TypeVar("TV_UnbatchGrad_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def unbatch_grad(original_input: Annotated[Any, TV_UnbatchGrad_T], batch_index: Annotated[Any, _atypes.Int64], grad: Annotated[Any, TV_UnbatchGrad_T], id: Annotated[Any, _atypes.Int64], container:str="", shared_name:str="", name=None) -> Annotated[Any, TV_UnbatchGrad_T]:
  r"""Gradient of Unbatch.

  Acts like Batch but using the given batch_index index of batching things as they
  become available. This ensures that the gradients are propagated back in the
  same session which did the forward pass.

  original_input: The input to the Unbatch operation this is the gradient of.
  batch_index: The batch_index given to the Unbatch operation this is the gradient
  of.
  grad: The downstream gradient.
  id: The id scalar emitted by Batch.
  batched_grad: The return value, either an empty tensor or the batched gradient.
  container: Container to control resource sharing.
  shared_name: Instances of UnbatchGrad with the same container and shared_name
   are assumed to possibly belong to the same batch. If left empty, the op name
   will be used as the shared name.

  Args:
    original_input: A `Tensor`.
    batch_index: A `Tensor` of type `int64`.
    grad: A `Tensor`. Must have the same type as `original_input`.
    id: A `Tensor` of type `int64`.
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `original_input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UnbatchGrad", name, original_input, batch_index, grad, id,
        "container", container, "shared_name", shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return unbatch_grad_eager_fallback(
          original_input, batch_index, grad, id, container=container,
          shared_name=shared_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UnbatchGrad", original_input=original_input, batch_index=batch_index,
                       grad=grad, id=id, container=container,
                       shared_name=shared_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"), "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UnbatchGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

UnbatchGrad = tf_export("raw_ops.UnbatchGrad")(_ops.to_raw_op(unbatch_grad))


def unbatch_grad_eager_fallback(original_input: Annotated[Any, TV_UnbatchGrad_T], batch_index: Annotated[Any, _atypes.Int64], grad: Annotated[Any, TV_UnbatchGrad_T], id: Annotated[Any, _atypes.Int64], container: str, shared_name: str, name, ctx) -> Annotated[Any, TV_UnbatchGrad_T]:
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([original_input, grad], ctx, [])
  (original_input, grad) = _inputs_T
  batch_index = _ops.convert_to_tensor(batch_index, _dtypes.int64)
  id = _ops.convert_to_tensor(id, _dtypes.int64)
  _inputs_flat = [original_input, batch_index, grad, id]
  _attrs = ("container", container, "shared_name", shared_name, "T", _attr_T)
  _result = _execute.execute(b"UnbatchGrad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UnbatchGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

