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

TV_ApplyAdaMax_T = TypeVar("TV_ApplyAdaMax_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def apply_ada_max(var: Annotated[Any, TV_ApplyAdaMax_T], m: Annotated[Any, TV_ApplyAdaMax_T], v: Annotated[Any, TV_ApplyAdaMax_T], beta1_power: Annotated[Any, TV_ApplyAdaMax_T], lr: Annotated[Any, TV_ApplyAdaMax_T], beta1: Annotated[Any, TV_ApplyAdaMax_T], beta2: Annotated[Any, TV_ApplyAdaMax_T], epsilon: Annotated[Any, TV_ApplyAdaMax_T], grad: Annotated[Any, TV_ApplyAdaMax_T], use_locking:bool=False, name=None) -> Annotated[Any, TV_ApplyAdaMax_T]:
  r"""Update '*var' according to the AdaMax algorithm.

  m_t <- beta1 * m_{t-1} + (1 - beta1) * g
  v_t <- max(beta2 * v_{t-1}, abs(g))
  variable <- variable - learning_rate / (1 - beta1^t) * m_t / (v_t + epsilon)

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Should be from a Variable().
    m: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    v: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    beta1_power: A `Tensor`. Must have the same type as `var`.
      Must be a scalar.
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    beta1: A `Tensor`. Must have the same type as `var`.
      Momentum factor. Must be a scalar.
    beta2: A `Tensor`. Must have the same type as `var`.
      Momentum factor. Must be a scalar.
    epsilon: A `Tensor`. Must have the same type as `var`.
      Ridge term. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var, m, and v tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("apply_ada_max op does not support eager execution. Arg 'out' is a ref.")
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ApplyAdaMax", var=var, m=m, v=v, beta1_power=beta1_power, lr=lr,
                       beta1=beta1, beta2=beta2, epsilon=epsilon, grad=grad,
                       use_locking=use_locking, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "use_locking",
              _op._get_attr_bool("use_locking"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ApplyAdaMax", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ApplyAdaMax = tf_export("raw_ops.ApplyAdaMax")(_ops.to_raw_op(apply_ada_max))


def apply_ada_max_eager_fallback(var: Annotated[Any, TV_ApplyAdaMax_T], m: Annotated[Any, TV_ApplyAdaMax_T], v: Annotated[Any, TV_ApplyAdaMax_T], beta1_power: Annotated[Any, TV_ApplyAdaMax_T], lr: Annotated[Any, TV_ApplyAdaMax_T], beta1: Annotated[Any, TV_ApplyAdaMax_T], beta2: Annotated[Any, TV_ApplyAdaMax_T], epsilon: Annotated[Any, TV_ApplyAdaMax_T], grad: Annotated[Any, TV_ApplyAdaMax_T], use_locking: bool, name, ctx) -> Annotated[Any, TV_ApplyAdaMax_T]:
  raise RuntimeError("apply_ada_max op does not support eager execution. Arg 'out' is a ref.")

TV_ApplyAdadelta_T = TypeVar("TV_ApplyAdadelta_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def apply_adadelta(var: Annotated[Any, TV_ApplyAdadelta_T], accum: Annotated[Any, TV_ApplyAdadelta_T], accum_update: Annotated[Any, TV_ApplyAdadelta_T], lr: Annotated[Any, TV_ApplyAdadelta_T], rho: Annotated[Any, TV_ApplyAdadelta_T], epsilon: Annotated[Any, TV_ApplyAdadelta_T], grad: Annotated[Any, TV_ApplyAdadelta_T], use_locking:bool=False, name=None) -> Annotated[Any, TV_ApplyAdadelta_T]:
  r"""Update '*var' according to the adadelta scheme.

  accum = rho() * accum + (1 - rho()) * grad.square();
  update = (update_accum + epsilon).sqrt() * (accum + epsilon()).rsqrt() * grad;
  update_accum = rho() * update_accum + (1 - rho()) * update.square();
  var -= update;

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Should be from a Variable().
    accum: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    accum_update: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    rho: A `Tensor`. Must have the same type as `var`.
      Decay factor. Must be a scalar.
    epsilon: A `Tensor`. Must have the same type as `var`.
      Constant factor. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, updating of the var, accum and update_accum tensors will be protected by
      a lock; otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("apply_adadelta op does not support eager execution. Arg 'out' is a ref.")
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ApplyAdadelta", var=var, accum=accum, accum_update=accum_update,
                         lr=lr, rho=rho, epsilon=epsilon, grad=grad,
                         use_locking=use_locking, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "use_locking",
              _op._get_attr_bool("use_locking"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ApplyAdadelta", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ApplyAdadelta = tf_export("raw_ops.ApplyAdadelta")(_ops.to_raw_op(apply_adadelta))


def apply_adadelta_eager_fallback(var: Annotated[Any, TV_ApplyAdadelta_T], accum: Annotated[Any, TV_ApplyAdadelta_T], accum_update: Annotated[Any, TV_ApplyAdadelta_T], lr: Annotated[Any, TV_ApplyAdadelta_T], rho: Annotated[Any, TV_ApplyAdadelta_T], epsilon: Annotated[Any, TV_ApplyAdadelta_T], grad: Annotated[Any, TV_ApplyAdadelta_T], use_locking: bool, name, ctx) -> Annotated[Any, TV_ApplyAdadelta_T]:
  raise RuntimeError("apply_adadelta op does not support eager execution. Arg 'out' is a ref.")

TV_ApplyAdagrad_T = TypeVar("TV_ApplyAdagrad_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def apply_adagrad(var: Annotated[Any, TV_ApplyAdagrad_T], accum: Annotated[Any, TV_ApplyAdagrad_T], lr: Annotated[Any, TV_ApplyAdagrad_T], grad: Annotated[Any, TV_ApplyAdagrad_T], use_locking:bool=False, update_slots:bool=True, name=None) -> Annotated[Any, TV_ApplyAdagrad_T]:
  r"""Update '*var' according to the adagrad scheme.

  accum += grad * grad
  var -= lr * grad * (1 / sqrt(accum))

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Should be from a Variable().
    accum: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    update_slots: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("apply_adagrad op does not support eager execution. Arg 'out' is a ref.")
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if update_slots is None:
    update_slots = True
  update_slots = _execute.make_bool(update_slots, "update_slots")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ApplyAdagrad", var=var, accum=accum, lr=lr, grad=grad,
                        use_locking=use_locking, update_slots=update_slots,
                        name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "use_locking",
              _op._get_attr_bool("use_locking"), "update_slots",
              _op._get_attr_bool("update_slots"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ApplyAdagrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ApplyAdagrad = tf_export("raw_ops.ApplyAdagrad")(_ops.to_raw_op(apply_adagrad))


def apply_adagrad_eager_fallback(var: Annotated[Any, TV_ApplyAdagrad_T], accum: Annotated[Any, TV_ApplyAdagrad_T], lr: Annotated[Any, TV_ApplyAdagrad_T], grad: Annotated[Any, TV_ApplyAdagrad_T], use_locking: bool, update_slots: bool, name, ctx) -> Annotated[Any, TV_ApplyAdagrad_T]:
  raise RuntimeError("apply_adagrad op does not support eager execution. Arg 'out' is a ref.")

TV_ApplyAdagradDA_T = TypeVar("TV_ApplyAdagradDA_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def apply_adagrad_da(var: Annotated[Any, TV_ApplyAdagradDA_T], gradient_accumulator: Annotated[Any, TV_ApplyAdagradDA_T], gradient_squared_accumulator: Annotated[Any, TV_ApplyAdagradDA_T], grad: Annotated[Any, TV_ApplyAdagradDA_T], lr: Annotated[Any, TV_ApplyAdagradDA_T], l1: Annotated[Any, TV_ApplyAdagradDA_T], l2: Annotated[Any, TV_ApplyAdagradDA_T], global_step: Annotated[Any, _atypes.Int64], use_locking:bool=False, name=None) -> Annotated[Any, TV_ApplyAdagradDA_T]:
  r"""Update '*var' according to the proximal adagrad scheme.

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Should be from a Variable().
    gradient_accumulator: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    gradient_squared_accumulator: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `var`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `var`.
      L2 regularization. Must be a scalar.
    global_step: A `Tensor` of type `int64`.
      Training step number. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, updating of the var and accum tensors will be protected by
      a lock; otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("apply_adagrad_da op does not support eager execution. Arg 'out' is a ref.")
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ApplyAdagradDA", var=var, gradient_accumulator=gradient_accumulator,
                          gradient_squared_accumulator=gradient_squared_accumulator,
                          grad=grad, lr=lr, l1=l1, l2=l2,
                          global_step=global_step, use_locking=use_locking,
                          name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "use_locking",
              _op._get_attr_bool("use_locking"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ApplyAdagradDA", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ApplyAdagradDA = tf_export("raw_ops.ApplyAdagradDA")(_ops.to_raw_op(apply_adagrad_da))


def apply_adagrad_da_eager_fallback(var: Annotated[Any, TV_ApplyAdagradDA_T], gradient_accumulator: Annotated[Any, TV_ApplyAdagradDA_T], gradient_squared_accumulator: Annotated[Any, TV_ApplyAdagradDA_T], grad: Annotated[Any, TV_ApplyAdagradDA_T], lr: Annotated[Any, TV_ApplyAdagradDA_T], l1: Annotated[Any, TV_ApplyAdagradDA_T], l2: Annotated[Any, TV_ApplyAdagradDA_T], global_step: Annotated[Any, _atypes.Int64], use_locking: bool, name, ctx) -> Annotated[Any, TV_ApplyAdagradDA_T]:
  raise RuntimeError("apply_adagrad_da op does not support eager execution. Arg 'out' is a ref.")

TV_ApplyAdagradV2_T = TypeVar("TV_ApplyAdagradV2_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def apply_adagrad_v2(var: Annotated[Any, TV_ApplyAdagradV2_T], accum: Annotated[Any, TV_ApplyAdagradV2_T], lr: Annotated[Any, TV_ApplyAdagradV2_T], epsilon: Annotated[Any, TV_ApplyAdagradV2_T], grad: Annotated[Any, TV_ApplyAdagradV2_T], use_locking:bool=False, update_slots:bool=True, name=None) -> Annotated[Any, TV_ApplyAdagradV2_T]:
  r"""Update '*var' according to the adagrad scheme.

  accum += grad * grad
  var -= lr * grad * (1 / sqrt(accum))

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Should be from a Variable().
    accum: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    epsilon: A `Tensor`. Must have the same type as `var`.
      Constant factor. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    update_slots: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("apply_adagrad_v2 op does not support eager execution. Arg 'out' is a ref.")
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if update_slots is None:
    update_slots = True
  update_slots = _execute.make_bool(update_slots, "update_slots")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ApplyAdagradV2", var=var, accum=accum, lr=lr, epsilon=epsilon,
                          grad=grad, use_locking=use_locking,
                          update_slots=update_slots, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "use_locking",
              _op._get_attr_bool("use_locking"), "update_slots",
              _op._get_attr_bool("update_slots"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ApplyAdagradV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ApplyAdagradV2 = tf_export("raw_ops.ApplyAdagradV2")(_ops.to_raw_op(apply_adagrad_v2))


def apply_adagrad_v2_eager_fallback(var: Annotated[Any, TV_ApplyAdagradV2_T], accum: Annotated[Any, TV_ApplyAdagradV2_T], lr: Annotated[Any, TV_ApplyAdagradV2_T], epsilon: Annotated[Any, TV_ApplyAdagradV2_T], grad: Annotated[Any, TV_ApplyAdagradV2_T], use_locking: bool, update_slots: bool, name, ctx) -> Annotated[Any, TV_ApplyAdagradV2_T]:
  raise RuntimeError("apply_adagrad_v2 op does not support eager execution. Arg 'out' is a ref.")

TV_ApplyAdam_T = TypeVar("TV_ApplyAdam_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def apply_adam(var: Annotated[Any, TV_ApplyAdam_T], m: Annotated[Any, TV_ApplyAdam_T], v: Annotated[Any, TV_ApplyAdam_T], beta1_power: Annotated[Any, TV_ApplyAdam_T], beta2_power: Annotated[Any, TV_ApplyAdam_T], lr: Annotated[Any, TV_ApplyAdam_T], beta1: Annotated[Any, TV_ApplyAdam_T], beta2: Annotated[Any, TV_ApplyAdam_T], epsilon: Annotated[Any, TV_ApplyAdam_T], grad: Annotated[Any, TV_ApplyAdam_T], use_locking:bool=False, use_nesterov:bool=False, name=None) -> Annotated[Any, TV_ApplyAdam_T]:
  r"""Update '*var' according to the Adam algorithm.

  $$\text{lr}_t := \mathrm{lr} \cdot \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}$$
  $$m_t := \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g$$
  $$v_t := \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g^2$$
  $$\text{var} := \begin{cases} \text{var} - (m_t \beta_1 + g \cdot (1 - \beta_1))\cdot\text{lr}_t/(\sqrt{v_t} + \epsilon), &\text{if use_nesterov}\\\\  \text{var} - m_t \cdot \text{lr}_t /(\sqrt{v_t} + \epsilon), &\text{otherwise} \end{cases}$$

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Should be from a Variable().
    m: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    v: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    beta1_power: A `Tensor`. Must have the same type as `var`.
      Must be a scalar.
    beta2_power: A `Tensor`. Must have the same type as `var`.
      Must be a scalar.
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    beta1: A `Tensor`. Must have the same type as `var`.
      Momentum factor. Must be a scalar.
    beta2: A `Tensor`. Must have the same type as `var`.
      Momentum factor. Must be a scalar.
    epsilon: A `Tensor`. Must have the same type as `var`.
      Ridge term. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var, m, and v tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    use_nesterov: An optional `bool`. Defaults to `False`.
      If `True`, uses the nesterov update.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("apply_adam op does not support eager execution. Arg 'out' is a ref.")
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if use_nesterov is None:
    use_nesterov = False
  use_nesterov = _execute.make_bool(use_nesterov, "use_nesterov")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ApplyAdam", var=var, m=m, v=v, beta1_power=beta1_power,
                     beta2_power=beta2_power, lr=lr, beta1=beta1, beta2=beta2,
                     epsilon=epsilon, grad=grad, use_locking=use_locking,
                     use_nesterov=use_nesterov, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "use_locking",
              _op._get_attr_bool("use_locking"), "use_nesterov",
              _op._get_attr_bool("use_nesterov"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ApplyAdam", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ApplyAdam = tf_export("raw_ops.ApplyAdam")(_ops.to_raw_op(apply_adam))


def apply_adam_eager_fallback(var: Annotated[Any, TV_ApplyAdam_T], m: Annotated[Any, TV_ApplyAdam_T], v: Annotated[Any, TV_ApplyAdam_T], beta1_power: Annotated[Any, TV_ApplyAdam_T], beta2_power: Annotated[Any, TV_ApplyAdam_T], lr: Annotated[Any, TV_ApplyAdam_T], beta1: Annotated[Any, TV_ApplyAdam_T], beta2: Annotated[Any, TV_ApplyAdam_T], epsilon: Annotated[Any, TV_ApplyAdam_T], grad: Annotated[Any, TV_ApplyAdam_T], use_locking: bool, use_nesterov: bool, name, ctx) -> Annotated[Any, TV_ApplyAdam_T]:
  raise RuntimeError("apply_adam op does not support eager execution. Arg 'out' is a ref.")

TV_ApplyAddSign_T = TypeVar("TV_ApplyAddSign_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def apply_add_sign(var: Annotated[Any, TV_ApplyAddSign_T], m: Annotated[Any, TV_ApplyAddSign_T], lr: Annotated[Any, TV_ApplyAddSign_T], alpha: Annotated[Any, TV_ApplyAddSign_T], sign_decay: Annotated[Any, TV_ApplyAddSign_T], beta: Annotated[Any, TV_ApplyAddSign_T], grad: Annotated[Any, TV_ApplyAddSign_T], use_locking:bool=False, name=None) -> Annotated[Any, TV_ApplyAddSign_T]:
  r"""Update '*var' according to the AddSign update.

  m_t <- beta1 * m_{t-1} + (1 - beta1) * g
  update <- (alpha + sign_decay * sign(g) *sign(m)) * g
  variable <- variable - lr_t * update

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Should be from a Variable().
    m: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    alpha: A `Tensor`. Must have the same type as `var`. Must be a scalar.
    sign_decay: A `Tensor`. Must have the same type as `var`.
      Must be a scalar.
    beta: A `Tensor`. Must have the same type as `var`. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and m tensors is
      protected by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("apply_add_sign op does not support eager execution. Arg 'out' is a ref.")
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ApplyAddSign", var=var, m=m, lr=lr, alpha=alpha,
                        sign_decay=sign_decay, beta=beta, grad=grad,
                        use_locking=use_locking, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "use_locking",
              _op._get_attr_bool("use_locking"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ApplyAddSign", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ApplyAddSign = tf_export("raw_ops.ApplyAddSign")(_ops.to_raw_op(apply_add_sign))


def apply_add_sign_eager_fallback(var: Annotated[Any, TV_ApplyAddSign_T], m: Annotated[Any, TV_ApplyAddSign_T], lr: Annotated[Any, TV_ApplyAddSign_T], alpha: Annotated[Any, TV_ApplyAddSign_T], sign_decay: Annotated[Any, TV_ApplyAddSign_T], beta: Annotated[Any, TV_ApplyAddSign_T], grad: Annotated[Any, TV_ApplyAddSign_T], use_locking: bool, name, ctx) -> Annotated[Any, TV_ApplyAddSign_T]:
  raise RuntimeError("apply_add_sign op does not support eager execution. Arg 'out' is a ref.")

TV_ApplyCenteredRMSProp_T = TypeVar("TV_ApplyCenteredRMSProp_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def apply_centered_rms_prop(var: Annotated[Any, TV_ApplyCenteredRMSProp_T], mg: Annotated[Any, TV_ApplyCenteredRMSProp_T], ms: Annotated[Any, TV_ApplyCenteredRMSProp_T], mom: Annotated[Any, TV_ApplyCenteredRMSProp_T], lr: Annotated[Any, TV_ApplyCenteredRMSProp_T], rho: Annotated[Any, TV_ApplyCenteredRMSProp_T], momentum: Annotated[Any, TV_ApplyCenteredRMSProp_T], epsilon: Annotated[Any, TV_ApplyCenteredRMSProp_T], grad: Annotated[Any, TV_ApplyCenteredRMSProp_T], use_locking:bool=False, name=None) -> Annotated[Any, TV_ApplyCenteredRMSProp_T]:
  r"""Update '*var' according to the centered RMSProp algorithm.

  The centered RMSProp algorithm uses an estimate of the centered second moment
  (i.e., the variance) for normalization, as opposed to regular RMSProp, which
  uses the (uncentered) second moment. This often helps with training, but is
  slightly more expensive in terms of computation and memory.

  Note that in dense implementation of this algorithm, mg, ms, and mom will
  update even if the grad is zero, but in this sparse implementation, mg, ms,
  and mom will not update in iterations during which the grad is zero.

  mean_square = decay * mean_square + (1-decay) * gradient ** 2
  mean_grad = decay * mean_grad + (1-decay) * gradient

  Delta = learning_rate * gradient / sqrt(mean_square + epsilon - mean_grad ** 2)

  mg <- rho * mg_{t-1} + (1-rho) * grad
  ms <- rho * ms_{t-1} + (1-rho) * grad * grad
  mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms - mg * mg + epsilon)
  var <- var - mom

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Should be from a Variable().
    mg: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    ms: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    mom: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    rho: A `Tensor`. Must have the same type as `var`.
      Decay rate. Must be a scalar.
    momentum: A `Tensor`. Must have the same type as `var`.
      Momentum Scale. Must be a scalar.
    epsilon: A `Tensor`. Must have the same type as `var`.
      Ridge term. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var, mg, ms, and mom tensors is
      protected by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("apply_centered_rms_prop op does not support eager execution. Arg 'out' is a ref.")
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ApplyCenteredRMSProp", var=var, mg=mg, ms=ms, mom=mom, lr=lr,
                                rho=rho, momentum=momentum, epsilon=epsilon,
                                grad=grad, use_locking=use_locking, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "use_locking",
              _op._get_attr_bool("use_locking"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ApplyCenteredRMSProp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ApplyCenteredRMSProp = tf_export("raw_ops.ApplyCenteredRMSProp")(_ops.to_raw_op(apply_centered_rms_prop))


def apply_centered_rms_prop_eager_fallback(var: Annotated[Any, TV_ApplyCenteredRMSProp_T], mg: Annotated[Any, TV_ApplyCenteredRMSProp_T], ms: Annotated[Any, TV_ApplyCenteredRMSProp_T], mom: Annotated[Any, TV_ApplyCenteredRMSProp_T], lr: Annotated[Any, TV_ApplyCenteredRMSProp_T], rho: Annotated[Any, TV_ApplyCenteredRMSProp_T], momentum: Annotated[Any, TV_ApplyCenteredRMSProp_T], epsilon: Annotated[Any, TV_ApplyCenteredRMSProp_T], grad: Annotated[Any, TV_ApplyCenteredRMSProp_T], use_locking: bool, name, ctx) -> Annotated[Any, TV_ApplyCenteredRMSProp_T]:
  raise RuntimeError("apply_centered_rms_prop op does not support eager execution. Arg 'out' is a ref.")

TV_ApplyFtrl_T = TypeVar("TV_ApplyFtrl_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def apply_ftrl(var: Annotated[Any, TV_ApplyFtrl_T], accum: Annotated[Any, TV_ApplyFtrl_T], linear: Annotated[Any, TV_ApplyFtrl_T], grad: Annotated[Any, TV_ApplyFtrl_T], lr: Annotated[Any, TV_ApplyFtrl_T], l1: Annotated[Any, TV_ApplyFtrl_T], l2: Annotated[Any, TV_ApplyFtrl_T], lr_power: Annotated[Any, TV_ApplyFtrl_T], use_locking:bool=False, multiply_linear_by_lr:bool=False, name=None) -> Annotated[Any, TV_ApplyFtrl_T]:
  r"""Update '*var' according to the Ftrl-proximal scheme.

  accum_new = accum + grad * grad
  linear += grad - (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
  quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
  var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
  accum = accum_new

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Should be from a Variable().
    accum: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    linear: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `var`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `var`.
      L2 regularization. Must be a scalar.
    lr_power: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    multiply_linear_by_lr: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("apply_ftrl op does not support eager execution. Arg 'out' is a ref.")
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if multiply_linear_by_lr is None:
    multiply_linear_by_lr = False
  multiply_linear_by_lr = _execute.make_bool(multiply_linear_by_lr, "multiply_linear_by_lr")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ApplyFtrl", var=var, accum=accum, linear=linear, grad=grad, lr=lr,
                     l1=l1, l2=l2, lr_power=lr_power, use_locking=use_locking,
                     multiply_linear_by_lr=multiply_linear_by_lr, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "use_locking",
              _op._get_attr_bool("use_locking"), "multiply_linear_by_lr",
              _op._get_attr_bool("multiply_linear_by_lr"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ApplyFtrl", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ApplyFtrl = tf_export("raw_ops.ApplyFtrl")(_ops.to_raw_op(apply_ftrl))


def apply_ftrl_eager_fallback(var: Annotated[Any, TV_ApplyFtrl_T], accum: Annotated[Any, TV_ApplyFtrl_T], linear: Annotated[Any, TV_ApplyFtrl_T], grad: Annotated[Any, TV_ApplyFtrl_T], lr: Annotated[Any, TV_ApplyFtrl_T], l1: Annotated[Any, TV_ApplyFtrl_T], l2: Annotated[Any, TV_ApplyFtrl_T], lr_power: Annotated[Any, TV_ApplyFtrl_T], use_locking: bool, multiply_linear_by_lr: bool, name, ctx) -> Annotated[Any, TV_ApplyFtrl_T]:
  raise RuntimeError("apply_ftrl op does not support eager execution. Arg 'out' is a ref.")

TV_ApplyFtrlV2_T = TypeVar("TV_ApplyFtrlV2_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def apply_ftrl_v2(var: Annotated[Any, TV_ApplyFtrlV2_T], accum: Annotated[Any, TV_ApplyFtrlV2_T], linear: Annotated[Any, TV_ApplyFtrlV2_T], grad: Annotated[Any, TV_ApplyFtrlV2_T], lr: Annotated[Any, TV_ApplyFtrlV2_T], l1: Annotated[Any, TV_ApplyFtrlV2_T], l2: Annotated[Any, TV_ApplyFtrlV2_T], l2_shrinkage: Annotated[Any, TV_ApplyFtrlV2_T], lr_power: Annotated[Any, TV_ApplyFtrlV2_T], use_locking:bool=False, multiply_linear_by_lr:bool=False, name=None) -> Annotated[Any, TV_ApplyFtrlV2_T]:
  r"""Update '*var' according to the Ftrl-proximal scheme.

  grad_with_shrinkage = grad + 2 * l2_shrinkage * var
  accum_new = accum + grad * grad
  linear += grad_with_shrinkage -
      (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
  quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
  var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
  accum = accum_new

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Should be from a Variable().
    accum: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    linear: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `var`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `var`.
      L2 shrinkage regularization. Must be a scalar.
    l2_shrinkage: A `Tensor`. Must have the same type as `var`.
    lr_power: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    multiply_linear_by_lr: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("apply_ftrl_v2 op does not support eager execution. Arg 'out' is a ref.")
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if multiply_linear_by_lr is None:
    multiply_linear_by_lr = False
  multiply_linear_by_lr = _execute.make_bool(multiply_linear_by_lr, "multiply_linear_by_lr")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ApplyFtrlV2", var=var, accum=accum, linear=linear, grad=grad, lr=lr,
                       l1=l1, l2=l2, l2_shrinkage=l2_shrinkage,
                       lr_power=lr_power, use_locking=use_locking,
                       multiply_linear_by_lr=multiply_linear_by_lr, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "use_locking",
              _op._get_attr_bool("use_locking"), "multiply_linear_by_lr",
              _op._get_attr_bool("multiply_linear_by_lr"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ApplyFtrlV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ApplyFtrlV2 = tf_export("raw_ops.ApplyFtrlV2")(_ops.to_raw_op(apply_ftrl_v2))


def apply_ftrl_v2_eager_fallback(var: Annotated[Any, TV_ApplyFtrlV2_T], accum: Annotated[Any, TV_ApplyFtrlV2_T], linear: Annotated[Any, TV_ApplyFtrlV2_T], grad: Annotated[Any, TV_ApplyFtrlV2_T], lr: Annotated[Any, TV_ApplyFtrlV2_T], l1: Annotated[Any, TV_ApplyFtrlV2_T], l2: Annotated[Any, TV_ApplyFtrlV2_T], l2_shrinkage: Annotated[Any, TV_ApplyFtrlV2_T], lr_power: Annotated[Any, TV_ApplyFtrlV2_T], use_locking: bool, multiply_linear_by_lr: bool, name, ctx) -> Annotated[Any, TV_ApplyFtrlV2_T]:
  raise RuntimeError("apply_ftrl_v2 op does not support eager execution. Arg 'out' is a ref.")

TV_ApplyGradientDescent_T = TypeVar("TV_ApplyGradientDescent_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def apply_gradient_descent(var: Annotated[Any, TV_ApplyGradientDescent_T], alpha: Annotated[Any, TV_ApplyGradientDescent_T], delta: Annotated[Any, TV_ApplyGradientDescent_T], use_locking:bool=False, name=None) -> Annotated[Any, TV_ApplyGradientDescent_T]:
  r"""Update '*var' by subtracting 'alpha' * 'delta' from it.

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Should be from a Variable().
    alpha: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    delta: A `Tensor`. Must have the same type as `var`. The change.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, the subtraction will be protected by a lock;
      otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("apply_gradient_descent op does not support eager execution. Arg 'out' is a ref.")
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ApplyGradientDescent", var=var, alpha=alpha, delta=delta,
                                use_locking=use_locking, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "use_locking",
              _op._get_attr_bool("use_locking"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ApplyGradientDescent", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ApplyGradientDescent = tf_export("raw_ops.ApplyGradientDescent")(_ops.to_raw_op(apply_gradient_descent))


def apply_gradient_descent_eager_fallback(var: Annotated[Any, TV_ApplyGradientDescent_T], alpha: Annotated[Any, TV_ApplyGradientDescent_T], delta: Annotated[Any, TV_ApplyGradientDescent_T], use_locking: bool, name, ctx) -> Annotated[Any, TV_ApplyGradientDescent_T]:
  raise RuntimeError("apply_gradient_descent op does not support eager execution. Arg 'out' is a ref.")

TV_ApplyMomentum_T = TypeVar("TV_ApplyMomentum_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def apply_momentum(var: Annotated[Any, TV_ApplyMomentum_T], accum: Annotated[Any, TV_ApplyMomentum_T], lr: Annotated[Any, TV_ApplyMomentum_T], grad: Annotated[Any, TV_ApplyMomentum_T], momentum: Annotated[Any, TV_ApplyMomentum_T], use_locking:bool=False, use_nesterov:bool=False, name=None) -> Annotated[Any, TV_ApplyMomentum_T]:
  r"""Update '*var' according to the momentum scheme.

  Set use_nesterov = True if you want to use Nesterov momentum.

  accum = accum * momentum + grad
  var -= lr * accum

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Should be from a Variable().
    accum: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    momentum: A `Tensor`. Must have the same type as `var`.
      Momentum. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    use_nesterov: An optional `bool`. Defaults to `False`.
      If `True`, the tensor passed to compute grad will be
      var - lr * momentum * accum, so in the end, the var you get is actually
      var - lr * momentum * accum.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("apply_momentum op does not support eager execution. Arg 'out' is a ref.")
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if use_nesterov is None:
    use_nesterov = False
  use_nesterov = _execute.make_bool(use_nesterov, "use_nesterov")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ApplyMomentum", var=var, accum=accum, lr=lr, grad=grad,
                         momentum=momentum, use_locking=use_locking,
                         use_nesterov=use_nesterov, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "use_locking",
              _op._get_attr_bool("use_locking"), "use_nesterov",
              _op._get_attr_bool("use_nesterov"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ApplyMomentum", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ApplyMomentum = tf_export("raw_ops.ApplyMomentum")(_ops.to_raw_op(apply_momentum))


def apply_momentum_eager_fallback(var: Annotated[Any, TV_ApplyMomentum_T], accum: Annotated[Any, TV_ApplyMomentum_T], lr: Annotated[Any, TV_ApplyMomentum_T], grad: Annotated[Any, TV_ApplyMomentum_T], momentum: Annotated[Any, TV_ApplyMomentum_T], use_locking: bool, use_nesterov: bool, name, ctx) -> Annotated[Any, TV_ApplyMomentum_T]:
  raise RuntimeError("apply_momentum op does not support eager execution. Arg 'out' is a ref.")

TV_ApplyPowerSign_T = TypeVar("TV_ApplyPowerSign_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def apply_power_sign(var: Annotated[Any, TV_ApplyPowerSign_T], m: Annotated[Any, TV_ApplyPowerSign_T], lr: Annotated[Any, TV_ApplyPowerSign_T], logbase: Annotated[Any, TV_ApplyPowerSign_T], sign_decay: Annotated[Any, TV_ApplyPowerSign_T], beta: Annotated[Any, TV_ApplyPowerSign_T], grad: Annotated[Any, TV_ApplyPowerSign_T], use_locking:bool=False, name=None) -> Annotated[Any, TV_ApplyPowerSign_T]:
  r"""Update '*var' according to the AddSign update.

  m_t <- beta1 * m_{t-1} + (1 - beta1) * g
  update <- exp(logbase * sign_decay * sign(g) * sign(m_t)) * g
  variable <- variable - lr_t * update

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Should be from a Variable().
    m: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    logbase: A `Tensor`. Must have the same type as `var`. Must be a scalar.
    sign_decay: A `Tensor`. Must have the same type as `var`.
      Must be a scalar.
    beta: A `Tensor`. Must have the same type as `var`. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and m tensors is
      protected by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("apply_power_sign op does not support eager execution. Arg 'out' is a ref.")
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ApplyPowerSign", var=var, m=m, lr=lr, logbase=logbase,
                          sign_decay=sign_decay, beta=beta, grad=grad,
                          use_locking=use_locking, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "use_locking",
              _op._get_attr_bool("use_locking"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ApplyPowerSign", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ApplyPowerSign = tf_export("raw_ops.ApplyPowerSign")(_ops.to_raw_op(apply_power_sign))


def apply_power_sign_eager_fallback(var: Annotated[Any, TV_ApplyPowerSign_T], m: Annotated[Any, TV_ApplyPowerSign_T], lr: Annotated[Any, TV_ApplyPowerSign_T], logbase: Annotated[Any, TV_ApplyPowerSign_T], sign_decay: Annotated[Any, TV_ApplyPowerSign_T], beta: Annotated[Any, TV_ApplyPowerSign_T], grad: Annotated[Any, TV_ApplyPowerSign_T], use_locking: bool, name, ctx) -> Annotated[Any, TV_ApplyPowerSign_T]:
  raise RuntimeError("apply_power_sign op does not support eager execution. Arg 'out' is a ref.")

TV_ApplyProximalAdagrad_T = TypeVar("TV_ApplyProximalAdagrad_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def apply_proximal_adagrad(var: Annotated[Any, TV_ApplyProximalAdagrad_T], accum: Annotated[Any, TV_ApplyProximalAdagrad_T], lr: Annotated[Any, TV_ApplyProximalAdagrad_T], l1: Annotated[Any, TV_ApplyProximalAdagrad_T], l2: Annotated[Any, TV_ApplyProximalAdagrad_T], grad: Annotated[Any, TV_ApplyProximalAdagrad_T], use_locking:bool=False, name=None) -> Annotated[Any, TV_ApplyProximalAdagrad_T]:
  r"""Update '*var' and '*accum' according to FOBOS with Adagrad learning rate.

  accum += grad * grad
  prox_v = var - lr * grad * (1 / sqrt(accum))
  var = sign(prox_v)/(1+lr*l2) * max{|prox_v|-lr*l1,0}

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Should be from a Variable().
    accum: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `var`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `var`.
      L2 regularization. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, updating of the var and accum tensors will be protected by
      a lock; otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("apply_proximal_adagrad op does not support eager execution. Arg 'out' is a ref.")
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ApplyProximalAdagrad", var=var, accum=accum, lr=lr, l1=l1, l2=l2,
                                grad=grad, use_locking=use_locking, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "use_locking",
              _op._get_attr_bool("use_locking"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ApplyProximalAdagrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ApplyProximalAdagrad = tf_export("raw_ops.ApplyProximalAdagrad")(_ops.to_raw_op(apply_proximal_adagrad))


def apply_proximal_adagrad_eager_fallback(var: Annotated[Any, TV_ApplyProximalAdagrad_T], accum: Annotated[Any, TV_ApplyProximalAdagrad_T], lr: Annotated[Any, TV_ApplyProximalAdagrad_T], l1: Annotated[Any, TV_ApplyProximalAdagrad_T], l2: Annotated[Any, TV_ApplyProximalAdagrad_T], grad: Annotated[Any, TV_ApplyProximalAdagrad_T], use_locking: bool, name, ctx) -> Annotated[Any, TV_ApplyProximalAdagrad_T]:
  raise RuntimeError("apply_proximal_adagrad op does not support eager execution. Arg 'out' is a ref.")

TV_ApplyProximalGradientDescent_T = TypeVar("TV_ApplyProximalGradientDescent_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def apply_proximal_gradient_descent(var: Annotated[Any, TV_ApplyProximalGradientDescent_T], alpha: Annotated[Any, TV_ApplyProximalGradientDescent_T], l1: Annotated[Any, TV_ApplyProximalGradientDescent_T], l2: Annotated[Any, TV_ApplyProximalGradientDescent_T], delta: Annotated[Any, TV_ApplyProximalGradientDescent_T], use_locking:bool=False, name=None) -> Annotated[Any, TV_ApplyProximalGradientDescent_T]:
  r"""Update '*var' as FOBOS algorithm with fixed learning rate.

  prox_v = var - alpha * delta
  var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Should be from a Variable().
    alpha: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `var`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `var`.
      L2 regularization. Must be a scalar.
    delta: A `Tensor`. Must have the same type as `var`. The change.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, the subtraction will be protected by a lock;
      otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("apply_proximal_gradient_descent op does not support eager execution. Arg 'out' is a ref.")
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ApplyProximalGradientDescent", var=var, alpha=alpha, l1=l1, l2=l2,
                                        delta=delta, use_locking=use_locking,
                                        name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "use_locking",
              _op._get_attr_bool("use_locking"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ApplyProximalGradientDescent", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ApplyProximalGradientDescent = tf_export("raw_ops.ApplyProximalGradientDescent")(_ops.to_raw_op(apply_proximal_gradient_descent))


def apply_proximal_gradient_descent_eager_fallback(var: Annotated[Any, TV_ApplyProximalGradientDescent_T], alpha: Annotated[Any, TV_ApplyProximalGradientDescent_T], l1: Annotated[Any, TV_ApplyProximalGradientDescent_T], l2: Annotated[Any, TV_ApplyProximalGradientDescent_T], delta: Annotated[Any, TV_ApplyProximalGradientDescent_T], use_locking: bool, name, ctx) -> Annotated[Any, TV_ApplyProximalGradientDescent_T]:
  raise RuntimeError("apply_proximal_gradient_descent op does not support eager execution. Arg 'out' is a ref.")

TV_ApplyRMSProp_T = TypeVar("TV_ApplyRMSProp_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def apply_rms_prop(var: Annotated[Any, TV_ApplyRMSProp_T], ms: Annotated[Any, TV_ApplyRMSProp_T], mom: Annotated[Any, TV_ApplyRMSProp_T], lr: Annotated[Any, TV_ApplyRMSProp_T], rho: Annotated[Any, TV_ApplyRMSProp_T], momentum: Annotated[Any, TV_ApplyRMSProp_T], epsilon: Annotated[Any, TV_ApplyRMSProp_T], grad: Annotated[Any, TV_ApplyRMSProp_T], use_locking:bool=False, name=None) -> Annotated[Any, TV_ApplyRMSProp_T]:
  r"""Update '*var' according to the RMSProp algorithm.

  Note that in dense implementation of this algorithm, ms and mom will
  update even if the grad is zero, but in this sparse implementation, ms
  and mom will not update in iterations during which the grad is zero.

  mean_square = decay * mean_square + (1-decay) * gradient ** 2
  Delta = learning_rate * gradient / sqrt(mean_square + epsilon)

  ms <- rho * ms_{t-1} + (1-rho) * grad * grad
  mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
  var <- var - mom

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Should be from a Variable().
    ms: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    mom: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    rho: A `Tensor`. Must have the same type as `var`.
      Decay rate. Must be a scalar.
    momentum: A `Tensor`. Must have the same type as `var`.
    epsilon: A `Tensor`. Must have the same type as `var`.
      Ridge term. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var, ms, and mom tensors is protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("apply_rms_prop op does not support eager execution. Arg 'out' is a ref.")
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ApplyRMSProp", var=var, ms=ms, mom=mom, lr=lr, rho=rho,
                        momentum=momentum, epsilon=epsilon, grad=grad,
                        use_locking=use_locking, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "use_locking",
              _op._get_attr_bool("use_locking"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ApplyRMSProp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ApplyRMSProp = tf_export("raw_ops.ApplyRMSProp")(_ops.to_raw_op(apply_rms_prop))


def apply_rms_prop_eager_fallback(var: Annotated[Any, TV_ApplyRMSProp_T], ms: Annotated[Any, TV_ApplyRMSProp_T], mom: Annotated[Any, TV_ApplyRMSProp_T], lr: Annotated[Any, TV_ApplyRMSProp_T], rho: Annotated[Any, TV_ApplyRMSProp_T], momentum: Annotated[Any, TV_ApplyRMSProp_T], epsilon: Annotated[Any, TV_ApplyRMSProp_T], grad: Annotated[Any, TV_ApplyRMSProp_T], use_locking: bool, name, ctx) -> Annotated[Any, TV_ApplyRMSProp_T]:
  raise RuntimeError("apply_rms_prop op does not support eager execution. Arg 'out' is a ref.")

TV_ResourceApplyAdaMax_T = TypeVar("TV_ResourceApplyAdaMax_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def resource_apply_ada_max(var: Annotated[Any, _atypes.Resource], m: Annotated[Any, _atypes.Resource], v: Annotated[Any, _atypes.Resource], beta1_power: Annotated[Any, TV_ResourceApplyAdaMax_T], lr: Annotated[Any, TV_ResourceApplyAdaMax_T], beta1: Annotated[Any, TV_ResourceApplyAdaMax_T], beta2: Annotated[Any, TV_ResourceApplyAdaMax_T], epsilon: Annotated[Any, TV_ResourceApplyAdaMax_T], grad: Annotated[Any, TV_ResourceApplyAdaMax_T], use_locking:bool=False, name=None):
  r"""Update '*var' according to the AdaMax algorithm.

  m_t <- beta1 * m_{t-1} + (1 - beta1) * g
  v_t <- max(beta2 * v_{t-1}, abs(g))
  variable <- variable - learning_rate / (1 - beta1^t) * m_t / (v_t + epsilon)

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    m: A `Tensor` of type `resource`. Should be from a Variable().
    v: A `Tensor` of type `resource`. Should be from a Variable().
    beta1_power: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Must be a scalar.
    lr: A `Tensor`. Must have the same type as `beta1_power`.
      Scaling factor. Must be a scalar.
    beta1: A `Tensor`. Must have the same type as `beta1_power`.
      Momentum factor. Must be a scalar.
    beta2: A `Tensor`. Must have the same type as `beta1_power`.
      Momentum factor. Must be a scalar.
    epsilon: A `Tensor`. Must have the same type as `beta1_power`.
      Ridge term. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `beta1_power`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var, m, and v tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceApplyAdaMax", name, var, m, v, beta1_power, lr, beta1,
        beta2, epsilon, grad, "use_locking", use_locking)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_apply_ada_max_eager_fallback(
          var, m, v, beta1_power, lr, beta1, beta2, epsilon, grad,
          use_locking=use_locking, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceApplyAdaMax", var=var, m=m, v=v, beta1_power=beta1_power,
                               lr=lr, beta1=beta1, beta2=beta2,
                               epsilon=epsilon, grad=grad,
                               use_locking=use_locking, name=name)
  return _op
ResourceApplyAdaMax = tf_export("raw_ops.ResourceApplyAdaMax")(_ops.to_raw_op(resource_apply_ada_max))


def resource_apply_ada_max_eager_fallback(var: Annotated[Any, _atypes.Resource], m: Annotated[Any, _atypes.Resource], v: Annotated[Any, _atypes.Resource], beta1_power: Annotated[Any, TV_ResourceApplyAdaMax_T], lr: Annotated[Any, TV_ResourceApplyAdaMax_T], beta1: Annotated[Any, TV_ResourceApplyAdaMax_T], beta2: Annotated[Any, TV_ResourceApplyAdaMax_T], epsilon: Annotated[Any, TV_ResourceApplyAdaMax_T], grad: Annotated[Any, TV_ResourceApplyAdaMax_T], use_locking: bool, name, ctx):
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([beta1_power, lr, beta1, beta2, epsilon, grad], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (beta1_power, lr, beta1, beta2, epsilon, grad) = _inputs_T
  var = _ops.convert_to_tensor(var, _dtypes.resource)
  m = _ops.convert_to_tensor(m, _dtypes.resource)
  v = _ops.convert_to_tensor(v, _dtypes.resource)
  _inputs_flat = [var, m, v, beta1_power, lr, beta1, beta2, epsilon, grad]
  _attrs = ("T", _attr_T, "use_locking", use_locking)
  _result = _execute.execute(b"ResourceApplyAdaMax", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


TV_ResourceApplyAdadelta_T = TypeVar("TV_ResourceApplyAdadelta_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def resource_apply_adadelta(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], accum_update: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceApplyAdadelta_T], rho: Annotated[Any, TV_ResourceApplyAdadelta_T], epsilon: Annotated[Any, TV_ResourceApplyAdadelta_T], grad: Annotated[Any, TV_ResourceApplyAdadelta_T], use_locking:bool=False, name=None):
  r"""Update '*var' according to the adadelta scheme.

  accum = rho() * accum + (1 - rho()) * grad.square();
  update = (update_accum + epsilon).sqrt() * (accum + epsilon()).rsqrt() * grad;
  update_accum = rho() * update_accum + (1 - rho()) * update.square();
  var -= update;

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    accum: A `Tensor` of type `resource`. Should be from a Variable().
    accum_update: A `Tensor` of type `resource`. Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Scaling factor. Must be a scalar.
    rho: A `Tensor`. Must have the same type as `lr`.
      Decay factor. Must be a scalar.
    epsilon: A `Tensor`. Must have the same type as `lr`.
      Constant factor. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, updating of the var, accum and update_accum tensors will be protected by
      a lock; otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceApplyAdadelta", name, var, accum, accum_update, lr,
        rho, epsilon, grad, "use_locking", use_locking)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_apply_adadelta_eager_fallback(
          var, accum, accum_update, lr, rho, epsilon, grad,
          use_locking=use_locking, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceApplyAdadelta", var=var, accum=accum,
                                 accum_update=accum_update, lr=lr, rho=rho,
                                 epsilon=epsilon, grad=grad,
                                 use_locking=use_locking, name=name)
  return _op
ResourceApplyAdadelta = tf_export("raw_ops.ResourceApplyAdadelta")(_ops.to_raw_op(resource_apply_adadelta))


def resource_apply_adadelta_eager_fallback(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], accum_update: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceApplyAdadelta_T], rho: Annotated[Any, TV_ResourceApplyAdadelta_T], epsilon: Annotated[Any, TV_ResourceApplyAdadelta_T], grad: Annotated[Any, TV_ResourceApplyAdadelta_T], use_locking: bool, name, ctx):
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([lr, rho, epsilon, grad], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (lr, rho, epsilon, grad) = _inputs_T
  var = _ops.convert_to_tensor(var, _dtypes.resource)
  accum = _ops.convert_to_tensor(accum, _dtypes.resource)
  accum_update = _ops.convert_to_tensor(accum_update, _dtypes.resource)
  _inputs_flat = [var, accum, accum_update, lr, rho, epsilon, grad]
  _attrs = ("T", _attr_T, "use_locking", use_locking)
  _result = _execute.execute(b"ResourceApplyAdadelta", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


TV_ResourceApplyAdagrad_T = TypeVar("TV_ResourceApplyAdagrad_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def resource_apply_adagrad(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceApplyAdagrad_T], grad: Annotated[Any, TV_ResourceApplyAdagrad_T], use_locking:bool=False, update_slots:bool=True, name=None):
  r"""Update '*var' according to the adagrad scheme.

  accum += grad * grad
  var -= lr * grad * (1 / sqrt(accum))

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    accum: A `Tensor` of type `resource`. Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Scaling factor. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    update_slots: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceApplyAdagrad", name, var, accum, lr, grad,
        "use_locking", use_locking, "update_slots", update_slots)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_apply_adagrad_eager_fallback(
          var, accum, lr, grad, use_locking=use_locking,
          update_slots=update_slots, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if update_slots is None:
    update_slots = True
  update_slots = _execute.make_bool(update_slots, "update_slots")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceApplyAdagrad", var=var, accum=accum, lr=lr, grad=grad,
                                use_locking=use_locking,
                                update_slots=update_slots, name=name)
  return _op
ResourceApplyAdagrad = tf_export("raw_ops.ResourceApplyAdagrad")(_ops.to_raw_op(resource_apply_adagrad))


def resource_apply_adagrad_eager_fallback(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceApplyAdagrad_T], grad: Annotated[Any, TV_ResourceApplyAdagrad_T], use_locking: bool, update_slots: bool, name, ctx):
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if update_slots is None:
    update_slots = True
  update_slots = _execute.make_bool(update_slots, "update_slots")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([lr, grad], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (lr, grad) = _inputs_T
  var = _ops.convert_to_tensor(var, _dtypes.resource)
  accum = _ops.convert_to_tensor(accum, _dtypes.resource)
  _inputs_flat = [var, accum, lr, grad]
  _attrs = ("T", _attr_T, "use_locking", use_locking, "update_slots",
  update_slots)
  _result = _execute.execute(b"ResourceApplyAdagrad", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


TV_ResourceApplyAdagradDA_T = TypeVar("TV_ResourceApplyAdagradDA_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def resource_apply_adagrad_da(var: Annotated[Any, _atypes.Resource], gradient_accumulator: Annotated[Any, _atypes.Resource], gradient_squared_accumulator: Annotated[Any, _atypes.Resource], grad: Annotated[Any, TV_ResourceApplyAdagradDA_T], lr: Annotated[Any, TV_ResourceApplyAdagradDA_T], l1: Annotated[Any, TV_ResourceApplyAdagradDA_T], l2: Annotated[Any, TV_ResourceApplyAdagradDA_T], global_step: Annotated[Any, _atypes.Int64], use_locking:bool=False, name=None):
  r"""Update '*var' according to the proximal adagrad scheme.

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    gradient_accumulator: A `Tensor` of type `resource`.
      Should be from a Variable().
    gradient_squared_accumulator: A `Tensor` of type `resource`.
      Should be from a Variable().
    grad: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      The gradient.
    lr: A `Tensor`. Must have the same type as `grad`.
      Scaling factor. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `grad`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `grad`.
      L2 regularization. Must be a scalar.
    global_step: A `Tensor` of type `int64`.
      Training step number. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, updating of the var and accum tensors will be protected by
      a lock; otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceApplyAdagradDA", name, var, gradient_accumulator,
        gradient_squared_accumulator, grad, lr, l1, l2, global_step,
        "use_locking", use_locking)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_apply_adagrad_da_eager_fallback(
          var, gradient_accumulator, gradient_squared_accumulator, grad, lr,
          l1, l2, global_step, use_locking=use_locking, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceApplyAdagradDA", var=var,
                                  gradient_accumulator=gradient_accumulator,
                                  gradient_squared_accumulator=gradient_squared_accumulator,
                                  grad=grad, lr=lr, l1=l1, l2=l2,
                                  global_step=global_step,
                                  use_locking=use_locking, name=name)
  return _op
ResourceApplyAdagradDA = tf_export("raw_ops.ResourceApplyAdagradDA")(_ops.to_raw_op(resource_apply_adagrad_da))


def resource_apply_adagrad_da_eager_fallback(var: Annotated[Any, _atypes.Resource], gradient_accumulator: Annotated[Any, _atypes.Resource], gradient_squared_accumulator: Annotated[Any, _atypes.Resource], grad: Annotated[Any, TV_ResourceApplyAdagradDA_T], lr: Annotated[Any, TV_ResourceApplyAdagradDA_T], l1: Annotated[Any, TV_ResourceApplyAdagradDA_T], l2: Annotated[Any, TV_ResourceApplyAdagradDA_T], global_step: Annotated[Any, _atypes.Int64], use_locking: bool, name, ctx):
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([grad, lr, l1, l2], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (grad, lr, l1, l2) = _inputs_T
  var = _ops.convert_to_tensor(var, _dtypes.resource)
  gradient_accumulator = _ops.convert_to_tensor(gradient_accumulator, _dtypes.resource)
  gradient_squared_accumulator = _ops.convert_to_tensor(gradient_squared_accumulator, _dtypes.resource)
  global_step = _ops.convert_to_tensor(global_step, _dtypes.int64)
  _inputs_flat = [var, gradient_accumulator, gradient_squared_accumulator, grad, lr, l1, l2, global_step]
  _attrs = ("T", _attr_T, "use_locking", use_locking)
  _result = _execute.execute(b"ResourceApplyAdagradDA", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_ResourceApplyAdagradV2_T = TypeVar("TV_ResourceApplyAdagradV2_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def resource_apply_adagrad_v2(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceApplyAdagradV2_T], epsilon: Annotated[Any, TV_ResourceApplyAdagradV2_T], grad: Annotated[Any, TV_ResourceApplyAdagradV2_T], use_locking:bool=False, update_slots:bool=True, name=None):
  r"""Update '*var' according to the adagrad scheme.

  accum += grad * grad
  var -= lr * grad * (1 / (sqrt(accum) + epsilon))

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    accum: A `Tensor` of type `resource`. Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Scaling factor. Must be a scalar.
    epsilon: A `Tensor`. Must have the same type as `lr`.
      Constant factor. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    update_slots: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceApplyAdagradV2", name, var, accum, lr, epsilon, grad,
        "use_locking", use_locking, "update_slots", update_slots)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_apply_adagrad_v2_eager_fallback(
          var, accum, lr, epsilon, grad, use_locking=use_locking,
          update_slots=update_slots, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if update_slots is None:
    update_slots = True
  update_slots = _execute.make_bool(update_slots, "update_slots")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceApplyAdagradV2", var=var, accum=accum, lr=lr,
                                  epsilon=epsilon, grad=grad,
                                  use_locking=use_locking,
                                  update_slots=update_slots, name=name)
  return _op
ResourceApplyAdagradV2 = tf_export("raw_ops.ResourceApplyAdagradV2")(_ops.to_raw_op(resource_apply_adagrad_v2))


def resource_apply_adagrad_v2_eager_fallback(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceApplyAdagradV2_T], epsilon: Annotated[Any, TV_ResourceApplyAdagradV2_T], grad: Annotated[Any, TV_ResourceApplyAdagradV2_T], use_locking: bool, update_slots: bool, name, ctx):
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if update_slots is None:
    update_slots = True
  update_slots = _execute.make_bool(update_slots, "update_slots")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([lr, epsilon, grad], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (lr, epsilon, grad) = _inputs_T
  var = _ops.convert_to_tensor(var, _dtypes.resource)
  accum = _ops.convert_to_tensor(accum, _dtypes.resource)
  _inputs_flat = [var, accum, lr, epsilon, grad]
  _attrs = ("T", _attr_T, "use_locking", use_locking, "update_slots",
  update_slots)
  _result = _execute.execute(b"ResourceApplyAdagradV2", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_ResourceApplyAdam_T = TypeVar("TV_ResourceApplyAdam_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def resource_apply_adam(var: Annotated[Any, _atypes.Resource], m: Annotated[Any, _atypes.Resource], v: Annotated[Any, _atypes.Resource], beta1_power: Annotated[Any, TV_ResourceApplyAdam_T], beta2_power: Annotated[Any, TV_ResourceApplyAdam_T], lr: Annotated[Any, TV_ResourceApplyAdam_T], beta1: Annotated[Any, TV_ResourceApplyAdam_T], beta2: Annotated[Any, TV_ResourceApplyAdam_T], epsilon: Annotated[Any, TV_ResourceApplyAdam_T], grad: Annotated[Any, TV_ResourceApplyAdam_T], use_locking:bool=False, use_nesterov:bool=False, name=None):
  r"""Update '*var' according to the Adam algorithm.

  $$\text{lr}_t := \mathrm{lr} \cdot \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}$$
  $$m_t := \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g$$
  $$v_t := \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g^2$$
  $$\text{var} := \begin{cases} \text{var} - (m_t \beta_1 + g \cdot (1 - \beta_1))\cdot\text{lr}_t/(\sqrt{v_t} + \epsilon), &\text{if use_nesterov}\\\\  \text{var} - m_t \cdot \text{lr}_t /(\sqrt{v_t} + \epsilon), &\text{otherwise} \end{cases}$$

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    m: A `Tensor` of type `resource`. Should be from a Variable().
    v: A `Tensor` of type `resource`. Should be from a Variable().
    beta1_power: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Must be a scalar.
    beta2_power: A `Tensor`. Must have the same type as `beta1_power`.
      Must be a scalar.
    lr: A `Tensor`. Must have the same type as `beta1_power`.
      Scaling factor. Must be a scalar.
    beta1: A `Tensor`. Must have the same type as `beta1_power`.
      Momentum factor. Must be a scalar.
    beta2: A `Tensor`. Must have the same type as `beta1_power`.
      Momentum factor. Must be a scalar.
    epsilon: A `Tensor`. Must have the same type as `beta1_power`.
      Ridge term. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `beta1_power`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var, m, and v tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    use_nesterov: An optional `bool`. Defaults to `False`.
      If `True`, uses the nesterov update.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceApplyAdam", name, var, m, v, beta1_power, beta2_power,
        lr, beta1, beta2, epsilon, grad, "use_locking", use_locking,
        "use_nesterov", use_nesterov)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_apply_adam_eager_fallback(
          var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon,
          grad, use_locking=use_locking, use_nesterov=use_nesterov, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if use_nesterov is None:
    use_nesterov = False
  use_nesterov = _execute.make_bool(use_nesterov, "use_nesterov")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceApplyAdam", var=var, m=m, v=v, beta1_power=beta1_power,
                             beta2_power=beta2_power, lr=lr, beta1=beta1,
                             beta2=beta2, epsilon=epsilon, grad=grad,
                             use_locking=use_locking,
                             use_nesterov=use_nesterov, name=name)
  return _op
ResourceApplyAdam = tf_export("raw_ops.ResourceApplyAdam")(_ops.to_raw_op(resource_apply_adam))


def resource_apply_adam_eager_fallback(var: Annotated[Any, _atypes.Resource], m: Annotated[Any, _atypes.Resource], v: Annotated[Any, _atypes.Resource], beta1_power: Annotated[Any, TV_ResourceApplyAdam_T], beta2_power: Annotated[Any, TV_ResourceApplyAdam_T], lr: Annotated[Any, TV_ResourceApplyAdam_T], beta1: Annotated[Any, TV_ResourceApplyAdam_T], beta2: Annotated[Any, TV_ResourceApplyAdam_T], epsilon: Annotated[Any, TV_ResourceApplyAdam_T], grad: Annotated[Any, TV_ResourceApplyAdam_T], use_locking: bool, use_nesterov: bool, name, ctx):
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if use_nesterov is None:
    use_nesterov = False
  use_nesterov = _execute.make_bool(use_nesterov, "use_nesterov")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad) = _inputs_T
  var = _ops.convert_to_tensor(var, _dtypes.resource)
  m = _ops.convert_to_tensor(m, _dtypes.resource)
  v = _ops.convert_to_tensor(v, _dtypes.resource)
  _inputs_flat = [var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad]
  _attrs = ("T", _attr_T, "use_locking", use_locking, "use_nesterov",
  use_nesterov)
  _result = _execute.execute(b"ResourceApplyAdam", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


TV_ResourceApplyAdamWithAmsgrad_T = TypeVar("TV_ResourceApplyAdamWithAmsgrad_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def resource_apply_adam_with_amsgrad(var: Annotated[Any, _atypes.Resource], m: Annotated[Any, _atypes.Resource], v: Annotated[Any, _atypes.Resource], vhat: Annotated[Any, _atypes.Resource], beta1_power: Annotated[Any, TV_ResourceApplyAdamWithAmsgrad_T], beta2_power: Annotated[Any, TV_ResourceApplyAdamWithAmsgrad_T], lr: Annotated[Any, TV_ResourceApplyAdamWithAmsgrad_T], beta1: Annotated[Any, TV_ResourceApplyAdamWithAmsgrad_T], beta2: Annotated[Any, TV_ResourceApplyAdamWithAmsgrad_T], epsilon: Annotated[Any, TV_ResourceApplyAdamWithAmsgrad_T], grad: Annotated[Any, TV_ResourceApplyAdamWithAmsgrad_T], use_locking:bool=False, name=None):
  r"""Update '*var' according to the Adam algorithm.

  $$\text{lr}_t := \mathrm{learning_rate} * \sqrt{1 - \beta_2^t} / (1 - \beta_1^t)$$
  $$m_t := \beta_1 * m_{t-1} + (1 - \beta_1) * g$$
  $$v_t := \beta_2 * v_{t-1} + (1 - \beta_2) * g * g$$
  $$\hat{v}_t := max{\hat{v}_{t-1}, v_t}$$
  $$\text{variable} := \text{variable} - \text{lr}_t * m_t / (\sqrt{\hat{v}_t} + \epsilon)$$

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    m: A `Tensor` of type `resource`. Should be from a Variable().
    v: A `Tensor` of type `resource`. Should be from a Variable().
    vhat: A `Tensor` of type `resource`. Should be from a Variable().
    beta1_power: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Must be a scalar.
    beta2_power: A `Tensor`. Must have the same type as `beta1_power`.
      Must be a scalar.
    lr: A `Tensor`. Must have the same type as `beta1_power`.
      Scaling factor. Must be a scalar.
    beta1: A `Tensor`. Must have the same type as `beta1_power`.
      Momentum factor. Must be a scalar.
    beta2: A `Tensor`. Must have the same type as `beta1_power`.
      Momentum factor. Must be a scalar.
    epsilon: A `Tensor`. Must have the same type as `beta1_power`.
      Ridge term. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `beta1_power`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var, m, and v tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceApplyAdamWithAmsgrad", name, var, m, v, vhat,
        beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad,
        "use_locking", use_locking)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_apply_adam_with_amsgrad_eager_fallback(
          var, m, v, vhat, beta1_power, beta2_power, lr, beta1, beta2,
          epsilon, grad, use_locking=use_locking, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceApplyAdamWithAmsgrad", var=var, m=m, v=v, vhat=vhat,
                                        beta1_power=beta1_power,
                                        beta2_power=beta2_power, lr=lr,
                                        beta1=beta1, beta2=beta2,
                                        epsilon=epsilon, grad=grad,
                                        use_locking=use_locking, name=name)
  return _op
ResourceApplyAdamWithAmsgrad = tf_export("raw_ops.ResourceApplyAdamWithAmsgrad")(_ops.to_raw_op(resource_apply_adam_with_amsgrad))


def resource_apply_adam_with_amsgrad_eager_fallback(var: Annotated[Any, _atypes.Resource], m: Annotated[Any, _atypes.Resource], v: Annotated[Any, _atypes.Resource], vhat: Annotated[Any, _atypes.Resource], beta1_power: Annotated[Any, TV_ResourceApplyAdamWithAmsgrad_T], beta2_power: Annotated[Any, TV_ResourceApplyAdamWithAmsgrad_T], lr: Annotated[Any, TV_ResourceApplyAdamWithAmsgrad_T], beta1: Annotated[Any, TV_ResourceApplyAdamWithAmsgrad_T], beta2: Annotated[Any, TV_ResourceApplyAdamWithAmsgrad_T], epsilon: Annotated[Any, TV_ResourceApplyAdamWithAmsgrad_T], grad: Annotated[Any, TV_ResourceApplyAdamWithAmsgrad_T], use_locking: bool, name, ctx):
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad) = _inputs_T
  var = _ops.convert_to_tensor(var, _dtypes.resource)
  m = _ops.convert_to_tensor(m, _dtypes.resource)
  v = _ops.convert_to_tensor(v, _dtypes.resource)
  vhat = _ops.convert_to_tensor(vhat, _dtypes.resource)
  _inputs_flat = [var, m, v, vhat, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad]
  _attrs = ("T", _attr_T, "use_locking", use_locking)
  _result = _execute.execute(b"ResourceApplyAdamWithAmsgrad", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_ResourceApplyAddSign_T = TypeVar("TV_ResourceApplyAddSign_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def resource_apply_add_sign(var: Annotated[Any, _atypes.Resource], m: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceApplyAddSign_T], alpha: Annotated[Any, TV_ResourceApplyAddSign_T], sign_decay: Annotated[Any, TV_ResourceApplyAddSign_T], beta: Annotated[Any, TV_ResourceApplyAddSign_T], grad: Annotated[Any, TV_ResourceApplyAddSign_T], use_locking:bool=False, name=None):
  r"""Update '*var' according to the AddSign update.

  m_t <- beta1 * m_{t-1} + (1 - beta1) * g
  update <- (alpha + sign_decay * sign(g) *sign(m)) * g
  variable <- variable - lr_t * update

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    m: A `Tensor` of type `resource`. Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Scaling factor. Must be a scalar.
    alpha: A `Tensor`. Must have the same type as `lr`. Must be a scalar.
    sign_decay: A `Tensor`. Must have the same type as `lr`. Must be a scalar.
    beta: A `Tensor`. Must have the same type as `lr`. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and m tensors is
      protected by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceApplyAddSign", name, var, m, lr, alpha, sign_decay,
        beta, grad, "use_locking", use_locking)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_apply_add_sign_eager_fallback(
          var, m, lr, alpha, sign_decay, beta, grad, use_locking=use_locking,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceApplyAddSign", var=var, m=m, lr=lr, alpha=alpha,
                                sign_decay=sign_decay, beta=beta, grad=grad,
                                use_locking=use_locking, name=name)
  return _op
ResourceApplyAddSign = tf_export("raw_ops.ResourceApplyAddSign")(_ops.to_raw_op(resource_apply_add_sign))


def resource_apply_add_sign_eager_fallback(var: Annotated[Any, _atypes.Resource], m: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceApplyAddSign_T], alpha: Annotated[Any, TV_ResourceApplyAddSign_T], sign_decay: Annotated[Any, TV_ResourceApplyAddSign_T], beta: Annotated[Any, TV_ResourceApplyAddSign_T], grad: Annotated[Any, TV_ResourceApplyAddSign_T], use_locking: bool, name, ctx):
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([lr, alpha, sign_decay, beta, grad], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (lr, alpha, sign_decay, beta, grad) = _inputs_T
  var = _ops.convert_to_tensor(var, _dtypes.resource)
  m = _ops.convert_to_tensor(m, _dtypes.resource)
  _inputs_flat = [var, m, lr, alpha, sign_decay, beta, grad]
  _attrs = ("T", _attr_T, "use_locking", use_locking)
  _result = _execute.execute(b"ResourceApplyAddSign", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


TV_ResourceApplyCenteredRMSProp_T = TypeVar("TV_ResourceApplyCenteredRMSProp_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def resource_apply_centered_rms_prop(var: Annotated[Any, _atypes.Resource], mg: Annotated[Any, _atypes.Resource], ms: Annotated[Any, _atypes.Resource], mom: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceApplyCenteredRMSProp_T], rho: Annotated[Any, TV_ResourceApplyCenteredRMSProp_T], momentum: Annotated[Any, TV_ResourceApplyCenteredRMSProp_T], epsilon: Annotated[Any, TV_ResourceApplyCenteredRMSProp_T], grad: Annotated[Any, TV_ResourceApplyCenteredRMSProp_T], use_locking:bool=False, name=None):
  r"""Update '*var' according to the centered RMSProp algorithm.

  The centered RMSProp algorithm uses an estimate of the centered second moment
  (i.e., the variance) for normalization, as opposed to regular RMSProp, which
  uses the (uncentered) second moment. This often helps with training, but is
  slightly more expensive in terms of computation and memory.

  Note that in dense implementation of this algorithm, mg, ms, and mom will
  update even if the grad is zero, but in this sparse implementation, mg, ms,
  and mom will not update in iterations during which the grad is zero.

  mean_square = decay * mean_square + (1-decay) * gradient ** 2
  mean_grad = decay * mean_grad + (1-decay) * gradient

  Delta = learning_rate * gradient / sqrt(mean_square + epsilon - mean_grad ** 2)

  mg <- rho * mg_{t-1} + (1-rho) * grad
  ms <- rho * ms_{t-1} + (1-rho) * grad * grad
  mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms - mg * mg + epsilon)
  var <- var - mom

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    mg: A `Tensor` of type `resource`. Should be from a Variable().
    ms: A `Tensor` of type `resource`. Should be from a Variable().
    mom: A `Tensor` of type `resource`. Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Scaling factor. Must be a scalar.
    rho: A `Tensor`. Must have the same type as `lr`.
      Decay rate. Must be a scalar.
    momentum: A `Tensor`. Must have the same type as `lr`.
      Momentum Scale. Must be a scalar.
    epsilon: A `Tensor`. Must have the same type as `lr`.
      Ridge term. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var, mg, ms, and mom tensors is
      protected by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceApplyCenteredRMSProp", name, var, mg, ms, mom, lr, rho,
        momentum, epsilon, grad, "use_locking", use_locking)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_apply_centered_rms_prop_eager_fallback(
          var, mg, ms, mom, lr, rho, momentum, epsilon, grad,
          use_locking=use_locking, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceApplyCenteredRMSProp", var=var, mg=mg, ms=ms, mom=mom, lr=lr,
                                        rho=rho, momentum=momentum,
                                        epsilon=epsilon, grad=grad,
                                        use_locking=use_locking, name=name)
  return _op
ResourceApplyCenteredRMSProp = tf_export("raw_ops.ResourceApplyCenteredRMSProp")(_ops.to_raw_op(resource_apply_centered_rms_prop))


def resource_apply_centered_rms_prop_eager_fallback(var: Annotated[Any, _atypes.Resource], mg: Annotated[Any, _atypes.Resource], ms: Annotated[Any, _atypes.Resource], mom: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceApplyCenteredRMSProp_T], rho: Annotated[Any, TV_ResourceApplyCenteredRMSProp_T], momentum: Annotated[Any, TV_ResourceApplyCenteredRMSProp_T], epsilon: Annotated[Any, TV_ResourceApplyCenteredRMSProp_T], grad: Annotated[Any, TV_ResourceApplyCenteredRMSProp_T], use_locking: bool, name, ctx):
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([lr, rho, momentum, epsilon, grad], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (lr, rho, momentum, epsilon, grad) = _inputs_T
  var = _ops.convert_to_tensor(var, _dtypes.resource)
  mg = _ops.convert_to_tensor(mg, _dtypes.resource)
  ms = _ops.convert_to_tensor(ms, _dtypes.resource)
  mom = _ops.convert_to_tensor(mom, _dtypes.resource)
  _inputs_flat = [var, mg, ms, mom, lr, rho, momentum, epsilon, grad]
  _attrs = ("T", _attr_T, "use_locking", use_locking)
  _result = _execute.execute(b"ResourceApplyCenteredRMSProp", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_ResourceApplyFtrl_T = TypeVar("TV_ResourceApplyFtrl_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def resource_apply_ftrl(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], linear: Annotated[Any, _atypes.Resource], grad: Annotated[Any, TV_ResourceApplyFtrl_T], lr: Annotated[Any, TV_ResourceApplyFtrl_T], l1: Annotated[Any, TV_ResourceApplyFtrl_T], l2: Annotated[Any, TV_ResourceApplyFtrl_T], lr_power: Annotated[Any, TV_ResourceApplyFtrl_T], use_locking:bool=False, multiply_linear_by_lr:bool=False, name=None):
  r"""Update '*var' according to the Ftrl-proximal scheme.

  accum_new = accum + grad * grad
  linear += grad - (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
  quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
  var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
  accum = accum_new

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    accum: A `Tensor` of type `resource`. Should be from a Variable().
    linear: A `Tensor` of type `resource`. Should be from a Variable().
    grad: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      The gradient.
    lr: A `Tensor`. Must have the same type as `grad`.
      Scaling factor. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `grad`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `grad`.
      L2 regularization. Must be a scalar.
    lr_power: A `Tensor`. Must have the same type as `grad`.
      Scaling factor. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    multiply_linear_by_lr: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceApplyFtrl", name, var, accum, linear, grad, lr, l1, l2,
        lr_power, "use_locking", use_locking, "multiply_linear_by_lr",
        multiply_linear_by_lr)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_apply_ftrl_eager_fallback(
          var, accum, linear, grad, lr, l1, l2, lr_power,
          use_locking=use_locking,
          multiply_linear_by_lr=multiply_linear_by_lr, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if multiply_linear_by_lr is None:
    multiply_linear_by_lr = False
  multiply_linear_by_lr = _execute.make_bool(multiply_linear_by_lr, "multiply_linear_by_lr")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceApplyFtrl", var=var, accum=accum, linear=linear, grad=grad,
                             lr=lr, l1=l1, l2=l2, lr_power=lr_power,
                             use_locking=use_locking,
                             multiply_linear_by_lr=multiply_linear_by_lr,
                             name=name)
  return _op
ResourceApplyFtrl = tf_export("raw_ops.ResourceApplyFtrl")(_ops.to_raw_op(resource_apply_ftrl))


def resource_apply_ftrl_eager_fallback(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], linear: Annotated[Any, _atypes.Resource], grad: Annotated[Any, TV_ResourceApplyFtrl_T], lr: Annotated[Any, TV_ResourceApplyFtrl_T], l1: Annotated[Any, TV_ResourceApplyFtrl_T], l2: Annotated[Any, TV_ResourceApplyFtrl_T], lr_power: Annotated[Any, TV_ResourceApplyFtrl_T], use_locking: bool, multiply_linear_by_lr: bool, name, ctx):
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if multiply_linear_by_lr is None:
    multiply_linear_by_lr = False
  multiply_linear_by_lr = _execute.make_bool(multiply_linear_by_lr, "multiply_linear_by_lr")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([grad, lr, l1, l2, lr_power], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (grad, lr, l1, l2, lr_power) = _inputs_T
  var = _ops.convert_to_tensor(var, _dtypes.resource)
  accum = _ops.convert_to_tensor(accum, _dtypes.resource)
  linear = _ops.convert_to_tensor(linear, _dtypes.resource)
  _inputs_flat = [var, accum, linear, grad, lr, l1, l2, lr_power]
  _attrs = ("T", _attr_T, "use_locking", use_locking, "multiply_linear_by_lr",
  multiply_linear_by_lr)
  _result = _execute.execute(b"ResourceApplyFtrl", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


TV_ResourceApplyFtrlV2_T = TypeVar("TV_ResourceApplyFtrlV2_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def resource_apply_ftrl_v2(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], linear: Annotated[Any, _atypes.Resource], grad: Annotated[Any, TV_ResourceApplyFtrlV2_T], lr: Annotated[Any, TV_ResourceApplyFtrlV2_T], l1: Annotated[Any, TV_ResourceApplyFtrlV2_T], l2: Annotated[Any, TV_ResourceApplyFtrlV2_T], l2_shrinkage: Annotated[Any, TV_ResourceApplyFtrlV2_T], lr_power: Annotated[Any, TV_ResourceApplyFtrlV2_T], use_locking:bool=False, multiply_linear_by_lr:bool=False, name=None):
  r"""Update '*var' according to the Ftrl-proximal scheme.

  accum_new = accum + grad * grad
  grad_with_shrinkage = grad + 2 * l2_shrinkage * var
  linear += grad_with_shrinkage +
      (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
  quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
  var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
  accum = accum_new

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    accum: A `Tensor` of type `resource`. Should be from a Variable().
    linear: A `Tensor` of type `resource`. Should be from a Variable().
    grad: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      The gradient.
    lr: A `Tensor`. Must have the same type as `grad`.
      Scaling factor. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `grad`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `grad`.
      L2 shrinkage regularization. Must be a scalar.
    l2_shrinkage: A `Tensor`. Must have the same type as `grad`.
    lr_power: A `Tensor`. Must have the same type as `grad`.
      Scaling factor. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    multiply_linear_by_lr: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceApplyFtrlV2", name, var, accum, linear, grad, lr, l1,
        l2, l2_shrinkage, lr_power, "use_locking", use_locking,
        "multiply_linear_by_lr", multiply_linear_by_lr)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_apply_ftrl_v2_eager_fallback(
          var, accum, linear, grad, lr, l1, l2, l2_shrinkage, lr_power,
          use_locking=use_locking,
          multiply_linear_by_lr=multiply_linear_by_lr, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if multiply_linear_by_lr is None:
    multiply_linear_by_lr = False
  multiply_linear_by_lr = _execute.make_bool(multiply_linear_by_lr, "multiply_linear_by_lr")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceApplyFtrlV2", var=var, accum=accum, linear=linear, grad=grad,
                               lr=lr, l1=l1, l2=l2, l2_shrinkage=l2_shrinkage,
                               lr_power=lr_power, use_locking=use_locking,
                               multiply_linear_by_lr=multiply_linear_by_lr,
                               name=name)
  return _op
ResourceApplyFtrlV2 = tf_export("raw_ops.ResourceApplyFtrlV2")(_ops.to_raw_op(resource_apply_ftrl_v2))


def resource_apply_ftrl_v2_eager_fallback(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], linear: Annotated[Any, _atypes.Resource], grad: Annotated[Any, TV_ResourceApplyFtrlV2_T], lr: Annotated[Any, TV_ResourceApplyFtrlV2_T], l1: Annotated[Any, TV_ResourceApplyFtrlV2_T], l2: Annotated[Any, TV_ResourceApplyFtrlV2_T], l2_shrinkage: Annotated[Any, TV_ResourceApplyFtrlV2_T], lr_power: Annotated[Any, TV_ResourceApplyFtrlV2_T], use_locking: bool, multiply_linear_by_lr: bool, name, ctx):
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if multiply_linear_by_lr is None:
    multiply_linear_by_lr = False
  multiply_linear_by_lr = _execute.make_bool(multiply_linear_by_lr, "multiply_linear_by_lr")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([grad, lr, l1, l2, l2_shrinkage, lr_power], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (grad, lr, l1, l2, l2_shrinkage, lr_power) = _inputs_T
  var = _ops.convert_to_tensor(var, _dtypes.resource)
  accum = _ops.convert_to_tensor(accum, _dtypes.resource)
  linear = _ops.convert_to_tensor(linear, _dtypes.resource)
  _inputs_flat = [var, accum, linear, grad, lr, l1, l2, l2_shrinkage, lr_power]
  _attrs = ("T", _attr_T, "use_locking", use_locking, "multiply_linear_by_lr",
  multiply_linear_by_lr)
  _result = _execute.execute(b"ResourceApplyFtrlV2", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


TV_ResourceApplyGradientDescent_T = TypeVar("TV_ResourceApplyGradientDescent_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def resource_apply_gradient_descent(var: Annotated[Any, _atypes.Resource], alpha: Annotated[Any, TV_ResourceApplyGradientDescent_T], delta: Annotated[Any, TV_ResourceApplyGradientDescent_T], use_locking:bool=False, name=None):
  r"""Update '*var' by subtracting 'alpha' * 'delta' from it.

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    alpha: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Scaling factor. Must be a scalar.
    delta: A `Tensor`. Must have the same type as `alpha`. The change.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, the subtraction will be protected by a lock;
      otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceApplyGradientDescent", name, var, alpha, delta,
        "use_locking", use_locking)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_apply_gradient_descent_eager_fallback(
          var, alpha, delta, use_locking=use_locking, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceApplyGradientDescent", var=var, alpha=alpha, delta=delta,
                                        use_locking=use_locking, name=name)
  return _op
ResourceApplyGradientDescent = tf_export("raw_ops.ResourceApplyGradientDescent")(_ops.to_raw_op(resource_apply_gradient_descent))


def resource_apply_gradient_descent_eager_fallback(var: Annotated[Any, _atypes.Resource], alpha: Annotated[Any, TV_ResourceApplyGradientDescent_T], delta: Annotated[Any, TV_ResourceApplyGradientDescent_T], use_locking: bool, name, ctx):
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([alpha, delta], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (alpha, delta) = _inputs_T
  var = _ops.convert_to_tensor(var, _dtypes.resource)
  _inputs_flat = [var, alpha, delta]
  _attrs = ("T", _attr_T, "use_locking", use_locking)
  _result = _execute.execute(b"ResourceApplyGradientDescent", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_ResourceApplyKerasMomentum_T = TypeVar("TV_ResourceApplyKerasMomentum_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def resource_apply_keras_momentum(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceApplyKerasMomentum_T], grad: Annotated[Any, TV_ResourceApplyKerasMomentum_T], momentum: Annotated[Any, TV_ResourceApplyKerasMomentum_T], use_locking:bool=False, use_nesterov:bool=False, name=None):
  r"""Update '*var' according to the momentum scheme.

  Set use_nesterov = True if you want to use Nesterov momentum.

  accum = accum * momentum - lr * grad
  var += accum

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    accum: A `Tensor` of type `resource`. Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Scaling factor. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    momentum: A `Tensor`. Must have the same type as `lr`.
      Momentum. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    use_nesterov: An optional `bool`. Defaults to `False`.
      If `True`, the tensor passed to compute grad will be
      var + momentum * accum, so in the end, the var you get is actually
      var + momentum * accum.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceApplyKerasMomentum", name, var, accum, lr, grad,
        momentum, "use_locking", use_locking, "use_nesterov", use_nesterov)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_apply_keras_momentum_eager_fallback(
          var, accum, lr, grad, momentum, use_locking=use_locking,
          use_nesterov=use_nesterov, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if use_nesterov is None:
    use_nesterov = False
  use_nesterov = _execute.make_bool(use_nesterov, "use_nesterov")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceApplyKerasMomentum", var=var, accum=accum, lr=lr, grad=grad,
                                      momentum=momentum,
                                      use_locking=use_locking,
                                      use_nesterov=use_nesterov, name=name)
  return _op
ResourceApplyKerasMomentum = tf_export("raw_ops.ResourceApplyKerasMomentum")(_ops.to_raw_op(resource_apply_keras_momentum))


def resource_apply_keras_momentum_eager_fallback(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceApplyKerasMomentum_T], grad: Annotated[Any, TV_ResourceApplyKerasMomentum_T], momentum: Annotated[Any, TV_ResourceApplyKerasMomentum_T], use_locking: bool, use_nesterov: bool, name, ctx):
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if use_nesterov is None:
    use_nesterov = False
  use_nesterov = _execute.make_bool(use_nesterov, "use_nesterov")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([lr, grad, momentum], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (lr, grad, momentum) = _inputs_T
  var = _ops.convert_to_tensor(var, _dtypes.resource)
  accum = _ops.convert_to_tensor(accum, _dtypes.resource)
  _inputs_flat = [var, accum, lr, grad, momentum]
  _attrs = ("T", _attr_T, "use_locking", use_locking, "use_nesterov",
  use_nesterov)
  _result = _execute.execute(b"ResourceApplyKerasMomentum", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_ResourceApplyMomentum_T = TypeVar("TV_ResourceApplyMomentum_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def resource_apply_momentum(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceApplyMomentum_T], grad: Annotated[Any, TV_ResourceApplyMomentum_T], momentum: Annotated[Any, TV_ResourceApplyMomentum_T], use_locking:bool=False, use_nesterov:bool=False, name=None):
  r"""Update '*var' according to the momentum scheme.

  Set use_nesterov = True if you want to use Nesterov momentum.

  accum = accum * momentum + grad
  var -= lr * accum

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    accum: A `Tensor` of type `resource`. Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Scaling factor. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    momentum: A `Tensor`. Must have the same type as `lr`.
      Momentum. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    use_nesterov: An optional `bool`. Defaults to `False`.
      If `True`, the tensor passed to compute grad will be
      var - lr * momentum * accum, so in the end, the var you get is actually
      var - lr * momentum * accum.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceApplyMomentum", name, var, accum, lr, grad, momentum,
        "use_locking", use_locking, "use_nesterov", use_nesterov)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_apply_momentum_eager_fallback(
          var, accum, lr, grad, momentum, use_locking=use_locking,
          use_nesterov=use_nesterov, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if use_nesterov is None:
    use_nesterov = False
  use_nesterov = _execute.make_bool(use_nesterov, "use_nesterov")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceApplyMomentum", var=var, accum=accum, lr=lr, grad=grad,
                                 momentum=momentum, use_locking=use_locking,
                                 use_nesterov=use_nesterov, name=name)
  return _op
ResourceApplyMomentum = tf_export("raw_ops.ResourceApplyMomentum")(_ops.to_raw_op(resource_apply_momentum))


def resource_apply_momentum_eager_fallback(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceApplyMomentum_T], grad: Annotated[Any, TV_ResourceApplyMomentum_T], momentum: Annotated[Any, TV_ResourceApplyMomentum_T], use_locking: bool, use_nesterov: bool, name, ctx):
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if use_nesterov is None:
    use_nesterov = False
  use_nesterov = _execute.make_bool(use_nesterov, "use_nesterov")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([lr, grad, momentum], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (lr, grad, momentum) = _inputs_T
  var = _ops.convert_to_tensor(var, _dtypes.resource)
  accum = _ops.convert_to_tensor(accum, _dtypes.resource)
  _inputs_flat = [var, accum, lr, grad, momentum]
  _attrs = ("T", _attr_T, "use_locking", use_locking, "use_nesterov",
  use_nesterov)
  _result = _execute.execute(b"ResourceApplyMomentum", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


TV_ResourceApplyPowerSign_T = TypeVar("TV_ResourceApplyPowerSign_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def resource_apply_power_sign(var: Annotated[Any, _atypes.Resource], m: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceApplyPowerSign_T], logbase: Annotated[Any, TV_ResourceApplyPowerSign_T], sign_decay: Annotated[Any, TV_ResourceApplyPowerSign_T], beta: Annotated[Any, TV_ResourceApplyPowerSign_T], grad: Annotated[Any, TV_ResourceApplyPowerSign_T], use_locking:bool=False, name=None):
  r"""Update '*var' according to the AddSign update.

  m_t <- beta1 * m_{t-1} + (1 - beta1) * g
  update <- exp(logbase * sign_decay * sign(g) * sign(m_t)) * g
  variable <- variable - lr_t * update

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    m: A `Tensor` of type `resource`. Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Scaling factor. Must be a scalar.
    logbase: A `Tensor`. Must have the same type as `lr`. Must be a scalar.
    sign_decay: A `Tensor`. Must have the same type as `lr`. Must be a scalar.
    beta: A `Tensor`. Must have the same type as `lr`. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and m tensors is
      protected by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceApplyPowerSign", name, var, m, lr, logbase, sign_decay,
        beta, grad, "use_locking", use_locking)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_apply_power_sign_eager_fallback(
          var, m, lr, logbase, sign_decay, beta, grad,
          use_locking=use_locking, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceApplyPowerSign", var=var, m=m, lr=lr, logbase=logbase,
                                  sign_decay=sign_decay, beta=beta, grad=grad,
                                  use_locking=use_locking, name=name)
  return _op
ResourceApplyPowerSign = tf_export("raw_ops.ResourceApplyPowerSign")(_ops.to_raw_op(resource_apply_power_sign))


def resource_apply_power_sign_eager_fallback(var: Annotated[Any, _atypes.Resource], m: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceApplyPowerSign_T], logbase: Annotated[Any, TV_ResourceApplyPowerSign_T], sign_decay: Annotated[Any, TV_ResourceApplyPowerSign_T], beta: Annotated[Any, TV_ResourceApplyPowerSign_T], grad: Annotated[Any, TV_ResourceApplyPowerSign_T], use_locking: bool, name, ctx):
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([lr, logbase, sign_decay, beta, grad], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (lr, logbase, sign_decay, beta, grad) = _inputs_T
  var = _ops.convert_to_tensor(var, _dtypes.resource)
  m = _ops.convert_to_tensor(m, _dtypes.resource)
  _inputs_flat = [var, m, lr, logbase, sign_decay, beta, grad]
  _attrs = ("T", _attr_T, "use_locking", use_locking)
  _result = _execute.execute(b"ResourceApplyPowerSign", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_ResourceApplyProximalAdagrad_T = TypeVar("TV_ResourceApplyProximalAdagrad_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def resource_apply_proximal_adagrad(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceApplyProximalAdagrad_T], l1: Annotated[Any, TV_ResourceApplyProximalAdagrad_T], l2: Annotated[Any, TV_ResourceApplyProximalAdagrad_T], grad: Annotated[Any, TV_ResourceApplyProximalAdagrad_T], use_locking:bool=False, name=None):
  r"""Update '*var' and '*accum' according to FOBOS with Adagrad learning rate.

  accum += grad * grad
  prox_v = var - lr * grad * (1 / sqrt(accum))
  var = sign(prox_v)/(1+lr*l2) * max{|prox_v|-lr*l1,0}

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    accum: A `Tensor` of type `resource`. Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Scaling factor. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `lr`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `lr`.
      L2 regularization. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, updating of the var and accum tensors will be protected by
      a lock; otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceApplyProximalAdagrad", name, var, accum, lr, l1, l2,
        grad, "use_locking", use_locking)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_apply_proximal_adagrad_eager_fallback(
          var, accum, lr, l1, l2, grad, use_locking=use_locking, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceApplyProximalAdagrad", var=var, accum=accum, lr=lr, l1=l1,
                                        l2=l2, grad=grad,
                                        use_locking=use_locking, name=name)
  return _op
ResourceApplyProximalAdagrad = tf_export("raw_ops.ResourceApplyProximalAdagrad")(_ops.to_raw_op(resource_apply_proximal_adagrad))


def resource_apply_proximal_adagrad_eager_fallback(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceApplyProximalAdagrad_T], l1: Annotated[Any, TV_ResourceApplyProximalAdagrad_T], l2: Annotated[Any, TV_ResourceApplyProximalAdagrad_T], grad: Annotated[Any, TV_ResourceApplyProximalAdagrad_T], use_locking: bool, name, ctx):
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([lr, l1, l2, grad], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (lr, l1, l2, grad) = _inputs_T
  var = _ops.convert_to_tensor(var, _dtypes.resource)
  accum = _ops.convert_to_tensor(accum, _dtypes.resource)
  _inputs_flat = [var, accum, lr, l1, l2, grad]
  _attrs = ("T", _attr_T, "use_locking", use_locking)
  _result = _execute.execute(b"ResourceApplyProximalAdagrad", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_ResourceApplyProximalGradientDescent_T = TypeVar("TV_ResourceApplyProximalGradientDescent_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def resource_apply_proximal_gradient_descent(var: Annotated[Any, _atypes.Resource], alpha: Annotated[Any, TV_ResourceApplyProximalGradientDescent_T], l1: Annotated[Any, TV_ResourceApplyProximalGradientDescent_T], l2: Annotated[Any, TV_ResourceApplyProximalGradientDescent_T], delta: Annotated[Any, TV_ResourceApplyProximalGradientDescent_T], use_locking:bool=False, name=None):
  r"""Update '*var' as FOBOS algorithm with fixed learning rate.

  prox_v = var - alpha * delta
  var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    alpha: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Scaling factor. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `alpha`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `alpha`.
      L2 regularization. Must be a scalar.
    delta: A `Tensor`. Must have the same type as `alpha`. The change.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, the subtraction will be protected by a lock;
      otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceApplyProximalGradientDescent", name, var, alpha, l1,
        l2, delta, "use_locking", use_locking)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_apply_proximal_gradient_descent_eager_fallback(
          var, alpha, l1, l2, delta, use_locking=use_locking, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceApplyProximalGradientDescent", var=var, alpha=alpha, l1=l1,
                                                l2=l2, delta=delta,
                                                use_locking=use_locking,
                                                name=name)
  return _op
ResourceApplyProximalGradientDescent = tf_export("raw_ops.ResourceApplyProximalGradientDescent")(_ops.to_raw_op(resource_apply_proximal_gradient_descent))


def resource_apply_proximal_gradient_descent_eager_fallback(var: Annotated[Any, _atypes.Resource], alpha: Annotated[Any, TV_ResourceApplyProximalGradientDescent_T], l1: Annotated[Any, TV_ResourceApplyProximalGradientDescent_T], l2: Annotated[Any, TV_ResourceApplyProximalGradientDescent_T], delta: Annotated[Any, TV_ResourceApplyProximalGradientDescent_T], use_locking: bool, name, ctx):
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([alpha, l1, l2, delta], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (alpha, l1, l2, delta) = _inputs_T
  var = _ops.convert_to_tensor(var, _dtypes.resource)
  _inputs_flat = [var, alpha, l1, l2, delta]
  _attrs = ("T", _attr_T, "use_locking", use_locking)
  _result = _execute.execute(b"ResourceApplyProximalGradientDescent", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_ResourceApplyRMSProp_T = TypeVar("TV_ResourceApplyRMSProp_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def resource_apply_rms_prop(var: Annotated[Any, _atypes.Resource], ms: Annotated[Any, _atypes.Resource], mom: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceApplyRMSProp_T], rho: Annotated[Any, TV_ResourceApplyRMSProp_T], momentum: Annotated[Any, TV_ResourceApplyRMSProp_T], epsilon: Annotated[Any, TV_ResourceApplyRMSProp_T], grad: Annotated[Any, TV_ResourceApplyRMSProp_T], use_locking:bool=False, name=None):
  r"""Update '*var' according to the RMSProp algorithm.

  Note that in dense implementation of this algorithm, ms and mom will
  update even if the grad is zero, but in this sparse implementation, ms
  and mom will not update in iterations during which the grad is zero.

  mean_square = decay * mean_square + (1-decay) * gradient ** 2
  Delta = learning_rate * gradient / sqrt(mean_square + epsilon)

  ms <- rho * ms_{t-1} + (1-rho) * grad * grad
  mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
  var <- var - mom

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    ms: A `Tensor` of type `resource`. Should be from a Variable().
    mom: A `Tensor` of type `resource`. Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Scaling factor. Must be a scalar.
    rho: A `Tensor`. Must have the same type as `lr`.
      Decay rate. Must be a scalar.
    momentum: A `Tensor`. Must have the same type as `lr`.
    epsilon: A `Tensor`. Must have the same type as `lr`.
      Ridge term. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var, ms, and mom tensors is protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceApplyRMSProp", name, var, ms, mom, lr, rho, momentum,
        epsilon, grad, "use_locking", use_locking)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_apply_rms_prop_eager_fallback(
          var, ms, mom, lr, rho, momentum, epsilon, grad,
          use_locking=use_locking, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceApplyRMSProp", var=var, ms=ms, mom=mom, lr=lr, rho=rho,
                                momentum=momentum, epsilon=epsilon, grad=grad,
                                use_locking=use_locking, name=name)
  return _op
ResourceApplyRMSProp = tf_export("raw_ops.ResourceApplyRMSProp")(_ops.to_raw_op(resource_apply_rms_prop))


def resource_apply_rms_prop_eager_fallback(var: Annotated[Any, _atypes.Resource], ms: Annotated[Any, _atypes.Resource], mom: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceApplyRMSProp_T], rho: Annotated[Any, TV_ResourceApplyRMSProp_T], momentum: Annotated[Any, TV_ResourceApplyRMSProp_T], epsilon: Annotated[Any, TV_ResourceApplyRMSProp_T], grad: Annotated[Any, TV_ResourceApplyRMSProp_T], use_locking: bool, name, ctx):
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([lr, rho, momentum, epsilon, grad], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (lr, rho, momentum, epsilon, grad) = _inputs_T
  var = _ops.convert_to_tensor(var, _dtypes.resource)
  ms = _ops.convert_to_tensor(ms, _dtypes.resource)
  mom = _ops.convert_to_tensor(mom, _dtypes.resource)
  _inputs_flat = [var, ms, mom, lr, rho, momentum, epsilon, grad]
  _attrs = ("T", _attr_T, "use_locking", use_locking)
  _result = _execute.execute(b"ResourceApplyRMSProp", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


TV_ResourceSparseApplyAdadelta_T = TypeVar("TV_ResourceSparseApplyAdadelta_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_ResourceSparseApplyAdadelta_Tindices = TypeVar("TV_ResourceSparseApplyAdadelta_Tindices", _atypes.Int32, _atypes.Int64)

def resource_sparse_apply_adadelta(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], accum_update: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceSparseApplyAdadelta_T], rho: Annotated[Any, TV_ResourceSparseApplyAdadelta_T], epsilon: Annotated[Any, TV_ResourceSparseApplyAdadelta_T], grad: Annotated[Any, TV_ResourceSparseApplyAdadelta_T], indices: Annotated[Any, TV_ResourceSparseApplyAdadelta_Tindices], use_locking:bool=False, name=None):
  r"""var: Should be from a Variable().

  Args:
    var: A `Tensor` of type `resource`.
    accum: A `Tensor` of type `resource`. Should be from a Variable().
    accum_update: A `Tensor` of type `resource`.
      : Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Learning rate. Must be a scalar.
    rho: A `Tensor`. Must have the same type as `lr`.
      Decay factor. Must be a scalar.
    epsilon: A `Tensor`. Must have the same type as `lr`.
      Constant factor. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, updating of the var and accum tensors will be protected by
      a lock; otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceSparseApplyAdadelta", name, var, accum, accum_update,
        lr, rho, epsilon, grad, indices, "use_locking", use_locking)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_sparse_apply_adadelta_eager_fallback(
          var, accum, accum_update, lr, rho, epsilon, grad, indices,
          use_locking=use_locking, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceSparseApplyAdadelta", var=var, accum=accum,
                                       accum_update=accum_update, lr=lr,
                                       rho=rho, epsilon=epsilon, grad=grad,
                                       indices=indices,
                                       use_locking=use_locking, name=name)
  return _op
ResourceSparseApplyAdadelta = tf_export("raw_ops.ResourceSparseApplyAdadelta")(_ops.to_raw_op(resource_sparse_apply_adadelta))


def resource_sparse_apply_adadelta_eager_fallback(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], accum_update: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceSparseApplyAdadelta_T], rho: Annotated[Any, TV_ResourceSparseApplyAdadelta_T], epsilon: Annotated[Any, TV_ResourceSparseApplyAdadelta_T], grad: Annotated[Any, TV_ResourceSparseApplyAdadelta_T], indices: Annotated[Any, TV_ResourceSparseApplyAdadelta_Tindices], use_locking: bool, name, ctx):
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([lr, rho, epsilon, grad], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (lr, rho, epsilon, grad) = _inputs_T
  _attr_Tindices, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ])
  var = _ops.convert_to_tensor(var, _dtypes.resource)
  accum = _ops.convert_to_tensor(accum, _dtypes.resource)
  accum_update = _ops.convert_to_tensor(accum_update, _dtypes.resource)
  _inputs_flat = [var, accum, accum_update, lr, rho, epsilon, grad, indices]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "use_locking",
  use_locking)
  _result = _execute.execute(b"ResourceSparseApplyAdadelta", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_ResourceSparseApplyAdagrad_T = TypeVar("TV_ResourceSparseApplyAdagrad_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_ResourceSparseApplyAdagrad_Tindices = TypeVar("TV_ResourceSparseApplyAdagrad_Tindices", _atypes.Int32, _atypes.Int64)

def resource_sparse_apply_adagrad(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceSparseApplyAdagrad_T], grad: Annotated[Any, TV_ResourceSparseApplyAdagrad_T], indices: Annotated[Any, TV_ResourceSparseApplyAdagrad_Tindices], use_locking:bool=False, update_slots:bool=True, name=None):
  r"""Update relevant entries in '*var' and '*accum' according to the adagrad scheme.

  That is for rows we have grad for, we update var and accum as follows:
  accum += grad * grad
  var -= lr * grad * (1 / sqrt(accum))

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    accum: A `Tensor` of type `resource`. Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Learning rate. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    update_slots: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceSparseApplyAdagrad", name, var, accum, lr, grad,
        indices, "use_locking", use_locking, "update_slots", update_slots)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_sparse_apply_adagrad_eager_fallback(
          var, accum, lr, grad, indices, use_locking=use_locking,
          update_slots=update_slots, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if update_slots is None:
    update_slots = True
  update_slots = _execute.make_bool(update_slots, "update_slots")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceSparseApplyAdagrad", var=var, accum=accum, lr=lr, grad=grad,
                                      indices=indices,
                                      use_locking=use_locking,
                                      update_slots=update_slots, name=name)
  return _op
ResourceSparseApplyAdagrad = tf_export("raw_ops.ResourceSparseApplyAdagrad")(_ops.to_raw_op(resource_sparse_apply_adagrad))


def resource_sparse_apply_adagrad_eager_fallback(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceSparseApplyAdagrad_T], grad: Annotated[Any, TV_ResourceSparseApplyAdagrad_T], indices: Annotated[Any, TV_ResourceSparseApplyAdagrad_Tindices], use_locking: bool, update_slots: bool, name, ctx):
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if update_slots is None:
    update_slots = True
  update_slots = _execute.make_bool(update_slots, "update_slots")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([lr, grad], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (lr, grad) = _inputs_T
  _attr_Tindices, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ])
  var = _ops.convert_to_tensor(var, _dtypes.resource)
  accum = _ops.convert_to_tensor(accum, _dtypes.resource)
  _inputs_flat = [var, accum, lr, grad, indices]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "use_locking",
  use_locking, "update_slots", update_slots)
  _result = _execute.execute(b"ResourceSparseApplyAdagrad", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_ResourceSparseApplyAdagradDA_T = TypeVar("TV_ResourceSparseApplyAdagradDA_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_ResourceSparseApplyAdagradDA_Tindices = TypeVar("TV_ResourceSparseApplyAdagradDA_Tindices", _atypes.Int32, _atypes.Int64)

def resource_sparse_apply_adagrad_da(var: Annotated[Any, _atypes.Resource], gradient_accumulator: Annotated[Any, _atypes.Resource], gradient_squared_accumulator: Annotated[Any, _atypes.Resource], grad: Annotated[Any, TV_ResourceSparseApplyAdagradDA_T], indices: Annotated[Any, TV_ResourceSparseApplyAdagradDA_Tindices], lr: Annotated[Any, TV_ResourceSparseApplyAdagradDA_T], l1: Annotated[Any, TV_ResourceSparseApplyAdagradDA_T], l2: Annotated[Any, TV_ResourceSparseApplyAdagradDA_T], global_step: Annotated[Any, _atypes.Int64], use_locking:bool=False, name=None):
  r"""Update entries in '*var' and '*accum' according to the proximal adagrad scheme.

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    gradient_accumulator: A `Tensor` of type `resource`.
      Should be from a Variable().
    gradient_squared_accumulator: A `Tensor` of type `resource`.
      Should be from a Variable().
    grad: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    lr: A `Tensor`. Must have the same type as `grad`.
      Learning rate. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `grad`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `grad`.
      L2 regularization. Must be a scalar.
    global_step: A `Tensor` of type `int64`.
      Training step number. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, updating of the var and accum tensors will be protected by
      a lock; otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceSparseApplyAdagradDA", name, var, gradient_accumulator,
        gradient_squared_accumulator, grad, indices, lr, l1, l2, global_step,
        "use_locking", use_locking)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_sparse_apply_adagrad_da_eager_fallback(
          var, gradient_accumulator, gradient_squared_accumulator, grad,
          indices, lr, l1, l2, global_step, use_locking=use_locking,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceSparseApplyAdagradDA", var=var,
                                        gradient_accumulator=gradient_accumulator,
                                        gradient_squared_accumulator=gradient_squared_accumulator,
                                        grad=grad, indices=indices, lr=lr,
                                        l1=l1, l2=l2, global_step=global_step,
                                        use_locking=use_locking, name=name)
  return _op
ResourceSparseApplyAdagradDA = tf_export("raw_ops.ResourceSparseApplyAdagradDA")(_ops.to_raw_op(resource_sparse_apply_adagrad_da))


def resource_sparse_apply_adagrad_da_eager_fallback(var: Annotated[Any, _atypes.Resource], gradient_accumulator: Annotated[Any, _atypes.Resource], gradient_squared_accumulator: Annotated[Any, _atypes.Resource], grad: Annotated[Any, TV_ResourceSparseApplyAdagradDA_T], indices: Annotated[Any, TV_ResourceSparseApplyAdagradDA_Tindices], lr: Annotated[Any, TV_ResourceSparseApplyAdagradDA_T], l1: Annotated[Any, TV_ResourceSparseApplyAdagradDA_T], l2: Annotated[Any, TV_ResourceSparseApplyAdagradDA_T], global_step: Annotated[Any, _atypes.Int64], use_locking: bool, name, ctx):
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([grad, lr, l1, l2], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (grad, lr, l1, l2) = _inputs_T
  _attr_Tindices, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ])
  var = _ops.convert_to_tensor(var, _dtypes.resource)
  gradient_accumulator = _ops.convert_to_tensor(gradient_accumulator, _dtypes.resource)
  gradient_squared_accumulator = _ops.convert_to_tensor(gradient_squared_accumulator, _dtypes.resource)
  global_step = _ops.convert_to_tensor(global_step, _dtypes.int64)
  _inputs_flat = [var, gradient_accumulator, gradient_squared_accumulator, grad, indices, lr, l1, l2, global_step]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "use_locking",
  use_locking)
  _result = _execute.execute(b"ResourceSparseApplyAdagradDA", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_ResourceSparseApplyAdagradV2_T = TypeVar("TV_ResourceSparseApplyAdagradV2_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_ResourceSparseApplyAdagradV2_Tindices = TypeVar("TV_ResourceSparseApplyAdagradV2_Tindices", _atypes.Int32, _atypes.Int64)

def resource_sparse_apply_adagrad_v2(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceSparseApplyAdagradV2_T], epsilon: Annotated[Any, TV_ResourceSparseApplyAdagradV2_T], grad: Annotated[Any, TV_ResourceSparseApplyAdagradV2_T], indices: Annotated[Any, TV_ResourceSparseApplyAdagradV2_Tindices], use_locking:bool=False, update_slots:bool=True, name=None):
  r"""Update relevant entries in '*var' and '*accum' according to the adagrad scheme.

  That is for rows we have grad for, we update var and accum as follows:
  accum += grad * grad
  var -= lr * grad * (1 / sqrt(accum))

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    accum: A `Tensor` of type `resource`. Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Learning rate. Must be a scalar.
    epsilon: A `Tensor`. Must have the same type as `lr`.
      Constant factor. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    update_slots: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceSparseApplyAdagradV2", name, var, accum, lr, epsilon,
        grad, indices, "use_locking", use_locking, "update_slots",
        update_slots)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_sparse_apply_adagrad_v2_eager_fallback(
          var, accum, lr, epsilon, grad, indices, use_locking=use_locking,
          update_slots=update_slots, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if update_slots is None:
    update_slots = True
  update_slots = _execute.make_bool(update_slots, "update_slots")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceSparseApplyAdagradV2", var=var, accum=accum, lr=lr,
                                        epsilon=epsilon, grad=grad,
                                        indices=indices,
                                        use_locking=use_locking,
                                        update_slots=update_slots, name=name)
  return _op
ResourceSparseApplyAdagradV2 = tf_export("raw_ops.ResourceSparseApplyAdagradV2")(_ops.to_raw_op(resource_sparse_apply_adagrad_v2))


def resource_sparse_apply_adagrad_v2_eager_fallback(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceSparseApplyAdagradV2_T], epsilon: Annotated[Any, TV_ResourceSparseApplyAdagradV2_T], grad: Annotated[Any, TV_ResourceSparseApplyAdagradV2_T], indices: Annotated[Any, TV_ResourceSparseApplyAdagradV2_Tindices], use_locking: bool, update_slots: bool, name, ctx):
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if update_slots is None:
    update_slots = True
  update_slots = _execute.make_bool(update_slots, "update_slots")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([lr, epsilon, grad], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (lr, epsilon, grad) = _inputs_T
  _attr_Tindices, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ])
  var = _ops.convert_to_tensor(var, _dtypes.resource)
  accum = _ops.convert_to_tensor(accum, _dtypes.resource)
  _inputs_flat = [var, accum, lr, epsilon, grad, indices]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "use_locking",
  use_locking, "update_slots", update_slots)
  _result = _execute.execute(b"ResourceSparseApplyAdagradV2", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_ResourceSparseApplyCenteredRMSProp_T = TypeVar("TV_ResourceSparseApplyCenteredRMSProp_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_ResourceSparseApplyCenteredRMSProp_Tindices = TypeVar("TV_ResourceSparseApplyCenteredRMSProp_Tindices", _atypes.Int32, _atypes.Int64)

def resource_sparse_apply_centered_rms_prop(var: Annotated[Any, _atypes.Resource], mg: Annotated[Any, _atypes.Resource], ms: Annotated[Any, _atypes.Resource], mom: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceSparseApplyCenteredRMSProp_T], rho: Annotated[Any, TV_ResourceSparseApplyCenteredRMSProp_T], momentum: Annotated[Any, TV_ResourceSparseApplyCenteredRMSProp_T], epsilon: Annotated[Any, TV_ResourceSparseApplyCenteredRMSProp_T], grad: Annotated[Any, TV_ResourceSparseApplyCenteredRMSProp_T], indices: Annotated[Any, TV_ResourceSparseApplyCenteredRMSProp_Tindices], use_locking:bool=False, name=None):
  r"""Update '*var' according to the centered RMSProp algorithm.

  The centered RMSProp algorithm uses an estimate of the centered second moment
  (i.e., the variance) for normalization, as opposed to regular RMSProp, which
  uses the (uncentered) second moment. This often helps with training, but is
  slightly more expensive in terms of computation and memory.

  Note that in dense implementation of this algorithm, mg, ms, and mom will
  update even if the grad is zero, but in this sparse implementation, mg, ms,
  and mom will not update in iterations during which the grad is zero.

  mean_square = decay * mean_square + (1-decay) * gradient ** 2
  mean_grad = decay * mean_grad + (1-decay) * gradient
  Delta = learning_rate * gradient / sqrt(mean_square + epsilon - mean_grad ** 2)

  ms <- rho * ms_{t-1} + (1-rho) * grad * grad
  mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
  var <- var - mom

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    mg: A `Tensor` of type `resource`. Should be from a Variable().
    ms: A `Tensor` of type `resource`. Should be from a Variable().
    mom: A `Tensor` of type `resource`. Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Scaling factor. Must be a scalar.
    rho: A `Tensor`. Must have the same type as `lr`.
      Decay rate. Must be a scalar.
    momentum: A `Tensor`. Must have the same type as `lr`.
    epsilon: A `Tensor`. Must have the same type as `lr`.
      Ridge term. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var, ms and mom.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var, mg, ms, and mom tensors is
      protected by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceSparseApplyCenteredRMSProp", name, var, mg, ms, mom,
        lr, rho, momentum, epsilon, grad, indices, "use_locking", use_locking)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_sparse_apply_centered_rms_prop_eager_fallback(
          var, mg, ms, mom, lr, rho, momentum, epsilon, grad, indices,
          use_locking=use_locking, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceSparseApplyCenteredRMSProp", var=var, mg=mg, ms=ms, mom=mom,
                                              lr=lr, rho=rho,
                                              momentum=momentum,
                                              epsilon=epsilon, grad=grad,
                                              indices=indices,
                                              use_locking=use_locking,
                                              name=name)
  return _op
ResourceSparseApplyCenteredRMSProp = tf_export("raw_ops.ResourceSparseApplyCenteredRMSProp")(_ops.to_raw_op(resource_sparse_apply_centered_rms_prop))


def resource_sparse_apply_centered_rms_prop_eager_fallback(var: Annotated[Any, _atypes.Resource], mg: Annotated[Any, _atypes.Resource], ms: Annotated[Any, _atypes.Resource], mom: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceSparseApplyCenteredRMSProp_T], rho: Annotated[Any, TV_ResourceSparseApplyCenteredRMSProp_T], momentum: Annotated[Any, TV_ResourceSparseApplyCenteredRMSProp_T], epsilon: Annotated[Any, TV_ResourceSparseApplyCenteredRMSProp_T], grad: Annotated[Any, TV_ResourceSparseApplyCenteredRMSProp_T], indices: Annotated[Any, TV_ResourceSparseApplyCenteredRMSProp_Tindices], use_locking: bool, name, ctx):
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([lr, rho, momentum, epsilon, grad], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (lr, rho, momentum, epsilon, grad) = _inputs_T
  _attr_Tindices, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ])
  var = _ops.convert_to_tensor(var, _dtypes.resource)
  mg = _ops.convert_to_tensor(mg, _dtypes.resource)
  ms = _ops.convert_to_tensor(ms, _dtypes.resource)
  mom = _ops.convert_to_tensor(mom, _dtypes.resource)
  _inputs_flat = [var, mg, ms, mom, lr, rho, momentum, epsilon, grad, indices]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "use_locking",
  use_locking)
  _result = _execute.execute(b"ResourceSparseApplyCenteredRMSProp", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_ResourceSparseApplyFtrl_T = TypeVar("TV_ResourceSparseApplyFtrl_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_ResourceSparseApplyFtrl_Tindices = TypeVar("TV_ResourceSparseApplyFtrl_Tindices", _atypes.Int32, _atypes.Int64)

def resource_sparse_apply_ftrl(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], linear: Annotated[Any, _atypes.Resource], grad: Annotated[Any, TV_ResourceSparseApplyFtrl_T], indices: Annotated[Any, TV_ResourceSparseApplyFtrl_Tindices], lr: Annotated[Any, TV_ResourceSparseApplyFtrl_T], l1: Annotated[Any, TV_ResourceSparseApplyFtrl_T], l2: Annotated[Any, TV_ResourceSparseApplyFtrl_T], lr_power: Annotated[Any, TV_ResourceSparseApplyFtrl_T], use_locking:bool=False, multiply_linear_by_lr:bool=False, name=None):
  r"""Update relevant entries in '*var' according to the Ftrl-proximal scheme.

  That is for rows we have grad for, we update var, accum and linear as follows:
  accum_new = accum + grad * grad
  linear += grad - (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
  quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
  var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
  accum = accum_new

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    accum: A `Tensor` of type `resource`. Should be from a Variable().
    linear: A `Tensor` of type `resource`. Should be from a Variable().
    grad: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    lr: A `Tensor`. Must have the same type as `grad`.
      Scaling factor. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `grad`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `grad`.
      L2 regularization. Must be a scalar.
    lr_power: A `Tensor`. Must have the same type as `grad`.
      Scaling factor. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    multiply_linear_by_lr: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceSparseApplyFtrl", name, var, accum, linear, grad,
        indices, lr, l1, l2, lr_power, "use_locking", use_locking,
        "multiply_linear_by_lr", multiply_linear_by_lr)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_sparse_apply_ftrl_eager_fallback(
          var, accum, linear, grad, indices, lr, l1, l2, lr_power,
          use_locking=use_locking,
          multiply_linear_by_lr=multiply_linear_by_lr, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if multiply_linear_by_lr is None:
    multiply_linear_by_lr = False
  multiply_linear_by_lr = _execute.make_bool(multiply_linear_by_lr, "multiply_linear_by_lr")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceSparseApplyFtrl", var=var, accum=accum, linear=linear,
                                   grad=grad, indices=indices, lr=lr, l1=l1,
                                   l2=l2, lr_power=lr_power,
                                   use_locking=use_locking,
                                   multiply_linear_by_lr=multiply_linear_by_lr,
                                   name=name)
  return _op
ResourceSparseApplyFtrl = tf_export("raw_ops.ResourceSparseApplyFtrl")(_ops.to_raw_op(resource_sparse_apply_ftrl))


def resource_sparse_apply_ftrl_eager_fallback(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], linear: Annotated[Any, _atypes.Resource], grad: Annotated[Any, TV_ResourceSparseApplyFtrl_T], indices: Annotated[Any, TV_ResourceSparseApplyFtrl_Tindices], lr: Annotated[Any, TV_ResourceSparseApplyFtrl_T], l1: Annotated[Any, TV_ResourceSparseApplyFtrl_T], l2: Annotated[Any, TV_ResourceSparseApplyFtrl_T], lr_power: Annotated[Any, TV_ResourceSparseApplyFtrl_T], use_locking: bool, multiply_linear_by_lr: bool, name, ctx):
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if multiply_linear_by_lr is None:
    multiply_linear_by_lr = False
  multiply_linear_by_lr = _execute.make_bool(multiply_linear_by_lr, "multiply_linear_by_lr")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([grad, lr, l1, l2, lr_power], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (grad, lr, l1, l2, lr_power) = _inputs_T
  _attr_Tindices, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ])
  var = _ops.convert_to_tensor(var, _dtypes.resource)
  accum = _ops.convert_to_tensor(accum, _dtypes.resource)
  linear = _ops.convert_to_tensor(linear, _dtypes.resource)
  _inputs_flat = [var, accum, linear, grad, indices, lr, l1, l2, lr_power]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "use_locking",
  use_locking, "multiply_linear_by_lr", multiply_linear_by_lr)
  _result = _execute.execute(b"ResourceSparseApplyFtrl", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_ResourceSparseApplyFtrlV2_T = TypeVar("TV_ResourceSparseApplyFtrlV2_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_ResourceSparseApplyFtrlV2_Tindices = TypeVar("TV_ResourceSparseApplyFtrlV2_Tindices", _atypes.Int32, _atypes.Int64)

def resource_sparse_apply_ftrl_v2(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], linear: Annotated[Any, _atypes.Resource], grad: Annotated[Any, TV_ResourceSparseApplyFtrlV2_T], indices: Annotated[Any, TV_ResourceSparseApplyFtrlV2_Tindices], lr: Annotated[Any, TV_ResourceSparseApplyFtrlV2_T], l1: Annotated[Any, TV_ResourceSparseApplyFtrlV2_T], l2: Annotated[Any, TV_ResourceSparseApplyFtrlV2_T], l2_shrinkage: Annotated[Any, TV_ResourceSparseApplyFtrlV2_T], lr_power: Annotated[Any, TV_ResourceSparseApplyFtrlV2_T], use_locking:bool=False, multiply_linear_by_lr:bool=False, name=None):
  r"""Update relevant entries in '*var' according to the Ftrl-proximal scheme.

  That is for rows we have grad for, we update var, accum and linear as follows:
  grad_with_shrinkage = grad + 2 * l2_shrinkage * var
  accum_new = accum + grad_with_shrinkage * grad_with_shrinkage
  linear += grad_with_shrinkage +
      (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
  quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
  var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
  accum = accum_new

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    accum: A `Tensor` of type `resource`. Should be from a Variable().
    linear: A `Tensor` of type `resource`. Should be from a Variable().
    grad: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    lr: A `Tensor`. Must have the same type as `grad`.
      Scaling factor. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `grad`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `grad`.
      L2 shrinkage regularization. Must be a scalar.
    l2_shrinkage: A `Tensor`. Must have the same type as `grad`.
    lr_power: A `Tensor`. Must have the same type as `grad`.
      Scaling factor. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    multiply_linear_by_lr: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceSparseApplyFtrlV2", name, var, accum, linear, grad,
        indices, lr, l1, l2, l2_shrinkage, lr_power, "use_locking",
        use_locking, "multiply_linear_by_lr", multiply_linear_by_lr)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_sparse_apply_ftrl_v2_eager_fallback(
          var, accum, linear, grad, indices, lr, l1, l2, l2_shrinkage,
          lr_power, use_locking=use_locking,
          multiply_linear_by_lr=multiply_linear_by_lr, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if multiply_linear_by_lr is None:
    multiply_linear_by_lr = False
  multiply_linear_by_lr = _execute.make_bool(multiply_linear_by_lr, "multiply_linear_by_lr")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceSparseApplyFtrlV2", var=var, accum=accum, linear=linear,
                                     grad=grad, indices=indices, lr=lr, l1=l1,
                                     l2=l2, l2_shrinkage=l2_shrinkage,
                                     lr_power=lr_power,
                                     use_locking=use_locking,
                                     multiply_linear_by_lr=multiply_linear_by_lr,
                                     name=name)
  return _op
ResourceSparseApplyFtrlV2 = tf_export("raw_ops.ResourceSparseApplyFtrlV2")(_ops.to_raw_op(resource_sparse_apply_ftrl_v2))


def resource_sparse_apply_ftrl_v2_eager_fallback(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], linear: Annotated[Any, _atypes.Resource], grad: Annotated[Any, TV_ResourceSparseApplyFtrlV2_T], indices: Annotated[Any, TV_ResourceSparseApplyFtrlV2_Tindices], lr: Annotated[Any, TV_ResourceSparseApplyFtrlV2_T], l1: Annotated[Any, TV_ResourceSparseApplyFtrlV2_T], l2: Annotated[Any, TV_ResourceSparseApplyFtrlV2_T], l2_shrinkage: Annotated[Any, TV_ResourceSparseApplyFtrlV2_T], lr_power: Annotated[Any, TV_ResourceSparseApplyFtrlV2_T], use_locking: bool, multiply_linear_by_lr: bool, name, ctx):
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if multiply_linear_by_lr is None:
    multiply_linear_by_lr = False
  multiply_linear_by_lr = _execute.make_bool(multiply_linear_by_lr, "multiply_linear_by_lr")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([grad, lr, l1, l2, l2_shrinkage, lr_power], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (grad, lr, l1, l2, l2_shrinkage, lr_power) = _inputs_T
  _attr_Tindices, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ])
  var = _ops.convert_to_tensor(var, _dtypes.resource)
  accum = _ops.convert_to_tensor(accum, _dtypes.resource)
  linear = _ops.convert_to_tensor(linear, _dtypes.resource)
  _inputs_flat = [var, accum, linear, grad, indices, lr, l1, l2, l2_shrinkage, lr_power]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "use_locking",
  use_locking, "multiply_linear_by_lr", multiply_linear_by_lr)
  _result = _execute.execute(b"ResourceSparseApplyFtrlV2", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_ResourceSparseApplyKerasMomentum_T = TypeVar("TV_ResourceSparseApplyKerasMomentum_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_ResourceSparseApplyKerasMomentum_Tindices = TypeVar("TV_ResourceSparseApplyKerasMomentum_Tindices", _atypes.Int32, _atypes.Int64)

def resource_sparse_apply_keras_momentum(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceSparseApplyKerasMomentum_T], grad: Annotated[Any, TV_ResourceSparseApplyKerasMomentum_T], indices: Annotated[Any, TV_ResourceSparseApplyKerasMomentum_Tindices], momentum: Annotated[Any, TV_ResourceSparseApplyKerasMomentum_T], use_locking:bool=False, use_nesterov:bool=False, name=None):
  r"""Update relevant entries in '*var' and '*accum' according to the momentum scheme.

  Set use_nesterov = True if you want to use Nesterov momentum.

  That is for rows we have grad for, we update var and accum as follows:

  accum = accum * momentum - lr * grad
  var += accum

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    accum: A `Tensor` of type `resource`. Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Learning rate. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    momentum: A `Tensor`. Must have the same type as `lr`.
      Momentum. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    use_nesterov: An optional `bool`. Defaults to `False`.
      If `True`, the tensor passed to compute grad will be
      var + momentum * accum, so in the end, the var you get is actually
      var + momentum * accum.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceSparseApplyKerasMomentum", name, var, accum, lr, grad,
        indices, momentum, "use_locking", use_locking, "use_nesterov",
        use_nesterov)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_sparse_apply_keras_momentum_eager_fallback(
          var, accum, lr, grad, indices, momentum, use_locking=use_locking,
          use_nesterov=use_nesterov, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if use_nesterov is None:
    use_nesterov = False
  use_nesterov = _execute.make_bool(use_nesterov, "use_nesterov")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceSparseApplyKerasMomentum", var=var, accum=accum, lr=lr,
                                            grad=grad, indices=indices,
                                            momentum=momentum,
                                            use_locking=use_locking,
                                            use_nesterov=use_nesterov,
                                            name=name)
  return _op
ResourceSparseApplyKerasMomentum = tf_export("raw_ops.ResourceSparseApplyKerasMomentum")(_ops.to_raw_op(resource_sparse_apply_keras_momentum))


def resource_sparse_apply_keras_momentum_eager_fallback(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceSparseApplyKerasMomentum_T], grad: Annotated[Any, TV_ResourceSparseApplyKerasMomentum_T], indices: Annotated[Any, TV_ResourceSparseApplyKerasMomentum_Tindices], momentum: Annotated[Any, TV_ResourceSparseApplyKerasMomentum_T], use_locking: bool, use_nesterov: bool, name, ctx):
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if use_nesterov is None:
    use_nesterov = False
  use_nesterov = _execute.make_bool(use_nesterov, "use_nesterov")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([lr, grad, momentum], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (lr, grad, momentum) = _inputs_T
  _attr_Tindices, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ])
  var = _ops.convert_to_tensor(var, _dtypes.resource)
  accum = _ops.convert_to_tensor(accum, _dtypes.resource)
  _inputs_flat = [var, accum, lr, grad, indices, momentum]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "use_locking",
  use_locking, "use_nesterov", use_nesterov)
  _result = _execute.execute(b"ResourceSparseApplyKerasMomentum", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_ResourceSparseApplyMomentum_T = TypeVar("TV_ResourceSparseApplyMomentum_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_ResourceSparseApplyMomentum_Tindices = TypeVar("TV_ResourceSparseApplyMomentum_Tindices", _atypes.Int32, _atypes.Int64)

def resource_sparse_apply_momentum(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceSparseApplyMomentum_T], grad: Annotated[Any, TV_ResourceSparseApplyMomentum_T], indices: Annotated[Any, TV_ResourceSparseApplyMomentum_Tindices], momentum: Annotated[Any, TV_ResourceSparseApplyMomentum_T], use_locking:bool=False, use_nesterov:bool=False, name=None):
  r"""Update relevant entries in '*var' and '*accum' according to the momentum scheme.

  Set use_nesterov = True if you want to use Nesterov momentum.

  That is for rows we have grad for, we update var and accum as follows:

  accum = accum * momentum + grad
  var -= lr * accum

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    accum: A `Tensor` of type `resource`. Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Learning rate. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    momentum: A `Tensor`. Must have the same type as `lr`.
      Momentum. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    use_nesterov: An optional `bool`. Defaults to `False`.
      If `True`, the tensor passed to compute grad will be
      var - lr * momentum * accum, so in the end, the var you get is actually
      var - lr * momentum * accum.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceSparseApplyMomentum", name, var, accum, lr, grad,
        indices, momentum, "use_locking", use_locking, "use_nesterov",
        use_nesterov)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_sparse_apply_momentum_eager_fallback(
          var, accum, lr, grad, indices, momentum, use_locking=use_locking,
          use_nesterov=use_nesterov, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if use_nesterov is None:
    use_nesterov = False
  use_nesterov = _execute.make_bool(use_nesterov, "use_nesterov")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceSparseApplyMomentum", var=var, accum=accum, lr=lr, grad=grad,
                                       indices=indices, momentum=momentum,
                                       use_locking=use_locking,
                                       use_nesterov=use_nesterov, name=name)
  return _op
ResourceSparseApplyMomentum = tf_export("raw_ops.ResourceSparseApplyMomentum")(_ops.to_raw_op(resource_sparse_apply_momentum))


def resource_sparse_apply_momentum_eager_fallback(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceSparseApplyMomentum_T], grad: Annotated[Any, TV_ResourceSparseApplyMomentum_T], indices: Annotated[Any, TV_ResourceSparseApplyMomentum_Tindices], momentum: Annotated[Any, TV_ResourceSparseApplyMomentum_T], use_locking: bool, use_nesterov: bool, name, ctx):
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if use_nesterov is None:
    use_nesterov = False
  use_nesterov = _execute.make_bool(use_nesterov, "use_nesterov")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([lr, grad, momentum], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (lr, grad, momentum) = _inputs_T
  _attr_Tindices, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ])
  var = _ops.convert_to_tensor(var, _dtypes.resource)
  accum = _ops.convert_to_tensor(accum, _dtypes.resource)
  _inputs_flat = [var, accum, lr, grad, indices, momentum]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "use_locking",
  use_locking, "use_nesterov", use_nesterov)
  _result = _execute.execute(b"ResourceSparseApplyMomentum", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_ResourceSparseApplyProximalAdagrad_T = TypeVar("TV_ResourceSparseApplyProximalAdagrad_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_ResourceSparseApplyProximalAdagrad_Tindices = TypeVar("TV_ResourceSparseApplyProximalAdagrad_Tindices", _atypes.Int32, _atypes.Int64)

def resource_sparse_apply_proximal_adagrad(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceSparseApplyProximalAdagrad_T], l1: Annotated[Any, TV_ResourceSparseApplyProximalAdagrad_T], l2: Annotated[Any, TV_ResourceSparseApplyProximalAdagrad_T], grad: Annotated[Any, TV_ResourceSparseApplyProximalAdagrad_T], indices: Annotated[Any, TV_ResourceSparseApplyProximalAdagrad_Tindices], use_locking:bool=False, name=None):
  r"""Sparse update entries in '*var' and '*accum' according to FOBOS algorithm.

  That is for rows we have grad for, we update var and accum as follows:
  accum += grad * grad
  prox_v = var
  prox_v -= lr * grad * (1 / sqrt(accum))
  var = sign(prox_v)/(1+lr*l2) * max{|prox_v|-lr*l1,0}

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    accum: A `Tensor` of type `resource`. Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Learning rate. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `lr`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `lr`.
      L2 regularization. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, updating of the var and accum tensors will be protected by
      a lock; otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceSparseApplyProximalAdagrad", name, var, accum, lr, l1,
        l2, grad, indices, "use_locking", use_locking)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_sparse_apply_proximal_adagrad_eager_fallback(
          var, accum, lr, l1, l2, grad, indices, use_locking=use_locking,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceSparseApplyProximalAdagrad", var=var, accum=accum, lr=lr,
                                              l1=l1, l2=l2, grad=grad,
                                              indices=indices,
                                              use_locking=use_locking,
                                              name=name)
  return _op
ResourceSparseApplyProximalAdagrad = tf_export("raw_ops.ResourceSparseApplyProximalAdagrad")(_ops.to_raw_op(resource_sparse_apply_proximal_adagrad))


def resource_sparse_apply_proximal_adagrad_eager_fallback(var: Annotated[Any, _atypes.Resource], accum: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceSparseApplyProximalAdagrad_T], l1: Annotated[Any, TV_ResourceSparseApplyProximalAdagrad_T], l2: Annotated[Any, TV_ResourceSparseApplyProximalAdagrad_T], grad: Annotated[Any, TV_ResourceSparseApplyProximalAdagrad_T], indices: Annotated[Any, TV_ResourceSparseApplyProximalAdagrad_Tindices], use_locking: bool, name, ctx):
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([lr, l1, l2, grad], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (lr, l1, l2, grad) = _inputs_T
  _attr_Tindices, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ])
  var = _ops.convert_to_tensor(var, _dtypes.resource)
  accum = _ops.convert_to_tensor(accum, _dtypes.resource)
  _inputs_flat = [var, accum, lr, l1, l2, grad, indices]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "use_locking",
  use_locking)
  _result = _execute.execute(b"ResourceSparseApplyProximalAdagrad", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_ResourceSparseApplyProximalGradientDescent_T = TypeVar("TV_ResourceSparseApplyProximalGradientDescent_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_ResourceSparseApplyProximalGradientDescent_Tindices = TypeVar("TV_ResourceSparseApplyProximalGradientDescent_Tindices", _atypes.Int32, _atypes.Int64)

def resource_sparse_apply_proximal_gradient_descent(var: Annotated[Any, _atypes.Resource], alpha: Annotated[Any, TV_ResourceSparseApplyProximalGradientDescent_T], l1: Annotated[Any, TV_ResourceSparseApplyProximalGradientDescent_T], l2: Annotated[Any, TV_ResourceSparseApplyProximalGradientDescent_T], grad: Annotated[Any, TV_ResourceSparseApplyProximalGradientDescent_T], indices: Annotated[Any, TV_ResourceSparseApplyProximalGradientDescent_Tindices], use_locking:bool=False, name=None):
  r"""Sparse update '*var' as FOBOS algorithm with fixed learning rate.

  That is for rows we have grad for, we update var as follows:
  prox_v = var - alpha * grad
  var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    alpha: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Scaling factor. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `alpha`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `alpha`.
      L2 regularization. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `alpha`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, the subtraction will be protected by a lock;
      otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceSparseApplyProximalGradientDescent", name, var, alpha,
        l1, l2, grad, indices, "use_locking", use_locking)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_sparse_apply_proximal_gradient_descent_eager_fallback(
          var, alpha, l1, l2, grad, indices, use_locking=use_locking,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceSparseApplyProximalGradientDescent", var=var, alpha=alpha,
                                                      l1=l1, l2=l2, grad=grad,
                                                      indices=indices,
                                                      use_locking=use_locking,
                                                      name=name)
  return _op
ResourceSparseApplyProximalGradientDescent = tf_export("raw_ops.ResourceSparseApplyProximalGradientDescent")(_ops.to_raw_op(resource_sparse_apply_proximal_gradient_descent))


def resource_sparse_apply_proximal_gradient_descent_eager_fallback(var: Annotated[Any, _atypes.Resource], alpha: Annotated[Any, TV_ResourceSparseApplyProximalGradientDescent_T], l1: Annotated[Any, TV_ResourceSparseApplyProximalGradientDescent_T], l2: Annotated[Any, TV_ResourceSparseApplyProximalGradientDescent_T], grad: Annotated[Any, TV_ResourceSparseApplyProximalGradientDescent_T], indices: Annotated[Any, TV_ResourceSparseApplyProximalGradientDescent_Tindices], use_locking: bool, name, ctx):
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([alpha, l1, l2, grad], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (alpha, l1, l2, grad) = _inputs_T
  _attr_Tindices, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ])
  var = _ops.convert_to_tensor(var, _dtypes.resource)
  _inputs_flat = [var, alpha, l1, l2, grad, indices]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "use_locking",
  use_locking)
  _result = _execute.execute(b"ResourceSparseApplyProximalGradientDescent", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_ResourceSparseApplyRMSProp_T = TypeVar("TV_ResourceSparseApplyRMSProp_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_ResourceSparseApplyRMSProp_Tindices = TypeVar("TV_ResourceSparseApplyRMSProp_Tindices", _atypes.Int32, _atypes.Int64)

def resource_sparse_apply_rms_prop(var: Annotated[Any, _atypes.Resource], ms: Annotated[Any, _atypes.Resource], mom: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceSparseApplyRMSProp_T], rho: Annotated[Any, TV_ResourceSparseApplyRMSProp_T], momentum: Annotated[Any, TV_ResourceSparseApplyRMSProp_T], epsilon: Annotated[Any, TV_ResourceSparseApplyRMSProp_T], grad: Annotated[Any, TV_ResourceSparseApplyRMSProp_T], indices: Annotated[Any, TV_ResourceSparseApplyRMSProp_Tindices], use_locking:bool=False, name=None):
  r"""Update '*var' according to the RMSProp algorithm.

  Note that in dense implementation of this algorithm, ms and mom will
  update even if the grad is zero, but in this sparse implementation, ms
  and mom will not update in iterations during which the grad is zero.

  mean_square = decay * mean_square + (1-decay) * gradient ** 2
  Delta = learning_rate * gradient / sqrt(mean_square + epsilon)

  ms <- rho * ms_{t-1} + (1-rho) * grad * grad
  mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
  var <- var - mom

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    ms: A `Tensor` of type `resource`. Should be from a Variable().
    mom: A `Tensor` of type `resource`. Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Scaling factor. Must be a scalar.
    rho: A `Tensor`. Must have the same type as `lr`.
      Decay rate. Must be a scalar.
    momentum: A `Tensor`. Must have the same type as `lr`.
    epsilon: A `Tensor`. Must have the same type as `lr`.
      Ridge term. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var, ms and mom.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var, ms, and mom tensors is protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceSparseApplyRMSProp", name, var, ms, mom, lr, rho,
        momentum, epsilon, grad, indices, "use_locking", use_locking)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resource_sparse_apply_rms_prop_eager_fallback(
          var, ms, mom, lr, rho, momentum, epsilon, grad, indices,
          use_locking=use_locking, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceSparseApplyRMSProp", var=var, ms=ms, mom=mom, lr=lr, rho=rho,
                                      momentum=momentum, epsilon=epsilon,
                                      grad=grad, indices=indices,
                                      use_locking=use_locking, name=name)
  return _op
ResourceSparseApplyRMSProp = tf_export("raw_ops.ResourceSparseApplyRMSProp")(_ops.to_raw_op(resource_sparse_apply_rms_prop))


def resource_sparse_apply_rms_prop_eager_fallback(var: Annotated[Any, _atypes.Resource], ms: Annotated[Any, _atypes.Resource], mom: Annotated[Any, _atypes.Resource], lr: Annotated[Any, TV_ResourceSparseApplyRMSProp_T], rho: Annotated[Any, TV_ResourceSparseApplyRMSProp_T], momentum: Annotated[Any, TV_ResourceSparseApplyRMSProp_T], epsilon: Annotated[Any, TV_ResourceSparseApplyRMSProp_T], grad: Annotated[Any, TV_ResourceSparseApplyRMSProp_T], indices: Annotated[Any, TV_ResourceSparseApplyRMSProp_Tindices], use_locking: bool, name, ctx):
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([lr, rho, momentum, epsilon, grad], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (lr, rho, momentum, epsilon, grad) = _inputs_T
  _attr_Tindices, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ])
  var = _ops.convert_to_tensor(var, _dtypes.resource)
  ms = _ops.convert_to_tensor(ms, _dtypes.resource)
  mom = _ops.convert_to_tensor(mom, _dtypes.resource)
  _inputs_flat = [var, ms, mom, lr, rho, momentum, epsilon, grad, indices]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "use_locking",
  use_locking)
  _result = _execute.execute(b"ResourceSparseApplyRMSProp", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_SparseApplyAdadelta_T = TypeVar("TV_SparseApplyAdadelta_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_SparseApplyAdadelta_Tindices = TypeVar("TV_SparseApplyAdadelta_Tindices", _atypes.Int32, _atypes.Int64)

def sparse_apply_adadelta(var: Annotated[Any, TV_SparseApplyAdadelta_T], accum: Annotated[Any, TV_SparseApplyAdadelta_T], accum_update: Annotated[Any, TV_SparseApplyAdadelta_T], lr: Annotated[Any, TV_SparseApplyAdadelta_T], rho: Annotated[Any, TV_SparseApplyAdadelta_T], epsilon: Annotated[Any, TV_SparseApplyAdadelta_T], grad: Annotated[Any, TV_SparseApplyAdadelta_T], indices: Annotated[Any, TV_SparseApplyAdadelta_Tindices], use_locking:bool=False, name=None) -> Annotated[Any, TV_SparseApplyAdadelta_T]:
  r"""var: Should be from a Variable().

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
    accum: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    accum_update: A mutable `Tensor`. Must have the same type as `var`.
      : Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Learning rate. Must be a scalar.
    rho: A `Tensor`. Must have the same type as `var`.
      Decay factor. Must be a scalar.
    epsilon: A `Tensor`. Must have the same type as `var`.
      Constant factor. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, updating of the var and accum tensors will be protected by
      a lock; otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("sparse_apply_adadelta op does not support eager execution. Arg 'out' is a ref.")
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseApplyAdadelta", var=var, accum=accum,
                               accum_update=accum_update, lr=lr, rho=rho,
                               epsilon=epsilon, grad=grad, indices=indices,
                               use_locking=use_locking, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "use_locking",
              _op._get_attr_bool("use_locking"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseApplyAdadelta", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseApplyAdadelta = tf_export("raw_ops.SparseApplyAdadelta")(_ops.to_raw_op(sparse_apply_adadelta))


def sparse_apply_adadelta_eager_fallback(var: Annotated[Any, TV_SparseApplyAdadelta_T], accum: Annotated[Any, TV_SparseApplyAdadelta_T], accum_update: Annotated[Any, TV_SparseApplyAdadelta_T], lr: Annotated[Any, TV_SparseApplyAdadelta_T], rho: Annotated[Any, TV_SparseApplyAdadelta_T], epsilon: Annotated[Any, TV_SparseApplyAdadelta_T], grad: Annotated[Any, TV_SparseApplyAdadelta_T], indices: Annotated[Any, TV_SparseApplyAdadelta_Tindices], use_locking: bool, name, ctx) -> Annotated[Any, TV_SparseApplyAdadelta_T]:
  raise RuntimeError("sparse_apply_adadelta op does not support eager execution. Arg 'out' is a ref.")

TV_SparseApplyAdagrad_T = TypeVar("TV_SparseApplyAdagrad_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_SparseApplyAdagrad_Tindices = TypeVar("TV_SparseApplyAdagrad_Tindices", _atypes.Int32, _atypes.Int64)

def sparse_apply_adagrad(var: Annotated[Any, TV_SparseApplyAdagrad_T], accum: Annotated[Any, TV_SparseApplyAdagrad_T], lr: Annotated[Any, TV_SparseApplyAdagrad_T], grad: Annotated[Any, TV_SparseApplyAdagrad_T], indices: Annotated[Any, TV_SparseApplyAdagrad_Tindices], use_locking:bool=False, update_slots:bool=True, name=None) -> Annotated[Any, TV_SparseApplyAdagrad_T]:
  r"""Update relevant entries in '*var' and '*accum' according to the adagrad scheme.

  That is for rows we have grad for, we update var and accum as follows:
  $$accum += grad * grad$$
  $$var -= lr * grad * (1 / sqrt(accum))$$

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Should be from a Variable().
    accum: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Learning rate. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    update_slots: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("sparse_apply_adagrad op does not support eager execution. Arg 'out' is a ref.")
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if update_slots is None:
    update_slots = True
  update_slots = _execute.make_bool(update_slots, "update_slots")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseApplyAdagrad", var=var, accum=accum, lr=lr, grad=grad,
                              indices=indices, use_locking=use_locking,
                              update_slots=update_slots, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "use_locking",
              _op._get_attr_bool("use_locking"), "update_slots",
              _op._get_attr_bool("update_slots"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseApplyAdagrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseApplyAdagrad = tf_export("raw_ops.SparseApplyAdagrad")(_ops.to_raw_op(sparse_apply_adagrad))


def sparse_apply_adagrad_eager_fallback(var: Annotated[Any, TV_SparseApplyAdagrad_T], accum: Annotated[Any, TV_SparseApplyAdagrad_T], lr: Annotated[Any, TV_SparseApplyAdagrad_T], grad: Annotated[Any, TV_SparseApplyAdagrad_T], indices: Annotated[Any, TV_SparseApplyAdagrad_Tindices], use_locking: bool, update_slots: bool, name, ctx) -> Annotated[Any, TV_SparseApplyAdagrad_T]:
  raise RuntimeError("sparse_apply_adagrad op does not support eager execution. Arg 'out' is a ref.")

TV_SparseApplyAdagradDA_T = TypeVar("TV_SparseApplyAdagradDA_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_SparseApplyAdagradDA_Tindices = TypeVar("TV_SparseApplyAdagradDA_Tindices", _atypes.Int32, _atypes.Int64)

def sparse_apply_adagrad_da(var: Annotated[Any, TV_SparseApplyAdagradDA_T], gradient_accumulator: Annotated[Any, TV_SparseApplyAdagradDA_T], gradient_squared_accumulator: Annotated[Any, TV_SparseApplyAdagradDA_T], grad: Annotated[Any, TV_SparseApplyAdagradDA_T], indices: Annotated[Any, TV_SparseApplyAdagradDA_Tindices], lr: Annotated[Any, TV_SparseApplyAdagradDA_T], l1: Annotated[Any, TV_SparseApplyAdagradDA_T], l2: Annotated[Any, TV_SparseApplyAdagradDA_T], global_step: Annotated[Any, _atypes.Int64], use_locking:bool=False, name=None) -> Annotated[Any, TV_SparseApplyAdagradDA_T]:
  r"""Update entries in '*var' and '*accum' according to the proximal adagrad scheme.

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Should be from a Variable().
    gradient_accumulator: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    gradient_squared_accumulator: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    lr: A `Tensor`. Must have the same type as `var`.
      Learning rate. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `var`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `var`.
      L2 regularization. Must be a scalar.
    global_step: A `Tensor` of type `int64`.
      Training step number. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, updating of the var and accum tensors will be protected by
      a lock; otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("sparse_apply_adagrad_da op does not support eager execution. Arg 'out' is a ref.")
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseApplyAdagradDA", var=var,
                                gradient_accumulator=gradient_accumulator,
                                gradient_squared_accumulator=gradient_squared_accumulator,
                                grad=grad, indices=indices, lr=lr, l1=l1,
                                l2=l2, global_step=global_step,
                                use_locking=use_locking, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "use_locking",
              _op._get_attr_bool("use_locking"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseApplyAdagradDA", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseApplyAdagradDA = tf_export("raw_ops.SparseApplyAdagradDA")(_ops.to_raw_op(sparse_apply_adagrad_da))


def sparse_apply_adagrad_da_eager_fallback(var: Annotated[Any, TV_SparseApplyAdagradDA_T], gradient_accumulator: Annotated[Any, TV_SparseApplyAdagradDA_T], gradient_squared_accumulator: Annotated[Any, TV_SparseApplyAdagradDA_T], grad: Annotated[Any, TV_SparseApplyAdagradDA_T], indices: Annotated[Any, TV_SparseApplyAdagradDA_Tindices], lr: Annotated[Any, TV_SparseApplyAdagradDA_T], l1: Annotated[Any, TV_SparseApplyAdagradDA_T], l2: Annotated[Any, TV_SparseApplyAdagradDA_T], global_step: Annotated[Any, _atypes.Int64], use_locking: bool, name, ctx) -> Annotated[Any, TV_SparseApplyAdagradDA_T]:
  raise RuntimeError("sparse_apply_adagrad_da op does not support eager execution. Arg 'out' is a ref.")

TV_SparseApplyAdagradV2_T = TypeVar("TV_SparseApplyAdagradV2_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_SparseApplyAdagradV2_Tindices = TypeVar("TV_SparseApplyAdagradV2_Tindices", _atypes.Int32, _atypes.Int64)

def sparse_apply_adagrad_v2(var: Annotated[Any, TV_SparseApplyAdagradV2_T], accum: Annotated[Any, TV_SparseApplyAdagradV2_T], lr: Annotated[Any, TV_SparseApplyAdagradV2_T], epsilon: Annotated[Any, TV_SparseApplyAdagradV2_T], grad: Annotated[Any, TV_SparseApplyAdagradV2_T], indices: Annotated[Any, TV_SparseApplyAdagradV2_Tindices], use_locking:bool=False, update_slots:bool=True, name=None) -> Annotated[Any, TV_SparseApplyAdagradV2_T]:
  r"""Update relevant entries in '*var' and '*accum' according to the adagrad scheme.

  That is for rows we have grad for, we update var and accum as follows:
  $$accum += grad * grad$$
  $$var -= lr * grad * (1 / sqrt(accum))$$

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Should be from a Variable().
    accum: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Learning rate. Must be a scalar.
    epsilon: A `Tensor`. Must have the same type as `var`.
      Constant factor. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    update_slots: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("sparse_apply_adagrad_v2 op does not support eager execution. Arg 'out' is a ref.")
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if update_slots is None:
    update_slots = True
  update_slots = _execute.make_bool(update_slots, "update_slots")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseApplyAdagradV2", var=var, accum=accum, lr=lr, epsilon=epsilon,
                                grad=grad, indices=indices,
                                use_locking=use_locking,
                                update_slots=update_slots, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "use_locking",
              _op._get_attr_bool("use_locking"), "update_slots",
              _op._get_attr_bool("update_slots"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseApplyAdagradV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseApplyAdagradV2 = tf_export("raw_ops.SparseApplyAdagradV2")(_ops.to_raw_op(sparse_apply_adagrad_v2))


def sparse_apply_adagrad_v2_eager_fallback(var: Annotated[Any, TV_SparseApplyAdagradV2_T], accum: Annotated[Any, TV_SparseApplyAdagradV2_T], lr: Annotated[Any, TV_SparseApplyAdagradV2_T], epsilon: Annotated[Any, TV_SparseApplyAdagradV2_T], grad: Annotated[Any, TV_SparseApplyAdagradV2_T], indices: Annotated[Any, TV_SparseApplyAdagradV2_Tindices], use_locking: bool, update_slots: bool, name, ctx) -> Annotated[Any, TV_SparseApplyAdagradV2_T]:
  raise RuntimeError("sparse_apply_adagrad_v2 op does not support eager execution. Arg 'out' is a ref.")

TV_SparseApplyCenteredRMSProp_T = TypeVar("TV_SparseApplyCenteredRMSProp_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_SparseApplyCenteredRMSProp_Tindices = TypeVar("TV_SparseApplyCenteredRMSProp_Tindices", _atypes.Int32, _atypes.Int64)

def sparse_apply_centered_rms_prop(var: Annotated[Any, TV_SparseApplyCenteredRMSProp_T], mg: Annotated[Any, TV_SparseApplyCenteredRMSProp_T], ms: Annotated[Any, TV_SparseApplyCenteredRMSProp_T], mom: Annotated[Any, TV_SparseApplyCenteredRMSProp_T], lr: Annotated[Any, TV_SparseApplyCenteredRMSProp_T], rho: Annotated[Any, TV_SparseApplyCenteredRMSProp_T], momentum: Annotated[Any, TV_SparseApplyCenteredRMSProp_T], epsilon: Annotated[Any, TV_SparseApplyCenteredRMSProp_T], grad: Annotated[Any, TV_SparseApplyCenteredRMSProp_T], indices: Annotated[Any, TV_SparseApplyCenteredRMSProp_Tindices], use_locking:bool=False, name=None) -> Annotated[Any, TV_SparseApplyCenteredRMSProp_T]:
  r"""Update '*var' according to the centered RMSProp algorithm.

  The centered RMSProp algorithm uses an estimate of the centered second moment
  (i.e., the variance) for normalization, as opposed to regular RMSProp, which
  uses the (uncentered) second moment. This often helps with training, but is
  slightly more expensive in terms of computation and memory.

  Note that in dense implementation of this algorithm, mg, ms, and mom will
  update even if the grad is zero, but in this sparse implementation, mg, ms,
  and mom will not update in iterations during which the grad is zero.

  mean_square = decay * mean_square + (1-decay) * gradient ** 2
  mean_grad = decay * mean_grad + (1-decay) * gradient
  Delta = learning_rate * gradient / sqrt(mean_square + epsilon - mean_grad ** 2)

  $$ms <- rho * ms_{t-1} + (1-rho) * grad * grad$$
  $$mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)$$
  $$var <- var - mom$$

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Should be from a Variable().
    mg: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    ms: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    mom: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    rho: A `Tensor`. Must have the same type as `var`.
      Decay rate. Must be a scalar.
    momentum: A `Tensor`. Must have the same type as `var`.
    epsilon: A `Tensor`. Must have the same type as `var`.
      Ridge term. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var, ms and mom.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var, mg, ms, and mom tensors is
      protected by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("sparse_apply_centered_rms_prop op does not support eager execution. Arg 'out' is a ref.")
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseApplyCenteredRMSProp", var=var, mg=mg, ms=ms, mom=mom, lr=lr,
                                      rho=rho, momentum=momentum,
                                      epsilon=epsilon, grad=grad,
                                      indices=indices,
                                      use_locking=use_locking, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "use_locking",
              _op._get_attr_bool("use_locking"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseApplyCenteredRMSProp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseApplyCenteredRMSProp = tf_export("raw_ops.SparseApplyCenteredRMSProp")(_ops.to_raw_op(sparse_apply_centered_rms_prop))


def sparse_apply_centered_rms_prop_eager_fallback(var: Annotated[Any, TV_SparseApplyCenteredRMSProp_T], mg: Annotated[Any, TV_SparseApplyCenteredRMSProp_T], ms: Annotated[Any, TV_SparseApplyCenteredRMSProp_T], mom: Annotated[Any, TV_SparseApplyCenteredRMSProp_T], lr: Annotated[Any, TV_SparseApplyCenteredRMSProp_T], rho: Annotated[Any, TV_SparseApplyCenteredRMSProp_T], momentum: Annotated[Any, TV_SparseApplyCenteredRMSProp_T], epsilon: Annotated[Any, TV_SparseApplyCenteredRMSProp_T], grad: Annotated[Any, TV_SparseApplyCenteredRMSProp_T], indices: Annotated[Any, TV_SparseApplyCenteredRMSProp_Tindices], use_locking: bool, name, ctx) -> Annotated[Any, TV_SparseApplyCenteredRMSProp_T]:
  raise RuntimeError("sparse_apply_centered_rms_prop op does not support eager execution. Arg 'out' is a ref.")

TV_SparseApplyFtrl_T = TypeVar("TV_SparseApplyFtrl_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_SparseApplyFtrl_Tindices = TypeVar("TV_SparseApplyFtrl_Tindices", _atypes.Int32, _atypes.Int64)

def sparse_apply_ftrl(var: Annotated[Any, TV_SparseApplyFtrl_T], accum: Annotated[Any, TV_SparseApplyFtrl_T], linear: Annotated[Any, TV_SparseApplyFtrl_T], grad: Annotated[Any, TV_SparseApplyFtrl_T], indices: Annotated[Any, TV_SparseApplyFtrl_Tindices], lr: Annotated[Any, TV_SparseApplyFtrl_T], l1: Annotated[Any, TV_SparseApplyFtrl_T], l2: Annotated[Any, TV_SparseApplyFtrl_T], lr_power: Annotated[Any, TV_SparseApplyFtrl_T], use_locking:bool=False, multiply_linear_by_lr:bool=False, name=None) -> Annotated[Any, TV_SparseApplyFtrl_T]:
  r"""Update relevant entries in '*var' according to the Ftrl-proximal scheme.

  That is for rows we have grad for, we update var, accum and linear as follows:
  $$accum_new = accum + grad * grad$$
  $$linear += grad + (accum_{new}^{-lr_{power}} - accum^{-lr_{power}} / lr * var$$
  $$quadratic = 1.0 / (accum_{new}^{lr_{power}} * lr) + 2 * l2$$
  $$var = (sign(linear) * l1 - linear) / quadratic\ if\ |linear| > l1\ else\ 0.0$$
  $$accum = accum_{new}$$

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Should be from a Variable().
    accum: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    linear: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `var`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `var`.
      L2 regularization. Must be a scalar.
    lr_power: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    multiply_linear_by_lr: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("sparse_apply_ftrl op does not support eager execution. Arg 'out' is a ref.")
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if multiply_linear_by_lr is None:
    multiply_linear_by_lr = False
  multiply_linear_by_lr = _execute.make_bool(multiply_linear_by_lr, "multiply_linear_by_lr")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseApplyFtrl", var=var, accum=accum, linear=linear, grad=grad,
                           indices=indices, lr=lr, l1=l1, l2=l2,
                           lr_power=lr_power, use_locking=use_locking,
                           multiply_linear_by_lr=multiply_linear_by_lr,
                           name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "use_locking",
              _op._get_attr_bool("use_locking"), "multiply_linear_by_lr",
              _op._get_attr_bool("multiply_linear_by_lr"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseApplyFtrl", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseApplyFtrl = tf_export("raw_ops.SparseApplyFtrl")(_ops.to_raw_op(sparse_apply_ftrl))


def sparse_apply_ftrl_eager_fallback(var: Annotated[Any, TV_SparseApplyFtrl_T], accum: Annotated[Any, TV_SparseApplyFtrl_T], linear: Annotated[Any, TV_SparseApplyFtrl_T], grad: Annotated[Any, TV_SparseApplyFtrl_T], indices: Annotated[Any, TV_SparseApplyFtrl_Tindices], lr: Annotated[Any, TV_SparseApplyFtrl_T], l1: Annotated[Any, TV_SparseApplyFtrl_T], l2: Annotated[Any, TV_SparseApplyFtrl_T], lr_power: Annotated[Any, TV_SparseApplyFtrl_T], use_locking: bool, multiply_linear_by_lr: bool, name, ctx) -> Annotated[Any, TV_SparseApplyFtrl_T]:
  raise RuntimeError("sparse_apply_ftrl op does not support eager execution. Arg 'out' is a ref.")

TV_SparseApplyFtrlV2_T = TypeVar("TV_SparseApplyFtrlV2_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_SparseApplyFtrlV2_Tindices = TypeVar("TV_SparseApplyFtrlV2_Tindices", _atypes.Int32, _atypes.Int64)

def sparse_apply_ftrl_v2(var: Annotated[Any, TV_SparseApplyFtrlV2_T], accum: Annotated[Any, TV_SparseApplyFtrlV2_T], linear: Annotated[Any, TV_SparseApplyFtrlV2_T], grad: Annotated[Any, TV_SparseApplyFtrlV2_T], indices: Annotated[Any, TV_SparseApplyFtrlV2_Tindices], lr: Annotated[Any, TV_SparseApplyFtrlV2_T], l1: Annotated[Any, TV_SparseApplyFtrlV2_T], l2: Annotated[Any, TV_SparseApplyFtrlV2_T], l2_shrinkage: Annotated[Any, TV_SparseApplyFtrlV2_T], lr_power: Annotated[Any, TV_SparseApplyFtrlV2_T], use_locking:bool=False, multiply_linear_by_lr:bool=False, name=None) -> Annotated[Any, TV_SparseApplyFtrlV2_T]:
  r"""Update relevant entries in '*var' according to the Ftrl-proximal scheme.

  That is for rows we have grad for, we update var, accum and linear as follows:
  grad_with_shrinkage = grad + 2 * l2_shrinkage * var
  accum_new = accum + grad * grad
  linear += grad_with_shrinkage -
      (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
  quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
  var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
  accum = accum_new

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Should be from a Variable().
    accum: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    linear: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `var`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `var`.
      L2 shrinkage regularization. Must be a scalar.
    l2_shrinkage: A `Tensor`. Must have the same type as `var`.
    lr_power: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    multiply_linear_by_lr: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("sparse_apply_ftrl_v2 op does not support eager execution. Arg 'out' is a ref.")
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if multiply_linear_by_lr is None:
    multiply_linear_by_lr = False
  multiply_linear_by_lr = _execute.make_bool(multiply_linear_by_lr, "multiply_linear_by_lr")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseApplyFtrlV2", var=var, accum=accum, linear=linear, grad=grad,
                             indices=indices, lr=lr, l1=l1, l2=l2,
                             l2_shrinkage=l2_shrinkage, lr_power=lr_power,
                             use_locking=use_locking,
                             multiply_linear_by_lr=multiply_linear_by_lr,
                             name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "use_locking",
              _op._get_attr_bool("use_locking"), "multiply_linear_by_lr",
              _op._get_attr_bool("multiply_linear_by_lr"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseApplyFtrlV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseApplyFtrlV2 = tf_export("raw_ops.SparseApplyFtrlV2")(_ops.to_raw_op(sparse_apply_ftrl_v2))


def sparse_apply_ftrl_v2_eager_fallback(var: Annotated[Any, TV_SparseApplyFtrlV2_T], accum: Annotated[Any, TV_SparseApplyFtrlV2_T], linear: Annotated[Any, TV_SparseApplyFtrlV2_T], grad: Annotated[Any, TV_SparseApplyFtrlV2_T], indices: Annotated[Any, TV_SparseApplyFtrlV2_Tindices], lr: Annotated[Any, TV_SparseApplyFtrlV2_T], l1: Annotated[Any, TV_SparseApplyFtrlV2_T], l2: Annotated[Any, TV_SparseApplyFtrlV2_T], l2_shrinkage: Annotated[Any, TV_SparseApplyFtrlV2_T], lr_power: Annotated[Any, TV_SparseApplyFtrlV2_T], use_locking: bool, multiply_linear_by_lr: bool, name, ctx) -> Annotated[Any, TV_SparseApplyFtrlV2_T]:
  raise RuntimeError("sparse_apply_ftrl_v2 op does not support eager execution. Arg 'out' is a ref.")

TV_SparseApplyMomentum_T = TypeVar("TV_SparseApplyMomentum_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_SparseApplyMomentum_Tindices = TypeVar("TV_SparseApplyMomentum_Tindices", _atypes.Int32, _atypes.Int64)

def sparse_apply_momentum(var: Annotated[Any, TV_SparseApplyMomentum_T], accum: Annotated[Any, TV_SparseApplyMomentum_T], lr: Annotated[Any, TV_SparseApplyMomentum_T], grad: Annotated[Any, TV_SparseApplyMomentum_T], indices: Annotated[Any, TV_SparseApplyMomentum_Tindices], momentum: Annotated[Any, TV_SparseApplyMomentum_T], use_locking:bool=False, use_nesterov:bool=False, name=None) -> Annotated[Any, TV_SparseApplyMomentum_T]:
  r"""Update relevant entries in '*var' and '*accum' according to the momentum scheme.

  Set use_nesterov = True if you want to use Nesterov momentum.

  That is for rows we have grad for, we update var and accum as follows:

  $$accum = accum * momentum + grad$$
  $$var -= lr * accum$$

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Should be from a Variable().
    accum: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Learning rate. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    momentum: A `Tensor`. Must have the same type as `var`.
      Momentum. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    use_nesterov: An optional `bool`. Defaults to `False`.
      If `True`, the tensor passed to compute grad will be
      var - lr * momentum * accum, so in the end, the var you get is actually
      var - lr * momentum * accum.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("sparse_apply_momentum op does not support eager execution. Arg 'out' is a ref.")
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  if use_nesterov is None:
    use_nesterov = False
  use_nesterov = _execute.make_bool(use_nesterov, "use_nesterov")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseApplyMomentum", var=var, accum=accum, lr=lr, grad=grad,
                               indices=indices, momentum=momentum,
                               use_locking=use_locking,
                               use_nesterov=use_nesterov, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "use_locking",
              _op._get_attr_bool("use_locking"), "use_nesterov",
              _op._get_attr_bool("use_nesterov"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseApplyMomentum", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseApplyMomentum = tf_export("raw_ops.SparseApplyMomentum")(_ops.to_raw_op(sparse_apply_momentum))


def sparse_apply_momentum_eager_fallback(var: Annotated[Any, TV_SparseApplyMomentum_T], accum: Annotated[Any, TV_SparseApplyMomentum_T], lr: Annotated[Any, TV_SparseApplyMomentum_T], grad: Annotated[Any, TV_SparseApplyMomentum_T], indices: Annotated[Any, TV_SparseApplyMomentum_Tindices], momentum: Annotated[Any, TV_SparseApplyMomentum_T], use_locking: bool, use_nesterov: bool, name, ctx) -> Annotated[Any, TV_SparseApplyMomentum_T]:
  raise RuntimeError("sparse_apply_momentum op does not support eager execution. Arg 'out' is a ref.")

TV_SparseApplyProximalAdagrad_T = TypeVar("TV_SparseApplyProximalAdagrad_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_SparseApplyProximalAdagrad_Tindices = TypeVar("TV_SparseApplyProximalAdagrad_Tindices", _atypes.Int32, _atypes.Int64)

def sparse_apply_proximal_adagrad(var: Annotated[Any, TV_SparseApplyProximalAdagrad_T], accum: Annotated[Any, TV_SparseApplyProximalAdagrad_T], lr: Annotated[Any, TV_SparseApplyProximalAdagrad_T], l1: Annotated[Any, TV_SparseApplyProximalAdagrad_T], l2: Annotated[Any, TV_SparseApplyProximalAdagrad_T], grad: Annotated[Any, TV_SparseApplyProximalAdagrad_T], indices: Annotated[Any, TV_SparseApplyProximalAdagrad_Tindices], use_locking:bool=False, name=None) -> Annotated[Any, TV_SparseApplyProximalAdagrad_T]:
  r"""Sparse update entries in '*var' and '*accum' according to FOBOS algorithm.

  That is for rows we have grad for, we update var and accum as follows:
  $$accum += grad * grad$$
  $$prox_v = var$$
  $$prox_v -= lr * grad * (1 / sqrt(accum))$$
  $$var = sign(prox_v)/(1+lr*l2) * max{|prox_v|-lr*l1,0}$$

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Should be from a Variable().
    accum: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Learning rate. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `var`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `var`.
      L2 regularization. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, updating of the var and accum tensors will be protected by
      a lock; otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("sparse_apply_proximal_adagrad op does not support eager execution. Arg 'out' is a ref.")
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseApplyProximalAdagrad", var=var, accum=accum, lr=lr, l1=l1,
                                      l2=l2, grad=grad, indices=indices,
                                      use_locking=use_locking, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "use_locking",
              _op._get_attr_bool("use_locking"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseApplyProximalAdagrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseApplyProximalAdagrad = tf_export("raw_ops.SparseApplyProximalAdagrad")(_ops.to_raw_op(sparse_apply_proximal_adagrad))


def sparse_apply_proximal_adagrad_eager_fallback(var: Annotated[Any, TV_SparseApplyProximalAdagrad_T], accum: Annotated[Any, TV_SparseApplyProximalAdagrad_T], lr: Annotated[Any, TV_SparseApplyProximalAdagrad_T], l1: Annotated[Any, TV_SparseApplyProximalAdagrad_T], l2: Annotated[Any, TV_SparseApplyProximalAdagrad_T], grad: Annotated[Any, TV_SparseApplyProximalAdagrad_T], indices: Annotated[Any, TV_SparseApplyProximalAdagrad_Tindices], use_locking: bool, name, ctx) -> Annotated[Any, TV_SparseApplyProximalAdagrad_T]:
  raise RuntimeError("sparse_apply_proximal_adagrad op does not support eager execution. Arg 'out' is a ref.")

TV_SparseApplyProximalGradientDescent_T = TypeVar("TV_SparseApplyProximalGradientDescent_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_SparseApplyProximalGradientDescent_Tindices = TypeVar("TV_SparseApplyProximalGradientDescent_Tindices", _atypes.Int32, _atypes.Int64)

def sparse_apply_proximal_gradient_descent(var: Annotated[Any, TV_SparseApplyProximalGradientDescent_T], alpha: Annotated[Any, TV_SparseApplyProximalGradientDescent_T], l1: Annotated[Any, TV_SparseApplyProximalGradientDescent_T], l2: Annotated[Any, TV_SparseApplyProximalGradientDescent_T], grad: Annotated[Any, TV_SparseApplyProximalGradientDescent_T], indices: Annotated[Any, TV_SparseApplyProximalGradientDescent_Tindices], use_locking:bool=False, name=None) -> Annotated[Any, TV_SparseApplyProximalGradientDescent_T]:
  r"""Sparse update '*var' as FOBOS algorithm with fixed learning rate.

  That is for rows we have grad for, we update var as follows:
  $$prox_v = var - alpha * grad$$
  $$var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}$$

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Should be from a Variable().
    alpha: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `var`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `var`.
      L2 regularization. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, the subtraction will be protected by a lock;
      otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("sparse_apply_proximal_gradient_descent op does not support eager execution. Arg 'out' is a ref.")
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseApplyProximalGradientDescent", var=var, alpha=alpha, l1=l1,
                                              l2=l2, grad=grad,
                                              indices=indices,
                                              use_locking=use_locking,
                                              name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "use_locking",
              _op._get_attr_bool("use_locking"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseApplyProximalGradientDescent", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseApplyProximalGradientDescent = tf_export("raw_ops.SparseApplyProximalGradientDescent")(_ops.to_raw_op(sparse_apply_proximal_gradient_descent))


def sparse_apply_proximal_gradient_descent_eager_fallback(var: Annotated[Any, TV_SparseApplyProximalGradientDescent_T], alpha: Annotated[Any, TV_SparseApplyProximalGradientDescent_T], l1: Annotated[Any, TV_SparseApplyProximalGradientDescent_T], l2: Annotated[Any, TV_SparseApplyProximalGradientDescent_T], grad: Annotated[Any, TV_SparseApplyProximalGradientDescent_T], indices: Annotated[Any, TV_SparseApplyProximalGradientDescent_Tindices], use_locking: bool, name, ctx) -> Annotated[Any, TV_SparseApplyProximalGradientDescent_T]:
  raise RuntimeError("sparse_apply_proximal_gradient_descent op does not support eager execution. Arg 'out' is a ref.")

TV_SparseApplyRMSProp_T = TypeVar("TV_SparseApplyRMSProp_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_SparseApplyRMSProp_Tindices = TypeVar("TV_SparseApplyRMSProp_Tindices", _atypes.Int32, _atypes.Int64)

def sparse_apply_rms_prop(var: Annotated[Any, TV_SparseApplyRMSProp_T], ms: Annotated[Any, TV_SparseApplyRMSProp_T], mom: Annotated[Any, TV_SparseApplyRMSProp_T], lr: Annotated[Any, TV_SparseApplyRMSProp_T], rho: Annotated[Any, TV_SparseApplyRMSProp_T], momentum: Annotated[Any, TV_SparseApplyRMSProp_T], epsilon: Annotated[Any, TV_SparseApplyRMSProp_T], grad: Annotated[Any, TV_SparseApplyRMSProp_T], indices: Annotated[Any, TV_SparseApplyRMSProp_Tindices], use_locking:bool=False, name=None) -> Annotated[Any, TV_SparseApplyRMSProp_T]:
  r"""Update '*var' according to the RMSProp algorithm.

  Note that in dense implementation of this algorithm, ms and mom will
  update even if the grad is zero, but in this sparse implementation, ms
  and mom will not update in iterations during which the grad is zero.

  mean_square = decay * mean_square + (1-decay) * gradient ** 2
  Delta = learning_rate * gradient / sqrt(mean_square + epsilon)

  $$ms <- rho * ms_{t-1} + (1-rho) * grad * grad$$
  $$mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)$$
  $$var <- var - mom$$

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Should be from a Variable().
    ms: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    mom: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    rho: A `Tensor`. Must have the same type as `var`.
      Decay rate. Must be a scalar.
    momentum: A `Tensor`. Must have the same type as `var`.
    epsilon: A `Tensor`. Must have the same type as `var`.
      Ridge term. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var, ms and mom.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var, ms, and mom tensors is protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("sparse_apply_rms_prop op does not support eager execution. Arg 'out' is a ref.")
  # Add nodes to the TensorFlow graph.
  if use_locking is None:
    use_locking = False
  use_locking = _execute.make_bool(use_locking, "use_locking")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseApplyRMSProp", var=var, ms=ms, mom=mom, lr=lr, rho=rho,
                              momentum=momentum, epsilon=epsilon, grad=grad,
                              indices=indices, use_locking=use_locking,
                              name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "use_locking",
              _op._get_attr_bool("use_locking"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseApplyRMSProp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SparseApplyRMSProp = tf_export("raw_ops.SparseApplyRMSProp")(_ops.to_raw_op(sparse_apply_rms_prop))


def sparse_apply_rms_prop_eager_fallback(var: Annotated[Any, TV_SparseApplyRMSProp_T], ms: Annotated[Any, TV_SparseApplyRMSProp_T], mom: Annotated[Any, TV_SparseApplyRMSProp_T], lr: Annotated[Any, TV_SparseApplyRMSProp_T], rho: Annotated[Any, TV_SparseApplyRMSProp_T], momentum: Annotated[Any, TV_SparseApplyRMSProp_T], epsilon: Annotated[Any, TV_SparseApplyRMSProp_T], grad: Annotated[Any, TV_SparseApplyRMSProp_T], indices: Annotated[Any, TV_SparseApplyRMSProp_Tindices], use_locking: bool, name, ctx) -> Annotated[Any, TV_SparseApplyRMSProp_T]:
  raise RuntimeError("sparse_apply_rms_prop op does not support eager execution. Arg 'out' is a ref.")
