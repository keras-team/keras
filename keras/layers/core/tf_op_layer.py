# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Contains the TFOpLambda layer."""
# pylint: disable=g-classes-have-attributes,g-direct-tensorflow-import,g-bad-import-order
import tensorflow.compat.v2 as tf
# pylint: enable=g-bad-import-order

from keras import backend
from keras.engine import keras_tensor
from keras.engine.base_layer import Layer

from tensorflow.python.platform import tf_logging
from tensorflow.python.util.tf_export import get_canonical_name_for_symbol
from tensorflow.python.util.tf_export import get_symbol_from_name


class ClassMethod(Layer):
  """Wraps a TF API Class's class method  in a `Layer` object.

  It is inserted by the Functional API construction whenever users call
  a supported TF Class's class method on KerasTensors.

  This is useful in the case where users do something like:
  x = keras.Input(...)
  y = keras.Input(...)
  out = tf.RaggedTensor.from_row_splits(x, y)
  """

  @tf.__internal__.tracking.no_automatic_dependency_tracking
  def __init__(self, cls_ref, method_name, **kwargs):
    self.cls_ref = cls_ref
    self.method_name = method_name
    self.cls_symbol = (
        get_canonical_name_for_symbol(
            self.cls_ref, add_prefix_to_v1_names=True) or
        get_canonical_name_for_symbol(
            self.cls_ref, api_name='keras', add_prefix_to_v1_names=True))
    if 'name' not in kwargs:
      kwargs['name'] = backend.unique_object_name(
          'tf.' + self.cls_symbol + '.' + self.method_name,
          zero_based=True,
          avoid_observed_names=True)
    kwargs['autocast'] = False

    # Do not individually trace op layers in the SavedModel.
    self._must_restore_from_config = True

    super(ClassMethod, self).__init__(**kwargs)

    # Preserve all argument data structures when saving/loading a config
    # (e.g., don't unnest lists that contain one element)
    self._preserve_input_structure_in_config = True

    self._expects_training_arg = False
    self._expects_mask_arg = False

  def call(self, args, kwargs):
    return getattr(self.cls_ref, self.method_name)(*args, **kwargs)

  def get_config(self):
    if not self.cls_symbol:
      raise ValueError(
          'This Keras class method conversion tried to convert '
          f'a method belonging to class {self.cls_symbol}, a class '
          'that is not publicly exposed in the TensorFlow API. '
          'To ensure cross-version compatibility of Keras models '
          'that use op layers, only op layers produced from '
          'public TensorFlow API symbols can be serialized.')

    config = {'cls_symbol': self.cls_symbol, 'method_name': self.method_name}
    base_config = super(ClassMethod, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    config = config.copy()
    symbol_name = config.pop('cls_symbol')
    cls_ref = get_symbol_from_name(symbol_name)
    if not cls_ref:
      raise ValueError(f'TensorFlow symbol `{symbol_name}` could not be found.')

    config['cls_ref'] = cls_ref

    return cls(**config)


class KerasOpDispatcher(tf.__internal__.dispatch.GlobalOpDispatcher):
  """A global dispatcher that allows building a functional model with TF Ops."""

  def handle(self, op, args, kwargs):
    """Handle the specified operation with the specified arguments."""
    if any(
        isinstance(x, keras_tensor.KerasTensor)
        for x in tf.nest.flatten([args, kwargs])):
      return TFOpLambda(op)(*args, **kwargs)
    else:
      return self.NOT_SUPPORTED


KerasOpDispatcher().register()


class InstanceProperty(Layer):
  """Wraps an instance property access (e.g.

  `x.foo`) in a Keras Layer.

  This layer takes an attribute name `attr_name` in the constructor and,
  when called on input tensor `obj` returns `obj.attr_name`.

  KerasTensors specialized for specific extension types use it to
  represent instance property accesses on the represented object in the
  case where the property needs to be dynamically accessed as opposed to
  being statically computed from the typespec, e.g.

  x = keras.Input(..., ragged=True)
  out = x.flat_values
  """

  @tf.__internal__.tracking.no_automatic_dependency_tracking
  def __init__(self, attr_name, **kwargs):
    self.attr_name = attr_name

    if 'name' not in kwargs:
      kwargs['name'] = backend.unique_object_name(
          'input.' + self.attr_name, zero_based=True, avoid_observed_names=True)
    kwargs['autocast'] = False

    # Do not individually trace op layers in the SavedModel.
    self._must_restore_from_config = True

    super(InstanceProperty, self).__init__(**kwargs)

    # Preserve all argument data structures when saving/loading a config
    # (e.g., don't unnest lists that contain one element)
    self._preserve_input_structure_in_config = True

  def call(self, obj):
    return getattr(obj, self.attr_name)

  def get_config(self):
    config = {'attr_name': self.attr_name}
    base_config = super(InstanceProperty, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)


class InstanceMethod(InstanceProperty):
  """Wraps an instance method access (e.g. `x.foo(arg)` in a Keras Layer.

  This layer takes an attribute name `attr_name` in the constructor and,
  when called on input tensor `obj` with additional arguments `args` and
  `kwargs` returns `obj.attr_name(*args, **kwargs)`.

  KerasTensors specialized for specific extension types use it to
  represent dynamic instance method calls on the represented object, e.g.

  x = keras.Input(..., ragged=True)
  new_values = keras.Input(...)
  out = x.with_values(new_values)
  """

  def call(self, obj, args, kwargs):
    method = getattr(obj, self.attr_name)
    return method(*args, **kwargs)


class TFOpLambda(Layer):
  """Wraps TF API symbols in a `Layer` object.

  It is inserted by the Functional API construction whenever users call
  a supported TF symbol on KerasTensors.

  Like Lambda layers, this layer tries to raise warnings when it detects users
  explicitly use variables in the call. (To let them know
  that the layer will not capture the variables).

  This is useful in the case where users do something like:
  x = keras.Input(...)
  y = tf.Variable(...)
  out = x * tf_variable
  """

  @tf.__internal__.tracking.no_automatic_dependency_tracking
  def __init__(self, function, **kwargs):
    self.function = function
    self.symbol = (
        get_canonical_name_for_symbol(
            self.function, add_prefix_to_v1_names=True) or
        get_canonical_name_for_symbol(
            self.function, api_name='keras', add_prefix_to_v1_names=True))
    if 'name' not in kwargs:
      # Generate a name.
      # TFOpLambda layers avoid already-observed names,
      # because users cannot easily control the generated names.
      # Without this avoidance, users would be more likely to run
      # into unavoidable duplicate layer name collisions.
      # (For standard layers users could just set `name` when creating the
      # layer to work around a collision, but they can't do that for
      # auto-generated layers)
      if self.symbol:
        name = 'tf.' + self.symbol
      else:
        name = self.function.__name__
      kwargs['name'] = backend.unique_object_name(
          name, zero_based=True, avoid_observed_names=True)
    kwargs['autocast'] = False

    # Decorate the function to produce this layer's call method
    def _call_wrapper(*args, **kwargs):
      return self._call_wrapper(*args, **kwargs)

    self.call = tf.__internal__.decorator.make_decorator(
        function, _call_wrapper)

    # Do not individually trace op layers in the SavedModel.
    self._must_restore_from_config = True

    super(TFOpLambda, self).__init__(**kwargs)

    # Preserve all argument data structures when saving/loading a config
    # (e.g., don't unnest lists that contain one element)
    self._preserve_input_structure_in_config = True

    # Warning on every invocation will be quite irksome in Eager mode.
    self._already_warned = False

    self._expects_training_arg = False
    self._expects_mask_arg = False

  def _call_wrapper(self, *args, **kwargs):
    created_variables = []

    def _variable_creator(next_creator, **creator_kwargs):
      var = next_creator(**creator_kwargs)
      created_variables.append(var)
      return var

    with tf.GradientTape(watch_accessed_variables=True) as tape, \
        tf.variable_creator_scope(_variable_creator):
      # We explicitly drop `name` arguments here,
      # to guard against the case where an op explicitly has a
      # `name` passed (which is susceptible to producing
      # multiple ops w/ the same name when the layer is reused)
      kwargs.pop('name', None)
      result = self.function(*args, **kwargs)
    self._check_variables(created_variables, tape.watched_variables())
    return result

  def _check_variables(self, created_variables, accessed_variables):
    if not created_variables and not accessed_variables:
      # In the common case that a Lambda layer does not touch a Variable, we
      # don't want to incur the runtime cost of assembling any state used for
      # checking only to immediately discard it.
      return

    tracked_weights = set(v.ref() for v in self.weights)
    untracked_new_vars = [
        v for v in created_variables if v.ref() not in tracked_weights
    ]
    if untracked_new_vars:
      variable_str = '\n'.join('  {}'.format(i) for i in untracked_new_vars)
      raise ValueError(
          'The following Variables were created within a Lambda layer '
          f'({self.name}) but are not tracked by said layer: {variable_str}\n'
          'The layer cannot safely ensure proper Variable reuse '
          'across multiple calls, and consquently this behavior is disallowed '
          'for safety reasons. Lambda layers are not well suited for stateful '
          'computation; instead, writing a subclassed Layer is the recommend '
          'way to define layers with Variables.')

    untracked_used_vars = [
        v for v in accessed_variables if v.ref() not in tracked_weights
    ]
    if untracked_used_vars and not self._already_warned:
      variable_str = '\n'.join('  {}'.format(i) for i in untracked_used_vars)
      self._warn(
          'The following Variables were used in a Lambda layer\'s call '
          f'({self.name}), but are not present in its tracked objects: '
          f'{variable_str}. This is a strong indication that the Lambda layer '
          'should be rewritten as a subclassed Layer.')
      self._already_warned = True

  def _warn(self, msg):
    # This method will be overridden in a unit test to raise an error, because
    # self.assertWarns is not universally implemented.
    return tf_logging.warning(msg)

  def get_config(self):
    if not self.symbol:
      raise ValueError(
          f'This Keras op layer was generated from {self.function}, a method '
          'that is not publicly exposed in the TensorFlow API. This '
          'may have happened if the method was explicitly '
          'decorated to add dispatching support, and it was used '
          'during Functional model construction. '
          'To ensure cross-version compatibility of Keras models '
          'that use op layers, only op layers produced from '
          'public TensorFlow API symbols can be serialized.')
    config = {'function': self.symbol}

    base_config = super(TFOpLambda, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    config = config.copy()
    symbol_name = config['function']
    function = get_symbol_from_name(symbol_name)
    if not function:
      raise ValueError(f'TF symbol `{symbol_name}` could not be found.')

    config['function'] = function

    return cls(**config)


def _delegate_property(keras_tensor_cls, property_name):  # pylint: disable=invalid-name
  """Register property on a KerasTensor class.

  Calling this multiple times with the same arguments should be a no-op.

  This method exposes a property on the KerasTensor class that will use an
  `InstanceProperty` layer to access the property on the represented
  intermediate values in the model.

  Args:
    keras_tensor_cls: The KerasTensor subclass that should expose the property.
    property_name: The name of the property to expose and delegate to the
      represented (Composite)Tensor.
  """
  # We use a lambda because we can't create a Keras layer at import time
  # due to dynamic layer class versioning.
  property_access = property(lambda self: InstanceProperty(property_name)(self))  # pylint: disable=unnecessary-lambda
  setattr(keras_tensor_cls, property_name, property_access)


def _delegate_method(keras_tensor_cls, method_name):  # pylint: disable=invalid-name
  """Register method on a KerasTensor class.

  Calling this function times with the same arguments should be a no-op.

  This method exposes an instance method on the KerasTensor class that will use
  an `InstanceMethod` layer to run the desired method on the represented
  intermediate values in the model.

  Args:
    keras_tensor_cls: The KerasTensor subclass that should expose the property.
    method_name: The name of the method to expose and delegate to the
      represented (Composite)Tensor.
  """

  def delegate(self, *args, **kwargs):
    return InstanceMethod(method_name)(self, args, kwargs)

  setattr(keras_tensor_cls, method_name, delegate)


# We do not support the `uniform_row_length` property because it
# returns either `None` or an int tensor, and code that relies on it tends
# to check `is None` directly. Delegating it here would always return a
# `KerasTensor`, regardless of what can be statically inferred. This would
# never equal `None`, breaking code that expects it to be partially-static
# in unpredictable ways.
for ragged_property in [
    'values', 'flat_values', 'row_splits', 'nested_row_splits'
]:
  _delegate_property(keras_tensor.RaggedKerasTensor, ragged_property)

for ragged_method_name in [
    'value_rowids',
    'nested_value_rowids',
    'nrows',
    'row_starts',
    'row_limits',
    'row_lengths',
    'nested_row_lengths',
    'bounding_shape',
    'with_values',
    'with_flat_values',
    'with_row_splits_dtype',
    'merge_dims',
    'to_tensor',
    'to_sparse',
]:
  _delegate_method(keras_tensor.RaggedKerasTensor, ragged_method_name)

for sparse_property in [
    'indices',
    'values',
    'dense_shape',
]:
  _delegate_property(keras_tensor.SparseKerasTensor, sparse_property)

for sparse_method in [
    'with_values',
]:
  _delegate_method(keras_tensor.SparseKerasTensor, sparse_method)


class TFClassMethodDispatcher(tf.__internal__.dispatch.OpDispatcher):
  """A class method dispatcher that allows building a functional model with TF class methods."""

  def __init__(self, cls, method_name):
    self.cls = cls
    self.method_name = method_name

  def handle(self, args, kwargs):
    """Handle the specified operation with the specified arguments."""
    if any(
        isinstance(x, keras_tensor.KerasTensor)
        for x in tf.nest.flatten([args, kwargs])):
      return ClassMethod(self.cls, self.method_name)(args[1:], kwargs)
    else:
      return self.NOT_SUPPORTED


for ragged_class_method in [
    'from_value_rowids',
    'from_row_splits',
    'from_row_lengths',
    'from_row_starts',
    'from_row_limits',
    'from_uniform_row_length',
    'from_nested_value_rowids',
    'from_nested_row_splits',
    'from_nested_row_lengths',
    'from_tensor',
    'from_sparse',
]:
  TFClassMethodDispatcher(tf.RaggedTensor, ragged_class_method).register(
      getattr(tf.RaggedTensor, ragged_class_method))


class SlicingOpLambda(TFOpLambda):
  """Wraps TF API symbols in a `Layer` object.

  It is inserted by the Functional API construction whenever users call
  a supported TF symbol on KerasTensors.

  Like Lambda layers, this layer tries to raise warnings when it detects users
  explicitly use variables in the call. (To let them know
  that the layer will not capture the variables).

  This is useful in the case where users do something like:
  x = keras.Input(...)
  y = tf.Variable(...)
  out = x * tf_variable
  """

  @tf.__internal__.tracking.no_automatic_dependency_tracking
  def __init__(self, function, **kwargs):
    super(SlicingOpLambda, self).__init__(function, **kwargs)

    original_call = self.call

    # Decorate the function to produce this layer's call method
    def _call_wrapper(*args, **kwargs):
      # Turn any slice dicts in the args back into `slice` objects.
      # This conversion cannot use nest.flatten/map_structure,
      # because dicts are flattened by nest while slices aren't.
      # So, map_structure would only see the individual elements in the
      # dict.
      # This can't use map_structure_up_to either because the 'shallowness' of
      # the shallow tree would have to vary depending on if only one dim or
      # multiple are being sliced.
      new_args = []
      for arg in args:
        arg = _dict_to_slice(arg)
        if isinstance(arg, (list, tuple)):
          new_arg = []
          for sub_arg in arg:
            new_arg.append(_dict_to_slice(sub_arg))
          arg = new_arg
        new_args.append(arg)

      # Handle the kwargs too.
      new_kwargs = {}
      for key, value in kwargs.items():
        value = _dict_to_slice(value)
        if isinstance(value, (list, tuple)):
          new_value = []
          for v in value:
            new_value.append(_dict_to_slice(v))
          value = new_value
        new_kwargs[key] = value

      return original_call(*new_args, **new_kwargs)

    self.call = tf.__internal__.decorator.make_decorator(
        original_call, _call_wrapper)


def _slice_to_dict(x):
  if isinstance(x, slice):
    return {'start': x.start, 'stop': x.stop, 'step': x.step}
  return x


def _dict_to_slice(x):
  if isinstance(x, dict):
    return slice(x['start'], x['stop'], x['step'])
  return x


class TFSlicingOpDispatcher(tf.__internal__.dispatch.OpDispatcher):
  """A global dispatcher that allows building a functional model with TF Ops."""

  def __init__(self, op):
    self.op = op

  def handle(self, args, kwargs):
    """Handle the specified operation with the specified arguments."""
    args = tf.nest.map_structure(_slice_to_dict, args)
    kwargs = tf.nest.map_structure(_slice_to_dict, kwargs)
    if any(
        isinstance(x, keras_tensor.KerasTensor)
        for x in tf.nest.flatten([args, kwargs])):
      return SlicingOpLambda(self.op)(*args, **kwargs)
    else:
      return self.NOT_SUPPORTED


for slicing_op in [
    tf.__operators__.getitem,  # pylint: disable=protected-access
    tf.compat.v1.boolean_mask,
    tf.boolean_mask,
    tf.__operators__.ragged_getitem
]:
  TFSlicingOpDispatcher(slicing_op).register(slicing_op)
