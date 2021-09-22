# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for TensorFlow 2.0 layer behavior."""
# pylint: disable=g-bad-import-order
import tensorflow.compat.v2 as tf

import copy
import os

import numpy as np
from keras import backend
from keras import combinations
from keras import keras_parameterized
from keras import layers
from keras import regularizers
from keras import testing_utils
from keras.engine import base_layer
from keras.engine import input_layer
from keras.engine import sequential
from keras.engine import training as training_lib
from keras.legacy_tf_layers import core as legacy_core
from keras.optimizer_v2 import rmsprop
from keras.utils import control_flow_util


class DynamicLayer(base_layer.Layer):

  def __init__(self, dynamic=False, **kwargs):
    super(DynamicLayer, self).__init__(dynamic=dynamic, **kwargs)

  def call(self, inputs):
    samples = tf.TensorArray(
        dtype=tf.float32, size=tf.shape(inputs)[0])
    for idx, sample in enumerate(inputs):
      samples = samples.write(idx, tf.square(sample))
    return samples.stack()

  def compute_output_shape(self, input_shape):
    return input_shape


class InvalidLayer(base_layer.Layer):

  def call(self, inputs):
    raise ValueError('You did something wrong!')


class BaseLayerTest(keras_parameterized.TestCase):

  @combinations.generate(combinations.keras_mode_combinations())
  def test_layer_instrumentation(self):
    layer = layers.Add()
    self.assertTrue(layer._instrumented_keras_api)
    self.assertTrue(layer._instrumented_keras_layer_class)
    self.assertFalse(layer._instrumented_keras_model_class)
    self.assertTrue(base_layer.keras_api_gauge.get_cell('tf.keras.layers.Add'))

    # Verify this was not instrumented as a legacy layer
    self.assertFalse(
        base_layer.keras_api_gauge.get_cell('legacy_layer').value())
    base_layer.keras_api_gauge.get_cell('tf.keras.layers.Add').set(False)

  @combinations.generate(combinations.keras_model_type_combinations())
  def test_dynamic_layer(self):
    model = testing_utils.get_model_from_layers([DynamicLayer(dynamic=True)],
                                                input_shape=(3,))
    self.assertEqual(model.dynamic, True)
    model.compile(rmsprop.RMSprop(0.001), loss='mse')
    self.assertEqual(model.run_eagerly, True)
    model.train_on_batch(np.random.random((2, 3)), np.random.random((2, 3)))

  @combinations.generate(combinations.keras_model_type_combinations())
  def test_dynamic_layer_error(self):
    # Functional Models hit the `dyanamic=True` error during construction.
    # Subclass Models should just throw the original autograph error during
    # execution.
    raised_error = False
    try:
      model = testing_utils.get_model_from_layers([DynamicLayer()],
                                                  input_shape=(3,))
      model.compile(rmsprop.RMSprop(0.001), loss='mse')
      model.train_on_batch(np.random.random((2, 3)), np.random.random((2, 3)))
    except tf.errors.OperatorNotAllowedInGraphError as e:
      if 'iterating over `tf.Tensor` is not allowed' in str(e):
        raised_error = True
    except TypeError as e:
      if 'attempting to use Python control flow' in str(e):
        raised_error = True
    self.assertTrue(raised_error)

  @combinations.generate(combinations.keras_model_type_combinations())
  def test_dynamic_layer_error_running_in_graph_mode(self):
    with tf.compat.v1.get_default_graph().as_default():
      model = testing_utils.get_model_from_layers([DynamicLayer(dynamic=True)],
                                                  input_shape=(3,))
      self.assertEqual(model.dynamic, True)
      # But then you cannot run the model since you're in a graph scope.
      with self.assertRaisesRegex(ValueError,
                                  'You must enable eager execution'):
        model.compile(rmsprop.RMSprop(0.001), loss='mse')

  def test_manual_compute_output_shape(self):

    class BuildCounter(base_layer.Layer):

      def __init__(self, *args, **kwargs):  # pylint: disable=redefined-outer-name
        super(BuildCounter, self).__init__(*args, **kwargs)
        self.build_counter = 0

      def build(self, input_shape):
        self.build_counter += 1
        self.build_shape = input_shape

      def call(self, inputs):
        return inputs

    layer = BuildCounter(dtype=tf.float64)
    output_shape = layer.compute_output_shape((None, 10))
    self.assertEqual(layer.build_counter, 1)
    self.assertEqual(layer.build_shape.as_list(), [None, 10])
    self.assertEqual(output_shape.as_list(), [None, 10])
    output_signature = layer.compute_output_signature(
        tf.TensorSpec(dtype=tf.float64, shape=[None, 10]))
    self.assertEqual(layer.build_counter, 1)
    self.assertEqual(layer.build_shape.as_list(), [None, 10])
    self.assertEqual(output_signature.dtype, tf.float64)
    self.assertEqual(output_signature.shape.as_list(), [None, 10])
    layer(np.ones((5, 10)))
    self.assertEqual(layer.build_counter, 1)
    self.assertEqual(layer.build_shape.as_list(), [None, 10])

  def test_dynamic_layer_with_deferred_sequential_model(self):
    model = sequential.Sequential([DynamicLayer(dynamic=True), layers.Dense(3)])
    self.assertEqual(model.dynamic, True)
    model.compile(rmsprop.RMSprop(0.001), loss='mse')
    self.assertEqual(model.run_eagerly, True)
    model.train_on_batch(np.random.random((2, 3)), np.random.random((2, 3)))

  def test_nested_dynamic_layers_in_eager_mode(self):
    inputs = input_layer.Input((3,))
    outputs = DynamicLayer(dynamic=True)(inputs)
    inner_model = training_lib.Model(inputs, outputs)
    self.assertEqual(inner_model.dynamic, True)

    inputs = input_layer.Input((3,))
    x = DynamicLayer(dynamic=True)(inputs)
    outputs = inner_model(x)

    model = training_lib.Model(inputs, outputs)
    self.assertEqual(model.dynamic, True)
    model.compile(rmsprop.RMSprop(0.001), loss='mse')
    self.assertEqual(model.run_eagerly, True)
    model.train_on_batch(np.random.random((2, 3)), np.random.random((2, 3)))

  def test_dynamic_subclassed_model_no_shape_inference(self):

    class MyModel(training_lib.Model):

      def __init__(self):
        super(MyModel, self).__init__(dynamic=True)
        self.layer1 = layers.Dense(3)
        self.layer2 = layers.Dense(3)

      def call(self, inputs):
        if tf.reduce_sum(inputs) > 0:
          return self.layer1(inputs)
        else:
          return self.layer2(inputs)

    model = MyModel()
    self.assertEqual(model.dynamic, True)
    model.compile(rmsprop.RMSprop(0.001), loss='mse')
    self.assertEqual(model.run_eagerly, True)
    model.train_on_batch(np.random.random((2, 3)), np.random.random((2, 3)))
    self.assertEqual(model.outputs, None)

  def test_dynamic_subclassed_model_with_shape_inference(self):

    class MyModel(training_lib.Model):

      def __init__(self):
        super(MyModel, self).__init__(dynamic=True)
        self.layer1 = layers.Dense(3)
        self.layer2 = layers.Dense(3)

      def call(self, inputs):
        if tf.reduce_sum(inputs) > 0:
          return self.layer1(inputs)
        else:
          return self.layer2(inputs)

      def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1].as_list()) + (3,)

    model = MyModel()
    self.assertEqual(model.dynamic, True)
    model.compile(rmsprop.RMSprop(0.001), loss='mse')
    x, y = np.random.random((2, 3)), np.random.random((2, 3))
    model.train_on_batch(x, y)
    outputs = model(x)
    self.assertEqual(outputs.shape.as_list(), [2, 3])

  def test_deepcopy(self):
    bias_reg = lambda x: 1e-3 * tf.reduce_sum(x)
    layer = layers.Conv2D(32, (3, 3), bias_regularizer=bias_reg)
    # Call the Layer on data to generate regularize losses.
    layer(tf.ones((1, 10, 10, 3)))
    self.assertLen(layer.losses, 1)
    new_layer = copy.deepcopy(layer)
    self.assertEqual(new_layer.bias_regularizer, bias_reg)
    self.assertEqual(layer.get_config(), new_layer.get_config())

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_invalid_forward_pass(self):
    inputs = input_layer.Input((3,))
    with self.assertRaisesRegex(ValueError, 'You did something wrong!'):
      _ = InvalidLayer()(inputs)

  def test_no_legacy_model(self):
    inputs = input_layer.Input((1,))
    legacy_dense_0 = legacy_core.Dense(1, name='legacy_dense_0')
    legacy_dense_1 = legacy_core.Dense(1, name='legacy_dense_1')

    layer = legacy_dense_0(inputs)
    layer = layers.Dense(1)(layer)
    layer = legacy_dense_1(layer)

    expected_regex = (r'The following are legacy tf\.layers\.Layers:\n  '
                      '{}\n  {}'.format(legacy_dense_0, legacy_dense_1))

    with self.assertRaisesRegex(TypeError, expected_regex):
      _ = training_lib.Model(inputs=[inputs], outputs=[layer])

    model = training_lib.Model(inputs=[inputs], outputs=[inputs])
    with self.assertRaisesRegex(TypeError, expected_regex):
      model._insert_layers([legacy_dense_0, legacy_dense_1])

  def test_no_legacy_sequential(self):
    layer = [layers.Dense(1), legacy_core.Dense(1, name='legacy_dense_0')]

    expected_regex = r'legacy tf\.layers\.Layers:\n  {}'.format(layer[1])
    with self.assertRaisesRegex(TypeError, expected_regex):
      _ = sequential.Sequential(layer)

    with self.assertRaisesRegex(TypeError, expected_regex):
      _ = sequential.Sequential([input_layer.Input(shape=(4,))] + layer)

    model = sequential.Sequential()
    with self.assertRaisesRegex(TypeError, expected_regex):
      for l in layer:
        model.add(l)

  @combinations.generate(
      combinations.times(
          combinations.keras_model_type_combinations(),
          combinations.combine(mode=['graph', 'eager'])))
  def test_build_with_numpy_data(self):
    model_layers = [
        layers.Dense(3, activation='relu', kernel_initializer='ones'),
        layers.Dense(1, activation='sigmoid', kernel_initializer='ones')
    ]
    model = testing_utils.get_model_from_layers(model_layers, input_shape=(4,))
    model(np.zeros((2, 4), dtype='float32'))
    self.assertTrue(model.built)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_default_add_weight(self):

    class TestLayer(base_layer.Layer):

      def __init__(self):
        super(TestLayer, self).__init__()
        self.default_weight = self.add_weight()
        self.weight_without_name = self.add_weight(shape=(3, 4))
        self.regularized_weight_without_name = self.add_weight(
            shape=(3, 4), regularizer='l2')

    layer = TestLayer()
    self.assertEqual(layer.default_weight.shape.as_list(), [])
    self.assertEqual(layer.weight_without_name.shape.as_list(), [3, 4])
    self.assertEqual(layer.default_weight.dtype.name, 'float32')
    self.assertEqual(layer.weight_without_name.dtype.name, 'float32')
    self.assertEqual(len(layer.losses), 1)
    if not tf.executing_eagerly():
      # Cannot access tensor.name in eager execution.
      self.assertIn('Variable_2/Regularizer', layer.losses[0].name)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_add_weight_by_getter(self):
    layer = base_layer.Layer()
    variable = tf.Variable('abc')
    added = layer.add_weight(
        dtype=tf.string, getter=lambda *_, **__: variable)
    self.assertIs(variable, added)

  @combinations.generate(combinations.keras_mode_combinations(mode=['eager']))
  def test_learning_phase_freezing_for_layers(self):

    class LearningPhaseLayer(base_layer.Layer):

      def call(self, inputs):
        return backend.in_train_phase(lambda: tf.ones_like(inputs),
                                      lambda: tf.zeros_like(inputs))

    def get_learning_phase_value():
      model = sequential.Sequential([LearningPhaseLayer(input_shape=(1,))])
      model._run_eagerly = testing_utils.should_run_eagerly()
      return np.sum(model(np.ones((1, 1))))

    self.assertEqual(get_learning_phase_value(), 0)

    # Test scope.
    with backend.learning_phase_scope(1):
      self.assertEqual(get_learning_phase_value(), 1)

    # The effects of the scope end after exiting it.
    self.assertEqual(get_learning_phase_value(), 0)

    # Test setting.
    backend.set_learning_phase(1)
    self.assertEqual(get_learning_phase_value(), 1)
    backend.set_learning_phase(0)
    self.assertEqual(get_learning_phase_value(), 0)

  # Cannot be enabled with `run_eagerly=True`, see b/123904578
  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_layer_can_return_variable(self):

    class ComputeSum(base_layer.Layer):

      def __init__(self):
        super(ComputeSum, self).__init__()
        self.total = tf.Variable(
            initial_value=tf.zeros((1, 1)), trainable=False)
        if not tf.executing_eagerly():
          backend.get_session().run(self.total.initializer)

      def call(self, inputs):
        self.total.assign_add(inputs)
        return self.total

    inputs = input_layer.Input(shape=(1,))
    model = training_lib.Model(inputs, ComputeSum()(inputs))
    model.predict(np.ones((1, 1)))

  def _get_layer_with_training_arg(self):

    class TrainingLayer(base_layer.Layer):
      """A layer with a `training` argument in a defuned `call`."""

      @tf.function
      def call(self, inputs, training=None):
        if training is None:
          training = backend.learning_phase()
        return control_flow_util.smart_cond(
            training, lambda: tf.ones_like(inputs),
            lambda: tf.zeros_like(inputs))

    return TrainingLayer()

  # b/124459427: can't test with `run_eagerly=True` for now.
  @combinations.generate(
      combinations.times(combinations.keras_mode_combinations(),
                         combinations.keras_model_type_combinations()))
  def test_training_arg_in_defun(self):
    layer = self._get_layer_with_training_arg()
    model = testing_utils.get_model_from_layers([layer], input_shape=(1,))
    model.compile(rmsprop.RMSprop(0.),
                  loss='mae')
    history = model.fit(np.zeros((1, 1)), np.zeros((1, 1)))
    self.assertEqual(history.history['loss'][0], 1.)
    loss = model.evaluate(np.zeros((1, 1)), np.zeros((1, 1)))
    self.assertEqual(loss, 0.)

    # Test that the argument injection performed in `call` is not active
    # when the argument is passed explicitly.
    layer = self._get_layer_with_training_arg()
    inputs = input_layer.Input(shape=(1,))
    # Pass `training` by name
    outputs = layer(inputs, training=False)
    model = training_lib.Model(inputs, outputs)
    model.compile(rmsprop.RMSprop(0.),
                  loss='mae')
    history = model.fit(np.zeros((1, 1)), np.zeros((1, 1)))
    self.assertEqual(history.history['loss'][0], 0.)

  @combinations.generate(
      combinations.times(combinations.keras_mode_combinations(),
                         combinations.keras_model_type_combinations()))
  def test_raw_variable_assignment(self):

    class RawVariableLayer(base_layer.Layer):

      def __init__(self, **kwargs):
        super(RawVariableLayer, self).__init__(**kwargs)
        # Test variables in nested structure.
        self.var_list = [tf.Variable(1.), {'a': tf.Variable(2.)}]

      def call(self, inputs):
        return inputs * self.var_list[0] * self.var_list[1]['a']

    model = testing_utils.get_model_from_layers([RawVariableLayer()],
                                                input_shape=(10,))
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    x, y = np.ones((10, 10)), np.ones((10, 10))
    # Checks that variables get initialized.
    model.fit(x, y, batch_size=2, epochs=2)

  @combinations.generate(combinations.combine(mode=['eager']))
  def test_composite_variable_assignment(self):

    class Spec(tf.TypeSpec):

      value_type = property(lambda self: CompositeVariable)

      def _component_specs(self):
        pass

      def _serialize(self):
        pass

      def _to_components(self, value):
        return value._variables

      def _from_components(self, variable_list):
        return CompositeVariable(variable_list)

    class CompositeVariable(tf.__internal__.CompositeTensor):

      def __init__(self, variable_list):
        self._variables = variable_list

      @property
      def _type_spec(self):
        return Spec()

    class CompositeVariableLayer(base_layer.Layer):

      def __init__(self):
        super().__init__()
        self.composite_var = CompositeVariable(
            [tf.Variable(1.),
             tf.Variable(2.)])

    layer = CompositeVariableLayer()
    self.assertLen(layer.weights, 2)
    self.assertIsInstance(layer.weights[0], tf.Variable)
    self.assertIsInstance(layer.weights[1], tf.Variable)
    self.assertEqual(self.evaluate(layer.weights[0]), 1.)
    self.assertEqual(self.evaluate(layer.weights[1]), 2.)

  def test_exception_if_trainable_not_boolean(self):
    base_layer.Layer(trainable=True)
    base_layer.Layer(trainable=tf.constant(True))
    base_layer.Layer(trainable=tf.Variable(tf.constant(True)))
    with self.assertRaisesRegex(
        TypeError, 'Expected `trainable` argument to be a boolean'):
      base_layer.Layer(trainable=0)

  def test_exception_if_dynamic_not_boolean(self):
    base_layer.Layer(dynamic=True)
    with self.assertRaisesRegex(TypeError,
                                'Expected `dynamic` argument to be a boolean'):
      base_layer.Layer(dynamic=0)

  def test_exception_if_name_not_string_or_none(self):
    base_layer.Layer(name=None)
    base_layer.Layer(name='layer_name')
    with self.assertRaisesRegex(TypeError,
                                'Expected `name` argument to be a string'):
      base_layer.Layer(name=0)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_layer_names(self):
    inputs = input_layer.Input(shape=[2])
    add1 = inputs + inputs
    add2 = layers.Add()([inputs, inputs])
    add3 = inputs + inputs
    add4 = layers.Add()([inputs, inputs])
    model = training_lib.Model(inputs=[inputs],
                               outputs=[add1, add2, add3, add4])
    actual_names = [l.name for l in model.layers]
    graph_names = [
        'input_1', 'tf_op_layer_add', 'add', 'tf_op_layer_add_2', 'add_1'
    ]
    eager_names = [
        'input_1', 'tf.__operators__.add', 'add', 'tf.__operators__.add_1',
        'add_1'
    ]
    for actual, eager, graph in zip(actual_names, graph_names, eager_names):
      self.assertIn(actual, {eager, graph})

  @combinations.generate(combinations.combine(mode=['eager']))
  def test_layer_names_after_loading(self):
    backend.clear_session()
    # Mimic loading a model that already contained add layers with
    # name = 'add_1' and 'tf.__operators__.add'
    layers.Add(name='add_1')
    layers.Add(name='tf.__operators__.add')

    inputs = input_layer.Input(shape=[2])
    add1 = inputs + inputs
    add2 = layers.Add()([inputs, inputs])
    add3 = inputs + inputs
    add4 = layers.Add()([inputs, inputs])
    model = training_lib.Model(
        inputs=[inputs], outputs=[add1, add2, add3, add4])
    actual_names = [l.name for l in model.layers]
    # The generated op layer names should have avoided layer names seen in
    # the loaded model. (This avoiance should not apply to non-op-layers)
    expected_names = [
        'input_1', 'tf.__operators__.add_1',
        'add', 'tf.__operators__.add_2', 'add_1'
    ]
    self.assertAllEqual(actual_names, expected_names)

  def test_add_trainable_weight_on_frozen_layer(self):

    class TestLayer(base_layer.Layer):

      def build(self, input_shape):
        self.w = self.add_weight(shape=(), trainable=True)

      def call(self, inputs):
        return self.w * inputs

    layer = TestLayer()
    layer.trainable = False
    layer.build(None)
    layer.trainable = True
    self.assertListEqual(layer.trainable_weights, [layer.w])

  @combinations.generate(
      combinations.times(combinations.keras_mode_combinations(),
                         combinations.keras_model_type_combinations()))
  def test_passing_initial_weights_values(self):
    kernel_value = np.random.random((10, 2))
    layer_with_weights = layers.Dense(2, use_bias=False, weights=[kernel_value])

    model = testing_utils.get_model_from_layers([layer_with_weights],
                                                input_shape=(10,))
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    inputs = np.random.random((3, 10))
    out = model.predict(inputs)
    self.assertAllClose(model.layers[-1].get_weights()[0], kernel_value)
    self.assertAllClose(out, np.dot(inputs, kernel_value))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_set_weights_and_get_weights(self):
    layer = layers.Dense(2)
    layer.build((None, 10))
    kernel = np.random.random((10, 2))
    bias = np.random.random((2,))
    layer.set_weights([kernel, bias])
    weights = layer.get_weights()
    self.assertEqual(len(weights), 2)
    self.assertAllClose(weights[0], kernel)
    self.assertAllClose(weights[1], bias)
    with self.assertRaisesRegex(ValueError,
                                'but the layer was expecting 2 weights'):
      layer.set_weights([1, 2, 3])
    with self.assertRaisesRegex(ValueError,
                                'not compatible with provided weight shape'):
      layer.set_weights([kernel.T, bias])

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_set_weights_accepts_output_of_get_weights(self):
    layer = layers.Layer()
    layer.add_weight(name='scalar_float', shape=(), dtype=tf.float32)
    layer.add_weight(name='scalar_string', shape=(), dtype=tf.string,
                     initializer=lambda *a, **k: 'abc')
    layer.add_weight(name='vector_float', shape=(3,), dtype=tf.float32)
    layer.add_weight(name='vector_string', shape=(2,), dtype=tf.string,
                     initializer=lambda *a, **k: 2 * ['abc'])
    layer.set_weights(layer.get_weights())

  def test_get_config_error(self):

    class MyLayer(base_layer.Layer):

      def __init__(self, my_kwarg='default', **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        self.my_kwarg = my_kwarg

    # `__init__` includes kwargs but `get_config` is not overridden, so
    # an error should be thrown:
    with self.assertRaisesRegex(NotImplementedError, 'Layer MyLayer has'):
      MyLayer('custom').get_config()

    class MyLayerNew(base_layer.Layer):

      def __init__(self, my_kwarg='default', **kwargs):
        super(MyLayerNew, self).__init__(**kwargs)
        self.my_kwarg = my_kwarg

      def get_config(self):
        config = super(MyLayerNew, self).get_config()
        config['my_kwarg'] = self.my_kwarg
        return config

    # Test to make sure that error is not raised if the method call is
    # from an overridden `get_config`:
    self.assertEqual(MyLayerNew('custom').get_config()['my_kwarg'], 'custom')

    class MyLayerNew2(base_layer.Layer):

      def __init__(self, name='MyLayerName', dtype=None, **kwargs):  # pylint:disable=redefined-outer-name
        super(MyLayerNew2, self).__init__(name=name, dtype=dtype, **kwargs)

    # Check that if the kwargs in `__init__` are base layer constructor
    # arguments, no error is thrown:
    self.assertEqual(MyLayerNew2(name='New').get_config()['name'], 'New')

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_count_params(self):
    dense = layers.Dense(16)
    dense.build((None, 4))
    self.assertEqual(dense.count_params(), 16 * 4 + 16)

    dense = layers.Dense(16)
    with self.assertRaisesRegex(ValueError, 'call `count_params`'):
      dense.count_params()

    model = sequential.Sequential(layers.Dense(16))
    with self.assertRaisesRegex(ValueError, 'call `count_params`'):
      model.count_params()

    dense = layers.Dense(16, input_dim=4)
    model = sequential.Sequential(dense)
    self.assertEqual(model.count_params(), 16 * 4 + 16)

  def test_super_not_called(self):

    class CustomLayerNotCallingSuper(base_layer.Layer):

      def __init__(self):
        pass

    layer = CustomLayerNotCallingSuper()
    with self.assertRaisesRegex(RuntimeError, 'You must call `super()'):
      layer(np.random.random((10, 2)))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_first_arg_not_called_inputs(self):
    x, y = tf.ones((10, 1)), tf.ones((10, 1))

    class ArgLayer(base_layer.Layer):

      def call(self, x, y):
        return x + y

    layer = ArgLayer()
    out = self.evaluate(layer(x=x, y=y))
    self.assertAllClose(out, 2 * np.ones((10, 1)))

    class KwargLayer(base_layer.Layer):

      def call(self, x=None, y=None):
        return x + y

    layer = KwargLayer()
    out = self.evaluate(layer(x=x, y=y))
    self.assertAllClose(out, 2 * np.ones((10, 1)))

    with self.assertRaisesRegex(ValueError, 'must always be passed'):
      layer(y=y)

    class TFFunctionLayer(base_layer.Layer):

      @tf.function
      def call(self, x, y=None):
        if y is None:
          return x
        return x + y

    layer = TFFunctionLayer()
    out = self.evaluate(layer(x=x, y=y))
    self.assertAllClose(out, 2 * np.ones((10, 1)))

  def test_build_input_shape(self):

    class CustomLayer(base_layer.Layer):

      def build(self, input_shape):
        self.add_weight('w', shape=input_shape[1:])
        super(CustomLayer, self).build(input_shape)

    layer = CustomLayer()
    self.assertFalse(layer.built)

    layer.build([None, 1, 2, 3])
    self.assertTrue(layer.built)
    self.assertEqual([None, 1, 2, 3], layer._build_input_shape)

    layer = CustomLayer()
    layer(input_layer.Input((3,)))
    self.assertTrue(layer.built)
    self.assertEqual([None, 3], layer._build_input_shape.as_list())

  @combinations.generate(combinations.combine(mode=['eager']))
  def test_custom_layer_training_arg(self):
    class CustomLayerNoTrainingArg(base_layer.Layer):

      def __init__(self, nested_layer=None):
        super(CustomLayerNoTrainingArg, self).__init__()
        self._nested_layer = nested_layer or tf.identity

      def call(self, inputs):
        return self._nested_layer(inputs)

    class CustomLayerDefaultTrainingMissing(base_layer.Layer):

      def __init__(self, nested_layer=None):
        super(CustomLayerDefaultTrainingMissing, self).__init__()
        self._nested_layer = nested_layer or tf.identity

      def call(self, inputs, training):
        if training:
          return self._nested_layer(inputs)
        else:
          return self._nested_layer(inputs) * 0.5

    class CustomLayerDefaultTrainingNone(base_layer.Layer):

      def __init__(self, nested_layer=None):
        super(CustomLayerDefaultTrainingNone, self).__init__()
        self._nested_layer = nested_layer or tf.identity

      def call(self, inputs, training=None):
        if training:
          return self._nested_layer(inputs)
        else:
          return self._nested_layer(inputs) * 0.5

    class CustomLayerDefaultTrainingFalse(base_layer.Layer):

      def __init__(self, nested_layer=None):
        super(CustomLayerDefaultTrainingFalse, self).__init__()
        self._nested_layer = nested_layer or tf.identity

      def call(self, inputs, training=False):
        if training:
          return self._nested_layer(inputs)
        else:
          return self._nested_layer(inputs) * 0.5

    class CustomLayerDefaultTrainingTrue(base_layer.Layer):

      def __init__(self, nested_layer=None):
        super(CustomLayerDefaultTrainingTrue, self).__init__()
        self._nested_layer = nested_layer or tf.identity

      def call(self, inputs, training=True):
        if training:
          return self._nested_layer(inputs)
        else:
          return self._nested_layer(inputs) * 0.5

    self._test_custom_layer_training_arg(
        CustomLayerNoTrainingArg=CustomLayerNoTrainingArg,
        CustomLayerDefaultTrainingMissing=CustomLayerDefaultTrainingMissing,
        CustomLayerDefaultTrainingNone=CustomLayerDefaultTrainingNone,
        CustomLayerDefaultTrainingFalse=CustomLayerDefaultTrainingFalse,
        CustomLayerDefaultTrainingTrue=CustomLayerDefaultTrainingTrue)

  @combinations.generate(combinations.combine(mode=['eager']))
  def test_custom_layer_training_arg_kwargonly(self):
    class CustomLayerNoTrainingArg(base_layer.Layer):

      def __init__(self, nested_layer=None):
        super(CustomLayerNoTrainingArg, self).__init__()
        self._nested_layer = nested_layer or tf.identity

      def call(self, inputs):
        return self._nested_layer(inputs)

    class CustomLayerDefaultTrainingMissing(base_layer.Layer):

      def __init__(self, nested_layer=None):
        super(CustomLayerDefaultTrainingMissing, self).__init__()
        self._nested_layer = nested_layer or tf.identity

      def call(self, inputs, *, training):
        if training:
          return self._nested_layer(inputs)
        else:
          return self._nested_layer(inputs) * 0.5

    class CustomLayerDefaultTrainingNone(base_layer.Layer):

      def __init__(self, nested_layer=None):
        super(CustomLayerDefaultTrainingNone, self).__init__()
        self._nested_layer = nested_layer or tf.identity

      def call(self, inputs, *, training=None):
        if training:
          return self._nested_layer(inputs)
        else:
          return self._nested_layer(inputs) * 0.5

    class CustomLayerDefaultTrainingFalse(base_layer.Layer):

      def __init__(self, nested_layer=None):
        super(CustomLayerDefaultTrainingFalse, self).__init__()
        self._nested_layer = nested_layer or tf.identity

      def call(self, inputs, *, training=False):
        if training:
          return self._nested_layer(inputs)
        else:
          return self._nested_layer(inputs) * 0.5

    class CustomLayerDefaultTrainingTrue(base_layer.Layer):

      def __init__(self, nested_layer=None):
        super(CustomLayerDefaultTrainingTrue, self).__init__()
        self._nested_layer = nested_layer or tf.identity

      def call(self, inputs, *, training=True):
        if training:
          return self._nested_layer(inputs)
        else:
          return self._nested_layer(inputs) * 0.5

    self._test_custom_layer_training_arg(
        CustomLayerNoTrainingArg=CustomLayerNoTrainingArg,
        CustomLayerDefaultTrainingMissing=CustomLayerDefaultTrainingMissing,
        CustomLayerDefaultTrainingNone=CustomLayerDefaultTrainingNone,
        CustomLayerDefaultTrainingFalse=CustomLayerDefaultTrainingFalse,
        CustomLayerDefaultTrainingTrue=CustomLayerDefaultTrainingTrue)

  def _test_custom_layer_training_arg(self,
                                      # pylint: disable=invalid-name
                                      CustomLayerNoTrainingArg,
                                      CustomLayerDefaultTrainingMissing,
                                      CustomLayerDefaultTrainingNone,
                                      CustomLayerDefaultTrainingFalse,
                                      CustomLayerDefaultTrainingTrue,
                                      # pylint: enable=invalid-name
                                      ):
    x = tf.ones(shape=(1, 1))

    # If the layer signature doesn't specify a default training arg,
    # run it in inference mode when to training arg is passed
    # to __call__
    layer = CustomLayerDefaultTrainingMissing()
    self.assertAllEqual(layer(x), x * 0.5)
    self.assertAllEqual(layer(x, training=False), x * 0.5)
    self.assertAllEqual(layer(x, training=True), x)

    # If the layer signature specifies `False` as the default training arg,
    # run it in inference mode when no training arg is passed
    # to __call__
    layer = CustomLayerDefaultTrainingFalse()
    self.assertAllEqual(layer(x), x * 0.5)
    self.assertAllEqual(layer(x, training=False), x * 0.5)
    self.assertAllEqual(layer(x, training=True), x)

    # If the layer signature specifies `True` as the default training arg,
    # explicitly run it in training mode when no training arg is passed
    # to __call__
    layer = CustomLayerDefaultTrainingTrue()
    self.assertAllEqual(layer(x), x)
    self.assertAllEqual(layer(x, training=False), x * 0.5)
    self.assertAllEqual(layer(x, training=True), x)

    # Outer layers/models should set the training context implicitly for all
    # nested layers, respecting whatever mode the outer layer was run with.
    layer = CustomLayerDefaultTrainingTrue(CustomLayerDefaultTrainingFalse())
    # No outer value passed: use local defaults
    self.assertAllEqual(layer(x), x)  # Use outer default True
    # Outer value passed: override local defaults
    self.assertAllEqual(layer(x, training=False), x * 0.25)
    self.assertAllEqual(layer(x, training=True), x)

    layer = CustomLayerDefaultTrainingFalse(CustomLayerDefaultTrainingTrue())
    # No outer value passed: use local defaults
    self.assertAllEqual(layer(x), x * 0.25)  # Use outer default False
    # Outer value passed: override local defaults
    self.assertAllEqual(layer(x, training=False), x * 0.25)
    self.assertAllEqual(layer(x, training=True), x)

    # If the outer layer `call` doesn't take a training argument at all,
    # it'll set the nested scope as None when no training arg is passed in.
    # If a training arg is passed in it won't use it directly in `call`, but
    # it will set the nested training mode.
    layer = CustomLayerNoTrainingArg(CustomLayerDefaultTrainingTrue())
    self.assertAllEqual(layer(x), x)  # Use local default True
    self.assertAllEqual(layer(x, training=False), x * 0.5)
    self.assertAllEqual(layer(x, training=True), x)

    layer = CustomLayerDefaultTrainingNone(CustomLayerDefaultTrainingTrue())
    self.assertAllEqual(layer(x), x * 0.5)  # Nested use local default True
    self.assertAllEqual(layer(x, training=False), x * 0.25)
    self.assertAllEqual(layer(x, training=True), x)

  def test_activity_regularizer_string(self):

    class MyLayer(base_layer.Layer):
      pass

    layer = MyLayer(activity_regularizer='l2')
    self.assertIsInstance(layer.activity_regularizer, regularizers.L2)

  def test_tf_module_tracking(self):

    class MyModule(tf.Module):

      def __init__(self):
        super(MyModule, self).__init__()
        self.v1 = tf.Variable(1., trainable=True, name='v1')
        self.v2 = tf.Variable(2., trainable=False, name='v2')

      def __call__(self, x):
        return x * self.v1 * self.v2

    class MyLayer(base_layer.Layer):

      def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        self.my_modules = {}
        self.my_modules['a'] = MyModule()

      def call(self, x):
        return self.my_modules['a'](x)

    layer = MyLayer()
    self.assertLen(layer.variables, 2)
    self.assertLen(layer.trainable_variables, 1)
    self.assertLen(layer.non_trainable_variables, 1)

    layer.trainable = False
    self.assertLen(layer.variables, 2)
    self.assertLen(layer.trainable_variables, 0)
    self.assertLen(layer.non_trainable_variables, 2)

    class MyModel(training_lib.Model):

      def __init__(self):
        super(MyModel, self).__init__()
        self.my_modules = []
        self.my_modules.append(MyModule())

      def call(self, x):
        return self.my_modules[0](x)

    model = MyModel()
    self.assertLen(model.variables, 2)
    self.assertLen(model.trainable_variables, 1)
    self.assertLen(model.non_trainable_variables, 1)

    model.trainable = False
    self.assertLen(model.variables, 2)
    self.assertLen(model.trainable_variables, 0)
    self.assertLen(model.non_trainable_variables, 2)


class SymbolicSupportTest(keras_parameterized.TestCase):

  def test_using_symbolic_tensors_with_tf_ops(self):
    # Single-input.
    x = input_layer.Input((3,))
    tf.square(x)

    # Multi-inputs.
    x1, x2 = input_layer.Input((3,)), input_layer.Input((3,))
    tf.concat([x1, x2], axis=1)

    # Mixing Keras symbolic tensors and graph tensors from the same graph works.
    with backend.get_graph().as_default():
      x1 = input_layer.Input((3,))
    x2 = input_layer.Input((3,))
    tf.matmul(x1, x2)

    # Creating same op type (matmul) multiple times in the Keras graph works.
    x1 = input_layer.Input((3,))
    x2 = input_layer.Input((3,))
    tf.matmul(x1, x2)

  def test_mixing_eager_and_graph_tensors(self):
    with tf.Graph().as_default():
      x1 = tf.ones((3, 3))
    x2 = tf.ones((3, 3))
    with self.assertRaises(TypeError):
      tf.matmul(x1, x2)

  def test_mixing_numpy_arrays_and_graph_tensors(self):
    with tf.Graph().as_default():
      x1 = tf.ones((3, 3))
    x2 = np.ones((3, 3), dtype='float32')
    with self.assertRaises(TypeError):
      tf.matmul(x1, x2)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_mixing_keras_symbolic_tensors_and_eager_tensors(self):
    x1 = input_layer.Input((3,))
    x2 = tf.ones((3, 3))
    y = tf.matmul(x1, x2)

    fn = backend.function(inputs=[x1], outputs=[y])
    x_val = np.random.random((3, 3))
    y_val = np.ones((3, 3))
    self.assertAllClose(fn([x_val])[0],
                        np.matmul(x_val, y_val),
                        atol=1e-5)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_mixing_keras_symbolic_tensors_and_numpy_arrays(self):
    x1 = input_layer.Input((3,))
    x2 = np.ones((3, 3), dtype='float32')
    y = tf.matmul(x1, x2)

    fn = backend.function(inputs=[x1], outputs=[y])
    x_val = np.random.random((3, 3))
    y_val = np.ones((3, 3))
    self.assertAllClose(fn([x_val])[0],
                        np.matmul(x_val, y_val),
                        atol=1e-5)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_reraising_exception(self):
    # When layer is not dynamic, we have some pattern matching during exception
    # handling to detect when the user is trying to use python control flow.
    # When an exception is thrown but the pattern doesn't match, we want to
    # preserve the originating stack trace. An early implementation of this
    # logic lost the stack trace. We test the correct behavior here.

    class TypeErrorLayer(base_layer.Layer):

      def call(self, inputs):
        def easily_identifiable_name():
          raise TypeError('Non-matching TypeError message.')
        easily_identifiable_name()

    inputs = input_layer.Input((3,))

    try:
      _ = TypeErrorLayer()(inputs)
    except TypeError as e:
      self.assertIn('easily_identifiable_name', str(e))  # pylint: disable=g-assert-in-except

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_summaries_in_tf_function(self):
    if not tf.executing_eagerly():
      return

    class MyLayer(base_layer.Layer):

      def call(self, inputs):
        tf.summary.scalar('mean', tf.reduce_mean(inputs))
        return inputs

    tmp_dir = self.get_temp_dir()
    writer = tf.summary.create_file_writer(tmp_dir)
    with writer.as_default(step=1), tf.summary.record_if(True):
      my_layer = MyLayer()
      x = tf.ones((10, 10))

      def my_fn(x):
        return my_layer(x)

      _ = my_fn(x)

    event_file = tf.compat.v1.gfile.Glob(os.path.join(tmp_dir, 'events*'))
    self.assertLen(event_file, 1)
    event_file = event_file[0]
    tags = set()
    for e in tf.compat.v1.train.summary_iterator(event_file):
      for val in e.summary.value:
        tags.add(val.tag)
    self.assertEqual(set(['my_layer/mean']), tags)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_error_when_passing_non_tensor(self):
    # layers that have an `input_spec` will raise an error when called on
    # non-tensors. This covers all built-in layers.
    layer = layers.Dense(3)
    x = object()
    with self.assertRaisesRegex(TypeError, r'should be tensors'):
      layer(x)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class NestedTrackingTest(tf.test.TestCase):

  def test_nested_layer_variable_tracking(self):
    # Test that variables from nested sublayers are
    # being tracked by subclassed layers.

    class MyLayer(base_layer.Layer):

      def __init__(self):
        super(MyLayer, self).__init__()
        self.dense1 = layers.Dense(1)
        self.dense2 = layers.BatchNormalization()

      def build(self, input_shape):
        self.v1 = self.add_weight('v1', shape=input_shape[1:].as_list())
        self.v2 = tf.Variable(
            name='v2',
            initial_value=np.zeros(input_shape[1:].as_list(), dtype='float32'),
            trainable=False)

      def call(self, inputs):
        x = self.dense1(inputs) + self.dense2(inputs)
        return x + self.v1 + self.v2

    layer = MyLayer()
    inputs = input_layer.Input((1,))
    _ = layer(inputs)

    self.assertEqual(len(layer.weights), 8)
    self.assertEqual(len(layer.trainable_weights), 5)
    self.assertEqual(len(layer.non_trainable_weights), 3)

    layer.dense1.trainable = False
    self.assertEqual(len(layer.weights), 8)
    self.assertEqual(len(layer.trainable_weights), 3)
    self.assertEqual(len(layer.non_trainable_weights), 5)

    layer.trainable = False
    self.assertEqual(len(layer.weights), 8)
    self.assertEqual(len(layer.trainable_weights), 0)
    self.assertEqual(len(layer.non_trainable_weights), 8)
    self.assertEqual(
        {id(v) for v in [layer.dense1, layer.dense2, layer.v1, layer.v2]},
        {id(v) for _, v in layer._checkpoint_dependencies})

  def test_nested_layer_updates_losses_tracking(self):
    # Test that updates and losses from nested sublayers are
    # being tracked by subclassed layers.

    class UpdateAndLossLayer(base_layer.Layer):

      def build(self, _):
        self.v1 = self.add_weight('v1', shape=())

      def call(self, inputs):
        self.add_loss(tf.reduce_sum(inputs))
        self.add_update(tf.compat.v1.assign_add(self.v1, 1))
        return inputs + 1

    class MyLayer(base_layer.Layer):

      def build(self, _):
        self.v1 = self.add_weight('v1', shape=())

      def __init__(self):
        super(MyLayer, self).__init__()
        self.ul1 = UpdateAndLossLayer()
        self.ul2 = UpdateAndLossLayer()

      def call(self, inputs):
        self.add_loss(tf.reduce_sum(inputs))
        self.add_update(tf.compat.v1.assign_add(self.v1, 1))
        x = self.ul1(inputs)
        return self.ul2(x)

    layer = MyLayer()

    if tf.executing_eagerly():
      inputs = tf.ones((3, 1))
      _ = layer(inputs)
      self.assertEqual(len(layer.losses), 3)
      self.assertLen(layer.get_losses_for(None), 3)
    else:
      inputs = input_layer.Input((1,))
      _ = layer(inputs)
      self.assertEqual(len(layer.losses), 3)
      self.assertEqual(len(layer.updates), 3)
      self.assertLen(layer.get_losses_for(None), 3)

  def test_attribute_reassignment(self):
    l = base_layer.Layer()
    l.a = base_layer.Layer()
    l.a = []
    l.a = tf.Variable(1.)
    l.a = base_layer.Layer()
    last_assignment = base_layer.Layer()
    l.a = last_assignment
    l.b = tf.Variable(1.)
    del l.b
    l.c = base_layer.Layer()
    del l.c
    l.d = last_assignment
    del l.d
    sublayers = list(l._flatten_layers(include_self=False, recursive=False))
    self.assertEqual([last_assignment], sublayers)
    self.assertEqual([], l.trainable_weights)
    self.assertEqual([], l.non_trainable_weights)
    self.assertEqual([], l.weights)
    del l.a
    self.assertEqual([], l._self_tracked_trackables)

  def test_layer_class_not_tracked_as_sublayer(self):
    # See https://github.com/tensorflow/tensorflow/issues/27431 for details.

    class LayerWithClassAttribute(base_layer.Layer):

      def __init__(self):
        super(LayerWithClassAttribute, self).__init__()
        self.layer_fn = layers.Dense

    layer = LayerWithClassAttribute()
    self.assertEmpty(layer.variables)
    self.assertEmpty(layer.submodules)

  def test_layer_call_fn_args(self):

    class NonDefunLayer(base_layer.Layer):

      def call(self, inputs, a, mask, b=None, training=None):
        return inputs

    class DefunLayer(base_layer.Layer):

      @tf.function
      def call(self, x, mask, a, training=None, b=None):
        return x

    nondefun_layer = NonDefunLayer()
    self.assertEqual(nondefun_layer._call_fn_args,
                     ['inputs', 'a', 'mask', 'b', 'training'])
    defun_layer = DefunLayer()
    self.assertEqual(defun_layer._call_fn_args,
                     ['x', 'mask', 'a', 'training', 'b'])

  def test_sequential_model(self):
    model = sequential.Sequential(
        [layers.Dense(10, input_shape=(10,)),
         layers.Dense(5)])
    self.assertLen(model.layers, 2)
    self.assertLen(model.weights, 4)

    # Make sure a subclass model also works when it is called 'Sequential'.
    class Sequential(training_lib.Model):

      def __init__(self):
        super(Sequential, self).__init__()
        self.dense_layers = [layers.Dense(10), layers.Dense(5)]

      def call(self, inputs):
        x = inputs
        for d in self.dense_layers:
          x = d(x)
        return x

    s = Sequential()
    self.assertLen(s.layers, 2)
    self.assertLen(s.weights, 0)

    s(input_layer.Input((10,)))
    self.assertLen(s.weights, 4)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class NameScopingTest(keras_parameterized.TestCase):

  def test_name_scope_layer(self):
    x = backend.placeholder(shape=(10, 10))
    layer = layers.Dense(10, name='MyName')
    layer(x)
    self.assertEqual(layer.bias.name, 'MyName/bias:0')
    self.assertEqual(layer.kernel.name, 'MyName/kernel:0')

  def test_name_scope_functional_api(self):
    inputs = input_layer.Input((3,))
    layer = layers.Dense(10, name='MyName')
    _ = layer(inputs)
    self.assertEqual(layer.bias.name, 'MyName/bias:0')
    self.assertEqual(layer.kernel.name, 'MyName/kernel:0')

  def test_name_scope_functional_api_nested(self):

    class NestedLayer(base_layer.Layer):

      def __init__(self, name='OuterName'):
        super(NestedLayer, self).__init__(name=name)
        self.dense = layers.Dense(10, name='InnerName')

      def call(self, inputs):
        return self.dense(inputs)

    inputs = input_layer.Input((3,))
    layer = NestedLayer()
    _ = layer(inputs)
    self.assertEqual(layer.dense.bias.name, 'OuterName/InnerName/bias:0')
    self.assertEqual(layer.dense.kernel.name, 'OuterName/InnerName/kernel:0')

  def test_name_scope_sublayer(self):

    class NameScopeTracker(base_layer.Layer):

      def call(self, inputs):
        self.active_name_scope = tf.__internal__.get_name_scope()
        return inputs

    x = backend.placeholder(shape=(10, 10))
    sublayer = NameScopeTracker(name='Sublayer')
    layer = layers.Dense(10, activation=sublayer, name='MyName2')
    layer(x)
    self.assertEqual(layer.bias.name, 'MyName2/bias:0')
    self.assertEqual(layer.kernel.name, 'MyName2/kernel:0')
    self.assertEqual(sublayer.active_name_scope, 'MyName2/Sublayer')

  def test_name_scope_tf_tensor(self):
    x = tf.convert_to_tensor(np.ones((10, 10)))
    layer = layers.Dense(
        10, activation=layers.ReLU(name='MyAct'), name='MyName3')
    layer(x)
    self.assertEqual(layer.bias.name, 'MyName3/bias:0')
    self.assertEqual(layer.kernel.name, 'MyName3/kernel:0')

  @testing_utils.run_v2_only
  def test_apply_name_scope_on_model_declaration(self):
    if not tf.executing_eagerly():
      self.skipTest('`apply_name_scope_on_model_declaration` API is supported'
                    ' only for V2 eager')

    base_layer._apply_name_scope_on_model_declaration(True)

    inputs = input_layer.Input((3,))
    x = layers.Dense(10, name='Dense1')(inputs)
    with tf.name_scope('outer'):
      x = layers.Dense(10, name='Dense2')(x)
      with tf.name_scope('inner'):
        x = layers.Dense(10, name='Dense3')(x)
      x = layers.Dense(10, name='Dense4')(x)
    outputs = layers.Dense(10, name='Dense5')(x)

    model = training_lib.Model(inputs, outputs)
    node_names = self._get_model_node_names(model, np.random.random((1, 3)),
                                            'call_scope')
    self.assertListEqual(node_names, [
        'call_scope/Const',
        'call_scope/model/Cast',
        'call_scope/model/Dense1/MatMul/ReadVariableOp/resource',
        'call_scope/model/Dense1/MatMul/ReadVariableOp',
        'call_scope/model/Dense1/MatMul',
        'call_scope/model/Dense1/BiasAdd/ReadVariableOp/resource',
        'call_scope/model/Dense1/BiasAdd/ReadVariableOp',
        'call_scope/model/Dense1/BiasAdd',
        'call_scope/model/outer/Dense2/MatMul/ReadVariableOp/resource',
        'call_scope/model/outer/Dense2/MatMul/ReadVariableOp',
        'call_scope/model/outer/Dense2/MatMul',
        'call_scope/model/outer/Dense2/BiasAdd/ReadVariableOp/resource',
        'call_scope/model/outer/Dense2/BiasAdd/ReadVariableOp',
        'call_scope/model/outer/Dense2/BiasAdd',
        'call_scope/model/outer/inner/Dense3/MatMul/ReadVariableOp/resource',
        'call_scope/model/outer/inner/Dense3/MatMul/ReadVariableOp',
        'call_scope/model/outer/inner/Dense3/MatMul',
        'call_scope/model/outer/inner/Dense3/BiasAdd/ReadVariableOp/resource',
        'call_scope/model/outer/inner/Dense3/BiasAdd/ReadVariableOp',
        'call_scope/model/outer/inner/Dense3/BiasAdd',
        'call_scope/model/outer/Dense4/MatMul/ReadVariableOp/resource',
        'call_scope/model/outer/Dense4/MatMul/ReadVariableOp',
        'call_scope/model/outer/Dense4/MatMul',
        'call_scope/model/outer/Dense4/BiasAdd/ReadVariableOp/resource',
        'call_scope/model/outer/Dense4/BiasAdd/ReadVariableOp',
        'call_scope/model/outer/Dense4/BiasAdd',
        'call_scope/model/Dense5/MatMul/ReadVariableOp/resource',
        'call_scope/model/Dense5/MatMul/ReadVariableOp',
        'call_scope/model/Dense5/MatMul',
        'call_scope/model/Dense5/BiasAdd/ReadVariableOp/resource',
        'call_scope/model/Dense5/BiasAdd/ReadVariableOp',
        'call_scope/model/Dense5/BiasAdd',
        'Identity',
        'NoOp'
    ])
    base_layer._apply_name_scope_on_model_declaration(False)

  def _get_model_node_names(self, model, inputs, call_name_scope):
    """Returns a list of model's node names."""

    @tf.function()
    def wrapper():
      with tf.name_scope(call_name_scope):
        return model(inputs)

    return [
        node.name
        for node in wrapper.get_concrete_function().graph.as_graph_def().node
    ]


@combinations.generate(combinations.keras_mode_combinations(mode=['eager']))
class AutographControlFlowTest(keras_parameterized.TestCase):

  def test_disabling_in_context_is_matched(self):

    test_obj = self

    class MyLayer(base_layer.Layer):

      def call(self, inputs, training=None):
        with test_obj.assertRaisesRegex(TypeError, 'Tensor.*as.*bool'):
          if tf.constant(False):
            return inputs * 1.
        return inputs * 0.

    @tf.function(autograph=False)
    def test_fn():
      return MyLayer()(tf.constant([[1., 2., 3.]]))

    test_fn()

  def test_if_training_pattern_output(self):

    class MyLayer(base_layer.Layer):

      def call(self, inputs, training=None):
        if training:
          return inputs * 1.
        return inputs * 0.

    inputs = input_layer.Input((3,))
    outputs = MyLayer()(inputs)
    model = training_lib.Model(inputs, outputs)
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    train_loss = model.train_on_batch(np.ones((2, 3)), np.ones((2, 3)))
    self.assertEqual(train_loss, 0.)
    test_loss = model.test_on_batch(np.ones((2, 3)), np.ones((2, 3)))
    self.assertEqual(test_loss, 1.)

  def test_if_training_pattern_loss(self):

    class MyLayer(base_layer.Layer):

      def call(self, inputs, training=None):
        if training:
          loss = tf.reduce_sum(inputs)
        else:
          loss = 0.
        self.add_loss(loss)
        return inputs

    inputs = input_layer.Input((3,))
    outputs = MyLayer()(inputs)
    model = training_lib.Model(inputs, outputs)
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    train_loss = model.train_on_batch(np.ones((2, 3)), np.ones((2, 3)))
    self.assertEqual(train_loss, 2 * 3)
    test_loss = model.test_on_batch(np.ones((2, 3)), np.ones((2, 3)))
    self.assertEqual(test_loss, 0)

  def test_if_training_pattern_metric(self):

    class MyLayer(base_layer.Layer):

      def call(self, inputs, training=None):
        if training:
          metric = tf.reduce_sum(inputs)
        else:
          metric = 0.
        self.add_metric(metric, name='my_metric', aggregation='mean')
        return inputs

    inputs = input_layer.Input((3,))
    outputs = MyLayer()(inputs)
    model = training_lib.Model(inputs, outputs)
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    for _ in range(3):
      _, train_metric = model.train_on_batch(np.ones((2, 3)),
                                             np.ones((2, 3)))

      self.assertEqual(train_metric, 2 * 3)
      _, test_metric = model.test_on_batch(np.ones((2, 3)),
                                           np.ones((2, 3)))
      self.assertEqual(test_metric, 0)

  def test_if_training_pattern_update(self):

    class MyLayer(base_layer.Layer):

      def build(self, input_shape):
        self.counter = self.add_weight(
            shape=(), trainable=False, initializer='zeros')

      def call(self, inputs, training=None):
        if training:
          increment = 1.
        else:
          increment = 0.
        self.counter.assign_add(increment)
        return inputs

    inputs = input_layer.Input((3,))
    layer = MyLayer()
    outputs = layer(inputs)
    model = training_lib.Model(inputs, outputs)
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    model.train_on_batch(np.ones((2, 3)), np.ones((2, 3)))
    self.assertEqual(backend.get_value(layer.counter), 1.)

  def test_conditional_losses_in_call(self):

    class MyLayer(base_layer.Layer):

      def __init__(self):
        super(MyLayer,
              self).__init__(dynamic=testing_utils.should_run_eagerly())

      def call(self, inputs, training=None):
        if training:
          self.add_loss(tf.reduce_sum(inputs))
        return inputs

      def compute_output_shape(self, input_shape):
        return input_shape

    inputs = input_layer.Input((3,))
    layer = MyLayer()
    outputs = layer(inputs)
    model = training_lib.Model(inputs, outputs)
    model.compile('sgd', 'mse', run_eagerly=testing_utils.should_run_eagerly())
    loss = model.train_on_batch(np.ones((2, 3)), np.ones((2, 3)))
    self.assertEqual(loss, 2 * 3)

  def test_conditional_callable_losses(self):
    model = sequential.Sequential([
        layers.Dense(
            1, kernel_regularizer=regularizers.l2(1e-4), input_shape=(1,))
    ])
    model._run_eagerly = testing_utils.should_run_eagerly()

    def assert_graph(t):
      if not tf.executing_eagerly():
        self.assertEqual(t.graph, tf.compat.v1.get_default_graph())

    @tf.function
    def get_losses(t):
      if t < 0:
        return tf.reduce_sum(model.losses) * t
      else:
        return tf.reduce_sum(model.losses)

    assert_graph(get_losses(tf.constant(2.)))
    assert_graph(get_losses(tf.constant(0.5)))

  def test_conditional_metrics_in_call(self):

    class MyLayer(base_layer.Layer):

      def __init__(self):
        super(MyLayer,
              self).__init__(dynamic=testing_utils.should_run_eagerly())

      def call(self, inputs, training=None):
        if training:
          self.add_metric(tf.reduce_sum(inputs),
                          name='sum',
                          aggregation='mean')
        return inputs

      def compute_output_shape(self, input_shape):
        return input_shape

    inputs = input_layer.Input((3,))
    layer = MyLayer()
    outputs = layer(inputs)
    model = training_lib.Model(inputs, outputs)
    model.compile('sgd', 'mse', run_eagerly=testing_utils.should_run_eagerly())
    history = model.fit(np.ones((2, 3)), np.ones((2, 3)))
    self.assertEqual(history.history['sum'][-1], 2 * 3)

  def test_conditional_activity_regularizer_in_call(self):

    class TestModel(training_lib.Model):

      def __init__(self):
        super(TestModel, self).__init__(
            name='test_model', dynamic=testing_utils.should_run_eagerly())
        self.layer = layers.Dense(2, activity_regularizer='l2')

      def call(self, x, training=None):
        if tf.greater(tf.reduce_sum(x), 0.0):
          return self.layer(x)
        else:
          return self.layer(x)

    model = TestModel()
    model.compile(
        loss='mse',
        optimizer='sgd',
        run_eagerly=testing_utils.should_run_eagerly())

    x = np.ones(shape=(10, 1))
    y = np.ones(shape=(10, 2))

    if testing_utils.should_run_eagerly():
      model.fit(x, y, epochs=2, batch_size=5)
    else:
      with self.assertRaisesRegex(ValueError, 'ActivityRegularizer'):
        model.fit(x, y, epochs=2, batch_size=5)

  def test_conditional_activity_regularizer_with_wrappers_in_call(self):

    class TestModel(training_lib.Model):

      def __init__(self):
        super(TestModel, self).__init__(
            name='test_model', dynamic=testing_utils.should_run_eagerly())
        self.layer = layers.TimeDistributed(
            layers.Dense(2, activity_regularizer='l2'), input_shape=(3, 4))

      def call(self, x, training=None):
        if tf.greater(tf.reduce_sum(x), 0.0):
          return self.layer(x)
        else:
          return self.layer(x)

    model = TestModel()
    model.compile(
        loss='mse',
        optimizer='sgd',
        run_eagerly=testing_utils.should_run_eagerly())

    x = np.ones(shape=(10, 3, 4))
    y = np.ones(shape=(10, 3, 2))

    if testing_utils.should_run_eagerly():
      model.fit(x, y, epochs=2, batch_size=5)
    else:
      with self.assertRaisesRegex(ValueError, 'ActivityRegularizer'):
        model.fit(x, y, epochs=2, batch_size=5)


class AddLayer(base_layer.Layer):
  """A layer which adds its input to a variable.

  Useful for testing a layer with a variable
  """

  def build(self, _):
    self.v = self.add_weight('v', (), initializer='ones')
    self.built = True

  def call(self, inputs):
    return inputs + self.v


class IdentityLayer(base_layer.Layer):
  """A layer that returns its input.

  Useful for testing a layer without a variable.
  """

  def call(self, inputs):
    return inputs


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class DTypeTest(keras_parameterized.TestCase):

  def _const(self, dtype):
    return tf.constant(1, dtype=dtype)

  @testing_utils.enable_v2_dtype_behavior
  def test_dtype_defaults_to_floatx(self):
    layer = AddLayer()
    self.assertEqual(layer.dtype, 'float32')
    layer(self._const('float64'))
    self.assertEqual(layer.dtype, 'float32')  # dtype should not change

    try:
      backend.set_floatx('float64')
      layer = AddLayer()
      self.assertEqual(layer.dtype, 'float64')
    finally:
      backend.set_floatx('float32')

  @testing_utils.enable_v2_dtype_behavior
  def test_passing_dtype_to_constructor(self):
    layer = IdentityLayer(dtype='float64')
    layer(self._const('float32'))
    self.assertEqual(layer.dtype, 'float64')

    layer = IdentityLayer(dtype='int32')
    layer(self._const('float32'))
    self.assertEqual(layer.dtype, 'int32')

    layer = IdentityLayer(dtype=tf.float64)
    layer(self._const('float32'))
    self.assertEqual(layer.dtype, 'float64')

  @testing_utils.enable_v2_dtype_behavior
  def input_cast_to_dtype(self):
    layer = AddLayer()

    # Input should be cast to layer.dtype, so output should also be layer.dtype
    self.assertEqual(layer(self._const('float64')).dtype, 'float32')

    layer = AddLayer(dtype='float64')
    self.assertEqual(layer(self._const('float32')).dtype, 'float64')

    # Test inputs are not casted if layer.dtype is not floating-point
    layer = IdentityLayer(dtype='int32')
    self.assertEqual(layer(self._const('float64')).dtype, 'float64')

    # Test inputs are not casted if the inputs are not floating-point
    layer = IdentityLayer(dtype='float32')
    self.assertEqual(layer(self._const('int32')).dtype, 'int32')

    # Test Numpy arrays are casted
    layer = IdentityLayer(dtype='float64')
    self.assertEqual(layer(np.array(1, dtype='float32')).dtype, 'float64')

    # Test Python floats are casted
    layer = IdentityLayer(dtype='float64')
    self.assertEqual(layer(1.).dtype, 'float64')

  @testing_utils.enable_v2_dtype_behavior
  def multiple_inputs_cast_to_dtype(self):

    class MultiIdentityLayer(base_layer.Layer):

      def call(self, inputs):
        return [tf.identity(x) for x in inputs]

    # Testing layer with default dtype of float32
    layer = MultiIdentityLayer()
    x, y = layer([self._const('float16'), self._const('float32')])
    self.assertEqual(x.dtype, 'float32')
    self.assertEqual(y.dtype, 'float32')

    # Test passing dtype to the constructor
    layer = MultiIdentityLayer(dtype='float64')
    x, y = layer([self._const('float16'), self._const('float32')])
    self.assertEqual(x.dtype, 'float64')
    self.assertEqual(y.dtype, 'float64')

    # Test several non-floating point types
    layer = MultiIdentityLayer(dtype='float64')
    x, y, z, w = layer([self._const('float16'), self._const('bool'),
                        self._const('float64'), self._constant('complex64')])
    self.assertEqual(x.dtype, 'float64')
    self.assertEqual(y.dtype, 'bool')
    self.assertEqual(z.dtype, 'float64')
    self.assertEqual(w.dtype, 'complex64')

  @testing_utils.enable_v2_dtype_behavior
  def test_extra_args_and_kwargs_not_casted(self):

    class IdentityLayerWithArgs(base_layer.Layer):

      def call(self, inputs, *args, **kwargs):
        kwargs.pop('training', None)
        return tf.nest.flatten([inputs, args, kwargs])

    layer = IdentityLayerWithArgs(dtype='float64')
    x, y, z = layer(self._const('float16'), self._const('float16'),
                    kwarg=self._const('float16'))
    self.assertEqual(x.dtype, 'float64')
    self.assertEqual(y.dtype, 'float16')
    self.assertEqual(z.dtype, 'float16')

  @testing_utils.enable_v2_dtype_behavior
  def test_layer_without_autocast(self):

    class IdentityLayerWithoutAutocast(IdentityLayer):

      def __init__(self, *args, **kwargs):
        kwargs['autocast'] = False
        super(IdentityLayerWithoutAutocast, self).__init__(*args, **kwargs)

    layer = IdentityLayerWithoutAutocast(dtype='float64')
    self.assertEqual(layer(self._const('float32')).dtype, 'float32')

  @testing_utils.enable_v2_dtype_behavior
  def test_compute_output_signature(self):

    class IdentityLayerWithOutputShape(IdentityLayer):

      def compute_output_shape(self, input_shape):
        return input_shape

    layer = IdentityLayerWithOutputShape(dtype='float64')
    output_signature = layer.compute_output_signature(
        tf.TensorSpec(shape=(), dtype='float32'))
    self.assertEqual(output_signature.shape, ())
    self.assertEqual(output_signature.dtype, 'float64')

  @testing_utils.enable_v2_dtype_behavior
  def test_composite_tensors_input_casting(self):
    sparse = tf.SparseTensor(
        indices=tf.constant([[0, 1], [2, 3]], dtype='int64'),
        values=tf.constant([0., 1.], dtype='float32'),
        dense_shape=tf.constant([4, 4], dtype='int64'))
    ragged = tf.RaggedTensor.from_row_splits(
        values=tf.constant([1., 2., 3.], dtype='float32'),
        row_splits=tf.constant([0, 2, 2, 3], dtype='int64'))

    layer = IdentityLayer(dtype='float16')

    for x in sparse, ragged:
      self.assertEqual(x.dtype, 'float32')
      y = layer(x)
      self.assertEqual(y.dtype, 'float16')
      self.assertEqual(type(x), type(y))

  @testing_utils.enable_v2_dtype_behavior
  def test_passing_non_tensor(self):
    layer = IdentityLayer()
    x = object()
    y = layer(x)  # Layer should not cast 'x', as it's not a tensor
    self.assertIs(x, y)

  @testing_utils.disable_v2_dtype_behavior
  def test_v1_behavior(self):
    # Test dtype defaults to None and inferred from input
    layer = IdentityLayer()
    self.assertIsNone(layer.dtype)
    layer(self._const('float64'))
    self.assertEqual(layer.dtype, 'float64')

    # Test layer does not cast to dtype
    self.assertEqual(layer(self._const('float32')).dtype, 'float32')


if __name__ == '__main__':
  tf.compat.v1.enable_eager_execution()
  tf.test.main()
