# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for unit-testing Keras."""
# pylint: disable=g-bad-import-order

import tensorflow.compat.v2 as tf

import collections
import functools
import itertools
import unittest

from absl.testing import parameterized

import keras
from keras.testing_infra import test_utils

try:
  import h5py  # pylint:disable=g-import-not-at-top
except ImportError:
  h5py = None

KERAS_MODEL_TYPES = ['functional', 'subclass', 'sequential']


class TestCase(tf.test.TestCase, parameterized.TestCase):

  def tearDown(self):
    keras.backend.clear_session()
    super(TestCase, self).tearDown()


def run_with_all_saved_model_formats(
    test_or_class=None,
    exclude_formats=None):
  """Execute the decorated test with all Keras saved model formats).

  This decorator is intended to be applied either to individual test methods in
  a `test_combinations.TestCase` class, or directly to a test class that
  extends it. Doing so will cause the contents of the individual test
  method (or all test methods in the class) to be executed multiple times - once
  for each Keras saved model format.

  The Keras saved model formats include:
  1. HDF5: 'h5'
  2. SavedModel: 'tf'

  Note: if stacking this decorator with absl.testing's parameterized decorators,
  those should be at the bottom of the stack.

  Various methods in `testing_utils` to get file path for saved models will
  auto-generate a string of the two saved model formats. This allows unittests
  to confirm the equivalence between the two Keras saved model formats.

  For example, consider the following unittest:

  ```python
  class MyTests(test_utils.KerasTestCase):

    @test_utils.run_with_all_saved_model_formats
    def test_foo(self):
      save_format = test_utils.get_save_format()
      saved_model_dir = '/tmp/saved_model/'
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_shape=(3,)))
      model.add(keras.layers.Dense(3))
      model.compile(loss='mse', optimizer='sgd', metrics=['acc'])

      keras.models.save_model(model, saved_model_dir, save_format=save_format)
      model = keras.models.load_model(saved_model_dir)

  if __name__ == "__main__":
    tf.test.main()
  ```

  This test tries to save the model into the formats of 'hdf5', 'h5', 'keras',
  'tensorflow', and 'tf'.

  We can also annotate the whole class if we want this to apply to all tests in
  the class:
  ```python
  @test_utils.run_with_all_saved_model_formats
  class MyTests(test_utils.KerasTestCase):

    def test_foo(self):
      save_format = test_utils.get_save_format()
      saved_model_dir = '/tmp/saved_model/'
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_shape=(3,)))
      model.add(keras.layers.Dense(3))
      model.compile(loss='mse', optimizer='sgd', metrics=['acc'])

      keras.models.save_model(model, saved_model_dir, save_format=save_format)
      model = tf.keras.models.load_model(saved_model_dir)

  if __name__ == "__main__":
    tf.test.main()
  ```

  Args:
    test_or_class: test method or class to be annotated. If None,
      this method returns a decorator that can be applied to a test method or
      test class. If it is not None this returns the decorator applied to the
      test or class.
    exclude_formats: A collection of Keras saved model formats to not run.
      (May also be a single format not wrapped in a collection).
      Defaults to None.

  Returns:
    Returns a decorator that will run the decorated test method multiple times:
    once for each desired Keras saved model format.

  Raises:
    ImportError: If abseil parameterized is not installed or not included as
      a target dependency.
  """
  # Exclude h5 save format if H5py isn't available.
  if h5py is None:
    exclude_formats.append(['h5'])
  saved_model_formats = ['h5', 'tf', 'tf_no_traces']
  params = [('_%s' % saved_format, saved_format)
            for saved_format in saved_model_formats
            if saved_format not in tf.nest.flatten(exclude_formats)]

  def single_method_decorator(f):
    """Decorator that constructs the test cases."""
    # Use named_parameters so it can be individually run from the command line
    @parameterized.named_parameters(*params)
    @functools.wraps(f)
    def decorated(self, saved_format, *args, **kwargs):
      """A run of a single test case w/ the specified model type."""
      if saved_format == 'h5':
        _test_h5_saved_model_format(f, self, *args, **kwargs)
      elif saved_format == 'tf':
        _test_tf_saved_model_format(f, self, *args, **kwargs)
      elif saved_format == 'tf_no_traces':
        _test_tf_saved_model_format_no_traces(f, self, *args, **kwargs)
      else:
        raise ValueError('Unknown model type: %s' % (saved_format,))
    return decorated

  return _test_or_class_decorator(test_or_class, single_method_decorator)


def _test_h5_saved_model_format(f, test_or_class, *args, **kwargs):
  with test_utils.saved_model_format_scope('h5'):
    f(test_or_class, *args, **kwargs)


def _test_tf_saved_model_format(f, test_or_class, *args, **kwargs):
  with test_utils.saved_model_format_scope('tf'):
    f(test_or_class, *args, **kwargs)


def _test_tf_saved_model_format_no_traces(f, test_or_class, *args, **kwargs):
  with test_utils.saved_model_format_scope('tf', save_traces=False):
    f(test_or_class, *args, **kwargs)


def run_with_all_weight_formats(test_or_class=None, exclude_formats=None):
  """Runs all tests with the supported formats for saving weights."""
  exclude_formats = exclude_formats or []
  exclude_formats.append('tf_no_traces')  # Only applies to saving models
  return run_with_all_saved_model_formats(test_or_class, exclude_formats)


# TODO(kaftan): Possibly enable 'subclass_custom_build' when tests begin to pass
# it. Or perhaps make 'subclass' always use a custom build method.
def run_with_all_model_types(
    test_or_class=None,
    exclude_models=None):
  """Execute the decorated test with all Keras model types.

  This decorator is intended to be applied either to individual test methods in
  a `test_combinations.TestCase` class, or directly to a test class that
  extends it. Doing so will cause the contents of the individual test
  method (or all test methods in the class) to be executed multiple times - once
  for each Keras model type.

  The Keras model types are: ['functional', 'subclass', 'sequential']

  Note: if stacking this decorator with absl.testing's parameterized decorators,
  those should be at the bottom of the stack.

  Various methods in `testing_utils` to get models will auto-generate a model
  of the currently active Keras model type. This allows unittests to confirm
  the equivalence between different Keras models.

  For example, consider the following unittest:

  ```python
  class MyTests(test_utils.KerasTestCase):

    @test_utils.run_with_all_model_types(
      exclude_models = ['sequential'])
    def test_foo(self):
      model = test_utils.get_small_mlp(1, 4, input_dim=3)
      optimizer = RMSPropOptimizer(learning_rate=0.001)
      loss = 'mse'
      metrics = ['mae']
      model.compile(optimizer, loss, metrics=metrics)

      inputs = np.zeros((10, 3))
      targets = np.zeros((10, 4))
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(10)

      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)

  if __name__ == "__main__":
    tf.test.main()
  ```

  This test tries building a small mlp as both a functional model and as a
  subclass model.

  We can also annotate the whole class if we want this to apply to all tests in
  the class:
  ```python
  @test_utils.run_with_all_model_types(exclude_models = ['sequential'])
  class MyTests(test_utils.KerasTestCase):

    def test_foo(self):
      model = test_utils.get_small_mlp(1, 4, input_dim=3)
      optimizer = RMSPropOptimizer(learning_rate=0.001)
      loss = 'mse'
      metrics = ['mae']
      model.compile(optimizer, loss, metrics=metrics)

      inputs = np.zeros((10, 3))
      targets = np.zeros((10, 4))
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(10)

      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)

  if __name__ == "__main__":
    tf.test.main()
  ```


  Args:
    test_or_class: test method or class to be annotated. If None,
      this method returns a decorator that can be applied to a test method or
      test class. If it is not None this returns the decorator applied to the
      test or class.
    exclude_models: A collection of Keras model types to not run.
      (May also be a single model type not wrapped in a collection).
      Defaults to None.

  Returns:
    Returns a decorator that will run the decorated test method multiple times:
    once for each desired Keras model type.

  Raises:
    ImportError: If abseil parameterized is not installed or not included as
      a target dependency.
  """
  model_types = ['functional', 'subclass', 'sequential']
  params = [('_%s' % model, model) for model in model_types
            if model not in tf.nest.flatten(exclude_models)]

  def single_method_decorator(f):
    """Decorator that constructs the test cases."""
    # Use named_parameters so it can be individually run from the command line
    @parameterized.named_parameters(*params)
    @functools.wraps(f)
    def decorated(self, model_type, *args, **kwargs):
      """A run of a single test case w/ the specified model type."""
      if model_type == 'functional':
        _test_functional_model_type(f, self, *args, **kwargs)
      elif model_type == 'subclass':
        _test_subclass_model_type(f, self, *args, **kwargs)
      elif model_type == 'sequential':
        _test_sequential_model_type(f, self, *args, **kwargs)
      else:
        raise ValueError('Unknown model type: %s' % (model_type,))
    return decorated

  return _test_or_class_decorator(test_or_class, single_method_decorator)


def _test_functional_model_type(f, test_or_class, *args, **kwargs):
  with test_utils.model_type_scope('functional'):
    f(test_or_class, *args, **kwargs)


def _test_subclass_model_type(f, test_or_class, *args, **kwargs):
  with test_utils.model_type_scope('subclass'):
    f(test_or_class, *args, **kwargs)


def _test_sequential_model_type(f, test_or_class, *args, **kwargs):
  with test_utils.model_type_scope('sequential'):
    f(test_or_class, *args, **kwargs)


def run_all_keras_modes(test_or_class=None,
                        config=None,
                        always_skip_v1=False,
                        always_skip_eager=False,
                        **kwargs):
  """Execute the decorated test with all keras execution modes.

  This decorator is intended to be applied either to individual test methods in
  a `test_combinations.TestCase` class, or directly to a test class that
  extends it. Doing so will cause the contents of the individual test
  method (or all test methods in the class) to be executed multiple times -
  once executing in legacy graph mode, once running eagerly and with
  `should_run_eagerly` returning True, and once running eagerly with
  `should_run_eagerly` returning False.

  If Tensorflow v2 behavior is enabled, legacy graph mode will be skipped, and
  the test will only run twice.

  Note: if stacking this decorator with absl.testing's parameterized decorators,
  those should be at the bottom of the stack.

  For example, consider the following unittest:

  ```python
  class MyTests(test_utils.KerasTestCase):

    @test_utils.run_all_keras_modes
    def test_foo(self):
      model = test_utils.get_small_functional_mlp(1, 4, input_dim=3)
      optimizer = RMSPropOptimizer(learning_rate=0.001)
      loss = 'mse'
      metrics = ['mae']
      model.compile(
          optimizer, loss, metrics=metrics,
          run_eagerly=test_utils.should_run_eagerly())

      inputs = np.zeros((10, 3))
      targets = np.zeros((10, 4))
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(10)

      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)

  if __name__ == "__main__":
    tf.test.main()
  ```

  This test will try compiling & fitting the small functional mlp using all
  three Keras execution modes.

  Args:
    test_or_class: test method or class to be annotated. If None,
      this method returns a decorator that can be applied to a test method or
      test class. If it is not None this returns the decorator applied to the
      test or class.
    config: An optional config_pb2.ConfigProto to use to configure the
      session when executing graphs.
    always_skip_v1: If True, does not try running the legacy graph mode even
      when Tensorflow v2 behavior is not enabled.
    always_skip_eager: If True, does not execute the decorated test
      with eager execution modes.
    **kwargs: Additional kwargs for configuring tests for
     in-progress Keras behaviors/ refactorings that we haven't fully
     rolled out yet

  Returns:
    Returns a decorator that will run the decorated test method multiple times.

  Raises:
    ImportError: If abseil parameterized is not installed or not included as
      a target dependency.
  """
  if kwargs:
    raise ValueError('Unrecognized keyword args: {}'.format(kwargs))

  params = [('_v2_function', 'v2_function')]
  if not always_skip_eager:
    params.append(('_v2_eager', 'v2_eager'))
  if not (always_skip_v1 or tf.__internal__.tf2.enabled()):
    params.append(('_v1_session', 'v1_session'))

  def single_method_decorator(f):
    """Decorator that constructs the test cases."""

    # Use named_parameters so it can be individually run from the command line
    @parameterized.named_parameters(*params)
    @functools.wraps(f)
    def decorated(self, run_mode, *args, **kwargs):
      """A run of a single test case w/ specified run mode."""
      if run_mode == 'v1_session':
        _v1_session_test(f, self, config, *args, **kwargs)
      elif run_mode == 'v2_eager':
        _v2_eager_test(f, self, *args, **kwargs)
      elif run_mode == 'v2_function':
        _v2_function_test(f, self, *args, **kwargs)
      else:
        return ValueError('Unknown run mode %s' % run_mode)

    return decorated

  return _test_or_class_decorator(test_or_class, single_method_decorator)


def _v1_session_test(f, test_or_class, config, *args, **kwargs):
  with tf.compat.v1.get_default_graph().as_default():
    with test_utils.run_eagerly_scope(False):
      with test_or_class.test_session(config=config):
        f(test_or_class, *args, **kwargs)


def _v2_eager_test(f, test_or_class, *args, **kwargs):
  with tf.__internal__.eager_context.eager_mode():
    with test_utils.run_eagerly_scope(True):
      f(test_or_class, *args, **kwargs)


def _v2_function_test(f, test_or_class, *args, **kwargs):
  with tf.__internal__.eager_context.eager_mode():
    with test_utils.run_eagerly_scope(False):
      f(test_or_class, *args, **kwargs)


def _test_or_class_decorator(test_or_class, single_method_decorator):
  """Decorate a test or class with a decorator intended for one method.

  If the test_or_class is a class:
    This will apply the decorator to all test methods in the class.

  If the test_or_class is an iterable of already-parameterized test cases:
    This will apply the decorator to all the cases, and then flatten the
    resulting cross-product of test cases. This allows stacking the Keras
    parameterized decorators w/ each other, and to apply them to test methods
    that have already been marked with an absl parameterized decorator.

  Otherwise, treat the obj as a single method and apply the decorator directly.

  Args:
    test_or_class: A test method (that may have already been decorated with a
      parameterized decorator, or a test class that extends
      test_combinations.TestCase
    single_method_decorator:
      A parameterized decorator intended for a single test method.
  Returns:
    The decorated result.
  """
  def _decorate_test_or_class(obj):
    if isinstance(obj, collections.abc.Iterable):
      return itertools.chain.from_iterable(
          single_method_decorator(method) for method in obj)
    if isinstance(obj, type):
      cls = obj
      for name, value in cls.__dict__.copy().items():
        if callable(value) and name.startswith(
            unittest.TestLoader.testMethodPrefix):
          setattr(cls, name, single_method_decorator(value))

      cls = type(cls).__new__(type(cls), cls.__name__, cls.__bases__,
                              cls.__dict__.copy())
      return cls

    return single_method_decorator(obj)

  if test_or_class is not None:
    return _decorate_test_or_class(test_or_class)

  return _decorate_test_or_class


def keras_mode_combinations(mode=None, run_eagerly=None):
  """Returns the default test combinations for tf.keras tests.

  Note that if tf2 is enabled, then v1 session test will be skipped.

  Args:
    mode: List of modes to run the tests. The valid options are 'graph' and
      'eager'. Default to ['graph', 'eager'] if not specified. If a empty list
      is provide, then the test will run under the context based on tf's
      version, eg graph for v1 and eager for v2.
    run_eagerly: List of `run_eagerly` value to be run with the tests.
      Default to [True, False] if not specified. Note that for `graph` mode,
      run_eagerly value will only be False.

  Returns:
    A list contains all the combinations to be used to generate test cases.
  """
  if mode is None:
    mode = ['eager'] if tf.__internal__.tf2.enabled() else ['graph', 'eager']
  if run_eagerly is None:
    run_eagerly = [True, False]
  result = []
  if 'eager' in mode:
    result += tf.__internal__.test.combinations.combine(mode=['eager'], run_eagerly=run_eagerly)
  if 'graph' in mode:
    result += tf.__internal__.test.combinations.combine(mode=['graph'], run_eagerly=[False])
  return result


def keras_model_type_combinations():
  return tf.__internal__.test.combinations.combine(model_type=KERAS_MODEL_TYPES)


class KerasModeCombination(tf.__internal__.test.combinations.TestCombination):
  """Combination for Keras test mode.

  It by default includes v1_session, v2_eager and v2_tf_function.
  """

  def context_managers(self, kwargs):
    run_eagerly = kwargs.pop('run_eagerly', None)

    if run_eagerly is not None:
      return [test_utils.run_eagerly_scope(run_eagerly)]
    else:
      return []

  def parameter_modifiers(self):
    return [tf.__internal__.test.combinations.OptionalParameter('run_eagerly')]


class KerasModelTypeCombination(tf.__internal__.test.combinations.TestCombination):
  """Combination for Keras model types when doing model test.

  It by default includes 'functional', 'subclass', 'sequential'.

  Various methods in `testing_utils` to get models will auto-generate a model
  of the currently active Keras model type. This allows unittests to confirm
  the equivalence between different Keras models.
  """

  def context_managers(self, kwargs):
    model_type = kwargs.pop('model_type', None)
    if model_type in KERAS_MODEL_TYPES:
      return [test_utils.model_type_scope(model_type)]
    else:
      return []

  def parameter_modifiers(self):
    return [tf.__internal__.test.combinations.OptionalParameter('model_type')]


_defaults = tf.__internal__.test.combinations.generate.keywords['test_combinations']
generate = functools.partial(
    tf.__internal__.test.combinations.generate,
    test_combinations=_defaults +
    (KerasModeCombination(), KerasModelTypeCombination()))
combine = tf.__internal__.test.combinations.combine
times = tf.__internal__.test.combinations.times
NamedObject = tf.__internal__.test.combinations.NamedObject
