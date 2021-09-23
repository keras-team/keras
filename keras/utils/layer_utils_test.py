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
"""Tests for layer_utils."""

import keras
import tensorflow.compat.v2 as tf

import collections
import contextlib
import multiprocessing.dummy
import os
import pickle
import shutil
import time
import timeit

import numpy as np
from keras.utils import layer_utils


_PICKLEABLE_CALL_COUNT = collections.Counter()


class MyPickleableObject(tf.__internal__.tracking.AutoTrackable):
  """Needed for InterfaceTests.test_property_cache_serialization.

  This class must be at the top level. This is a constraint of pickle,
  unrelated to `cached_per_instance`.
  """

  @property
  @layer_utils.cached_per_instance
  def my_id(self):
    _PICKLEABLE_CALL_COUNT[self] += 1
    return id(self)


class LayerUtilsTest(tf.test.TestCase):

  def test_print_summary(self):
    model = keras.Sequential()
    model.add(
        keras.layers.Conv2D(
            filters=2, kernel_size=(2, 3), input_shape=(3, 5, 5), name='conv'))
    model.add(keras.layers.Flatten(name='flat'))
    model.add(keras.layers.Dense(5, name='dense'))

    file_name = 'model_1.txt'
    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
    fpath = os.path.join(temp_dir, file_name)
    writer = open(fpath, 'w')

    def print_to_file(text):
      print(text, file=writer)

    try:
      layer_utils.print_summary(model, print_fn=print_to_file)
      self.assertTrue(tf.io.gfile.exists(fpath))
      writer.close()
      reader = open(fpath, 'r')
      lines = reader.readlines()
      reader.close()
      self.assertEqual(len(lines), 15)
    except ImportError:
      pass

  def test_print_summary_expand_nested(self):
    shape = (None, None, 3)

    def make_model():
      x = inputs = keras.Input(shape)
      x = keras.layers.Conv2D(3, 1)(x)
      x = keras.layers.BatchNormalization()(x)
      return keras.Model(inputs, x)

    x = inner_inputs = keras.Input(shape)
    x = make_model()(x)
    inner_model = keras.Model(inner_inputs, x)

    inputs = keras.Input(shape)
    model = keras.Model(inputs, inner_model(inputs))

    file_name = 'model_2.txt'
    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
    fpath = os.path.join(temp_dir, file_name)
    writer = open(fpath, 'w')

    def print_to_file(text):
      print(text, file=writer)

    try:
      layer_utils.print_summary(
          model, print_fn=print_to_file, expand_nested=True)
      self.assertTrue(tf.io.gfile.exists(fpath))
      writer.close()
      reader = open(fpath, 'r')
      lines = reader.readlines()
      reader.close()
      check_str = (
          'Model: "model_2"\n'
          '_________________________________________________________________\n'
          ' Layer (type)                Output Shape              Param #   \n'
          '=================================================================\n'
          ' input_3 (InputLayer)        [(None, None, None, 3)]   0         \n'
          '                                                                 \n'
          ' model_1 (Functional)        (None, None, None, 3)     24        \n'
          '|¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|\n'
          '| input_1 (InputLayer)      [(None, None, None, 3)]   0         |\n'
          '|                                                               |\n'
          '| model (Functional)        (None, None, None, 3)     24        |\n'
          '||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||\n'
          '|| input_2 (InputLayer)    [(None, None, None, 3)]   0         ||\n'
          '||                                                             ||\n'
          '|| conv2d (Conv2D)         (None, None, None, 3)     12        ||\n'
          '||                                                             ||\n'
          '|| batch_normalization (BatchN  (None, None, None, 3)  12      ||\n'
          '|| ormalization)                                               ||\n'
          '|¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|\n'
          '¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯\n'
          '=================================================================\n'
          'Total params: 24\n'
          'Trainable params: 18\n'
          'Non-trainable params: 6\n'
          '_________________________________________________________________\n')

      fin_str = ''
      for line in lines:
        fin_str += line

      self.assertIn(fin_str, check_str)
      self.assertEqual(len(lines), 25)
    except ImportError:
      pass

  def test_summary_subclass_model_expand_nested(self):

    class Sequential(keras.Model):

      def __init__(self, *args):
        super(Sequential, self).__init__()
        self.module_list = list(args) if args else []

      def call(self, x):
        for module in self.module_list:
          x = module(x)
        return x

    class Block(keras.Model):

      def __init__(self):
        super(Block, self).__init__()
        self.module = Sequential(
            keras.layers.Dense(10),
            keras.layers.Dense(10),
        )

      def call(self, input_tensor):
        x = self.module(input_tensor)
        return x

    class Base(keras.Model):

      def __init__(self):
        super(Base, self).__init__()
        self.module = Sequential(Block(), Block())

      def call(self, input_tensor):
        x = self.module(input_tensor)
        y = self.module(x)
        return x, y

    class Network(keras.Model):

      def __init__(self):
        super(Network, self).__init__()
        self.child = Base()

      def call(self, inputs):
        return self.child(inputs)

    net = Network()
    inputs = keras.Input(shape=(10,))
    outputs = net(inputs)
    model = keras.models.Model(inputs=inputs, outputs=outputs)

    file_name = 'model_3.txt'
    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
    fpath = os.path.join(temp_dir, file_name)
    writer = open(fpath, 'w')

    def print_to_file(text):
      print(text, file=writer)

    try:
      layer_utils.print_summary(
          model, line_length=120, print_fn=print_to_file, expand_nested=True)
      self.assertTrue(tf.io.gfile.exists(fpath))
      writer.close()
      reader = open(fpath, 'r')
      lines = reader.readlines()
      reader.close()
      # The output content are slightly different for the input shapes between
      # v1 and v2.
      if tf.__internal__.tf2.enabled():
        self.assertEqual(len(lines), 39)
      else:
        self.assertEqual(len(lines), 40)
    except ImportError:
      pass

  def test_property_cache(self):
    test_counter = collections.Counter()

    class MyObject(tf.__internal__.tracking.AutoTrackable):

      def __init__(self):
        super(MyObject, self).__init__()
        self._frozen = True

      def __setattr__(self, key, value):
        """Enforce that cache does not set attribute on MyObject."""
        if getattr(self, '_frozen', False):
          raise ValueError('Cannot mutate when frozen.')
        return super(MyObject, self).__setattr__(key, value)

      @property
      @layer_utils.cached_per_instance
      def test_property(self):
        test_counter[id(self)] += 1
        return id(self)

    first_object = MyObject()
    second_object = MyObject()

    # Make sure the objects return the correct values
    self.assertEqual(first_object.test_property, id(first_object))
    self.assertEqual(second_object.test_property, id(second_object))

    # Make sure the cache does not share across objects
    self.assertNotEqual(first_object.test_property, second_object.test_property)

    # Check again (Now the values should be cached.)
    self.assertEqual(first_object.test_property, id(first_object))
    self.assertEqual(second_object.test_property, id(second_object))

    # Count the function calls to make sure the cache is actually being used.
    self.assertAllEqual(tuple(test_counter.values()), (1, 1))

  def test_property_cache_threaded(self):
    call_count = collections.Counter()

    class MyObject(tf.__internal__.tracking.AutoTrackable):

      @property
      @layer_utils.cached_per_instance
      def test_property(self):
        # Random sleeps to ensure that the execution thread changes
        # mid-computation.
        call_count['test_property'] += 1
        time.sleep(np.random.random() + 1.)

        # Use a RandomState which is seeded off the instance's id (the mod is
        # because numpy limits the range of seeds) to ensure that an instance
        # returns the same value in different threads, but different instances
        # return different values.
        return int(np.random.RandomState(id(self) % (2 ** 31)).randint(2 ** 16))

      def get_test_property(self, _):
        """Function provided to .map for threading test."""
        return self.test_property

    # Test that multiple threads return the same value. This requires that
    # the underlying function is repeatable, as cached_property makes no attempt
    # to prioritize the first call.
    test_obj = MyObject()
    with contextlib.closing(multiprocessing.dummy.Pool(32)) as pool:
      # Intentionally make a large pool (even when there are only a small number
      # of cpus) to ensure that the runtime switches threads.
      results = pool.map(test_obj.get_test_property, range(64))
    self.assertEqual(len(set(results)), 1)

    # Make sure we actually are testing threaded behavior.
    self.assertGreater(call_count['test_property'], 1)

    # Make sure new threads still cache hit.
    with contextlib.closing(multiprocessing.dummy.Pool(2)) as pool:
      start_time = timeit.default_timer()  # Don't time pool instantiation.
      results = pool.map(test_obj.get_test_property, range(4))
    total_time = timeit.default_timer() - start_time

    # Note(taylorrobie): The reason that it is safe to time a unit test is that
    #                    a cache hit will be << 1 second, and a cache miss is
    #                    guaranteed to be >= 1 second. Empirically confirmed by
    #                    100,000 runs with no flakes.
    self.assertLess(total_time, 0.95)

  def test_property_cache_serialization(self):
    # Reset call count. .keys() must be wrapped in a list, because otherwise we
    # would mutate the iterator while iterating.
    for k in list(_PICKLEABLE_CALL_COUNT.keys()):
      _PICKLEABLE_CALL_COUNT.pop(k)

    first_instance = MyPickleableObject()
    self.assertEqual(id(first_instance), first_instance.my_id)

    # Test that we can pickle and un-pickle
    second_instance = pickle.loads(pickle.dumps(first_instance))

    self.assertEqual(id(second_instance), second_instance.my_id)
    self.assertNotEqual(first_instance.my_id, second_instance.my_id)

    # Make sure de-serialized object uses the cache.
    self.assertEqual(_PICKLEABLE_CALL_COUNT[second_instance], 1)

    # Make sure the decorator cache is not being serialized with the object.
    expected_size = len(pickle.dumps(second_instance))
    for _ in range(5):
      # Add some more entries to the cache.
      _ = MyPickleableObject().my_id
    self.assertEqual(len(_PICKLEABLE_CALL_COUNT), 7)
    size_check_instance = MyPickleableObject()
    _ = size_check_instance.my_id
    self.assertEqual(expected_size, len(pickle.dumps(size_check_instance)))


if __name__ == '__main__':
  tf.test.main()
