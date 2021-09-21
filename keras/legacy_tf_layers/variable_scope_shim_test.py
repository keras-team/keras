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
"""Tests for variable store."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

import gc
import threading

from absl.testing import parameterized
import numpy
from tensorflow.python.framework import test_util
from keras import combinations
from keras import regularizers
from keras.engine import input_layer as input_layer_module
from keras.engine import training as training_module
from keras.layers import core
from keras.legacy_tf_layers import core as core_layers
from keras.legacy_tf_layers import variable_scope_shim
from tensorflow.python.ops import variable_scope


def run_inside_wrap_function_in_eager_mode(graph_function):
  """Decorator to execute the same graph code in eager and graph modes.

  In graph mode, we just execute the graph_function passed as argument. In eager
  mode, we wrap the function using wrap_function and then execute the wrapped
  result.

  Args:
    graph_function: python function containing graph code to be wrapped

  Returns:
    decorated function
  """
  def wrap_and_execute(self):
    store = variable_scope_shim._EagerVariableStore()
    with variable_scope.with_variable_store(store):
      # use the original function
      graph_function(self)
  return wrap_and_execute


class VariableScopeTest(tf.test.TestCase):

  def tearDown(self):
    gc.collect()
    # This will only contain uncollectable garbage, i.e. reference cycles
    # involving objects with __del__ defined.
    self.assertEqual(0, len(gc.garbage))

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testGetVar(self):
    vs = variable_scope._get_default_variable_store()
    v = vs.get_variable("v", [1])
    v1 = vs.get_variable("v", [1])
    self.assertIs(v, v1)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testNameExists(self):
    vs = variable_scope._get_default_variable_store()
    # No check by default, so we can both create and get existing names.
    v = vs.get_variable("v", [1])
    v1 = vs.get_variable("v", [1])
    self.assertIs(v, v1)

    self.assertIsNot(v, vs.get_variable("u", [1], reuse=False))

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testNamelessStore(self):
    vs = variable_scope._get_default_variable_store()
    vs.get_variable("v1", [2])
    vs.get_variable("v2", [2])
    expected_names = ["%s:0" % name for name in ["v1", "v2"]]
    self.assertEqual(
        set(expected_names), set(v.name for v in vs._vars.values()))

  # TODO(mihaimaruseac): Not converted to use wrap_function because of
  # TypeError: Expected tf.group() expected Tensor arguments not 'None' with
  # type '<type 'NoneType'>'
  @test_util.run_in_graph_and_eager_modes
  def testVarScopeInitializer(self):
    init = tf.compat.v1.constant_initializer(0.3)
    with tf.compat.v1.variable_scope("tower0") as tower:
      with tf.compat.v1.variable_scope("foo", initializer=init):
        v = tf.compat.v1.get_variable("v", [])
        self.evaluate(tf.compat.v1.variables_initializer([v]))
        self.assertAllClose(self.evaluate(v.value()), 0.3)
      with tf.compat.v1.variable_scope(tower, initializer=init):
        w = tf.compat.v1.get_variable("w", [])
        self.evaluate(tf.compat.v1.variables_initializer([w]))
        self.assertAllClose(self.evaluate(w.value()), 0.3)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarScopeConstraint(self):
    constraint = lambda x: 0. * x
    with tf.compat.v1.variable_scope("tower1") as tower:
      with tf.compat.v1.variable_scope("foo", constraint=constraint):
        v = tf.compat.v1.get_variable("v", [])
        self.assertIsNotNone(v.constraint)
      with tf.compat.v1.variable_scope(tower, constraint=constraint):
        w = tf.compat.v1.get_variable("w", [])
        self.assertIsNotNone(w.constraint)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarScopeDType(self):
    with tf.compat.v1.variable_scope("tower2") as tower:
      with tf.compat.v1.variable_scope("foo", dtype=tf.float16):
        v = tf.compat.v1.get_variable("v", [])
        self.assertEqual(v.dtype.base_dtype, tf.float16)
      with tf.compat.v1.variable_scope(tower, dtype=tf.float16):
        w = tf.compat.v1.get_variable("w", [])
        self.assertEqual(w.dtype.base_dtype, tf.float16)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testInitFromNonTensorValue(self):
    v = tf.compat.v1.get_variable("v4", initializer=4, dtype=tf.int32)
    self.evaluate(tf.compat.v1.variables_initializer([v]))
    self.assertAllClose(self.evaluate(v.value()), 4)

    w = tf.compat.v1.get_variable(
        "w4", initializer=numpy.array([1, 2, 3]), dtype=tf.int64)
    self.evaluate(tf.compat.v1.variables_initializer([w]))
    self.assertAllClose(self.evaluate(w.value()), [1, 2, 3])

    # A quirk to be revisited?
    error = ValueError if tf.executing_eagerly() else TypeError
    with self.assertRaises(error):
      tf.compat.v1.get_variable("x4", initializer={})

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testInitFromNonInitializer(self):
    # Test various dtypes with zeros initializer as following:
    types = [
        tf.int8, tf.uint8, tf.int16, tf.uint16, tf.int32,
        tf.int64, tf.bool
    ]

    # Use different variable_name to distinguish various dtypes
    for (i, dtype) in enumerate(types):
      x = tf.compat.v1.get_variable(
          name="xx%d" % i, shape=(3, 4), dtype=dtype)
      y = tf.compat.v1.get_variable(
          name="yy%d" % i,
          shape=(3, 4),
          dtype=dtype,
          initializer=tf.compat.v1.zeros_initializer(dtype=dtype))

      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.assertAllEqual(self.evaluate(x.value()), self.evaluate(y.value()))

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarScopeRegularizer(self):
    init = tf.compat.v1.constant_initializer(0.3)

    def regularizer1(v):
      return tf.reduce_mean(v) + 0.1

    def regularizer2(v):
      return tf.reduce_mean(v) + 0.2

    with tf.compat.v1.variable_scope(
        "tower3", regularizer=regularizer1) as tower:
      with tf.compat.v1.variable_scope("foo", initializer=init):
        v = tf.compat.v1.get_variable("v", [])
        self.evaluate(tf.compat.v1.variables_initializer([v]))
      with tf.compat.v1.variable_scope(tower, initializer=init) as vs:
        tf.compat.v1.get_variable("u", [])
        vs.set_regularizer(regularizer2)
        tf.compat.v1.get_variable("w", [])
        # Next 3 variable not regularized to test disabling regularization.
        tf.compat.v1.get_variable(
            "x", [], regularizer=tf.compat.v1.no_regularizer)
        with tf.compat.v1.variable_scope(
            "baz", regularizer=tf.compat.v1.no_regularizer):
          tf.compat.v1.get_variable("y", [])
        vs.set_regularizer(tf.compat.v1.no_regularizer)
        tf.compat.v1.get_variable("z", [])

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testInitializeFromValue(self):
    init = tf.constant(0.1)
    w = tf.compat.v1.get_variable("v", initializer=init)
    self.evaluate(tf.compat.v1.variables_initializer([w]))
    self.assertAllClose(self.evaluate(w.value()), 0.1)

    with self.assertRaisesRegex(ValueError, "shape"):
      # We disallow explicit shape specification when initializer is constant.
      tf.compat.v1.get_variable("u", [1], initializer=init)

    with tf.compat.v1.variable_scope("foo", initializer=init):
      # Constant initializer can be passed through scopes if needed.
      v = tf.compat.v1.get_variable("v")
      self.evaluate(tf.compat.v1.variables_initializer([v]))
      self.assertAllClose(self.evaluate(v.value()), 0.1)

    # Check that non-float32 initializer creates a non-float32 variable.
    init = tf.constant(1, dtype=tf.int32)
    t = tf.compat.v1.get_variable("t", initializer=init)
    self.assertEqual(t.dtype.base_dtype, tf.int32)

    # Raise error if `initializer` dtype and `dtype` are not identical.
    with self.assertRaisesRegex(ValueError, "don't match"):
      tf.compat.v1.get_variable("s", initializer=init, dtype=tf.float64)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarScopeGetOrCreateReuse(self):
    with self.cached_session():

      def test_value(value):
        x = tf.constant(value)
        with tf.compat.v1.variable_scope(
            "testVarScopeGetOrCreateReuse_bar",
            reuse=tf.compat.v1.AUTO_REUSE):
          _ = tf.compat.v1.assign(tf.compat.v1.get_variable("var", []), x)
        with tf.compat.v1.variable_scope(
            "testVarScopeGetOrCreateReuse_bar",
            reuse=tf.compat.v1.AUTO_REUSE):
          _ = tf.compat.v1.get_variable("var", [])
        self.assertEqual(value, self.evaluate(x))

      test_value(42.)  # Variable is created.
      test_value(13.)  # Variable is reused hereafter.
      test_value(17.)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarScopeGetOrCreateReuseIgnoreFalse(self):
    with self.cached_session():

      def test_value(value):
        x = tf.constant(value)
        with tf.compat.v1.variable_scope(
            "testVarScopeGetOrCreateReuse_bar",
            reuse=False):
          _ = tf.compat.v1.assign(tf.compat.v1.get_variable("var", []), x)
        # We need to ignore reuse=False in the shim, because the
        # code is expected to get rerun each time the user calls the shim.
        with tf.compat.v1.variable_scope(
            "testVarScopeGetOrCreateReuse_bar",
            reuse=False):
          _ = tf.compat.v1.get_variable("var", [])
        self.assertEqual(value, self.evaluate(x))

      test_value(42.)  # Variable is created.
      test_value(13.)  # Variable is reused hereafter.
      test_value(17.)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarOpScope(self):
    with self.cached_session():
      with tf.name_scope("testVarOpScope1"):
        with tf.compat.v1.variable_scope("tower", "default", []):
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name, "tower/w:0")

      with tf.name_scope("testVarOpScope2"):
        with tf.compat.v1.variable_scope(None, "default", []):
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name, "default/w:0")
        with tf.compat.v1.variable_scope(None, "default", []):
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name, "default_1/w:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarOpScopeUniqueNamesInterleavedSubstringScopes(self):
    with self.cached_session():
      with tf.compat.v1.variable_scope(None, "defaultScope1"):
        with tf.compat.v1.variable_scope(None, "layer"):
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name,
              "defaultScope1/layer/w:0")
      with tf.compat.v1.variable_scope(None, "defaultScope1"):
        with tf.compat.v1.variable_scope(None, "layer"):
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name,
              "defaultScope1_1/layer/w:0")
      with tf.compat.v1.variable_scope(None, "defaultScope"):
        with tf.compat.v1.variable_scope(None, "layer"):
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name,
              "defaultScope/layer/w:0")
      with tf.compat.v1.variable_scope(None, "defaultScope1"):
        with tf.compat.v1.variable_scope(None, "layer"):
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name,
              "defaultScope1_2/layer/w:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarOpScopeUniqueNamesWithJump(self):
    with self.cached_session():
      with tf.compat.v1.variable_scope("default") as default:
        with tf.compat.v1.variable_scope(None, "layer"):
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name, "default/layer/w:0")
        with tf.compat.v1.variable_scope(None, "layer"):
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name,
              "default/layer_1/w:0")
        with tf.compat.v1.variable_scope(default):
          pass
        # No matter the jump in the middle, unique numbering continues.
        with tf.compat.v1.variable_scope(None, "layer"):
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name,
              "default/layer_2/w:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarOpScopeReuse(self):
    with self.cached_session():
      with tf.compat.v1.variable_scope("outer") as outer:
        with tf.compat.v1.variable_scope("tower", "default", []):
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name, "outer/tower/w:0")
        with tf.compat.v1.variable_scope(None, "default", []):
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name, "outer/default/w:0")

      with tf.compat.v1.variable_scope(outer, reuse=True) as outer:
        with tf.compat.v1.variable_scope("tower", "default", []):
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name, "outer/tower/w:0")
        with tf.compat.v1.variable_scope(None, "default", []):
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name, "outer/default/w:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarScopeGetVar(self):
    with self.cached_session():
      with tf.compat.v1.variable_scope("root"):
        with tf.compat.v1.variable_scope("towerA") as tower_a:
          va = tf.compat.v1.get_variable("v", [1])
          self.assertEqual(va.name, "root/towerA/v:0")

        with tf.compat.v1.variable_scope(tower_a, reuse=True):
          va2 = tf.compat.v1.get_variable("v", [1])
          self.assertIs(va2, va)

        with tf.compat.v1.variable_scope("towerB"):
          vb = tf.compat.v1.get_variable("v", [1])
          self.assertEqual(vb.name, "root/towerB/v:0")

        with tf.compat.v1.variable_scope("towerA", reuse=True):
          va2 = tf.compat.v1.get_variable("v", [1])
          self.assertIs(va2, va)

        with tf.compat.v1.variable_scope("foo"):
          with tf.compat.v1.variable_scope("bar"):
            v = tf.compat.v1.get_variable("v", [1])
            self.assertEqual(v.name, "root/foo/bar/v:0")
            with tf.compat.v1.variable_scope(tower_a, reuse=True):
              va3 = tf.compat.v1.get_variable("v", [1])
              self.assertIs(va, va3)

        with self.assertRaises(ValueError) as exc:
          with tf.compat.v1.variable_scope(tower_a, reuse=True):
            tf.compat.v1.get_variable("v", [2])  # Different shape.
        self.assertEqual("shape" in str(exc.exception), True)

        with self.assertRaises(ValueError) as exc:
          with tf.compat.v1.variable_scope(tower_a, reuse=True):
            tf.compat.v1.get_variable("v", [1], dtype=tf.int32)
        self.assertEqual("dtype" in str(exc.exception), True)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarScopeOuterScope(self):
    with self.cached_session():
      with tf.compat.v1.variable_scope("outer") as outer:
        pass
      with tf.compat.v1.variable_scope(outer):
        self.assertEqual(
            tf.compat.v1.get_variable("w", []).name, "outer/w:0")
        with tf.compat.v1.variable_scope("default"):
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name, "outer/default/w:0")

      with tf.compat.v1.variable_scope(outer, reuse=True):
        self.assertEqual(
            tf.compat.v1.get_variable("w", []).name, "outer/w:0")
        with tf.compat.v1.variable_scope("default", reuse=True):
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name, "outer/default/w:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarScopeNestedOuterScope(self):
    with self.cached_session():
      with tf.compat.v1.variable_scope("outer") as outer:
        with tf.compat.v1.variable_scope(outer):
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name, "outer/w:0")
        with tf.compat.v1.variable_scope("default"):
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name, "outer/default/w:0")

        with tf.compat.v1.variable_scope(outer, reuse=True):
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name, "outer/w:0")
        with tf.compat.v1.variable_scope("default", reuse=True):
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name, "outer/default/w:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarOpScopeReuseParam(self):
    with self.cached_session():
      with tf.compat.v1.variable_scope("outer") as outer:
        with tf.compat.v1.variable_scope("tower", "default", []):
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name, "outer/tower/w:0")
        with tf.compat.v1.variable_scope(None, "default", []):
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name, "outer/default/w:0")

      with tf.compat.v1.variable_scope(outer) as outer:
        with tf.compat.v1.variable_scope("tower", "default", reuse=True):
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name, "outer/tower/w:0")
        outer.reuse_variables()
        with tf.compat.v1.variable_scope(None, "default", []):
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name, "outer/default/w:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarOpScopeReuseError(self):
    with self.cached_session():
      with self.assertRaises(ValueError):
        with tf.compat.v1.variable_scope(None, "default", reuse=True):
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name, "outer/tower/w:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarOpScopeOuterScope(self):
    with self.cached_session():
      with tf.compat.v1.variable_scope("outer") as outer:
        pass
      with tf.compat.v1.variable_scope(outer, "default", []):
        self.assertEqual(
            tf.compat.v1.get_variable("w", []).name, "outer/w:0")
        with tf.compat.v1.variable_scope(None, "default", []):
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name, "outer/default/w:0")

      with tf.compat.v1.variable_scope(outer, "default", reuse=True):
        self.assertEqual(
            tf.compat.v1.get_variable("w", []).name, "outer/w:0")
        outer.reuse_variables()
        with tf.compat.v1.variable_scope(None, "default", []):
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name, "outer/default/w:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarOpScopeNestedOuterScope(self):
    with self.cached_session():
      with tf.compat.v1.variable_scope("outer") as outer:
        with tf.compat.v1.variable_scope(outer, "default", []):
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name, "outer/w:0")
        with tf.compat.v1.variable_scope(None, "default", []):
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name, "outer/default/w:0")

      with tf.compat.v1.variable_scope(outer, "default", reuse=True):
        self.assertEqual(
            tf.compat.v1.get_variable("w", []).name, "outer/w:0")
        with tf.compat.v1.variable_scope(None, "default", []):
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name, "outer/default/w:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testBasicWhenAuxiliaryNameScopeIsFalse(self):
    with self.cached_session():
      with tf.compat.v1.variable_scope(
          "scope", auxiliary_name_scope=False) as scope:
        self.assertEqual(
            tf.compat.v1.get_variable("w", []).name, "scope/w:0")
      with tf.compat.v1.variable_scope(scope, auxiliary_name_scope=False):
        self.assertEqual(
            tf.compat.v1.get_variable("w1", []).name, "scope/w1:0")

      with tf.compat.v1.variable_scope("outer"):
        with tf.compat.v1.variable_scope(
            "inner", auxiliary_name_scope=False) as inner:
          self.assertEqual(inner.original_name_scope, "outer/")
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name, "outer/inner/w:0")
        with tf.compat.v1.variable_scope(
            inner, auxiliary_name_scope=False) as inner1:
          self.assertEqual(inner1.original_name_scope, "outer/")
          self.assertEqual(
              tf.compat.v1.get_variable("w1", []).name, "outer/inner/w1:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testCreatedByDefaultNameWhenAuxiliaryNameScopeIsFalse(self):
    with self.cached_session():
      with tf.compat.v1.variable_scope(
          None, default_name="default", auxiliary_name_scope=False):
        self.assertEqual(
            tf.compat.v1.get_variable("w", []).name, "default/w:0")

      with tf.compat.v1.variable_scope("outer"):
        with tf.compat.v1.variable_scope(
            None, default_name="default",
            auxiliary_name_scope=False) as inner:
          self.assertEqual(inner.original_name_scope, "outer/")
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name, "outer/default/w:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testReenterRootScopeWhenAuxiliaryNameScopeIsFalse(self):
    with self.cached_session():
      root_scope = tf.compat.v1.get_variable_scope()
      with tf.compat.v1.variable_scope(
          root_scope, auxiliary_name_scope=False):
        self.assertEqual(tf.compat.v1.get_variable("w", []).name, "w:0")

      with tf.compat.v1.variable_scope("outer"):
        with tf.compat.v1.variable_scope(
            root_scope, auxiliary_name_scope=False) as inner:
          self.assertEqual(inner.original_name_scope, "")
          self.assertEqual(tf.compat.v1.get_variable("w1", []).name, "w1:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testAuxiliaryNameScopeIsInvalid(self):
    with self.cached_session():
      with self.assertRaisesRegex(TypeError, "auxiliary_name_scope"):
        with tf.compat.v1.variable_scope(
            None, default_name="scope", auxiliary_name_scope="invalid"):
          pass

      with self.assertRaisesRegex(TypeError, "auxiliary_name_scope"):
        with tf.compat.v1.variable_scope(
            "scope", auxiliary_name_scope="invalid"):
          pass

      with tf.compat.v1.variable_scope("scope") as scope:
        pass
      with self.assertRaisesRegex(TypeError, "auxiliary_name_scope"):
        with tf.compat.v1.variable_scope(
            scope, auxiliary_name_scope="invalid"):
          pass

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testReuseScopeWithoutNameScopeCollision(self):
    # Github issue: #13429
    with self.cached_session():
      with tf.compat.v1.variable_scope("outer"):
        with tf.compat.v1.variable_scope("inner") as inner:
          pass

      with tf.compat.v1.variable_scope(
          inner, auxiliary_name_scope=False) as scope:
        with tf.name_scope(scope.original_name_scope):
          self.assertEqual(
              tf.compat.v1.get_variable("w", []).name, "outer/inner/w:0")

      with tf.compat.v1.variable_scope("another"):
        with tf.compat.v1.variable_scope(
            inner, auxiliary_name_scope=False) as scope1:
          with tf.name_scope(scope1.original_name_scope):
            self.assertEqual(
                tf.compat.v1.get_variable("w1", []).name,
                "outer/inner/w1:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testGetVarWithDevice(self):
    g = tf.Graph()
    varname_type = []

    def device_func(op):
      if op.type in ["Variable", "VariableV2", "VarHandleOp"]:
        varname_type.append((op.name, op.get_attr("dtype")))
      return "/device:GPU:0"

    with g.as_default():
      with tf.compat.v1.device(device_func):
        _ = tf.compat.v1.get_variable("x", (100, 200))
        _ = tf.compat.v1.get_variable(
            "y", dtype=tf.int64, initializer=numpy.arange(73))
    self.assertEqual(varname_type[0], ("x", tf.float32))
    self.assertEqual(varname_type[1], ("y", tf.int64))

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testGetVariableWithRefDtype(self):
    v = tf.compat.v1.get_variable("v", shape=[3, 4], dtype=tf.float32)
    # Ensure it is possible to do get_variable with a _ref dtype passed in.
    _ = tf.compat.v1.get_variable("w", shape=[5, 6], dtype=v.dtype)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testGetVariableWithInitializerWhichTakesNoArgs(self):
    v = tf.compat.v1.get_variable("foo", initializer=lambda: [2])
    self.assertEqual(v.name, "foo:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testGetVariableWithInitializerWhichTakesOptionalArgs(self):
    v = tf.compat.v1.get_variable("foo", initializer=lambda x=True: [2])
    self.assertEqual(v.name, "foo:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testTwoGraphs(self):

    def f():
      g1 = tf.Graph()
      g2 = tf.Graph()
      with g1.as_default():
        with g2.as_default():
          with tf.compat.v1.variable_scope("_"):
            pass

    self.assertRaisesRegex(ValueError,
                           "'_' is not a valid (?:root )?scope name", f)


class VariableScopeWithCustomGetterTest(tf.test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testNonCallableGetterFails(self):
    with self.assertRaisesRegex(ValueError, r"custom_getter .* not callable:"):
      with tf.compat.v1.variable_scope("scope0", custom_getter=3):
        tf.compat.v1.get_variable("name0")
    with self.assertRaisesRegex(ValueError, r"custom_getter .* not callable:"):
      tf.compat.v1.get_variable("name0", custom_getter=3)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testNoSideEffectsWithIdentityCustomGetter(self):
    called = [0]

    def custom_getter(getter, *args, **kwargs):
      called[0] += 1
      return getter(*args, **kwargs)

    with tf.compat.v1.variable_scope(
        "scope", custom_getter=custom_getter) as scope:
      v = tf.compat.v1.get_variable("v", [1])
    with tf.compat.v1.variable_scope(scope, reuse=True):
      v2 = tf.compat.v1.get_variable("v", [1])
    with tf.compat.v1.variable_scope("new_scope") as new_scope:
      v3 = tf.compat.v1.get_variable("v3", [1])
    with tf.compat.v1.variable_scope(
        new_scope, reuse=True, custom_getter=custom_getter):
      v4 = tf.compat.v1.get_variable("v3", [1])

    self.assertIs(v, v2)
    self.assertIs(v3, v4)
    self.assertEqual(3, called[0])  # skipped one in the first new_scope

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testSynchronizationAndAggregationWithCustomGetter(self):
    called = [0]
    synchronization = tf.VariableSynchronization.AUTO
    aggregation = tf.compat.v1.VariableAggregation.NONE

    def custom_getter(getter, *args, **kwargs):
      called[0] += 1

      # Verify synchronization and aggregation kwargs are as expected.
      self.assertEqual(kwargs["synchronization"], synchronization)
      self.assertEqual(kwargs["aggregation"], aggregation)
      return getter(*args, **kwargs)

    with tf.compat.v1.variable_scope("scope", custom_getter=custom_getter):
      tf.compat.v1.get_variable("v", [1])
    self.assertEqual(1, called[0])

    with tf.compat.v1.variable_scope("scope", custom_getter=custom_getter):
      synchronization = tf.VariableSynchronization.ON_READ
      aggregation = tf.compat.v1.VariableAggregation.MEAN
      tf.compat.v1.get_variable(
          "v1", [1], synchronization=synchronization, aggregation=aggregation)

    self.assertEqual(2, called[0])

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVariableCreator(self):
    variable_names = []

    def creator_a(next_creator, **kwargs):
      variable_names.append(kwargs.get("name", ""))
      return next_creator(**kwargs)

    def creator_b(next_creator, **kwargs):
      kwargs["name"] = "forced_name"
      return next_creator(**kwargs)

    with tf.variable_creator_scope(creator_a):
      with tf.variable_creator_scope(creator_b):
        tf.compat.v1.Variable(1.0, name="one_name")

    self.assertEqual(variable_names[0], "forced_name")

    called = [False]

    def creater_c(next_creator, **kwargs):
      called[0] = True
      self.assertEqual(kwargs["synchronization"],
                       tf.VariableSynchronization.ON_WRITE)
      self.assertEqual(kwargs["aggregation"],
                       tf.compat.v1.VariableAggregation.MEAN)
      return next_creator(**kwargs)

    with tf.variable_creator_scope(creater_c):
      tf.compat.v1.get_variable(
          "v", [],
          synchronization=tf.VariableSynchronization.ON_WRITE,
          aggregation=tf.compat.v1.VariableAggregation.MEAN)
    self.assertTrue(called[0])

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVariableCreatorNestingError(self):

    def creator(next_creator, **kwargs):
      return next_creator(**kwargs)

    # Save the state so we can clean up at the end.
    graph = tf.compat.v1.get_default_graph()
    old_creator_stack = graph._variable_creator_stack

    try:
      scope = tf.variable_creator_scope(creator)
      scope.__enter__()
      with tf.variable_creator_scope(creator):
        with self.assertRaises(RuntimeError):
          scope.__exit__(None, None, None)
    finally:
      graph._variable_creator_stack = old_creator_stack


class VariableScopeMultithreadedTest(tf.test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testReenterMainScope(self):

    def thread_fn(graph, main_thread_scope):
      with graph.as_default():
        # Variable created with main scope will have prefix "main".
        with tf.compat.v1.variable_scope(main_thread_scope):
          with tf.compat.v1.variable_scope("foo"):
            v = tf.compat.v1.get_variable("v", [])
            self.assertEqual("main/foo/v:0", v.name)

        # Variable created outside main scope will not have prefix "main".
        with tf.compat.v1.variable_scope("bar"):
          v = tf.compat.v1.get_variable("v", [])
          self.assertEqual("bar/v:0", v.name)

    graph = tf.compat.v1.get_default_graph()
    with tf.compat.v1.variable_scope("main") as main_thread_scope:
      thread = threading.Thread(
          target=thread_fn, args=(graph, main_thread_scope))
      thread.start()
      thread.join()


class CompatV1TemplateScaleByY(variable_scope_shim.VariableScopeLayer):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    def my_op(x, scalar_name):
      var1 = tf.compat.v1.get_variable(
          scalar_name,
          shape=[],
          regularizer=regularizers.L2(),
          initializer=tf.compat.v1.constant_initializer(1.5))
      return x * var1
    self.scale_by_y = tf.compat.v1.make_template(
        "scale_by_y", my_op, scalar_name="y")

  def forward_pass(self, inputs):
    with tf.compat.v1.variable_scope("foo"):
      return self.scale_by_y(inputs)


class VariableScopeModule(tf.Module):
  """Module that uses the shim."""

  @variable_scope_shim.track_tf1_style_variables
  def __call__(self, *args, **kwargs):
    with self.name_scope:
      return self.forward_pass(*args, **kwargs)

  def get_compat_v1_regularization_losses(self):
    """Dict w/ regularization losses from `get_variable`&`compat.v1.layers`."""
    return {name: regularizer() for name, regularizer
            in self._tf1_style_var_store._regularizers.items()}  # pylint: disable=protected-access


@combinations.generate(combinations.combine(mode=["eager"]))
class TF1VariableScopeLayerTest(tf.test.TestCase, parameterized.TestCase):

  def test_get_variable(self):
    # Test the shim when using `get_variable` (and regularizers) directly

    class WrappedDenseLayer(variable_scope_shim.VariableScopeLayer):

      def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units

      def forward_pass(self, inputs, training=None):
        out = inputs
        with tf.compat.v1.variable_scope("dense_one"):
          # The weights are created with a `regularizer`,
          # so the layer should track their regularization losses
          kernel = tf.compat.v1.get_variable(
              shape=[out.shape[-1], self.units],
              regularizer=regularizers.L2(),
              initializer=tf.compat.v1.ones_initializer(),
              name="kernel")
          bias = tf.compat.v1.get_variable(
              shape=[self.units,],
              initializer=tf.compat.v1.zeros_initializer(),
              name="bias")
          out = tf.matmul(out, kernel)
          out = tf.nn.bias_add(out, bias)
        with tf.compat.v1.variable_scope("nested_scope"):
          with tf.compat.v1.variable_scope("dense_two"):
            kernel = tf.compat.v1.get_variable(
                shape=[out.shape[-1], self.units],
                regularizer=regularizers.L2(),
                initializer=tf.compat.v1.ones_initializer(),
                name="kernel")
            bias = tf.compat.v1.get_variable(
                shape=[self.units,],
                initializer=tf.compat.v1.zeros_initializer(),
                name="bias")
            out = tf.matmul(out, kernel)
            out = tf.nn.bias_add(out, bias)
        return out

    layer = WrappedDenseLayer(10)
    out = layer(tf.ones(shape=(5, 5)))
    weights = {x.name: x for x in layer.variables}

    # Verify the correct output, regularization losses, + variables were made
    self.assertEqual(weights.keys(), {"dense_one/bias:0",
                                      "dense_one/kernel:0",
                                      "nested_scope/dense_two/bias:0",
                                      "nested_scope/dense_two/kernel:0"})
    self.assertAllEqual(out, tf.ones(shape=(5, 10)) * 50)
    self.assertAllEqual(tf.add_n(layer.losses), 1.5)

    # Verify reuse by updating the variables then re-running
    weights["dense_one/kernel:0"].assign(tf.ones(shape=(5, 10)) * 2)
    weights["nested_scope/dense_two/kernel:0"].assign(
        tf.ones(shape=(10, 10)) * 2)
    out = layer(tf.ones(shape=(5, 5)))
    self.assertAllEqual(out, tf.ones(shape=(5, 10)) * 200)
    self.assertAllEqual(tf.add_n(layer.losses), 6)

  def test_compat_v1_layer(self):
    # Test the shim when using `compat.v1` layers

    class WrappedDenseLayer(variable_scope_shim.VariableScopeLayer):

      def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units

      def forward_pass(self, inputs, training=None):
        out = core_layers.dense(
            inputs, self.units, name="dense_one",
            kernel_initializer=tf.compat.v1.ones_initializer(),
            kernel_regularizer="l2")
        with tf.compat.v1.variable_scope("nested_scope"):
          out = core_layers.dense(
              out, self.units, name="dense_two",
              kernel_initializer=tf.compat.v1.ones_initializer(),
              kernel_regularizer="l2")
        return out

    layer = WrappedDenseLayer(10)
    out = layer(tf.ones(shape=(5, 5)))
    weights = {x.name: x for x in layer.variables}

    # Verify the correct output, losses, + variables were made
    self.assertEqual(weights.keys(), {"dense_one/bias:0",
                                      "dense_one/kernel:0",
                                      "nested_scope/dense_two/bias:0",
                                      "nested_scope/dense_two/kernel:0"})
    self.assertAllEqual(out, tf.ones(shape=(5, 10)) * 50)
    self.assertAllEqual(tf.add_n(layer.losses), 1.5)

    # Verify reuse by updating the variables then re-running
    weights["dense_one/kernel:0"].assign(tf.ones(shape=(5, 10)) * 2)
    weights["nested_scope/dense_two/kernel:0"].assign(
        tf.ones(shape=(10, 10)) * 2)
    out = layer(tf.ones(shape=(5, 5)))
    self.assertAllEqual(out, tf.ones(shape=(5, 10)) * 200)
    self.assertAllEqual(tf.add_n(layer.losses), 6)

  def test_shim_exporting(self):

    class WrappedDenseLayer(variable_scope_shim.VariableScopeLayer):

      def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units

      def forward_pass(self, inputs, training=None):
        out = core_layers.dense(
            inputs,
            self.units,
            name="dense_one",
            kernel_initializer=tf.compat.v1.ones_initializer(),
            kernel_regularizer="l2")
        with tf.compat.v1.variable_scope("nested_scope"):
          out = core_layers.dense(
              out,
              self.units,
              name="dense_two",
              kernel_initializer=tf.compat.v1.ones_initializer(),
              kernel_regularizer="l2")
        return out

    layer = WrappedDenseLayer(10)
    layer(tf.ones(shape=(5, 5)))

    tmp_dir = self.get_temp_dir()
    tf.saved_model.save(layer, tmp_dir)

  def test_variable_store_scope_get_variable(self):
    # Test the module shim when using `get_variable` (and regularizers) directly

    class WrappedDenseLayer(tf.Module):

      def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self._variable_store = variable_scope_shim._EagerVariableStore()

      def get_compat_v1_regularization_losses(self):
        """Dict w/ regularization losses from `get_variable`."""
        return {name: regularizer() for name, regularizer
                in self._variable_store._regularizers.items()}  # pylint: disable=protected-access

      def __call__(self, inputs, training=None):
        with self._variable_store.scope():
          out = inputs
          with tf.compat.v1.variable_scope("dense_one"):
            # The weights are created with a `regularizer`,
            # so the layer should track their regularization losses
            kernel = tf.compat.v1.get_variable(
                shape=[out.shape[-1], self.units],
                regularizer=regularizers.L2(),
                initializer=tf.compat.v1.ones_initializer(),
                name="kernel")
            bias = tf.compat.v1.get_variable(
                shape=[self.units,],
                initializer=tf.compat.v1.zeros_initializer(),
                name="bias")
            out = tf.matmul(out, kernel)
            out = tf.nn.bias_add(out, bias)
          with tf.compat.v1.variable_scope("nested_scope"):
            with tf.compat.v1.variable_scope("dense_two"):
              kernel = tf.compat.v1.get_variable(
                  shape=[out.shape[-1], self.units],
                  regularizer=regularizers.L2(),
                  initializer=tf.compat.v1.ones_initializer(),
                  name="kernel")
              bias = tf.compat.v1.get_variable(
                  shape=[self.units,],
                  initializer=tf.compat.v1.zeros_initializer(),
                  name="bias")
              out = tf.matmul(out, kernel)
              out = tf.nn.bias_add(out, bias)
          return out

    layer = WrappedDenseLayer(10)
    out = layer(tf.ones(shape=(5, 5)))
    weights = {x.name: x for x in layer.variables}

    # Verify the correct output, regularization losses, + variables were made
    self.assertEqual(weights.keys(), {"dense_one/bias:0",
                                      "dense_one/kernel:0",
                                      "nested_scope/dense_two/bias:0",
                                      "nested_scope/dense_two/kernel:0"})
    self.assertAllEqual(out, tf.ones(shape=(5, 10)) * 50)
    self.assertAllEqual(
        tf.add_n(layer.get_compat_v1_regularization_losses().values()), 1.5)

    # Verify reuse by updating the variables then re-running
    weights["dense_one/kernel:0"].assign(tf.ones(shape=(5, 10)) * 2)
    weights["nested_scope/dense_two/kernel:0"].assign(
        tf.ones(shape=(10, 10)) * 2)
    out = layer(tf.ones(shape=(5, 5)))
    self.assertAllEqual(out, tf.ones(shape=(5, 10)) * 200)
    self.assertAllEqual(
        tf.add_n(layer.get_compat_v1_regularization_losses().values()), 6)

  def test_module_get_variable(self):
    # Test the module shim when using `get_variable` (and regularizers) directly

    class WrappedDenseLayer(VariableScopeModule):

      def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units

      def forward_pass(self, inputs, training=None):
        out = inputs
        with tf.compat.v1.variable_scope("dense_one"):
          # The weights are created with a `regularizer`,
          # so the layer should track their regularization losses
          kernel = tf.compat.v1.get_variable(
              shape=[out.shape[-1], self.units],
              regularizer=regularizers.L2(),
              initializer=tf.compat.v1.ones_initializer(),
              name="kernel")
          bias = tf.compat.v1.get_variable(
              shape=[self.units,],
              initializer=tf.compat.v1.zeros_initializer(),
              name="bias")
          out = tf.matmul(out, kernel)
          out = tf.nn.bias_add(out, bias)
        with tf.compat.v1.variable_scope("nested_scope"):
          with tf.compat.v1.variable_scope("dense_two"):
            kernel = tf.compat.v1.get_variable(
                shape=[out.shape[-1], self.units],
                regularizer=regularizers.L2(),
                initializer=tf.compat.v1.ones_initializer(),
                name="kernel")
            bias = tf.compat.v1.get_variable(
                shape=[self.units,],
                initializer=tf.compat.v1.zeros_initializer(),
                name="bias")
            out = tf.matmul(out, kernel)
            out = tf.nn.bias_add(out, bias)
        return out

    layer = WrappedDenseLayer(10)
    out = layer(tf.ones(shape=(5, 5)))
    weights = {x.name: x for x in layer.variables}

    # Verify the correct output, regularization losses, + variables were made
    self.assertEqual(weights.keys(), {"dense_one/bias:0",
                                      "dense_one/kernel:0",
                                      "nested_scope/dense_two/bias:0",
                                      "nested_scope/dense_two/kernel:0"})
    self.assertAllEqual(out, tf.ones(shape=(5, 10)) * 50)
    self.assertAllEqual(
        tf.add_n(layer.get_compat_v1_regularization_losses().values()), 1.5)

    # Verify reuse by updating the variables then re-running
    weights["dense_one/kernel:0"].assign(tf.ones(shape=(5, 10)) * 2)
    weights["nested_scope/dense_two/kernel:0"].assign(
        tf.ones(shape=(10, 10)) * 2)
    out = layer(tf.ones(shape=(5, 5)))
    self.assertAllEqual(out, tf.ones(shape=(5, 10)) * 200)
    self.assertAllEqual(
        tf.add_n(layer.get_compat_v1_regularization_losses().values()), 6)

  def test_module_compat_v1_layer(self):
    # Test the module shim when using `compat.v1` layers

    class WrappedDenseLayer(VariableScopeModule):

      def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units

      def forward_pass(self, inputs, training=None):
        out = core_layers.dense(
            inputs, self.units, name="dense_one",
            kernel_initializer=tf.compat.v1.ones_initializer(),
            kernel_regularizer="l2")
        with tf.compat.v1.variable_scope("nested_scope"):
          out = core_layers.dense(
              out, self.units, name="dense_two",
              kernel_initializer=tf.compat.v1.ones_initializer(),
              kernel_regularizer="l2")
        return out

    layer = WrappedDenseLayer(10)
    out = layer(tf.ones(shape=(5, 5)))
    weights = {x.name: x for x in layer.variables}

    # Verify the correct output, losses, + variables were made
    self.assertEqual(weights.keys(), {"dense_one/bias:0",
                                      "dense_one/kernel:0",
                                      "nested_scope/dense_two/bias:0",
                                      "nested_scope/dense_two/kernel:0"})
    self.assertAllEqual(out, tf.ones(shape=(5, 10)) * 50)
    self.assertAllEqual(tf.add_n(
        layer.get_compat_v1_regularization_losses().values()), 1.5)

    # Verify reuse by updating the variables then re-running
    weights["dense_one/kernel:0"].assign(tf.ones(shape=(5, 10)) * 2)
    weights["nested_scope/dense_two/kernel:0"].assign(
        tf.ones(shape=(10, 10)) * 2)
    out = layer(tf.ones(shape=(5, 5)))
    self.assertAllEqual(out, tf.ones(shape=(5, 10)) * 200)
    self.assertAllEqual(tf.add_n(
        layer.get_compat_v1_regularization_losses().values()), 6)

  def test_shim_nesting(self):
    # Test that nesting the shim in itself works

    class NestedLayer(variable_scope_shim.VariableScopeLayer):

      def __init__(self, units, name, *args, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.units = units

      def forward_pass(self, inputs):
        out = inputs
        with tf.compat.v1.variable_scope(self.name):
          # The weights are created with a `regularizer`,
          # so the layer should track their regularization losses
          kernel = tf.compat.v1.get_variable(
              shape=[out.shape[-1], self.units],
              regularizer=regularizers.L2(1.0),
              initializer=tf.compat.v1.ones_initializer(),
              name="kernel")
          bias = tf.compat.v1.get_variable(
              shape=[self.units,],
              initializer=tf.compat.v1.initializers.zeros,
              name="bias")
          out = tf.linalg.matmul(out, kernel)
          out = tf.compat.v1.nn.bias_add(out, bias)
        return out

    class WrappedDenseLayer(variable_scope_shim.VariableScopeLayer):

      def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense_layer_a = None
        self.dense_layer_b = None

      def forward_pass(self, inputs):
        # Only create the nested tf.variable/module/layer/model if it has not
        # already been created!
        if not self.dense_layer_a:
          self.dense_layer_a = NestedLayer(self.units * 2, "dense_one")
        out = self.dense_layer_a(inputs)
        if not self.dense_layer_b:
          self.dense_layer_b = NestedLayer(self.units, "dense_two")
        out = self.dense_layer_b(out)
        return out

    layer = WrappedDenseLayer(5)
    out = layer(tf.ones(shape=(1, 3)))
    weights = {x.name: x for x in layer.variables}

    # Verify the correct output, losses, + variables were made
    # (Specifically: no double-counting of any weights or reg. losses
    # between nested components!)
    self.assertEqual({var.name for var in layer.trainable_weights},
                     {"dense_one/bias:0",
                      "dense_one/kernel:0",
                      "dense_two/bias:0",
                      "dense_two/kernel:0"})
    self.assertEqual({var.name for var in layer.dense_layer_a.weights},
                     {"dense_one/bias:0",
                      "dense_one/kernel:0"})
    self.assertEqual({var.name for var in layer.dense_layer_b.weights},
                     {"dense_two/bias:0",
                      "dense_two/kernel:0"})
    self.assertAllEqual(out, tf.ones(shape=(1, 5)) * 30)
    self.assertAllEqual(tf.add_n(layer.dense_layer_a.losses), 30)
    self.assertAllEqual(tf.add_n(layer.dense_layer_b.losses), 50)
    self.assertAllEqual(tf.add_n(layer.losses), 80)

    # Verify reuse by updating the variables then re-running
    weights["dense_one/kernel:0"].assign(tf.ones(shape=(3, 10)) * 2)
    weights["dense_two/kernel:0"].assign(
        tf.ones(shape=(10, 5)) * 2)
    out = layer(tf.ones(shape=(1, 3)))
    self.assertAllEqual(out, tf.ones(shape=(1, 5)) * 120)
    self.assertAllEqual(tf.add_n(layer.losses), 320)

  def test_compat_v1_make_template_in_shim_eager(self):
    # Test the shim when using `compat.v1.make_template`
    # Verify it works correctly in eager
    layer = CompatV1TemplateScaleByY()
    for _ in range(3):
      # Use multiple calls to verify that no new weights get created
      self.assertAllEqual(layer(tf.ones(shape=(2, 3))),
                          tf.constant(1.5, shape=(2, 3)))
    self.assertAllEqual({var.name: var.numpy() for var in layer.weights},
                        {"foo/scale_by_y/y:0": 1.5})
    self.assertAllEqual(tf.add_n(layer.losses),
                        regularizers.L2()(layer.weights[0]))

  def test_compat_v1_make_template_in_shim_tf_function(self):
    # Test the shim when using `compat.v1.make_template`
    # Verify it works correctly in a tf.function
    # when made outside the function
    layer = CompatV1TemplateScaleByY()

    @tf.function
    def foo(x):
      return layer(x), tf.add_n(layer.losses)

    for _ in range(3):
      # Use multiple calls to verify that no new weights get created
      out, loss = foo(tf.ones(shape=(2, 3)))
      self.assertAllEqual(out, tf.constant(1.5, shape=(2, 3)))
      self.assertAllEqual(loss, regularizers.L2()(layer.weights[0]))
    self.assertAllEqual({var.name: var.numpy() for var in layer.weights},
                        {"foo/scale_by_y/y:0": 1.5})

  def test_compat_v1_make_template_in_trace_in_shim(self):
    # Test the shim when using `compat.v1.make_template`
    # Verify it works correctly when the make_template/layer/shim
    # is created on the first tf.function trace!
    layers = {}
    @tf.function
    def bar(x):
      if "layer" not in layers:
        layers["layer"] = CompatV1TemplateScaleByY()
      layer = layers["layer"]
      return layer(x), tf.add_n(layer.losses)

    for _ in range(3):
      # Use multiple calls to verify that no new weights get created
      out, loss = bar(tf.ones(shape=(2, 3)))
      self.assertAllEqual(out, tf.constant(1.5, shape=(2, 3)))
      self.assertAllEqual(loss, regularizers.L2()(layers["layer"].weights[0]))
    self.assertAllEqual(
        {var.name: var.numpy() for var in layers["layer"].weights},
        {"foo/scale_by_y/y:0": 1.5})

  def test_only_track_get_variable(self):
    # Test the shim does not try tracking or reusing variables
    # that were not created by get_variable. These variables/modules/layers
    # need to be tracked separately

    class WrappedDenseLayer(variable_scope_shim.VariableScopeLayer):

      def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self._dense_model = None

      def forward_pass(self, inputs):
        dense_layer = core.Dense(
            self.units, name="dense",
            kernel_initializer=tf.compat.v1.ones_initializer(),
            kernel_regularizer="l2")
        return dense_layer(inputs)

    layer = WrappedDenseLayer(10)
    out = layer(tf.ones(shape=(5, 5)))
    self.assertAllEqual(out, tf.ones(shape=(5, 10)) * 5)

    self.assertEmpty(layer.weights)

  def test_embedded_keras_model(self):
    # Test the shim when embedding a Keras model inside of it
    # And assigning the model to an attribute

    class WrappedDenseLayer(variable_scope_shim.VariableScopeLayer):

      def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self._dense_model = None

      def forward_pass(self, inputs):
        if not self._dense_model:
          inp = input_layer_module.Input(shape=inputs.shape)
          dense_layer = core.Dense(
              self.units, name="dense",
              kernel_initializer=tf.compat.v1.ones_initializer(),
              kernel_regularizer="l2")
          self._dense_model = training_module.Model(
              inputs=inp, outputs=dense_layer(inp))
        return self._dense_model(inputs)

    layer = WrappedDenseLayer(10)
    out = layer(tf.ones(shape=(5, 5)))
    weights = {x.name: x for x in layer.variables}

    # Verify the correct output, losses, + variables were made
    self.assertEqual(weights.keys(), {"dense/bias:0",
                                      "dense/kernel:0"})
    self.assertAllEqual(out, tf.ones(shape=(5, 10)) * 5)
    self.assertAllEqual(tf.add_n(layer.losses), 0.5)

    # Verify reuse by updating the variables then re-running
    weights["dense/kernel:0"].assign(
        tf.ones(shape=(5, 10)) * 2)
    out = layer(tf.ones(shape=(5, 5)))
    self.assertAllEqual(out, tf.ones(shape=(5, 10)) * 10)
    self.assertAllEqual(tf.add_n(layer.losses), 2)

  def test_embedded_keras_model_in_module(self):
    # Test the module shim when embedding a Keras model inside of it
    # And assigning the model to an attribute

    class WrappedDenseLayer(VariableScopeModule):

      def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self._dense_model = None

      def forward_pass(self, inputs):
        if not self._dense_model:
          inp = input_layer_module.Input(shape=inputs.shape)
          dense_layer = core.Dense(
              self.units, name="dense",
              kernel_initializer=tf.compat.v1.ones_initializer(),
              kernel_regularizer="l2")
          self._dense_model = training_module.Model(
              inputs=inp, outputs=dense_layer(inp))
        return self._dense_model(inputs)

    layer = WrappedDenseLayer(10)
    out = layer(tf.ones(shape=(5, 5)))
    weights = {x.name: x for x in layer.variables}

    # Verify the correct output, losses, + variables were made
    self.assertEqual(weights.keys(), {"dense/bias:0",
                                      "dense/kernel:0"})
    self.assertAllEqual(out, tf.ones(shape=(5, 10)) * 5)

    # The module shim will only track regularization losses made by
    # compat.v1.layers and compat.v1.get_variable. Other regularization
    # losses must be tracked by separate user-created mechanisms.
    self.assertEmpty(layer.get_compat_v1_regularization_losses())

    # Verify reuse by updating the variables then re-running
    weights["dense/kernel:0"].assign(
        tf.ones(shape=(5, 10)) * 2)
    out = layer(tf.ones(shape=(5, 5)))
    self.assertAllEqual(out, tf.ones(shape=(5, 10)) * 10)

    # The module shim will only track regularization losses made by
    # compat.v1.layers and compat.v1.get_variable. Other regularization
    # losses must be tracked by separate user-created mechanisms.
    self.assertEmpty(layer.get_compat_v1_regularization_losses())

  def test_training_arg(self):
    # Test the shim when passing in a Keras `training` arg

    class TrainingCheckLayer(variable_scope_shim.VariableScopeLayer):

      def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units

      def forward_pass(self, inputs, training=None):
        if training:
          out = core_layers.dense(inputs, self.units, name="dense_training")
        else:
          out = core_layers.dense(inputs, self.units, name="dense_no_training")
        return out

    layer = TrainingCheckLayer(10)
    layer(tf.ones(shape=(5, 5)), training=True)
    weights = {x.name: x for x in layer.variables}

    # Verify the correct variables were made
    self.assertEqual(weights.keys(),
                     {"dense_training/bias:0", "dense_training/kernel:0"})

    layer = TrainingCheckLayer(10)
    layer(tf.ones(shape=(5, 5)))
    weights = {x.name: x for x in layer.variables}

    # Verify the correct variables were made
    self.assertEqual(weights.keys(),
                     {"dense_no_training/bias:0", "dense_no_training/kernel:0"})

  def test_incorrect_decoration(self):
    # Raise an error if you incorrectly decorate a method
    # that is not a method of a Module, layer, or model:
    @variable_scope_shim.track_tf1_style_variables
    def foo(x):
      return x * 2

    with self.assertRaisesRegex(ValueError, "does not extend"):
      foo(tf.ones(shape=(4, 4)))

if __name__ == "__main__":
  tf.test.main()
