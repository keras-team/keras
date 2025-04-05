# Copyright 2020 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for JAX2TF converted.

Specific JAX primitive conversion tests are in primitives_test."""
import collections
import contextlib
import math
import os
import re
import unittest

from absl import logging
from absl.testing import absltest, parameterized

import jax
from jax import ad_checkpoint
from jax import dtypes
from jax import export
from jax import lax
from jax import numpy as jnp
from jax import sharding
from jax._src import config
from jax._src import core
from jax._src import source_info_util
from jax._src import test_util as jtu
from jax._src import xla_bridge as xb
from jax.experimental import jax2tf
from jax.experimental.jax2tf.tests import tf_test_util
from jax.experimental.shard_map import shard_map
from jax.experimental import pjit
from jax.sharding import PartitionSpec as P

import numpy as np
import tensorflow as tf

config.parse_flags_with_absl()


class Jax2TfTest(tf_test_util.JaxToTfTestCase):

  def setUp(self):
    super().setUp()
    # One TF device of each device_type
    self.tf_devices = []
    for tf_device in (tf.config.list_logical_devices("TPU") +
                      tf.config.list_logical_devices("GPU") +
                      tf.config.list_logical_devices()):
      if tf_device.device_type == "TPU_SYSTEM":
        continue  # A virtual device
      if all(tf_device.device_type != d.device_type for d in self.tf_devices):
        self.tf_devices.append(tf_device)
    self.warning_ctx = jtu.ignore_warning(
        message="jax2tf.convert with native_serialization=False has been deprecated"
    )
    self.warning_ctx.__enter__()

  def tearDown(self):
    self.warning_ctx.__exit__(None, None, None)
    super().tearDown()

  def test_empty(self):
    f_jax = lambda x, y: x
    self.ConvertAndCompare(f_jax, 0.7, 1)

  def test_sin(self):
    f_tf = jax2tf.convert(jnp.sin)
    x = np.float32(.5)
    sin_x = np.sin(x)
    self.assertAllClose(sin_x, f_tf(x))
    self.assertAllClose(sin_x, tf.function(f_tf, autograph=False,
                                           jit_compile=True)(x))

    tf_preferred_device = (
        tf.config.list_logical_devices("TPU")
        + tf.config.list_logical_devices("GPU")
        + tf.config.list_logical_devices()
    )[0]
    logging.info("Running TF on %s", tf_preferred_device)

    # The following, with jit_compile=False, fails with native serialization
    # because TF executes the function where it is instantiated (For example,
    # XlaCallModule op on CPU). The workaround here is that we can
    # wrap it and add device assignment inside the tf.function.
    @tf.function(autograph=False, jit_compile=False)
    def f_tf_wrapped(x):
      with tf.device(tf_preferred_device.name):
        return f_tf(x)

    with tf.device(tf_preferred_device.name):
      self.assertAllClose(sin_x, f_tf_wrapped(x))

  def test_basics(self):
    f_jax = lambda x: jnp.sin(jnp.cos(x))
    self.ConvertAndCompare(f_jax, 0.7)

  def test_input_output_naming(self):
    @jax2tf.convert
    def f(xs, y):
      return [jnp.add(x, y) for x in xs]

    @tf.function(autograph=False)
    def u(xs, y):
      xs = tf.nest.map_structure(tf.convert_to_tensor, xs)
      with tf.GradientTape() as tape:
        tf.nest.map_structure(tape.watch, xs)
        y = f(xs, y)
        tape.gradient(y, xs)
        return y

    cf = u.get_concrete_function([1., 2., 3.], 4.)
    g = cf.graph
    g.get_operation_by_name("jax2tf_arg_0")
    g.get_operation_by_name("jax2tf_arg_1")
    g.get_operation_by_name("jax2tf_arg_2")
    g.get_operation_by_name("jax2tf_arg_3")
    g.get_operation_by_name("jax2tf_out")
    g.get_operation_by_name("jax2tf_out_1")
    g.get_operation_by_name("jax2tf_out_2")
    with self.assertRaises(KeyError):
      g.get_operation_by_name("jax2tf_arg_4")
    with self.assertRaises(KeyError):
      g.get_operation_by_name("jax2tf_out_3")
    g.get_operation_by_name("jax2tf_vjp/jax2tf_arg_0")
    g.get_operation_by_name("jax2tf_vjp/jax2tf_arg_1")
    g.get_operation_by_name("jax2tf_vjp/jax2tf_arg_2")
    g.get_operation_by_name("jax2tf_vjp/jax2tf_arg_3")
    g.get_operation_by_name("jax2tf_vjp/jax2tf_out")
    g.get_operation_by_name("jax2tf_vjp/jax2tf_out_1")
    g.get_operation_by_name("jax2tf_vjp/jax2tf_out_2")
    g.get_operation_by_name("jax2tf_vjp/jax2tf_out_3")

  def test_pytrees(self):
    # Take and return pytrees
    def f_jax(x: tuple[float, dict[str, float]]) -> tuple[float, dict[str, float]]:
      x_a, x_dict = x
      return x_a * 2., {k: v * 3. for k, v in x_dict.items()}

    x = (.7, {"a": .8, "b": .9})
    self.ConvertAndCompare(f_jax, x)

  def test_variable_input(self):
    f_jax = lambda x: jnp.sin(jnp.cos(x))
    f_tf = jax2tf.convert(f_jax)
    v = tf.Variable(0.7, dtype=jax2tf.dtype_of_val(0.7))
    self.assertIsInstance(f_tf(v), tf.Tensor)
    self.assertAllClose(f_jax(0.7), f_tf(v))

  def test_jit(self):
    f_jax = jax.jit(lambda x: jnp.sin(jnp.cos(x)))
    self.ConvertAndCompare(f_jax, 0.7)

  def test_nested_jit(self):
    f_jax = jax.jit(lambda x: jnp.sin(jax.jit(jnp.cos)(x)))
    x = 0.7
    self.ConvertAndCompare(f_jax, x)

  def test_nested_jit_pytree(self):
    @jax.jit
    def f_jax(xy):
      x, y = xy
      return x + y
    xy = (0.7, 0.8)
    self.ConvertAndCompare(f_jax, xy)

  def test_nested_jit_is_compiled(self):
    # Check that nested jax.jit are compiled with tf.function(jit_compile=True)
    # We do this by looking for the _XlaMustCompile attribute in the function graph
    def has_xla_must_compile(f_tf, x):
      f_conc = tf.function(f_tf, autograph=True).get_concrete_function(tf.convert_to_tensor(x))
      for n in f_conc.graph._nodes_by_id.values():
        try:
          n.get_attr("_XlaMustCompile")
          return True
        except ValueError:
          continue
      return False

    x = np.array(0.7)
    f_no_jit = lambda x: x
    self.assertFalse(has_xla_must_compile(jax2tf.convert(f_no_jit), x))
    f_jit = lambda x: jax.jit(jnp.sin)(x)
    # TODO(b/207464757): TF compilation is disabled
    self.assertFalse(has_xla_must_compile(jax2tf.convert(f_jit), x))

  def test_converts_jax_arrays(self):
    f_tf = tf.function(lambda x: x)
    self.assertEqual(f_tf(jnp.zeros([])).numpy(), 0.)
    self.assertEqual(f_tf(jnp.ones([])).numpy(), 1.)
    f_tf = tf.function(lambda x: x + x)
    self.assertEqual(f_tf(jnp.ones([])).numpy(), 2.)

    # Test with a PmapSharding-sharded Array.
    n = jax.local_device_count()
    mk_sharded = lambda f: jax.pmap(lambda x: x)(f([n]))
    f_tf = tf.function(lambda x: x)
    self.assertAllClose(f_tf(mk_sharded(jnp.zeros)).numpy(),
                        jnp.zeros([n]))
    self.assertAllClose(f_tf(mk_sharded(jnp.ones)).numpy(),
                        jnp.ones([n]))

  @jtu.skip_on_devices("gpu")
  def test_bfloat16_passed_by_tf(self):
    f_jax = lambda a, b: a + b
    f_tf = tf.function(jax2tf.convert(f_jax), autograph=False,
                       input_signature=[tf.TensorSpec([512, 512], tf.bfloat16),
                                        tf.TensorSpec([512, 512], tf.bfloat16)])
    self.assertIsNotNone(f_tf.get_concrete_function())

  @jtu.skip_on_devices("gpu")
  def test_bfloat16_returned_by_jax(self):
    f_jax = lambda a, b: (a + b).astype(jnp.bfloat16)
    f_tf = jax2tf.convert(f_jax)
    self.assertEqual(f_tf(1., 2.).dtype, tf.bfloat16)

  @jtu.skip_on_devices("gpu")
  def test_bfloat16_tf_grad(self):
    f_jax = lambda a, b: a + b

    def _tf_grad(a, b):
      with tf.GradientTape() as tape:
        tape.watch(a)
        result = jax2tf.convert(f_jax)(a, b)
      return result, tape.gradient(result, a)

    f_tf = tf.function(_tf_grad, autograph=False,
                       input_signature=[tf.TensorSpec([512, 512], tf.bfloat16),
                                        tf.TensorSpec([512, 512], tf.bfloat16)])

    self.assertIsNotNone(f_tf.get_concrete_function())

  @jtu.sample_product(
    dtype=[np.int64, np.float64],
    with_function=[True, False],
  )
  def test_converts_64bit(self, dtype=np.int64, with_function=False):
    if not config.enable_x64.value:
      self.skipTest("requires x64 mode")
    big_const = np.full((5,), 2 ** 33, dtype=dtype)
    self.ConvertAndCompare(jnp.sin, big_const)
    f_conv = jax2tf.convert(jnp.sin)
    if with_function:
      f_conv = tf.function(f_conv, autograph=False)
    # We check also when we pass tf.Variable or tf.Tensor into the
    # converted function
    self.assertAllClose(jnp.sin(big_const),
                        f_conv(tf.Variable(big_const)))
    self.assertAllClose(jnp.sin(big_const),
                        f_conv(tf.constant(big_const)))

  def test_64bit_behavior_enable_x64_readme(self):
    # Tests some of the examples from the README
    if not config.enable_x64.value:
      self.skipTest("requires x64 mode")

    # JAX and TF have different default float types if JAX_ENABLE_X64=1
    self.assertEqual(tf.math.sin(3.14).dtype, tf.float32)
    self.assertEqual(jnp.sin(3.14).dtype, jnp.float64)

    # jax2tf.convert has the same behavior as JAX
    self.assertEqual(jax2tf.convert(jnp.sin)(3.14).dtype, tf.float64)
    # The following will compute `sin` in float64.
    self.assertEqual(tf.function(jax2tf.convert(jnp.sin), autograph=False)(tf.Variable(3.14, dtype=tf.float64)).dtype, tf.float64)

    # The following will compute `sin` in float32.
    self.assertEqual(tf.function(jax2tf.convert(jnp.sin), autograph=False)(tf.Variable(3.14)).dtype, tf.float32)

  def test_64bit_behavior_not_enable_x64_readme(self):
    # Tests some of the examples from the README
    if config.enable_x64.value:
      self.skipTest("requires not x64 mode")

    # JAX and TF have same default float types if JAX_ENABLE_X64=0
    self.assertEqual(tf.math.sin(3.14).dtype, tf.float32)
    self.assertEqual(jnp.sin(3.14).dtype, jnp.float32)

    self.assertEqual(tf.math.sin(np.float64(3.14)).dtype, tf.float64)
    # JAX forces values to 32-bit
    self.assertEqual(jnp.sin(np.float64(3.14)).dtype, jnp.float32)

    # jax2tf.convert has the same behavior as JAX
    self.assertEqual(jax2tf.convert(jnp.sin)(3.14).dtype, tf.float32)
    self.assertEqual(jax2tf.convert(jnp.sin)(np.float64(3.14)).dtype, tf.float32)
    self.assertEqual(tf.function(jax2tf.convert(jnp.sin), autograph=False)(tf.Variable(3.14, dtype=tf.float64)).dtype, tf.float32)

  def test_function(self):
    f_jax = jax.jit(lambda x: jnp.sin(jnp.cos(x)))
    self.ConvertAndCompare(f_jax, 0.7)

  @jtu.sample_product(with_function=[False, True])
  def test_gradients_disabled(self, with_function=False):
    if tf.version.VERSION.split(".") <= ["2", "17", "0"]:
      self.skipTest("This test works only with newer versions of TF")
    f_tf = jax2tf.convert(jnp.tan, with_gradient=False)
    if with_function:
      f_tf = tf.function(f_tf, autograph=False)
    x = tf.ones([])

    # With tf.function the error is raised when we evaluate f_tf(x), in
    # eager mode when we evaluate tape.gradient(y, x)
    with self.assertRaisesRegex(LookupError,
                                "Gradient explicitly disabled.*The jax2tf-converted function does not support gradients"):
      with tf.GradientTape() as tape:
        tape.watch(x)
        y = f_tf(x)
        _ = tape.gradient(y, x)

  @jtu.sample_product(with_function=[False, True])
  def test_gradients(self, with_function=True):
    def f(x, y):
      return x * x, x * y
    f_tf = jax2tf.convert(f, with_gradient=True)
    if with_function:
      f_tf = tf.function(f_tf, autograph=False)
    default_float_type = jax2tf.dtype_of_val(4.)
    x = tf.Variable(4., dtype=jax2tf.dtype_of_val(4.))
    y = tf.Variable(5., dtype=default_float_type)
    with tf.GradientTape(persistent=True) as tape:
      u, v = f_tf(x, y)

    self.assertAllClose(2. * 4., tape.gradient(u, x))
    self.assertAllClose(0., tape.gradient(u, y))
    self.assertAllClose(5., tape.gradient(v, x))
    self.assertAllClose(4., tape.gradient(v, y))

  def test_higher_order_gradients(self):
    f = lambda x: x ** 3
    f_tf = jax2tf.convert(f)
    x = tf.Variable(4.0, dtype=tf.float32)  # Create a Tensorflow variable initialized to 4.0
    with tf.GradientTape() as t2:
      with tf.GradientTape() as t1:
        y = f_tf(x)

      # Compute the gradient inside the outer `t2` context manager
      # which means the gradient computation is differentiable as well.
      dy_dx = t1.gradient(y, x)
    d2y_dx2 = t2.gradient(dy_dx, x)

    self.assertAllClose(np.float32(48.), dy_dx.numpy())
    self.assertAllClose(np.float32(24.), d2y_dx2.numpy())

  @jtu.sample_product(with_function=[False, True])
  def test_gradients_pytree(self, with_function=False):
    def f(xy: tuple[float, float]) -> dict[str, float]:
      x, y = xy
      return dict(one=x * x, two=x * y)

    f_tf = jax2tf.convert(f, with_gradient=True)
    if with_function:
      f_tf = tf.function(f_tf, autograph=False)
    default_float_dtype = jax2tf.dtype_of_val(4.)
    x = tf.Variable(4., dtype=default_float_dtype)
    y = tf.Variable(5., dtype=default_float_dtype)
    with tf.GradientTape(persistent=True) as tape:
      uv = f_tf((x, y))

    self.assertAllClose(2. * 4., tape.gradient(uv["one"], x))
    self.assertAllClose(0., tape.gradient(uv["one"], y))
    self.assertAllClose(5., tape.gradient(uv["two"], x))
    self.assertAllClose(4., tape.gradient(uv["two"], y))

  def test_custom_pytree_readme(self):
    # Code examples from README.md
    class CustomPair:
      def __init__(self, a, b):
        self.a = a
        self.b = b

    jax.tree_util.register_pytree_node(CustomPair,
                                       lambda x: ((x.a, x.b), None),
                                       lambda _, ab: CustomPair(*ab))
    def f_jax(pair: CustomPair):
      return np.float32(2.) * pair.a + np.float32(3.) * pair.b
    f_tf = jax2tf.convert(f_jax)

    x = CustomPair(np.float32(4.), np.float32(5.))
    res_jax = f_jax(x)
    # TF execution works as long as JAX can flatten the arguments and results
    res_tf = f_tf(x)
    self.assertAllClose(res_jax, res_tf.numpy())
    res_tf_2 = tf.function(f_tf, autograph=False, jit_compile=True)(x)
    self.assertAllClose(res_jax, res_tf_2)

    # wrapped TF function to use only standard containers
    def f_tf_wrapped(a, b):
      return f_tf(CustomPair(a, b))

    # Try to put into SavedModel
    my_model = tf.Module()
    # Save a function that can take scalar inputs.
    my_model.f = tf.function(f_tf_wrapped, autograph=False,
                             input_signature=[tf.TensorSpec([], tf.float32),
                                              tf.TensorSpec([], tf.float32)])
    model_dir = os.path.join(absltest.get_default_test_tmpdir(), str(id(my_model)))
    tf.saved_model.save(my_model, model_dir,
                        options=tf.saved_model.SaveOptions(experimental_custom_gradients=True))

    # Restoring (note: the restored model does *not* require JAX to run, just XLA).
    restored_model = tf.saved_model.load(model_dir)
    def restored_f(pair: CustomPair):
      return restored_model.f(pair.a, pair.b)

    res_tf_3 = restored_f(x)
    self.assertAllClose(res_jax, res_tf_3)
    grad_jax = jax.grad(f_jax)(x)

    x_v = [tf.Variable(x.a), tf.Variable(x.b)]
    with tf.GradientTape() as tape:
      res = f_tf_wrapped(*x_v)

      grad_tf = tape.gradient(res, x_v)
    self.assertAllClose(grad_jax.a, grad_tf[0])
    self.assertAllClose(grad_jax.b, grad_tf[1])

  @jtu.sample_product(with_function=[False, True])
  def test_gradients_with_ordered_dict_input(self, with_function=True):
    def f(inputs):
      out = 0.0
      for v in inputs.values():
        out += jnp.sum(v)
      return out

    f_tf = jax2tf.convert(f, with_gradient=True)
    if with_function:
      f_tf = tf.function(f_tf, autograph=False)
    default_float_type = jax2tf.dtype_of_val(4.)
    x = tf.Variable([4.], dtype=default_float_type)
    y = tf.Variable([4., 5.], dtype=default_float_type)
    inputs = collections.OrderedDict()
    inputs['r'] = x
    inputs['d'] = y
    with tf.GradientTape(persistent=True) as tape:
      u = f_tf(inputs)

    self.assertAllClose(np.array([1.]), tape.gradient(u, x).numpy())
    self.assertAllClose(np.array([1., 1.]), tape.gradient(u, y).numpy())

  @jtu.sample_product(with_function=[False, True])
  def test_gradients_with_custom_jvp(self, with_function=True):
    """Check gradients, for a function with custom JVP."""
    @jax.custom_jvp
    def f(x):
      return x * x

    @f.defjvp
    def f_jvp(primals, tangents):
      # 3 * x * x_t
      x, = primals
      x_dot, = tangents
      primal_out = f(x)
      tangent_out = 3. * x * x_dot
      return primal_out, tangent_out

    self.assertAllClose(4. * 4., f(4.))
    self.assertAllClose(3. * 4., jax.grad(f)(4.))

    f_tf = jax2tf.convert(f, with_gradient=True)
    if with_function:
      f_tf = tf.function(f_tf, autograph=False)
    self.assertAllClose(4. * 4., f_tf(4.))
    x = tf.Variable(4., dtype=jax2tf.dtype_of_val(4.))
    with tf.GradientTape() as tape:
      tape.watch(x)
      y = f_tf(x)

    self.assertAllClose(4. * 4., y)
    self.assertAllClose(3. * 4., tape.gradient(y, x))

  @jtu.sample_product(with_function=[False, True])
  def test_gradients_with_custom_vjp(self, with_function=True):
    """Check gradients, for a function with custom VJP."""
    @jax.custom_vjp
    def f(x):
      return x * x

    # f_fwd: a -> (b, residual)
    def f_fwd(x):
      return f(x), 3. * x
    # f_bwd: (residual, CT b) -> [CT a]
    def f_bwd(residual, ct_b):
      return residual * ct_b,

    f.defvjp(f_fwd, f_bwd)

    self.assertAllClose(4. * 4., f(4.))
    self.assertAllClose(3. * 4., jax.grad(f)(4.))

    f_tf = jax2tf.convert(f, with_gradient=True)
    if with_function:
      f_tf = tf.function(f_tf, autograph=False)
    self.assertAllClose(4. * 4., f_tf(4.))
    x = tf.Variable(4., dtype=jax2tf.dtype_of_val(4.))
    with tf.GradientTape() as tape:
      tape.watch(x)
      y = f_tf(x)

    self.assertAllClose(4. * 4., y)
    self.assertAllClose(3. * 4., tape.gradient(y, x))

  def test_gradient_with_float0_intermediate(self):
    # Gradient over integer-argument functions
    def f(x, y):  # x is an int, y is a float
      return 2 * x + y

    def g(x):  # x: f32
      return 2. * f(3 * x.astype("int32"), x * 4.)

    x = 2.
    grad_g = jax.grad(g)
    self.ConvertAndCompare(grad_g, x)

  def test_gradient_with_float0_result(self):
    # Gradient over integer-argument functions, with float0 result
    def f(x, y):  # x is an int, y is a float
      return 2 * x + y

    def g(x):  # x: i32
      return jnp.sum(2. * f(3 * x, 4. * jnp.array(x, jnp.dtype("float32"))))

    grad_g = jax.grad(g, allow_int=True)
    x = 2
    d_dx_jax = grad_g(x)
    d_dx_tf = jax2tf.convert(grad_g)(x)
    self.assertEqual(d_dx_jax.dtype, dtypes.float0)
    self.assertAllClose(jnp.zeros(np.shape(d_dx_jax), np.bool_),
                        d_dx_tf.numpy())

    shape = (3, 4)
    x = np.ones(shape, dtype=np.int32)
    d_dx_jax = grad_g(x)
    d_dx_tf = jax2tf.convert(grad_g)(x)
    self.assertEqual(d_dx_jax.dtype, dtypes.float0)
    self.assertAllClose(jnp.zeros(np.shape(d_dx_jax), np.bool_),
                        d_dx_tf.numpy())

  @jtu.sample_product(with_function=[False, True])
  def test_gradients_unused_argument_readme(self, with_function=False):
    # x1 and x3 are not used. x3 has integer type.
    def fn(x0, x1, x2, x3):
      return x0 * 0. + x2 * 2.

    xs = [tf.Variable(x) for x in [10., 11., 12., 13]]
    with tf.GradientTape(persistent=True) as tape:
      res = fn(*xs)

    g_tf_native = tape.gradient(res, xs)
    self.assertAllClose(g_tf_native[0].numpy(), np.float32(0.))
    self.assertIsNone(g_tf_native[1])
    self.assertAllClose(g_tf_native[2].numpy(), np.float32(2.))
    self.assertIsNone(g_tf_native[3])

    g_tf_native_0 = tape.gradient(res, xs,
                                  unconnected_gradients=tf.UnconnectedGradients.ZERO)
    self.assertAllClose(g_tf_native_0[0].numpy(), np.float32(0.))
    self.assertAllClose(g_tf_native_0[1].numpy(), np.float32(0.))
    self.assertAllClose(g_tf_native_0[2].numpy(), np.float32(2.))
    self.assertAllClose(g_tf_native_0[3].numpy(), np.int32(0))

    # Now with jax2tf.convert
    with tf.GradientTape(persistent=True) as tape:
      conv_fn = jax2tf.convert(fn, with_gradient=True)
      if with_function:
        conv_fn = tf.function(conv_fn, autograph=False)
      res = conv_fn(*xs)

    g_jax2tf = tape.gradient(res, xs)
    # Returns: 0., 0., 2., None
    # Note that the gradient for x1 is 0.
    self.assertAllClose(g_jax2tf[0].numpy(), np.float32(0.))
    self.assertAllClose(g_jax2tf[1].numpy(), np.float32(0.))
    self.assertAllClose(g_jax2tf[2].numpy(), np.float32(2.))
    self.assertIsNone(g_jax2tf[3])

    g_jax2tf = tape.gradient(res, xs,
                               unconnected_gradients=tf.UnconnectedGradients.ZERO)
    self.assertAllClose(g_jax2tf[0].numpy(), np.float32(0.))
    self.assertAllClose(g_jax2tf[1].numpy(), np.float32(0.))
    self.assertAllClose(g_jax2tf[2].numpy(), np.float32(2.))
    self.assertAllClose(g_jax2tf[3].numpy(), np.int32(0))

  @jtu.sample_product(with_function=[False, True])
  def test_gradients_int_argument(self, with_function=False):
    # https://github.com/jax-ml/jax/issues/6975
    # Also issue #6975.
    # An expanded version of test_gradients_unused_argument
    state = dict(
        float_used=np.array([0.7, 0.9], dtype=np.float32),
        float_passthrough=np.float16(1.),
        float_unused=np.array([1.1, 2.2, 3.3], dtype=np.float32),
        int_used=np.int16(5),
        int_passthrough=np.int8(7),
        int_unused=np.array([1, 2, 3], dtype=np.uint32),
        bool_used=np.array([True, False, False, True], dtype=np.bool_),
        bool_passthrough=np.array([True, False, False, True, False], dtype=np.bool_),
        bool_unused=np.array([[True, False], [False, True]], dtype=np.bool_),
    )
    def jax_f(state):
      res = dict(state,
                 float_used=2. * state["float_used"],
                 int_used=3 * state["int_used"],
                 bool_used=(state["bool_used"] == state["bool_used"]))
      del res["float_unused"]
      del res["int_unused"]
      del res["bool_unused"]
      return res

    args = (state,)
    res_jax = jax_f(*args)
    # Native JAX AD
    vjp_jax_fun, args_vjp = tf_test_util.TransformJaxVJP(jax_f, args, res_jax)
    grad_jax, = vjp_jax_fun(*args_vjp)

    def compare_with_overrides(*, what, expected, **expected_overrides):
      what_keys = set(what.keys())
      expected_keys = set(expected.keys())
      self.assertEqual(what_keys, expected_keys)
      for k, w in what.items():
        e = expected[k]
        if k in expected_overrides:
          if expected_overrides[k] == "ZERO":
            e = np.zeros_like(w)
          elif expected_overrides[k] == "ZERO_BOOL":
            e = np.zeros(np.shape(w), dtype=np.bool_)
          elif expected_overrides[k] == "ONE":
            e = np.ones_like(w)
          else:
            e = expected_overrides[k]

        if e is None:
          self.assertIsNone(w, msg=k)
        else:
          self.assertIsNotNone(w, msg=k)
        w = w.numpy() if isinstance(w, tf.Tensor) else e
        e = e.numpy() if isinstance(e, tf.Tensor) else e
        try:
          self.assertAllClose(e, w, err_msg=k)
        except:
          print(f"Failed at {k}")
          raise

    # compare_with_overrides(g_jax, {},
    #   bool_passthrough=np.zeros(state["bool_passthrough"].shape, dtype=dtypes.float0),
    #   bool_unused=np.zeros(state["bool_unused"].shape, dtype=dtypes.float0),
    #   bool_used=np.zeros(state["bool_used"].shape, dtype=dtypes.float0),
    #   float_passthrough=np.ones_like(state["float_passthrough"]),
    #   float_unused=np.zeros_like(state["float_unused"]),
    #   float_used=np.ones_like(state["float_used"]) * np.array(2., dtype=state["float_used"].dtype),
    #   int_passthrough=np.zeros(state["int_passthrough"].shape, dtype=dtypes.float0),
    #   int_unused=np.zeros(state["int_unused"].shape, dtype=dtypes.float0),
    #   int_used=np.zeros(state["int_used"].shape, dtype=dtypes.float0))

    # Now native TF gradients, only to test how native TF AD works
    _, (grad_tf_0,) = tf_test_util.ComputeTfValueAndGrad(
        jax_f, args, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    compare_with_overrides(what=grad_tf_0,
                           expected=grad_jax,
                           float_unused="ZERO",
                           bool_used="ZERO", bool_passthrough="ONE", bool_unused="ZERO",
                           int_used="ZERO", int_passthrough="ONE", int_unused="ZERO")

    _, (grad_tf_None,) = tf_test_util.ComputeTfValueAndGrad(
        jax_f, args,
        unconnected_gradients=tf.UnconnectedGradients.NONE)
    compare_with_overrides(what=grad_tf_None,
                           expected=grad_tf_0,
                           float_unused=None, int_used=None, int_unused=None,
                           bool_used=None, bool_unused=None)

    f_tf_jax = jax2tf.convert(jax_f)
    if with_function:
      f_tf_jax = tf.function(f_tf_jax, autograph=False)

    _, (grad_tf_jax_0,) = tf_test_util.ComputeTfValueAndGrad(f_tf_jax, args)
    # Same results as TF native AD with tf.UnconnectedGradients.ZERO
    compare_with_overrides(what=grad_tf_jax_0,
                           expected=grad_tf_0,
                           int_passthrough="ZERO", bool_passthrough="ZERO")

    _, (grad_tf_jax_None,) = tf_test_util.ComputeTfValueAndGrad(
        f_tf_jax, args,
        unconnected_gradients=tf.UnconnectedGradients.NONE)
    compare_with_overrides(what=grad_tf_jax_None,
                           expected=grad_tf_0,
                           int_used=None, int_passthrough=None, int_unused=None,
                           bool_unused=None, bool_used=None, bool_passthrough=None)

    # Not convert the JAX gradient function
    tf_vjp_jax_fun = jax2tf.convert(vjp_jax_fun)
    grad_tf_vjp_jax, = tf_vjp_jax_fun(*args_vjp)
    compare_with_overrides(what=grad_tf_vjp_jax,
                           expected=grad_tf_0,
                           bool_passthrough="ZERO_BOOL",
                           bool_unused="ZERO_BOOL", bool_used="ZERO_BOOL",
                           int_passthrough="ZERO_BOOL", int_unused="ZERO_BOOL",
                           int_used="ZERO_BOOL")

  def test_readme_gradient_int(self):
    x = np.array(2, dtype=np.int16)

    def f_jax(x):  # x: int16
      return x.astype(np.float32) * 2.

    print(jax.grad(f_jax, allow_int=True)(x))
    # returns a special `float0`: array((b'',), dtype=[('float0', 'V')])

    print(jax2tf.convert(jax.grad(f_jax, allow_int=True))(x))
    # returns a 0 with same shape as x, but with dtype int32

    def f_tf(x):  # x: int16
      return tf.cast(x, tf.float32) * 2.

    xv = tf.Variable(x)
    with tf.GradientTape(persistent=True) as tape:
      print(tape.gradient(f_tf(xv), xv))
      # returns None
      print(tape.gradient(f_tf(xv), xv,
                          unconnected_gradients=tf.UnconnectedGradients.ZERO))
      # returns 0 with the same shape and dtype as x

  def test_convert_argument_non_callable_error(self):
    with self.assertRaisesRegex(TypeError, "Expected a callable value"):
      jax2tf.convert(5.)

  def test_convert_argument_non_tensor_error(self):
    with self.assertRaisesRegex(TypeError,
                                "Argument.*is not a valid JAX type"):
      jax2tf.convert(lambda x: x)(lambda y: y)

  def test_argument_eager_tensor(self):
    x = jax2tf.convert(jnp.sin)(1.)
    jax2tf.convert(jnp.cos)(x)  # No error

  def test_checkpoint_wrapper_types(self):
    m = tf.Module()
    m.a = [tf.Module(), tf.Module()]
    m.b = (tf.Module(), tf.Module())
    m.c = {'a': tf.Module(), 'b': tf.Module()}
    self.assertNotEqual(type(m.a), list)
    self.assertNotEqual(type(m.b), tuple)
    self.assertNotEqual(type(m.c), dict)
    self.assertLen(jax.tree_util.tree_leaves(m.a), 2)
    self.assertLen(jax.tree_util.tree_leaves(m.b), 2)
    self.assertLen(jax.tree_util.tree_leaves(m.c), 2)

  @unittest.skip("Test fails at head")
  def test_issue_10586(self):

    class JaxModule(tf.Module):
      def __init__(self):
        self._params = {'w': tf.Variable(tf.ones([784, 10]), name='w'),
                        'b': tf.Variable(tf.ones([10]), name='b')}

      def __call__(self, x):
        return jax2tf.convert(lambda p, x: x @ p['w'] + p['b'])(self._params, x)

    net = JaxModule()
    images = tf.ones([1, 784])

    with tf.GradientTape() as tape:
      loss = tf.reduce_sum(net(images))
    params = tape.watched_variables()
    grads = tape.gradient(loss, params)
    for var, grad in zip(params, grads):
      self.assertEqual(var.shape, grad.shape, msg=var.name)

  def test_custom_jvp(self):
    """Conversion of function with custom JVP"""

    @jax.custom_jvp
    def f(x):
      return x * x

    @f.defjvp
    def f_jvp(primals, tangents):
      x, = primals
      x_dot, = tangents
      primal_out = f(x)
      tangent_out = 3. * x * x_dot
      return primal_out, tangent_out

    arg = 0.7
    self.TransformConvertAndCompare(f, arg, None)
    self.TransformConvertAndCompare(f, arg, "jvp")
    self.TransformConvertAndCompare(f, arg, "vmap")
    self.TransformConvertAndCompare(f, arg, "jvp_vmap")
    self.TransformConvertAndCompare(f, arg, "grad")
    self.TransformConvertAndCompare(f, arg, "grad_vmap")

  def test_custom_vjp(self):
    """Conversion of function with custom VJP"""

    @jax.custom_vjp
    def f(x):
      return x * x

    # f_fwd: a -> (b, residual)
    def f_fwd(x):
      return f(x), 3. * x

    # f_bwd: (residual, CT b) -> [CT a]
    def f_bwd(residual, ct_b):
      return residual * ct_b,

    f.defvjp(f_fwd, f_bwd)
    arg = 0.7
    self.TransformConvertAndCompare(f, arg, None)
    self.TransformConvertAndCompare(f, arg, "vmap")
    self.TransformConvertAndCompare(f, arg, "grad")
    self.TransformConvertAndCompare(f, arg, "grad_vmap")

  def test_remat(self):
    def f(x1):
      x2 = jnp.sin(x1)
      x3 = jnp.sin(x2)
      x4 = jnp.sin(x3)
      return x4
    remat_f = ad_checkpoint.checkpoint(f)

    # The computation of grad_f computes "sin" 5 times, 3 for the forward pass
    # and then to rematerialize "x2" and "x3" in the backward pass.
    arg = np.array(3.)
    f_tf = jax2tf.convert(jax.grad(remat_f))
    f_tf_hlo = self.TfToHlo(f_tf, arg)
    if config.remat_opt_barrier.value:
      self.assertRegex(f_tf_hlo, r"opt-barrier")
    else:
      self.assertRegex(f_tf_hlo,
                       r'transpose/jax2tf_f_/jvp/checkpoint/cond/branch_1_fun/Sin')

  def test_remat_free_var(self):
    def f(x):
      y = 2 * x

      @ad_checkpoint.checkpoint
      def g():
        return y

      return g()
    arg = 3.
    self.TransformConvertAndCompare(f, arg, None)
    self.TransformConvertAndCompare(f, arg, "grad")

  def test_checkpoint_name(self):
    def f_jax(x):
      return ad_checkpoint.checkpoint_name(jnp.sin(x), "sin")
    jax2tf.convert(f_jax)(1.)  # No error.

  def test_convert_of_nested_independent_jit(self):
    def func(x):
      def inner1(y):
        return x + y
      # The JIT does not have data dependency
      return jax.jit(inner1)(1.)

    jax2tf.convert(func)(2.)

  def test_convert_of_nested_dependent_jit(self):
    def func(x):
      def inner1(y):
        return x + y
      # The JIT does have data dependency
      return jax.jit(inner1)(x)

    jax2tf.convert(func)(2.)  # No error

  def test_jit_unused(self):
    def f_jax(x, y_unused):
      return x * np.float32(2.)
    x, y_unused = np.float32(5.), np.arange(7, dtype=np.int32)
    res_tf = jax2tf.convert(jax.jit(f_jax, keep_unused=False))(x, y_unused)
    self.assertAllClose(f_jax(x, None), res_tf)

  @parameterized.named_parameters(
      dict(testcase_name=mode, mode=mode)
      for mode in ("eager", "graph", "compiled"))
  def test_jit_unused_grad(self, mode="eager"):
    def f_jax(x, y_unused):
      return x * np.float32(2.)

    x, y_unused = np.float32(5.), np.arange(7, dtype=np.int32)
    res_jax = f_jax(x, y_unused)
    f_tf = jax2tf.convert(jax.jit(f_jax, keep_unused=False))

    x_tf, y_unused_tf = tf.constant(x), tf.constant(y_unused)
    def grad_tf(x, y_unused):
      with tf.GradientTape() as tape:
        tape.watch(x)
        tape.watch(y_unused)
        res_tf = f_tf(x, y_unused)
        grad_tf_x, grad_tf_y = tape.gradient(res_tf, (x, y_unused))
      return res_tf, grad_tf_x, grad_tf_y

    if mode == "graph":
      grad_tf = tf.function(grad_tf, autograph=False)
    elif mode == "compiled":
      grad_tf = tf.function(grad_tf, autograph=False, jit_compile=True)

    res_tf, grad_tf_x, grad_tf_y = grad_tf(x_tf, y_unused_tf)
    self.assertAllClose(res_jax, res_tf)
    self.assertAllClose(np.float32(2.), grad_tf_x)
    self.assertIsNone(grad_tf_y)

  def test_nested_convert_error(self):
    def outer(y):
      return jax2tf.convert(jnp.sin)(y)  # Inner convert takes tracer args
    with self.assertRaisesRegex(
        ValueError, "convert must be used outside all JAX transformations"):
      jax2tf.convert(outer)(np.ones((4,), dtype=np.float32))

  def test_nested_convert_error_non_tracer(self):
    """The inner convert takes non-tracer arguments"""
    def outer(y):
      sin_1 = jax2tf.convert(jnp.sin)(1.)  # Inner convert takes non-tracer arg
      return y + sin_1

    with self.assertRaisesRegex(
        ValueError, "convert must be used outside all JAX transformations"):
      jax2tf.convert(outer)(2.)

  @jtu.sample_product(transform=["jit", "jvp", "grad", "vmap"])
  def test_convert_under_transform_error(self, transform="vmap"):
    def outer(y):
      return jax2tf.convert(jnp.sin)(y)  # Inner convert takes tracer args

    with self.assertRaisesRegex(
        ValueError, "convert must be used outside all JAX transformations"):
      self.TransformConvertAndCompare(outer, np.ones((4,)), transform)

  @jtu.sample_product(transform=["jit", "jvp", "grad", "vmap"])
  def test_convert_under_transform_error_non_tracer(self, transform="vmap"):
    def outer(y):
      sin_1 = jax2tf.convert(jnp.sin)(1.)  # Inner convert takes non-tracer arg
      return y + sin_1

    with self.assertRaisesRegex(
        ValueError, "convert must be used outside all JAX transformations"):
      self.TransformConvertAndCompare(outer, np.ones((4,)), transform)

  def test_name_scope(self):
    def run_tf():
      @jax.named_call
      def my_test_function_jax(x):
        return x * x

      def caller_jax(x):
        return my_test_function_jax(jnp.sin(x))

      out = jax2tf.convert(caller_jax, with_gradient=False)(2.)
      return out
    if config.jax2tf_default_native_serialization.value:
      self.assertIn("my_test_function_jax/mul", self.TfToHlo(run_tf))
    else:
      graph_def = str(tf.function(run_tf, autograph=False).get_concrete_function().graph.as_graph_def())
      if "my_test_function_jax/pjit_multiply_/Mul" not in graph_def:
        self.assertIn("my_test_function_jax/jit_multiply_/Mul", graph_def)

  def test_bfloat16_constant(self):
    # Re: https://github.com/jax-ml/jax/issues/3942
    def jax_fn_scalar(x):
      x = x.astype(jnp.bfloat16)
      x *= 2.
      return x

    def jax_fn_array(x):
      x = x.astype(jnp.bfloat16)
      x *= np.array([1.5, 2.5, 3.5], jnp.bfloat16)
      return x

    tf_fn_scalar = jax2tf.convert(jax_fn_scalar)
    self.assertAllClose(tf_fn_scalar(1.375).numpy(), jnp.bfloat16(2.750))

    tf_fn_array = jax2tf.convert(jax_fn_array)
    self.assertAllClose(
        tf_fn_array(np.array([3, 4, 5])), np.array([4.5, 10, 17.5],
                                                   jnp.bfloat16))

  def test_shared_constants(self):
    # Check that the constants are shared properly in converted functions
    # See https://github.com/jax-ml/jax/issues/7992.
    if config.jax2tf_default_native_serialization.value:
      raise unittest.SkipTest("shared constants tests not interesting for native serialization")
    const = np.random.uniform(size=256).astype(np.float32)  # A shared constant
    def f(x):
      return x + const + const + const + const

    f_tf_consts = self.FindLargeTfConstants(jax2tf.convert(f), const)
    self.assertLen(f_tf_consts, 1)

  def test_shared_constants_under_cond(self):
    # Check that the constants are shared properly in converted functions
    # See https://github.com/jax-ml/jax/issues/7992.
    if config.jax2tf_default_native_serialization.value:
      raise unittest.SkipTest("shared constants tests not interesting for native serialization")
    const_size = 512
    const = np.random.uniform(size=const_size).astype(np.float32)  # A shared constant
    x = np.ones((const_size,), dtype=np.float32)
    def f1(x):
      # Ensure that we first see the constants in the inside jaxpr
      return lax.cond(x[0] >= 0., lambda x: x + const, lambda x: x * const, x) + const
    def f2(x):
      return f1(x) + const  # The extra const should not cost anything
    f1_consts = self.FindLargeTfConstants(jax2tf.convert(f1), x, at_least=const_size)
    f2_consts = self.FindLargeTfConstants(jax2tf.convert(f2), x, at_least=const_size)
    self.assertLen(f2_consts, len(f1_consts))

  def test_shared_constants_under_scan(self):
    # See https://github.com/jax-ml/jax/issues/7992.
    if config.jax2tf_default_native_serialization.value:
      raise unittest.SkipTest("shared constants tests not interesting for native serialization")
    const_size = 512
    const = np.random.uniform(size=const_size).astype(np.float32)  # A shared constant
    xs = np.ones((8, const_size), dtype=np.float32)
    def f1(xs):
      res, _ = lax.scan(lambda carry, x: (carry + x + const, None),
                        jnp.zeros((const_size,), dtype=np.float32), xs)
      return res

    def f2(xs):
      return f1(xs) + const  # The extra const should not be saved

    f1_consts = self.FindLargeTfConstants(jax2tf.convert(f1), xs, at_least=const_size)
    f2_consts = self.FindLargeTfConstants(jax2tf.convert(f2), xs, at_least=const_size)
    self.assertLen(f2_consts, len(f1_consts))

  def test_shared_constants_under_jit(self):
    # We do not share constants under jit.
    if config.jax2tf_default_native_serialization.value:
      raise unittest.SkipTest("shared constants tests not interesting for native serialization")
    const = np.random.uniform(size=(16, 16)).astype(np.float32)  # A shared constant
    @jax.jit
    def g_jit(x):
      return x * const
    def f(x):
      return g_jit(x) + const + const

    f_tf_graph_consts = self.FindLargeTfConstants(jax2tf.convert(f), const)
    self.assertLen(f_tf_graph_consts, 1)

  def test_shared_constants_randint(self):
    # randint has the property that the TF lowering of the randbits_p
    # primitive generates constants that did not exist in the Jaxpr. As such
    # it has created new errors related to the sharing of the constants.
    if config.jax2tf_default_native_serialization.value:
      raise unittest.SkipTest("shared constants tests not interesting for native serialization")

    key = jax.random.PRNGKey(42)

    def f_nested_jax(x):
      # Lowering this will generate a tf.constant(shape=(1,), dtype=np.int32)
      # that was not already in the Jaxpr, and hence JAX did not get a chance
      # to share.
      return x + jax.random.randint(key, shape=x.shape,
                                    minval=0, maxval=100, dtype=np.int32)
    def f_jax(x):
      res = lax.cond(x[0] >= 2, lambda: f_nested_jax(x), lambda: f_nested_jax(x))
      res += lax.while_loop(lambda x: f_nested_jax(x)[0] <= 0, f_nested_jax, x)
      # We also generate tf.while in the batching rule for cond
      res += jax.vmap(lambda x: lax.cond(x[0] >= 2,
                                         lambda: f_nested_jax(x),
                                         lambda: f_nested_jax(x)))(jnp.stack([x, x]))
      res += f_nested_jax(x)
      return res

    # Must be odd to trigger the failure
    x = np.array([123, 456, 789], dtype=np.int32)

    f_tf = tf.function(jax2tf.convert(f_jax), autograph=False)
    res_tf = f_tf(x)
    self.assertAllClose(res_tf, f_jax(x))

  def test_weak_types(self):
    mul = jax.jit(jnp.multiply)
    # The value `2` here should be weakly typed, and should not lead to
    # promotion.
    tf_fn = jax2tf.convert(lambda x: mul(x, 2.))
    self.assertAllClose(tf_fn(tf.constant(1.375, tf.bfloat16)).numpy(),
                        jnp.bfloat16(2.750))

  @jtu.sample_product(with_function=[False, True])
  def test_kwargs(self, with_function=False):
    # Re: https://github.com/jax-ml/jax/issues/6791
    def f_jax(*, x):
      return jnp.sum(x)
    f_tf = jax2tf.convert(f_jax)
    if with_function:
      f_tf = tf.function(f_tf, autograph=False)
    self.assertAllClose(
      f_tf(x=np.zeros(3, dtype=np.float32)),  # Call with kwargs.
      np.zeros((), dtype=np.float32))

  @jtu.sample_product(with_function=[False, True])
  def test_grad_kwargs(self, with_function=False):
    # Re: https://github.com/jax-ml/jax/issues/6791
    x = (np.zeros(3, dtype=np.float32),
         np.zeros(4, dtype=np.float32))
    def f_jax(*, x=(1., 2.)):
      return jnp.sum(x[0]) + 2. * jnp.sum(x[1])
    f_tf = jax2tf.convert(f_jax)
    if with_function:
      f_tf = tf.function(f_tf, autograph=False)
    xv = tf.nest.map_structure(tf.Variable, x)
    with tf.GradientTape() as tape:
      res = f_tf(x=xv)
    grad_tf = tape.gradient(res, xv)
    self.assertAllClose((np.full_like(x[0], fill_value=1.),
                         np.full_like(x[1], fill_value=2.)),
                        (grad_tf[0].numpy(), grad_tf[1].numpy()))

  def test_device_array_arg(self):
    self.ConvertAndCompare(jnp.sin, jnp.zeros((2, 3), jnp.float32))

  def test_randint(self):
    def randint():
      return jax.random.randint(
          jax.random.PRNGKey(42), shape=(), minval=0, maxval=1)

    self.ConvertAndCompare(randint)

  def test_op_metadata_simple(self):
    self.skipTest("include_xla_op_metadata not yet enabled")
    # A simple example
    # The user_frame is used to compute line numbers for ops in the test.
    user_frame = source_info_util.user_frame(source_info_util.current())
    def f_simple(x):
      return jnp.sin(x)

    x = np.ones((2, 3), np.float32)
    self.CheckOpMetadata(
        f_simple, x,
        [tf_test_util.OpMetadataGraph(tf_type="Sin",
                                      source_file=__file__,
                                      source_line=user_frame.start_line + 2,
                                      op_name="jax2tf(f_simple)/sin",
                                      op_type="sin")
         ]
    )

  def test_op_metadata_sub_jit(self):
    self.skipTest("include_xla_op_metadata not yet enabled")
    # Calling a jitted-function
    # The user_frame is used to compute line numbers for ops in the test.
    user_frame = source_info_util.user_frame(source_info_util.current())
    def f_callee(x):
      return jnp.cos(x)
    def f_caller(x):
      y = jnp.tanh(x)
      z = jax.jit(f_callee)(y)
      return jnp.sin(z)

    x = np.ones((2, 3), np.float32)

    self.CheckOpMetadata(
        f_caller, x,
        [tf_test_util.OpMetadataGraph(tf_type="Tanh",
                                      source_file=__file__,
                                      source_line=user_frame.start_line + 4,
                                      op_name="jax2tf(f_caller)/tanh",
                                      op_type="tanh"),
         tf_test_util.OpMetadataGraph(tf_type="Cos",
                                      source_file=__file__,
                                      source_line=user_frame.start_line + 2,
                                      op_name="jax2tf(f_caller)/jit(f_callee)/cos",
                                      op_type="cos"),
         tf_test_util.OpMetadataGraph(tf_type="Sin",
                                      source_file=__file__,
                                      source_line=user_frame.start_line + 6,
                                      op_name="jax2tf(f_caller)/sin",
                                      op_type="sin"),
         ]
    )

  def test_op_metadata_named(self):
    self.skipTest("include_xla_op_metadata not yet enabled")
    # Calling a jax.named_call
    # The user_frame is used to compute line numbers for ops in the test.
    user_frame = source_info_util.user_frame(source_info_util.current())
    def f_callee(x):
      return jnp.cos(x)
    def f_caller(x):
      y = jnp.tanh(x)
      z = jax.named_call(f_callee, name="callee")(y)
      return jnp.sin(z)

    x = np.ones((2, 3), np.float32)

    self.CheckOpMetadata(
        f_caller, x,
        [tf_test_util.OpMetadataGraph(tf_type="Tanh",
                                      source_file=__file__,
                                      source_line=user_frame.start_line + 4,
                                      op_name="jax2tf(f_caller)/tanh",
                                      op_type="tanh"),
         tf_test_util.OpMetadataGraph(tf_type="Cos",
                                      source_file=__file__,
                                      source_line=user_frame.start_line + 2,
                                      op_name="jax2tf(f_caller)/named(callee)/cos",
                                      op_type="cos"),
         tf_test_util.OpMetadataGraph(tf_type="Sin",
                                      source_file=__file__,
                                      source_line=user_frame.start_line + 6,
                                      op_name="jax2tf(f_caller)/sin",
                                      op_type="sin"),
         ]
    )

  def test_op_metadata_while_and_cond(self):
    self.skipTest("include_xla_op_metadata not yet enabled")
    # An example with while and cond
    # The user_frame is used to compute line numbers for ops in the test.
    user_frame = source_info_util.user_frame(source_info_util.current())
    def f_while_cond(x):
      def body_fun(i_acc):
        i, acc = i_acc
        return (i + 1,
                (jnp.cos(acc) +
                 lax.cond(jnp.mod(i, 2) == 0,
                          lambda acc: jnp.sin(acc),
                          lambda acc: acc,
                          acc)))

      _, acc = lax.while_loop(
          lambda i_acc: i_acc[0] <= 5,
          body_fun, (0, x))
      return acc

    x = np.ones((2, 3), np.float32)
    self.CheckOpMetadata(
        f_while_cond, x,
        [tf_test_util.OpMetadataGraph(tf_type="Cos",
                                      source_file=__file__,
                                      source_line=user_frame.start_line + 5,
                                      op_name="jax2tf(f_while_cond)/while/body/cos",
                                      op_type="cos"),
         tf_test_util.OpMetadataGraph(tf_type="Sin",
                                      source_file=__file__,
                                      source_line=user_frame.start_line + 7,
                                      op_name="jax2tf(f_while_cond)/while/body/branch_1_fun/sin",
                                      op_type="sin"),
         tf_test_util.OpMetadataGraph(tf_type="FloorMod",
                                      source_file=__file__,
                                      source_line=user_frame.start_line + 6,
                                      op_name="jax2tf(f_while_cond)/while/body/rem",
                                      op_type="rem"),
         ]
    )

  def test_op_metadata_batched_while(self):
    self.skipTest("include_xla_op_metadata not yet enabled")
    # An example with while and cond
    # The user_frame is used to compute line numbers for ops in the test.
    user_frame = source_info_util.user_frame(source_info_util.current())
    @jax.vmap
    def f_while(x):
      def body_fun(carry):
        new_carry = jnp.sin(carry)  # We look for "sin" in the graph
        return new_carry

      _, carry = lax.while_loop(
          lambda carry: jnp.all(carry <= x),  # We look for "le" in the graph
          body_fun, x)
      return carry

    shape = (3, 2)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)

    jax_comp = jax.jit(f_while).lower(x).compiler_ir('hlo')
    backend = xb.get_backend()
    modules = backend.compile(jax_comp).hlo_modules()
    jax_opt_hlo = modules[0].to_string()
    print(f"JAX OPT HLO = {jax_opt_hlo}")

    self.CheckOpMetadata(
        f_while, x,
        [tf_test_util.OpMetadataGraph(tf_type="Sin",
                                      source_file=__file__,
                                      source_line=user_frame.start_line + 4,
                                      op_name="jax2tf(f_while)/while/body/sin",
                                      op_type="sin"),
         tf_test_util.OpMetadataGraph(tf_type="LessEqual",
                                      source_file=__file__,
                                      source_line=user_frame.start_line + 8,
                                      op_name="jax2tf(f_while)/while/body_pred/le",
                                      op_type="le"),
         ]
    )

  def test_op_metadata_disabled(self):
    self.skipTest("include_xla_op_metadata not yet enabled")
    def f_simple(x):
      return jnp.sin(x)

    x = np.ones((2, 3), np.float32)
    self.CheckOpMetadata(
        f_simple, x,
        [],
        include_xla_op_metadata=False
    )

  def assertAllOperationStartWith(self, g: tf.Graph, scope_name: str):
    """Assert all operations name start with ```scope_name```.

    Also the scope_name only occur one time.
    """
    result = g.get_operations()
    if not result:
      self.fail("result is empty.")
    for op in result:
      logging.info("tf op.name = %s", op.name)
      if not op.name.startswith(scope_name):
        self.fail(f"{op.name} does not start with {scope_name}.")

  def test_name_scope_polymorphic(self):
    if (config.jax2tf_default_native_serialization.value and
        not config.dynamic_shapes.value):
      self.skipTest("shape polymorphism but --jax_dynamic_shapes is not set.")

    def func_jax(x, y):
      return jnp.sin(x) + jnp.cos(y)

    func_tf = jax2tf.convert(
        func_jax, polymorphic_shapes="(b,...)", with_gradient=True)

    outer_scope = "output_a"

    g = tf.Graph()
    with g.as_default() as g:
      with tf.name_scope(outer_scope):
        x = tf.Variable(
            tf.zeros(shape=(1, 5), dtype=tf.dtypes.float32), name="x")
        y = tf.compat.v1.placeholder(tf.dtypes.float32, (None, 5), "y")
        _ = func_tf(x, y)
    self.assertAllOperationStartWith(g, outer_scope)

    # wrap tf.function
    g2 = tf.Graph()
    with g2.as_default() as g:
      with tf.name_scope(outer_scope):
        x = tf.Variable(
            tf.zeros(shape=(1, 5), dtype=tf.dtypes.float32), name="x")
        y = tf.compat.v1.placeholder(tf.dtypes.float32, (None, 5), "y")
        _ = tf.function(func_tf, jit_compile=True, autograph=False)(x, y)
    self.assertAllOperationStartWith(g2, outer_scope)

  def test_name_scope_cond(self):
    def f(x):
      def f_pos(x):
        with jax.named_scope("jax_f_pos"):
          return lax.cond(x < 1., jnp.cos, jnp.sin, x)

      with jax.named_scope("jax_f_outer"):
        return lax.cond(x > 0., f_pos, lambda x: x, x)

    @tf.function(jit_compile=True, autograph=False)
    def outer_forward():
      with tf.name_scope("tf_outer_forward"):
        x = 0.5
        f_tf = jax2tf.convert(f)
        _ = f_tf(x)

    g = outer_forward.get_concrete_function().graph
    self.assertAllOperationStartWith(g, "tf_outer_forward")
    for func in g._functions.values():
      self.assertAllOperationStartWith(
          func.graph, "tf_outer_forward/jax2tf_f_/jax_f_outer")

    x = tf.Variable(0.5, name="tf_outer_back/x")

    @tf.function(jit_compile=True, autograph=False)
    def outer_back():
      with tf.name_scope("tf_outer_back"):
        f_tf = jax2tf.convert(f)
        with tf.GradientTape() as tape:
          res_tf = f_tf(x)
          _ = tape.gradient(res_tf, x)

    g = outer_back.get_concrete_function().graph
    self.assertAllOperationStartWith(g, "tf_outer_back")
    for func in g._functions.values():
      self.assertAllOperationStartWith(func.graph, "tf_outer_back")

  def test_name_scope_while_loop(self):
    def f(x):
      with tf.name_scope("outer_scope"):
        def condition(x):
          return jnp.sum(x, keepdims=False) < 100
        def body(x):
          return jnp.add(x, 2.0)

        result = jax.lax.while_loop(condition, body, x)
        return result

    tf_f = tf.function(jax2tf.convert(f), jit_compile=True, autograph=False)
    g = tf_f.get_concrete_function(tf.zeros((1, 3))).graph

    for func in g._functions.values():
      for op in func.graph.get_operations():
        if op.name.count(f"outer_scope/jax2tf_{f.__name__}_/while") > 1:
          self.fail(
              "tf graph has repeated name issue on when converting lax.while to tf.while."
              f"See op.name = : {op.name}")

  @parameterized.named_parameters(
      dict(testcase_name=(
          f"{'with_mesh_' if with_mesh else ''}"
          f"2={transform2 if transform2 != 'none' else ''}"
          f"_1={transform1 if transform1 != 'none' else ''}"
          f"{'_nullary' if nullary else ''}"),
          with_mesh=with_mesh, transform1=transform1,
          transform2=transform2, nullary=nullary)
      # Test transform2(transform1(func)
      for transform1 in [
          "none",
          "jit",
          "pjit", "pjit_in_shardings_None", "pjit_in_shardings_P",
          "pjit_in_shardings_Sharding", "shard_map", "pmap"]
      for transform2 in (
          ["none", "pjit_in_shardings_None", "pjit_in_shardings_P",
           "pjit_in_shardings_Sharding"]
      )
      # Whether the function can be nullary
      for nullary in (
          # To reduce the number of tests
          [True, False] if transform2 == "none" else
          [False])
      # Whether we use a "with mesh"
      for with_mesh in (
          [True] if (transform1 not in ["base", "jit", "pjit"] or
                     transform2 != "none") else
          [False, True])
  )
  def test_cross_platform(self, with_mesh=True, transform1="pjit_in_shardings_P",
                          transform2="pjit_in_shardings_P", nullary=False):
    # Tests cross-lowering for
    #  with mesh:
    #   transform2(transform1(func))
    if transform2 == "none" and (
        transform1 == "shard_map" or
        transform1 in ["pjit_in_shardings_P", "pjit_in_shardings_Sharding"] and nullary):
      raise unittest.SkipTest("Skip because must have pjit at top level")

    x = np.ones((4, 6), dtype=np.float32)
    mesh = sharding.Mesh(jax.devices()[:1], ("a",))
    # cummax has distinctive lowering for TPU, using a reduce-window op
    func = lambda x: lax.cummax(x, axis=0, reverse=False)
    # For shard_map we cannot use cummax :-( because it does not have a
    # replication rule. But we use lax.all_gather which on TPU is lowered with
    # an all-gather op
    func_shard_map = lambda x: lax.all_gather(x, 'a', axis=1, tiled=True)

    def apply_transform(func, transform: str):
      transformed_func = dict(
          none=func,
          jit=jax.jit(func),
          jit_in_shardings_None=jax.jit(func, in_shardings=None),
          jit_in_shardings_P=jax.jit(func, in_shardings=(P("a"),)),
          jit_in_shardings_Sharding=jax.jit(
              func, in_shardings=(sharding.NamedSharding(mesh, P("a")),)),
          pjit=pjit.pjit(func),
          pjit_in_shardings_None=pjit.pjit(func, in_shardings=None,
                                           out_shardings=None),
          pjit_in_shardings_P=pjit.pjit(func, in_shardings=(P("a"),),
                                        out_shardings=P("a")),
          pjit_in_shardings_Sharding=pjit.pjit(
              func,
              in_shardings=(sharding.NamedSharding(mesh, P("a")),),
              out_shardings=sharding.NamedSharding(mesh, P("a"))),
          shard_map=(
              shard_map(func, mesh, in_specs=(P("a", None),),
                        out_specs=P("a", None))),
          pmap=jax.pmap(func, in_axes=0, out_axes=0),
      )[transform]
      return transformed_func

    transformed1_func = apply_transform(
        (func_shard_map if transform1 == "shard_map" else func),
        transform1)
    assert transform2 not in ["shard_map"]
    transformed2_func = apply_transform(transformed1_func, transform2)

    if transform1 == "pmap":
      x = x.reshape((1, -1))  # Since we use 1 device
    if not nullary:
      func_to_convert = transformed2_func
      args = [x]
    else:
      func_to_convert = lambda: transformed2_func(jnp.ones(x.shape,
                                                           dtype=x.dtype))
      args = []

    if transform1 == "pmap":
      if nullary:
        raise unittest.SkipTest("Cannot lower nested pmap: jit-of-pmap warning")
      raise unittest.SkipTest("TODO: figure out how to invoke pmap from TF")

    f_tf = jax2tf.convert(func_to_convert,
                          native_serialization=True,
                          native_serialization_platforms=('tpu',))
    f_tf = tf.function(f_tf, jit_compile=True, autograph=False)
    with contextlib.ExitStack() as stack:
      if with_mesh:
        stack.enter_context(mesh)
      # Run the JAX native version, to check it works, and to fill caches.
      _ = func_to_convert(*args)
      exported = export.export(
          (jax.jit(func_to_convert) if not hasattr(func_to_convert, "trace") else func_to_convert),
          platforms=("tpu",)
      )(*(core.ShapedArray(a.shape, a.dtype) for a in args))

    if transform1 == "shard_map":
      self.assertIn("stablehlo.all_gather", str(exported.mlir_module()))
    else:
      self.assertIn("stablehlo.reduce_window", str(exported.mlir_module()))

  def test_cross_platform_error(self):
    f_tf = jax2tf.convert(jnp.sin, native_serialization=True,
                          native_serialization_platforms=('tpu',))
    x = np.float32(.5)
    if jtu.test_device_matches(["tpu"]):
      self.assertAllClose(jnp.sin(x), f_tf(x))
    else:
      # We can construct the tf.Graph
      f_tf_fun = tf.function(f_tf, jit_compile=True, autograph=False)
      graph_def = f_tf_fun.get_concrete_function(x).graph.as_graph_def()
      self.assertIn("XlaCallModule", str(graph_def))
      with self.assertRaisesRegex(tf.errors.NotFoundError,
          "The current platform .* is not among the platforms required by the module"):
        f_tf(x)

  @jtu.ignore_warning(message="using native_serialization_platforms without native_serialization")
  def test_native_parameters_for_non_native(self):
    # We can use the native_serialization_platforms even for non-native
    # serialization.
    f_tf = jax2tf.convert(jnp.sin,
                          native_serialization_platforms=('cpu',))
    x = np.float32(.5)
    # Run the TF code on CPU
    tf_cpus = tf.config.list_logical_devices("CPU")
    self.assertNotEmpty(tf_cpus)
    with tf.device(tf_cpus[0]):
      self.assertAllClose(jnp.sin(x), f_tf(x))

    f_tf = jax2tf.convert(jnp.sin,
                          native_serialization_disabled_checks=(
                            jax2tf.DisabledSafetyCheck.platform(),))
    self.assertAllClose(jnp.sin(x), f_tf(x))

  def test_native_serialization_grad(self):
    # Check that the grad function uses the same native serialization parameters
    # as the primal function.
    f_tf = jax2tf.convert(jnp.sin, native_serialization=True,
                          native_serialization_platforms=('tpu',))
    x = np.arange(4, dtype=np.float32)
    x_v = tf.Variable(x)

    @tf.function(autograph=False)
    def f_grad_tf(x_v):
      with tf.GradientTape() as tape:
        tape.watch(x_v)
        res_tf = f_tf(x_v)
        return tape.gradient(res_tf, x_v)

    # Make sure that we have 2x XlaCallModule in the graph of the gradient
    # function
    f_grad_tf_fun = tf.function(f_grad_tf, autograph=False)
    graph_def = f_grad_tf_fun.get_concrete_function(x).graph.as_graph_def()
    logging.info("Found graph_def: %s", graph_def)
    self.assertLen(re.findall(r'op:\s*"XlaCallModule"', str(graph_def)), 2)

    if not jtu.test_device_matches(["tpu"]):
      with self.assertRaisesRegex(
          tf.errors.NotFoundError,
          r"The current platform .* is not among the platforms required by the module: \[TPU\]"):
        f_grad_tf(x_v)

  def test_effects_error(self):
    def f_jax(x):
      jax.debug.print("{}", x)
      return jnp.sin(x)

    with self.assertRaisesRegex(NotImplementedError,
                                "serialization of host_callbacks is not yet implemented"):
      jax2tf.convert(f_jax, native_serialization=True)(np.float32(42.))

    def f_ordered_jax(x):
      jax.debug.print("{}", x, ordered=True)
      return jnp.sin(x)

    with self.assertRaisesRegex(NotImplementedError,
                                "serialization of host_callbacks is not yet implemented"):
      jax2tf.convert(f_ordered_jax, native_serialization=True)(np.float32(42.))

  def test_tuple_args(self):
    # On TPU if we have more than 2000 arguments, we pass them as a tuple.
    # This is a compiler option, and should have no effect on lowering.
    if not jtu.test_device_matches(["tpu"]):
      raise unittest.SkipTest("Test enabled on TPU only")
    def f_jax(*many_args):
      acc = 0.
      for a in many_args:
        acc += a
      return acc

    many_args = [np.float32(i) for i in range(2001)]
    # Test that we do set lowered.compile_args[tuple_args]
    lowered = jax.jit(f_jax).lower(*many_args)
    self.assertTrue(lowered._lowering.compile_args["tuple_args"])
    res = jax2tf.convert(f_jax, native_serialization=True)(*many_args)
    self.assertAllClose(f_jax(*many_args), res)

  @jtu.ignore_warning(message="Calling from_dlpack with a DLPack tensor",
                      category=DeprecationWarning)
  def test_nested_convert(self):
    # Test call sequence: convert -> call_tf -> convert.

    @jax.jit
    def f_jax(x):
      return x + 1

    inputs = np.ones((10), dtype=np.float32)

    res = f_jax(inputs)

    f_tf = jax2tf.convert(f_jax, native_serialization=True)
    self.assertAllClose(res, f_tf(inputs))

    f_jax_nested = jax2tf.call_tf(f_tf)
    self.assertAllClose(res, f_jax_nested(inputs))

    f_tf_nested = jax2tf.convert(f_jax_nested, native_serialization=True)
    self.assertAllClose(res, f_tf_nested(inputs))

  def test_multi_platform(self):
    if config.enable_x64.value:
      self.skipTest("TODO: enable when we can handle i64 platform_index_argument")
    # Checks that we dispatch from TF to the proper JAX platform lowering.

    # We add a different value to it: cpu=2., tpu=3., cuda=.4, rocm=5.
    _testing_multi_platform_to_add = dict(cpu=2., tpu=3., cuda=4., rocm=5.)

    def f_jax(x):
      return x + lax.platform_dependent(
        tpu=lambda: _testing_multi_platform_to_add["tpu"],
        cuda=lambda: _testing_multi_platform_to_add["cuda"],
        rocm=lambda: _testing_multi_platform_to_add["rocm"],
        default=lambda: _testing_multi_platform_to_add["cpu"]
      )

    x = np.float32(.42)
    f_tf = jax2tf.convert(
      f_jax,
      native_serialization=True,
      native_serialization_platforms=("cpu", "cuda", "tpu"))
    for tf_device in self.tf_devices:
      logging.info(
        f"Running on tf_device = {tf_device} of device_type = {tf_device.device_type}")
      with tf.device(tf_device):
        res = f_tf(x)
      tf_device_jax_platform = dict(
        CPU="cpu", GPU="cuda", TPU="tpu"
      )[tf_device.device_type]
      self.assertAllClose(
        res,
        x + _testing_multi_platform_to_add[tf_device_jax_platform])

  def test_dot_algorithm(self):
    # ref: https://github.com/jax-ml/jax/issues/24236
    if tf.version.VERSION.split(".") <= ["2", "18", "0"]:
      self.skipTest("Because of an XLA bug this test segfaults with TF v2.18.0")

    if jtu.test_device_matches(["tpu"]):
      algorithm = "BF16_BF16_F32"
    else:
      algorithm = "F32_F32_F32"

    def f_jax(x):
      return jax.lax.dot(x, x, precision=algorithm)

    f_tf = jax2tf.convert(f_jax, native_serialization=True)
    f_tf(np.ones((128, 128), dtype=np.float32))  # no crash

  def test_dot_algorithm_non_native_unsupported(self):
    def f_jax(x):
      return jax.lax.dot(x, x, precision="F32_F32_F32")

    x = np.ones((128, 128), dtype=np.float32)
    with self.assertRaisesRegex(NotImplementedError,
                                "Unsupported precision in dot_general"):
      jax2tf.convert(f_jax, native_serialization=False)(x)


@jtu.with_config(jax_enable_custom_prng=True)
class Jax2tfWithCustomPRNGTest(tf_test_util.JaxToTfTestCase):
  def setUp(self):
    super().setUp()
    self.warning_ctx = jtu.ignore_warning(
        message="jax2tf.convert with native_serialization=False has been deprecated"
    )
    self.warning_ctx.__enter__()

  def tearDown(self):
    self.warning_ctx.__exit__(None, None, None)
    super().tearDown()

  def test_key_argument(self):
    func = lambda key: jax.random.uniform(key, ())
    key = jax.random.PRNGKey(0)
    key_raw = jax.random.key_data(key)
    with self.assertWarnsRegex(FutureWarning, "Raw arrays as random keys.*"):
      tf_result = jax2tf.convert(func)(key_raw)
    jax_result = func(key)
    self.assertEqual(tf_result, jax_result)

  def test_key_from_seed(self):
    func = lambda seed: jax.random.uniform(jax.random.PRNGKey(seed), ())
    seed = 1701
    tf_result = jax2tf.convert(func)(seed)
    jax_result = func(seed)
    self.assertEqual(tf_result, jax_result)

  def test_key_closure(self):
    def func():
      # Include nontrivial shape operations to catch tracing bugs.
      key = global_key.reshape(1).squeeze()
      return jax.random.uniform(key)
    global_key = jax.random.PRNGKey(0)
    tf_result = jax2tf.convert(func)()
    jax_result = func()
    self.assertEqual(tf_result, jax_result)

class Jax2TfVersioningTest(tf_test_util.JaxToTfTestCase):
  # Use a separate test case with the default jax_serialization_version
  def setUp(self):
    self.use_max_serialization_version = False
    super().setUp()

  @jtu.ignore_warning(
      message="jax2tf.convert with native_serialization=False has been deprecated"
  )
  def test_simple(self):
    self.ConvertAndCompare(jnp.sin, 0.7)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
