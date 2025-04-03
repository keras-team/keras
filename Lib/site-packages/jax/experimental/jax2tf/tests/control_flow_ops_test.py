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
"""Tests for the jax2tf conversion for control-flow primitives."""

from absl.testing import absltest

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax._src import test_util as jtu
import numpy as np

from jax.experimental.jax2tf.tests import tf_test_util

jax.config.parse_flags_with_absl()


class ControlFlowOpsTest(tf_test_util.JaxToTfTestCase):

  @jtu.ignore_warning(category=UserWarning,
                      message="Explicitly requested dtype .* requested in array is not available")
  def test_cond(self):
    def f_jax(pred, x):
      return lax.cond(pred, lambda t: t + 1., lambda f: f, x)

    self.ConvertAndCompare(f_jax, jnp.bool_(True), 1.)
    self.ConvertAndCompare(f_jax, jnp.bool_(False), 1.)

  @jtu.ignore_warning(category=UserWarning,
                      message="Explicitly requested dtype .* requested in array is not available")
  def test_cond_multiple_results(self):
    def f_jax(pred, x):
      return lax.cond(pred, lambda t: (t + 1., 1.), lambda f: (f + 2., 2.), x)

    self.ConvertAndCompare(f_jax, jnp.bool_(True), 1.)
    self.ConvertAndCompare(f_jax, jnp.bool_(False), 1.)

  @jtu.ignore_warning(category=UserWarning,
                      message="Explicitly requested dtype .* requested in array is not available")
  def test_cond_partial_eval(self):
    def f(x):
      res = lax.cond(True, lambda op: op * x, lambda op: op + x, x)
      return res
    self.ConvertAndCompare(jax.grad(f), 1.)

  @jtu.ignore_warning(category=UserWarning,
                      message="Explicitly requested dtype .* requested in array is not available")
  def test_cond_units(self):
    def g(x):
      return lax.cond(True, lambda x: x, lambda y: y, x)

    self.ConvertAndCompare(g, 0.7)
    self.ConvertAndCompare(jax.grad(g), 0.7)

  @jtu.ignore_warning(category=UserWarning,
                      message="Explicitly requested dtype .* requested in array is not available")
  def test_cond_custom_jvp(self):
    """Conversion of function with custom JVP, inside cond.
    This exercises the custom_jvp_call_jaxpr primitives."""
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

    def g(x):
      return lax.cond(True, f, lambda y: y, x)

    arg = 0.7
    self.TransformConvertAndCompare(g, arg, None)
    self.TransformConvertAndCompare(g, arg, "jvp")
    self.TransformConvertAndCompare(g, arg, "vmap")
    self.TransformConvertAndCompare(g, arg, "jvp_vmap")
    self.TransformConvertAndCompare(g, arg, "grad")
    self.TransformConvertAndCompare(g, arg, "grad_vmap")

  @jtu.ignore_warning(category=UserWarning,
                      message="Explicitly requested dtype .* requested in array is not available")
  def test_cond_custom_vjp(self):
    """Conversion of function with custom VJP, inside cond.
    This exercises the custom_vjp_call_jaxpr primitives."""
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

    def g(x):
      return lax.cond(True, f, lambda y: y, x)

    arg = 0.7
    self.TransformConvertAndCompare(g, arg, None)
    self.TransformConvertAndCompare(g, arg, "vmap")
    self.TransformConvertAndCompare(g, arg, "grad_vmap")

  @jtu.ignore_warning(category=UserWarning,
                      message="Explicitly requested dtype .* requested in array is not available")
  def test_while_single_carry(self):
    """A while with a single carry"""
    def func(x):
      # Equivalent to:
      #      for(i=x; i < 4; i++);
      return lax.while_loop(lambda c: c < 4, lambda c: c + 1, x)

    self.ConvertAndCompare(func, 0)

  def test_while(self):
    # Some constants to capture in the conditional branches
    cond_const = np.ones(3, dtype=np.float32)
    body_const1 = np.full_like(cond_const, 1.)
    body_const2 = np.full_like(cond_const, 2.)

    def func(x):
      # Equivalent to:
      #      c = [1, 1, 1]
      #      for(i=0; i < 3; i++)
      #        c += [1, 1, 1] + [2, 2, 2]
      #
      # The function is set-up so that it captures constants in the
      # body of the functionals. This covers some cases in the representation
      # of the lax.while primitive.
      def cond(idx_carry):
        i, c = idx_carry
        return i < jnp.sum(cond_const)  # Capture cond_const

      def body(idx_carry):
        i, c = idx_carry
        return (i + 1, c + body_const1 + body_const2)

      return lax.while_loop(cond, body, (0, x))

    self.ConvertAndCompare(func, cond_const)

  def test_while_batched_cond(self):
    """A while with a single carry"""
    def product(x, y):
      # Equivalent to "x * y" implemented as:
      #      res = 0.
      #      for(i=0; i < y; i++)
      #         res += x
      return lax.while_loop(lambda idx_carry: idx_carry[0] < y,
                            lambda idx_carry: (idx_carry[0] + 1,
                                               idx_carry[1] + x),
                            (0, 0.))

    # We use vmap to compute result[i, j] = i * j
    xs = np.arange(4, dtype=np.int32)
    ys = np.arange(5, dtype=np.int32)

    def product_xs_y(xs, y):
      return jax.vmap(product, in_axes=(0, None))(xs, y)
    def product_xs_ys(xs, ys):
      return jax.vmap(product_xs_y, in_axes=(None, 0))(xs, ys)

    self.ConvertAndCompare(product_xs_ys, xs, ys)

  @jtu.ignore_warning(category=UserWarning,
                      message="Explicitly requested dtype .* requested in array is not available")
  def test_while_custom_jvp(self):
    """Conversion of function with custom JVP, inside while.
    This exercises the custom_jvp_call_jaxpr primitives."""
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

    def g(x):
      return lax.while_loop(lambda carry: carry[0] < 10,
                            lambda carry: (carry[0] + 1., f(carry[1])),
                            (0., x))

    arg = 0.7
    self.TransformConvertAndCompare(g, arg, None)
    self.TransformConvertAndCompare(g, arg, "jvp")
    self.TransformConvertAndCompare(g, arg, "vmap")
    self.TransformConvertAndCompare(g, arg, "jvp_vmap")


  def test_scan(self):
    def f_jax(xs, ys):
      body_const = np.ones((2, ), dtype=np.float32)  # Test constant capture
      def body(res0, inputs):
        x, y = inputs
        return res0 + x * y, body_const
      return lax.scan(body, 0., (xs, ys))

    arg = np.arange(10, dtype=np.float32)
    self.ConvertAndCompare(f_jax, arg, arg)

  def test_scan_partial_eval(self):
    def f_jax(xs, ys):
      body_const = np.ones((2, ), dtype=np.float32)  # Test constant capture
      def body(res0, inputs):
        x, y = inputs
        return res0 + x * y, body_const
      c_out, _ = lax.scan(body, 0., (xs, ys))
      return c_out

    arg = np.arange(10, dtype=np.float32)
    self.ConvertAndCompare(jax.grad(f_jax), arg, arg)


  def test_scan_custom_jvp(self):
    """Conversion of function with custom JVP, inside scan.
    This exercises the custom_jvp_call_jaxpr primitives."""
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

    def g(x):
      return lax.scan(lambda carry, inp: (carry + f(inp), 0.),
                      np.full(x.shape[1:], 0.),  # Like x w/o leading dim
                      x)[0]

    arg = np.full((5,), 0.7)
    self.TransformConvertAndCompare(g, arg, None)
    self.TransformConvertAndCompare(g, arg, "jvp")
    self.TransformConvertAndCompare(g, arg, "vmap")
    self.TransformConvertAndCompare(g, arg, "jvp_vmap")
    self.TransformConvertAndCompare(g, arg, "grad")
    self.TransformConvertAndCompare(g, arg, "grad_vmap")

  def test_scan_custom_vjp(self):
    """Conversion of function with custom VJP, inside scan.
    This exercises the custom_vjp_call_jaxpr primitives."""
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

    def g(x):
      return lax.scan(lambda carry, inp: (carry + f(inp), 0.),
                      np.full(x.shape[1:], 0.),  # Like x w/o leading dim
                      x)[0]

    arg = np.full((5,), 0.7)
    self.TransformConvertAndCompare(g, arg, None)
    self.TransformConvertAndCompare(g, arg, "vmap")
    self.TransformConvertAndCompare(g, arg, "grad")
    self.TransformConvertAndCompare(g, arg, "grad_vmap")

  def test_scan_remat(self):
    def f_jax(xs):

      @jax.remat
      def body_fun(carry, x):
        return carry * x, xs   # capture xs from the environment
      res1, res2 = lax.scan(body_fun, 0., xs + 1.)
      return jnp.sum(res1) + jnp.sum(res2)

    arg = np.arange(10, dtype=np.float32) + 1.
    self.TransformConvertAndCompare(f_jax, arg, None)
    self.TransformConvertAndCompare(f_jax, arg, "grad")


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
