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

from absl.testing import absltest
import os

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from jax.experimental import jax2tf
from jax.experimental.jax2tf.tests import tf_test_util
from jax._src import test_util as jtu

jax.config.parse_flags_with_absl()


class SavedModelTest(tf_test_util.JaxToTfTestCase):

  def setUp(self):
    super().setUp()
    self.warning_ctx = jtu.ignore_warning(
        message="jax2tf.convert with native_serialization=False is deprecated"
    )
    self.warning_ctx.__enter__()

  def tearDown(self):
    self.warning_ctx.__exit__(None, None, None)
    super().tearDown()

  def test_eval(self):
    f_jax = jax.jit(lambda x: jnp.sin(jnp.cos(x)))
    model = tf.Module()
    model.f = tf.function(jax2tf.convert(f_jax),
                          autograph=False,
                          input_signature=[tf.TensorSpec([], tf.float32)]
                          )
    x = np.array(0.7, dtype=jnp.float32)
    self.assertAllClose(model.f(x), f_jax(x))
    restored_model = tf_test_util.SaveAndLoadModel(model)
    self.assertAllClose(restored_model.f(x), f_jax(x))

  def test_gradient(self):
    """Save and restore the custom gradient."""
    @jax.custom_jvp
    def f_jax(x):
      return x * x

    @f_jax.defjvp
    def f_jax_jvp(primals, tangents):
      # 3 * x * x_t
      x, = primals
      x_dot, = tangents
      primal_out = f_jax(x)
      tangent_out = x * x_dot * 3.
      return primal_out, tangent_out

    model = tf.Module()
    model.f = tf.function(jax2tf.convert(f_jax, with_gradient=True),
                          autograph=False,
                          input_signature=[tf.TensorSpec([], tf.float32)])
    x = np.array(0.7, dtype=jnp.float32)
    self.assertAllClose(model.f(x), f_jax(x))
    restored_model = tf_test_util.SaveAndLoadModel(model)
    xv = tf.Variable(x)
    self.assertAllClose(restored_model.f(x), f_jax(x))
    with tf.GradientTape() as tape:
      y = restored_model.f(xv)
    self.assertAllClose(tape.gradient(y, xv).numpy(),
                        jax.grad(f_jax)(x))

  def test_gradient_nested(self):
    """Save and restore the custom gradient, when combined with other TF code."""
    @jax.custom_jvp
    def f_jax(x):
      return x * x

    @f_jax.defjvp
    def f_jax_jvp(primals, tangents):
      # 3 * x * x_t
      x, = primals
      x_dot, = tangents
      primal_out = f_jax(x)
      tangent_out = x * x_dot * 3.
      return primal_out, tangent_out

    model = tf.Module()
    # After conversion, we wrap with some pure TF code
    model.f = tf.function(lambda x: tf.math.sin(jax2tf.convert(f_jax, with_gradient=True)(x)),
                          autograph=False,
                          input_signature=[tf.TensorSpec([], tf.float32)])
    f_jax_equiv = lambda x: jnp.sin(f_jax(x))
    x = np.array(0.7, dtype=jnp.float32)
    self.assertAllClose(model.f(x), f_jax_equiv(x))
    restored_model = tf_test_util.SaveAndLoadModel(model)
    xv = tf.Variable(x)
    self.assertAllClose(restored_model.f(x), f_jax_equiv(x))
    with tf.GradientTape() as tape:
      y = restored_model.f(xv)
    self.assertAllClose(tape.gradient(y, xv).numpy(),
                        jax.grad(f_jax_equiv)(x))

  def test_gradient_disabled(self):
    f_jax = lambda x: x * x

    model = tf.Module()
    model.f = tf.function(jax2tf.convert(f_jax, with_gradient=False),
                          autograph=False,
                          input_signature=[tf.TensorSpec([], tf.float32)])
    x = np.array(0.7, dtype=jnp.float32)
    self.assertAllClose(model.f(x), f_jax(x))
    restored_model = tf_test_util.SaveAndLoadModel(model)
    xv = tf.Variable(0.7, dtype=jnp.float32)
    self.assertAllClose(restored_model.f(x), f_jax(x))

    with self.assertRaisesRegex(LookupError,
                                "Gradient explicitly disabled.*The jax2tf-converted function does not support gradients"):
      with tf.GradientTape():
        _ = restored_model.f(xv)

  def test_save_without_gradients(self):
    f_jax = lambda x: x * x

    x = np.array(0.7, dtype=jnp.float32)
    model = tf.Module()
    model.f = tf.function(jax2tf.convert(f_jax, with_gradient=True),
                          autograph=False,
                          input_signature=[tf.TensorSpec(x.shape, x.dtype)])

    self.assertAllClose(model.f(x), f_jax(x))
    restored_model = tf_test_util.SaveAndLoadModel(model,
                                                   save_gradients=False)
    self.assertAllClose(restored_model.f(x), f_jax(x))

    xv = tf.Variable(x)
    with tf.GradientTape():
      _ = restored_model.f(xv)
      # TODO: clean this up b/191117111: it should fail with a clear error
      # The following results in a confusing error:
      # TypeError: An op outside of the function building code is being passed
      # a "Graph" tensor. It is possible to have Graph tensors
      # leak out of the function building context by including a
      # tf.init_scope in your function building code.
      # For example, the following function will fail:
      #   @tf.function
      #   def has_init_scope():
      #     my_constant = tf.constant(1.)
      #     with tf.init_scope():
      #       added = my_constant * 2
      # The graph tensor has name: args_0:0
      # g = tape.gradient(res, xv)
    #self.assertAllClose(g.numpy(), jax.grad(f_jax)(x))

  def test_save_without_embedding_params(self):
    def model_jax(params, inputs):
      return params[0] + params[1] * inputs

    params = (np.array(1.0, dtype=jnp.float32),
              np.array(2.0, dtype=jnp.float32))
    params_vars = tf.nest.map_structure(tf.Variable, params)

    prediction_tf = lambda x: jax2tf.convert(model_jax)(params_vars, x)

    model = tf.Module()
    model._variables = tf.nest.flatten(params_vars)
    model.f = tf.function(prediction_tf, jit_compile=True, autograph=False)

    x = np.array(0.7, dtype=jnp.float32)
    self.assertAllClose(model.f(x), model_jax(params, x))
    restored_model = tf_test_util.SaveAndLoadModel(model,
                                                   save_gradients=False)
    self.assertAllClose(restored_model.f(x), model_jax(params, x))


  def test_save_grad_integers(self):
    # https://github.com/jax-ml/jax/issues/7123
    # In the end this is a test that does not involve JAX at all
    batch_size = 5
    state = np.array([1], dtype=np.int32)  # Works if float32
    params = np.ones((3, 3), dtype=np.float32)

    # params: f32[3, 3], state: i32[1]
    # returns f32[5, 2] constant, and the state
    def tf_predict(params, state):
      # state = tf.cast(state, tf.float32)
      # Setup a custom-gradient, like jax2tf would
      @tf.custom_gradient
      def converted_fun_with_custom_gradient(params, state):
        res_out = tf.zeros((batch_size, 2), dtype=tf.float32)
        state_out = state  # tf.zeros((4, 4), np.int32),

        return ((res_out, state_out), converted_grad_fn)

      def converted_grad_fn(res_out_ct, state_out_ct, variables=None):
        # The gradients for params and the state
        return tf.zeros(params.shape, dtype=params.dtype), state_out_ct

      res, state_out = converted_fun_with_custom_gradient(params, state)
      # state_out = tf.cast(state_out, tf.int32)
      return res, state_out

    # Compute the gradient before saving. This works!
    params_v = tf.Variable(params)
    with tf.GradientTape() as tape:
      preds = tf_predict(params_v, state)[0]
      loss = tf.reduce_mean(preds)
      g = tape.gradient(loss, params_v)
    self.assertAllClose(g.numpy(), np.zeros(params.shape, dtype=params.dtype))

    # TF -> SavedModel
    model = tf.Module()
    model.fn = tf.function(tf_predict, autograph=False)
    model.fn.get_concrete_function(
        tf.TensorSpec(params.shape, params.dtype),
        tf.TensorSpec(state.shape, state.dtype))
    save_dir = os.path.join(absltest.get_default_test_tmpdir(), str(id(model)))
    options = tf.saved_model.SaveOptions(experimental_custom_gradients=True)
    _ = tf.saved_model.save(model, save_dir, options=options)
    restored_module = tf.saved_model.load(save_dir)

    # It seems that saving and reloading is important
    restored_fn = restored_module.fn

    # Compute the gradients after saving and restoring. Fails!
    with tf.GradientTape() as tape:
      preds = restored_fn(params_v, state)[0]
      loss = tf.reduce_mean(preds)
      g = tape.gradient(loss, params_v)
    self.assertAllClose(g.numpy(), np.zeros(params.shape, dtype=params.dtype))

  def _compare_with_saved_model(self, f_jax, *args):
    # Certain ops are converted to ensure an XLA context, e.g.,
    # tf.gather, so that the index-out-of-bounds behavior matches that of
    # JAX. We check that this information is preserved through a savedmodel
    f_tf = jax2tf.convert(f_jax)
    res = f_tf(*args)
    restored_f, _ = tf_test_util.SaveAndLoadFunction(f_tf, input_args=args)
    res_restored = restored_f(*args)
    self.assertAllClose(res, res_restored)

  def test_pytree(self):
    def f_jax(params, x):
      # params is a dict
      return x @ params["w"] + params["b"]

    x = np.ones((2, 3), dtype=np.float32)
    params = dict(w=np.ones((3, 4), dtype=np.float32),
                  b=np.ones((2, 4), dtype=np.float32))
    res_jax = f_jax(params, x)
    f_tf = jax2tf.convert(f_jax)

    res_tf = f_tf(params, x)
    self.assertAllClose(res_jax, res_tf.numpy())

    restored_f, restored_model = tf_test_util.SaveAndLoadFunction(f_tf, input_args=(params, x),
                                                                  save_gradients=True)
    self.assertAllClose(restored_f(params, x).numpy(), res_tf.numpy())

    # Gradients for the converted function
    params_v = tf.nest.map_structure(tf.Variable, params)
    with tf.GradientTape() as tape:
      res = f_tf(params_v, x)
      loss = tf.reduce_sum(res)
      g_tf = tape.gradient(loss, params_v)

    params_v = tf.nest.map_structure(tf.Variable, params)
    with tf.GradientTape() as tape:
      res = restored_f(params_v, x)
      loss = tf.reduce_sum(res)
      g_restored_f = tape.gradient(loss, params_v)

    self.assertAllClose(g_tf["w"].numpy(), g_restored_f["w"].numpy())
    self.assertAllClose(g_tf["b"].numpy(), g_restored_f["b"].numpy())


  def test_xla_context_preserved_slice(self):
    arr = np.arange(10, dtype=np.float32)
    def f_jax(arr):
      return lax.dynamic_slice(arr, [100], [1])  # out of bounds, should return the last element
    self._compare_with_saved_model(f_jax, arr)

  def test_xla_context_preserved_gather(self):
    def f_jax(arr):
      return arr[100]  # out of bounds, should return the last element
    arr = np.arange(10, dtype=np.float32)
    self._compare_with_saved_model(f_jax, arr)

  # Test does not work on GPU/TPU; would need something like TPU inference
  # converter to separate the model on what needs to run on CPU or accelerator.
  @jtu.skip_on_devices("gpu", "tpu")
  def test_tf_mix_jax_with_uncompilableble(self):
    """Show how to combine TF-uncompilableble code with compiled JAX-converted code."""
    def tf_fn(x_str, compute_tf_fn=lambda x: x):
      # Some TF preprocessing code that cannot be compiled with XLA because it
      # uses strings.
      numbers_f32 = tf.strings.to_number(x_str, out_type=tf.float32)
      numbers_f16 = tf.cast(numbers_f32, tf.float16)
      return compute_tf_fn(numbers_f16)

    x_str = np.array(["3.14", "2.78"])

    # Test that we get an error if we try to TF-compile `tf_fn`
    with self.assertRaisesRegex(
        Exception,
        "Detected unsupported operations when trying to compile graph"):
      tf.function(tf_fn, jit_compile=True, autograph=False)(x_str)

    # Plug in the TF-compiled JAX-converted `compute_jax_fn`.
    composed_fn = lambda x_str: tf_fn(
        x_str,
        compute_tf_fn=tf.function(jax2tf.convert(jnp.sin),
                                  autograph=False,
                                  jit_compile=True))
    res_tf = composed_fn(x_str)
    self.assertAllClose(res_tf.numpy(),
                        jnp.sin(np.array([3.14, 2.78], dtype=np.float16)))

    # Save and restore SavedModel
    restored_f, _ = tf_test_util.SaveAndLoadFunction(composed_fn,
                                                     input_args=[x_str])
    res_tf_restored = restored_f(x_str)
    self.assertAllClose(res_tf_restored.numpy(), res_tf.numpy())


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
