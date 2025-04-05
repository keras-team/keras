# Copyright 2021 The JAX Authors.
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
"""Tests for the jax2tf conversion of pjit.

 To verify that the tests do run indeed on multiple devices you can run

  perftools/gputools/profiler/jfprof.sh jax/experimental/jax2tf/tests:sharding_test_tpu -- -c opt --test_filter=ShardingTest.test_shmap_all_to_all --test_arg=--vmodule=jax2tf=3 --

"""
from collections.abc import Sequence
from functools import partial
import logging
import re
from typing import Any
import unittest

from absl import app
from absl.testing import absltest

import jax
from jax._src import compiler
from jax._src import config
from jax._src import test_util as jtu
from jax._src import xla_bridge
from jax import lax
from jax.experimental import jax2tf
from jax.experimental import pjit
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import jax.numpy as jnp

import numpy as np

import tensorflow as tf

config.parse_flags_with_absl()
jtu.request_cpu_devices(8)

# Must come after initializing the flags
from jax.experimental.jax2tf.tests import tf_test_util

topology = None


def initialize_tf_tpu():
  global topology
  if jtu.test_device_matches(["tpu"]):
    with jtu.ignore_warning(message="the imp module is deprecated"):
      resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tf.config.experimental_connect_to_cluster(resolver)
    # Do TPU init at beginning since it will wipe out all HBMs.
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
  else:
    topology = None

app.call_after_init(initialize_tf_tpu)


class ShardingTest(tf_test_util.JaxToTfTestCase):
  """Tests that inspect the HLO for the sharding annotations.
  """
  def setUp(self):
    super().setUp()
    if jtu.test_device_matches(["gpu"]):
      raise unittest.SkipTest("Sharding HLO tests not useful for GPU")

    if len(jax.devices()) < 2:
      raise unittest.SkipTest("Test requires at least 2 local devices")
    self.devices = np.array(jax.devices()[:2])  # use 2 devices

    self.warning_ctx = jtu.ignore_warning(
        message="jax2tf.convert with native_serialization=False is deprecated"
    )
    self.warning_ctx.__enter__()

  def tearDown(self):
    self.warning_ctx.__exit__(None, None, None)
    super().tearDown()

  def log_jax_hlo(self, f_jax, args: Sequence[Any], *,
                  num_replicas=1, num_partitions=2):
    """Log the HLO generated from JAX before and after optimizations"""
    jax_comp = f_jax.lower(*args).compiler_ir(dialect="stablehlo")
    jax_hlo = str(jax_comp)
    logging.info("[%s] got JAX HLO %s", self._testMethodName, jax_hlo)

    # We only dump JAX optimized code on the TPU
    if jtu.test_device_matches(["tpu"]):
      backend = xla_bridge.get_backend()
      device_assignment = np.arange(num_partitions * num_replicas)
      device_assignment = np.reshape(device_assignment, (-1, num_partitions))
      use_spmd_partitioning = num_partitions > 1
      compile_options = compiler.get_compile_options(
          num_replicas=num_replicas,
          num_partitions=num_partitions,
          device_assignment=device_assignment,
          use_spmd_partitioning=use_spmd_partitioning,
      )
      jax_optimized_hlo = backend.compile(
          jax_hlo, compile_options).hlo_modules()[0].to_string()
      logging.info("[%s] got JAX optimized HLO for platform %s %s",
                   self._testMethodName, backend.platform, jax_optimized_hlo)

  def device_assignment(self,
                        computation_shape=(1, 1, 1, 2),
                        num_replicas=1):
    self.assertEqual(jtu.device_under_test(), "tpu")
    return tf.tpu.experimental.DeviceAssignment.build(
        topology, computation_shape=computation_shape,
        num_replicas=num_replicas)

  def tf_hlo(self, f_tf, args_tf: Sequence[Any]) -> str:
    """Get the unoptimized HLO from TF"""
    f_tf_fun = tf.function(f_tf, autograph=False, jit_compile=True)
    logging.info("[%s] Got TF graph %s",
                 self._testMethodName,
                 f_tf_fun.get_concrete_function(*args_tf).graph.as_graph_def())
    device_name = f"/device:{jtu.device_under_test().upper()}:0"
    tf_hlo_generator = f_tf_fun.experimental_get_compiler_ir(*args_tf)
    tf_hlo = tf_hlo_generator(
        stage="hlo", platform_name=jtu.device_under_test().upper()
    )
    logging.info("[%s] got TF HLO %s", self._testMethodName, tf_hlo)
    # TODO(necula): TensorFlow doesn't support getting the optimized_hlo on TFRT
    # TPU devices. But it doesn't seem like we're using it anyway.
    #
    # tf_optimized_hlo = tf_hlo_generator(stage="optimized_hlo",
    #                                     platform_name=platform_name)
    # logging.info("[%s] got TF optimized HLO for %s: %s", self._testMethodName,
    #              platform_name, tf_optimized_hlo)
    # Before we check, we drop the metadata= at the end of tf_hlo
    return re.sub(r'metadata=.*', '', tf_hlo)

  def GEQ(self, value):
    # Construct an expected >= value. See `check_sharding`.
    return (">=", value)

  def check_sharding(self, f_tf, args_tf: Sequence[Any], *,
                     checks=()):
    """Check the sharding in TF.

    Args:
      f_tf: the TF callable
      args_tf: the TF args
      checks: a list of tuples. The first element is a regular expression, the
        second element is an integer representing the expected number of
        occurrences of the regular expression in the TF HLO. As a special case,
        the second element can be the result of `self.GEQ(v)` to check that
        the number of occurrences is greater or equal to a value.
    """
    tf_hlo = self.tf_hlo(f_tf, args_tf)
    for check_re, expected_count in checks:
      count = len(re.findall(check_re, tf_hlo))
      if isinstance(expected_count, int):
        self.assertEqual(
            count, expected_count,
            (f"regular expression `{check_re}` expected to occur "
            f"{expected_count} times but occurs {count} times in "
            f"the TF HLO.\nThis is the TF HLO:\n{tf_hlo}"))
      elif isinstance(expected_count, tuple) and expected_count[0] == ">=":
        self.assertGreaterEqual(
            count, expected_count[1],
            (f"regular expression `{check_re}` expected to occur "
            f"at least {expected_count[1]} times but occurs {count} times in "
            f"the TF HLO.\nThis is the TF HLO:\n{tf_hlo}"))
      else:
        assert False

  @jtu.parameterized_filterable(
    kwargs=[
      dict(testcase_name=f"_in_shardings={in_shardings}_out_shardings={out_shardings}",
           in_shardings=in_shardings, out_shardings=out_shardings)
      for in_shardings in ("missing", None, "P")
      for out_shardings in ("missing", None, "P")
  ])
  @jtu.with_mesh([("x", 2)])
  def test_pjit_basic(self, in_shardings="P", out_shardings="P"):
    # Ensure that we can distinguish the inputs and outputs by shape
    def f_jax(x):  # f32[10,20] -> f32[20,10]
      return jnp.sin(x.T)

    pjit_kwargs = {}
    if in_shardings != "missing":
      pjit_kwargs["in_shardings"] = (P(None, "x") if in_shardings == "P" else None)
    if out_shardings != "missing":
      pjit_kwargs["out_shardings"] = (P("x", None) if out_shardings == "P" else None)
    f_jax = pjit.pjit(f_jax, **pjit_kwargs)

    x_shape = (10, 20)
    x = np.arange(np.prod(x_shape), dtype=np.float32).reshape(x_shape)

    self.log_jax_hlo(f_jax, [x], num_partitions=2)

    @tf.function(autograph=False, jit_compile=True)
    def f_tf(x):
      f_converted = jax2tf.convert(f_jax)
      if jtu.test_device_matches(["tpu"]):
        return tf.compat.v1.tpu.rewrite(
            f_converted, [tf.convert_to_tensor(x)],
            device_assignment=self.device_assignment(
                computation_shape=[1, 1, 1, 2],
            ))[0]
      else:
        return f_converted(x)

    # Annotation count for the input
    count_in_P = 1 if in_shardings == "P" else 0
    if config.jax2tf_default_native_serialization.value:
      # With native serialization even unspecified in_shardings turn into replicated
      count_in_replicated = 1 if in_shardings in [None, "missing"] else 0
    else:
      count_in_replicated = 1 if in_shardings is None else 0
    # Annotation count for the output
    count_out_P = 1 if out_shardings == "P" else 0
    count_out_replicated = 1 if out_shardings is None else 0

    self.check_sharding(
        jax2tf.convert(f_jax), [x],
        checks=[
            # The argument
            (r"f32\[10,20\].*custom_call_target.*Sharding.*sharding.*devices=\[1,2\]",
             count_in_P),
            # The result
            (r"f32\[20,10\].*custom_call_target.*Sharding.*sharding.*devices=\[2,1\]",
             count_out_P),
        ])
    # TODO(b/326476605): Change the condition below if required.
    if in_shardings not in [None, "missing"] and out_shardings is not None:
      self.check_sharding(
        jax2tf.convert(f_jax), [x],
        checks=[
            (r"f32\[10,20\].*custom_call_target.*Sharding.*sharding.*replicated",
             count_in_replicated),
            (r"f32\[20,10\].*custom_call_target.*Sharding.*sharding.*replicated",
             count_out_replicated),
            (r"custom_call_target.*Sharding",
             count_in_P + count_in_replicated + count_out_P + count_out_replicated),
        ])

    res_jax = f_jax(x)
    res_tf = f_tf(x)
    self.assertAllClose(res_tf.numpy(), res_jax)

  @jtu.with_mesh([("x", 2)])
  def test_pjit_variable_arg(self):
    # The first argument is a tf.Variable
    @partial(pjit.pjit, in_shardings=(P(None, "x"), P("x", None)),
             out_shardings=None)
    def f_jax(x, y):  # f32[10,20] , f32[20,30] -> f32[10,30]
      return x @ y

    shape_x = (10, 20)
    x = np.arange(np.prod(shape_x), dtype=np.float32).reshape(shape_x)
    shape_y = (20, 30)
    y = np.arange(np.prod(shape_y), dtype=np.float32).reshape(shape_y)

    self.log_jax_hlo(f_jax, [x, y], num_partitions=2)

    x_v = tf.Variable(x)
    f_tf = lambda y: jax2tf.convert(f_jax)(x_v, y)

    self.check_sharding(
        f_tf, [y],
        checks=[
            # The variable argument
            (r"f32\[10,20\].*custom_call_target.*Sharding.*sharding.*devices=\[1,2\]", 1),
            # The y argument
            (r"f32\[20,30\].*custom_call_target.*Sharding.*sharding.*devices=\[2,1\]", 1),
            # The output sharding
            (r"f32\[10,30\].*custom_call_target.*Sharding.*sharding.*replicated", 1),
            # No other annotations
            (r"custom_call_target.*Sharding", 3)
        ])

  @jtu.with_mesh([("x", 2)])
  def test_pjit_closed_over_const(self):
    x = np.ones((10, 20), dtype=np.float32)
    const = jnp.full((10, 20), 7, dtype=np.float32)

    @partial(pjit.pjit, in_shardings=(P("x"),), out_shardings=None)
    def f_jax(x):  # f32[10,20] -> f32[20,10]
      return (x * const).T

    @tf.function(autograph=False, jit_compile=True)
    def f_tf(x):
      f_converted = jax2tf.convert(f_jax)
      if jtu.test_device_matches(["tpu"]):
        return tf.compat.v1.tpu.rewrite(
            f_converted, [tf.convert_to_tensor(x)],
            device_assignment=self.device_assignment(
                computation_shape=[1, 1, 1, 2])
        )[0]
      else:
        return f_converted(x)

    self.check_sharding(
        jax2tf.convert(f_jax), [x],
        checks=[
            # x
            (r"f32\[10,20\].*custom_call_target.*Sharding.*sharding.*devices=\[2,1\]",
             1),
            # The result
            (r"f32\[20,10\].*custom_call_target.*Sharding.*sharding.*replicated",
             self.GEQ(1)),
        ])

    res_jax = f_jax(x)
    res_tf = f_tf(x)
    self.assertAllClose(res_tf, res_jax)

  @jtu.parameterized_filterable(
    kwargs=[
      dict(testcase_name=f"_nested_pjit={nested_pjit}_constraint={constraint}_poly={poly}",
           nested_pjit=nested_pjit, constraint=constraint, poly=poly)
      # We add a constraint either with a nested pjit or with a sharding_constraint
      for nested_pjit in (True, False)
      for constraint in (None, "P")
      for poly in (None, "2*b1,_", "_,b2", "2*b1,b2")
  ])
  @jtu.with_mesh([("x", 2)])
  def test_pjit_sharding_constraint(self, nested_pjit=True, constraint="P", poly="2*b1,b2"):
    constraint_sharding = P("x", None) if constraint == "P" else None
    @partial(pjit.pjit, in_shardings=None,
             out_shardings=None)
    def f_jax(x):  # x: f32[10, 20], optionally some axes as polymorphic
      y = jnp.concatenate([x, x], axis=1)  # y: f32[10, 40]
      if nested_pjit:
        y = pjit.pjit(lambda y: y, in_shardings=constraint_sharding,
                      out_shardings=constraint_sharding)(y)
      else:
        y = jax.lax.with_sharding_constraint(y, constraint_sharding)
      return jnp.concatenate([y, y], axis=1)  # res: f32[10, 80]

    shape = (10, 20)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    self.log_jax_hlo(f_jax, [x], num_partitions=2)
    f_tf = jax2tf.convert(f_jax, polymorphic_shapes=poly)

    # If we use a pjit then we see two constraints, otherwise only 1
    count_inner_sharding = (2 if nested_pjit else 1) if constraint == "P" else 0
    count_inner_replicated = (2 if nested_pjit else 1) if constraint != "P" else 0
    self.check_sharding(
        f_tf, [x],
        checks=[
            # The input argument
            (r"f32\[10,20\].*custom_call_target.*Sharding.*sharding.*replicated", 1),
            # The y argument
            (r"f32\[10,40\].*custom_call_target.*Sharding.*sharding.*devices=\[2,1\]",
             count_inner_sharding),
            (r"f32\[10,40\].*custom_call_target.*Sharding.*sharding.*replicated",
             count_inner_replicated),
            # The output sharding
            (r"f32\[10,80\].*custom_call_target.*Sharding.*sharding.*replicated", 1),
            # No other annotations
            (r"custom_call_target.*Sharding", 2 + count_inner_sharding + count_inner_replicated)
        ])

  @jtu.parameterized_filterable(
    kwargs=[
      dict(testcase_name=f"_in_shardings={in_shardings}_out_shardings={out_shardings}",
           in_shardings=in_shardings, out_shardings=out_shardings)
      for in_shardings in ("missing", None, "P")
      for out_shardings in ("missing", None, "P")
  ])
  def test_grad_pjit(self, in_shardings="P", out_shardings=None):
    if not config.jax2tf_default_native_serialization.value:
      self.skipTest("TODO: failure in non-native serialization")
    local_devices = list(jax.local_devices())
    size = 2
    if len(local_devices) < size:
      raise unittest.SkipTest(f"Test requires {size} local devices")
    mesh_devices = np.array(local_devices[:size]).reshape((2,))
    mesh = jax.sharding.Mesh(mesh_devices, ("x",))
    def f_jax(x):  # x: f32[10,20] -> f32[20,10]
      return jnp.sin(x.T)

    pjit_kwargs = {}
    if in_shardings != "missing":
      pjit_kwargs["in_shardings"] = (
        NamedSharding(mesh, P(None, "x")) if in_shardings == "P" else None)
    if out_shardings != "missing":
      pjit_kwargs["out_shardings"] = (
        NamedSharding(mesh, P("x", None)) if out_shardings == "P" else None)
    f_jax = pjit.pjit(f_jax, **pjit_kwargs)
    x_shape = (10, 20)
    x = np.arange(np.prod(x_shape), dtype=np.float32).reshape(x_shape)

    def f_grad_tf(x_v, res_ct):
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_v)
        with tf.GradientTape() as tape2:
          tape2.watch(x_v)
          res_tf = jax2tf.convert(f_jax)(x_v)
        dy_dx = tape.gradient(res_tf, x_v, output_gradients=res_ct)
      d2y_dx2 = tape.gradient(dy_dx, x_v)
      return d2y_dx2

    # Annotation count for the primal input and the grad output
    count_in_P = self.GEQ(2) if in_shardings == "P" else 0
    if config.jax2tf_default_native_serialization.value:
      # With native serialization even unspecified shardings turn into replicated
      count_in_replicated = self.GEQ(2) if in_shardings in [None, "missing"] else 0
    else:
      count_in_replicated = self.GEQ(2) if in_shardings is None else 0
    # Annotation count for the contangent input
    count_out_P = self.GEQ(1) if out_shardings == "P" else 0
    if config.jax2tf_default_native_serialization.value:
      # With native serialization even unspecified shardings turn into replicated
      count_out_replicated = self.GEQ(1) if out_shardings in [None, "missing"] else 0
    else:
      count_out_replicated = self.GEQ(1) if out_shardings is None else 0

    self.check_sharding(f_grad_tf, [x, x.T],
        checks=[
            # The input primal argument, and the output grad
            (r"f32\[10,20\].*custom_call_target.*Sharding.*sharding.*devices=\[1,2\]", count_in_P),
            # The primal result, and the input cotangent
            (r"f32\[20,10\].*custom_call_target.*Sharding.*sharding.*devices=\[2,1\]", count_out_P),
        ])
    # TODO(b/326476605): Change the condition below if required.
    if out_shardings not in [None, "missing"] and in_shardings not in [None, "missing"]:
      self.check_sharding(f_grad_tf, [x, x.T],
        checks=[
            (r"f32\[10,20\].*custom_call_target.*Sharding.*sharding.*replicated", count_in_replicated),
            # The primal result, and the input cotangent
            (r"f32\[20,10\].*custom_call_target.*Sharding.*sharding.*devices=\[2,1\]", count_out_P),
        ])

  def test_grad_sharding_different_mesh(self):
    # Convert with two similar meshes, the only difference being
    # the order of the devices. grad should not fail.
    # https://github.com/jax-ml/jax/issues/21314
    devices = jax.local_devices()[:2]
    if len(devices) < 2:
      raise unittest.SkipTest("Test requires 2 local devices")
    def f_jax(x):
      return jnp.sum(x * 2.)

    mesh = Mesh(devices, "i")
    # The same mesh with reversed order of devices
    mesh_rev = Mesh(list(reversed(devices)), "i")
    shardings = NamedSharding(mesh, jax.sharding.PartitionSpec(("i",)))
    shardings_rev = NamedSharding(mesh_rev, jax.sharding.PartitionSpec(("i",)))

    f_tf = tf.function(jax2tf.convert(pjit.pjit(f_jax, in_shardings=shardings)),
                       autograph=False)
    f_tf_rev = tf.function(jax2tf.convert(pjit.pjit(f_jax, in_shardings=shardings_rev)),
                           autograph=False)
    inp = np.ones((2, 4), dtype=np.float32)

    input_v = tf.Variable(inp)
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(input_v)
      res_tf = f_tf(input_v)
      g = tape.gradient(res_tf, input_v)

    with tf.GradientTape(persistent=True) as tape:
      tape.watch(input_v)
      res_tf_rev = f_tf_rev(input_v)
      g_rev = tape.gradient(res_tf_rev, input_v)
    self.assertAllClose(g, g_rev)

  @jtu.parameterized_filterable(
    kwargs=[
      dict(testcase_name=f"_func={func}", func=func)
      for func in ("pjit_sharded", "pjit_replicated",
                   "nested_pjit_sharded", "nested_pjit_replicated")
  ])
  def test_pjit_eager_error(self, func="pjit_sharded"):
    if config.jax2tf_default_native_serialization.value:
      raise unittest.SkipTest("There is no error in eager mode for native serialization")

    # Define some test functions
    @partial(pjit.pjit, in_shardings=(P("x"),),
             out_shardings=None)
    def f_pjit_sharded(a):
      return a + a

    @partial(pjit.pjit, in_shardings=None,
             out_shardings=None)
    def f_pjit_replicated(a):
      return a + a

    def f_nested_pjit_sharded(a):
      return a + pjit.pjit(jnp.sin, in_shardings=(P("x"),), out_shardings=None)(a)

    def f_nested_pjit_replicated(a):
      return a + pjit.pjit(jnp.sin, in_shardings=None, out_shardings=None)(a)

    shape = (8, 10)
    a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    if func == "pjit_sharded":
      f_jax = f_pjit_sharded
    elif func == "pjit_replicated":
      f_jax = f_pjit_replicated
    elif func == "nested_pjit_sharded":
      f_jax = f_nested_pjit_sharded
    elif func == "nested_pjit_replicated":
      f_jax = f_nested_pjit_replicated
    else:
      assert False

    with Mesh(self.devices, axis_names=("x",)):
      _ = f_jax(a)
      with self.assertRaisesRegex(
          ValueError,
          "function with sharded arguments or results must be used under a `tf.function` context"):
        jax2tf.convert(f_jax)(a)

  @jtu.ignore_warning(category=UserWarning,
                      message="all_to_all .* are only implemented properly for TPUs and GPUs .*")
  def test_shmap_all_to_all(self):
    if jtu.test_device_matches(["cpu"]):
      raise unittest.SkipTest("TODO(b/268295912): ShardingRemover crash")

    mesh = Mesh(self.devices, axis_names=('x'))
    a = np.arange(4 * 4, dtype=np.float32).reshape((4, 4))

    @partial(pjit.pjit,
             in_shardings=(P('x', None),), out_shardings=P(None, 'x'))
    @partial(shard_map, mesh=mesh,
             in_specs=(P('x', None),), out_specs=P(None, 'x'))
    def f_jax(b):  # b: f32[2, 4]
      return lax.all_to_all(b, 'x', split_axis=1, concat_axis=1, tiled=True)

    @tf.function(autograph=False, jit_compile=True)
    def f_tf(a):
      f_converted = jax2tf.convert(f_jax, native_serialization=True)
      if jtu.test_device_matches(["tpu"]):
        return tf.compat.v1.tpu.rewrite(
            f_converted, [tf.convert_to_tensor(a)],
            device_assignment=self.device_assignment(
                computation_shape=[1, 1, 1, 2])
        )[0]
      else:
        return f_converted(a)

    with mesh:
      res_jax = f_jax(a)  # res: f32[2, 8]
      b0, b1 = np.split(a, 2, axis=0)  # The shard_map in_specs splits on axis 0
      b00, b01 = np.split(b0, 2, axis=1)  # split_axis=1
      b10, b11 = np.split(b1, 2, axis=1)
      b0 = np.concatenate([b00, b10], axis=1)  # concat_axis=1
      b1 = np.concatenate([b01, b11], axis=1)
      res = np.concatenate([b0, b1], axis=1)  # out_specs concatenates on axis 1
      self.assertAllClose(res_jax, res)
      res_tf = f_tf(a)
      self.assertAllClose(res_tf, res_jax)

      # TODO(b/274648842): Failed to GetCompilerIr
      # self.check_sharding(
      #     jax2tf.convert(f_jax, native_serialization=True), [a],
      #     checks=[])

  @unittest.skip("TODO(b/268295912): ShardingRemover crash,on all platforms!!!")
  def test_repro_xla_bug_shmap_collective_permute(self):
    mesh = Mesh(self.devices, axis_names=('x'))

    @partial(pjit.pjit,
             in_shardings=(P('x', None),), out_shardings=P('x', None))
    @partial(shard_map, mesh=mesh,
             in_specs=(P('x', None),), out_specs=P('x', None))
    def f_jax(b):  # b: f32[2, 4]
      axis_size = lax.psum(1, 'x')
      perm = [(j, (j + 1) % axis_size) for j in range(axis_size)]
      return lax.ppermute(b, 'x', perm=perm)

    with mesh:
      a = np.arange(4 * 4).reshape((4, 4))
      res_jax = f_jax(a)
      b0, b1 = np.split(a, 2, axis=0)  # The shard_map splits on axis 0
      b0, b1 = b1, b0
      expected = np.concatenate([b0, b1], axis=0)  # out_specs concatenates on axis 0
      self.assertAllClose(res_jax, expected)

      # XLA bug: invoke the f_tf without tpu.replicate
      f_tf = tf.function(
          jax2tf.convert(f_jax, native_serialization=True),
          autograph=False, jit_compile=True)

      res_tf = f_tf(a)
      self.assertAllClose(res_tf, expected)

  @jtu.parameterized_filterable(
    kwargs=[
      dict(testcase_name=f"_poly={poly}", poly=poly)
      for poly in (None, "2*b1,_", "_,b2", "2*b1,b2")
    ])
  def test_shmap_collective_permute(self, poly=None):
    if jtu.test_device_matches(["cpu"]):
      raise unittest.SkipTest("TODO(b/268295912): ShardingRemover crash")
    mesh = Mesh(self.devices, axis_names=('x'))
    a = np.arange(4 * 4, dtype=np.float32).reshape((4, 4))

    @partial(pjit.pjit,
             in_shardings=(P('x', None),), out_shardings=P('x', None))
    @partial(shard_map, mesh=mesh,
             in_specs=(P('x', None),), out_specs=P('x', None))
    def f_jax(b):  # b: f32[2, 4]
      axis_size = lax.psum(1, 'x')
      perm = [(j, (j + 1) % axis_size) for j in range(axis_size)]
      return lax.ppermute(b, 'x', perm=perm)

    @tf.function(autograph=False, jit_compile=True)
    def f_tf(a):
      f_converted = jax2tf.convert(f_jax, native_serialization=True,
                                   polymorphic_shapes=poly)
      if jtu.test_device_matches(["tpu"]):
        res = tf.compat.v1.tpu.rewrite(
            f_converted, [tf.convert_to_tensor(a)],
            device_assignment=self.device_assignment(
                computation_shape=[1, 1, 1, 2])
        )[0]
      else:
        res = f_converted(a)
      return res

    with mesh:
      res_jax = f_jax(a)
      b0, b1 = np.split(a, 2, axis=0)  # The shard_map splits on axis 0
      b0, b1 = b1, b0
      expected = np.concatenate([b0, b1], axis=0)  # out_specs concatenates on axis 0
      self.assertAllClose(res_jax, expected)
      res_tf = f_tf(a)
      self.assertAllClose(res_tf, expected)
      # TODO(b/274648842): Failed to GetCompilerIr
      # self.check_sharding(
      #     jax2tf.convert(f_jax, native_serialization=True), [a],
      #     checks=[])

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
