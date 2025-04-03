# Copyright 2023 The JAX Authors.
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
"""Tests for backwards compatibility of custom calls involving TensorFlow.

See the back_compat_test_util module docstring for how to setup and update
these tests.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Sequence
import io
import os
import tarfile

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax._src.internal_test_util import export_back_compat_test_util as bctu
from jax._src.lib import xla_extension
from jax.experimental import jax2tf
from jax.experimental.jax2tf.tests.back_compat_testdata import tf_call_tf_function
import jax.numpy as jnp
import tensorflow as tf


jax.config.parse_flags_with_absl()


def serialize_directory(directory_path):
  """Seriliaze the directory as a string."""
  tar_buffer = io.BytesIO()
  with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
    tar.add(directory_path, arcname=os.path.basename(directory_path))

  # Convert the binary data to a base64-encoded string
  serialized_string = base64.b64encode(tar_buffer.getvalue())
  return serialized_string


def deserialize_directory(serialized_string, output_directory):
  """Deserialize the string to the directory."""
  # Convert the base64-encoded string back to binary data
  tar_data = base64.b64decode(serialized_string)

  # Extract the tar archive to the output directory
  with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r") as tar:
    tar.extractall(output_directory)


class CompatTensoflowTest(bctu.CompatTestBase):
  """Compatibility tests that use TF.

  Uses tf.Graph to serialize and run the functions; expects that `func`
  contains a `jax2tf.call_tf` and uses `jax2tf.convert` to generate a
  `tf.Graph` containing a XlaCallModule with the actual MLIR module.
  """

  def run_current(self, func: Callable, data: bctu.CompatTestData):
    # Here we use tf.saved_model and provide  string serialize/deserialize methods
    # for the whole directory.
    @tf.function(autograph=False, jit_compile=True)
    def tf_func(the_input):  # Use recognizable names for input and result
      res = jax2tf.convert(func, native_serialization=True)(the_input)
      return tf.identity(res, name="the_result")

    self.tf_func = tf_func
    return tf_func(*data.inputs)

  def serialize(
      self,
      func: Callable,
      data: bctu.CompatTestData,
      polymorphic_shapes: Sequence[str] | None = None,
      allow_unstable_custom_call_targets: Sequence[str] = (),
  ):
    # We serialize as a tf.Graph
    assert len(data.inputs) == 1  # We only support a single input now
    tf_graph = self.tf_func.get_concrete_function(*data.inputs).graph
    for op in tf_graph.get_operations():
      if op.type == "XlaCallModule":
        serialized_module = op.get_attr("module")
        module_str = xla_extension.mlir.deserialize_portable_artifact(
            serialized_module
        )
        module_version = op.get_attr("version")
        break
    else:
      raise ValueError("Cannot find an XlaCallModule")
    tf_graph_def = tf_graph.as_graph_def()
    # module_str is just for human readability, add both the MLIR module
    # and the tf.Graph
    module_str = (
        "# First the MLIR module:\n"
        + module_str
        + "\n# Then the tf.Graph:\n"
        + str(tf_graph_def)
    )
    # serialized = tf_graph_def.SerializeToString()
    module = tf.Module()
    module.call = self.tf_func.get_concrete_function(*data.inputs)
    root_dir = self.create_tempdir()
    saved_model_dir = os.path.join(root_dir, "saved_model")
    os.mkdir(saved_model_dir)
    tf.saved_model.save(
        module,
        saved_model_dir,
        options=tf.saved_model.SaveOptions(experimental_custom_gradients=False),
    )
    serialized =  serialize_directory(saved_model_dir)
    nr_devices = 1
    return serialized, module_str, module_version, nr_devices

  def run_serialized(
      self,
      data: bctu.CompatTestData,
      polymorphic_shapes: Sequence[str] | None = None,
  ):
    root_dir = self.create_tempdir()
    deserialize_directory(data.mlir_module_serialized, root_dir)
    saved_model_dir = os.path.join(root_dir, "saved_model")
    loaded_model = tf.saved_model.load(saved_model_dir)
    return (loaded_model.call(*data.inputs).numpy(),)

  def test_tf_call_tf_function(self):
    # A custom call tf.call_tf_function is generated when we lower call_tf
    # with the call_tf_graph=True option.
    def func(x):
      def func_tf(x):
        return tf.math.sin(x)

      return jnp.cos(
          jax2tf.call_tf(func_tf, output_shape_dtype=x, call_tf_graph=True)(x)
      )

    data = self.load_testdata(tf_call_tf_function.data_2023_07_29)
    self.run_one_test(func, data)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
