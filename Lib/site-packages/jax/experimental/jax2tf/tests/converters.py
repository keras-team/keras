# Copyright 2022 The JAX Authors.
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
"""Converters for jax2tf."""

from collections.abc import Callable
import dataclasses
import functools
import tempfile
from typing import Any

from jax.experimental import jax2tf
import tensorflowjs as tfjs

from jax.experimental.jax2tf.tests.model_harness import ModelHarness


@dataclasses.dataclass
class Converter:
  name: str
  convert_fn: Callable[..., Any]
  compare_numerics: bool = True


def jax2tf_convert(harness: ModelHarness, enable_xla: bool = True):
  return jax2tf.convert(
      harness.apply_with_vars,
      enable_xla=enable_xla,
      polymorphic_shapes=harness.polymorphic_shapes)


def jax2tfjs(harness: ModelHarness):
  """Converts the given `test_case` using the TFjs converter."""
  with tempfile.TemporaryDirectory() as model_dir:
    tfjs.converters.convert_jax(
        apply_fn=harness.apply,
        params=harness.variables,
        input_signatures=harness.tf_input_signature,
        polymorphic_shapes=harness.polymorphic_shapes,
        model_dir=model_dir)


ALL_CONVERTERS = [
    # jax2tf with XLA support (enable_xla=True).
    Converter(name='jax2tf_xla', convert_fn=jax2tf_convert),
    # jax2tf without XLA support (enable_xla=False).
    Converter(
        name='jax2tf_noxla',
        convert_fn=functools.partial(jax2tf_convert, enable_xla=False),
    ),
    # Convert JAX to Tensorflow.JS.
    Converter(name='jax2tfjs', convert_fn=jax2tfjs, compare_numerics=False),
]
