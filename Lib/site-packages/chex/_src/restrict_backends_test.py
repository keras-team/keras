# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for `restrict_backends.py`."""
from absl.testing import absltest
from chex._src import restrict_backends
import jax
import jax.numpy as jnp
import numpy as np


def compute_cube(side):
  return jnp.sum(jnp.ones((side, side)) * side)


class RestrictBackendsTest(absltest.TestCase):

  # These tests need an accelerator of some sort, so that JAX can try to use it.
  def setUp(self):
    super().setUp()

    try:
      jax.devices('gpu')
      gpu_backend_available = True
    except RuntimeError:
      gpu_backend_available = False

    try:
      jax.devices('tpu')
      tpu_backend_available = True
    except RuntimeError:
      tpu_backend_available = False

    if not tpu_backend_available or gpu_backend_available:
      self.skipTest('No known accelerator backends are available, so these '
                    'tests will not test anything useful.')

  def test_detects_implicitly_forbidden_tpu_computation(self):
    with self.assertRaisesRegex(restrict_backends.RestrictedBackendError,
                                r'forbidden by restrict_backends'):
      with restrict_backends.restrict_backends(allowed=['cpu']):
        compute_cube(3)
    # Make sure the restriction is no longer in place.
    np.testing.assert_array_equal(compute_cube(3), 27)

  def test_detects_explicitly_forbidden_tpu_computation(self):
    with self.assertRaisesRegex(restrict_backends.RestrictedBackendError,
                                r'forbidden by restrict_backends'):
      with restrict_backends.restrict_backends(forbidden=['tpu', 'gpu']):
        compute_cube(2)
    # Make sure the restriction is no longer in place.
    np.testing.assert_array_equal(compute_cube(2), 8)

  def test_detects_implicitly_forbidden_cpu_computation(self):
    with self.assertRaisesRegex(restrict_backends.RestrictedBackendError,
                                r'forbidden by restrict_backends'):
      with restrict_backends.restrict_backends(allowed=['tpu', 'gpu']):
        jax.jit(lambda: compute_cube(8), backend='cpu')()
    # Make sure the restriction is no longer in place.
    np.testing.assert_array_equal(compute_cube(8), 512)

  def test_detects_explicitly_forbidden_cpu_computation(self):
    with self.assertRaisesRegex(restrict_backends.RestrictedBackendError,
                                r'forbidden by restrict_backends'):
      with restrict_backends.restrict_backends(forbidden=['cpu']):
        jax.jit(lambda: compute_cube(9), backend='cpu')()
    # Make sure the restriction is no longer in place.
    np.testing.assert_array_equal(compute_cube(9), 729)

  def test_ignores_explicitly_allowed_cpu_computation(self):
    with restrict_backends.restrict_backends(allowed=['cpu']):
      c = jax.jit(lambda: compute_cube(4), backend='cpu')()
    np.testing.assert_array_equal(c, 64)

  def test_ignores_implicitly_allowed_cpu_computation(self):
    with restrict_backends.restrict_backends(forbidden=['tpu', 'gpu']):
      c = jax.jit(lambda: compute_cube(5), backend='cpu')()
    np.testing.assert_array_equal(c, 125)

  def test_ignores_explicitly_allowed_tpu_computation(self):
    with restrict_backends.restrict_backends(allowed=['tpu', 'gpu']):
      c = jax.jit(lambda: compute_cube(6))()
    np.testing.assert_array_equal(c, 216)

  def test_ignores_implicitly_allowed_tpu_computation(self):
    with restrict_backends.restrict_backends(forbidden=['cpu']):
      c = jax.jit(lambda: compute_cube(7))()
    np.testing.assert_array_equal(c, 343)

  def test_raises_if_no_restrictions_specified(self):
    with self.assertRaisesRegex(ValueError, r'No restrictions specified'):
      with restrict_backends.restrict_backends():
        pass

  def test_raises_if_contradictory_restrictions_specified(self):
    with self.assertRaisesRegex(ValueError, r"can't be both"):
      with restrict_backends.restrict_backends(
          allowed=['cpu'], forbidden=['cpu']):
        pass


if __name__ == '__main__':
  absltest.main()
