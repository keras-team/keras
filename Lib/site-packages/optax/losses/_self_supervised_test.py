# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for self-supervised losses in `optax.losses._self_supervised.py`."""

from absl.testing import absltest
import chex
import jax.numpy as jnp
import numpy as np
from optax.losses import _self_supervised


class NtxentTest(chex.TestCase):

  def setUp(self):
    super().setUp()
    self.ys = jnp.array([
        [-1.9540, 1.0780],
        [0.2380, -0.5703],
        [1.8745, -0.0195],
        [-0.6719, -1.9210],
    ])
    self.ys_2 = jnp.array([
        [0.0, 0.0],
        [0.2380, -0.5703],
        [1.8745, -0.0195],
        [-0.6719, -1.9210],
    ])
    self.ts_1 = jnp.array([0, 0, 1, 1])
    self.ts_2 = jnp.array([0, 0, 0, 1])
    # Calculated expected output
    self.exp_1 = jnp.array(14.01032)
    self.exp_2 = jnp.array(8.968544)
    self.exp_3 = jnp.array(9.2889)

  @chex.all_variants
  def test_batched(self):
    """Tests for a full batch."""
    np.testing.assert_allclose(
        self.variant(_self_supervised.ntxent)(self.ys, self.ts_1),
        self.exp_1,
        atol=1e-4,
    )

    np.testing.assert_allclose(
        self.variant(_self_supervised.ntxent)(self.ys, self.ts_2),
        self.exp_2,
        atol=1e-4,
    )

    np.testing.assert_allclose(
        self.variant(_self_supervised.ntxent)(self.ys_2, self.ts_1),
        self.exp_3,
        atol=1e-4,
    )


if __name__ == '__main__':
  absltest.main()
