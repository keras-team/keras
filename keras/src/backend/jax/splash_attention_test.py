# Copyright 2024 The Keras Authors. All Rights Reserved.
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

import unittest.mock

import jax
import jax.numpy as jnp
import pytest

from keras.src import backend
from keras.src import testing
from keras.src.backend.jax import nn as jax_nn


@pytest.mark.skipif(backend.backend() != "jax", reason="Test is JAX-specific.")
class SplashAttentionTest(testing.TestCase):
    def test_splash_attention_with_tracer_mask_fallback(self):
        # Reproduces behavior described in https://github.com/keras-team/keras/issues/21916
        # When compiling with JIT, the mask becomes a Tracer.
        # Splash attention requires a concrete mask for hashing.
        # We ensure it falls back gracefully instead of crashing.

        # Mock is_tpu=True to trigger the Splash Attention path
        # We can't actually run on TPU in CI, but we want to test the logic path
        # up to the fallback check.

        # We also need to mock _can_use_flash_attention to return True
        # so we enter the block where the check happens.

        with unittest.mock.patch(
            "keras.src.backend.jax.nn._can_use_flash_attention",
            return_value=True,
        ):
            # We mock jax.devices() to simulate TPU platform
            # The actual device object needs a 'platform' attribute
            mock_device = unittest.mock.Mock()
            mock_device.platform = "tpu"

            with unittest.mock.patch("jax.devices", return_value=[mock_device]):

                @jax.jit
                def run_attention(query, key, value, mask):
                    return jax_nn.dot_product_attention(
                        query, key, value, mask=mask
                    )

                # Concrete inputs
                B, T, H, D = 1, 4, 2, 8
                query = jnp.ones((B, T, H, D))
                key = jnp.ones((B, T, H, D))
                value = jnp.ones((B, T, H, D))
                mask = jnp.ones((B, H, T, T))

                # This should run without ConcretizationTypeError
                # because the code should detect `mask` is a Tracer and disable
                # flash_attention
                out = run_attention(query, key, value, mask)
                self.assertIsNotNone(out)
                self.assertEqual(out.shape, (B, T, H, D))
