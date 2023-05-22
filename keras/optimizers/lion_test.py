# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Lion."""

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.framework import dtypes

from keras.optimizers.lion import Lion


def lion_update_numpy(
    params,
    grads,
    momentums,
    learning_rate=0.0001,
    beta_1=0.9,
    beta_2=0.99,
):
    params = params - learning_rate * np.sign(
        beta_1 * momentums + (1 - beta_1) * grads
    )
    momentums = beta_2 * momentums + (1 - beta_2) * grads
    return params, momentums


class LionOptimizerTest(tf.test.TestCase):
    def testDense(self):
        for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
            learning_rate = 0.0001
            beta_1 = 0.9
            beta_2 = 0.99
            with self.cached_session():
                m0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
                m1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
                var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
                var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
                grads0_np = np.array([0.9, 0.0], dtype=dtype.as_numpy_dtype)
                grads1_np = np.array([0.1, 0.0], dtype=dtype.as_numpy_dtype)

                var0 = tf.Variable(var0_np)
                var1 = tf.Variable(var1_np)
                grads0 = tf.constant(grads0_np)
                grads1 = tf.constant(grads1_np)
                optimizer = Lion(
                    learning_rate=learning_rate,
                    beta_1=beta_1,
                    beta_2=beta_2,
                )

                # Run 3 steps of Lion
                for _ in range(3):
                    optimizer.apply_gradients(
                        zip([grads0, grads1], [var0, var1])
                    )
                    var0_np, m0_np = lion_update_numpy(
                        var0_np,
                        grads0_np,
                        m0_np,
                        learning_rate=learning_rate,
                        beta_1=beta_1,
                        beta_2=beta_2,
                    )
                    var1_np, m1_np = lion_update_numpy(
                        var1_np,
                        grads1_np,
                        m1_np,
                        learning_rate=learning_rate,
                        beta_1=beta_1,
                        beta_2=beta_2,
                    )
                    # Validate updated params
                    self.assertAllCloseAccordingToType(var0_np, var0)
                    self.assertAllCloseAccordingToType(var1_np, var1)

    def testSparse(self):
        for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
            learning_rate = 0.0001
            beta_1 = 0.9
            beta_2 = 0.99
            with self.cached_session():
                m0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
                m1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
                var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
                var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
                grads0_np = np.array([0.9, 0.0], dtype=dtype.as_numpy_dtype)
                grads1_np = np.array([0.1, 0.0], dtype=dtype.as_numpy_dtype)

                var0 = tf.Variable(var0_np)
                var1 = tf.Variable(var1_np)
                grads0_np_indices = np.array([0], dtype=np.int32)
                grads0 = tf.IndexedSlices(
                    tf.constant(grads0_np[grads0_np_indices]),
                    tf.constant(grads0_np_indices),
                    tf.constant([2]),
                )
                grads1_np_indices = np.array([0], dtype=np.int32)
                grads1 = tf.IndexedSlices(
                    tf.constant(grads1_np[grads1_np_indices]),
                    tf.constant(grads1_np_indices),
                    tf.constant([2]),
                )

                optimizer = Lion(
                    learning_rate=learning_rate,
                    beta_1=beta_1,
                    beta_2=beta_2,
                )

                # Run 3 steps of Lion
                for _ in range(3):
                    optimizer.apply_gradients(
                        zip([grads0, grads1], [var0, var1])
                    )
                    var0_np, m0_np = lion_update_numpy(
                        var0_np,
                        grads0_np,
                        m0_np,
                        learning_rate=learning_rate,
                        beta_1=beta_1,
                        beta_2=beta_2,
                    )
                    var1_np, m1_np = lion_update_numpy(
                        var1_np,
                        grads1_np,
                        m1_np,
                        learning_rate=learning_rate,
                        beta_1=beta_1,
                        beta_2=beta_2,
                    )
                    # Validate updated params
                    self.assertAllCloseAccordingToType(var0_np, var0)
                    self.assertAllCloseAccordingToType(var1_np, var1)


if __name__ == "__main__":
    tf.test.main()
