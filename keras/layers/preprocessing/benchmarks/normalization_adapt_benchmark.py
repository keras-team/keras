# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Benchmark for Keras text vectorization preprocessing layer's adapt method."""

import tensorflow.compat.v2 as tf

import time

import numpy as np

import keras
from keras.layers.preprocessing import normalization

tf.compat.v1.enable_v2_behavior()


def reduce_fn(state, values):
  """tf.data.Dataset-friendly implementation of mean and variance."""
  k, n, ex, ex2 = state
  # If this is the first iteration, we pick the first value to be 'k',
  # which helps with precision - we assume that k is close to an average
  # value and calculate mean and variance with respect to that.
  k = tf.cond(tf.equal(n, 0), lambda: values[0], lambda: k)

  sum_v = tf.reduce_sum(values, axis=0)
  sum_v2 = tf.reduce_sum(tf.square(values), axis=0)
  ones = tf.ones_like(values, dtype=tf.int32)
  batch_size = tf.reduce_sum(ones, axis=0)
  batch_size_f = tf.cast(batch_size, tf.float32)

  ex = 0 + sum_v - tf.multiply(batch_size_f, k)
  ex2 = 0 + sum_v2 + tf.multiply(
      batch_size_f, (tf.square(k) -
                     tf.multiply(tf.multiply(2.0, k), sum_v)))

  return (k, n + batch_size, ex, ex2)


class BenchmarkAdapt(tf.test.Benchmark):
  """Benchmark adapt."""

  def run_dataset_implementation(self, num_elements, batch_size):
    input_t = keras.Input(shape=(1,))
    layer = normalization.Normalization()
    _ = layer(input_t)

    num_repeats = 5
    starts = []
    ends = []
    for _ in range(num_repeats):
      ds = tf.data.Dataset.range(num_elements)
      ds = ds.map(
          lambda x: tf.expand_dims(tf.cast(x, tf.float32), -1))
      ds = ds.batch(batch_size)

      starts.append(time.time())
      # Benchmarked code begins here.
      k, n, ex, ex2 = ds.reduce((0.0, 0, 0.0, 0.0), reduce_fn)
      mean = k.numpy() + ex.numpy() / n.numpy()
      var = (ex2.numpy() - (ex.numpy() * ex.numpy()) / n.numpy()) / (
          n.numpy() - 1)
      layer.set_weights([mean, var])
      # Benchmarked code ends here.
      ends.append(time.time())

    avg_time = np.mean(np.array(ends) - np.array(starts))
    return avg_time

  def bm_adapt_implementation(self, num_elements, batch_size):
    """Test the KPL adapt implementation."""
    input_t = keras.Input(shape=(1,), dtype=tf.float32)
    layer = normalization.Normalization()
    _ = layer(input_t)

    num_repeats = 5
    starts = []
    ends = []
    for _ in range(num_repeats):
      ds = tf.data.Dataset.range(num_elements)
      ds = ds.map(
          lambda x: tf.expand_dims(tf.cast(x, tf.float32), -1))
      ds = ds.batch(batch_size)

      starts.append(time.time())
      # Benchmarked code begins here.
      layer.adapt(ds)
      # Benchmarked code ends here.
      ends.append(time.time())

    avg_time = np.mean(np.array(ends) - np.array(starts))
    name = "normalization_adapt|%s_elements|batch_%s" % (num_elements,
                                                         batch_size)
    baseline = self.run_dataset_implementation(num_elements, batch_size)
    extras = {
        "tf.data implementation baseline": baseline,
        "delta seconds": (baseline - avg_time),
        "delta percent": ((baseline - avg_time) / baseline) * 100
    }
    self.report_benchmark(
        iters=num_repeats, wall_time=avg_time, extras=extras, name=name)

  def benchmark_vocab_size_by_batch(self):
    for vocab_size in [100, 1000, 10000, 100000, 1000000]:
      for batch in [1, 16, 2048]:
        self.bm_adapt_implementation(vocab_size, batch)


if __name__ == "__main__":
  tf.test.main()
