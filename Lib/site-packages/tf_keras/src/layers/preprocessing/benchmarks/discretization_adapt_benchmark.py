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
"""Benchmark for TF-Keras discretization preprocessing layer's adapt method."""

import time

import numpy as np
import tensorflow.compat.v2 as tf

import tf_keras.src as keras
from tf_keras.src.layers.preprocessing import discretization

EPSILON = 0.1


def reduce_fn(state, values, epsilon=EPSILON):
    """tf.data.Dataset-friendly implementation of mean and variance."""

    (state_,) = state
    summary = discretization.summarize(values, epsilon)
    if np.sum(state_[:, 0]) == 0:
        return (summary,)
    return (discretization.merge_summaries(state_, summary, epsilon),)


class BenchmarkAdapt(tf.test.Benchmark):
    """Benchmark adapt."""

    def run_dataset_implementation(self, num_elements, batch_size):
        input_t = keras.Input(shape=(1,))
        layer = discretization.Discretization()
        _ = layer(input_t)

        num_repeats = 5
        starts = []
        ends = []
        for _ in range(num_repeats):
            ds = tf.data.Dataset.range(num_elements)
            ds = ds.map(lambda x: tf.expand_dims(tf.cast(x, tf.float32), -1))
            ds = ds.batch(batch_size)

            starts.append(time.time())
            # Benchmarked code begins here.
            state = ds.reduce((np.zeros((1, 2)),), reduce_fn)

            bins = discretization.get_bucket_boundaries(state, 100)
            layer.set_weights([bins])
            # Benchmarked code ends here.
            ends.append(time.time())

        avg_time = np.mean(np.array(ends) - np.array(starts))
        return avg_time

    def bm_adapt_implementation(self, num_elements, batch_size):
        """Test the KPL adapt implementation."""
        input_t = keras.Input(shape=(1,), dtype=tf.float32)
        layer = discretization.Discretization()
        _ = layer(input_t)

        num_repeats = 5
        starts = []
        ends = []
        for _ in range(num_repeats):
            ds = tf.data.Dataset.range(num_elements)
            ds = ds.map(lambda x: tf.expand_dims(tf.cast(x, tf.float32), -1))
            ds = ds.batch(batch_size)

            starts.append(time.time())
            # Benchmarked code begins here.
            layer.adapt(ds)
            # Benchmarked code ends here.
            ends.append(time.time())

        avg_time = np.mean(np.array(ends) - np.array(starts))
        name = "discretization_adapt|%s_elements|batch_%s" % (
            num_elements,
            batch_size,
        )
        baseline = self.run_dataset_implementation(num_elements, batch_size)
        extras = {
            "tf.data implementation baseline": baseline,
            "delta seconds": (baseline - avg_time),
            "delta percent": ((baseline - avg_time) / baseline) * 100,
        }
        self.report_benchmark(
            iters=num_repeats, wall_time=avg_time, extras=extras, name=name
        )

    def benchmark_vocab_size_by_batch(self):
        for vocab_size in [100, 1000, 10000, 100000, 1000000]:
            for batch in [64 * 2048]:
                self.bm_adapt_implementation(vocab_size, batch)


if __name__ == "__main__":
    tf.test.main()

