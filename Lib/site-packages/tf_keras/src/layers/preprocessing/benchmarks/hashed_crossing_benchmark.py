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
"""Benchmark for KPL implementation of categorical cross hash columns with dense
inputs."""


import tensorflow.compat.v2 as tf

import tf_keras.src as keras
from tf_keras.src.layers.preprocessing import hashed_crossing
from tf_keras.src.layers.preprocessing.benchmarks import (
    feature_column_benchmark as fc_bm,
)

# isort: off
from tensorflow.python.eager.def_function import (
    function as tf_function,
)

NUM_REPEATS = 10
BATCH_SIZES = [32, 256]


def embedding_varlen(batch_size):
    """Benchmark a variable-length embedding."""
    # Data and constants.
    num_buckets = 10000
    data_a = tf.random.uniform(
        shape=(batch_size * NUM_REPEATS, 1), maxval=32768, dtype=tf.int64
    )
    data_b = tf.strings.as_string(data_a)

    # TF-Keras implementation
    input_1 = keras.Input(shape=(1,), name="data_a", dtype=tf.int64)
    input_2 = keras.Input(shape=(1,), name="data_b", dtype=tf.string)
    outputs = hashed_crossing.HashedCrossing(num_buckets)([input_1, input_2])
    model = keras.Model([input_1, input_2], outputs)

    # FC implementation
    fc = tf.feature_column.crossed_column(["data_a", "data_b"], num_buckets)

    # Wrap the FC implementation in a tf.function for a fair comparison
    @tf_function()
    def fc_fn(tensors):
        fc.transform_feature(
            tf.__internal__.feature_column.FeatureTransformationCache(tensors),
            None,
        )

    # Benchmark runs
    keras_data = {
        "data_a": data_a,
        "data_b": data_b,
    }
    k_avg_time = fc_bm.run_keras(keras_data, model, batch_size, NUM_REPEATS)

    fc_data = {
        "data_a": data_a,
        "data_b": data_b,
    }
    fc_avg_time = fc_bm.run_fc(fc_data, fc_fn, batch_size, NUM_REPEATS)

    return k_avg_time, fc_avg_time


class BenchmarkLayer(fc_bm.LayerBenchmark):
    """Benchmark the layer forward pass."""

    def benchmark_layer(self):
        for batch in BATCH_SIZES:
            name = f"hashed_cross|dense|batch_{batch}"
            k_time, f_time = embedding_varlen(batch_size=batch)
            self.report(name, k_time, f_time, NUM_REPEATS)


if __name__ == "__main__":
    tf.test.main()

