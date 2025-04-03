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
"""Benchmark for KPL implementation of vocabulary columns + indicator from lists
with varying-length inputs."""

import tensorflow.compat.v2 as tf

import tf_keras.src as keras
from tf_keras.src.layers.preprocessing import category_encoding
from tf_keras.src.layers.preprocessing import string_lookup
from tf_keras.src.layers.preprocessing.benchmarks import (
    feature_column_benchmark as fc_bm,
)

# isort: off
from tensorflow.python.eager.def_function import (
    function as tf_function,
)

NUM_REPEATS = 10
BATCH_SIZES = [32, 256]


def embedding_varlen(batch_size, max_length):
    """Benchmark a variable-length embedding."""
    # Data and constants.
    vocab_size = 32768
    vocab = fc_bm.create_vocabulary(vocab_size)
    data = fc_bm.create_string_data(
        max_length, batch_size * NUM_REPEATS, vocab, pct_oov=0.15
    )

    # TF-Keras implementation
    model = keras.Sequential()
    model.add(
        keras.Input(
            shape=(max_length,), name="data", ragged=True, dtype=tf.string
        )
    )
    model.add(string_lookup.StringLookup(vocabulary=vocab, mask_token=None))
    model.add(
        category_encoding.CategoryEncoding(
            num_tokens=vocab_size + 1, output_mode="count"
        )
    )

    # FC implementation
    fc = tf.feature_column.indicator_column(
        tf.feature_column.sequence_categorical_column_with_vocabulary_list(
            key="data", vocabulary_list=vocab, num_oov_buckets=1
        )
    )

    # Wrap the FC implementation in a tf.function for a fair comparison
    @tf_function()
    def fc_fn(tensors):
        fc.transform_feature(
            tf.__internal__.feature_column.FeatureTransformationCache(tensors),
            None,
        )

    # Benchmark runs
    keras_data = {"data": data}
    k_avg_time = fc_bm.run_keras(keras_data, model, batch_size, NUM_REPEATS)

    fc_data = {"data": data.to_sparse()}
    fc_avg_time = fc_bm.run_fc(fc_data, fc_fn, batch_size, NUM_REPEATS)

    return k_avg_time, fc_avg_time


class BenchmarkLayer(fc_bm.LayerBenchmark):
    """Benchmark the layer forward pass."""

    def benchmark_layer(self):
        for batch in BATCH_SIZES:
            name = f"vocab_list_indicator|varlen|batch_{batch}"
            k_time, f_time = embedding_varlen(batch_size=batch, max_length=256)
            self.report(name, k_time, f_time, NUM_REPEATS)


if __name__ == "__main__":
    tf.test.main()

