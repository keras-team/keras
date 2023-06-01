import time

import numpy as np
import tensorflow as tf
from absl import flags

import keras_core

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "benchmark_name",
    None,
    "The name of benchmark to run.",
)

flags.DEFINE_integer(
    "num_samples",
    1000,
    "Number of input data samples.",
)

flags.DEFINE_integer(
    "batch_size",
    20,
    "Batch size of data.",
)

flags.DEFINE_bool(
    "jit_compile",
    True,
    "If True, the benchmark will run with XLA compilation.",
)


class BenchmarkMetricsCallback:
    def __init__(self, start_batch=2, stop_batch=None):
        self.start_batch = start_batch
        self.stop_batch = stop_batch

        self.state = {}

    def on_train_batch_begin(self, batch, logs=None):
        if batch == self.start_batch:
            self.state["benchmark_begin"] = time.time()

    def on_train_batch_end(self, batch, logs=None):
        if batch == self.stop_batch:
            self.state["benchmark_end"] = time.time()
            throughput = (self.stop_batch - self.start_batch) / (
                self.state["benchmark_end"] - self.state["benchmark_begin"]
            )
            self.state["throughput"] = throughput

    def on_predict_batch_begin(self, batch, logs=None):
        if batch == self.start_batch:
            self.state["benchmark_begin"] = time.time()

    def on_predict_batch_end(self, batch, logs=None):
        if batch == self.stop_batch:
            self.state["benchmark_end"] = time.time()
            throughput = (self.stop_batch - self.start_batch) / (
                self.state["benchmark_end"] - self.state["benchmark_begin"]
            )
            self.state["throughput"] = throughput


class KerasCoreBenchmarkMetricsCallback(keras_core.callbacks.Callback):
    def __init__(self, start_batch=2, stop_batch=None):
        self._callback = BenchmarkMetricsCallback(start_batch, stop_batch)

    def on_train_batch_begin(self, batch, logs=None):
        self._callback.on_train_batch_begin(batch, logs)

    def on_train_batch_end(self, batch, logs=None):
        self._callback.on_train_batch_end(batch, logs)

    def on_predict_batch_begin(self, batch, logs=None):
        self._callback.on_predict_batch_begin(batch, logs)

    def on_predict_batch_end(self, batch, logs=None):
        self._callback.on_predict_batch_end(batch, logs)


class TFKerasBenchmarkMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, start_batch=2, stop_batch=None):
        self._callback = BenchmarkMetricsCallback(start_batch, stop_batch)

    def on_train_batch_begin(self, batch, logs=None):
        self._callback.on_train_batch_begin(batch, logs)

    def on_train_batch_end(self, batch, logs=None):
        self._callback.on_train_batch_end(batch, logs)

    def on_predict_batch_begin(self, batch, logs=None):
        self._callback.on_predict_batch_begin(batch, logs)

    def on_predict_batch_end(self, batch, logs=None):
        self._callback.on_predict_batch_end(batch, logs)


class LayerBenchmark:
    def __init__(
        self,
        layer_name,
        init_args,
        input_shape,
        jit_compile=True,
    ):
        self.layer_name = layer_name
        self.input_shape = input_shape
        _keras_core_layer_class = getattr(keras_core.layers, layer_name)
        _tf_keras_layer_class = getattr(tf.keras.layers, layer_name)

        self._keras_core_layer = _keras_core_layer_class(**init_args)
        self._tf_keras_layer = _tf_keras_layer_class(**init_args)

        self._keras_core_model = keras_core.Sequential([self._keras_core_layer])
        self._tf_keras_model = tf.keras.Sequential([self._tf_keras_layer])

        self._keras_core_model.compile(
            loss="mse", optimizer="sgd", jit_compile=jit_compile
        )
        self._tf_keras_model.compile(
            loss="mse", optimizer="sgd", jit_compile=jit_compile
        )

        self.jit_compile = jit_compile
        self.input_shape = input_shape

    def benchmark_predict(self, num_samples, batch_size):
        data_shape = [num_samples] + list(self.input_shape)
        data = np.random.normal(size=data_shape)
        num_iterations = num_samples // batch_size - 1
        callback = KerasCoreBenchmarkMetricsCallback(stop_batch=num_iterations)
        tf_keras_callback = TFKerasBenchmarkMetricsCallback(
            stop_batch=num_iterations
        )

        self._keras_core_model.predict(
            data,
            batch_size=batch_size,
            callbacks=[callback],
        )

        self._tf_keras_model.predict(
            data,
            batch_size=batch_size,
            callbacks=[tf_keras_callback],
        )

        keras_core_throughput = callback._callback.state["throughput"]
        tf_keras_throughput = tf_keras_callback._callback.state["throughput"]
        print(
            f"Keras Core throughput of forward pass of {self.layer_name}: "
            f"{keras_core_throughput} samples/sec."
        )
        print(
            f"TF Keras throughput of forward pass of {self.layer_name}: "
            f"{tf_keras_throughput} samples/sec."
        )

    def benchmark_train(self, num_samples, batch_size):
        data_shape = [num_samples] + list(self.input_shape)
        data = np.random.normal(size=data_shape)
        # Scale by a small factor to avoid zero gradients.
        label = np.array(self._keras_core_layer(data)) * 1.001

        num_iterations = num_samples // batch_size - 1
        callback = KerasCoreBenchmarkMetricsCallback(stop_batch=num_iterations)
        tf_keras_callback = TFKerasBenchmarkMetricsCallback(
            stop_batch=num_iterations
        )

        self._keras_core_model.fit(
            data,
            label,
            batch_size=batch_size,
            callbacks=[callback],
        )
        self._tf_keras_model.fit(
            data,
            label,
            batch_size=batch_size,
            callbacks=[tf_keras_callback],
        )

        keras_core_throughput = callback._callback.state["throughput"]
        tf_keras_throughput = tf_keras_callback._callback.state["throughput"]
        print(
            f"Keras Core throughput of forward & backward pass of "
            f"{self.layer_name}: {keras_core_throughput} samples/sec."
        )
        print(
            f"TF Keras  throughput of forward & backward pass of "
            f"{self.layer_name}: {tf_keras_throughput} samples/sec."
        )
