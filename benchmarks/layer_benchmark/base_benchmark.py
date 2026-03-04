import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
from absl import flags

import keras

FLAGS = flags.FLAGS

_RESULT_PREFIX = "BENCHMARK_RESULT:"
_BENCHMARK_BACKEND_ENV = "_BENCHMARK_BACKEND"
_single_backend_mode = os.environ.get(_BENCHMARK_BACKEND_ENV) is not None

try:
    import tensorflow as tf
except ImportError:
    tf = None

flags.DEFINE_string(
    "benchmark_name",
    None,
    "The name of benchmark to run. If None, all benchmarks in the file "
    "will be run.",
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

flags.DEFINE_string(
    "backend",
    None,
    "Comma-separated list of Keras backends to compare "
    "(e.g. jax,torch,tensorflow). When set, runs the benchmark with each "
    "backend in a separate process and prints a comparison.",
)

flags.DEFINE_string(
    "output_format",
    "table",
    "Output format for multi-backend benchmarks: 'table' (default) or "
    "'json'.",
)


def _get_hardware_info():
    return platform.machine()


class BenchmarkMetricsCallback:
    def __init__(self, start_batch=1, stop_batch=None):
        self.start_batch = start_batch
        self.stop_batch = stop_batch

        self.state = {}

    def on_train_batch_begin(self, batch, logs=None):
        if batch == self.start_batch:
            self.state["benchmark_begin"] = time.time()

    def on_train_batch_end(self, batch, logs=None):
        if batch == self.stop_batch:
            self.state["benchmark_end"] = time.time()
            throughput = (self.stop_batch - self.start_batch + 1) / (
                self.state["benchmark_end"] - self.state["benchmark_begin"]
            )
            self.state["throughput"] = throughput

    def on_predict_batch_begin(self, batch, logs=None):
        if batch == self.start_batch:
            self.state["benchmark_begin"] = time.time()

    def on_predict_batch_end(self, batch, logs=None):
        if batch == self.stop_batch:
            self.state["benchmark_end"] = time.time()
            throughput = (self.stop_batch - self.start_batch + 1) / (
                self.state["benchmark_end"] - self.state["benchmark_begin"]
            )
            self.state["throughput"] = throughput


class KerasCoreBenchmarkMetricsCallback(keras.callbacks.Callback):
    def __init__(self, start_batch=1, stop_batch=None):
        self._callback = BenchmarkMetricsCallback(start_batch, stop_batch)

    def on_train_batch_begin(self, batch, logs=None):
        self._callback.on_train_batch_begin(batch, logs)

    def on_train_batch_end(self, batch, logs=None):
        self._callback.on_train_batch_end(batch, logs)

    def on_predict_batch_begin(self, batch, logs=None):
        self._callback.on_predict_batch_begin(batch, logs)

    def on_predict_batch_end(self, batch, logs=None):
        self._callback.on_predict_batch_end(batch, logs)


if tf is not None:

    class TFKerasBenchmarkMetricsCallback(tf.keras.callbacks.Callback):
        def __init__(self, start_batch=1, stop_batch=None):
            self._callback = BenchmarkMetricsCallback(
                start_batch, stop_batch
            )

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
        flat_call_inputs=True,
        jit_compile=True,
        keras_layer=None,
        tf_keras_layer=None,
    ):
        self.layer_name = layer_name
        self._single_backend = _single_backend_mode

        _keras_layer_class = getattr(keras.layers, layer_name)

        if keras_layer is None:
            # Sometimes you want to initialize the keras layer and tf_keras
            # layer in a different way. For example, `Bidirectional` layer,
            # which takes in `keras.layers.Layer` and
            # `tf.keras.layer.Layer` separately.
            self._keras_layer = _keras_layer_class(**init_args)
        else:
            self._keras_layer = keras_layer

        self.input_shape = input_shape
        self._keras_model = self._build_keras_model(
            input_shape, flat_call_inputs
        )
        self._keras_model.compile(
            loss="mse", optimizer="sgd", jit_compile=jit_compile
        )

        if not self._single_backend:
            # Legacy mode: also build tf.keras model for comparison.
            if tf is None:
                raise ImportError(
                    "TensorFlow is required for the default benchmark mode "
                    "(Keras 3 vs tf.keras comparison). Install TensorFlow "
                    "or use --backend to run multi-backend comparisons."
                )
            _tf_keras_layer_class = getattr(tf.keras.layers, layer_name)

            if tf_keras_layer is None:
                self._tf_keras_layer = _tf_keras_layer_class(**init_args)
            else:
                self._tf_keras_layer = tf_keras_layer

            self._tf_keras_model = self._build_tf_keras_model(
                input_shape, flat_call_inputs
            )
            self._tf_keras_model.compile(
                loss="mse", optimizer="sgd", jit_compile=jit_compile
            )

        self.flat_call_inputs = flat_call_inputs
        self.jit_compile = jit_compile
        self.input_shape = input_shape

    def _build_keras_model(self, input_shape, flat_call_inputs=True):
        inputs = []
        if not isinstance(input_shape[0], (tuple, list)):
            input_shape = [input_shape]

        for shape in input_shape:
            inputs.append(keras.Input(shape=shape))

        if flat_call_inputs:
            outputs = self._keras_layer(*inputs)
        else:
            outputs = self._keras_layer(inputs)
        return keras.Model(inputs=inputs, outputs=outputs)

    def _build_tf_keras_model(self, input_shape, flat_call_inputs=True):
        inputs = []
        if not isinstance(input_shape[0], (tuple, list)):
            input_shape = [input_shape]

        for shape in input_shape:
            inputs.append(tf.keras.Input(shape=shape))

        if flat_call_inputs:
            outputs = self._tf_keras_layer(*inputs)
        else:
            outputs = self._tf_keras_layer(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def benchmark_predict(self, num_samples, batch_size, data=None):
        if data is None:
            # Generate default data if not provided.
            if isinstance(self.input_shape[0], (tuple, list)):
                # The layer has multiple inputs.
                data = []
                for data_shape in self.input_shape:
                    data_shape = [num_samples] + list(data_shape)
                    data.append(np.random.normal(size=data_shape))
            else:
                data_shape = [num_samples] + list(self.input_shape)
                data = np.random.normal(size=data_shape)

        num_iterations = num_samples // batch_size - 1
        callback = KerasCoreBenchmarkMetricsCallback(
            stop_batch=num_iterations
        )

        self._keras_model.predict(
            data,
            batch_size=batch_size,
            callbacks=[callback],
            verbose=0 if self._single_backend else "auto",
        )

        keras_throughput = callback._callback.state["throughput"] * batch_size

        if self._single_backend:
            result = {
                "backend": os.environ.get(_BENCHMARK_BACKEND_ENV),
                "layer_name": self.layer_name,
                "benchmark_type": "predict",
                "mean_samples_per_sec": round(keras_throughput, 2),
                "hardware": _get_hardware_info(),
            }
            print(f"{_RESULT_PREFIX}{json.dumps(result)}")
        else:
            tf_keras_callback = TFKerasBenchmarkMetricsCallback(
                stop_batch=num_iterations
            )
            self._tf_keras_model.predict(
                data,
                batch_size=batch_size,
                callbacks=[tf_keras_callback],
            )
            tf_keras_throughput = (
                tf_keras_callback._callback.state["throughput"] * batch_size
            )
            print(
                f"Keras 3 throughput of forward pass of "
                f"{self.layer_name}: "
                f"{keras_throughput:.2f} samples/sec."
            )
            print(
                f"TF Keras throughput of forward pass of "
                f"{self.layer_name}: "
                f"{tf_keras_throughput:.2f} samples/sec."
            )

    def benchmark_train(self, num_samples, batch_size, data=None, label=None):
        if data is None:
            # Generate default data if not provided.
            if isinstance(self.input_shape[0], (tuple, list)):
                # The layer has multiple inputs.
                data = []
                for data_shape in self.input_shape:
                    data_shape = [num_samples] + list(data_shape)
                    data.append(np.random.normal(size=data_shape))
            else:
                data_shape = [num_samples] + list(self.input_shape)
                data = [np.random.normal(size=data_shape)]

        if label is None:
            # Generate default label if not provided.
            if self.flat_call_inputs:
                # Scale by a small factor to avoid zero gradients.
                label = (
                    keras.backend.convert_to_numpy(self._keras_layer(*data))
                    * 1.001
                )
            else:
                label = (
                    keras.backend.convert_to_numpy(self._keras_layer(data))
                    * 1.001
                )

        num_iterations = num_samples // batch_size - 1
        callback = KerasCoreBenchmarkMetricsCallback(
            stop_batch=num_iterations
        )

        self._keras_model.fit(
            data,
            label,
            batch_size=batch_size,
            callbacks=[callback],
            verbose=0 if self._single_backend else "auto",
        )

        keras_throughput = callback._callback.state["throughput"] * batch_size

        if self._single_backend:
            result = {
                "backend": os.environ.get(_BENCHMARK_BACKEND_ENV),
                "layer_name": self.layer_name,
                "benchmark_type": "train",
                "mean_samples_per_sec": round(keras_throughput, 2),
                "hardware": _get_hardware_info(),
            }
            print(f"{_RESULT_PREFIX}{json.dumps(result)}")
        else:
            tf_keras_callback = TFKerasBenchmarkMetricsCallback(
                stop_batch=num_iterations
            )
            self._tf_keras_model.fit(
                data,
                label,
                batch_size=batch_size,
                callbacks=[tf_keras_callback],
            )
            tf_keras_throughput = (
                tf_keras_callback._callback.state["throughput"] * batch_size
            )
            print(
                f"Keras 3 throughput of forward & backward pass of "
                f"{self.layer_name}: {keras_throughput:.2f} samples/sec."
            )
            print(
                f"TF Keras  throughput of forward & backward pass of "
                f"{self.layer_name}: {tf_keras_throughput:.2f} samples/sec."
            )


def _print_comparison_table(results):
    """Print benchmark results as a formatted comparison table."""
    groups = {}
    backends = []
    for r in results:
        key = (r["layer_name"], r["benchmark_type"])
        if key not in groups:
            groups[key] = {}
        groups[key][r["backend"]] = r["mean_samples_per_sec"]
        if r["backend"] not in backends:
            backends.append(r["backend"])

    if not backends:
        return

    label_width = max(
        max(len(f"{ln} ({bt})") for ln, bt in groups), len("Benchmark")
    )
    col_width = max(max(len(b) for b in backends), 14)

    header = f"| {'Benchmark':<{label_width}} |"
    separator = f"|{'-' * (label_width + 2)}|"
    for b in backends:
        header += f" {b:>{col_width}} |"
        separator += f"{'-' * (col_width + 2)}|"

    print()
    print(separator)
    print(header)
    print(separator)

    for (layer_name, benchmark_type), backend_results in groups.items():
        label = f"{layer_name} ({benchmark_type})"
        row = f"| {label:<{label_width}} |"
        for b in backends:
            value = backend_results.get(b)
            if value is not None:
                formatted = f"{value:.2f}"
                row += f" {formatted:>{col_width}} |"
            else:
                row += f" {'N/A':>{col_width}} |"
        print(row)

    print(separator)
    print()


def run_multi_backend_benchmark(module_path, benchmark_fn_map):
    """Run benchmarks across multiple Keras backends and display comparison.

    Spawns a separate process for each backend with the appropriate
    KERAS_BACKEND environment variable and collects results.

    Args:
        module_path: Python module path for the benchmark script
            (e.g. "benchmarks.layer_benchmark.conv_benchmark").
        benchmark_fn_map: Dict mapping benchmark names to functions, used
            to validate the benchmark_name flag.
    """
    backends = [b.strip() for b in FLAGS.backend.split(",")]
    valid_backends = {"jax", "torch", "tensorflow"}
    for b in backends:
        if b not in valid_backends:
            raise ValueError(
                f"Invalid backend: {b}. Must be one of {valid_backends}."
            )

    benchmark_name = FLAGS.benchmark_name
    if benchmark_name is not None and benchmark_name not in benchmark_fn_map:
        raise ValueError(
            f"Invalid benchmark name: {benchmark_name}, `benchmark_name` "
            f"must be one of {benchmark_fn_map.keys()}"
        )

    all_results = []

    for backend in backends:
        env = os.environ.copy()
        env["KERAS_BACKEND"] = backend
        env[_BENCHMARK_BACKEND_ENV] = backend

        cmd = [
            sys.executable,
            "-m",
            module_path,
            f"--num_samples={FLAGS.num_samples}",
            f"--batch_size={FLAGS.batch_size}",
            f"--jit_compile={FLAGS.jit_compile}",
        ]
        if benchmark_name:
            cmd.append(f"--benchmark_name={benchmark_name}")

        print(f"Running benchmarks with {backend} backend...")
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=env
        )

        if proc.returncode != 0:
            print(f"  Warning: {backend} backend failed:")
            stderr_lines = proc.stderr.strip().splitlines()
            for line in stderr_lines[-5:]:
                print(f"    {line}")
            continue

        for line in proc.stdout.splitlines():
            if line.startswith(_RESULT_PREFIX):
                result = json.loads(line[len(_RESULT_PREFIX) :])
                all_results.append(result)

    if not all_results:
        print("No benchmark results collected.")
        return

    if FLAGS.output_format == "json":
        print(json.dumps(all_results, indent=2))
    else:
        _print_comparison_table(all_results)
