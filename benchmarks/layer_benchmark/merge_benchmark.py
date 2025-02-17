"""Benchmark merge layers.

To run benchmarks, see the following command for an example, please change the
flag to your custom value:

```
python3 -m benchmarks.layer_benchmark.merge_benchmark \
    --benchmark_name=benchmark_add \
    --num_samples=2048 \
    --batch_size=256 \
    --jit_compile=True
```
"""

from absl import app
from absl import flags

from benchmarks.layer_benchmark.base_benchmark import LayerBenchmark

FLAGS = flags.FLAGS


def benchmark_add(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "Add"
    init_args = {}
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[[256, 256], [256, 256]],
        flat_call_inputs=False,
        jit_compile=jit_compile,
    )

    benchmark.benchmark_predict(
        num_samples=num_samples,
        batch_size=batch_size,
    )

    benchmark.benchmark_train(
        num_samples=num_samples,
        batch_size=batch_size,
    )


def benchmark_average(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "Average"
    init_args = {}
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[[256, 256], [256, 256]],
        flat_call_inputs=False,
        jit_compile=jit_compile,
    )

    benchmark.benchmark_predict(
        num_samples=num_samples,
        batch_size=batch_size,
    )

    benchmark.benchmark_train(
        num_samples=num_samples,
        batch_size=batch_size,
    )


def benchmark_concatenate(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "Concatenate"
    init_args = {}
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[[256, 256], [256, 256]],
        flat_call_inputs=False,
        jit_compile=jit_compile,
    )

    benchmark.benchmark_predict(
        num_samples=num_samples,
        batch_size=batch_size,
    )

    benchmark.benchmark_train(
        num_samples=num_samples,
        batch_size=batch_size,
    )


def benchmark_dot(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "Dot"
    init_args = {"axes": [2, 1]}
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[[256, 32], [32, 64]],
        flat_call_inputs=False,
        jit_compile=jit_compile,
    )

    benchmark.benchmark_predict(
        num_samples=num_samples,
        batch_size=batch_size,
    )

    benchmark.benchmark_train(
        num_samples=num_samples,
        batch_size=batch_size,
    )


def benchmark_maximum(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "Maximum"
    init_args = {}
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[[256, 256], [256, 256]],
        flat_call_inputs=False,
        jit_compile=jit_compile,
    )

    benchmark.benchmark_predict(
        num_samples=num_samples,
        batch_size=batch_size,
    )

    benchmark.benchmark_train(
        num_samples=num_samples,
        batch_size=batch_size,
    )


def benchmark_minimum(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "Minimum"
    init_args = {}
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[[256, 256], [256, 256]],
        flat_call_inputs=False,
        jit_compile=jit_compile,
    )

    benchmark.benchmark_predict(
        num_samples=num_samples,
        batch_size=batch_size,
    )

    benchmark.benchmark_train(
        num_samples=num_samples,
        batch_size=batch_size,
    )


def benchmark_multiply(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "Multiply"
    init_args = {}
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[[256, 64], [256, 64]],
        flat_call_inputs=False,
        jit_compile=jit_compile,
    )

    benchmark.benchmark_predict(
        num_samples=num_samples,
        batch_size=batch_size,
    )

    benchmark.benchmark_train(
        num_samples=num_samples,
        batch_size=batch_size,
    )


def benchmark_subtract(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "Subtract"
    init_args = {}
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[[256, 256], [256, 256]],
        flat_call_inputs=False,
        jit_compile=jit_compile,
    )

    benchmark.benchmark_predict(
        num_samples=num_samples,
        batch_size=batch_size,
    )

    benchmark.benchmark_train(
        num_samples=num_samples,
        batch_size=batch_size,
    )


BENCHMARK_NAMES = {
    "benchmark_add": benchmark_add,
    "benchmark_average": benchmark_average,
    "benchmark_concatenate": benchmark_concatenate,
    "benchmark_dot": benchmark_dot,
    "benchmark_maximum": benchmark_maximum,
    "benchmark_minimum": benchmark_minimum,
    "benchmark_multiply": benchmark_multiply,
    "benchmark_subtract": benchmark_subtract,
}


def main(_):
    benchmark_name = FLAGS.benchmark_name
    num_samples = FLAGS.num_samples
    batch_size = FLAGS.batch_size
    jit_compile = FLAGS.jit_compile

    if benchmark_name is None:
        for name, benchmark_fn in BENCHMARK_NAMES.items():
            benchmark_fn(num_samples, batch_size, jit_compile)
        return

    if benchmark_name not in BENCHMARK_NAMES:
        raise ValueError(
            f"Invalid benchmark name: {benchmark_name}, `benchmark_name` must "
            f"be one of {BENCHMARK_NAMES.keys()}"
        )
    benchmark_fn = BENCHMARK_NAMES[benchmark_name]
    benchmark_fn(num_samples, batch_size, jit_compile)


if __name__ == "__main__":
    app.run(main)
