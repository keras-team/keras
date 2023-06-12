"""Benchmark normalization layers.

To run benchmarks, see the following command for an example, please change the
flag to your custom value:

```
python3 -m benchmarks.layer_benchmark.normalization_benchmark \
    --benchmark_name=benchmark_batch_normalization \
    --num_samples=2048 \
    --batch_size=256 \
    --jit_compile=True
```
"""

from absl import app
from absl import flags

from benchmarks.layer_benchmark.base_benchmark import LayerBenchmark

FLAGS = flags.FLAGS


def benchmark_batch_normalization(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "BatchNormalization"
    init_args = {}
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[256, 256, 4],
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


def benchmark_group_normalization(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "GroupNormalization"
    init_args = {
        "groups": 2,
    }
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[256, 256, 4],
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


def benchmark_layer_normalization(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "LayerNormalization"
    init_args = {}
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[256, 128, 4],
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


def benchmark_unit_normalization(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "UnitNormalization"
    init_args = {}
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[256, 128, 4],
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
    "benchmark_batch_normalization": benchmark_batch_normalization,
    "benchmark_group_normalization": benchmark_group_normalization,
    "benchmark_layer_normalization": benchmark_layer_normalization,
    "benchmark_unit_normalization": benchmark_unit_normalization,
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
