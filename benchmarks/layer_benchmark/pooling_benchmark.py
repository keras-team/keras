"""Benchmark pooling layers.

To run benchmarks, see the following command for an example, please change the
flag to your custom value:

```
python3 -m benchmarks.layer_benchmark.pooling_benchmark \
    --benchmark_name=benchmark_max_pooling1d \
    --num_samples=2048 \
    --batch_size=256 \
    --jit_compile=True
```
"""


from absl import app
from absl import flags

from benchmarks.layer_benchmark.base_benchmark import LayerBenchmark

FLAGS = flags.FLAGS


def benchmark_average_pooling1d(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "AveragePooling1D"
    init_args = {
        "pool_size": 2,
    }
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[1024, 256],
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


def benchmark_average_pooling2d(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "AveragePooling2D"
    init_args = {
        "pool_size": 2,
    }
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[256, 256, 3],
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


def benchmark_average_pooling3d(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "AveragePooling3D"
    init_args = {
        "pool_size": 2,
    }
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[64, 64, 32, 3],
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


def benchmark_max_pooling1d(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "MaxPooling1D"
    init_args = {
        "pool_size": 2,
    }
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[1024, 256],
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


def benchmark_max_pooling2d(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "MaxPooling2D"
    init_args = {
        "pool_size": 2,
    }
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[256, 256, 3],
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


def benchmark_max_pooling3d(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "MaxPooling3D"
    init_args = {
        "pool_size": 2,
    }
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[64, 64, 32, 3],
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


def benchmark_global_average_pooling1d(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "GlobalAveragePooling1D"
    init_args = {}
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[1024, 256],
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


def benchmark_global_average_pooling2d(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "GlobalAveragePooling2D"
    init_args = {}
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[256, 256, 3],
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


def benchmark_global_average_pooling3d(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "GlobalAveragePooling3D"
    init_args = {}
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[64, 64, 32, 3],
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


def benchmark_global_max_pooling1d(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "GlobalMaxPooling1D"
    init_args = {}
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[1024, 256],
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


def benchmark_global_max_pooling2d(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "GlobalMaxPooling2D"
    init_args = {}
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[256, 256, 3],
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


def benchmark_global_max_pooling3d(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "GlobalMaxPooling3D"
    init_args = {}
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[64, 64, 32, 3],
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
    "benchmark_average_pooling1d": benchmark_average_pooling1d,
    "benchmark_average_pooling2d": benchmark_average_pooling2d,
    "benchmark_average_pooling3d": benchmark_average_pooling3d,
    "benchmark_max_pooling1d": benchmark_max_pooling1d,
    "benchmark_max_pooling2d": benchmark_max_pooling2d,
    "benchmark_max_pooling3d": benchmark_max_pooling3d,
    "benchmark_global_average_pooling1d": benchmark_global_average_pooling1d,
    "benchmark_global_average_pooling2d": benchmark_global_average_pooling2d,
    "benchmark_global_average_pooling3d": benchmark_global_average_pooling3d,
    "benchmark_global_max_pooling1d": benchmark_global_max_pooling1d,
    "benchmark_global_max_pooling2d": benchmark_global_max_pooling2d,
    "benchmark_global_max_pooling3d": benchmark_global_max_pooling3d,
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
