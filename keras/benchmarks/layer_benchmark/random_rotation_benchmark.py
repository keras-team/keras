"""Benchmark RandomRotation layer."""

from absl import app
from absl import flags

from keras.benchmarks.layer_benchmark.base_benchmark import LayerBenchmark

FLAGS = flags.FLAGS


def benchmark_random_rotation(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "RandomRotation"
    init_args = {"factor": 0.1}

    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[224, 224, 3],
        jit_compile=jit_compile,
    )

    # Predict is effectively a no-op for preprocessing layers,
    # but we still call it to follow the standard benchmark structure.
    benchmark.benchmark_predict(
        num_samples=num_samples,
        batch_size=batch_size,
    )

    benchmark.benchmark_train(
        num_samples=num_samples,
        batch_size=batch_size,
    )


BENCHMARK_NAMES = {
    "benchmark_random_rotation": benchmark_random_rotation,
}


def main(_):
    benchmark_name = FLAGS.benchmark_name
    num_samples = FLAGS.num_samples
    batch_size = FLAGS.batch_size
    jit_compile = FLAGS.jit_compile

    if benchmark_name is None:
        for benchmark_fn in BENCHMARK_NAMES.values():
            benchmark_fn(num_samples, batch_size, jit_compile)
        return

    if benchmark_name not in BENCHMARK_NAMES:
        raise ValueError(
            f"Invalid benchmark name: {benchmark_name}, "
            f"`benchmark_name` must be one of {BENCHMARK_NAMES.keys()}"
        )

    BENCHMARK_NAMES[benchmark_name](num_samples, batch_size, jit_compile)


if __name__ == "__main__":
    app.run(main)
