""" Benchmark activation layers.

To run benchmarks, see the following command for an example, please change the
flag to your custom value:

```
python3 -m benchmarks.layer_benchmark.activation_benchmark \
    --benchmark_name=benchmark_elu \
    --num_samples=8192 \
    --batch_size=1024 \
    --jit_compile=True
```
"""

from absl import app
from absl import flags

from benchmarks.layer_benchmark.base_benchmark import LayerBenchmark

FLAGS = flags.FLAGS


def benchmark_elu(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "ELU"
    init_args = {}
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[256, 256],
        jit_compile=jit_compile,
    )

    benchmark.benchmark_predict(
        num_samples=num_samples,
        batch_size=batch_size,
    )


def benchmark_prelu(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "PReLU"
    init_args = {}
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[256, 256],
        jit_compile=jit_compile,
    )

    benchmark.benchmark_predict(
        num_samples=num_samples,
        batch_size=batch_size,
    )


def benchmark_softmax(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "Softmax"
    init_args = {}
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[256, 256],
        jit_compile=jit_compile,
    )

    benchmark.benchmark_predict(
        num_samples=num_samples,
        batch_size=batch_size,
    )


BENCHMARK_NAMES = {
    "benchmark_elu": benchmark_elu,
    "benchmark_prelu": benchmark_prelu,
    "benchmark_softmax": benchmark_softmax,
}


def main(_):
    benchmark_name = FLAGS.benchmark_name
    num_samples = FLAGS.num_samples
    batch_size = FLAGS.batch_size
    jit_compile = FLAGS.jit_compile

    if benchmark_name is None:
        for name, benchmark_fn in BENCHMARK_NAMES:
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
