"""Benchmark core layers.

To run benchmarks, see the following command for an example, please change the
flag to your custom value:

```
python3 -m benchmarks.layer_benchmark.core_benchmark \
    --benchmark_name=benchmark_dense \
    --num_samples=2048 \
    --batch_size=256 \
    --jit_compile=True
```
"""

import numpy as np
from absl import app
from absl import flags

from benchmarks.layer_benchmark.base_benchmark import LayerBenchmark

FLAGS = flags.FLAGS


def benchmark_dense(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "Dense"
    init_args = {"units": 256}
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

    benchmark.benchmark_train(
        num_samples=num_samples,
        batch_size=batch_size,
    )


def benchmark_einsum_dense(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "EinsumDense"
    init_args = {
        "equation": "abc,cd->abd",
        "output_shape": (None, 256),
    }
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

    benchmark.benchmark_train(
        num_samples=num_samples,
        batch_size=batch_size,
    )


def benchmark_embedding(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "Embedding"
    init_args = {
        "input_dim": 128,
        "output_dim": 256,
    }
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[
            256,
        ],
        jit_compile=jit_compile,
    )

    data = [np.random.randint(30, size=(num_samples, 256))]
    benchmark.benchmark_predict(
        num_samples=num_samples,
        batch_size=batch_size,
        data=data,
    )

    benchmark.benchmark_train(
        num_samples=num_samples,
        batch_size=batch_size,
        data=data,
    )


BENCHMARK_NAMES = {
    "benchmark_dense": benchmark_dense,
    "benchmark_einsum_dense": benchmark_einsum_dense,
    "benchmark_embedding": benchmark_embedding,
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
