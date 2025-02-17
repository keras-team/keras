"""Benchmark attention layers.

To run benchmarks, see the following command for an example, please change the
flag to your custom value:

```
python3 -m benchmarks.layer_benchmark.attention_benchmark \
    --benchmark_name=benchmark_attention \
    --num_samples=2048 \
    --batch_size=256 \
    --jit_compile=True
```
"""

from absl import app
from absl import flags

from benchmarks.layer_benchmark.base_benchmark import LayerBenchmark

FLAGS = flags.FLAGS


def benchmark_attention(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "Attention"
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


def benchmark_multi_head_attention(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "MultiHeadAttention"
    init_args = {
        "num_heads": 4,
        "key_dim": 16,
    }
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[[256, 64], [256, 64], [256, 64]],
        flat_call_inputs=True,
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


def benchmark_additive_attention(
    num_samples,
    batch_size,
    jit_compile=True,
):
    layer_name = "AdditiveAttention"
    init_args = {}
    benchmark = LayerBenchmark(
        layer_name,
        init_args,
        input_shape=[[256, 64], [256, 64], [256, 64]],
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
    "benchmark_attention": benchmark_attention,
    "benchmark_multi_head_attention": benchmark_multi_head_attention,
    "benchmark_additive_attention": benchmark_additive_attention,
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
