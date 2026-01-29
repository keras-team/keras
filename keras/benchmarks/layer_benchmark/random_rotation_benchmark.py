import tensorflow as tf

from keras import layers
from keras.benchmarks.layer_benchmark.layer_benchmark import LayerBenchmark


def benchmark_random_rotation():
    benchmark = LayerBenchmark(
        layer=layers.RandomRotation(0.1),
        input_shape=(256, 224, 224, 3),
        num_batches=100,
    )

    # For preprocessing layers, predict is effectively a no-op,
    # but we still call it to follow the standard benchmark structure.
    benchmark.benchmark_predict()
    benchmark.benchmark_train()


BENCHMARK_NAMES = ("random_rotation",)


if __name__ == "__main__":
    benchmark_random_rotation()
