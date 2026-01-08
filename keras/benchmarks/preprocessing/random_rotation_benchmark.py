"""
Benchmark for keras.layers.RandomRotation inside tf.data pipelines.

Measures end-to-end dataset iteration time for:
- eager execution
- tf.function-wrapped execution
"""

import time

import tensorflow as tf

from keras import layers


def make_dataset(
    batch_size=32,
    image_size=224,
    num_batches=100,
):
    images = tf.random.uniform(
        shape=(batch_size * num_batches, image_size, image_size, 3),
        dtype=tf.float32,
    )
    ds = tf.data.Dataset.from_tensor_slices(images)
    ds = ds.batch(batch_size)
    return ds


def run_benchmark(use_tf_function: bool):
    layer = layers.RandomRotation(0.1)

    def apply(x):
        return layer(x, training=True)

    if use_tf_function:
        apply = tf.function(apply)

    ds = make_dataset()
    ds = ds.map(apply, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    # Warm up run to compile the function and fill the pipeline.
    for _ in ds:
        pass

    # Timed runs.
    durations = []
    for _ in range(5):  # Run multiple times to get more stable results.
        start = time.time()
        for _ in ds:
            pass
        durations.append(time.time() - start)
    duration = min(durations)

    mode = "tf.function" if use_tf_function else "eager"
    print(f"{mode:12s} | {duration:.3f} sec")


if __name__ == "__main__":
    print("RandomRotation benchmark")
    run_benchmark(use_tf_function=False)
    run_benchmark(use_tf_function=True)
