import tensorflow as tf

from keras import layers
from keras.benchmarks.layer_benchmark.layer_benchmark import LayerBenchmark


class RandomRotationBenchmark(LayerBenchmark):
    def __init__(self):
        super().__init__(
            layer=layers.RandomRotation(0.1),
            input_shape=(256, 224, 224, 3),
            num_batches=100,
        )

    def make_dataset(self):
        images = tf.random.uniform(self.input_shape, dtype=tf.float32)
        ds = tf.data.Dataset.from_tensor_slices(images)
        ds = ds.batch(32)
        return ds
