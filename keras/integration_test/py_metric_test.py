"""Test Model.fit with a PyMetric."""

import tensorflow.compat.v2 as tf
from absl.testing import parameterized

from keras import Sequential
from keras import layers
from keras import losses
from keras import metrics
from keras.testing_infra import test_combinations


def get_dataset(num_batches=5, batch_size=2):
    x = tf.random.uniform((num_batches * batch_size, 100))
    y = tf.random.uniform((num_batches * batch_size, 2))
    dataset = (
        tf.data.Dataset.from_tensor_slices((x, y))
        .prefetch(batch_size * 2)
        .batch(batch_size)
    )
    return dataset


class CountingPyMetric(metrics.PyMetric):
    """A test-only PyMetric which simply counts how many results it's seen."""

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.y_pred.append(y_pred)

    def reset_state(self):
        self.y_pred = []

    def result(self):
        return len(self.y_pred)


class PyMetricTest(test_combinations.TestCase):
    @parameterized.named_parameters(("eager", True), ("graph", False))
    def test_fit(self, run_eagerly):
        num_batches = 5
        dataset = get_dataset(num_batches=num_batches)

        counting_metric = CountingPyMetric()

        model = Sequential(layers.Dense(2))
        model.compile(
            loss=losses.BinaryCrossentropy(),
            metrics=[counting_metric],
            run_eagerly=run_eagerly,
        )
        model.fit(dataset, epochs=1)

        self.assertEqual(counting_metric.result(), num_batches)

    @parameterized.named_parameters(("eager", True), ("graph", False))
    def test_evaluate(self, run_eagerly):
        num_batches = 5
        dataset = get_dataset(num_batches=num_batches)

        model = Sequential(layers.Dense(2))
        model.compile(
            loss=losses.BinaryCrossentropy(),
            metrics=[CountingPyMetric()],
            run_eagerly=run_eagerly,
        )
        loss, count = model.evaluate(dataset)

        self.assertEqual(count, num_batches)


if __name__ == "__main__":
    tf.test.main()
