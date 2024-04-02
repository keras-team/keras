"""Tests for tf.distribute related functionality under tf implementation."""

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.eager import context

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.backend.tensorflow import trainer as tf_trainer


@pytest.mark.skipif(
    backend.backend() != "tensorflow",
    reason="The distribute test can only run with TF backend.",
)
class DistributeTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        # Need at least 2 devices for distribution related tests.
        cpus = tf.config.list_physical_devices("CPU")
        context._reset_context()
        tf.config.set_logical_device_configuration(
            cpus[0],
            [
                tf.config.LogicalDeviceConfiguration(),
                tf.config.LogicalDeviceConfiguration(),
            ],
        )

    def test_variable_creation(self):
        strategy = tf.distribute.MirroredStrategy(["CPU:0", "CPU:1"])
        with strategy.scope():
            dense = layers.Dense(2)
            dense.build([4, 2])

        self.assertIsInstance(dense.kernel, backend.Variable)
        self.assertIsInstance(
            dense.kernel.value, tf.distribute.DistributedValues
        )
        self.assertIn("MirroredVariable", dense.kernel.value.__class__.__name__)

        self.assertIsInstance(dense.kernel, backend.Variable)
        self.assertIsInstance(dense.bias.value, tf.distribute.DistributedValues)
        self.assertIn("MirroredVariable", dense.bias.value.__class__.__name__)

    def test_strategy_run(self):
        strategy = tf.distribute.MirroredStrategy(["CPU:0", "CPU:1"])

        with strategy.scope():
            inputs = layers.Input(shape=[4])
            dense = layers.Dense(2)
            output = dense(inputs)
            model = models.Functional(inputs, output)

        self.assertIsInstance(dense.kernel, backend.Variable)
        self.assertIsInstance(
            dense.kernel.value, tf.distribute.DistributedValues
        )

        def input_fn(ctx):
            if ctx.replica_id_in_sync_group == 1:
                return tf.ones([8, 4])
            else:
                return tf.zeros([8, 4])

        distributed_inputs = (
            strategy.experimental_distribute_values_from_function(input_fn)
        )

        @tf.function
        def run_fn(data):
            return model(data)

        result = strategy.run(run_fn, args=(distributed_inputs,))

        self.assertIsInstance(
            result, tf.types.experimental.distributed.PerReplica
        )
        self.assertLen(result.values, 2)
        self.assertEqual(result.values[0].shape, [8, 2])
        self.assertEqual(result.values[1].shape, [8, 2])
        self.assertNotAllClose(result.values[0], result.values[1])
        self.assertAllClose(result.values[0], tf.zeros([8, 2]))

    def test_epoch_iterator(self):
        x = np.random.random((100, 16))
        y = np.random.random((100, 4))
        sample_weight = np.random.random((100,))
        batch_size = 16
        shuffle = True

        strategy = tf.distribute.MirroredStrategy(["CPU:0", "CPU:1"])

        epoch_iterator = tf_trainer.TFEpochIterator(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            shuffle=shuffle,
            distribute_strategy=strategy,
        )
        steps_seen = []
        for step, data_iterator in epoch_iterator.enumerate_epoch():
            steps_seen.append(step)
            batch = next(data_iterator)
            self.assertEqual(len(batch), 3)
            x, y, sample_weight = batch
            self.assertTrue(
                isinstance(x, tf.types.experimental.distributed.PerReplica)
            )
            # Make sure the local batch size is 8
            if step < 6:
                self.assertEqual(x.values[0].shape, [8, 16])
                self.assertEqual(y.values[0].shape, [8, 4])
                self.assertEqual(sample_weight.values[0].shape, [8])
            else:
                # Last partial batch
                self.assertEqual(x.values[0].shape, [2, 16])
                self.assertEqual(y.values[0].shape, [2, 4])
                self.assertEqual(sample_weight.values[0].shape, [2])
        self.assertEqual(steps_seen, [0, 1, 2, 3, 4, 5, 6])
