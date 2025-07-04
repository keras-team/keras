"""Tests for tf.distribute related functionality under tf implementation."""

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.eager import context

import keras
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
        for step, _, data_iterator in epoch_iterator:
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

    def test_variable_aggregation(self):
        strategy = tf.distribute.MirroredStrategy(["CPU:0", "CPU:1"])

        with strategy.scope():
            x = np.random.random((4, 4))
            v1 = backend.Variable(x, dtype="float32")
            self.assertEqual(v1.aggregation, "none")
            self.assertEqual(v1.value.aggregation, tf.VariableAggregation.NONE)

            v2 = backend.Variable(x, dtype="float32", aggregation="sum")
            self.assertEqual(v2.aggregation, "sum")
            self.assertEqual(v2.value.aggregation, tf.VariableAggregation.SUM)

    def test_variable_synchronization(self):
        strategy = tf.distribute.MirroredStrategy(["CPU:0", "CPU:1"])

        with strategy.scope():
            x = np.random.random((4, 4))
            v1 = backend.Variable(x, dtype="float32")
            self.assertEqual(v1.synchronization, "auto")
            # AUTO with MirroredStrategy defaults to ON_WRITE
            self.assertEqual(
                v1.value.synchronization, tf.VariableSynchronization.ON_WRITE
            )

            v2 = backend.Variable(x, dtype="float32", synchronization="on_read")
            self.assertEqual(v2.synchronization, "on_read")
            self.assertEqual(
                v2.value.synchronization, tf.VariableSynchronization.ON_READ
            )

    def test_seed_generator(self):
        strategy = tf.distribute.MirroredStrategy(["CPU:0", "CPU:1"])
        with strategy.scope():
            seed_generator = keras.random.SeedGenerator(42)
            states = strategy.run(lambda: seed_generator.state.value).values
            for s in states:
                self.assertAllClose(keras.ops.convert_to_numpy(s), (42, 0))

    def test_correctness_with_fit_and_regularizer(self):
        strategy = tf.distribute.MirroredStrategy(["CPU:0", "CPU:1"])

        batch_size = 12
        x = keras.ops.ones((batch_size, 1))
        y = keras.ops.zeros((batch_size, 1))

        # Runs without a strategy to get expected weights.
        inputs = layers.Input(shape=(1,))
        layer = layers.Dense(
            1,
            use_bias=False,
            kernel_initializer=keras.initializers.Constant(1),
            kernel_regularizer=keras.regularizers.L1L2(l1=0.01, l2=0.01),
        )
        model = models.Model(inputs, layer(inputs))
        model.compile(loss="mse", optimizer="sgd")
        history = model.fit(x, y, batch_size=batch_size, epochs=1)
        expected_loss = history.history["loss"]
        expected_weights = keras.ops.convert_to_numpy(layer.kernel)

        # Runs with a mirrored strategy.
        with strategy.scope():
            inputs = layers.Input(shape=(1,))
            layer = layers.Dense(
                1,
                use_bias=False,
                kernel_initializer=keras.initializers.Constant(1),
                kernel_regularizer=keras.regularizers.L1L2(l1=0.01, l2=0.01),
            )
            model = models.Model(inputs, layer(inputs))
            model.compile(loss="mse", optimizer="sgd")
            history = model.fit(x, y, batch_size=batch_size, epochs=1)
            weights = strategy.run(lambda: layer.kernel.value).values

            self.assertAllClose(history.history["loss"], expected_loss)
            for w in weights:
                self.assertAllClose(
                    keras.ops.convert_to_numpy(w), expected_weights
                )
