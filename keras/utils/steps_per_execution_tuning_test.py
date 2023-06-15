# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Test steps_per_execution_tuning."""

import time

import tensorflow.compat.v2 as tf

from keras import Input
from keras import Model
from keras import losses
from keras import optimizers
from keras.layers import Dense
from keras.testing_infra import test_combinations
from keras.utils import steps_per_execution_tuning


class mockOptimizer:
    def __init__(self, iterations):
        self.iterations = tf.Variable(iterations)


@test_combinations.run_with_all_model_types
@test_combinations.run_all_keras_modes(always_skip_v1=True)
class StepsPerExecutionTuningTest(test_combinations.TestCase):
    def test_variables(self):
        spe_variable = tf.Variable(1)
        tuner = steps_per_execution_tuning.StepsPerExecutionTuner(
            mockOptimizer(5), spe_variable, 5, 50, 0.5
        )
        assert tuner.optimizer.iterations.numpy() == 5
        assert tuner._steps_per_execution.numpy().item() == 1
        assert tuner.interval == 5
        assert tuner.change_spe_interval == 50
        assert tuner.spe_change_threshold == 0.5
        assert not tuner.steps_per_execution_stop_event.is_set()

    def test_start_stop(self):
        spe_variable = tf.Variable(1)
        tuner = steps_per_execution_tuning.StepsPerExecutionTuner(
            mockOptimizer(5), spe_variable, interval=0.2
        )
        tuner.start()
        assert not tuner.steps_per_execution_stop_event.is_set()
        assert tuner.start_time > 0
        time.sleep(0.5)  # should be enough time for 2 measurements
        tuner.stop()
        assert tuner.steps_per_execution_stop_event.is_set()
        assert tuner.spe_measurement_count > 0

    def test_settable_steps_per_execution(self):
        spe_variable = tf.Variable(1)
        tuner = steps_per_execution_tuning.StepsPerExecutionTuner(
            mockOptimizer(5), spe_variable, interval=0.2
        )
        tuner.start()
        tuner.stop()
        assert tuner.init_spe == 1
        tuner.steps_per_execution = 5
        assert spe_variable.numpy().item() == 5
        assert tuner.init_spe == 5

    def test_custom_training_loop(self):
        dataset = _get_dataset()
        iterator = iter(dataset)

        inputs = Input(shape=(784,), name="digits")
        x = Dense(64, activation="relu", name="dense_1")(inputs)
        x = Dense(64, activation="relu", name="dense_2")(x)
        outputs = Dense(10, name="predictions")(x)
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = optimizers.SGD(learning_rate=1e-3)
        loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)

        # Create our steps per execution variable
        steps_per_execution = tf.Variable(
            1,
            dtype="int64",
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        )

        # Create the tuner
        tuner = steps_per_execution_tuning.StepsPerExecutionTuner(
            optimizer, steps_per_execution
        )

        # Create a step function that runs a single training step
        @tf.function
        def step_fn(iterator):
            batch_data, labels = next(iterator)
            print(batch_data.shape, labels.shape)
            with tf.GradientTape() as tape:
                logits = model(batch_data, training=True)
                loss_value = loss_fn(labels, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # We can now pack multiple execution steps into one call
        @tf.function
        def multi_step_train_fn(iterator, steps_per_execution):
            for _ in tf.range(steps_per_execution):
                step_fn(iterator)
            return

        steps_per_epoch = 10
        epochs = 2

        # Start the tuner before training
        tuner.start()

        for _ in range(epochs):
            for _ in range(steps_per_epoch):
                multi_step_train_fn(iterator, steps_per_execution)

        # End the tuner after training
        tuner.stop()


def _get_dataset():
    inputs = tf.zeros((1000, 784), dtype=tf.float32)
    targets = tf.zeros((1000,), dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    dataset = dataset.batch(10)
    return dataset


if __name__ == "__main__":
    tf.test.main()
