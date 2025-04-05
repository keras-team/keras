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
"""Steps per execution autotuning for TF-Keras engine."""

import logging
import threading
import time

import numpy as np
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.utils.StepsPerExecutionTuner")
class StepsPerExecutionTuner:
    """Steps per execution tuner class.

    Args:
        optimizer: The optimizer used for training/evaluation/prediction. Used
            to measure iterations and global throughput
            (`optimizer.iterations`/second).
        spe_variable: A `tf.Variable` representing the `steps_per_execution`
            variable used during training/evaluation/prediction. Must be
            updatable with `spe_variable.assign`.
        interval: Optional int, the amount of seconds to wait between calls to
            measure throughput and tune `spe_variable`. Defaults to 5.
        change_spe_interval: Optional int, the number of throughput measurements
            before tuning. Defaults to 10.
        change_threshold: Optional float, the percent different in throughput to
            trigger a `steps_per_execution` change. For example, `0.1` triggers
            changes if throughput changes more than 10%.

    Examples:

    If you're using `model.compile` and `model.fit`, this functionality is
    available at compile time with `steps_per_execution='auto'`

    ```python
    model.compile(..., steps_per_execution='auto')
    ```

    Custom training loop usage:

    ```python
    # Get model
    inputs = keras.Input(shape=(784,), name="digits")
    x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(10, name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Instantiate an optimizer to train the model.
    optimizer = keras.optimizers.SGD(learning_rate=1e-3)
    # Instantiate a loss function.
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Prepare the training dataset.
    batch_size = 64
    (x_train, y_train), (_, _) = keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, (-1, 784))
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    # Create our steps per execution variable
    steps_per_execution = tf.Variable(
        1,
        dtype="int64",
        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA
    )

    # Create the tuner
    tuner = StepsPerExecutionTuner(
        optimizer, steps_per_execution
    )

    # Create a step function that runs a single training step
    @tf.function
    def step_fn(iterator):
        batch_data, labels = next(iterator)
        with tf.GradientTape() as tape:
            logits = model(batch_data, training=True)
            loss_value = loss_fn(labels, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # We can now pack multiple execution steps into one call
    @tf.function
    def multi_step_train_fn(iterator, steps_per_execution):
        for _ in tf.range(steps_per_execution):
            outputs = step_fn(iterator)
        return

    initial_steps_per_execution = 1
    steps_per_epoch = 100
    epochs = 2

    # Start the tuner before training
    tuner.start()

    # We can now call our multi step training with our data
    for epoch in range(epochs):
        for _ in range(steps_per_epoch):
            multi_step_train_fn(iterator, steps_per_execution)

    # End the tuner after training
    tuner.stop()
    ```
    """

    def __init__(
        self,
        optimizer,
        spe_variable,
        interval=5,
        change_spe_interval=10,
        change_threshold=0.1,
    ):
        self.optimizer = optimizer
        self._steps_per_execution = spe_variable
        self.interval = interval
        self.change_spe_interval = change_spe_interval
        self.spe_change_threshold = change_threshold
        self.steps_per_execution_stop_event = threading.Event()
        self.thread = None

    def start(self):
        """Starts steps per execution tuning thread.

        Returns a `threading.Thread` which will run every `self.interval`
            seconds to measure throughput and tune steps per execution.
        """
        if self.thread and self.thread.is_alive():
            return self.thread
        self._begin_tuning()
        self.thread = threading.Thread(
            target=self._steps_per_execution_interval_call, daemon=True
        )  # needed to shut down successfully
        self.thread.start()
        return self.thread

    @property
    def steps_per_execution(self):
        """Settable attribute representing`steps_per_execution` variable."""
        return self._steps_per_execution

    @steps_per_execution.setter
    def steps_per_execution(self, value):
        self._steps_per_execution.assign(value)
        self.init_spe = value

    def _steps_per_execution_interval_call(self):
        while not self.steps_per_execution_stop_event.is_set():
            self._measure_and_tune()
            self.steps_per_execution_stop_event.wait(self.interval)

    def _begin_tuning(self):
        self.start_time = time.time()
        self.init_iterations = self.optimizer.iterations.numpy()
        self.init_spe = self._steps_per_execution.numpy().item()
        self.spe_last_logged = {
            "iteration": self.init_iterations,
            "time_secs": self.start_time,
        }
        self.rgsps = []  # rgsps = recent global steps per second
        self.avg_rgsps = 0
        self.prev_avg_rgsps = 0
        self.spe_tune_last_action_add = True
        self.spe_measurement_count = 0

    def stop(self):
        """Stops steps per execution tuning thread."""
        if not self.steps_per_execution_stop_event.is_set():
            self.steps_per_execution_stop_event.set()

    def _should_tune(self):
        epoch_boundary = False
        if self.rgsps[-1] == 0:
            epoch_boundary = True

        return (
            self.spe_measurement_count % self.change_spe_interval == 0
            and self.rgsps
            and not epoch_boundary
        )

    def _tune(self):
        """Changes the steps per execution using the following algorithm.

        If there is more than a 10% increase in the throughput, then the last
        recorded action is repeated (i.e. if increasing the SPE caused an
        increase in throughput, it is increased again). If there is more than a
        10% decrease in the throughput, then the opposite of the last action is
        performed (i.e. if increasing the SPE decreased the throughput, then the
        SPE is decreased).
        """
        self.avg_rgsps = sum(self.rgsps) / len(self.rgsps)
        fast_threshold = (1 + self.spe_change_threshold) * self.prev_avg_rgsps
        slow_threshold = (1 - self.spe_change_threshold) * self.prev_avg_rgsps

        if self.spe_tune_last_action_add:
            repeat_action_mult = 1.5
            opposite_action_mult = 0.5
        else:
            repeat_action_mult = 0.5
            opposite_action_mult = 1.5

        spe_variable = self._steps_per_execution
        spe_limit = spe_variable.dtype.max / 1.5
        current_spe = spe_variable.numpy().item()
        if self.avg_rgsps > fast_threshold:
            # Note that our first iteration will always trigger this as our
            # threshold should be 0
            new_spe = current_spe * repeat_action_mult
        elif self.avg_rgsps < slow_threshold:
            new_spe = current_spe * opposite_action_mult
            self.spe_tune_last_action_add = not self.spe_tune_last_action_add
        else:
            new_spe = current_spe

        if current_spe >= spe_limit:
            new_spe = current_spe
        elif current_spe <= 0:
            new_spe = self.init_spe

        self._steps_per_execution.assign(np.round(new_spe))
        self.prev_avg_rgsps = self.avg_rgsps

    def _measure_and_tune(self):
        self.spe_measurement_count += 1

        cur_iteration = self.optimizer.iterations.numpy()

        cur_time_secs = time.time()
        recent_gsps = (cur_iteration - self.spe_last_logged["iteration"]) / (
            cur_time_secs - self.spe_last_logged["time_secs"]
        )

        self.rgsps.append(recent_gsps)
        if len(self.rgsps) > self.change_spe_interval:
            self.rgsps.pop(0)

        if cur_iteration == 0:  # No need to tune, we have no measurements
            self.start_time = cur_time_secs
            return

        self.spe_last_logged["iteration"] = cur_iteration
        self.spe_last_logged["time_secs"] = cur_time_secs

        try:
            if self._should_tune():
                self._tune()
        except RuntimeError:
            logging.exception("Steps per execution autotuner failed to run.")
            return

