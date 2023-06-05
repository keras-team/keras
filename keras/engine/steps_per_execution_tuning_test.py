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

from keras.engine import steps_per_execution_tuning
from keras.testing_infra import test_combinations


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


if __name__ == "__main__":
    tf.test.main()
