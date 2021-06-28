# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for distributed training utility functions."""

import tensorflow.compat.v2 as tf

from keras import callbacks
from keras.distribute import distributed_training_utils_v1
from keras.optimizer_v2 import adam


class DistributedTrainingUtilsTest(tf.test.TestCase):

  def test_validate_callbacks_predefined_callbacks(self):
    supported_predefined_callbacks = [
        callbacks.TensorBoard(),
        callbacks.CSVLogger(filename='./log.csv'),
        callbacks.EarlyStopping(),
        callbacks.ModelCheckpoint(filepath='./checkpoint'),
        callbacks.TerminateOnNaN(),
        callbacks.ProgbarLogger(),
        callbacks.History(),
        callbacks.RemoteMonitor()
    ]

    distributed_training_utils_v1.validate_callbacks(
        supported_predefined_callbacks, adam.Adam())

    unsupported_predefined_callbacks = [
        callbacks.ReduceLROnPlateau(),
        callbacks.LearningRateScheduler(schedule=lambda epoch: 0.001)
    ]

    for callback in unsupported_predefined_callbacks:
      with self.assertRaisesRegex(ValueError,
                                  'You must specify a Keras Optimizer V2'):
        distributed_training_utils_v1.validate_callbacks(
            [callback], tf.compat.v1.train.AdamOptimizer())


if __name__ == '__main__':
  tf.test.main()
