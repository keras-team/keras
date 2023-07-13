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
"""Tests of `worker_training_state.py` utilities."""

import os
import sys

import tensorflow.compat.v2 as tf
from absl.testing import parameterized

from keras import callbacks
from keras.distribute import multi_worker_testing_utils


class WorkerTrainingStateTest(tf.test.TestCase, parameterized.TestCase):
    @tf.__internal__.distribute.combinations.generate(
        tf.__internal__.test.combinations.combine(mode=["eager"])
    )
    def testCheckpointExists(self):
        train_ds, _ = multi_worker_testing_utils.mnist_synthetic_dataset(64, 2)
        model = multi_worker_testing_utils.get_mnist_model((28, 28, 1))
        saving_dir = self.get_temp_dir()
        callbacks_list = [
            callbacks.BackupAndRestore(
                backup_dir=saving_dir, delete_checkpoint=False
            )
        ]
        self.assertLen(tf.io.gfile.glob(os.path.join(saving_dir, "*")), 0)
        model.fit(
            x=train_ds, epochs=2, steps_per_epoch=2, callbacks=callbacks_list
        )
        # By default worker_training_state only keeps the results from one
        # checkpoint. Even though the test is expected to checkpoint twice, it
        # only keeps the checkpoint files from the second checkpoint.
        checkpoint_path = os.path.join(saving_dir, "chief", "ckpt-2.index")
        self.assertLen(tf.io.gfile.glob(checkpoint_path), 1)


if __name__ == "__main__":
    with tf.compat.v1.test.mock.patch.object(sys, "exit", os._exit):
        tf.test.main()
