# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Test to demonstrate Keras training with MultiWorkerMirroredStrategy."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow.compat.v2 as tf
from absl import logging

NUM_WORKERS = 2
NUM_EPOCHS = 2
NUM_STEPS_PER_EPOCH = 50


class MwmsMultiProcessRunnerTest(tf.test.TestCase):
    """Test to demonstrate Keras training with MultiWorkerMirroredStrategy."""

    def testMwmsWithModelFit(self):
        def worker_fn():
            def dataset_fn(input_context):
                # User should shard data accordingly. Omitted here.
                del input_context
                return tf.data.Dataset.from_tensor_slices(
                    (tf.random.uniform((6, 10)), tf.random.uniform((6, 10)))
                ).batch(2)

            strategy = tf.distribute.MultiWorkerMirroredStrategy()
            with strategy.scope():
                model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
            model.compile(
                loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                metrics=["accuracy"],
            )

            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(self.get_temp_dir(), "checkpoint")
                )
            ]
            dataset = strategy.distribute_datasets_from_function(dataset_fn)
            model.fit(
                dataset,
                epochs=NUM_EPOCHS,
                steps_per_epoch=NUM_STEPS_PER_EPOCH,
                callbacks=callbacks,
            )

            logging.info("testMwmsWithModelFit successfully ends")

        mpr_result = tf.__internal__.distribute.multi_process_runner.run(
            worker_fn,
            tf.__internal__.distribute.multi_process_runner.create_cluster_spec(
                num_workers=NUM_WORKERS
            ),
            return_output=True,
        )

        # Verifying the worker functions ended successfully.
        self.assertTrue(
            any(
                [
                    "testMwmsWithModelFit successfully ends" in msg
                    for msg in mpr_result.stdout
                ]
            )
        )


if __name__ == "__main__":
    tf.__internal__.distribute.multi_process_runner.test_main()
