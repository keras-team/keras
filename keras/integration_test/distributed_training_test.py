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
"""Test to demonstrate basic Keras training with a variety of strategies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os

import tensorflow.compat.v2 as tf

ds_combinations = tf.__internal__.distribute.combinations

# Note: Strategy combinations are not (yet) public APIs, so they are subject
# to API changes and backward-compatibility is not guaranteed.
# TODO(b/188763034): Proceed to export the strategy combinations as public APIs.
STRATEGIES = [
    ds_combinations.default_strategy,
    ds_combinations.mirrored_strategy_with_two_cpus,
    ds_combinations.mirrored_strategy_with_two_gpus,
    ds_combinations.tpu_strategy,
    ds_combinations.cloud_tpu_strategy,
    ds_combinations.parameter_server_strategy_3worker_2ps_cpu,
    ds_combinations.parameter_server_strategy_3worker_2ps_1gpu,
    ds_combinations.multi_worker_mirrored_2x1_cpu,
    ds_combinations.multi_worker_mirrored_2x2_gpu,
    ds_combinations.central_storage_strategy_with_two_gpus,
]


@ds_combinations.generate(
    tf.__internal__.test.combinations.combine(strategy=STRATEGIES, mode="eager")
)
class DistributedTrainingTest(tf.test.TestCase):
    """Test to demonstrate basic Keras training with a variety of strategies."""

    def testKerasTrainingAPI(self, strategy):
        if not tf.__internal__.tf2.enabled() and isinstance(
            strategy, tf.distribute.experimental.ParameterServerStrategy
        ):
            self.skipTest(
                "Parameter Server strategy with dataset creator need to be run "
                "when eager execution is enabled."
            )

        # A `dataset_fn` is required for `Model.fit` to work across all
        # strategies.
        def dataset_fn(input_context):
            batch_size = input_context.get_per_replica_batch_size(
                global_batch_size=64
            )
            x = tf.random.uniform((10, 10))
            y = tf.random.uniform((10,))
            dataset = (
                tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10).repeat()
            )
            dataset = dataset.shard(
                input_context.num_input_pipelines,
                input_context.input_pipeline_id,
            )
            return dataset.batch(batch_size).prefetch(2)

        with strategy.scope():
            model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
            optimizer = tf.keras.optimizers.SGD()
            model.compile(optimizer, loss="mse", steps_per_execution=5)

        x = tf.keras.utils.experimental.DatasetCreator(dataset_fn)

        logdir = os.path.join(self.get_temp_dir(), "logdir")
        model.fit(
            x,
            epochs=2,
            steps_per_epoch=20,
            callbacks=[
                tf.keras.callbacks.TensorBoard(
                    logdir,
                    update_freq=5,
                    write_steps_per_second=True,
                )
            ],
        )

        events_got = []
        for event_file in glob.glob(logdir + "/train/events.out.*"):
            for event in tf.compat.v1.train.summary_iterator(event_file):
                if not event.summary:
                    continue
                for value in event.summary.value:
                    if value.tag != "batch_loss":
                        continue
                    events_got += [event.step]

        # total steps = epochs * steps_per_epoch
        events_expected = [5, 10, 15, 20, 25, 30, 35, 40]

        if isinstance(
            strategy, tf.distribute.experimental.ParameterServerStrategy
        ):
            # Metrics are not logged with this strategy as they are not
            # immediately available on batch end
            events_expected = []
        if (
            strategy.cluster_resolver
            and strategy.cluster_resolver.task_type == "worker"
        ):
            # The below assertion is run by both chief and workers when using
            # `tf.distribute.MultiWorkerMirroredStrategy`, but only the chief
            # will log events.
            events_expected = []

        self.assertEqual(events_got, events_expected)


if __name__ == "__main__":
    tf.__internal__.distribute.multi_process_runner.test_main()
