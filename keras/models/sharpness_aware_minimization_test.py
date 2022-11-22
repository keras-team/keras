"""Tests for sharpness_aware_minimization."""

import os

import tensorflow.compat.v2 as tf
from absl.testing import parameterized

import keras
from keras.models import sharpness_aware_minimization
from keras.optimizers import adam
from keras.testing_infra import test_utils

ds_combinations = tf.__internal__.distribute.combinations

STRATEGIES = [
    ds_combinations.one_device_strategy,
    ds_combinations.mirrored_strategy_with_two_gpus,
    ds_combinations.tpu_strategy,
    ds_combinations.parameter_server_strategy_3worker_2ps_1gpu,
    ds_combinations.multi_worker_mirrored_2x1_cpu,
    ds_combinations.multi_worker_mirrored_2x2_gpu,
    ds_combinations.central_storage_strategy_with_two_gpus,
]


@test_utils.run_v2_only
class SharpnessAwareMinimizationTest(tf.test.TestCase, parameterized.TestCase):
    def test_sam_model_call(self):
        model = keras.Sequential(
            [
                keras.Input([2, 2]),
                keras.layers.Dense(4),
            ]
        )
        sam_model = sharpness_aware_minimization.SharpnessAwareMinimization(
            model
        )
        data = tf.random.uniform([2, 2])
        self.assertAllClose(model(data), sam_model(data))

    @ds_combinations.generate(
        tf.__internal__.test.combinations.combine(strategy=STRATEGIES)
    )
    def test_sam_model_fit(self, strategy):
        with strategy.scope():
            model = keras.Sequential(
                [
                    keras.Input([2, 2]),
                    keras.layers.Dense(4),
                    keras.layers.Dense(1),
                ]
            )
            sam_model = sharpness_aware_minimization.SharpnessAwareMinimization(
                model
            )
            data = tf.random.uniform([2, 2])
            label = data[:, 0] > 0.5

            sam_model.compile(
                optimizer=adam.Adam(),
                loss=keras.losses.BinaryCrossentropy(from_logits=True),
            )

            sam_model.fit(data, label, steps_per_epoch=1)

    @ds_combinations.generate(
        tf.__internal__.test.combinations.combine(strategy=STRATEGIES)
    )
    def test_sam_model_fit_with_sub_batch(self, strategy):
        with strategy.scope():
            model = keras.Sequential(
                [
                    keras.Input([2, 2]),
                    keras.layers.Dense(4),
                    keras.layers.Dense(1),
                ]
            )
            sam_model = sharpness_aware_minimization.SharpnessAwareMinimization(
                model, num_batch_splits=4
            )
            data = tf.random.uniform([48, 2])
            label = data[:, 0] > 0.5

            sam_model.compile(
                optimizer=adam.Adam(),
                loss=keras.losses.BinaryCrossentropy(from_logits=True),
            )

            sam_model.fit(data, label, steps_per_epoch=1)

    def test_save_sam(self):
        model = keras.Sequential(
            [
                keras.Input([2, 2]),
                keras.layers.Dense(4),
                keras.layers.Dense(1),
            ]
        )
        sam_model = sharpness_aware_minimization.SharpnessAwareMinimization(
            model
        )
        data = tf.random.uniform([1, 2, 2])
        label = data[:, 0] > 0.5

        sam_model.compile(
            optimizer=adam.Adam(),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
        )

        sam_model.fit(data, label)

        path = os.path.join(self.get_temp_dir(), "model")
        sam_model.save(path)
        loaded_sam_model = keras.models.load_model(path)
        loaded_sam_model.load_weights(path)

        self.assertAllClose(sam_model(data), loaded_sam_model(data))

    def test_checkpoint_sam(self):
        model = keras.Sequential(
            [
                keras.Input([2, 2]),
                keras.layers.Dense(4),
                keras.layers.Dense(1),
            ]
        )
        sam_model_1 = sharpness_aware_minimization.SharpnessAwareMinimization(
            model
        )
        sam_model_2 = sharpness_aware_minimization.SharpnessAwareMinimization(
            model
        )
        data = tf.random.uniform([1, 2, 2])
        label = data[:, 0] > 0.5

        sam_model_1.compile(
            optimizer=adam.Adam(),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
        )

        sam_model_1.fit(data, label)

        checkpoint = tf.train.Checkpoint(sam_model_1)
        checkpoint2 = tf.train.Checkpoint(sam_model_2)
        temp_dir = self.get_temp_dir()
        save_path = checkpoint.save(temp_dir)
        checkpoint2.restore(save_path)

        self.assertAllClose(sam_model_1(data), sam_model_2(data))


if __name__ == "__main__":
    tf.__internal__.distribute.multi_process_runner.test_main()
