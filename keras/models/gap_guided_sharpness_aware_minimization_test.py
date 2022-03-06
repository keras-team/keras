"""Tests for GapGuidedSharpnessAwareMinimization."""

import os

from absl.testing import parameterized
import keras
from keras.models import GapGuidedSharpnessAwareMinimization
from keras.optimizers.optimizer_experimental import adam
from keras.testing_infra import test_utils
import tensorflow.compat.v2 as tf

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
class GapGuidedSharpnessAwareMinimizationTest(tf.test.TestCase, parameterized.TestCase):

  def test_gsam_model_call(self):
    model = keras.Sequential([
        keras.Input([2, 2]),
        keras.layers.Dense(4),
    ])
    gsam_model = GapGuidedSharpnessAwareMinimization(model)
    data = tf.random.uniform([2, 2])
    self.assertAllClose(model(data), gsam_model(data))

  @ds_combinations.generate(
      tf.__internal__.test.combinations.combine(strategy=STRATEGIES))
  def test_gsam_model_fit(self, strategy):
    with strategy.scope():
      model = keras.Sequential([
          keras.Input([2, 2]),
          keras.layers.Dense(4),
          keras.layers.Dense(1),
      ])
      gsam_model = GapGuidedSharpnessAwareMinimization(model)
      data = tf.random.uniform([2, 2])
      label = data[:, 0] > 0.5

      gsam_model.compile(
          optimizer=adam.Adam(),
          loss=keras.losses.BinaryCrossentropy(from_logits=True),
      )

      gsam_model.fit(data, label)

  def test_save_gsam(self):
    model = keras.Sequential([
        keras.Input([2, 2]),
        keras.layers.Dense(4),
        keras.layers.Dense(1),
    ])
    gsam_model = GapGuidedSharpnessAwareMinimization(model)
    data = tf.random.uniform([1, 2, 2])
    label = data[:, 0] > 0.5

    gsam_model.compile(
        optimizer=adam.Adam(),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
    )

    gsam_model.fit(data, label)

    path = os.path.join(self.get_temp_dir(), "model")
    gsam_model.save(path)
    loaded_gsam_model = keras.models.load_model(path)
    loaded_gsam_model.load_weights(path)

    self.assertAllClose(gsam_model(data), loaded_gsam_model(data))

  def test_checkpoint_gsam(self):
    model = keras.Sequential([
        keras.Input([2, 2]),
        keras.layers.Dense(4),
        keras.layers.Dense(1),
    ])
    gsam_model_1 = GapGuidedSharpnessAwareMinimization(model)
    gsam_model_2 = GapGuidedSharpnessAwareMinimization(model)
    data = tf.random.uniform([1, 2, 2])
    label = data[:, 0] > 0.5

    gsam_model_1.compile(
        optimizer=adam.Adam(),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
    )

    gsam_model_1.fit(data, label)

    checkpoint = tf.train.Checkpoint(sam_model_1)
    checkpoint2 = tf.train.Checkpoint(sam_model_2)
    temp_dir = self.get_temp_dir()
    save_path = checkpoint.save(temp_dir)
    checkpoint2.restore(save_path)

    self.assertAllClose(gsam_model_1(data), gsam_model_2(data))


if __name__ == "__main__":
  tf.__internal__.distribute.multi_process_runner.test_main()
