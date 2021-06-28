# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

import os

import tensorflow.compat.v2 as tf

from absl.testing import parameterized
from keras.optimizer_v2 import adam


class TrainingCheckpointTests(tf.test.TestCase, parameterized.TestCase):

  @tf.__internal__.distribute.combinations.generate(
      tf.__internal__.test.combinations.combine(
          distribution=[
              tf.__internal__.distribute.combinations.mirrored_strategy_with_one_cpu,
              tf.__internal__.distribute.combinations.mirrored_strategy_with_gpu_and_cpu,
              tf.__internal__.distribute.combinations.tpu_strategy,
              tf.__internal__.distribute.combinations.tpu_strategy_packed_var,
              tf.__internal__.distribute.combinations.central_storage_strategy_with_two_gpus,
          ],
          mode=["eager"]))
  def testCheckpointRestoreOptimizerSlots(self, distribution):
    def state():
      with distribution.scope():
        v = tf.Variable(tf.random.normal([]))
      opt = adam.Adam(0.001)

      @tf.function
      def step():
        def f():
          with tf.GradientTape() as tape:
            loss = v + v
          gradients = tape.gradient(loss, [v])
          opt.apply_gradients(zip(gradients, [v]))

        distribution.run(f)

      return v, opt, step

    def checkpoint():
      v, opt, step = state()
      step()

      # Save random weights into checkpoint.
      checkpoint = tf.train.Checkpoint(v=v, opt=opt)
      prefix = os.path.join(self.get_temp_dir(), "ckpt")
      with self.test_session():
        save_path = checkpoint.save(prefix)
      return save_path

    save_path = checkpoint()

    v, opt, step = state()
    checkpoint = tf.train.Checkpoint(v=v, opt=opt)
    # Restore from the checkpoint inside a distribution.scope().
    with self.test_session():
      with distribution.scope():
        checkpoint.restore(save_path)
    step()
    slot = opt.get_slot(v, "m")
    self.assertEqual(v._distribute_strategy, slot._distribute_strategy)

    v, opt, step = state()
    checkpoint = tf.train.Checkpoint(v=v, opt=opt)
    # Restore from the checkpoint outside a distribution.scope().
    with self.test_session():
      with self.assertRaisesRegex(
          ValueError, "optimizer slot variable under the scope"):
        checkpoint.restore(save_path)

  @tf.__internal__.distribute.combinations.generate(
      tf.__internal__.test.combinations.combine(
          distribution=[
              tf.__internal__.distribute.combinations.mirrored_strategy_with_one_cpu,
              tf.__internal__.distribute.combinations.mirrored_strategy_with_gpu_and_cpu,
              tf.__internal__.distribute.combinations.cloud_tpu_strategy,
              tf.__internal__.distribute.combinations.tpu_strategy,
              tf.__internal__.distribute.combinations.tpu_strategy_packed_var,
              tf.__internal__.distribute.combinations.central_storage_strategy_with_two_gpus,
          ],
          mode=["eager"]))
  def testCheckpointSaveRestoreIoDevice(self, distribution):

    def state():
      with distribution.scope():
        v = tf.Variable(tf.random.normal([]))
        return v

    ckpt_options = tf.train.CheckpointOptions(
        experimental_io_device="/job:localhost")

    def checkpoint():
      v = state()
      # Save random weights into checkpoint.
      checkpoint = tf.train.Checkpoint(v=v)
      prefix = os.path.join(self.get_temp_dir(), "ckpt")
      with self.test_session():
        save_path = checkpoint.save(prefix, options=ckpt_options)
      return save_path

    save_path = checkpoint()

    v = state()
    checkpoint = tf.train.Checkpoint(v=v)
    # Restore from the checkpoint inside a distribution.scope().
    # Check that restore works without error.
    with self.test_session():
      with distribution.scope():
        checkpoint.restore(save_path, options=ckpt_options)


if __name__ == "__main__":
  tf.compat.v1.enable_eager_execution()
  tf.test.main()
