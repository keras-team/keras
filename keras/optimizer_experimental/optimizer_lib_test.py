"""Tests for optimizer_lib."""

from keras.optimizer_experimental import optimizer_lib
import tensorflow.compat.v2 as tf


class OptimizerLibTest(tf.test.TestCase):

  def test_gradients_clip_option(self):
    gradients_clip_option = optimizer_lib.GradientsClipOption(
        clipnorm=1, clipvalue=1)
    self.assertEqual(gradients_clip_option.clipnorm, 1)
    self.assertEqual(gradients_clip_option.clipvalue, 1)
    with self.assertRaisesRegex(ValueError, "At most one of.*"):
      _ = optimizer_lib.GradientsClipOption(clipnorm=1, global_clipnorm=1)

    with self.assertRaisesRegex(ValueError, ".*should be a positive number.*"):
      _ = optimizer_lib.GradientsClipOption(clipnorm=-1)

  def test_get_and_from_config(self):
    gradients_clip_option = optimizer_lib.GradientsClipOption(
        clipnorm=1, clipvalue=1)
    config = gradients_clip_option.get_config()
    restored = optimizer_lib.GradientsClipOption(**config)
    self.assertDictEqual(restored.get_config(), config)

  def test_invalid_ema_option(self):
    ema_option = optimizer_lib.EMAOption(
        use_ema=True, ema_momentum=0.5, ema_overwrite_frequency=50)
    self.assertEqual(ema_option.ema_momentum, 0.5)
    self.assertEqual(ema_option.ema_overwrite_frequency, 50)
    with self.assertRaisesRegex(ValueError, "`ema_momentum` must be in the*"):
      _ = optimizer_lib.EMAOption(use_ema=True, ema_momentum=-1)


if __name__ == "__main__":
  tf.test.main()
