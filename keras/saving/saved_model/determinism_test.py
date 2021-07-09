"""Saves the same model twice and ensures that they are serialized the same."""

import subprocess

from absl import flags
import tensorflow.compat.v2 as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.core.protobuf import saved_model_pb2
# pylint: enable=g-direct-tensorflow-import

FLAGS = flags.FLAGS


class DeterminismTest(tf.test.TestCase):

  def test_saving_is_deterministic(self):
    create_saved_model = f'{FLAGS.test_srcdir}/create_test_saved_model.par'
    saved_model_a_path = f'{FLAGS.test_tmpdir}/a'
    saved_model_b_path = f'{FLAGS.test_tmpdir}/b'

    save_a = subprocess.Popen(
        [create_saved_model, '--output_path', saved_model_a_path])
    save_b = subprocess.Popen(
        [create_saved_model, '--output_path', saved_model_b_path])
    save_a.wait()
    save_b.wait()
    saved_model_a = saved_model_pb2.SavedModel()
    with tf.io.gfile.GFile(f'{saved_model_a_path}/saved_model.pb') as f:
      saved_model_a.MergeFromString(f.read())
    saved_model_b = saved_model_pb2.SavedModel()
    with tf.io.gfile.GFile(f'{saved_model_b_path}/saved_model.pb') as f:
      saved_model_b.MergeFromString(f.read())

    self.assertProtoEquals(saved_model_a, saved_model_b)
