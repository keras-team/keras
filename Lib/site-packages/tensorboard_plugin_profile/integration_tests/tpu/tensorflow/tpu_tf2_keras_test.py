# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Profiler tests for TPUs."""
import glob
import json
import os

from absl import flags
from absl import logging
from absl.testing import absltest
import tensorflow.compat.v1 as tfv1
import tensorflow.compat.v2 as tf

from tensorboard_plugin_profile.convert import raw_to_tool_data
from tensorboard_plugin_profile.integration_tests import tf_mnist
from tensorboard_plugin_profile.integration_tests import tf_profiler_session

FLAGS = flags.FLAGS

_EPOCHS = 1
_TRAIN_STEPS = 40
_EVAL_STEPS = 10

log_dir: str = None

LOG_DIRECTORY = flags.DEFINE_string(
    'log_directory', None, 'Location of trace file, if already available'
)


def setUpModule():
  global log_dir
  # Runs the profiler on mnist model, then assert on the generated trace.
  if not LOG_DIRECTORY.value:
    log_dir = os.path.join(FLAGS.test_tmpdir, 'tensorboard')
    with tf_profiler_session.TensorflowProfilerSession(tf, log_dir, 0):
      _do_mnist_training(True)
  # Assert on the trace present at the log directory.
  else:
    log_dir = LOG_DIRECTORY.value


def _do_mnist_training(use_xla: bool):
  tf.config.optimizer.set_jit(use_xla)
  # For non xla test case, use the legacy optimizer to avoid XLA compilation.
  optimizer = tf.keras.optimizers.legacy.Adam()
  if use_xla:
    optimizer = tf.keras.optimizers.Adam()

    train_ds, eval_ds, _, input_shape = tf_mnist.get_input_datasets(tf)

    resolver = tf.distribute.cluster_resolver.TPUClusterResolver('local')
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)

    with strategy.scope():
      model = tf_mnist.get_model(tf, input_shape)
      model.compile(
          optimizer=optimizer,
          loss=tf.keras.losses.categorical_crossentropy,
          metrics=['accuracy'])

      model.fit(
          x=train_ds,
          epochs=_EPOCHS,
          steps_per_epoch=_TRAIN_STEPS,
          validation_steps=_EVAL_STEPS,
          validation_data=eval_ds)


class TpuKerasTest(absltest.TestCase):

  def _get_session_snapshot(self):
    """Gets a session snapshot of current session. assume only one session."""
    profile_plugin_root = os.path.join(log_dir, 'plugins/profile')
    # The session exists under a director whose name is time-dependent.
    profile_session_glob = os.path.join(profile_plugin_root, '*', '*.xplane.pb')
    return glob.glob(profile_session_glob)

  def test_xplane_is_present(self):
    files = self._get_session_snapshot()
    self.assertLen(files, 1)

  def test_tools_are_in_list(self):
    xspace_filenames = self._get_session_snapshot()
    result = raw_to_tool_data.xspace_to_tool_names(xspace_filenames)
    result.sort()
    expected = [
        'trace_viewer@^',
        'overview_page^',
        'input_pipeline_analyzer^',
        'framework_op_stats^',
        'memory_profile^',
        'pod_viewer^',
        'tf_data_bottleneck_analysis^',
        'op_profile^',
        'memory_viewer^',
        'graph_viewer^',
        'hlo_stats^',
        'inference_profile^',
        'roofline_model^',
    ]
    expected.sort()
    self.assertListEqual(expected, result)

  def test_overview_page(self):
    xspace_filenames = self._get_session_snapshot()
    result, _ = raw_to_tool_data.xspace_to_tool_data(xspace_filenames,
                                                     'overview_page^', {})
    result = json.loads(result)
    run_environment = result[2]
    self.assertEqual(run_environment['p']['host_count'], '1')
    self.assertRegex(run_environment['p']['device_type'], 'TPU.*')

  def test_op_profile(self):
    xspace_filenames = self._get_session_snapshot()
    result, _ = raw_to_tool_data.xspace_to_tool_data(
        xspace_filenames, 'op_profile^', {}
    )
    result = json.loads(result)
    logging.info(result)
    self.assertIn('byCategory', result)
    self.assertIn('metrics', result['byCategory'])
    overall_metrics = result['byCategory']['metrics']
    self.assertIn('flops', overall_metrics)
    self.assertIn('bandwidthUtils', overall_metrics)
    self.assertGreater(overall_metrics['flops'], 0)
    contains_value = False
    for m in overall_metrics['bandwidthUtils']:
      if (m > 0):
        contains_value = True
    self.assertTrue(contains_value)

  def test_device_trace_contains_threads(self):
    xspace_filenames = self._get_session_snapshot()
    result, _ = raw_to_tool_data.xspace_to_tool_data(
        xspace_filenames, 'trace_viewer^', {}
    )
    result = json.loads(result)
    thread_names = []
    process_names = []
    for event in result['traceEvents']:
      if 'name' in event:
        if event['name'] == 'thread_name':
          thread_names.append((event['args']['name']))
        elif event['name'] == 'process_name':
          process_names.append((event['args']['name']))
    self.assertContainsSubset(
        [
            'Framework Name Scope',
            'Framework Ops',
            'XLA Modules',
            'XLA Ops',
            'XLA TraceMe',
            'Steps',
        ],
        thread_names,
    )
    self.assertContainsSubset(
        ['/device:TPU:0', '/device:TPU:1', '/host:CPU'], process_names
    )


if __name__ == '__main__':
  tfv1.enable_v2_behavior()
  absltest.main()
