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
"""Programmatic functionality for tensorflow profile session."""

import types


class TensorflowProfilerSession:
  """context manager for Tensorflow programmatic profile session."""

  def __init__(
      self, tf: types.ModuleType, log_dir: str, python_tracer_level: int = 1
  ):
    """tf object is version dependent object."""
    self.profiler = tf.profiler.experimental
    self.log_dir = log_dir
    self.python_tracer_level = python_tracer_level

  def __enter__(self):
    """Starts profile session and serializes data to temp directory."""
    options = self.profiler.ProfilerOptions(
        host_tracer_level=2, python_tracer_level=self.python_tracer_level
    )
    self.profiler.start(self.log_dir, options=options)

  def __exit__(self, exc_type, exc_value, traceback):
    """Ends current profile session and verifies test expectations."""
    self.profiler.stop()
