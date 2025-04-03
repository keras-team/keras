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
"""Functions for dealing with metrics."""


import os


from tensorboard.plugins.hparams import api_pb2


def run_tag_from_session_and_metric(session_name, metric_name):
    """Returns a (run,tag) tuple storing the evaluations of the specified
    metric.

    Args:
      session_name: str.
      metric_name: MetricName protobuffer.
    Returns: (run, tag) tuple.
    """
    assert isinstance(session_name, str)
    assert isinstance(metric_name, api_pb2.MetricName)
    # os.path.join() will append a final slash if the group is empty; it seems
    # like multiplexer.Tensors won't recognize paths that end with a '/' so
    # we remove the final '/' in that case.
    run = os.path.join(session_name, metric_name.group)
    if run.endswith("/"):
        run = run[:-1]
    tag = metric_name.tag
    return run, tag
