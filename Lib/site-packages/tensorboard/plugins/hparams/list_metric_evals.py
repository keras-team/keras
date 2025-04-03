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
"""Classes and functions for handling the ListMetricEvals API call."""


from tensorboard.plugins.hparams import metrics
from tensorboard.plugins.scalar import scalars_plugin


class Handler:
    """Handles a ListMetricEvals request."""

    def __init__(
        self, request_context, request, scalars_plugin_instance, experiment
    ):
        """Constructor.

        Args:
            request_context: A tensorboard.context.RequestContext.
            request: A ListSessionGroupsRequest protobuf.
            scalars_plugin_instance: A scalars_plugin.ScalarsPlugin.
            experiment: A experiment ID, as a possibly-empty `str`.
        """
        self._request_context = request_context
        self._request = request
        self._scalars_plugin_instance = scalars_plugin_instance
        self._experiment = experiment

    def run(self):
        """Executes the request.

        Returns:
            An array of tuples representing the metric evaluations--each of the
            form (<wall time in secs>, <training step>, <metric value>).
        """
        run, tag = metrics.run_tag_from_session_and_metric(
            self._request.session_name, self._request.metric_name
        )
        body, _ = self._scalars_plugin_instance.scalars_impl(
            self._request_context,
            tag,
            run,
            self._experiment,
            scalars_plugin.OutputFormat.JSON,
        )
        return body
