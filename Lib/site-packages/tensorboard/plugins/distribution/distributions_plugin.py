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
"""The TensorBoard Distributions (a.k.a. compressed histograms) plugin.

See `http_api.md` in this directory for specifications of the routes for
this plugin.
"""


from werkzeug import wrappers

from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin
from tensorboard.plugins.distribution import compressor
from tensorboard.plugins.distribution import metadata
from tensorboard.plugins.histogram import histograms_plugin


class DistributionsPlugin(base_plugin.TBPlugin):
    """Distributions Plugin for TensorBoard.

    This supports both old-style summaries (created with TensorFlow ops
    that output directly to the `histo` field of the proto) and new-
    style summaries (as created by the
    `tensorboard.plugins.histogram.summary` module).
    """

    plugin_name = metadata.PLUGIN_NAME

    # Use a round number + 1 since sampling includes both start and end steps,
    # so N+1 samples corresponds to dividing the step sequence into N intervals.
    SAMPLE_SIZE = 501

    def __init__(self, context):
        """Instantiates DistributionsPlugin via TensorBoard core.

        Args:
          context: A base_plugin.TBContext instance.
        """
        self._histograms_plugin = histograms_plugin.HistogramsPlugin(context)

    def get_plugin_apps(self):
        return {
            "/distributions": self.distributions_route,
            "/tags": self.tags_route,
        }

    def is_active(self):
        """This plugin is active iff any run has at least one histogram tag.

        (The distributions plugin uses the same data source as the
        histogram plugin.)
        """
        return self._histograms_plugin.is_active()

    def data_plugin_names(self):
        return (self._histograms_plugin.plugin_name,)

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(
            element_name="tf-distribution-dashboard",
        )

    def distributions_impl(self, ctx, tag, run, experiment):
        """Result of the form `(body, mime_type)`.

        Raises:
          tensorboard.errors.PublicError: On invalid request.
        """
        (histograms, mime_type) = self._histograms_plugin.histograms_impl(
            ctx, tag, run, experiment=experiment, downsample_to=self.SAMPLE_SIZE
        )
        return (
            [self._compress(histogram) for histogram in histograms],
            mime_type,
        )

    def _compress(self, histogram):
        (wall_time, step, buckets) = histogram
        converted_buckets = compressor.compress_histogram(buckets)
        return [wall_time, step, converted_buckets]

    def index_impl(self, ctx, experiment):
        return self._histograms_plugin.index_impl(ctx, experiment=experiment)

    @wrappers.Request.application
    def tags_route(self, request):
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        index = self.index_impl(ctx, experiment=experiment)
        return http_util.Respond(request, index, "application/json")

    @wrappers.Request.application
    def distributions_route(self, request):
        """Given a tag and single run, return an array of compressed
        histograms."""
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        tag = request.args.get("tag")
        run = request.args.get("run")
        (body, mime_type) = self.distributions_impl(
            ctx, tag, run, experiment=experiment
        )
        return http_util.Respond(request, body, mime_type)
