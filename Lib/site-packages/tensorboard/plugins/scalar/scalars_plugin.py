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
"""The TensorBoard Scalars plugin.

See `http_api.md` in this directory for specifications of the routes for
this plugin.
"""


import csv
import io

import werkzeug.exceptions
from werkzeug import wrappers

from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.scalar import metadata

_DEFAULT_DOWNSAMPLING = 1000  # scalars per time series


class OutputFormat:
    """An enum used to list the valid output formats for API calls."""

    JSON = "json"
    CSV = "csv"


class ScalarsPlugin(base_plugin.TBPlugin):
    """Scalars Plugin for TensorBoard."""

    plugin_name = metadata.PLUGIN_NAME

    def __init__(self, context):
        """Instantiates ScalarsPlugin via TensorBoard core.

        Args:
          context: A base_plugin.TBContext instance.
        """
        self._downsample_to = (context.sampling_hints or {}).get(
            self.plugin_name, _DEFAULT_DOWNSAMPLING
        )
        self._data_provider = context.data_provider
        self._version_checker = plugin_util._MetadataVersionChecker(
            data_kind="scalar",
            latest_known_version=0,
        )

    def get_plugin_apps(self):
        return {
            "/scalars": self.scalars_route,
            "/scalars_multirun": self.scalars_multirun_route,
            "/tags": self.tags_route,
        }

    def is_active(self):
        return False  # `list_plugins` as called by TB core suffices

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(element_name="tf-scalar-dashboard")

    def index_impl(self, ctx, experiment=None):
        """Return {runName: {tagName: {displayName: ..., description:
        ...}}}."""
        mapping = self._data_provider.list_scalars(
            ctx,
            experiment_id=experiment,
            plugin_name=metadata.PLUGIN_NAME,
        )
        result = {run: {} for run in mapping}
        for run, tag_to_content in mapping.items():
            for tag, metadatum in tag_to_content.items():
                md = metadata.parse_plugin_metadata(metadatum.plugin_content)
                if not self._version_checker.ok(md.version, run, tag):
                    continue
                description = plugin_util.markdown_to_safe_html(
                    metadatum.description
                )
                result[run][tag] = {
                    "displayName": metadatum.display_name,
                    "description": description,
                }
        return result

    def scalars_impl(self, ctx, tag, run, experiment, output_format):
        """Result of the form `(body, mime_type)`."""
        all_scalars = self._data_provider.read_scalars(
            ctx,
            experiment_id=experiment,
            plugin_name=metadata.PLUGIN_NAME,
            downsample=self._downsample_to,
            run_tag_filter=provider.RunTagFilter(runs=[run], tags=[tag]),
        )
        scalars = all_scalars.get(run, {}).get(tag, None)
        if scalars is None:
            raise errors.NotFoundError(
                "No scalar data for run=%r, tag=%r" % (run, tag)
            )
        values = [(x.wall_time, x.step, x.value) for x in scalars]
        if output_format == OutputFormat.CSV:
            string_io = io.StringIO()
            writer = csv.writer(string_io)
            writer.writerow(["Wall time", "Step", "Value"])
            writer.writerows(values)
            return (string_io.getvalue(), "text/csv")
        else:
            return (values, "application/json")

    def scalars_multirun_impl(self, ctx, tag, runs, experiment):
        """Result of the form `(body, mime_type)`."""
        all_scalars = self._data_provider.read_scalars(
            ctx,
            experiment_id=experiment,
            plugin_name=metadata.PLUGIN_NAME,
            downsample=self._downsample_to,
            run_tag_filter=provider.RunTagFilter(runs=runs, tags=[tag]),
        )
        body = {
            run: [(x.wall_time, x.step, x.value) for x in run_data[tag]]
            for (run, run_data) in all_scalars.items()
        }
        return (body, "application/json")

    @wrappers.Request.application
    def tags_route(self, request):
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        index = self.index_impl(ctx, experiment=experiment)
        return http_util.Respond(request, index, "application/json")

    @wrappers.Request.application
    def scalars_route(self, request):
        """Given a tag and single run, return array of ScalarEvents."""
        tag = request.args.get("tag")
        run = request.args.get("run")
        if tag is None or run is None:
            raise errors.InvalidArgumentError(
                "Both run and tag must be specified: tag=%r, run=%r"
                % (tag, run)
            )

        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        output_format = request.args.get("format")
        (body, mime_type) = self.scalars_impl(
            ctx, tag, run, experiment, output_format
        )
        return http_util.Respond(request, body, mime_type)

    @wrappers.Request.application
    def scalars_multirun_route(self, request):
        """Given a tag and list of runs, return dict of ScalarEvent arrays."""
        if request.method != "POST":
            raise werkzeug.exceptions.MethodNotAllowed(["POST"])
        tags = request.form.getlist("tag")
        runs = request.form.getlist("runs")
        if len(tags) != 1:
            raise errors.InvalidArgumentError(
                "tag must be specified exactly once"
            )
        tag = tags[0]

        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        (body, mime_type) = self.scalars_multirun_impl(
            ctx, tag, runs, experiment
        )
        return http_util.Respond(request, body, mime_type)
