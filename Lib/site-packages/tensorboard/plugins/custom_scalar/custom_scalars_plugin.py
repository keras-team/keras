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
"""The TensorBoard Custom Scalars plugin.

This plugin lets the user create scalars plots with custom run-tag combinations
by specifying regular expressions.

See `http_api.md` in this directory for specifications of the routes for
this plugin.
"""

import re

from werkzeug import wrappers

from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.compat import tf
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.custom_scalar import layout_pb2
from tensorboard.plugins.custom_scalar import metadata
from tensorboard.plugins.scalar import metadata as scalars_metadata
from tensorboard.plugins.scalar import scalars_plugin


# The name of the property in the response for whether the regex is valid.
_REGEX_VALID_PROPERTY = "regex_valid"

# The name of the property in the response for the payload (tag to ScalarEvents
# mapping).
_TAG_TO_EVENTS_PROPERTY = "tag_to_events"

# The number of seconds to wait in between checks for the config file specifying
# layout.
_CONFIG_FILE_CHECK_THROTTLE = 60


class CustomScalarsPlugin(base_plugin.TBPlugin):
    """CustomScalars Plugin for TensorBoard."""

    plugin_name = metadata.PLUGIN_NAME

    def __init__(self, context):
        """Instantiates ScalarsPlugin via TensorBoard core.

        Args:
          context: A base_plugin.TBContext instance.
        """
        self._logdir = context.logdir
        self._data_provider = context.data_provider
        self._plugin_name_to_instance = context.plugin_name_to_instance

    def _get_scalars_plugin(self):
        """Tries to get the scalars plugin.

        Returns:
          The scalars plugin. Or None if it is not yet registered.
        """
        if scalars_metadata.PLUGIN_NAME in self._plugin_name_to_instance:
            # The plugin is registered.
            return self._plugin_name_to_instance[scalars_metadata.PLUGIN_NAME]
        # The plugin is not yet registered.
        return None

    def get_plugin_apps(self):
        return {
            "/download_data": self.download_data_route,
            "/layout": self.layout_route,
            "/scalars": self.scalars_route,
        }

    def is_active(self):
        """Plugin is active if there is a custom layout for the dashboard."""
        return False  # `list_plugins` as called by TB core suffices

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(
            element_name="tf-custom-scalar-dashboard",
            tab_name="Custom Scalars",
        )

    @wrappers.Request.application
    def download_data_route(self, request):
        ctx = plugin_util.context(request.environ)
        run = request.args.get("run")
        tag = request.args.get("tag")
        experiment = plugin_util.experiment_id(request.environ)
        response_format = request.args.get("format")
        try:
            body, mime_type = self.download_data_impl(
                ctx, run, tag, experiment, response_format
            )
        except ValueError as e:
            return http_util.Respond(
                request=request,
                content=str(e),
                content_type="text/plain",
                code=400,
            )
        return http_util.Respond(request, body, mime_type)

    def download_data_impl(self, ctx, run, tag, experiment, response_format):
        """Provides a response for downloading scalars data for a data series.

        Args:
          ctx: A tensorboard.context.RequestContext value.
          run: The run.
          tag: The specific tag.
          experiment: An experiment ID, as a possibly-empty `str`.
          response_format: A string. One of the values of the OutputFormat enum
            of the scalar plugin.

        Raises:
          ValueError: If the scalars plugin is not registered.

        Returns:
          2 entities:
            - A JSON object response body.
            - A mime type (string) for the response.
        """
        scalars_plugin_instance = self._get_scalars_plugin()
        if not scalars_plugin_instance:
            raise ValueError(
                (
                    "Failed to respond to request for /download_data. "
                    "The scalars plugin is oddly not registered."
                )
            )

        body, mime_type = scalars_plugin_instance.scalars_impl(
            ctx, tag, run, experiment, response_format
        )
        return body, mime_type

    @wrappers.Request.application
    def scalars_route(self, request):
        """Given a tag regex and single run, return ScalarEvents.

        This route takes 2 GET params:
        run: A run string to find tags for.
        tag: A string that is a regex used to find matching tags.
        The response is a JSON object:
        {
          // Whether the regular expression is valid. Also false if empty.
          regexValid: boolean,

          // An object mapping tag name to a list of ScalarEvents.
          payload: Object<string, ScalarEvent[]>,
        }
        """
        ctx = plugin_util.context(request.environ)
        tag_regex_string = request.args.get("tag")
        run = request.args.get("run")
        experiment = plugin_util.experiment_id(request.environ)
        mime_type = "application/json"

        try:
            body = self.scalars_impl(ctx, run, tag_regex_string, experiment)
        except ValueError as e:
            return http_util.Respond(
                request=request,
                content=str(e),
                content_type="text/plain",
                code=400,
            )

        # Produce the response.
        return http_util.Respond(request, body, mime_type)

    def scalars_impl(self, ctx, run, tag_regex_string, experiment):
        """Given a tag regex and single run, return ScalarEvents.

        Args:
          ctx: A tensorboard.context.RequestContext value.
          run: A run string.
          tag_regex_string: A regular expression that captures portions of tags.

        Raises:
          ValueError: if the scalars plugin is not registered.

        Returns:
          A dictionary that is the JSON-able response.
        """
        if not tag_regex_string:
            # The user provided no regex.
            return {
                _REGEX_VALID_PROPERTY: False,
                _TAG_TO_EVENTS_PROPERTY: {},
            }

        # Construct the regex.
        try:
            regex = re.compile(tag_regex_string)
        except re.error:
            return {
                _REGEX_VALID_PROPERTY: False,
                _TAG_TO_EVENTS_PROPERTY: {},
            }

        # Fetch the tags for the run. Filter for tags that match the regex.
        run_to_data = self._data_provider.list_scalars(
            ctx,
            experiment_id=experiment,
            plugin_name=scalars_metadata.PLUGIN_NAME,
            run_tag_filter=provider.RunTagFilter(runs=[run]),
        )

        tag_to_data = None
        try:
            tag_to_data = run_to_data[run]
        except KeyError:
            # The run could not be found. Perhaps a configuration specified a run that
            # TensorBoard has not read from disk yet.
            payload = {}

        if tag_to_data:
            scalars_plugin_instance = self._get_scalars_plugin()
            if not scalars_plugin_instance:
                raise ValueError(
                    (
                        "Failed to respond to request for /scalars. "
                        "The scalars plugin is oddly not registered."
                    )
                )

            form = scalars_plugin.OutputFormat.JSON
            payload = {
                tag: scalars_plugin_instance.scalars_impl(
                    ctx, tag, run, experiment, form
                )[0]
                for tag in tag_to_data.keys()
                if regex.match(tag)
            }

        return {
            _REGEX_VALID_PROPERTY: True,
            _TAG_TO_EVENTS_PROPERTY: payload,
        }

    @wrappers.Request.application
    def layout_route(self, request):
        """Fetches the custom layout specified by the config file in the logdir.

        If more than 1 run contains a layout, this method merges the layouts by
        merging charts within individual categories. If 2 categories with the same
        name are found, the charts within are merged. The merging is based on the
        order of the runs to which the layouts are written.

        The response is a JSON object mirroring properties of the Layout proto if a
        layout for any run is found.

        The response is an empty object if no layout could be found.
        """
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        body = self.layout_impl(ctx, experiment)
        return http_util.Respond(request, body, "application/json")

    def layout_impl(self, ctx, experiment):
        # Keep a mapping between and category so we do not create duplicate
        # categories.
        title_to_category = {}

        merged_layout = None
        data = self._data_provider.read_tensors(
            ctx,
            experiment_id=experiment,
            plugin_name=metadata.PLUGIN_NAME,
            run_tag_filter=provider.RunTagFilter(
                tags=[metadata.CONFIG_SUMMARY_TAG]
            ),
            downsample=1,
        )
        for run in sorted(data):
            points = data[run][metadata.CONFIG_SUMMARY_TAG]
            content = points[0].numpy.item()
            layout_proto = layout_pb2.Layout()
            layout_proto.ParseFromString(tf.compat.as_bytes(content))

            if merged_layout:
                # Append the categories within this layout to the merged layout.
                for category in layout_proto.category:
                    if category.title in title_to_category:
                        # A category with this name has been seen before. Do not create a
                        # new one. Merge their charts, skipping any duplicates.
                        title_to_category[category.title].chart.extend(
                            [
                                c
                                for c in category.chart
                                if c
                                not in title_to_category[category.title].chart
                            ]
                        )
                    else:
                        # This category has not been seen before.
                        merged_layout.category.add().MergeFrom(category)
                        title_to_category[category.title] = category
            else:
                # This is the first layout encountered.
                merged_layout = layout_proto
                for category in layout_proto.category:
                    title_to_category[category.title] = category

        if merged_layout:
            return plugin_util.proto_to_json(merged_layout)
        else:
            # No layout was found.
            return {}
