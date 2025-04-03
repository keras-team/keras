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


import numpy as np
from werkzeug import wrappers

from tensorboard import plugin_util
from tensorboard.data import provider
from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin
from tensorboard.plugins.pr_curve import metadata


_DEFAULT_DOWNSAMPLING = 100  # PR curves per time series


class PrCurvesPlugin(base_plugin.TBPlugin):
    """A plugin that serves PR curves for individual classes."""

    plugin_name = metadata.PLUGIN_NAME

    def __init__(self, context):
        """Instantiates a PrCurvesPlugin.

        Args:
          context: A base_plugin.TBContext instance. A magic container that
            TensorBoard uses to make objects available to the plugin.
        """
        self._data_provider = context.data_provider
        self._downsample_to = (context.sampling_hints or {}).get(
            metadata.PLUGIN_NAME, _DEFAULT_DOWNSAMPLING
        )
        self._version_checker = plugin_util._MetadataVersionChecker(
            data_kind="PR curve",
            latest_known_version=0,
        )

    @wrappers.Request.application
    def pr_curves_route(self, request):
        """A route that returns a JSON mapping between runs and PR curve data.

        Returns:
          Given a tag and a comma-separated list of runs (both stored within GET
          parameters), fetches a JSON object that maps between run name and objects
          containing data required for PR curves for that run. Runs that either
          cannot be found or that lack tags will be excluded from the response.
        """
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)

        runs = request.args.getlist("run")
        if not runs:
            return http_util.Respond(
                request, "No runs provided when fetching PR curve data", 400
            )

        tag = request.args.get("tag")
        if not tag:
            return http_util.Respond(
                request, "No tag provided when fetching PR curve data", 400
            )

        try:
            response = http_util.Respond(
                request,
                self.pr_curves_impl(ctx, experiment, runs, tag),
                "application/json",
            )
        except ValueError as e:
            return http_util.Respond(request, str(e), "text/plain", 400)

        return response

    def pr_curves_impl(self, ctx, experiment, runs, tag):
        """Creates the JSON object for the PR curves response for a run-tag
        combo.

        Arguments:
          runs: A list of runs to fetch the curves for.
          tag: The tag to fetch the curves for.

        Raises:
          ValueError: If no PR curves could be fetched for a run and tag.

        Returns:
          The JSON object for the PR curves route response.
        """
        response_mapping = {}
        rtf = provider.RunTagFilter(runs, [tag])
        read_result = self._data_provider.read_tensors(
            ctx,
            experiment_id=experiment,
            plugin_name=metadata.PLUGIN_NAME,
            run_tag_filter=rtf,
            downsample=self._downsample_to,
        )
        for run in runs:
            data = read_result.get(run, {}).get(tag)
            if data is None:
                raise ValueError(
                    "No PR curves could be found for run %r and tag %r"
                    % (run, tag)
                )
            response_mapping[run] = [self._process_datum(d) for d in data]
        return response_mapping

    @wrappers.Request.application
    def tags_route(self, request):
        """A route (HTTP handler) that returns a response with tags.

        Returns:
          A response that contains a JSON object. The keys of the object
          are all the runs. Each run is mapped to a (potentially empty) dictionary
          whose keys are tags associated with run and whose values are metadata
          (dictionaries).

          The metadata dictionaries contain 2 keys:
            - displayName: For the display name used atop visualizations in
                TensorBoard.
            - description: The description that appears near visualizations upon the
                user hovering over a certain icon.
        """
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        return http_util.Respond(
            request, self.tags_impl(ctx, experiment), "application/json"
        )

    def tags_impl(self, ctx, experiment):
        """Creates the JSON object for the tags route response.

        Returns:
          The JSON object for the tags route response.
        """
        mapping = self._data_provider.list_tensors(
            ctx, experiment_id=experiment, plugin_name=metadata.PLUGIN_NAME
        )
        result = {run: {} for run in mapping}
        for run, tag_to_time_series in mapping.items():
            for tag, time_series in tag_to_time_series.items():
                md = metadata.parse_plugin_metadata(time_series.plugin_content)
                if not self._version_checker.ok(md.version, run, tag):
                    continue
                result[run][tag] = {
                    "displayName": time_series.display_name,
                    "description": plugin_util.markdown_to_safe_html(
                        time_series.description
                    ),
                }
        return result

    def get_plugin_apps(self):
        """Gets all routes offered by the plugin.

        Returns:
          A dictionary mapping URL path to route that handles it.
        """
        return {
            "/tags": self.tags_route,
            "/pr_curves": self.pr_curves_route,
        }

    def is_active(self):
        return False  # `list_plugins` as called by TB core suffices

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(
            element_name="tf-pr-curve-dashboard",
            tab_name="PR Curves",
        )

    def _process_datum(self, datum):
        """Converts a TensorDatum into a dict that encapsulates information on
        it.

        Args:
          datum: The TensorDatum to convert.

        Returns:
          A JSON-able dictionary of PR curve data for 1 step.
        """
        return self._make_pr_entry(datum.step, datum.wall_time, datum.numpy)

    def _make_pr_entry(self, step, wall_time, data_array):
        """Creates an entry for PR curve data. Each entry corresponds to 1
        step.

        Args:
          step: The step.
          wall_time: The wall time.
          data_array: A numpy array of PR curve data stored in the summary format.

        Returns:
          A PR curve entry.
        """
        tp_index = metadata.TRUE_POSITIVES_INDEX
        fp_index = metadata.FALSE_POSITIVES_INDEX
        tn_index = metadata.TRUE_NEGATIVES_INDEX
        fn_index = metadata.FALSE_NEGATIVES_INDEX

        # Trim entries for which TP + FP = 0 (precision is undefined) at the tail of
        # the data.
        positives = data_array[[tp_index, fp_index], :].astype(int).sum(axis=0)
        # Searching from the end, find the farthest index where TP + FP = 0.
        end_index_inclusive = len(positives) - 1
        while end_index_inclusive > 0 and positives[end_index_inclusive] == 0:
            end_index_inclusive -= 1
        end_index = end_index_inclusive + 1
        # Generate thresholds in [0, 1].
        num_thresholds = data_array.shape[1]
        thresholds = np.linspace(0.0, 1.0, num_thresholds)

        true_positives = [int(v) for v in data_array[tp_index]]
        false_positives = [int(v) for v in data_array[fp_index]]
        true_negatives = [int(v) for v in data_array[tn_index]]
        false_negatives = [int(v) for v in data_array[fn_index]]

        return {
            "wall_time": wall_time,
            "step": step,
            "precision": data_array[
                metadata.PRECISION_INDEX, :end_index
            ].tolist(),
            "recall": data_array[metadata.RECALL_INDEX, :end_index].tolist(),
            "true_positives": true_positives[:end_index],
            "false_positives": false_positives[:end_index],
            "true_negatives": true_negatives[:end_index],
            "false_negatives": false_negatives[:end_index],
            "thresholds": thresholds[:end_index].tolist(),
        }
