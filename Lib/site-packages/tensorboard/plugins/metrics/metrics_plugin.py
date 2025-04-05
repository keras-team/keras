# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""The TensorBoard metrics plugin."""


import collections
import imghdr
import json

from werkzeug import wrappers

from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.histogram import metadata as histogram_metadata
from tensorboard.plugins.image import metadata as image_metadata
from tensorboard.plugins.metrics import metadata
from tensorboard.plugins.scalar import metadata as scalar_metadata


_IMGHDR_TO_MIMETYPE = {
    "bmp": "image/bmp",
    "gif": "image/gif",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "svg": "image/svg+xml",
}

_DEFAULT_IMAGE_MIMETYPE = "application/octet-stream"

_SINGLE_RUN_PLUGINS = frozenset(
    [histogram_metadata.PLUGIN_NAME, image_metadata.PLUGIN_NAME]
)

_SAMPLED_PLUGINS = frozenset([image_metadata.PLUGIN_NAME])


def _get_tag_description_info(mapping):
    """Gets maps from tags to descriptions, and descriptions to runs.

    Args:
        mapping: a nested map `d` such that `d[run][tag]` is a time series
          produced by DataProvider's `list_*` methods.

    Returns:
        A tuple containing
            tag_to_descriptions: A map from tag strings to a set of description
                strings.
            description_to_runs: A map from description strings to a set of run
                strings.
    """
    tag_to_descriptions = collections.defaultdict(set)
    description_to_runs = collections.defaultdict(set)
    for run, tag_to_content in mapping.items():
        for tag, metadatum in tag_to_content.items():
            description = metadatum.description
            if len(description):
                tag_to_descriptions[tag].add(description)
                description_to_runs[description].add(run)

    return tag_to_descriptions, description_to_runs


def _build_combined_description(descriptions, description_to_runs):
    """Creates a single description from a set of descriptions.

    Descriptions may be composites when a single tag has different descriptions
    across multiple runs.

    Args:
        descriptions: A list of description strings.
        description_to_runs: A map from description strings to a set of run
            strings.

    Returns:
        The combined description string.
    """
    prefixed_descriptions = []
    for description in descriptions:
        runs = sorted(description_to_runs[description])
        run_or_runs = "runs" if len(runs) > 1 else "run"
        run_header = "## For " + run_or_runs + ": " + ", ".join(runs)
        description_html = run_header + "\n" + description
        prefixed_descriptions.append(description_html)

    header = "# Multiple descriptions\n"
    return header + "\n".join(prefixed_descriptions)


def _get_tag_to_description(mapping):
    """Returns a map of tags to descriptions.

    Args:
        mapping: a nested map `d` such that `d[run][tag]` is a time series
          produced by DataProvider's `list_*` methods.

    Returns:
        A map from tag strings to description HTML strings. E.g.
        {
            "loss": "<h1>Multiple descriptions</h1><h2>For runs: test, train
            </h2><p>...</p>",
            "loss2": "<p>The lossy details</p>",
        }
    """
    tag_to_descriptions, description_to_runs = _get_tag_description_info(
        mapping
    )

    result = {}
    for tag in tag_to_descriptions:
        descriptions = sorted(tag_to_descriptions[tag])
        if len(descriptions) == 1:
            description = descriptions[0]
        else:
            description = _build_combined_description(
                descriptions, description_to_runs
            )
        result[tag] = plugin_util.markdown_to_safe_html(description)

    return result


def _get_run_tag_info(mapping):
    """Returns a map of run names to a list of tag names.

    Args:
        mapping: a nested map `d` such that `d[run][tag]` is a time series
          produced by DataProvider's `list_*` methods.

    Returns:
        A map from run strings to a list of tag strings. E.g.
            {"loss001a": ["actor/loss", "critic/loss"], ...}
    """
    return {run: sorted(mapping[run]) for run in mapping}


def _format_basic_mapping(mapping):
    """Prepares a scalar or histogram mapping for client consumption.

    Args:
        mapping: a nested map `d` such that `d[run][tag]` is a time series
          produced by DataProvider's `list_*` methods.

    Returns:
        A dict with the following fields:
            runTagInfo: the return type of `_get_run_tag_info`
            tagDescriptions: the return type of `_get_tag_to_description`
    """
    return {
        "runTagInfo": _get_run_tag_info(mapping),
        "tagDescriptions": _get_tag_to_description(mapping),
    }


def _format_image_blob_sequence_datum(sorted_datum_list, sample):
    """Formats image metadata from a list of BlobSequenceDatum's for clients.

    This expects that frontend clients need to access images based on the
    run+tag+sample.

    Args:
        sorted_datum_list: a list of DataProvider's `BlobSequenceDatum`, sorted by
            step. This can be produced via DataProvider's `read_blob_sequences`.
        sample: zero-indexed integer for the requested sample.

    Returns:
        A list of `ImageStepDatum` (see http_api.md).
    """
    # For images, ignore the first 2 items of a BlobSequenceDatum's values, which
    # correspond to width, height.
    index = sample + 2
    step_data = []
    for datum in sorted_datum_list:
        if len(datum.values) <= index:
            continue

        step_data.append(
            {
                "step": datum.step,
                "wallTime": datum.wall_time,
                "imageId": datum.values[index].blob_key,
            }
        )
    return step_data


def _get_tag_run_image_info(mapping):
    """Returns a map of tag names to run information.

    Args:
        mapping: the result of DataProvider's `list_blob_sequences`.

    Returns:
        A nested map from run strings to tag string to image info, where image
        info is an object of form {"maxSamplesPerStep": num}. For example,
        {
            "reshaped": {
                "test": {"maxSamplesPerStep": 1},
                "train": {"maxSamplesPerStep": 1}
            },
            "convolved": {"test": {"maxSamplesPerStep": 50}},
        }
    """
    tag_run_image_info = collections.defaultdict(dict)
    for run, tag_to_content in mapping.items():
        for tag, metadatum in tag_to_content.items():
            tag_run_image_info[tag][run] = {
                "maxSamplesPerStep": metadatum.max_length - 2  # width, height
            }
    return dict(tag_run_image_info)


def _format_image_mapping(mapping):
    """Prepares an image mapping for client consumption.

    Args:
        mapping: the result of DataProvider's `list_blob_sequences`.

    Returns:
        A dict with the following fields:
            tagRunSampledInfo: the return type of `_get_tag_run_image_info`
            tagDescriptions: the return type of `_get_tag_description_info`
    """
    return {
        "tagDescriptions": _get_tag_to_description(mapping),
        "tagRunSampledInfo": _get_tag_run_image_info(mapping),
    }


class MetricsPlugin(base_plugin.TBPlugin):
    """Metrics Plugin for TensorBoard."""

    plugin_name = metadata.PLUGIN_NAME

    def __init__(self, context):
        """Instantiates MetricsPlugin.

        Args:
            context: A base_plugin.TBContext instance. MetricsLoader checks that
                it contains a valid `data_provider`.
        """
        self._data_provider = context.data_provider

        # For histograms, use a round number + 1 since sampling includes both start
        # and end steps, so N+1 samples corresponds to dividing the step sequence
        # into N intervals.
        sampling_hints = context.sampling_hints or {}
        self._plugin_downsampling = {
            "scalars": sampling_hints.get(scalar_metadata.PLUGIN_NAME, 1000),
            "histograms": sampling_hints.get(
                histogram_metadata.PLUGIN_NAME, 51
            ),
            "images": sampling_hints.get(image_metadata.PLUGIN_NAME, 10),
        }
        self._scalar_version_checker = plugin_util._MetadataVersionChecker(
            data_kind="scalar time series",
            latest_known_version=0,
        )
        self._histogram_version_checker = plugin_util._MetadataVersionChecker(
            data_kind="histogram time series",
            latest_known_version=0,
        )
        self._image_version_checker = plugin_util._MetadataVersionChecker(
            data_kind="image time series",
            latest_known_version=0,
        )

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(
            is_ng_component=True, tab_name="Time Series"
        )

    def get_plugin_apps(self):
        return {
            "/tags": self._serve_tags,
            "/timeSeries": self._serve_time_series,
            "/imageData": self._serve_image_data,
        }

    def data_plugin_names(self):
        return (
            scalar_metadata.PLUGIN_NAME,
            histogram_metadata.PLUGIN_NAME,
            image_metadata.PLUGIN_NAME,
        )

    def is_active(self):
        return False  # 'data_plugin_names' suffices.

    @wrappers.Request.application
    def _serve_tags(self, request):
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        index = self._tags_impl(ctx, experiment=experiment)
        return http_util.Respond(request, index, "application/json")

    def _tags_impl(self, ctx, experiment=None):
        """Returns tag metadata for a given experiment's logged metrics.

        Args:
            ctx: A `tensorboard.context.RequestContext` value.
            experiment: optional string ID of the request's experiment.

        Returns:
            A nested dict 'd' with keys in ("scalars", "histograms", "images")
                and values being the return type of _format_*mapping.
        """
        scalar_mapping = self._data_provider.list_scalars(
            ctx,
            experiment_id=experiment,
            plugin_name=scalar_metadata.PLUGIN_NAME,
        )
        scalar_mapping = self._filter_by_version(
            scalar_mapping,
            scalar_metadata.parse_plugin_metadata,
            self._scalar_version_checker,
        )

        histogram_mapping = self._data_provider.list_tensors(
            ctx,
            experiment_id=experiment,
            plugin_name=histogram_metadata.PLUGIN_NAME,
        )
        if histogram_mapping is None:
            histogram_mapping = {}
        histogram_mapping = self._filter_by_version(
            histogram_mapping,
            histogram_metadata.parse_plugin_metadata,
            self._histogram_version_checker,
        )

        image_mapping = self._data_provider.list_blob_sequences(
            ctx,
            experiment_id=experiment,
            plugin_name=image_metadata.PLUGIN_NAME,
        )
        if image_mapping is None:
            image_mapping = {}
        image_mapping = self._filter_by_version(
            image_mapping,
            image_metadata.parse_plugin_metadata,
            self._image_version_checker,
        )

        result = {}
        result["scalars"] = _format_basic_mapping(scalar_mapping)
        result["histograms"] = _format_basic_mapping(histogram_mapping)
        result["images"] = _format_image_mapping(image_mapping)
        return result

    def _filter_by_version(self, mapping, parse_metadata, version_checker):
        """Filter `DataProvider.list_*` output by summary metadata version."""
        result = {run: {} for run in mapping}
        for run, tag_to_content in mapping.items():
            for tag, metadatum in tag_to_content.items():
                md = parse_metadata(metadatum.plugin_content)
                if not version_checker.ok(md.version, run, tag):
                    continue
                result[run][tag] = metadatum
        return result

    @wrappers.Request.application
    def _serve_time_series(self, request):
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        if request.method == "POST":
            series_requests_string = request.form.get("requests")
        else:
            series_requests_string = request.args.get("requests")
        if not series_requests_string:
            raise errors.InvalidArgumentError("Missing 'requests' field")
        try:
            series_requests = json.loads(series_requests_string)
        except ValueError:
            raise errors.InvalidArgumentError(
                "Unable to parse 'requests' as JSON"
            )

        response = self._time_series_impl(ctx, experiment, series_requests)
        return http_util.Respond(request, response, "application/json")

    def _time_series_impl(self, ctx, experiment, series_requests):
        """Constructs a list of responses from a list of series requests.

        Args:
            ctx: A `tensorboard.context.RequestContext` value.
            experiment: string ID of the request's experiment.
            series_requests: a list of `TimeSeriesRequest` dicts (see http_api.md).

        Returns:
            A list of `TimeSeriesResponse` dicts (see http_api.md).
        """
        responses = [
            self._get_time_series(ctx, experiment, request)
            for request in series_requests
        ]
        return responses

    def _create_base_response(self, series_request):
        tag = series_request.get("tag")
        run = series_request.get("run")
        plugin = series_request.get("plugin")
        sample = series_request.get("sample")
        response = {"plugin": plugin, "tag": tag}
        if isinstance(run, str):
            response["run"] = run
        if isinstance(sample, int):
            response["sample"] = sample

        return response

    def _get_invalid_request_error(self, series_request):
        tag = series_request.get("tag")
        plugin = series_request.get("plugin")
        run = series_request.get("run")
        sample = series_request.get("sample")

        if not isinstance(tag, str):
            return "Missing tag"

        if (
            plugin != scalar_metadata.PLUGIN_NAME
            and plugin != histogram_metadata.PLUGIN_NAME
            and plugin != image_metadata.PLUGIN_NAME
        ):
            return "Invalid plugin"

        if plugin in _SINGLE_RUN_PLUGINS and not isinstance(run, str):
            return "Missing run"

        if plugin in _SAMPLED_PLUGINS and not isinstance(sample, int):
            return "Missing sample"

        return None

    def _get_time_series(self, ctx, experiment, series_request):
        """Returns time series data for a given tag, plugin.

        Args:
            ctx: A `tensorboard.context.RequestContext` value.
            experiment: string ID of the request's experiment.
            series_request: a `TimeSeriesRequest` (see http_api.md).

        Returns:
            A `TimeSeriesResponse` dict (see http_api.md).
        """
        tag = series_request.get("tag")
        run = series_request.get("run")
        plugin = series_request.get("plugin")
        sample = series_request.get("sample")
        response = self._create_base_response(series_request)
        request_error = self._get_invalid_request_error(series_request)
        if request_error:
            response["error"] = request_error
            return response

        runs = [run] if run else None
        run_to_series = None
        if plugin == scalar_metadata.PLUGIN_NAME:
            run_to_series = self._get_run_to_scalar_series(
                ctx, experiment, tag, runs
            )

        if plugin == histogram_metadata.PLUGIN_NAME:
            run_to_series = self._get_run_to_histogram_series(
                ctx, experiment, tag, runs
            )

        if plugin == image_metadata.PLUGIN_NAME:
            run_to_series = self._get_run_to_image_series(
                ctx, experiment, tag, sample, runs
            )

        response["runToSeries"] = run_to_series
        return response

    def _get_run_to_scalar_series(self, ctx, experiment, tag, runs):
        """Builds a run-to-scalar-series dict for client consumption.

        Args:
            ctx: A `tensorboard.context.RequestContext` value.
            experiment: a string experiment id.
            tag: string of the requested tag.
            runs: optional list of run names as strings.

        Returns:
            A map from string run names to `ScalarStepDatum` (see http_api.md).
        """
        mapping = self._data_provider.read_scalars(
            ctx,
            experiment_id=experiment,
            plugin_name=scalar_metadata.PLUGIN_NAME,
            downsample=self._plugin_downsampling["scalars"],
            run_tag_filter=provider.RunTagFilter(runs=runs, tags=[tag]),
        )

        run_to_series = {}
        for result_run, tag_data in mapping.items():
            if tag not in tag_data:
                continue
            values = [
                {
                    "wallTime": datum.wall_time,
                    "step": datum.step,
                    "value": datum.value,
                }
                for datum in tag_data[tag]
            ]
            run_to_series[result_run] = values

        return run_to_series

    def _format_histogram_datum_bins(self, datum):
        """Formats a histogram datum's bins for client consumption.

        Args:
            datum: a DataProvider's TensorDatum.

        Returns:
            A list of `HistogramBin`s (see http_api.md).
        """
        numpy_list = datum.numpy.tolist()
        bins = [{"min": x[0], "max": x[1], "count": x[2]} for x in numpy_list]
        return bins

    def _get_run_to_histogram_series(self, ctx, experiment, tag, runs):
        """Builds a run-to-histogram-series dict for client consumption.

        Args:
            ctx: A `tensorboard.context.RequestContext` value.
            experiment: a string experiment id.
            tag: string of the requested tag.
            runs: optional list of run names as strings.

        Returns:
            A map from string run names to `HistogramStepDatum` (see http_api.md).
        """
        mapping = self._data_provider.read_tensors(
            ctx,
            experiment_id=experiment,
            plugin_name=histogram_metadata.PLUGIN_NAME,
            downsample=self._plugin_downsampling["histograms"],
            run_tag_filter=provider.RunTagFilter(runs=runs, tags=[tag]),
        )

        run_to_series = {}
        for result_run, tag_data in mapping.items():
            if tag not in tag_data:
                continue
            values = [
                {
                    "wallTime": datum.wall_time,
                    "step": datum.step,
                    "bins": self._format_histogram_datum_bins(datum),
                }
                for datum in tag_data[tag]
            ]
            run_to_series[result_run] = values

        return run_to_series

    def _get_run_to_image_series(self, ctx, experiment, tag, sample, runs):
        """Builds a run-to-image-series dict for client consumption.

        Args:
            ctx: A `tensorboard.context.RequestContext` value.
            experiment: a string experiment id.
            tag: string of the requested tag.
            sample: zero-indexed integer for the requested sample.
            runs: optional list of run names as strings.

        Returns:
            A `RunToSeries` dict (see http_api.md).
        """
        mapping = self._data_provider.read_blob_sequences(
            ctx,
            experiment_id=experiment,
            plugin_name=image_metadata.PLUGIN_NAME,
            downsample=self._plugin_downsampling["images"],
            run_tag_filter=provider.RunTagFilter(runs, tags=[tag]),
        )

        run_to_series = {}
        for result_run, tag_data in mapping.items():
            if tag not in tag_data:
                continue
            blob_sequence_datum_list = tag_data[tag]
            series = _format_image_blob_sequence_datum(
                blob_sequence_datum_list, sample
            )
            if series:
                run_to_series[result_run] = series

        return run_to_series

    @wrappers.Request.application
    def _serve_image_data(self, request):
        """Serves an individual image."""
        ctx = plugin_util.context(request.environ)
        blob_key = request.args["imageId"]
        if not blob_key:
            raise errors.InvalidArgumentError("Missing 'imageId' field")

        (data, content_type) = self._image_data_impl(ctx, blob_key)
        return http_util.Respond(request, data, content_type)

    def _image_data_impl(self, ctx, blob_key):
        """Gets the image data for a blob key.

        Args:
            ctx: A `tensorboard.context.RequestContext` value.
            blob_key: a string identifier for a DataProvider blob.

        Returns:
            A tuple containing:
              data: a raw bytestring of the requested image's contents.
              content_type: a string HTTP content type.
        """
        data = self._data_provider.read_blob(ctx, blob_key=blob_key)
        image_type = imghdr.what(None, data)
        content_type = _IMGHDR_TO_MIMETYPE.get(
            image_type, _DEFAULT_IMAGE_MIMETYPE
        )
        return (data, content_type)
