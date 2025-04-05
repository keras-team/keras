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
"""The TensorBoard Images plugin."""


import imghdr
import urllib.parse

from werkzeug import wrappers

from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.image import metadata


_IMGHDR_TO_MIMETYPE = {
    "bmp": "image/bmp",
    "gif": "image/gif",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "svg": "image/svg+xml",
}

_DEFAULT_IMAGE_MIMETYPE = "application/octet-stream"
_DEFAULT_DOWNSAMPLING = 10  # images per time series


# Extend imghdr.tests to include svg.
def detect_svg(data, f):
    del f  # Unused.
    # Assume XML documents attached to image tag to be SVG.
    if data.startswith(b"<?xml ") or data.startswith(b"<svg "):
        return "svg"


imghdr.tests.append(detect_svg)


class ImagesPlugin(base_plugin.TBPlugin):
    """Images Plugin for TensorBoard."""

    plugin_name = metadata.PLUGIN_NAME

    def __init__(self, context):
        """Instantiates ImagesPlugin via TensorBoard core.

        Args:
          context: A base_plugin.TBContext instance.
        """
        self._downsample_to = (context.sampling_hints or {}).get(
            self.plugin_name, _DEFAULT_DOWNSAMPLING
        )
        self._data_provider = context.data_provider
        self._version_checker = plugin_util._MetadataVersionChecker(
            data_kind="image",
            latest_known_version=0,
        )

    def get_plugin_apps(self):
        return {
            "/images": self._serve_image_metadata,
            "/individualImage": self._serve_individual_image,
            "/tags": self._serve_tags,
        }

    def is_active(self):
        return False  # `list_plugins` as called by TB core suffices

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(element_name="tf-image-dashboard")

    def _index_impl(self, ctx, experiment):
        mapping = self._data_provider.list_blob_sequences(
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
                    "samples": metadatum.max_length - 2,  # width, height
                }
        return result

    @wrappers.Request.application
    def _serve_image_metadata(self, request):
        """Given a tag and list of runs, serve a list of metadata for images.

        Note that the images themselves are not sent; instead, we respond with URLs
        to the images. The frontend should treat these URLs as opaque and should not
        try to parse information about them or generate them itself, as the format
        may change.

        Args:
          request: A werkzeug.wrappers.Request object.

        Returns:
          A werkzeug.Response application.
        """
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        tag = request.args.get("tag")
        run = request.args.get("run")
        sample = int(request.args.get("sample", 0))
        try:
            response = self._image_response_for_run(
                ctx, experiment, run, tag, sample
            )
        except KeyError:
            return http_util.Respond(
                request, "Invalid run or tag", "text/plain", code=400
            )
        return http_util.Respond(request, response, "application/json")

    def _image_response_for_run(self, ctx, experiment, run, tag, sample):
        """Builds a JSON-serializable object with information about images.

        Args:
          run: The name of the run.
          tag: The name of the tag the images all belong to.
          sample: The zero-indexed sample of the image for which to retrieve
            information. For instance, setting `sample` to `2` will fetch
            information about only the third image of each batch. Steps with
            fewer than three images will be omitted from the results.

        Returns:
          A list of dictionaries containing the wall time, step, and URL
          for each image.

        Raises:
          KeyError, NotFoundError: If no image data exists for the given
            parameters.
        """
        all_images = self._data_provider.read_blob_sequences(
            ctx,
            experiment_id=experiment,
            plugin_name=metadata.PLUGIN_NAME,
            downsample=self._downsample_to,
            run_tag_filter=provider.RunTagFilter(runs=[run], tags=[tag]),
        )
        images = all_images.get(run, {}).get(tag, None)
        if images is None:
            raise errors.NotFoundError(
                "No image data for run=%r, tag=%r" % (run, tag)
            )
        return [
            {
                "wall_time": datum.wall_time,
                "step": datum.step,
                "query": self._data_provider_query(datum.values[sample + 2]),
            }
            for datum in images
            if len(datum.values) - 2 > sample
        ]

    def _filter_by_sample(self, tensor_events, sample):
        return [
            tensor_event
            for tensor_event in tensor_events
            if (
                len(tensor_event.tensor_proto.string_val) - 2  # width, height
                > sample
            )
        ]

    def _query_for_individual_image(self, run, tag, sample, index):
        """Builds a URL for accessing the specified image.

        This should be kept in sync with _serve_image_metadata. Note that the URL is
        *not* guaranteed to always return the same image, since images may be
        unloaded from the reservoir as new images come in.

        Args:
          run: The name of the run.
          tag: The tag.
          sample: The relevant sample index, zero-indexed. See documentation
            on `_image_response_for_run` for more details.
          index: The index of the image. Negative values are OK.

        Returns:
          A string representation of a URL that will load the index-th sampled image
          in the given run with the given tag.
        """
        query_string = urllib.parse.urlencode(
            {
                "run": run,
                "tag": tag,
                "sample": sample,
                "index": index,
            }
        )
        return query_string

    def _data_provider_query(self, blob_reference):
        return urllib.parse.urlencode({"blob_key": blob_reference.blob_key})

    def _get_generic_data_individual_image(self, ctx, blob_key):
        """Returns the actual image bytes for a given image.

        Args:
          blob_key: As returned by a previous `read_blob_sequences` call.

        Returns:
          A bytestring of the raw image bytes.
        """
        return self._data_provider.read_blob(ctx, blob_key=blob_key)

    @wrappers.Request.application
    def _serve_individual_image(self, request):
        """Serves an individual image."""
        try:
            ctx = plugin_util.context(request.environ)
            blob_key = request.args["blob_key"]
            data = self._get_generic_data_individual_image(ctx, blob_key)
        except (KeyError, IndexError):
            return http_util.Respond(
                request,
                "Invalid run, tag, index, or sample",
                "text/plain",
                code=400,
            )
        image_type = imghdr.what(None, data)
        content_type = _IMGHDR_TO_MIMETYPE.get(
            image_type, _DEFAULT_IMAGE_MIMETYPE
        )
        return http_util.Respond(request, data, content_type)

    @wrappers.Request.application
    def _serve_tags(self, request):
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        index = self._index_impl(ctx, experiment)
        return http_util.Respond(request, index, "application/json")
