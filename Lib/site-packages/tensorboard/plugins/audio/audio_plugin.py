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
"""The TensorBoard Audio plugin."""


import urllib.parse

from werkzeug import wrappers

from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.audio import metadata


_DEFAULT_MIME_TYPE = "application/octet-stream"
_DEFAULT_DOWNSAMPLING = 10  # audio clips per time series
_MIME_TYPES = {
    metadata.Encoding.Value("WAV"): "audio/wav",
}
_ALLOWED_MIME_TYPES = frozenset(
    list(_MIME_TYPES.values()) + [_DEFAULT_MIME_TYPE]
)


class AudioPlugin(base_plugin.TBPlugin):
    """Audio Plugin for TensorBoard."""

    plugin_name = metadata.PLUGIN_NAME

    def __init__(self, context):
        """Instantiates AudioPlugin via TensorBoard core.

        Args:
          context: A base_plugin.TBContext instance.
        """
        self._data_provider = context.data_provider
        self._downsample_to = (context.sampling_hints or {}).get(
            self.plugin_name, _DEFAULT_DOWNSAMPLING
        )
        self._version_checker = plugin_util._MetadataVersionChecker(
            data_kind="audio",
            latest_known_version=0,
        )

    def get_plugin_apps(self):
        return {
            "/audio": self._serve_audio_metadata,
            "/individualAudio": self._serve_individual_audio,
            "/tags": self._serve_tags,
        }

    def is_active(self):
        return False  # `list_plugins` as called by TB core suffices

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(element_name="tf-audio-dashboard")

    def _index_impl(self, ctx, experiment):
        """Return information about the tags in each run.

        Result is a dictionary of the form

            {
              "runName1": {
                "tagName1": {
                  "displayName": "The first tag",
                  "description": "<p>Long ago there was just one tag...</p>",
                  "samples": 3
                },
                "tagName2": ...,
                ...
              },
              "runName2": ...,
              ...
            }

        For each tag, `samples` is the greatest number of audio clips that
        appear at any particular step. (It's not related to "samples of a
        waveform.") For example, if for tag `minibatch_input` there are
        five audio clips at step 0 and ten audio clips at step 1, then the
        dictionary for `"minibatch_input"` will contain `"samples": 10`.
        """
        mapping = self._data_provider.list_blob_sequences(
            ctx,
            experiment_id=experiment,
            plugin_name=metadata.PLUGIN_NAME,
        )
        result = {run: {} for run in mapping}
        for run, tag_to_time_series in mapping.items():
            for tag, time_series in tag_to_time_series.items():
                md = metadata.parse_plugin_metadata(time_series.plugin_content)
                if not self._version_checker.ok(md.version, run, tag):
                    continue
                description = plugin_util.markdown_to_safe_html(
                    time_series.description
                )
                result[run][tag] = {
                    "displayName": time_series.display_name,
                    "description": description,
                    "samples": time_series.max_length,
                }
        return result

    @wrappers.Request.application
    def _serve_audio_metadata(self, request):
        """Given a tag and list of runs, serve a list of metadata for audio.

        Note that the actual audio data are not sent; instead, we respond
        with URLs to the audio. The frontend should treat these URLs as
        opaque and should not try to parse information about them or
        generate them itself, as the format may change.

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

        response = self._audio_response_for_run(
            ctx, experiment, run, tag, sample
        )
        return http_util.Respond(request, response, "application/json")

    def _audio_response_for_run(self, ctx, experiment, run, tag, sample):
        """Builds a JSON-serializable object with information about audio.

        Args:
          run: The name of the run.
          tag: The name of the tag the audio entries all belong to.
          sample: The zero-indexed sample of the audio sample for which to
          retrieve information. For instance, setting `sample` to `2` will
            fetch information about only the third audio clip of each batch,
            and steps with fewer than three audio clips will be omitted from
            the results.

        Returns:
          A list of dictionaries containing the wall time, step, label,
          content type, and query string for each audio entry.
        """
        all_audio = self._data_provider.read_blob_sequences(
            ctx,
            experiment_id=experiment,
            plugin_name=metadata.PLUGIN_NAME,
            downsample=self._downsample_to,
            run_tag_filter=provider.RunTagFilter(runs=[run], tags=[tag]),
        )
        audio = all_audio.get(run, {}).get(tag, None)
        if audio is None:
            raise errors.NotFoundError(
                "No audio data for run=%r, tag=%r" % (run, tag)
            )
        content_type = self._get_mime_type(ctx, experiment, run, tag)
        response = []
        for datum in audio:
            if len(datum.values) < sample:
                continue
            query = urllib.parse.urlencode(
                {
                    "blob_key": datum.values[sample].blob_key,
                    "content_type": content_type,
                }
            )
            response.append(
                {
                    "wall_time": datum.wall_time,
                    "label": "",
                    "step": datum.step,
                    "contentType": content_type,
                    "query": query,
                }
            )
        return response

    def _get_mime_type(self, ctx, experiment, run, tag):
        # TODO(@wchargin): Move this call from `/audio` (called many
        # times) to `/tags` (called few times) to reduce data provider
        # calls.
        mapping = self._data_provider.list_blob_sequences(
            ctx,
            experiment_id=experiment,
            plugin_name=metadata.PLUGIN_NAME,
        )
        time_series = mapping.get(run, {}).get(tag, None)
        if time_series is None:
            raise errors.NotFoundError(
                "No audio data for run=%r, tag=%r" % (run, tag)
            )
        parsed = metadata.parse_plugin_metadata(time_series.plugin_content)
        return _MIME_TYPES.get(parsed.encoding, _DEFAULT_MIME_TYPE)

    @wrappers.Request.application
    def _serve_individual_audio(self, request):
        """Serve encoded audio data."""
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        mime_type = request.args["content_type"]
        if mime_type not in _ALLOWED_MIME_TYPES:
            raise errors.InvalidArgumentError(
                "Illegal mime type %r" % mime_type
            )
        blob_key = request.args["blob_key"]
        data = self._data_provider.read_blob(ctx, blob_key=blob_key)
        return http_util.Respond(request, data, mime_type)

    @wrappers.Request.application
    def _serve_tags(self, request):
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        index = self._index_impl(ctx, experiment)
        return http_util.Respond(request, index, "application/json")
