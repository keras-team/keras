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
"""TensorBoard 3D mesh visualizer plugin."""

import numpy as np
from werkzeug import wrappers

from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.mesh import metadata
from tensorboard.plugins.mesh import plugin_data_pb2
from tensorboard import plugin_util

_DEFAULT_DOWNSAMPLING = 100  # meshes per time series


class MeshPlugin(base_plugin.TBPlugin):
    """A plugin that serves 3D visualization of meshes."""

    plugin_name = metadata.PLUGIN_NAME

    def __init__(self, context):
        """Instantiates a MeshPlugin via TensorBoard core.

        Args:
          context: A base_plugin.TBContext instance. A magic container that
            TensorBoard uses to make objects available to the plugin.
        """
        self._data_provider = context.data_provider
        self._downsample_to = (context.sampling_hints or {}).get(
            self.plugin_name, _DEFAULT_DOWNSAMPLING
        )
        self._version_checker = plugin_util._MetadataVersionChecker(
            data_kind="mesh",
            latest_known_version=0,
        )

    def _instance_tag_metadata(self, ctx, experiment, run, instance_tag):
        """Gets the `MeshPluginData` proto for an instance tag."""
        results = self._data_provider.list_tensors(
            ctx,
            experiment_id=experiment,
            plugin_name=metadata.PLUGIN_NAME,
            run_tag_filter=provider.RunTagFilter(
                runs=[run], tags=[instance_tag]
            ),
        )
        content = results[run][instance_tag].plugin_content
        return metadata.parse_plugin_metadata(content)

    def _tag(self, ctx, experiment, run, instance_tag):
        """Gets the user-facing tag name for an instance tag."""
        return self._instance_tag_metadata(
            ctx, experiment, run, instance_tag
        ).name

    def _instance_tags(self, ctx, experiment, run, tag):
        """Gets the instance tag names for a user-facing tag."""
        index = self._data_provider.list_tensors(
            ctx,
            experiment_id=experiment,
            plugin_name=metadata.PLUGIN_NAME,
            run_tag_filter=provider.RunTagFilter(runs=[run]),
        )
        return [
            instance_tag
            for (instance_tag, ts) in index.get(run, {}).items()
            if tag == metadata.parse_plugin_metadata(ts.plugin_content).name
        ]

    @wrappers.Request.application
    def _serve_tags(self, request):
        """A route (HTTP handler) that returns a response with tags.

        Args:
          request: The werkzeug.Request object.

        Returns:
          A response that contains a JSON object. The keys of the object
          are all the runs. Each run is mapped to a (potentially empty)
          list of all tags that are relevant to this plugin.
        """
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        all_runs = self._data_provider.list_tensors(
            ctx,
            experiment_id=experiment,
            plugin_name=metadata.PLUGIN_NAME,
        )

        # tagToContent is itself a dictionary mapping tag name to string
        # SummaryMetadata.plugin_data.content. Retrieve the keys of that dictionary
        # to obtain a list of tags associated with each run. For each tag estimate
        # number of samples.
        response = dict()
        for run, tags in all_runs.items():
            response[run] = dict()
            for instance_tag, metadatum in tags.items():
                md = metadata.parse_plugin_metadata(metadatum.plugin_content)
                if not self._version_checker.ok(md.version, run, instance_tag):
                    continue
                # Make sure we only operate on user-defined tags here.
                tag = self._tag(ctx, experiment, run, instance_tag)
                meta = self._instance_tag_metadata(
                    ctx, experiment, run, instance_tag
                )
                # Batch size must be defined, otherwise we don't know how many
                # samples were there.
                response[run][tag] = {"samples": meta.shape[0]}
        return http_util.Respond(request, response, "application/json")

    def get_plugin_apps(self):
        """Gets all routes offered by the plugin.

        This method is called by TensorBoard when retrieving all the
        routes offered by the plugin.

        Returns:
          A dictionary mapping URL path to route that handles it.
        """
        # Note that the methods handling routes are decorated with
        # @wrappers.Request.application.
        return {
            "/tags": self._serve_tags,
            "/meshes": self._serve_mesh_metadata,
            "/data": self._serve_mesh_data,
        }

    def is_active(self):
        return False  # `list_plugins` as called by TB core suffices

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(element_name="mesh-dashboard")

    def _get_sample(self, tensor_datum, sample):
        """Returns a single sample from a batch of samples."""
        return tensor_datum.numpy[sample].tolist()

    def _get_tensor_metadata(
        self, event, content_type, components, data_shape, config
    ):
        """Converts a TensorDatum into a JSON-compatible response.

        Args:
          event: TensorDatum object containing data in proto format.
          content_type: enum plugin_data_pb2.MeshPluginData.ContentType value,
            representing content type in TensorDatum.
          components: Bitmask representing all parts (vertices, colors, etc.) that
            belong to the summary.
          data_shape: list of dimensions sizes of the tensor.
          config: rendering scene configuration as dictionary.

        Returns:
          Dictionary of transformed metadata.
        """
        return {
            "wall_time": event.wall_time,
            "step": event.step,
            "content_type": content_type,
            "components": components,
            "config": config,
            "data_shape": list(data_shape),
        }

    def _get_tensor_data(self, event, sample):
        """Convert a TensorDatum into a JSON-compatible response."""
        data = self._get_sample(event, sample)
        return data

    def _collect_tensor_events(self, request, step=None):
        """Collects list of tensor events based on request."""
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        run = request.args.get("run")
        tag = request.args.get("tag")

        tensor_events = []  # List of tuples (meta, tensor) that contain tag.
        for instance_tag in self._instance_tags(ctx, experiment, run, tag):
            tensors = self._data_provider.read_tensors(
                ctx,
                experiment_id=experiment,
                plugin_name=metadata.PLUGIN_NAME,
                run_tag_filter=provider.RunTagFilter(
                    runs=[run], tags=[instance_tag]
                ),
                downsample=self._downsample_to,
            )[run][instance_tag]
            meta = self._instance_tag_metadata(
                ctx, experiment, run, instance_tag
            )
            tensor_events += [(meta, tensor) for tensor in tensors]

        if step is not None:
            tensor_events = [
                event for event in tensor_events if event[1].step == step
            ]
        else:
            # Make sure tensors sorted by step in ascending order.
            tensor_events = sorted(
                tensor_events, key=lambda tensor_data: tensor_data[1].step
            )

        return tensor_events

    @wrappers.Request.application
    def _serve_mesh_data(self, request):
        """A route that returns data for particular summary of specified type.

        Data can represent vertices coordinates, vertices indices in faces,
        vertices colors and so on. Each mesh may have different combination of
        abovementioned data and each type/part of mesh summary must be served as
        separate roundtrip to the server.

        Args:
          request: werkzeug.Request containing content_type as a name of enum
            plugin_data_pb2.MeshPluginData.ContentType.

        Returns:
          werkzeug.Response either float32 or int32 data in binary format.
        """
        step = float(request.args.get("step", 0.0))
        tensor_events = self._collect_tensor_events(request, step)
        content_type = request.args.get("content_type")
        try:
            content_type = plugin_data_pb2.MeshPluginData.ContentType.Value(
                content_type
            )
        except ValueError:
            return http_util.Respond(
                request, "Bad content_type", "text/plain", 400
            )
        sample = int(request.args.get("sample", 0))

        response = [
            self._get_tensor_data(tensor, sample)
            for meta, tensor in tensor_events
            if meta.content_type == content_type
        ]

        np_type = {
            plugin_data_pb2.MeshPluginData.VERTEX: np.float32,
            plugin_data_pb2.MeshPluginData.FACE: np.int32,
            plugin_data_pb2.MeshPluginData.COLOR: np.uint8,
        }[content_type]

        response = np.array(response, dtype=np_type)
        # Looks like reshape can take around 160ms, so why not store it reshaped.
        response = response.reshape(-1).tobytes()

        return http_util.Respond(request, response, "arraybuffer")

    @wrappers.Request.application
    def _serve_mesh_metadata(self, request):
        """A route that returns the mesh metadata associated with a tag.

        Metadata consists of wall time, type of elements in tensor, scene
        configuration and so on.

        Args:
          request: The werkzeug.Request object.

        Returns:
          A JSON list of mesh data associated with the run and tag
          combination.
        """
        tensor_events = self._collect_tensor_events(request)

        # We convert the tensor data to text.
        response = [
            self._get_tensor_metadata(
                tensor,
                meta.content_type,
                meta.components,
                meta.shape,
                meta.json_config,
            )
            for meta, tensor in tensor_events
        ]
        return http_util.Respond(request, response, "application/json")
