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
"""Internal information about the mesh plugin."""

import dataclasses

from typing import Any

from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.mesh import plugin_data_pb2

PLUGIN_NAME = "mesh"

# The most recent value for the `version` field of the
# `MeshPluginData` proto.
_PROTO_VERSION = 0


@dataclasses.dataclass(frozen=True)
class MeshTensor:
    """A mesh tensor.

    Attributes:
      data: Tensor of shape `[dim_1, ..., dim_n, 3]` representing the mesh data
        of one of the following:
          - 3D coordinates of vertices
          - Indices of vertices within each triangle
          - Colors for each vertex
      content_type: Type of the mesh plugin data content.
      data_type: Data type of the elements in the tensor.
    """

    data: Any  # Expects type `tf.Tensor`, not specified here to avoid heavy TF dep.
    content_type: plugin_data_pb2.MeshPluginData.ContentType
    data_type: Any  # Expects type `tf.DType`, not specified here to avoid heavy TF dep.


def get_components_bitmask(content_types):
    """Creates bitmask for all existing components of the summary.

    Args:
      content_type: list of plugin_data_pb2.MeshPluginData.ContentType,
        representing all components related to the summary.
    Returns: bitmask based on passed tensors.
    """
    components = 0
    for content_type in content_types:
        if content_type == plugin_data_pb2.MeshPluginData.UNDEFINED:
            raise ValueError("Cannot include UNDEFINED content type in mask.")
        components = components | (1 << content_type)
    return components


def get_current_version():
    """Returns current verions of the proto."""
    return _PROTO_VERSION


def get_instance_name(name, content_type):
    """Returns a unique instance name for a given summary related to the
    mesh."""
    return "%s_%s" % (
        name,
        plugin_data_pb2.MeshPluginData.ContentType.Name(content_type),
    )


def create_summary_metadata(
    name,
    display_name,
    content_type,
    components,
    shape,
    description=None,
    json_config=None,
):
    """Creates summary metadata which defined at MeshPluginData proto.

    Arguments:
      name: Original merged (summaries of different types) summary name.
      display_name: The display name used in TensorBoard.
      content_type: Value from MeshPluginData.ContentType enum describing data.
      components: Bitmask representing present parts (vertices, colors, etc.) that
        belong to the summary.
      shape: list of dimensions sizes of the tensor.
      description: The description to show in TensorBoard.
      json_config: A string, JSON-serialized dictionary of ThreeJS classes
        configuration.

    Returns:
      A `summary_pb2.SummaryMetadata` protobuf object.
    """
    # Shape should be at least BxNx3 where B represents the batch dimensions
    # and N - the number of points, each with x,y,z coordinates.
    if len(shape) != 3:
        raise ValueError(
            "Tensor shape should be of shape BxNx3, but got %s." % str(shape)
        )
    mesh_plugin_data = plugin_data_pb2.MeshPluginData(
        version=get_current_version(),
        name=name,
        content_type=content_type,
        components=components,
        shape=shape,
        json_config=json_config,
    )
    content = mesh_plugin_data.SerializeToString()
    return summary_pb2.SummaryMetadata(
        display_name=display_name,  # Will not be used in TensorBoard UI.
        summary_description=description,
        plugin_data=summary_pb2.SummaryMetadata.PluginData(
            plugin_name=PLUGIN_NAME, content=content
        ),
    )


def parse_plugin_metadata(content):
    """Parse summary metadata to a Python object.

    Arguments:
      content: The `content` field of a `SummaryMetadata` proto
        corresponding to the mesh plugin.

    Returns:
      A `MeshPluginData` protobuf object.
    Raises: Error if the version of the plugin is not supported.
    """
    if not isinstance(content, bytes):
        raise TypeError("Content type must be bytes.")
    result = plugin_data_pb2.MeshPluginData.FromString(content)
    # Add components field to older version of the proto.
    if result.components == 0:
        result.components = get_components_bitmask(
            [
                plugin_data_pb2.MeshPluginData.VERTEX,
                plugin_data_pb2.MeshPluginData.FACE,
                plugin_data_pb2.MeshPluginData.COLOR,
            ]
        )
    return result
