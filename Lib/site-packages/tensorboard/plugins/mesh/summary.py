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
"""Mesh summaries and TensorFlow operations to create them.

This file is deprecated. See `summary_v2.py` instead.
"""

import json
import tensorflow as tf

from tensorboard.plugins.mesh import metadata
from tensorboard.plugins.mesh import plugin_data_pb2
from tensorboard.plugins.mesh import summary_v2

# Export V2 versions.
mesh = summary_v2.mesh
mesh_pb = summary_v2.mesh_pb


def _get_tensor_summary(
    name,
    display_name,
    description,
    tensor,
    content_type,
    components,
    json_config,
    collections,
):
    """Creates a tensor summary with summary metadata.

    Args:
      name: Uniquely identifiable name of the summary op. Could be replaced by
        combination of name and type to make it unique even outside of this
        summary.
      display_name: Will be used as the display name in TensorBoard.
        Defaults to `tag`.
      description: A longform readable description of the summary data. Markdown
        is supported.
      tensor: Tensor to display in summary.
      content_type: Type of content inside the Tensor.
      components: Bitmask representing present parts (vertices, colors, etc.) that
        belong to the summary.
      json_config: A string, JSON-serialized dictionary of ThreeJS classes
        configuration.
      collections: List of collections to add this summary to.

    Returns:
      Tensor summary with metadata.
    """
    tensor = tf.convert_to_tensor(value=tensor)
    shape = tensor.shape.as_list()
    shape = [dim if dim is not None else -1 for dim in shape]
    tensor_metadata = metadata.create_summary_metadata(
        name,
        display_name,
        content_type,
        components,
        shape,
        description,
        json_config=json_config,
    )
    tensor_summary = tf.compat.v1.summary.tensor_summary(
        metadata.get_instance_name(name, content_type),
        tensor,
        summary_metadata=tensor_metadata,
        collections=collections,
    )
    return tensor_summary


def _get_display_name(name, display_name):
    """Returns display_name from display_name and name."""
    if display_name is None:
        return name
    return display_name


def _get_json_config(config_dict):
    """Parses and returns JSON string from python dictionary."""
    json_config = "{}"
    if config_dict is not None:
        json_config = json.dumps(config_dict, sort_keys=True)
    return json_config


def op(
    name,
    vertices,
    faces=None,
    colors=None,
    display_name=None,
    description=None,
    collections=None,
    config_dict=None,
):
    """Creates a TensorFlow summary op for mesh rendering.

    DEPRECATED: see `summary_v2.py` instead.

    Args:
      name: A name for this summary operation.
      vertices: Tensor of shape `[dim_1, ..., dim_n, 3]` representing the 3D
        coordinates of vertices.
      faces: Tensor of shape `[dim_1, ..., dim_n, 3]` containing indices of
        vertices within each triangle.
      colors: Tensor of shape `[dim_1, ..., dim_n, 3]` containing colors for each
        vertex.
      display_name: If set, will be used as the display name in TensorBoard.
        Defaults to `name`.
      description: A longform readable description of the summary data. Markdown
        is supported.
      collections: Which TensorFlow graph collections to add the summary op to.
        Defaults to `['summaries']`. Can usually be ignored.
      config_dict: Dictionary with ThreeJS classes names and configuration.

    Returns:
      Merged summary for mesh/point cloud representation.
    """
    display_name = _get_display_name(name, display_name)
    json_config = _get_json_config(config_dict)

    # All tensors representing a single mesh will be represented as separate
    # summaries internally. Those summaries will be regrouped on the client before
    # rendering.
    summaries = []
    tensors = [
        metadata.MeshTensor(
            vertices, plugin_data_pb2.MeshPluginData.VERTEX, tf.float32
        ),
        metadata.MeshTensor(
            faces, plugin_data_pb2.MeshPluginData.FACE, tf.int32
        ),
        metadata.MeshTensor(
            colors, plugin_data_pb2.MeshPluginData.COLOR, tf.uint8
        ),
    ]
    tensors = [tensor for tensor in tensors if tensor.data is not None]

    components = metadata.get_components_bitmask(
        [tensor.content_type for tensor in tensors]
    )

    for tensor in tensors:
        summaries.append(
            _get_tensor_summary(
                name,
                display_name,
                description,
                tensor.data,
                tensor.content_type,
                components,
                json_config,
                collections,
            )
        )

    all_summaries = tf.compat.v1.summary.merge(
        summaries, collections=collections, name=name
    )
    return all_summaries


def pb(
    name,
    vertices,
    faces=None,
    colors=None,
    display_name=None,
    description=None,
    config_dict=None,
):
    """Create a mesh summary to save in pb format.

    DEPRECATED: see `summary_v2.py` instead.

    Args:
      name: A name for this summary operation.
      vertices: numpy array of shape `[dim_1, ..., dim_n, 3]` representing the 3D
        coordinates of vertices.
      faces: numpy array of shape `[dim_1, ..., dim_n, 3]` containing indices of
        vertices within each triangle.
      colors: numpy array of shape `[dim_1, ..., dim_n, 3]` containing colors for
        each vertex.
      display_name: If set, will be used as the display name in TensorBoard.
        Defaults to `name`.
      description: A longform readable description of the summary data. Markdown
        is supported.
      config_dict: Dictionary with ThreeJS classes names and configuration.

    Returns:
      Instance of tf.Summary class.
    """
    display_name = _get_display_name(name, display_name)
    json_config = _get_json_config(config_dict)

    summaries = []
    tensors = [
        metadata.MeshTensor(
            vertices, plugin_data_pb2.MeshPluginData.VERTEX, tf.float32
        ),
        metadata.MeshTensor(
            faces, plugin_data_pb2.MeshPluginData.FACE, tf.int32
        ),
        metadata.MeshTensor(
            colors, plugin_data_pb2.MeshPluginData.COLOR, tf.uint8
        ),
    ]
    tensors = [tensor for tensor in tensors if tensor.data is not None]
    components = metadata.get_components_bitmask(
        [tensor.content_type for tensor in tensors]
    )
    for tensor in tensors:
        shape = tensor.data.shape
        shape = [dim if dim is not None else -1 for dim in shape]
        tensor_proto = tf.compat.v1.make_tensor_proto(
            tensor.data, dtype=tensor.data_type
        )
        summary_metadata = metadata.create_summary_metadata(
            name,
            display_name,
            tensor.content_type,
            components,
            shape,
            description,
            json_config=json_config,
        )
        tag = metadata.get_instance_name(name, tensor.content_type)
        summaries.append((tag, summary_metadata, tensor_proto))

    summary = tf.compat.v1.Summary()
    for tag, summary_metadata, tensor_proto in summaries:
        tf_summary_metadata = tf.compat.v1.SummaryMetadata.FromString(
            summary_metadata.SerializeToString()
        )
        summary.value.add(
            tag=tag, metadata=tf_summary_metadata, tensor=tensor_proto
        )
    return summary
