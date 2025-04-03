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

V2 versions
"""

import json

from tensorboard.compat import tf2 as tf
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.mesh import metadata
from tensorboard.plugins.mesh import plugin_data_pb2
from tensorboard.util import tensor_util


def _write_summary(
    name, description, tensor, content_type, components, json_config, step
):
    """Creates a tensor summary with summary metadata.

    Args:
      name: A name for this summary. The summary tag used for TensorBoard will
        be this name prefixed by any active name scopes.
      description: Optional long-form description for this summary, as a
        constant `str`. Markdown is supported. Defaults to empty.
      tensor: Tensor to display in summary.
      content_type: Type of content inside the Tensor.
      components: Bitmask representing present parts (vertices, colors, etc.) that
        belong to the summary.
      json_config: A string, JSON-serialized dictionary of ThreeJS classes
        configuration.
      step: Explicit `int64`-castable monotonic step value for this summary. If
        omitted, this defaults to `tf.summary.experimental.get_step()`, which must
        not be None.

    Returns:
      A boolean indicating if summary was saved successfully or not.
    """
    tensor = tf.convert_to_tensor(value=tensor)
    shape = tensor.shape.as_list()
    shape = [dim if dim is not None else -1 for dim in shape]
    tensor_metadata = metadata.create_summary_metadata(
        name,
        None,  # display_name
        content_type,
        components,
        shape,
        description,
        json_config=json_config,
    )
    return tf.summary.write(
        tag=metadata.get_instance_name(name, content_type),
        tensor=tensor,
        step=step,
        metadata=tensor_metadata,
    )


def _get_json_config(config_dict):
    """Parses and returns JSON string from python dictionary."""
    json_config = "{}"
    if config_dict is not None:
        json_config = json.dumps(config_dict, sort_keys=True)
    return json_config


def mesh(
    name,
    vertices,
    faces=None,
    colors=None,
    config_dict=None,
    step=None,
    description=None,
):
    """Writes a TensorFlow mesh summary.

    Args:
      name: A name for this summary. The summary tag used for TensorBoard will
        be this name prefixed by any active name scopes.
      vertices: Tensor of shape `[dim_1, ..., dim_n, 3]` representing the 3D
        coordinates of vertices.
      faces: Tensor of shape `[dim_1, ..., dim_n, 3]` containing indices of
        vertices within each triangle.
      colors: Tensor of shape `[dim_1, ..., dim_n, 3]` containing colors for each
        vertex.
      config_dict: Dictionary with ThreeJS classes names and configuration.
      step: Explicit `int64`-castable monotonic step value for this summary. If
        omitted, this defaults to `tf.summary.experimental.get_step()`, which must
        not be None.
      description: Optional long-form description for this summary, as a
        constant `str`. Markdown is supported. Defaults to empty.

    Returns:
      True if all components of the mesh were saved successfully and False
        otherwise.
    """
    json_config = _get_json_config(config_dict)

    # All tensors representing a single mesh will be represented as separate
    # summaries internally. Those summaries will be regrouped on the client before
    # rendering.
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

    summary_scope = (
        getattr(tf.summary.experimental, "summary_scope", None)
        or tf.summary.summary_scope
    )
    all_success = True
    with summary_scope(name, "mesh_summary", values=tensors):
        for tensor in tensors:
            all_success = all_success and _write_summary(
                name,
                description,
                tensor.data,
                tensor.content_type,
                components,
                json_config,
                step,
            )

    return all_success


def mesh_pb(
    tag, vertices, faces=None, colors=None, config_dict=None, description=None
):
    """Create a mesh summary to save in pb format.

    Args:
      tag: String tag for the summary.
      vertices: numpy array of shape `[dim_1, ..., dim_n, 3]` representing the 3D
        coordinates of vertices.
      faces: numpy array of shape `[dim_1, ..., dim_n, 3]` containing indices of
        vertices within each triangle.
      colors: numpy array of shape `[dim_1, ..., dim_n, 3]` containing colors for
        each vertex.
      config_dict: Dictionary with ThreeJS classes names and configuration.
      description: Optional long-form description for this summary, as a
        constant `str`. Markdown is supported. Defaults to empty.

    Returns:
      Instance of tf.Summary class.
    """
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
        tensor_proto = tensor_util.make_tensor_proto(
            tensor.data, dtype=tensor.data_type
        )
        summary_metadata = metadata.create_summary_metadata(
            tag,
            None,  # display_name
            tensor.content_type,
            components,
            shape,
            description,
            json_config=json_config,
        )
        instance_tag = metadata.get_instance_name(tag, tensor.content_type)
        summaries.append((instance_tag, summary_metadata, tensor_proto))

    summary = summary_pb2.Summary()
    for instance_tag, summary_metadata, tensor_proto in summaries:
        summary.value.add(
            tag=instance_tag, metadata=summary_metadata, tensor=tensor_proto
        )
    return summary
