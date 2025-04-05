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
"""Utilities to migrate legacy summaries/events to generic data form.

For legacy summaries, this populates the `SummaryMetadata.data_class`
field and makes any necessary transformations to the tensor value. For
`graph_def` events, this creates a new summary event.

This should be effected after the `data_compat` transformation.
"""


from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.audio import metadata as audio_metadata
from tensorboard.plugins.custom_scalar import (
    metadata as custom_scalars_metadata,
)
from tensorboard.plugins.graph import metadata as graphs_metadata
from tensorboard.plugins.histogram import metadata as histograms_metadata
from tensorboard.plugins.hparams import metadata as hparams_metadata
from tensorboard.plugins.image import metadata as images_metadata
from tensorboard.plugins.mesh import metadata as mesh_metadata
from tensorboard.plugins.pr_curve import metadata as pr_curves_metadata
from tensorboard.plugins.scalar import metadata as scalars_metadata
from tensorboard.plugins.text import metadata as text_metadata
from tensorboard.util import tensor_util


def migrate_event(event, initial_metadata):
    """Migrate an event to a sequence of events.

    Args:
      event: An `event_pb2.Event`. The caller transfers ownership of the
        event to this method; the event may be mutated, and may or may
        not appear in the returned sequence.
      initial_metadata: Map from tag name (string) to `SummaryMetadata`
        proto for the initial occurrence of the given tag within the
        enclosing run. While loading a given run, the caller should
        always pass the same dictionary here, initially `{}`; this
        function will mutate it and reuse it for future calls.

    Returns:
      A sequence of `event_pb2.Event`s to use instead of `event`.
    """
    what = event.WhichOneof("what")
    if what == "graph_def":
        return _migrate_graph_event(event)
    if what == "tagged_run_metadata":
        return _migrate_tagged_run_metadata_event(event)
    if what == "summary":
        return _migrate_summary_event(event, initial_metadata)
    return (event,)


def _migrate_graph_event(old_event):
    result = event_pb2.Event()
    result.wall_time = old_event.wall_time
    result.step = old_event.step
    value = result.summary.value.add(tag=graphs_metadata.RUN_GRAPH_NAME)
    graph_bytes = old_event.graph_def
    value.tensor.CopyFrom(tensor_util.make_tensor_proto([graph_bytes]))
    value.metadata.plugin_data.plugin_name = graphs_metadata.PLUGIN_NAME
    # `value.metadata.plugin_data.content` left empty
    value.metadata.data_class = summary_pb2.DATA_CLASS_BLOB_SEQUENCE
    # As long as the graphs plugin still reads the old format, keep both
    # the old event and the new event to maintain compatibility.
    return (old_event, result)


def _migrate_tagged_run_metadata_event(old_event):
    result = event_pb2.Event()
    result.wall_time = old_event.wall_time
    result.step = old_event.step
    trm = old_event.tagged_run_metadata
    value = result.summary.value.add(tag=trm.tag)
    value.tensor.CopyFrom(tensor_util.make_tensor_proto([trm.run_metadata]))
    value.metadata.plugin_data.plugin_name = (
        graphs_metadata.PLUGIN_NAME_TAGGED_RUN_METADATA
    )
    # `value.metadata.plugin_data.content` left empty
    value.metadata.data_class = summary_pb2.DATA_CLASS_BLOB_SEQUENCE
    return (result,)


def _migrate_summary_event(event, initial_metadata):
    values = event.summary.value
    new_values = [
        new for old in values for new in _migrate_value(old, initial_metadata)
    ]
    # Optimization: Don't create a new event if there were no shallow
    # changes (there may still have been in-place changes).
    if len(values) == len(new_values) and all(
        x is y for (x, y) in zip(values, new_values)
    ):
        return (event,)
    del event.summary.value[:]
    event.summary.value.extend(new_values)
    return (event,)


def _migrate_value(value, initial_metadata):
    """Convert an old value to a stream of new values. May mutate."""
    metadata = initial_metadata.get(value.tag)
    initial = False
    if metadata is None:
        initial = True
        # Retain a copy of the initial metadata, so that even after we
        # update its data class we know whether to also transform later
        # events in this time series.
        metadata = summary_pb2.SummaryMetadata()
        metadata.CopyFrom(value.metadata)
        initial_metadata[value.tag] = metadata
    if metadata.data_class != summary_pb2.DATA_CLASS_UNKNOWN:
        return (value,)
    plugin_name = metadata.plugin_data.plugin_name
    if plugin_name == histograms_metadata.PLUGIN_NAME:
        return _migrate_histogram_value(value)
    if plugin_name == images_metadata.PLUGIN_NAME:
        return _migrate_image_value(value)
    if plugin_name == audio_metadata.PLUGIN_NAME:
        return _migrate_audio_value(value)
    if plugin_name == scalars_metadata.PLUGIN_NAME:
        return _migrate_scalar_value(value)
    if plugin_name == text_metadata.PLUGIN_NAME:
        return _migrate_text_value(value)
    if plugin_name == hparams_metadata.PLUGIN_NAME:
        return _migrate_hparams_value(value)
    if plugin_name == pr_curves_metadata.PLUGIN_NAME:
        return _migrate_pr_curve_value(value)
    if plugin_name == mesh_metadata.PLUGIN_NAME:
        return _migrate_mesh_value(value)
    if plugin_name == custom_scalars_metadata.PLUGIN_NAME:
        return _migrate_custom_scalars_value(value)
    if plugin_name in [
        graphs_metadata.PLUGIN_NAME_RUN_METADATA,
        graphs_metadata.PLUGIN_NAME_RUN_METADATA_WITH_GRAPH,
        graphs_metadata.PLUGIN_NAME_KERAS_MODEL,
    ]:
        return _migrate_graph_sub_plugin_value(value)
    return (value,)


def _migrate_scalar_value(value):
    if value.HasField("metadata"):
        value.metadata.data_class = summary_pb2.DATA_CLASS_SCALAR
    return (value,)


def _migrate_histogram_value(value):
    if value.HasField("metadata"):
        value.metadata.data_class = summary_pb2.DATA_CLASS_TENSOR
    return (value,)


def _migrate_image_value(value):
    if value.HasField("metadata"):
        value.metadata.data_class = summary_pb2.DATA_CLASS_BLOB_SEQUENCE
    return (value,)


def _migrate_text_value(value):
    if value.HasField("metadata"):
        value.metadata.data_class = summary_pb2.DATA_CLASS_TENSOR
    return (value,)


def _migrate_audio_value(value):
    if value.HasField("metadata"):
        value.metadata.data_class = summary_pb2.DATA_CLASS_BLOB_SEQUENCE
    tensor = value.tensor
    # Project out just the first axis: actual audio clips.
    stride = 1
    while len(tensor.tensor_shape.dim) > 1:
        stride *= tensor.tensor_shape.dim.pop().size
    if stride != 1:
        tensor.string_val[:] = tensor.string_val[::stride]
    return (value,)


def _migrate_hparams_value(value):
    if value.HasField("metadata"):
        value.metadata.data_class = summary_pb2.DATA_CLASS_TENSOR
    if not value.HasField("tensor"):
        value.tensor.CopyFrom(hparams_metadata.NULL_TENSOR)
    return (value,)


def _migrate_pr_curve_value(value):
    if value.HasField("metadata"):
        value.metadata.data_class = summary_pb2.DATA_CLASS_TENSOR
    return (value,)


def _migrate_mesh_value(value):
    if value.HasField("metadata"):
        value.metadata.data_class = summary_pb2.DATA_CLASS_TENSOR
    return (value,)


def _migrate_custom_scalars_value(value):
    if value.HasField("metadata"):
        value.metadata.data_class = summary_pb2.DATA_CLASS_TENSOR
    return (value,)


def _migrate_graph_sub_plugin_value(value):
    if value.HasField("metadata"):
        value.metadata.data_class = summary_pb2.DATA_CLASS_BLOB_SEQUENCE
    shape = value.tensor.tensor_shape.dim
    if not shape:
        shape.add(size=1)
    return (value,)
