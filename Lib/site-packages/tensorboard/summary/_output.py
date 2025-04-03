# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Generalized output options for writing tensor-formatted summary data."""

from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto import summary_pb2
from tensorboard.summary.writer import event_file_writer
from tensorboard.util import tensor_util

import abc


class Output(abc.ABC):
    """Interface for emitting tensor-formatted summary data.

    Implementations of this interface can be passed to Writer to customize
    how summary data is actually persisted (e.g. to disk, to memory, over
    the network, etc.).

    TODO(#4581): This API should be considered EXPERIMENTAL and subject to
    backwards-incompatible changes without notice.
    """

    @abc.abstractmethod
    def emit_scalar(
        self,
        *,
        plugin_name,
        tag,
        data,
        step,
        wall_time,
        tag_metadata=None,
        description=None,
    ):
        """Emits one scalar data point to this Output.

        Args:
          plugin_name: string name to uniquely identify the type of time series
            (historically associated with a TensorBoard plugin).
          tag: string tag used to uniquely identify this time series.
          data: `np.float32` scalar value for this data point.
          step: `np.int64` scalar step value for this data point.
          wall_time: `float` seconds since the Unix epoch, representing the
            real-world timestamp for this data point.
          tag_metadata: optional bytes containing metadata for this entire time
            series. This should be constant for a given tag; only the first
            value encountered will be used.
          description: optional string description for this entire time series.
            This should be constant for a given tag; only the first value
            encountered will be used.
        """
        pass

    @abc.abstractmethod
    def flush(self):
        """Flushes any data that has been buffered."""
        pass

    @abc.abstractmethod
    def close(self):
        """Closes the Output and also flushes any buffered data."""
        pass


class DirectoryOutput(Output):
    """Outputs summary data by writing event files to a log directory.

    TODO(#4581): This API should be considered EXPERIMENTAL and subject to
    backwards-incompatible changes without notice.
    """

    def __init__(self, path):
        """Creates a `DirectoryOutput` for the given path."""
        self._ev_writer = event_file_writer.EventFileWriter(path)

    def emit_scalar(
        self,
        *,
        plugin_name,
        tag,
        data,
        step,
        wall_time,
        tag_metadata=None,
        description=None,
    ):
        """See `Output`."""
        # TODO(#4581): cache summary metadata to emit only once.
        summary_metadata = summary_pb2.SummaryMetadata(
            plugin_data=summary_pb2.SummaryMetadata.PluginData(
                plugin_name=plugin_name, content=tag_metadata
            ),
            summary_description=description,
            data_class=summary_pb2.DataClass.DATA_CLASS_SCALAR,
        )
        tensor_proto = tensor_util.make_tensor_proto(data)
        event = event_pb2.Event(wall_time=wall_time, step=step)
        event.summary.value.add(
            tag=tag, tensor=tensor_proto, metadata=summary_metadata
        )
        self._ev_writer.add_event(event)

    def flush(self):
        """See `Output`."""
        self._ev_writer.flush()

    def close(self):
        """See `Output`."""
        # No need to call flush first since EventFileWriter already
        # will do this for us when we call close().
        self._ev_writer.close()
