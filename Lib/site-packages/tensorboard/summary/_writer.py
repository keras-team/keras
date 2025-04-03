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
"""Implementation for tensorboard.summary.Writer and related symbols.

This provides a TensorBoard-native summary writing API that only depends
on numpy and not any particular ML framework.
"""

import time

import numpy as np

from tensorboard.plugins.scalar import metadata as scalars_metadata
from tensorboard.summary import _output


class Writer:
    """Writes summary data for visualization in TensorBoard.

    This class is not thread-safe.

    TODO(#4581): This API should be considered EXPERIMENTAL and subject to
    backwards-incompatible changes without notice.
    """

    def __init__(self, output):
        """Constructs a Writer.

        Args:
          output: `tensorboard.summary.Output` object, or a string which will be
            interpreted as shorthand for an `Output` of the appropriate type. The
            only currently supported type is `DirectoryOutput`, where the string
            value given here will be used as the directory path.
        """
        if isinstance(output, _output.Output):
            self._output = output
        elif isinstance(output, str):
            self._output = _output.DirectoryOutput(output)
        else:
            raise TypeError("Unsupported output object %r" % output)
        self._closed = False

    def _check_not_closed(self):
        if self._closed:
            raise RuntimeError("Writer is already closed")

    def flush(self):
        """Flushes any buffered data."""
        self._check_not_closed()
        self._output.flush()

    def close(self):
        """Closes the writer and prevents further use."""
        self._check_not_closed()
        self._output.close()
        self._closed = True

    def add_scalar(self, tag, data, step, *, wall_time=None, description=None):
        """Adds a scalar summary.

        Args:
          tag: string tag used to uniquely identify this time series.
          data: numeric scalar value for this data point. Accepts any value that
            can be converted to a `np.float32` scalar.
          step: integer step value for this data point. Accepts any value that
            can be converted to a `np.int64` scalar.
          wall_time: optional `float` seconds since the Unix epoch, representing
            the real-world timestamp for this data point. Defaults to None in
            which case the current time will be used.
          description: optional string description for this entire time series.
            This should be constant for a given tag; only the first value
            encountered will be used.
        """
        self._check_not_closed()
        validated_data = _validate_scalar_shape(np.float32(data), "data")
        validated_step = _validate_scalar_shape(np.int64(step), "step")
        wall_time = wall_time if wall_time is not None else time.time()
        self._output.emit_scalar(
            plugin_name=scalars_metadata.PLUGIN_NAME,
            tag=tag,
            data=validated_data,
            step=validated_step,
            wall_time=wall_time,
            description=description,
        )


def _validate_scalar_shape(ndarray, name):
    if ndarray.ndim != 0:
        raise ValueError(
            "Expected scalar value for %r but got %r" % (name, ndarray)
        )
    return ndarray
