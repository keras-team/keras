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

import struct
from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c


class RecordWriter:
    """Write encoded protobuf to a file with packing defined in tensorflow."""

    def __init__(self, writer):
        """Open a file to keep the tensorboard records.

        Args:
        writer: A file-like object that implements `write`, `flush` and `close`.
        """
        self._writer = writer

    # Format of a single record: (little-endian)
    # uint64    length
    # uint32    masked crc of length
    # byte      data[length]
    # uint32    masked crc of data
    def write(self, data):
        header = struct.pack("<Q", len(data))
        header_crc = struct.pack("<I", masked_crc32c(header))
        footer_crc = struct.pack("<I", masked_crc32c(data))
        self._writer.write(header + header_crc + data + footer_crc)

    def flush(self):
        self._writer.flush()

    def close(self):
        self._writer.close()

    @property
    def closed(self):
        return self._writer.closed
