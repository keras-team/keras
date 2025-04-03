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
"""Internal information about the text plugin."""


from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.text import plugin_data_pb2

PLUGIN_NAME = "text"

# The most recent value for the `version` field of the
# `TextPluginData` proto.
PROTO_VERSION = 0


def create_summary_metadata(display_name, description):
    """Create a `summary_pb2.SummaryMetadata` proto for text plugin data.

    Returns:
      A `summary_pb2.SummaryMetadata` protobuf object.
    """
    content = plugin_data_pb2.TextPluginData(version=PROTO_VERSION)
    metadata = summary_pb2.SummaryMetadata(
        display_name=display_name,
        summary_description=description,
        plugin_data=summary_pb2.SummaryMetadata.PluginData(
            plugin_name=PLUGIN_NAME, content=content.SerializeToString()
        ),
    )
    return metadata


def parse_plugin_metadata(content):
    """Parse summary metadata to a Python object.

    Arguments:
      content: The `content` field of a `SummaryMetadata` proto corresponding to
        the text plugin.
    Returns:
      A `TextPluginData` protobuf object.
    """
    if not isinstance(content, bytes):
        raise TypeError("Content type must be bytes")
    if content == b"{}":
        # Old-style JSON format. Equivalent to an all-default proto.
        return plugin_data_pb2.TextPluginData()
    result = plugin_data_pb2.TextPluginData.FromString(content)
    if result.version == 0:
        return result
    # No other versions known at this time, so no migrations to do.
    return result
