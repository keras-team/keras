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
"""Information on the custom scalars plugin."""


from tensorboard.compat.proto import summary_pb2

# A special tag named used for the summary that stores the layout.
CONFIG_SUMMARY_TAG = "custom_scalars__config__"

PLUGIN_NAME = "custom_scalars"


def create_summary_metadata():
    """Create a `SummaryMetadata` proto for custom scalar plugin data.

    Returns:
      A `summary_pb2.SummaryMetadata` protobuf object.
    """
    return summary_pb2.SummaryMetadata(
        plugin_data=summary_pb2.SummaryMetadata.PluginData(
            plugin_name=PLUGIN_NAME
        )
    )
