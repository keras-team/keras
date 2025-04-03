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
"""Internal information about the pr_curves plugin."""


from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.pr_curve import plugin_data_pb2

PLUGIN_NAME = "pr_curves"

# Indices for obtaining various values from the tensor stored in a summary.
TRUE_POSITIVES_INDEX = 0
FALSE_POSITIVES_INDEX = 1
TRUE_NEGATIVES_INDEX = 2
FALSE_NEGATIVES_INDEX = 3
PRECISION_INDEX = 4
RECALL_INDEX = 5

# The most recent value for the `version` field of the
# `PrCurvePluginData` proto.
PROTO_VERSION = 0


def create_summary_metadata(display_name, description, num_thresholds):
    """Create a `summary_pb2.SummaryMetadata` proto for pr_curves plugin data.

    Arguments:
      display_name: The display name used in TensorBoard.
      description: The description to show in TensorBoard.
      num_thresholds: The number of thresholds to use for PR curves.

    Returns:
      A `summary_pb2.SummaryMetadata` protobuf object.
    """
    pr_curve_plugin_data = plugin_data_pb2.PrCurvePluginData(
        version=PROTO_VERSION, num_thresholds=num_thresholds
    )
    content = pr_curve_plugin_data.SerializeToString()
    return summary_pb2.SummaryMetadata(
        display_name=display_name,
        summary_description=description,
        plugin_data=summary_pb2.SummaryMetadata.PluginData(
            plugin_name=PLUGIN_NAME, content=content
        ),
    )


def parse_plugin_metadata(content):
    """Parse summary metadata to a Python object.

    Arguments:
      content: The `content` field of a `SummaryMetadata` proto
        corresponding to the pr_curves plugin.

    Returns:
      A `PrCurvesPlugin` protobuf object.
    """
    if not isinstance(content, bytes):
        raise TypeError("Content type must be bytes")
    result = plugin_data_pb2.PrCurvePluginData.FromString(content)
    if result.version == 0:
        return result
    # No other versions known at this time, so no migrations to do.
    return result
