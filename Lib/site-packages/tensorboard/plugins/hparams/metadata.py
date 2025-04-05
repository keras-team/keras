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
"""Constants used in the HParams plugin."""


from tensorboard.compat.proto import summary_pb2
from tensorboard.compat.proto import types_pb2
from tensorboard.plugins.hparams import error
from tensorboard.plugins.hparams import plugin_data_pb2
from tensorboard.util import tensor_util


PLUGIN_NAME = "hparams"
PLUGIN_DATA_VERSION = 0

# Tensor value for use in summaries that really only need to store
# metadata. A length-0 float vector is of minimal serialized length
# (6 bytes) among valid tensors. Cache this: computing it takes on the
# order of tens of microseconds.
NULL_TENSOR = tensor_util.make_tensor_proto(
    [], dtype=types_pb2.DT_FLOAT, shape=(0,)
)

EXPERIMENT_TAG = "_hparams_/experiment"
SESSION_START_INFO_TAG = "_hparams_/session_start_info"
SESSION_END_INFO_TAG = "_hparams_/session_end_info"


def create_summary_metadata(hparams_plugin_data_pb):
    """Returns a summary metadata for the HParams plugin.

    Returns a summary_pb2.SummaryMetadata holding a copy of the given
    HParamsPluginData message in its plugin_data.content field.
    Sets the version field of the hparams_plugin_data_pb copy to
    PLUGIN_DATA_VERSION.

    Args:
      hparams_plugin_data_pb: the HParamsPluginData protobuffer to use.
    """
    if not isinstance(
        hparams_plugin_data_pb, plugin_data_pb2.HParamsPluginData
    ):
        raise TypeError(
            "Needed an instance of plugin_data_pb2.HParamsPluginData."
            " Got: %s" % type(hparams_plugin_data_pb)
        )
    content = plugin_data_pb2.HParamsPluginData()
    content.CopyFrom(hparams_plugin_data_pb)
    content.version = PLUGIN_DATA_VERSION
    return summary_pb2.SummaryMetadata(
        plugin_data=summary_pb2.SummaryMetadata.PluginData(
            plugin_name=PLUGIN_NAME, content=content.SerializeToString()
        )
    )


def parse_experiment_plugin_data(content):
    """Returns the experiment from HParam's
    SummaryMetadata.plugin_data.content.

    Raises HParamsError if the content doesn't have 'experiment' set or
    this file is incompatible with the version of the metadata stored.

    Args:
      content: The SummaryMetadata.plugin_data.content to use.
    """
    return _parse_plugin_data_as(content, "experiment")


def parse_session_start_info_plugin_data(content):
    """Returns session_start_info from the plugin_data.content.

    Raises HParamsError if the content doesn't have 'session_start_info' set or
    this file is incompatible with the version of the metadata stored.

    Args:
      content: The SummaryMetadata.plugin_data.content to use.
    """
    return _parse_plugin_data_as(content, "session_start_info")


def parse_session_end_info_plugin_data(content):
    """Returns session_end_info from the plugin_data.content.

    Raises HParamsError if the content doesn't have 'session_end_info' set or
    this file is incompatible with the version of the metadata stored.

    Args:
      content: The SummaryMetadata.plugin_data.content to use.
    """
    return _parse_plugin_data_as(content, "session_end_info")


def _parse_plugin_data_as(content, data_oneof_field):
    """Returns a data oneof's field from plugin_data.content.

    Raises HParamsError if the content doesn't have 'data_oneof_field' set or
    this file is incompatible with the version of the metadata stored.

    Args:
      content: The SummaryMetadata.plugin_data.content to use.
      data_oneof_field: string. The name of the data oneof field to return.
    """
    plugin_data = plugin_data_pb2.HParamsPluginData.FromString(content)
    if plugin_data.version != PLUGIN_DATA_VERSION:
        raise error.HParamsError(
            "Only supports plugin_data version: %s; found: %s in: %s"
            % (PLUGIN_DATA_VERSION, plugin_data.version, plugin_data)
        )
    if not plugin_data.HasField(data_oneof_field):
        raise error.HParamsError(
            "Expected plugin_data.%s to be set. Got: %s"
            % (data_oneof_field, plugin_data)
        )
    return getattr(plugin_data, data_oneof_field)
