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
"""Information on the graph plugin."""


# This name is used as the plugin prefix route and to identify this plugin
# generally, and is also the `plugin_name` for run graphs after data-compat
# transformations.
PLUGIN_NAME = "graphs"
# The Summary API is implemented in TensorFlow because it uses TensorFlow internal APIs.
# As a result, this SummaryMetadata is a bit unconventional and uses non-public
# hardcoded name as the plugin name. Please refer to link below for the summary ops.
# https://github.com/tensorflow/tensorflow/blob/11f4ecb54708865ec757ca64e4805957b05d7570/tensorflow/python/ops/summary_ops_v2.py#L757
PLUGIN_NAME_RUN_METADATA = "graph_run_metadata"
# https://github.com/tensorflow/tensorflow/blob/11f4ecb54708865ec757ca64e4805957b05d7570/tensorflow/python/ops/summary_ops_v2.py#L788
PLUGIN_NAME_RUN_METADATA_WITH_GRAPH = "graph_run_metadata_graph"
# https://github.com/tensorflow/tensorflow/blob/565952cc2f17fdfd995e25171cf07be0f6f06180/tensorflow/python/ops/summary_ops_v2.py#L825
PLUGIN_NAME_KERAS_MODEL = "graph_keras_model"
# Plugin name used for `Event.tagged_run_metadata`. This doesn't fall into one
# of the above cases because (despite the name) `PLUGIN_NAME_RUN_METADATA` is
# _required_ to have both profile and op graphs, whereas tagged run metadata
# need only have profile data.
PLUGIN_NAME_TAGGED_RUN_METADATA = "graph_tagged_run_metadata"

# In the context of the data provider interface, tag name given to a
# graph read from the `graph_def` field of an `Event` proto, which is
# not attached to a summary and thus does not have a proper tag name of
# its own. Run level graphs always represent `GraphDef`s (graphs of
# TensorFlow ops), never conceptual graphs, profile graphs, etc. This is
# the only tag name used by the `"graphs"` plugin.
RUN_GRAPH_NAME = "__run_graph__"
