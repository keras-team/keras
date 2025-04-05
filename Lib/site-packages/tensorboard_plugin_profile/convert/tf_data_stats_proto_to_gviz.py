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
"""For conversion of CombinedTfDataStats protos to GViz DataTables.

Usage:
    gviz_data_tables = generate_all_chart_tables(combined_tf_data_stats)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gviz_api

from tensorboard_plugin_profile.protobuf import tf_data_stats_pb2


def format_iterator_stat(iterator_metadata, iterator_stat):
  return ("<div style='padding: 1px;'><b>{}</b><br/><div style='text-align: "
          "left;'>Start Time: {} us<br/>Total Duration: {} us<br/>Self "
          "Duration: {} us<br/># Calls: {}</div></div>").format(
              iterator_metadata.name,
              int(iterator_stat.start_time_ps / 1000_000),
              int(iterator_stat.duration_ps / 1000_000),
              int(iterator_stat.self_time_ps / 1000_000),
              iterator_stat.num_calls)


def get_graph_table_args(combined_tf_data_stats):
  """Creates a gviz DataTable object from a CombinedTfDataStats proto.

  Args:
    combined_tf_data_stats: A tf_data_stats_pb2.CombinedTfDataStats.

  Returns:
    Returns a gviz_api.DataTable
  """
  table_description = [
      ("host", "string", "Host"),
      ("input_pipeline", "string", "Input Pipeline"),
      ("rank", "number", "Rank"),
      ("name", "string", "Name"),
      ("parent", "string", "Parent"),
      ("type", "number", "Type"),
  ]
  data = []
  for host in combined_tf_data_stats.tf_data_stats:
    iterator_metadata_map = combined_tf_data_stats.tf_data_stats[
        host].iterator_metadata
    input_pipelines = combined_tf_data_stats.tf_data_stats[host].input_pipelines
    for input_pipeline_id in input_pipelines:
      input_pipeline_stats = input_pipelines[input_pipeline_id]
      rank = 0
      for input_pipeline_stat in input_pipeline_stats.stats:
        for iterator_id in sorted(input_pipeline_stat.iterator_stats):
          iterator_stat = input_pipeline_stat.iterator_stats[iterator_id]
          iterator_metadata = iterator_metadata_map[iterator_id]
          blocking_type = 0
          if iterator_stat.is_blocking:
            if iterator_id == input_pipeline_stat.bottleneck_iterator_id:
              blocking_type = 2
            else:
              blocking_type = 1
          row = [
              host,
              input_pipeline_stats.metadata.name,
              rank,
              (str(iterator_id),
               format_iterator_stat(iterator_metadata, iterator_stat)),
              str(iterator_metadata.parent_id)
              if iterator_metadata.parent_id else "",
              blocking_type,
          ]
          data.append(row)
        rank += 1
  return (table_description, data, {})


def generate_graph_table(combined_tf_data_stats):
  (table_description, data,
   custom_properties) = get_graph_table_args(combined_tf_data_stats)
  return gviz_api.DataTable(table_description, data, custom_properties)


def get_summary_table_args(combined_tf_data_stats):
  """Creates a gviz DataTable object from a CombinedTfDataStats proto.

  Args:
    combined_tf_data_stats: A tf_data_stats_pb2.CombinedTfDataStats.

  Returns:
    Returns a gviz_api.DataTable
  """
  table_description = [
      ("host", "string", "Host"),
      ("input_pipeline", "string", "Input Pipeline"),
      ("min", "number", "Min (us)"),
      ("mean", "number", "Avg (us)"),
      ("max", "number", "Max (us)"),
      ("num_calls", "number", "# calls"),
      ("num_slow_calls", "number", "# slow calls"),
  ]
  data = []
  for host in combined_tf_data_stats.tf_data_stats:
    input_pipelines = combined_tf_data_stats.tf_data_stats[host].input_pipelines
    for input_pipeline_id in input_pipelines:
      input_pipeline_stats = input_pipelines[input_pipeline_id]
      row = [
          host,
          input_pipeline_stats.metadata.name,
          int(input_pipeline_stats.min_latency_ps / 1000_000),
          int(input_pipeline_stats.avg_latency_ps / 1000_000),
          int(input_pipeline_stats.max_latency_ps / 1000_000),
          len(input_pipeline_stats.stats),
          input_pipeline_stats.num_slow_calls,
      ]
      data.append(row)
  return (table_description, data, {})


def generate_summary_table(combined_tf_data_stats):
  (table_description, data,
   custom_properties) = get_summary_table_args(combined_tf_data_stats)
  return gviz_api.DataTable(table_description, data, custom_properties)


def format_bottleneck(iterator_name, iterator_long_name, iterator_latency_ps):
  return ("<u>Iterator Type</u>: <b>{}</b><br/><u>Long Name</u>: "
          "{}<br/><u>Latency</u>: {:,} us").format(
              iterator_name, iterator_long_name,
              int(iterator_latency_ps / 1000_000))


def get_bottleneck_analysis_table_args(combined_tf_data_stats):
  """Creates a gviz DataTable object from a CombinedTfDataStats proto.

  Args:
    combined_tf_data_stats: A tf_data_stats_pb2.CombinedTfDataStats.

  Returns:
    Returns a gviz_api.DataTable
  """
  table_description = [
      ("host", "string", "Host"),
      ("input_pipeline", "string", "Input Pipeline"),
      ("max_latency", "number", "Max Latency (us)"),
      ("bottleneck", "string", "Bottleneck"),
      ("suggestion", "string", "Suggestion"),
  ]
  data = []
  for bottleneck_analysis in combined_tf_data_stats.bottleneck_analysis:
    row = [
        bottleneck_analysis.host,
        bottleneck_analysis.input_pipeline,
        int(bottleneck_analysis.max_latency_ps / 1000_000),
        format_bottleneck(bottleneck_analysis.iterator_name,
                          bottleneck_analysis.iterator_long_name,
                          bottleneck_analysis.iterator_latency_ps),
        bottleneck_analysis.suggestion,
    ]
    data.append(row)
  custom_properties = {
      "is_input_bound":
          "true" if combined_tf_data_stats.is_input_bound else "false",
      "summary_message":
          combined_tf_data_stats.summary,
  }
  return (table_description, data, custom_properties)


def generate_bottleneck_analysis_table(bottleneck_analysis):
  (table_description, data,
   custom_properties) = get_bottleneck_analysis_table_args(bottleneck_analysis)
  return gviz_api.DataTable(table_description, data, custom_properties)


def generate_all_chart_tables(combined_tf_data_stats):
  """Converts a CombinedTfDataStats proto to gviz DataTables."""
  return [
      generate_graph_table(combined_tf_data_stats),
      generate_summary_table(combined_tf_data_stats),
      generate_bottleneck_analysis_table(combined_tf_data_stats)
  ]


def to_json(raw_data):
  """Converts a serialized CombinedTfDataStats string to json."""
  combined_tf_data_stats = tf_data_stats_pb2.CombinedTfDataStats()
  combined_tf_data_stats.ParseFromString(raw_data)
  all_chart_tables = generate_all_chart_tables(combined_tf_data_stats)
  json_join = ",".join(x.ToJSon() if x else "{}" for x in all_chart_tables)
  return "[" + json_join + "]"
