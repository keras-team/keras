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
"""Contains utilities for conversion of TF stats proto types to GViz types.

Usage:
    gviz_data_table = generate_chart_table(stats_table)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gviz_api

from tensorboard_plugin_profile.protobuf import tf_stats_pb2


def get_chart_table_args(stats_table, device_type):
  """Creates gviz DataTable object from a TensorFlow stats table.

  Args:
    stats_table: A tf_stats_pb2.TfStatsTable.
    device_type: A string representing device type of this profile.

  Returns:
    Returns a gviz_api.DataTable
  """

  # Create schema
  table_description = [
      ("rank", "number", "Rank"),
      ("host_or_device", "string", "Host/device"),
      ("type", "string", "Type"),
      ("operation", "string", "Operation"),
      ("occurrences", "number", "#Occurrences"),
      ("total_time", "number", "Total time (us)"),
      ("avg_time", "number", "Avg. time (us)"),
      ("total_self_time", "number", "Total self-time (us)"),
      ("avg_self_time", "number", "Avg. self-time (us)"),
      ("device_total_self_time_percent", "number",
       "Total self-time on Device (%)"),
      ("device_cumulative_total_self_time_percent", "number",
       "Cumulative total-self time on Device (%)"),
      ("host_total_self_time_percent", "number", "Total self-time on Host (%)"),
      ("host_cumulative_total_self_time_percent", "number",
       "Cumulative total-self time on Host (%)"),
  ]

  # Add GPU TensorCore utilization only when hardware device type is GPU, for
  # other devices, the GPU TensorCore utilization is always 0 so there is no
  # need to display this column.
  if device_type == "GPU":
    table_description.append(
        ("gpu_tensorcore_utilization", "number", "GPU TensorCore utilization"))

  data = []
  for record in stats_table.tf_stats_record:
    row = [
        record.rank,
        record.host_or_device,
        record.op_type,
        record.op_name,
        record.occurrences,
        record.total_time_in_us,
        record.avg_time_in_us,
        record.total_self_time_in_us,
        record.avg_self_time_in_us,
        record.device_total_self_time_as_fraction,
        record.device_cumulative_total_self_time_as_fraction,
        record.host_total_self_time_as_fraction,
        record.host_cumulative_total_self_time_as_fraction,
    ]

    if device_type == "GPU":
      row.append(record.gpu_tensorcore_utilization)

    data.append(row)

  return (table_description, data, [])


def generate_chart_table(stats_table, device_type):
  (table_description, data,
   custom_properties) = get_chart_table_args(stats_table, device_type)
  return gviz_api.DataTable(table_description, data, custom_properties)


def generate_all_chart_tables(tf_stats_db):
  """Converts a TfStatsDatabase proto to gviz DataTable."""
  return [
      generate_chart_table(tf_stats_db.with_idle, tf_stats_db.device_type),
      generate_chart_table(tf_stats_db.without_idle, tf_stats_db.device_type),
  ]


def to_json(raw_data):
  """Converts a serialized TfStatsDatabase string to gviz json string."""
  tf_stats_db = tf_stats_pb2.TfStatsDatabase()
  tf_stats_db.ParseFromString(raw_data)
  all_chart_tables = generate_all_chart_tables(tf_stats_db)
  json_join = ",".join(x.ToJSon() for x in all_chart_tables)
  return "[" + json_join + "]"


def to_csv(raw_data):
  """Converts a serialized TfStatsDatabase string to CSV."""
  tf_stats_db = tf_stats_pb2.TfStatsDatabase()
  tf_stats_db.ParseFromString(raw_data)
  return generate_chart_table(tf_stats_db.with_idle,
                              tf_stats_db.device_type).ToCsv()
