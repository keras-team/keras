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
"""For conversion of TF GPU Kernel Stats protos to GViz DataTables.

Usage:
    gviz_data_tables = generate_all_chart_tables(kernel_stats_db)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gviz_api

from tensorboard_plugin_profile.protobuf import kernel_stats_pb2


def get_kernel_reports_table_args(kernel_reports):
  """Creates kernel reports from a kernel reports proto.

  Args:
    kernel_reports: A list of kernel_stats_pb2.KernelReport.

  Returns:
    Returns a gviz_api.DataTable
  """

  table_description = [
      ("rank", "number", "Rank"),
      ("kernel_name", "string", "Kernel Name"),
      ("registers_per_thread", "number", "Registers per thread"),
      ("shmem_bytes", "number", "Shared Mem bytes"),
      ("block_dim", "string", "Block dim"),
      ("grid_dim", "string", "Grid dim"),
      ("occupancy_pct", "number", "Occupancy %"),
      ("is_op_tensor_core_eligible", "boolean", "Op is TensorCore eligible"),
      ("is_kernel_using_tensor_core", "boolean", "Kernel uses TensorCore"),
      ("op_name", "string", "Op Name"),
      ("occurrences", "number", "Occurrences"),
      ("total_duration_us", "number", "Total Duration (us)"),
      ("avg_duration_us", "number", "Avg Duration (us)"),
      ("min_duration_us", "number", "Min Duration (us)"),
      ("max_duration_us", "number", "Max Duration (us)"),
  ]

  data = []
  rank = 1
  for kernel in kernel_reports:
    row = [
        rank,
        kernel.name,
        kernel.registers_per_thread,
        kernel.static_shmem_bytes + kernel.dynamic_shmem_bytes,
        ",".join(str(x) for x in kernel.block_dim[:3]),
        ",".join(str(x) for x in kernel.grid_dim[:3]),
        kernel.occupancy_pct,
        kernel.is_op_tensor_core_eligible,
        kernel.is_kernel_using_tensor_core,
        kernel.op_name,
        kernel.occurrences,
        kernel.total_duration_ns / 1000,
        (kernel.total_duration_ns / kernel.occurrences) / 1000,
        kernel.min_duration_ns / 1000,
        kernel.max_duration_ns / 1000,
    ]
    data.append(row)
    rank += 1

  return (table_description, data, [])


def generate_kernel_reports_table(kernel_reports):
  (table_description, data,
   custom_properties) = get_kernel_reports_table_args(kernel_reports)
  return gviz_api.DataTable(table_description, data, custom_properties)


def generate_all_chart_tables(kernel_stats_db):
  """Generates a list of gviz tables from KernelStatsDb."""

  return [
      generate_kernel_reports_table(kernel_stats_db.reports)
  ]


def to_json(raw_data):
  """Converts a serialized KernelStatsDb string to json."""
  kernel_stats_db = kernel_stats_pb2.KernelStatsDb()
  kernel_stats_db.ParseFromString(raw_data)
  all_chart_tables = generate_all_chart_tables(kernel_stats_db)
  json_join = ",".join(x.ToJSon() for x in all_chart_tables)
  return "[" + json_join + "]"


def to_csv(raw_data):
  """Converts a serialized KernelStatsDb string to CSV."""
  kernel_stats_db = kernel_stats_pb2.KernelStatsDb()
  kernel_stats_db.ParseFromString(raw_data)
  return generate_kernel_reports_table(kernel_stats_db.reports).ToCsv()
