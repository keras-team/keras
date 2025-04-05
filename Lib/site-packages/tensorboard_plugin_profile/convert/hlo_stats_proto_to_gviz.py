# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
"""For conversion of HloStats protos to GViz DataTables.

Usage:
    gviz_data_tables = generate_hlo_stats_table(hlo_stats_db)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any

import gviz_api

from tensorboard_plugin_profile.protobuf import hlo_stats_pb2


def get_hlo_op_name_from_expression(hlo_op_expression: str) -> str:
  """Returns the hlo op name from the hlo op expression.

  Args:
    hlo_op_expression: A string of hlo op expression.

  Returns:
    Returns the hlo op name extracted from the hlo op expression.
  """
  # The parse logic based on the assumption that the hlo op text is in format of
  # '%op_name = <long name>'
  parts = hlo_op_expression.split(" = ")
  hlo_op_name = parts[0]
  if hlo_op_name and hlo_op_name[0] == "%":
    hlo_op_name = hlo_op_name[1:]
  return hlo_op_name


TableColumnDescription = tuple[str, str, str]
TableRow = list[Any]
TableProperties = dict[str, Any]


def get_hlo_stats_table_args(
    hlo_stats_db: hlo_stats_pb2.HloStatsDatabase,
) -> tuple[list[TableColumnDescription], list[TableRow], TableProperties]:
  """Creates hlo op stats table from a hlo stats proto.

  Args:
    hlo_stats_db: A HloStatsDatabase proto.

  Returns:
    Returns table description(column defubutuibs), data(rows data) and custom
    properties for preparing the HloStats gviz table.
  """

  table_description = [
      ("rank", "number", "Rank"),
      ("program_id", "string", "Program id"),
      ("category", "string", "HLO op category"),
      ("hlo_op_name", "string", "HLO op name"),
      ("hlo_op_expression", "string", "HLO op text"),
      ("tf_op_name", "string", "Framework op name"),
      ("occurrences", "number", "#Occurrences"),
      ("total_time", "number", "Total time (us)"),
      ("avg_time", "number", "Avg. time (us)"),
      ("total_self_time", "number", "Total self time (us)"),
      ("avg_self_time", "number", "Avg. self time (us)"),
      ("total_self_time_percent", "number", "Total self time (%)"),
      (
          "cumulative_total_self_time_percent",
          "number",
          "Cumulative total self time (%)",
      ),
      ("dma_stall_percent", "number", "%time stalled by DMA"),
      ("model_flop_rate", "number", "Model GFLOP/s"),
      ("normalized_flop_rate", "number", "Normalized GFLOP/s"),
      ("measured_memory_bw", "number", "Measured memory BW (GiB/s)"),
      ("hbm_bw", "number", "HBM BW (GiB/s)"),
      ("cmem_read_bw", "number", "CMEM Read BW (GiB/s)"),
      ("cmem_write_bw", "number", "CMEM Write BW (GiB/s)"),
      ("operational_intensity", "number", "Operational intensity (FLOPS/Byte)"),
      ("bound_by", "string", "Bound by"),
      ("hlo_rematerialization", "string", "Rematerialization"),
      ("outside_compilation", "string", "Outside Compilation"),
      ("autotuned", "string", "Autotuned"),
  ]

  data = []
  for record in hlo_stats_db.hlo_stats_record:
    row = [
        record.rank,
        str(record.program_id),
        record.hlo_category,
        get_hlo_op_name_from_expression(record.hlo_expression),
        record.hlo_expression,
        record.tf_op_name,
        record.occurrences,
        record.total_time_in_us,
        record.avg_time_in_us,
        record.total_self_time_in_us,
        record.avg_self_time_in_us,
        record.total_self_time_as_fraction,
        record.cumulative_total_self_time_as_fraction,
        record.dma_stall_fraction,
        record.model_flop_rate,
        record.measured_flop_rate,
        record.measured_memory_bw,
        record.hbm_bw,
        record.cmem_read_bw,
        record.cmem_write_bw,
        record.operational_intensity,
        record.bound_by,
        "Yes" if record.rematerialization else "No",
        "Yes" if record.outside_compilation else "No",
        "Yes" if record.autotuned else "No",
    ]
    data.append(row)

  return (table_description, data, {})


def generate_hlo_stats_table(
    hlo_stats_db: hlo_stats_pb2.HloStatsDatabase,
) -> gviz_api.DataTable:
  """Converts a HloStatsDb proto to a GViz DataTable.

  Args:
    hlo_stats_db: A HloStatsDatabase proto.

  Returns:
    Returns a GViz DataTable for HloStats.
  """
  (table_description, data, custom_properties) = get_hlo_stats_table_args(
      hlo_stats_db
  )
  return gviz_api.DataTable(table_description, data, custom_properties)


def to_json(raw_data: bytes) -> str:
  """Converts a serialized HloStatsDb bytes to json.

  Args:
    raw_data: Bytes of serialized HloStatsDb.

  Returns:
    Returns a json string of HloStats Table.
  """
  hlo_stats_db = hlo_stats_pb2.HloStatsDatabase()
  hlo_stats_db.ParseFromString(raw_data)
  return generate_hlo_stats_table(hlo_stats_db).ToJSon()
