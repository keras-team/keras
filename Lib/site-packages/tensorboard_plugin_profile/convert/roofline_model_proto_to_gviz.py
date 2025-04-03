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
"""For conversion of RooflineModel protos to GViz DataTables.

Usage:
    gviz_data_tables = generate_roofline_model_table(roofline_model_db)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gviz_api

from tensorboard_plugin_profile.protobuf import roofline_model_pb2


def get_step_string(record_type, step_num):
  match record_type:
    case roofline_model_pb2.RecordType.INVALID_RECORD_TYPE:
      return "Invalid"
    case roofline_model_pb2.RecordType.ALL:
      return "Total"
    case roofline_model_pb2.RecordType.ALL_HW:
      return "Total (HW)"
    case roofline_model_pb2.RecordType.AVERAGE_STEP:
      return "Average"
    case roofline_model_pb2.RecordType.PER_STEP:
      return step_num


def gibi_to_giga(gibibytes):
  return gibibytes * ((1 << 30) / 1.0e9)


def ridge_point(peak_gigaflops_per_second, peak_gibibytes_per_second):
  if peak_gibibytes_per_second == 0:
    return 0.0
  return peak_gigaflops_per_second / gibi_to_giga(peak_gibibytes_per_second)


def get_roofline_model_table_args_for_gpu(roofline_model_db):
  """Creates roofline model table args from a roofline model proto for gpu.

  Args:
    roofline_model_db: A RooflineModelDatabase proto.

  Returns:
    Returns table description(columns), data(rows) and custom properties.
  """
  table_description = [
      ("step", "string", "Step"),
      ("rank", "number", "Rank"),
      ("category", "string", "Category"),
      ("operation", "string", "Operation"),
      ("occurrences", "number", "# Occurrences"),
      ("total_time", "number", "Total Time (us)"),
      ("avg_time", "number", "Avg. time (us)"),
      ("total_self_time", "number", "Total self time (us)"),
      ("avg_self_time", "number", "Avg. self time (us)"),
      ("total_self_time_percent", "number", "Total self time (%)"),
      (
          "cumulative_total_self_time_percent",
          "number",
          "Cumulative total self time (%)",
      ),
      ("measured_flop_rate", "number", "Normalized FLOP Rate (GFLOP/s)"),
      ("model_flop_rate", "number", "Model FLOP Rate (GFLOP/s)"),
      ("measured_memory_bw", "number", "Memory BW (GiB/s)"),
      ("hbm_bw", "number", "HBM BW (GiB/s)"),
      # For nvidia gpu, currently no vmem_read_bw field, and
      # vmem_write_bw is used for SHM/L1.
      ("vmem_write_bw", "number", "SHM/L1 BW (GiB/s)"),
      ("operational_intensity", "number", "Operational Intensity (FLOP/Byte)"),
      (
          "hbm_operational_intensity",
          "number",
          "HBM Operational Intensity (FLOP/Byte)",
      ),
      # for nvidia gpu, currently novmem_read_operational_intensity field, and
      # vmem_write_operational_intensity used for SHM/L1.
      (
          "vmem_write_operational_intensity",
          "number",
          "SHM/L1 Operational Intensity (FLOP/Byte)",
      ),
      (
          "bottleneck_operational_intensity",
          "number",
          "Bottleneck Operational Intensity (FLOP/Byte)",
      ),
      ("bound_by", "string", "Bound by"),
      ("total_time_per_core", "number", "Total Time per core (us)"),
      ("total_time_in_percentage", "number", "Total Time (%)"),
      ("optimal_flop_rate", "number", "Optimal FLOP Rate (GFLOP/s)"),
      ("roofline_efficiency", "number", "Roofline efficiency (%)"),
      ("compute_efficiency", "number", "FLOP Rate / Peak (%)"),
      (
          "max_mem_bw_utilization",
          "number",
          "Max memory (cmem or hbm) bandwidth utilization (%)",
      ),
      ("include_infeed_outfeed", "boolean", "Include Infeed/Outfeed"),
      ("hlo_module_id", "string", "Program ID"),
  ]

  data = []
  for record in roofline_model_db.roofline_model_record:
    row = [
        get_step_string(record.record_type, record.step_num),
        record.rank,
        record.hlo_category,
        record.hlo_name,
        record.occurrences,
        record.total_time_in_us,
        record.avg_time_in_us,
        record.total_self_time_in_us,
        record.avg_self_time_in_us,
        record.total_self_time_as_fraction,
        record.cumulative_total_self_time_as_fraction,
        record.measured_flop_rate,
        record.model_flop_rate,
        record.measured_memory_bw,
        record.hbm_bw,
        record.vmem_write_bw,
        record.operational_intensity,
        record.hbm_operational_intensity,
        record.vmem_write_operational_intensity,
        record.bottleneck_operational_intensity,
        record.bound_by,
        record.total_time_per_core_in_us,
        record.total_time_in_percentage,
        record.optimal_flop_rate,
        record.roofline_efficiency,
        record.flop_rate_relative_to_hw_limit,
        record.memory_bw_relative_to_hw_limit,
        record.include_infeed_outfeed,
        record.hlo_module_id,
    ]
    data.append(row)
  custom_properties = {
      "device_type": roofline_model_db.device_type,
      "has_cmem": roofline_model_db.has_cmem,
      "has_merged_vmem": roofline_model_db.has_merged_vmem,
      "peak_flop_rate": roofline_model_db.peak_flop_rate,
      "peak_hbm_bw": roofline_model_db.peak_hbm_bw,
      "peak_shml1_write_bw": roofline_model_db.peak_vmem_write_bw,
      "hbm_ridge_point": ridge_point(
          roofline_model_db.peak_flop_rate, roofline_model_db.peak_hbm_bw
      ),
      "shml1_write_ridge_point": ridge_point(
          roofline_model_db.peak_flop_rate, roofline_model_db.peak_vmem_write_bw
      ),
  }
  return (table_description, data, custom_properties)


def get_roofline_model_table_args(roofline_model_db):
  """Creates roofline model table args from a roofline model proto.

  Args:
    roofline_model_db: A RooflineModelDatabase proto.

  Returns:
    Returns table description(columns), data(rows) and custom properties.
  """

  table_description = [
      ("step", "string", "Step"),
      ("rank", "number", "Rank"),
      ("category", "string", "Category"),
      ("operation", "string", "Operation"),
      ("occurrences", "number", "# Occurrences"),
      ("total_time", "number", "Total Time (us)"),
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
      ("measured_flop_rate", "number", "Normalized FLOP Rate (GFLOP/s)"),
      ("model_flop_rate", "number", "Model FLOP Rate (GFLOP/s)"),
      ("measured_memory_bw", "number", "Memory BW (GiB/s)"),
      ("hbm_bw", "number", "HBM BW (GiB/s)"),
      ("cmem_read_bw", "number", "CMEM Read BW (GiB/s)"),
      ("cmem_write_bw", "number", "CMEM Write BW (GiB/s)"),
      ("vmem_read_bw", "number", "VMEM Read BW (GiB/s)"),
      ("vmem_write_bw", "number", "VMEM Write BW (GiB/s)"),
      ("operational_intensity", "number", "Operational Intensity (FLOP/Byte)"),
      (
          "hbm_operational_intensity",
          "number",
          "HBM Operational Intensity (FLOP/Byte)",
      ),
      (
          "cmem_read_operational_intensity",
          "number",
          "CMEM Read Operational Intensity (FLOP/Byte)",
      ),
      (
          "cmem_write_operational_intensity",
          "number",
          "CMEM Write Operational Intensity (FLOP/Byte)",
      ),
      (
          "vmem_read_operational_intensity",
          "number",
          "VMEM Read Operational Intensity (FLOP/Byte)",
      ),
      (
          "vmem_write_operational_intensity",
          "number",
          "VMEM Write Operational Intensity (FLOP/Byte)",
      ),
      (
          "bottleneck_operational_intensity",
          "number",
          "Bottleneck Operational Intensity (FLOP/Byte)",
      ),
      ("bound_by", "string", "Bound by"),
      ("total_time_per_core", "number", "Total Time per core (us)"),
      ("total_time_in_percentage", "number", "Total Time (%)"),
      ("optimal_flop_rate", "number", "Optimal FLOP Rate (GFLOP/s)"),
      ("roofline_efficiency", "number", "Roofline efficiency (%)"),
      ("compute_efficiency", "number", "FLOP Rate / Peak (%)"),
      (
          "max_mem_bw_utilization",
          "number",
          "Max memory (cmem or hbm) bandwidth utilization (%)",
      ),
      ("include_infeed_outfeed", "boolean", "Include Infeed/Outfeed"),
      ("hlo_module_id", "string", "Program ID"),
  ]

  data = []
  for record in roofline_model_db.roofline_model_record:
    row = [
        get_step_string(record.record_type, record.step_num),
        record.rank,
        record.hlo_category,
        record.hlo_name,
        record.occurrences,
        record.total_time_in_us,
        record.avg_time_in_us,
        record.total_self_time_in_us,
        record.avg_self_time_in_us,
        record.total_self_time_as_fraction,
        record.cumulative_total_self_time_as_fraction,
        record.dma_stall_fraction,
        record.measured_flop_rate,
        record.model_flop_rate,
        record.measured_memory_bw,
        record.hbm_bw,
        record.cmem_read_bw,
        record.cmem_write_bw,
        record.vmem_read_bw,
        record.vmem_write_bw,
        record.operational_intensity,
        record.hbm_operational_intensity,
        record.cmem_read_operational_intensity,
        record.cmem_write_operational_intensity,
        record.vmem_read_operational_intensity,
        record.vmem_write_operational_intensity,
        record.bottleneck_operational_intensity,
        record.bound_by,
        record.total_time_per_core_in_us,
        record.total_time_in_percentage,
        record.optimal_flop_rate,
        record.roofline_efficiency,
        record.flop_rate_relative_to_hw_limit,
        record.memory_bw_relative_to_hw_limit,
        record.include_infeed_outfeed,
        record.hlo_module_id,
    ]
    data.append(row)
  custom_properties = {
      "device_type": roofline_model_db.device_type,
      "megacore": str(int(roofline_model_db.megacore)),
      "has_cmem": str(int(roofline_model_db.has_cmem)),
      "has_merged_vmem": str(int(roofline_model_db.has_merged_vmem)),
      "peak_flop_rate": str(roofline_model_db.peak_flop_rate),
      "peak_hbm_bw": str(roofline_model_db.peak_hbm_bw),
      "peak_cmem_read_bw": str(roofline_model_db.peak_cmem_read_bw),
      "peak_cmem_write_bw": str(roofline_model_db.peak_cmem_write_bw),
      "peak_vmem_read_bw": str(roofline_model_db.peak_vmem_read_bw),
      "peak_vmem_write_bw": str(roofline_model_db.peak_vmem_write_bw),
      "hbm_ridge_point": str(
          ridge_point(
              roofline_model_db.peak_flop_rate, roofline_model_db.peak_hbm_bw
          )
      ),
      "cmem_read_ridge_point": str(
          ridge_point(
              roofline_model_db.peak_flop_rate,
              roofline_model_db.peak_cmem_read_bw,
          )
      ),
      "cmem_write_ridge_point": str(
          ridge_point(
              roofline_model_db.peak_flop_rate,
              roofline_model_db.peak_cmem_write_bw,
          )
      ),
      "vmem_read_ridge_point": str(
          ridge_point(
              roofline_model_db.peak_flop_rate,
              roofline_model_db.peak_vmem_read_bw,
          )
      ),
      "vmem_write_ridge_point": str(
          ridge_point(
              roofline_model_db.peak_flop_rate,
              roofline_model_db.peak_vmem_write_bw,
          )
      ),
  }

  return (table_description, data, custom_properties)


def generate_roofline_model_table(roofline_model_db):
  """Creates roofline model table from a list of roofline model protos.

  Args:
    roofline_model_db: a RooflineModelDatabase proto.

  included and one without..

  Returns:
    Returns a gviz_api.DataTable
  """
  device_type_str = roofline_model_db.device_type
  if "GPU" not in device_type_str:
    table_description, data, custom_properties = get_roofline_model_table_args(
        roofline_model_db
    )
  else:
    table_description, data, custom_properties = (
        get_roofline_model_table_args_for_gpu(roofline_model_db)
    )

  return gviz_api.DataTable(table_description, data, custom_properties)


def get_diagnostics_table_args(roofline_model_db):
  """Creates diagnostics table from a roofline model proto."""
  table_description = [
      ("severity", "string", "Severity"),
      ("message", "string", "Message"),
  ]
  data = []
  for info in roofline_model_db.diagnostics.info:
    data.append(["INFO", info])
  for warning in roofline_model_db.diagnostics.warnings:
    data.append(["WARNING", warning])
  for error in roofline_model_db.diagnostics.errors:
    data.append(["ERROR", error])
  return (table_description, data, {})


def generate_diagnostics_table(roofline_model_db):
  table_description, data, custom_properties = get_diagnostics_table_args(
      roofline_model_db
  )
  return gviz_api.DataTable(table_description, data, custom_properties)


def to_json(raw_data):
  """Converts a serialized HloStatsDb string to json."""
  roofline_model_db = roofline_model_pb2.RooflineModelDatabase()
  roofline_model_db.ParseFromString(raw_data)
  roofline_model_table = generate_roofline_model_table(
      roofline_model_db
  ).ToJSon()
  diagnostics_table = generate_diagnostics_table(roofline_model_db).ToJSon()
  return "[" + roofline_model_table + "," + diagnostics_table + "]"
