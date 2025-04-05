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
"""For conversion of TF Input Pipeline Analyzer protos to GViz DataTables.

Usage:
    gviz_data_tables = generate_all_chart_tables(ipa)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import gviz_api

from tensorboard_plugin_profile.convert import diagnostics as diag
from tensorboard_plugin_profile.protobuf import input_pipeline_pb2
from tensorboard_plugin_profile.protobuf import tpu_input_pipeline_pb2


def compute_time_ms(details: tpu_input_pipeline_pb2.PerTpuStepDetails):
  return details.tc_compute_time_ms + details.scv0_compute_time_ms


def infeed_time_ms(details: tpu_input_pipeline_pb2.PerTpuStepDetails):
  return details.tc_infeed_time_ms + details.scv0_infeed_time_ms


def all_reduce_time_ms(details: tpu_input_pipeline_pb2.PerTpuStepDetails):
  return details.all_reduce_compute_time_ms + details.all_reduce_sync_time_ms


def non_idle_time_ms(details: tpu_input_pipeline_pb2.PerTpuStepDetails):
  return (
      compute_time_ms(details)
      + infeed_time_ms(details)
      + all_reduce_time_ms(details)
      + details.tc_outfeed_time_ms
  )


# Time spent by a training step on TPU.
def step_time_ms(details: tpu_input_pipeline_pb2.PerTpuStepDetails):
  return non_idle_time_ms(details) + details.tc_idle_time_ms


def get_step_breakdown_table_args_for_tpu(ipa):
  """Creates a step breakdown from an Input Pipeline Analyzer proto for TPU.

  Args:
    ipa: An input_pipeline_pb2.InputPipelineAnalysisResult.

  Returns:
    Returns a gviz_api.DataTable
  """

  breakdown = tpu_input_pipeline_pb2.TpuStepTimeBreakdown()
  has_breakdown = isinstance(
      ipa.step_time_breakdown, tpu_input_pipeline_pb2.TpuStepTimeBreakdown
  )
  if not ipa.step_time_breakdown.Unpack(breakdown):
    warnings.warn("Could not unpack to TpuStepBreakdown")
  # scv0 and scv activities are mutually exclusive.
  # This check is an estimation and needs to be updated with more accurate flag.
  has_legacy_sc_summary = not breakdown.HasField("sparse_core_step_summary")

  table_description = [
      ("stepnum", "number", "stepnum"),
      ("tcComputeTimeMs", "number", "TensorCore compute (in ms)"),
  ]
  if has_legacy_sc_summary:
    table_description += [
        ("scv0ComputeTimeMs", "number", "SparseCoreV0 compute (in ms)"),
        ("scv0InfeedTimeMs", "number", "SparseCoreV0 input (in ms)"),
    ]
  table_description += [
      ("tcInfeedTimeMs", "number", "TensorCore input (in ms)"),
      ("tcOutfeedTimeMs", "number", "TensorCore output (in ms)"),
      ("tcIdleTimeMs", "number", "TensorCore idle (in ms)"),
      ("hostTransferTimeMs", "number", "Host transfer (in ms)"),
      ("tooltip", "string", "tooltip", {"role": "tooltip"}),
      (
          "infeedPercentAverage",
          "number",
          "number% step time waiting for input data",
      ),
      ("infeedPercentMin", "number", "Infeed percent min"),
      ("infeedPercentMax", "number", "Infeed percent max"),
  ]

  data = []
  for step_details in ipa.step_details:
    details = tpu_input_pipeline_pb2.PerTpuStepDetails()
    step_details.Unpack(details)
    tooltip = (
        "step {}: \nTime waiting for input data = {:.3f} ms, Step time ="
        " {:.3f} ms".format(
            details.step_number,
            infeed_time_ms(details),
            step_time_ms(details) - all_reduce_time_ms(details),
        )
    )
    row = [details.step_number, details.tc_compute_time_ms]
    if has_legacy_sc_summary:
      row += [
          details.scv0_compute_time_ms,
          details.scv0_infeed_time_ms,
      ]
    row += [
        details.tc_infeed_time_ms,
        details.tc_outfeed_time_ms,
        details.tc_idle_time_ms,
        details.host_transfer_ms,
        tooltip,
        details.infeed_percent_average,
        details.infeed_percent_minimum,
        details.infeed_percent_maximum,
    ]
    data.append(row)

  step_time_summary = ipa.step_time_summary
  input_percent_summary = ipa.input_percent_summary
  custom_properties = {
      "steptime_ms_average": "{:.1f}".format(step_time_summary.average),
      "steptime_ms_standard_deviation": "{:.1f}".format(
          step_time_summary.standard_deviation
      ),
      "steptime_ms_minimum": "{:.1f}".format(step_time_summary.minimum),
      "steptime_ms_maximum": "{:.1f}".format(step_time_summary.maximum),
      "infeed_percent_average": "{:.1f}".format(input_percent_summary.average),
      "infeed_percent_standard_deviation": "{:.1f}".format(
          input_percent_summary.standard_deviation
      ),
      "infeed_percent_minimum": "{:.1f}".format(input_percent_summary.minimum),
      "infeed_percent_maximum": "{:.1f}".format(input_percent_summary.maximum),
  }

  # Add TPU step time breakdown to table properties
  if has_breakdown:
    scv0_compute_summary = breakdown.scv0_compute_ms_summary
    scv0_infeed_summary = breakdown.scv0_infeed_ms_summary
    tc_compute_summary = breakdown.tc_compute_ms_summary
    tc_infeed_summary = breakdown.tc_infeed_ms_summary
    tc_outfeed_summary = breakdown.tc_outfeed_ms_summary
    tc_idle_summary = breakdown.tc_idle_ms_summary
    sc_compute_summary = (
        breakdown.sparse_core_step_summary.sc_compute_ms_summary
    )
    sc_infeed_summary = breakdown.sparse_core_step_summary.sc_infeed_ms_summary
    sc_outfeed_summary = (
        breakdown.sparse_core_step_summary.sc_outfeed_ms_summary
    )
    sc_idle_summary = breakdown.sparse_core_step_summary.sc_idle_ms_summary
    sc_step_summary = breakdown.sparse_core_step_summary.sc_step_time_ms_summary
    host_transfer_summary = breakdown.host_transfer_ms_summary

    if has_legacy_sc_summary:
      custom_properties.update({
          "scv0_compute_ms_average": "{:.2f}".format(
              scv0_compute_summary.average
          ),
          "scv0_infeed_ms_average": "{:.2f}".format(
              scv0_infeed_summary.average
          ),
      })
    custom_properties.update({
        "tc_compute_ms_average": "{:.2f}".format(tc_compute_summary.average),
        "tc_infeed_ms_average": "{:.2f}".format(tc_infeed_summary.average),
        "tc_outfeed_ms_average": "{:.2f}".format(tc_outfeed_summary.average),
        "tc_idle_ms_average": "{:.2f}".format(tc_idle_summary.average),
        "host_transfer_ms_average": "{:.2f}".format(
            host_transfer_summary.average
        ),
    })
    if sc_step_summary.minimum > 0:
      custom_properties.update({
          "sc_compute_ms_average": "{:.2f}".format(sc_compute_summary.average),
          "sc_infeed_ms_average": "{:.2f}".format(sc_infeed_summary.average),
          "sc_outfeed_ms_average": "{:.2f}".format(sc_outfeed_summary.average),
          "sc_idle_ms_average": "{:.2f}".format(sc_idle_summary.average),
          "sc_step_time_ms_average": "{:.1f}".format(sc_step_summary.average),
      })

  # Add TPU bottleneck summary analysis to table properties
  bottleneck = tpu_input_pipeline_pb2.TpuBottleneckAnalysis()
  has_bottleneck = isinstance(
      ipa.recommendation.bottleneck_analysis,
      tpu_input_pipeline_pb2.TpuBottleneckAnalysis,
  )
  if not has_bottleneck:
    warnings.warn("Could not unpack to TpuBottleneckAnalysis")
  ipa.recommendation.bottleneck_analysis.Unpack(bottleneck)
  if has_bottleneck:
    custom_properties.update({
        "input_conclusion": bottleneck.input_statement,
        "output_conclusion": bottleneck.output_statement,
    })
  custom_properties.update({
      "summary_nextstep": ipa.recommendation.summary_next_step,
  })

  return (table_description, data, custom_properties)


# Get generic step breakdown table
def get_step_breakdown_table_args(ipa):
  """Creates a step breakdown from an Input Pipeline Analyzer proto.

  Args:
    ipa: An input_pipeline_pb2.InputPipelineAnalysisResult.

  Returns:
    Returns a gviz_api.DataTable
  """

  table_description = [
      ("stepname", "string", "Step Name"),
      ("deviceComputeTimeMs", "number", "Device compute"),
      ("deviceToDeviceTimeMs", "number", "Device to device"),
      ("deviceCollectivesTimeMs", "number", "Device collectives"),
      ("hostComputeTimeMs", "number", "Host compute"),
      ("kernelLaunchTimeMs", "number", "Kernel launch"),
      ("infeedTimeMs", "number", "Input"),
      ("outfeedTimeMs", "number", "Output"),
      ("compileTimeMs", "number", "Compilation"),
      ("otherTimeMs", "number", "All others"),
      ("tooltip", "string", "tooltip", {
          "role": "tooltip"
      }),
      ("stepTimeMs", "number", "Step time"),
  ]

  # Parameters for input analysis summary.
  total_step_time_ms = 0.0
  total_input_ms = 0.0
  total_output_ms = 0.0
  total_host_compute_ms = 0.0
  total_host_prepare_ms = 0.0
  total_host_compile_ms = 0.0
  total_device_to_device_ms = 0.0
  total_device_collectives_ms = 0.0
  total_unknown_ms = 0.0

  data = []
  for step_details in ipa.step_details:
    details = input_pipeline_pb2.PerGenericStepDetails()
    step_details.Unpack(details)

    tooltip = ("Step {}, duration: {:.2f} ms\n"
               "-All others: {:.2f} ms\n"
               "-Compilation: {:.2f} ms\n"
               "-Output: {:.2f} ms\n"
               "-Input: {:.2f} ms\n"
               "-Kernel launch: {:.2f} ms\n"
               "-Host compute: {:.2f} ms\n"
               "-Device collectives: {:.2f} ms\n"
               "-Device to device: {:.2f} ms\n"
               "-Device compute: {:.2f} ms").format(
                   details.step_name, details.step_time_ms,
                   details.unknown_time_ms, details.host_compile_ms,
                   details.output_ms,
                   details.host_wait_input_ms + details.host_to_device_ms,
                   details.host_prepare_ms, details.host_compute_ms,
                   details.device_collectives_ms, details.device_to_device_ms,
                   details.device_compute_ms)

    row = [
        details.step_name,
        details.device_compute_ms,
        details.device_to_device_ms,
        details.device_collectives_ms,
        details.host_compute_ms,
        details.host_prepare_ms,
        details.host_wait_input_ms + details.host_to_device_ms,
        details.output_ms,
        details.host_compile_ms,
        details.unknown_time_ms,
        tooltip,
        details.step_time_ms,
    ]
    data.append(row)

    total_step_time_ms += details.step_time_ms
    total_input_ms += details.host_wait_input_ms + details.host_to_device_ms
    total_output_ms += details.output_ms
    total_host_prepare_ms += details.host_prepare_ms
    total_device_to_device_ms += details.device_to_device_ms
    total_device_collectives_ms += details.device_collectives_ms
    total_host_compute_ms += details.host_compute_ms
    total_host_compile_ms += details.host_compile_ms
    total_unknown_ms += details.unknown_time_ms

  bottleneck_analysis = input_pipeline_pb2.BottleneckAnalysis()
  ipa.recommendation.bottleneck_analysis.Unpack(bottleneck_analysis)
  kernel_launch_classification = \
      bottleneck_analysis.kernel_launch_classification
  kernel_launch_statement = bottleneck_analysis.kernel_launch_statement
  all_other_classification = bottleneck_analysis.all_other_classification
  all_other_statement = bottleneck_analysis.all_other_statement
  device_collectives_classification = \
      bottleneck_analysis.device_collectives_classification
  device_collectives_statement = \
      bottleneck_analysis.device_collectives_statement
  input_conclusion = bottleneck_analysis.input_statement
  summary_next_step = ipa.recommendation.summary_next_step

  # Add step time summary
  steptime_ms_average = "{:.1f}".format(ipa.step_time_summary.average)
  steptime_ms_standard_deviation = "{:.1f}".format(
      ipa.step_time_summary.standard_deviation)
  steptime_ms_minimum = "{:.1f}".format(ipa.step_time_summary.minimum)
  steptime_ms_maximum = "{:.1f}".format(ipa.step_time_summary.maximum)

  # Add step time breakdown
  breakdown = input_pipeline_pb2.GenericStepTimeBreakdown()
  ipa.step_time_breakdown.Unpack(breakdown)
  device_compute_time_ms_avg = "{:.1f}".format(
      breakdown.device_compute_ms_summary.average)
  device_compute_time_ms_sdv = "{:.1f}".format(
      breakdown.device_compute_ms_summary.standard_deviation)
  device_to_device_time_ms_avg = "{:.1f}".format(
      breakdown.device_to_device_ms_summary.average)
  device_to_device_time_ms_sdv = "{:.1f}".format(
      breakdown.device_to_device_ms_summary.standard_deviation)
  device_collectives_time_ms_avg = "{:.1f}".format(
      breakdown.device_collectives_ms_summary.average)
  device_collectives_time_ms_sdv = "{:.1f}".format(
      breakdown.device_collectives_ms_summary.standard_deviation)
  infeed_time_ms_avg = "{:.1f}".format(breakdown.input_ms_summary.average)
  infeed_time_ms_sdv = "{:.1f}".format(
      breakdown.input_ms_summary.standard_deviation)
  outfeed_time_ms_avg = "{:.1f}".format(breakdown.output_ms_summary.average)
  outfeed_time_ms_sdv = "{:.1f}".format(
      breakdown.output_ms_summary.standard_deviation)
  host_compute_time_ms_avg = "{:.1f}".format(
      breakdown.host_compute_ms_summary.average)
  host_compute_time_ms_sdv = "{:.1f}".format(
      breakdown.host_compute_ms_summary.standard_deviation)
  kernel_launch_time_ms_avg = "{:.1f}".format(
      breakdown.host_prepare_ms_summary.average)
  kernel_launch_time_ms_sdv = "{:.1f}".format(
      breakdown.host_prepare_ms_summary.standard_deviation)
  compile_time_ms_avg = "{:.1f}".format(
      breakdown.host_compile_ms_summary.average)
  compile_time_ms_sdv = "{:.1f}".format(
      breakdown.host_compile_ms_summary.standard_deviation)
  other_time_ms_avg = "{:.1f}".format(breakdown.unknown_time_ms_summary.average)
  other_time_ms_sdv = "{:.1f}".format(
      breakdown.unknown_time_ms_summary.standard_deviation)

  custom_properties = {
      "hardware_type": ipa.hardware_type,
      # Step time summary
      "steptime_ms_average": steptime_ms_average,
      "steptime_ms_standard_deviation": steptime_ms_standard_deviation,
      "steptime_ms_minimum": steptime_ms_minimum,
      "steptime_ms_maximum": steptime_ms_maximum,
      # Step time breakdown
      "device_compute_time_ms_avg": device_compute_time_ms_avg,
      "device_compute_time_ms_sdv": device_compute_time_ms_sdv,
      "device_to_device_time_ms_avg": device_to_device_time_ms_avg,
      "device_to_device_time_ms_sdv": device_to_device_time_ms_sdv,
      "device_collectives_time_ms_avg": device_collectives_time_ms_avg,
      "device_collectives_time_ms_sdv": device_collectives_time_ms_sdv,
      "infeed_time_ms_avg": infeed_time_ms_avg,
      "infeed_time_ms_sdv": infeed_time_ms_sdv,
      "outfeed_time_ms_avg": outfeed_time_ms_avg,
      "outfeed_time_ms_sdv": outfeed_time_ms_sdv,
      "host_compute_time_ms_avg": host_compute_time_ms_avg,
      "host_compute_time_ms_sdv": host_compute_time_ms_sdv,
      "kernel_launch_time_ms_avg": kernel_launch_time_ms_avg,
      "kernel_launch_time_ms_sdv": kernel_launch_time_ms_sdv,
      "compile_time_ms_avg": compile_time_ms_avg,
      "compile_time_ms_sdv": compile_time_ms_sdv,
      "other_time_ms_avg": other_time_ms_avg,
      "other_time_ms_sdv": other_time_ms_sdv,
      # Input analysis summary
      "input_conclusion": input_conclusion,
      "summary_nextstep": summary_next_step,
      # Generic recommendation
      "device_collectives_bottleneck": device_collectives_classification,
      "device_collectives_statement": device_collectives_statement,
      "kernel_launch_bottleneck": kernel_launch_classification,
      "kernel_launch_statement": kernel_launch_statement,
      "all_other_bottleneck": all_other_classification,
      "all_other_statement": all_other_statement,
  }

  return (table_description, data, custom_properties)


def get_input_op_table_args(ipa):
  """Creates an input operator from an Input Pipeline Analyzer proto.

  Args:
    ipa: An input_pipeline_pb2.InputPipelineAnalysisResult.

  Returns:
    Returns a gviz_api.DataTable
  """

  table_description = [
      ("opName", "string", "Input Op"),
      ("count", "number", "Count"),
      ("timeInMs", "number", "Total Time (in ms)"),
      ("timeInPercent", "number",
       "Total Time (as % of total input-processing time)"),
      ("selfTimeInMs", "number", "Total Self Time (in ms)"),
      ("selfTimeInPercent", "number",
       "Total Self Time (as % of total input-processing time)"),
      ("category", "string", "Category"),
  ]

  data = []
  for details in ipa.input_op_details:
    row = [
        details.op_name,
        details.count,
        details.time_in_ms,
        details.time_in_percent / 100.0,
        details.self_time_in_ms,
        details.self_time_in_percent / 100.0,
        details.category,
    ]
    data.append(row)

  enqueue_us = "{:.3f}".format(ipa.input_time_breakdown.enqueue_us)
  demanded_file_read_us = "{:.3f}".format(
      ipa.input_time_breakdown.demanded_file_read_us)
  advanced_file_read_us = "{:.3f}".format(
      ipa.input_time_breakdown.advanced_file_read_us)
  preprocessing_us = "{:.3f}".format(ipa.input_time_breakdown.preprocessing_us)
  unclassified_non_enqueue_us = "{:.3f}".format(
      ipa.input_time_breakdown.unclassified_non_enqueue_us)

  custom_properties = {
      "enqueue_us": enqueue_us,
      "demanded_file_read_us": demanded_file_read_us,
      "advanced_file_read_us": advanced_file_read_us,
      "preprocessing_us": preprocessing_us,
      "unclassified_nonequeue_us": unclassified_non_enqueue_us,
  }

  return (table_description, data, custom_properties)


def get_recommendation_table_args(ipa):
  """Creates an recommendation table from an Input Pipeline Analyzer proto.

  Args:
    ipa: An input_pipeline_pb2.InputPipelineAnalysisResult.

  Returns:
    Returns a gviz_api.DataTable
  """

  table_description = [("link", "string", "link")]

  data = [[detail] for detail in ipa.recommendation.details]

  return (table_description, data, None)


def generate_max_infeed_core_table(ipa):
  """Creates a table that reports the name of the TPU core that has the maximum infeed time at each step."""
  table_description = [
      ("index", "string", "Index"),
  ]
  step_number_row = ["Step Number"]
  core_name_row = ["Core Name"]
  for idx, raw_step_details in enumerate(ipa.step_details):
    table_description.append((str(idx), "string", str(idx)))
    step_details = tpu_input_pipeline_pb2.PerTpuStepDetails()
    if not raw_step_details.Unpack(step_details):
      continue
    step_number_row.append(step_details.step_number)
    core_name_row.append(step_details.max_infeed_time_core_name)

  return gviz_api.DataTable(
      table_description, (step_number_row, core_name_row), None
  )


def generate_step_breakdown_table_for_tpu(ipa):
  table_description, data, custom_properties = (
      get_step_breakdown_table_args_for_tpu(ipa)
  )
  return gviz_api.DataTable(table_description, data, custom_properties)


def generate_step_breakdown_table(ipa):
  (table_description, data, custom_properties) = get_step_breakdown_table_args(
      ipa
  )
  return gviz_api.DataTable(table_description, data, custom_properties)


def generate_input_op_table(ipa):
  (table_description, data, custom_properties) = get_input_op_table_args(ipa)
  return gviz_api.DataTable(table_description, data, custom_properties)


def generate_recommendation_table(ipa):
  (table_description, data,
   custom_properties) = get_recommendation_table_args(ipa)
  return gviz_api.DataTable(table_description, data, custom_properties)


def generate_all_chart_tables(ipa):
  """Generates a list of gviz tables from InputPipelineAnalysisResult."""

  tables = []
  if ipa.tag:
    tables.append(generate_step_breakdown_table_for_tpu(ipa))
    tables.append(generate_max_infeed_core_table(ipa))
  else:
    tables.append(generate_step_breakdown_table(ipa))

  return tables + [
      generate_input_op_table(ipa),
      generate_recommendation_table(ipa),
      diag.generate_diagnostics_table(ipa.diagnostics),
  ]


def to_json(raw_data):
  """Converts a serialized InputPipelineAnalysisResult string to json."""
  ipa = input_pipeline_pb2.InputPipelineAnalysisResult()
  ipa.ParseFromString(raw_data)
  all_chart_tables = generate_all_chart_tables(ipa)
  json_join = ",".join(x.ToJSon() for x in all_chart_tables)
  return "[" + json_join + "]"
