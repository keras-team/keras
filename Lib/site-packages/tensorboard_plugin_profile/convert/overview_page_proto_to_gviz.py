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
"""For conversion of TF Overview Page protos to GViz DataTables.

Usage:
    gviz_data_tables = generate_all_chart_tables(overview_page)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import gviz_api

from tensorboard_plugin_profile.convert import diagnostics as diag
from tensorboard_plugin_profile.convert import input_pipeline_proto_to_gviz
from tensorboard_plugin_profile.protobuf import overview_page_pb2


def get_run_environment_table_args(run_environment):
  """Creates a gviz DataTable object from a RunEnvironment proto.

  Args:
    run_environment: An op_stats_pb2.RunEnvironment.

  Returns:
    Returns a gviz_api.DataTable
  """

  table_description = [
      ("host_id", "string", "host_id"),
      ("command_line", "string", "command_line"),
      ("start_time", "string", "start_time"),
      ("bns_address", "string", "bns_address"),
  ]

  data = []
  for job in run_environment.host_dependent_job_info:
    row = [
        str(job.host_id),
        str(job.command_line),
        str(datetime.datetime.utcfromtimestamp(job.start_time)),
        str(job.bns_address),
    ]
    data.append(row)

  host_count = "unknown"
  if run_environment.host_count >= 0:
    host_count = str(run_environment.host_count)

  task_count = "unknown"
  if run_environment.task_count >= 0:
    task_count = str(run_environment.task_count)

  if run_environment.task_count > 0 and run_environment.host_count > 0:
    tasks_per_host = run_environment.task_count / run_environment.host_count
    task_count += " (num tasks per host = {})".format(tasks_per_host)

  device_core_count = "unknown"
  if run_environment.device_core_count >= 0:
    device_core_count = str(run_environment.device_core_count)

  if run_environment.replica_count > 0 and \
      run_environment.num_cores_per_replica > 0:
    device_core_count += " (Replica count = {}, num cores per replica = {})".\
        format(run_environment.replica_count(),
               run_environment.num_cores_per_replica())

  custom_properties = {
      "host_count": host_count,
      "task_count": task_count,
      "device_type": run_environment.device_type,
      "device_core_count": device_core_count,
  }

  return (table_description, data, custom_properties)


def generate_run_environment_table(run_environment):
  (table_description, data,
   custom_properties) = get_run_environment_table_args(run_environment)
  return gviz_api.DataTable(table_description, data, custom_properties)


def get_overview_page_analysis_table_args(overview_page_analysis):
  """Creates a gviz DataTable object from an OverviewPageAnalysis proto.

  Args:
    overview_page_analysis: An overview_page_pb2.OverviewPageAnalysis.

  Returns:
    Returns a gviz_api.DataTable
  """

  table_description = [
      ("selfTimePercent", "number", "Time (%)"),
      ("cumulativeTimePercent", "number", "Cumulative time (%)"),
      ("category", "string", "Category"),
      ("operation", "string", "Operation"),
      ("flopRate", "number", "GFLOPs/Sec"),
      ("tcEligibility", "boolean", "TensorCore eligibility"),
      ("tcUtilization", "boolean", "Op is using TensorCore"),
  ]

  data = []
  for op in overview_page_analysis.top_device_ops:
    row = [
        op.self_time_fraction,
        op.cumulative_time_fraction,
        str(op.category),
        str(op.name),
        op.flop_rate,
        op.is_op_tensorcore_eligible,
        op.is_op_using_tensorcore,
    ]
    data.append(row)

  host_tf_op_percent = "{:.1f}%".format(
      overview_page_analysis.host_tf_op_percent)
  device_tf_op_percent = "{:.1f}%".format(
      overview_page_analysis.device_tf_op_percent)
  host_op_time_eager_percent = "{:.1f}%".format(
      overview_page_analysis.host_op_time_eager_percent)
  device_op_time_eager_percent = "{:.1f}%".format(
      overview_page_analysis.device_op_time_eager_percent)
  device_compute_16bit_percent = "{:.1f}%".format(
      overview_page_analysis.device_compute_16bit_percent)
  device_compute_32bit_percent = "{:.1f}%".format(
      overview_page_analysis.device_compute_32bit_percent)
  remark_text = overview_page_analysis.remark_text
  remark_color = overview_page_analysis.remark_color
  mxu_utilization_percent = "{:.1f}%".format(
      overview_page_analysis.mxu_utilization_percent)

  custom_properties = {
      "host_tf_op_percent": host_tf_op_percent,
      "device_tf_op_percent": device_tf_op_percent,
      "host_op_time_eager_percent": host_op_time_eager_percent,
      "device_op_time_eager_percent": device_op_time_eager_percent,
      "device_compute_16bit_percent": device_compute_16bit_percent,
      "device_compute_32bit_percent": device_compute_32bit_percent,
      "remark_text": remark_text,
      "remark_color": remark_color,
      "mxu_utilization_percent": mxu_utilization_percent,
  }

  return (table_description, data, custom_properties)


def generate_overview_page_analysis_table(overview_page_analysis):
  (table_description, data, custom_properties) = \
      get_overview_page_analysis_table_args(overview_page_analysis)
  return gviz_api.DataTable(table_description, data, custom_properties)


def get_recommendation_table_args(overview_page_recommendation):
  """Creates a gviz DataTable object from an OverviewPageRecommendation proto.

  Args:
    overview_page_recommendation: An
      overview_page_pb2.OverviewPageRecommendation.

  Returns:
    Returns a gviz_api.DataTable
  """

  table_description = [
      ("tip_type", "string", "tip_type"),
      ("link", "string", "link"),
  ]

  data = []
  for faq_tip in overview_page_recommendation.faq_tips:
    data.append(["faq", faq_tip.link])

  for host_tip in overview_page_recommendation.host_tips:
    data.append(["host", host_tip.link])

  for device_tip in overview_page_recommendation.device_tips:
    data.append(["device", device_tip.link])

  for doc_tip in overview_page_recommendation.documentation_tips:
    data.append(["doc", doc_tip.link])

  for inference_tip in overview_page_recommendation.inference_tips:
    data.append(["inference", inference_tip.link])

  bottleneck = overview_page_recommendation.bottleneck
  statement = overview_page_recommendation.statement

  recommendation = overview_page_pb2.GenericRecommendation()
  overview_page_recommendation.recommendation.Unpack(recommendation)
  device_collectives_bottleneck = recommendation.device_collectives_bottleneck
  device_collectives_statement = recommendation.device_collectives_statement
  kernel_launch_bottleneck = recommendation.kernel_launch_bottleneck
  kernel_launch_statement = recommendation.kernel_launch_statement
  all_other_bottleneck = recommendation.all_other_bottleneck
  all_other_statement = recommendation.all_other_statement
  precision_statement = recommendation.precision_statement

  custom_properties = {
      "bottleneck": bottleneck,
      "statement": statement,
      "device_collectives_bottleneck": device_collectives_bottleneck,
      "device_collectives_statement": device_collectives_statement,
      "kernel_launch_bottleneck": kernel_launch_bottleneck,
      "kernel_launch_statement": kernel_launch_statement,
      "all_other_bottleneck": all_other_bottleneck,
      "all_other_statement": all_other_statement,
      "precision_statement": precision_statement,
  }

  return (table_description, data, custom_properties)


def add_inference_latency_row(
    label, breakdown: overview_page_pb2.OverviewLatencyBreakdown
):
  return [
      label,
      breakdown.host_latency_us / 1e3,
      breakdown.device_latency_us / 1e3,
      breakdown.communication_latency_us / 1e3,
      breakdown.total_latency_us / 1e3,
  ]


def generate_inference_latency_chart(
    inference_latency: overview_page_pb2.OverviewInferenceLatency,
):
  """Creates a gviz DataTable object from an InferenceLatency proto."""
  columns = [
      ("percentile", "string", "Percentile"),
      ("hostTimeMs", "number", "Host Time (ms)"),
      ("deviceTimeMs", "number", "Device Time (ms)"),
      ("communicationTimeMs", "number", "Host-Device Communication Time (ms)"),
      ("totalTimeMs", "number", "Total Time (ms)"),
  ]
  data = []
  if inference_latency.latency_breakdowns:
    data.append(
        add_inference_latency_row(
            "Avg", inference_latency.latency_breakdowns[0]
        )
    )
  for index, percentile in enumerate(inference_latency.percentile_numbers):
    data.append(
        add_inference_latency_row(
            "{:.1f}".format(percentile),
            inference_latency.latency_breakdowns[index],
        )
    )
  return gviz_api.DataTable(columns, data)


def generate_recommendation_table(overview_page_recommendation):
  (table_description, data, custom_properties) = get_recommendation_table_args(
      overview_page_recommendation
  )
  return gviz_api.DataTable(table_description, data, custom_properties)


def generate_all_chart_tables(overview_page):
  """Converts a OverviewPage proto to gviz DataTables."""
  return [
      generate_overview_page_analysis_table(overview_page.analysis),
      input_pipeline_proto_to_gviz.generate_step_breakdown_table(
          overview_page.input_analysis),
      generate_run_environment_table(overview_page.run_environment),
      generate_recommendation_table(overview_page.recommendation),
      generate_inference_latency_chart(overview_page.inference_latency),
      "",
      diag.generate_diagnostics_table(overview_page.diagnostics),
  ]


def to_json(raw_data):
  """Converts a serialized OverviewPage string to json."""
  overview_page = overview_page_pb2.OverviewPage()
  overview_page.ParseFromString(raw_data)
  all_chart_tables = generate_all_chart_tables(overview_page)
  json_join = ",".join(x.ToJSon() if x else "{}" for x in all_chart_tables)
  return "[" + json_join + "]"
