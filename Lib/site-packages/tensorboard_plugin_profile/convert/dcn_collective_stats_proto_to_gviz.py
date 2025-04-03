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
"""For conversion of Dcn Collective Stats page protos to GViz DataTables.

Usage:
    gviz_data_tables = generate_all_chart_tables(dcn_slack_analysis)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import gviz_api

from tensorboard_plugin_profile.protobuf import dcn_slack_analysis_pb2

_BYTES_TO_GIGABITS_CONVERSION_FACTOR = 8e-9
_MICRO_TO_SECONDS_CONVERSION_FACTOR = 1e-6
_BANDWIDTH_CONVERSION_FACTOR = (
    _BYTES_TO_GIGABITS_CONVERSION_FACTOR / _MICRO_TO_SECONDS_CONVERSION_FACTOR
)


def convert_bytes_to_human_readable_format(value):
  """Convert number of bytes to a human readable format (e.g. 10 KB).

  Args:
      value: bytes to convert.

  Returns:
      str: Human readable representation of data.
  """
  suffix_list = ["KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]

  base = 1000
  actual_bytes = float(value)
  abs_bytes = abs(actual_bytes)

  if abs_bytes == 1:
    return "%d Byte" % actual_bytes

  if abs_bytes < base:
    return "%d Bytes" % actual_bytes

  unit = base**2
  suffix = suffix_list[0]
  index = 0
  while index < len(suffix_list):
    unit = base ** (index + 2)
    suffix = suffix_list[index]
    if abs_bytes < unit:
      break
    index += 1

  return "{0} {1}".format(base * actual_bytes / unit, suffix)


def get_dcn_collective_stats_table_args(dcn_slack_analysis):
  """Creates a gviz DataTable object from DcnSlackAnalysis proto.

  Args:
    dcn_slack_analysis: dcn_slack_analysis_pb2.DcnSlackAnalysis.

  Returns:
    Returns a gviz_api.DataTable
  """

  table_description = [
      ("dcnCollectiveName", "string", "DCN Collective Name"),
      ("recvOpName", "string", "Recv Op Name"),
      ("sendOpName", "string", "Send Op Name"),
      ("slackTime", "number", "Slack Time (ms)"),
      ("observedDuration", "number", "Observed Duration (ms)"),
      ("stallDuration", "number", "Stall Duration (ms)"),
      ("occurrences", "number", "Occurrences"),
      ("totalStallDuration", "number", "Aggregated Total Stall (ms)"),
      ("dataTransmittedSize", "string", "Data Transmitted Size"),
      ("requiredBandwidth", "number", "Required Bandwidth (Gbps)"),
  ]

  data = []
  for slack in dcn_slack_analysis.dcn_slack_summary:
    row = [
        slack.rendezvous,
        slack.recv_op_name,
        slack.send_op_name,
        slack.slack_us / 1000,
        slack.observed_duration_us / 1000,
        slack.stall_duration_us / 1000,
        slack.occurrences,
        (slack.stall_duration_us * slack.occurrences) / 1000,
        convert_bytes_to_human_readable_format(
            slack.bytes_transmitted_over_network
        ),
    ]
    if slack.slack_us == 0:
      row.append(float(sys.maxsize))
    else:
      row.append(
          (slack.bytes_transmitted_over_network / slack.slack_us)
          * _BANDWIDTH_CONVERSION_FACTOR
      )
    data.append(row)

  return (table_description, data, [])


def generate_dcn_collective_stats_table(dcn_slack_analysis):
  (table_description, data, custom_properties) = (
      get_dcn_collective_stats_table_args(dcn_slack_analysis)
  )
  return gviz_api.DataTable(table_description, data, custom_properties)


def generate_all_chart_tables(dcn_slack_analysis):
  """Converts a DcnSlackAnalysis proto to gviz DataTables."""
  return [
      generate_dcn_collective_stats_table(dcn_slack_analysis),
  ]


def to_json(raw_data):
  """Converts a serialized DcnCollectiveAnalysis string to json."""
  dcn_slack_analysis = dcn_slack_analysis_pb2.DcnSlackAnalysis()
  dcn_slack_analysis.ParseFromString(raw_data)
  all_chart_tables = generate_all_chart_tables(dcn_slack_analysis)
  json_join = ",".join(x.ToJSon() if x else "{}" for x in all_chart_tables)
  return "[" + json_join + "]"
