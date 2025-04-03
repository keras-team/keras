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
"""For conversion of InferenceStats proto to gviz tables.

Usage:
    gviz_data_tables = generate_all_chart_tables(inference_stats)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gviz_api

from tensorboard_plugin_profile.protobuf import inference_stats_pb2


def pico_to_milli(ps: float) -> float:
  """Converts picoseconds to milliseconds."""
  return ps / 1e9


def _add_request_details(
    request_detail: inference_stats_pb2.RequestDetail,
    percentile: str,
    request_id: str,
    has_batching: bool,
    is_tpu: bool,
    throughput: str,
):
  """Generates the request details row."""
  row = [
      percentile,
      request_id,
      pico_to_milli(request_detail.end_time_ps - request_detail.start_time_ps),
  ]
  if has_batching:
    row.append(request_detail.batching_request_size)
    row.append(pico_to_milli(request_detail.batching_request_delay_ps))
    row.append(throughput)
  if is_tpu:
    row.append(pico_to_milli(request_detail.host_preprocessing_ps))
    row.append(pico_to_milli(request_detail.host_runtime_ps))
    row.append(pico_to_milli(request_detail.write_to_device_time_ps))
    row.append(pico_to_milli(request_detail.read_from_device_time_ps))
    row.append(pico_to_milli(request_detail.device_time_ps))
    row.append(pico_to_milli(request_detail.host_postprocessing_ps))
    row.append(pico_to_milli(request_detail.idle_time_ps))
  return row


def _create_request_table(
    per_model_stats: inference_stats_pb2.PerModelInferenceStats,
    sampled_per_model_stats: inference_stats_pb2.SampledPerModelInferenceStatsProto,
    has_batching: bool,
    is_tpu: bool,
):
  """Generates the request table."""
  columns = [
      ("percentile", "string", "Percentile"),
      ("request_id", "string", "Request ID"),
      ("latency_ms", "number", "Latency (ms)"),
  ]
  if has_batching:
    columns.append(("batching_request_size", "number", "Batching Request Size"))
    columns.append(
        ("host_batch_information", "number", "Host Batch Information")
    )
    columns.append(("throughput", "string", "Throughput"))
  if is_tpu:
    columns.append(("host_preprocessing", "number", "Host Preprocessing"))
    columns.append(("host_runtime", "number", "Host Runtime"))
    columns.append(("data_transfer_h2d", "number", "Data transfer H2D"))
    columns.append(("data_transfer_d2h", "number", "Data transfer D2H"))
    columns.append(("device_compute", "number", "Device compute"))
    columns.append(("host_postprocess", "number", "Host Postprocessing"))
    columns.append(("idle_time", "number", "Idle Time"))
  data = []
  for request_detail in sampled_per_model_stats.sampled_requests:
    data.append(
        _add_request_details(
            request_detail,
            "{:.3f}".format(request_detail.percentile),
            "N/A",
            has_batching,
            is_tpu,
            "N/A",
        )
    )
  for request_detail in per_model_stats.per_batch_size_aggregated_result:
    data.append(
        _add_request_details(
            request_detail.aggregated_request_result,
            "Batch Size {}".format(request_detail.batch_size),
            "N/A",
            has_batching,
            is_tpu,
            "{:.1f}".format(request_detail.batch_throughput),
        )
    )
  data.append(
      _add_request_details(
          per_model_stats.aggregated_request_detail,
          "Aggregated",
          "N/A",
          has_batching,
          is_tpu,
          "{:.1f}".format(per_model_stats.request_throughput),
      )
  )
  custom_properties = {
      "throughput": "{:.1f}".format(per_model_stats.request_throughput),
      "averageLatencyMs": "{:.3f}".format(
          per_model_stats.request_average_latency_us / 1e3
      ),
  }
  return gviz_api.DataTable(columns, data, custom_properties)


def _generate_batch_details(
    batch_detail: inference_stats_pb2.BatchDetail,
    percentile: str,
    batch_id: str,
    throughput: str,
):
  """Generates the batch details row."""
  return [
      percentile,
      batch_id,
      pico_to_milli(batch_detail.end_time_ps - batch_detail.start_time_ps),
      batch_detail.padding_amount,
      batch_detail.batch_size_after_padding,
      (batch_detail.batch_size_after_padding - batch_detail.padding_amount)
      / batch_detail.batch_size_after_padding,
      pico_to_milli(batch_detail.batch_delay_ps),
      throughput,
  ]


def _generate_batch_table(
    per_model_stats: inference_stats_pb2.PerModelInferenceStats,
    sampled_per_model_stats: inference_stats_pb2.SampledPerModelInferenceStatsProto,
    model_id_database: inference_stats_pb2.ModelIdDatabase,
    model_id: str,
):
  """Generates the batch table."""
  columns = [
      ("percentile", "string", "Percentile"),
      ("batch_id", "string", "Batch ID"),
      ("latency", "number", "Latency (ms)"),
      ("padding_amount", "number", "Padding Amount"),
      ("batch_size_after_padding", "number", "Batch Size After Padding"),
      ("batching_efficiency", "number", "Batch Efficiency"),
      ("batch_delay_ms", "number", "Batch Delay (ms)"),
      ("throughput", "string", "Throughput"),
  ]
  data = []
  properties = {}
  properties["throughput"] = "{:.1f}".format(per_model_stats.batch_throughput)
  properties["averageLatencyMs"] = "{:.3f}".format(
      per_model_stats.batch_average_latency_us / 1e3
  )

  if model_id in model_id_database.id_to_batching_params:
    params = model_id_database.id_to_batching_params[model_id]
    properties["hasBatchingParam"] = "true"
    properties["batchingParamNumBatchThreads"] = str(params.num_batch_threads)
    properties["batchingParamMaxBatchSize"] = str(params.max_batch_size)
    properties["batchingParamBatchTimeoutMicros"] = str(
        params.batch_timeout_micros
    )
    properties["batchingParamMaxEnqueuedBatches"] = str(
        params.max_enqueued_batches
    )
    properties["batchingParamAllowedBatchSizes"] = str(
        params.allowed_batch_sizes
    )
  else:
    properties["hasBatchingParam"] = "false"
  for batch_detail in sampled_per_model_stats.sampled_batches:
    data.append(
        _generate_batch_details(
            batch_detail,
            "{:.3f}".format(batch_detail.percentile),
            "N/A",
            "N/A",
        )
    )
  for batch_detail in per_model_stats.per_batch_size_aggregated_result:
    data.append(
        _generate_batch_details(
            batch_detail.aggregated_batch_result,
            "Batch Size {}".format(batch_detail.batch_size),
            "N/A",
            "{:.1f}".format(batch_detail.batch_throughput),
        )
    )
  data.append(
      _generate_batch_details(
          per_model_stats.aggregated_batch_detail,
          "Aggregated",
          "N/A",
          "{:.1f}".format(per_model_stats.batch_throughput),
      )
  )
  return gviz_api.DataTable(columns, data, properties)


def _generate_tensor_pattern_table(
    per_model_inference_stats: inference_stats_pb2.PerModelInferenceStats,
    tensor_pattern_db: inference_stats_pb2.TensorPatternDatabase,
):
  """Generates the tensor pattern table."""
  table_description = [
      ("id", "number", "ID"),
      ("tensor_pattern", "string", "Tensor Pattern"),
      ("count", "number", "Number of Occurrence"),
      ("percentile", "string", "Linearize/Delinearize latency"),
  ]
  data = []
  for counter, aggregated_result in enumerate(
      per_model_inference_stats.tensor_transfer_aggregated_result.tensor_pattern_results
  ):
    tensor_pattern = tensor_pattern_db.tensor_pattern[
        aggregated_result.tensor_pattern_index
    ]
    data.append([
        counter,
        tensor_pattern,
        aggregated_result.count,
        aggregated_result.linearize_delinearize_percentile_time,
    ])
  return gviz_api.DataTable(table_description, data)


def _generate_per_model_inference_table(
    inference_stats: inference_stats_pb2.InferenceStats,
    sorted_model_ids: list[str],
    has_batching: bool,
    is_tpu: bool,
):
  """Generates the per model inference table."""
  tables = []
  for model_id in sorted_model_ids:
    try:
      model_index = inference_stats.model_id_db.id_to_index[model_id]
      per_model_stats = inference_stats.inference_stats_per_model[model_index]
      sampled_per_model_stats = inference_stats.sampled_inference_stats.sampled_inference_stats_per_model[
          model_index
      ]
      tables.append(
          _create_request_table(
              per_model_stats, sampled_per_model_stats, has_batching, is_tpu
          )
      )
      if has_batching:
        tables.append(
            _generate_batch_table(
                per_model_stats,
                sampled_per_model_stats,
                inference_stats.model_id_db,
                model_id,
            )
        )
      if inference_stats.tensor_pattern_db.tensor_pattern:
        tables.append(
            _generate_tensor_pattern_table(
                per_model_stats, inference_stats.tensor_pattern_db
            )
        )
    except KeyError:
      continue
  return tables


def generate_all_chart_tables(
    inference_stats: inference_stats_pb2.InferenceStats,
):
  """Converts a InferenceStats proto to gviz DataTables."""
  sorted_model_ids = [x for x in inference_stats.model_id_db.ids]
  sorted_model_ids.sort()
  has_batching = False
  for _, per_model_stats in inference_stats.inference_stats_per_model.items():
    if per_model_stats.batch_details:
      has_batching = True
      break
  is_tpu = True
  table_properties = {
      "hasBatching": "{}".format(has_batching).lower(),
      "hasTensorPattern": "false",
  }
  columns = [
      ("model_name", "string", "Model Name"),
  ]
  data = []
  for model_id in sorted_model_ids:
    data.append([model_id])
  return [
      gviz_api.DataTable(columns, data, table_properties),
      *_generate_per_model_inference_table(
          inference_stats,
          sorted_model_ids,
          has_batching,
          is_tpu,
      ),
  ]


def to_json(raw_data):
  """Converts a serialized DcnCollectiveAnalysis string to json."""
  inference_stats = inference_stats_pb2.InferenceStats()
  inference_stats.ParseFromString(raw_data)
  all_chart_tables = generate_all_chart_tables(inference_stats)
  json_join = ",".join(x.ToJSon() if x else "{}" for x in all_chart_tables)
  return "[" + json_join + "]"
