# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A data provider that talks to a gRPC server."""

import collections
import contextlib

import grpc

from tensorboard.util import tensor_util
from tensorboard.util import timing
from tensorboard import errors
from tensorboard.data import provider
from tensorboard.data.proto import data_provider_pb2
from tensorboard.data.proto import data_provider_pb2_grpc


def make_stub(channel):
    """Wraps a gRPC channel with a service stub."""
    return data_provider_pb2_grpc.TensorBoardDataProviderStub(channel)


class GrpcDataProvider(provider.DataProvider):
    """Data provider that talks over gRPC."""

    def __init__(self, addr, stub):
        """Initializes a GrpcDataProvider.

        Args:
          addr: String address of the remote peer. Used cosmetically for
            data location.
          stub: `data_provider_pb2_grpc.TensorBoardDataProviderStub`
            value. See `make_stub` to construct one from a channel.
        """
        self._addr = addr
        self._stub = stub

    def __str__(self):
        return "GrpcDataProvider(addr=%r)" % self._addr

    def experiment_metadata(self, ctx, *, experiment_id):
        req = data_provider_pb2.GetExperimentRequest()
        req.experiment_id = experiment_id
        with _translate_grpc_error():
            res = self._stub.GetExperiment(req)
        res = provider.ExperimentMetadata(
            data_location=res.data_location,
            experiment_name=res.name,
            experiment_description=res.description,
            creation_time=_timestamp_proto_to_float(res.creation_time),
        )
        return res

    def list_plugins(self, ctx, *, experiment_id):
        req = data_provider_pb2.ListPluginsRequest()
        req.experiment_id = experiment_id
        with _translate_grpc_error():
            res = self._stub.ListPlugins(req)
        return [p.name for p in res.plugins]

    def list_runs(self, ctx, *, experiment_id):
        req = data_provider_pb2.ListRunsRequest()
        req.experiment_id = experiment_id
        with _translate_grpc_error():
            res = self._stub.ListRuns(req)
        return [
            provider.Run(
                run_id=run.name,
                run_name=run.name,
                start_time=run.start_time,
            )
            for run in res.runs
        ]

    @timing.log_latency
    def list_scalars(
        self, ctx, *, experiment_id, plugin_name, run_tag_filter=None
    ):
        with timing.log_latency("build request"):
            req = data_provider_pb2.ListScalarsRequest()
            req.experiment_id = experiment_id
            req.plugin_filter.plugin_name = plugin_name
            _populate_rtf(run_tag_filter, req.run_tag_filter)
        with timing.log_latency("_stub.ListScalars"):
            with _translate_grpc_error():
                res = self._stub.ListScalars(req)
        with timing.log_latency("build result"):
            result = {}
            for run_entry in res.runs:
                tags = {}
                result[run_entry.run_name] = tags
                for tag_entry in run_entry.tags:
                    time_series = tag_entry.metadata
                    tags[tag_entry.tag_name] = provider.ScalarTimeSeries(
                        max_step=time_series.max_step,
                        max_wall_time=time_series.max_wall_time,
                        plugin_content=time_series.summary_metadata.plugin_data.content,
                        description=time_series.summary_metadata.summary_description,
                        display_name=time_series.summary_metadata.display_name,
                    )
            return result

    @timing.log_latency
    def read_scalars(
        self,
        ctx,
        *,
        experiment_id,
        plugin_name,
        downsample=None,
        run_tag_filter=None,
    ):
        with timing.log_latency("build request"):
            req = data_provider_pb2.ReadScalarsRequest()
            req.experiment_id = experiment_id
            req.plugin_filter.plugin_name = plugin_name
            _populate_rtf(run_tag_filter, req.run_tag_filter)
            req.downsample.num_points = downsample
        with timing.log_latency("_stub.ReadScalars"):
            with _translate_grpc_error():
                res = self._stub.ReadScalars(req)
        with timing.log_latency("build result"):
            result = {}
            for run_entry in res.runs:
                tags = {}
                result[run_entry.run_name] = tags
                for tag_entry in run_entry.tags:
                    series = []
                    tags[tag_entry.tag_name] = series
                    d = tag_entry.data
                    for step, wt, value in zip(d.step, d.wall_time, d.value):
                        point = provider.ScalarDatum(
                            step=step,
                            wall_time=wt,
                            value=value,
                        )
                        series.append(point)
            return result

    @timing.log_latency
    def read_last_scalars(
        self,
        ctx,
        *,
        experiment_id,
        plugin_name,
        run_tag_filter=None,
    ):
        with timing.log_latency("build request"):
            req = data_provider_pb2.ReadScalarsRequest()
            req.experiment_id = experiment_id
            req.plugin_filter.plugin_name = plugin_name
            _populate_rtf(run_tag_filter, req.run_tag_filter)
            # `ReadScalars` always includes the most recent datum, therefore
            # downsampling to one means fetching the latest value.
            req.downsample.num_points = 1
        with timing.log_latency("_stub.ReadScalars"):
            with _translate_grpc_error():
                res = self._stub.ReadScalars(req)
        with timing.log_latency("build result"):
            result = collections.defaultdict(dict)
            for run_entry in res.runs:
                run_name = run_entry.run_name
                for tag_entry in run_entry.tags:
                    d = tag_entry.data
                    # There should be no more than one datum in
                    # `tag_entry.data` since downsample was set to 1.
                    for step, wt, value in zip(d.step, d.wall_time, d.value):
                        result[run_name][tag_entry.tag_name] = (
                            provider.ScalarDatum(
                                step=step,
                                wall_time=wt,
                                value=value,
                            )
                        )
            return result

    @timing.log_latency
    def list_tensors(
        self, ctx, *, experiment_id, plugin_name, run_tag_filter=None
    ):
        with timing.log_latency("build request"):
            req = data_provider_pb2.ListTensorsRequest()
            req.experiment_id = experiment_id
            req.plugin_filter.plugin_name = plugin_name
            _populate_rtf(run_tag_filter, req.run_tag_filter)
        with timing.log_latency("_stub.ListTensors"):
            with _translate_grpc_error():
                res = self._stub.ListTensors(req)
        with timing.log_latency("build result"):
            result = {}
            for run_entry in res.runs:
                tags = {}
                result[run_entry.run_name] = tags
                for tag_entry in run_entry.tags:
                    time_series = tag_entry.metadata
                    tags[tag_entry.tag_name] = provider.TensorTimeSeries(
                        max_step=time_series.max_step,
                        max_wall_time=time_series.max_wall_time,
                        plugin_content=time_series.summary_metadata.plugin_data.content,
                        description=time_series.summary_metadata.summary_description,
                        display_name=time_series.summary_metadata.display_name,
                    )
            return result

    @timing.log_latency
    def read_tensors(
        self,
        ctx,
        *,
        experiment_id,
        plugin_name,
        downsample=None,
        run_tag_filter=None,
    ):
        with timing.log_latency("build request"):
            req = data_provider_pb2.ReadTensorsRequest()
            req.experiment_id = experiment_id
            req.plugin_filter.plugin_name = plugin_name
            _populate_rtf(run_tag_filter, req.run_tag_filter)
            req.downsample.num_points = downsample
        with timing.log_latency("_stub.ReadTensors"):
            with _translate_grpc_error():
                res = self._stub.ReadTensors(req)
        with timing.log_latency("build result"):
            result = {}
            for run_entry in res.runs:
                tags = {}
                result[run_entry.run_name] = tags
                for tag_entry in run_entry.tags:
                    series = []
                    tags[tag_entry.tag_name] = series
                    d = tag_entry.data
                    for step, wt, value in zip(d.step, d.wall_time, d.value):
                        point = provider.TensorDatum(
                            step=step,
                            wall_time=wt,
                            numpy=tensor_util.make_ndarray(value),
                        )
                        series.append(point)
            return result

    @timing.log_latency
    def list_blob_sequences(
        self, ctx, experiment_id, plugin_name, run_tag_filter=None
    ):
        with timing.log_latency("build request"):
            req = data_provider_pb2.ListBlobSequencesRequest()
            req.experiment_id = experiment_id
            req.plugin_filter.plugin_name = plugin_name
            _populate_rtf(run_tag_filter, req.run_tag_filter)
        with timing.log_latency("_stub.ListBlobSequences"):
            with _translate_grpc_error():
                res = self._stub.ListBlobSequences(req)
        with timing.log_latency("build result"):
            result = {}
            for run_entry in res.runs:
                tags = {}
                result[run_entry.run_name] = tags
                for tag_entry in run_entry.tags:
                    time_series = tag_entry.metadata
                    tags[tag_entry.tag_name] = provider.BlobSequenceTimeSeries(
                        max_step=time_series.max_step,
                        max_wall_time=time_series.max_wall_time,
                        max_length=time_series.max_length,
                        plugin_content=time_series.summary_metadata.plugin_data.content,
                        description=time_series.summary_metadata.summary_description,
                        display_name=time_series.summary_metadata.display_name,
                    )
            return result

    @timing.log_latency
    def read_blob_sequences(
        self,
        ctx,
        experiment_id,
        plugin_name,
        downsample=None,
        run_tag_filter=None,
    ):
        with timing.log_latency("build request"):
            req = data_provider_pb2.ReadBlobSequencesRequest()
            req.experiment_id = experiment_id
            req.plugin_filter.plugin_name = plugin_name
            _populate_rtf(run_tag_filter, req.run_tag_filter)
            req.downsample.num_points = downsample
        with timing.log_latency("_stub.ReadBlobSequences"):
            with _translate_grpc_error():
                res = self._stub.ReadBlobSequences(req)
        with timing.log_latency("build result"):
            result = {}
            for run_entry in res.runs:
                tags = {}
                result[run_entry.run_name] = tags
                for tag_entry in run_entry.tags:
                    series = []
                    tags[tag_entry.tag_name] = series
                    d = tag_entry.data
                    for step, wt, blob_sequence in zip(
                        d.step, d.wall_time, d.values
                    ):
                        values = []
                        for ref in blob_sequence.blob_refs:
                            values.append(
                                provider.BlobReference(
                                    blob_key=ref.blob_key, url=ref.url or None
                                )
                            )
                        point = provider.BlobSequenceDatum(
                            step=step, wall_time=wt, values=tuple(values)
                        )
                        series.append(point)
            return result

    @timing.log_latency
    def read_blob(self, ctx, blob_key):
        with timing.log_latency("build request"):
            req = data_provider_pb2.ReadBlobRequest()
            req.blob_key = blob_key
        with timing.log_latency("list(_stub.ReadBlob)"):
            with _translate_grpc_error():
                responses = list(self._stub.ReadBlob(req))
        with timing.log_latency("build result"):
            return b"".join(res.data for res in responses)


@contextlib.contextmanager
def _translate_grpc_error():
    try:
        yield
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.INVALID_ARGUMENT:
            raise errors.InvalidArgumentError(e.details())
        if e.code() == grpc.StatusCode.NOT_FOUND:
            raise errors.NotFoundError(e.details())
        if e.code() == grpc.StatusCode.PERMISSION_DENIED:
            raise errors.PermissionDeniedError(e.details())
        raise


def _populate_rtf(run_tag_filter, rtf_proto):
    """Copies `run_tag_filter` into `rtf_proto`."""
    if run_tag_filter is None:
        return
    if run_tag_filter.runs is not None:
        rtf_proto.runs.names[:] = sorted(run_tag_filter.runs)
    if run_tag_filter.tags is not None:
        rtf_proto.tags.names[:] = sorted(run_tag_filter.tags)


def _timestamp_proto_to_float(ts):
    """Converts `timestamp_pb2.Timestamp` to float seconds since epoch."""
    return ts.ToNanoseconds() / 1e9
