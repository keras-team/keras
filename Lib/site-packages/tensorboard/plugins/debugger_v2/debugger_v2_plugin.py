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
"""The TensorBoard Debugger V2 plugin."""


import threading

from werkzeug import wrappers

from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.plugins import base_plugin
from tensorboard.plugins.debugger_v2 import debug_data_provider
from tensorboard.backend import http_util


def _error_response(request, error_message):
    return http_util.Respond(
        request,
        {"error": error_message},
        "application/json",
        code=400,
    )


def _missing_run_error_response(request):
    return _error_response(request, "run parameter is not provided")


class DebuggerV2Plugin(base_plugin.TBPlugin):
    """Debugger V2 Plugin for TensorBoard."""

    plugin_name = debug_data_provider.PLUGIN_NAME

    def __init__(self, context):
        """Instantiates Debugger V2 Plugin via TensorBoard core.

        Args:
          context: A base_plugin.TBContext instance.
        """
        super().__init__(context)
        self._logdir = context.logdir
        self._underlying_data_provider = None
        # Held while initializing `_underlying_data_provider` for the first
        # time, to make sure that we only construct one.
        self._data_provider_init_lock = threading.Lock()

    @property
    def _data_provider(self):
        if self._underlying_data_provider is not None:
            return self._underlying_data_provider
        with self._data_provider_init_lock:
            if self._underlying_data_provider is not None:
                return self._underlying_data_provider
            # TODO(cais): Implement factory for DataProvider that takes into account
            # the settings.
            dp = debug_data_provider.LocalDebuggerV2DataProvider(self._logdir)
            self._underlying_data_provider = dp
            return dp

    def get_plugin_apps(self):
        # TODO(cais): Add routes as they are implemented.
        return {
            "/runs": self.serve_runs,
            "/alerts": self.serve_alerts,
            "/execution/digests": self.serve_execution_digests,
            "/execution/data": self.serve_execution_data,
            "/graph_execution/digests": self.serve_graph_execution_digests,
            "/graph_execution/data": self.serve_graph_execution_data,
            "/graphs/graph_info": self.serve_graph_info,
            "/graphs/op_info": self.serve_graph_op_info,
            "/source_files/list": self.serve_source_files_list,
            "/source_files/file": self.serve_source_file,
            "/stack_frames/stack_frames": self.serve_stack_frames,
        }

    def is_active(self):
        """The Debugger V2 plugin must be manually selected."""
        return False

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(
            is_ng_component=True, tab_name="Debugger V2", disable_reload=False
        )

    @wrappers.Request.application
    def serve_runs(self, request):
        experiment = plugin_util.experiment_id(request.environ)
        runs = self._data_provider.list_runs(experiment_id=experiment)
        run_listing = dict()
        for run in runs:
            run_listing[run.run_id] = {"start_time": run.start_time}
        return http_util.Respond(request, run_listing, "application/json")

    @wrappers.Request.application
    def serve_alerts(self, request):
        experiment = plugin_util.experiment_id(request.environ)
        run = request.args.get("run")
        if run is None:
            return _missing_run_error_response(request)
        begin = int(request.args.get("begin", "0"))
        end = int(request.args.get("end", "-1"))
        alert_type = request.args.get("alert_type", None)
        run_tag_filter = debug_data_provider.alerts_run_tag_filter(
            run, begin, end, alert_type=alert_type
        )
        blob_sequences = self._data_provider.read_blob_sequences(
            experiment_id=experiment,
            plugin_name=self.plugin_name,
            run_tag_filter=run_tag_filter,
        )
        tag = next(iter(run_tag_filter.tags))
        try:
            return http_util.Respond(
                request,
                self._data_provider.read_blob(
                    blob_key=blob_sequences[run][tag][0].blob_key
                ),
                "application/json",
            )
        except errors.InvalidArgumentError as e:
            return _error_response(request, str(e))

    @wrappers.Request.application
    def serve_execution_digests(self, request):
        experiment = plugin_util.experiment_id(request.environ)
        run = request.args.get("run")
        if run is None:
            return _missing_run_error_response(request)
        begin = int(request.args.get("begin", "0"))
        end = int(request.args.get("end", "-1"))
        run_tag_filter = debug_data_provider.execution_digest_run_tag_filter(
            run, begin, end
        )
        blob_sequences = self._data_provider.read_blob_sequences(
            experiment_id=experiment,
            plugin_name=self.plugin_name,
            run_tag_filter=run_tag_filter,
        )
        tag = next(iter(run_tag_filter.tags))
        try:
            return http_util.Respond(
                request,
                self._data_provider.read_blob(
                    blob_key=blob_sequences[run][tag][0].blob_key
                ),
                "application/json",
            )
        except errors.InvalidArgumentError as e:
            return _error_response(request, str(e))

    @wrappers.Request.application
    def serve_execution_data(self, request):
        experiment = plugin_util.experiment_id(request.environ)
        run = request.args.get("run")
        if run is None:
            return _missing_run_error_response(request)
        begin = int(request.args.get("begin", "0"))
        end = int(request.args.get("end", "-1"))
        run_tag_filter = debug_data_provider.execution_data_run_tag_filter(
            run, begin, end
        )
        blob_sequences = self._data_provider.read_blob_sequences(
            experiment_id=experiment,
            plugin_name=self.plugin_name,
            run_tag_filter=run_tag_filter,
        )
        tag = next(iter(run_tag_filter.tags))
        try:
            return http_util.Respond(
                request,
                self._data_provider.read_blob(
                    blob_key=blob_sequences[run][tag][0].blob_key
                ),
                "application/json",
            )
        except errors.InvalidArgumentError as e:
            return _error_response(request, str(e))

    @wrappers.Request.application
    def serve_graph_execution_digests(self, request):
        """Serve digests of intra-graph execution events.

        As the names imply, this route differs from `serve_execution_digests()`
        in that it is for intra-graph execution, while `serve_execution_digests()`
        is for top-level (eager) execution.
        """
        experiment = plugin_util.experiment_id(request.environ)
        run = request.args.get("run")
        if run is None:
            return _missing_run_error_response(request)
        begin = int(request.args.get("begin", "0"))
        end = int(request.args.get("end", "-1"))
        run_tag_filter = (
            debug_data_provider.graph_execution_digest_run_tag_filter(
                run, begin, end
            )
        )
        blob_sequences = self._data_provider.read_blob_sequences(
            experiment_id=experiment,
            plugin_name=self.plugin_name,
            run_tag_filter=run_tag_filter,
        )
        tag = next(iter(run_tag_filter.tags))
        try:
            return http_util.Respond(
                request,
                self._data_provider.read_blob(
                    blob_key=blob_sequences[run][tag][0].blob_key
                ),
                "application/json",
            )
        except errors.InvalidArgumentError as e:
            return _error_response(request, str(e))

    @wrappers.Request.application
    def serve_graph_execution_data(self, request):
        """Serve detailed data objects of intra-graph execution events.

        As the names imply, this route differs from `serve_execution_data()`
        in that it is for intra-graph execution, while `serve_execution_data()`
        is for top-level (eager) execution.

        Unlike `serve_graph_execution_digests()`, this method serves the
        full-sized data objects for intra-graph execution events.
        """
        experiment = plugin_util.experiment_id(request.environ)
        run = request.args.get("run")
        if run is None:
            return _missing_run_error_response(request)
        begin = int(request.args.get("begin", "0"))
        end = int(request.args.get("end", "-1"))
        run_tag_filter = (
            debug_data_provider.graph_execution_data_run_tag_filter(
                run, begin, end
            )
        )
        blob_sequences = self._data_provider.read_blob_sequences(
            experiment_id=experiment,
            plugin_name=self.plugin_name,
            run_tag_filter=run_tag_filter,
        )
        tag = next(iter(run_tag_filter.tags))
        try:
            return http_util.Respond(
                request,
                self._data_provider.read_blob(
                    blob_key=blob_sequences[run][tag][0].blob_key
                ),
                "application/json",
            )
        except errors.InvalidArgumentError as e:
            return _error_response(request, str(e))

    @wrappers.Request.application
    def serve_graph_info(self, request):
        """Serve basic information about a TensorFlow graph.

        The request specifies the debugger-generated ID of the graph being
        queried.

        The response contains a JSON object with the following fields:
          - graph_id: The debugger-generated ID (echoing the request).
          - name: The name of the graph (if any). For TensorFlow 2.x
            Function Graphs (FuncGraphs), this is typically the name of
            the underlying Python function, optionally prefixed with
            TensorFlow-generated prefixed such as "__inference_".
            Some graphs (e.g., certain outermost graphs) may have no names,
            in which case this field is `null`.
          - outer_graph_id: Outer graph ID (if any). For an outermost graph
            without an outer graph context, this field is `null`.
          - inner_graph_ids: Debugger-generated IDs of all the graphs
            nested inside this graph. For a graph without any graphs nested
            inside, this field is an empty array.
        """
        experiment = plugin_util.experiment_id(request.environ)
        run = request.args.get("run")
        if run is None:
            return _missing_run_error_response(request)
        graph_id = request.args.get("graph_id")
        run_tag_filter = debug_data_provider.graph_info_run_tag_filter(
            run, graph_id
        )
        blob_sequences = self._data_provider.read_blob_sequences(
            experiment_id=experiment,
            plugin_name=self.plugin_name,
            run_tag_filter=run_tag_filter,
        )
        tag = next(iter(run_tag_filter.tags))
        try:
            return http_util.Respond(
                request,
                self._data_provider.read_blob(
                    blob_key=blob_sequences[run][tag][0].blob_key
                ),
                "application/json",
            )
        except errors.NotFoundError as e:
            return _error_response(request, str(e))

    @wrappers.Request.application
    def serve_graph_op_info(self, request):
        """Serve information for ops in graphs.

        The request specifies the op name and the ID of the graph that
        contains the op.

        The response contains a JSON object with the following fields:
          - op_type
          - op_name
          - graph_ids: Stack of graph IDs that the op is located in, from
            outermost to innermost. The length of this array is always >= 1.
            The length is 1 if and only if the graph is an outermost graph.
          - num_outputs: Number of output tensors.
          - output_tensor_ids: The debugger-generated number IDs for the
            symbolic output tensors of the op (an array of numbers).
          - host_name: Name of the host on which the op is created.
          - stack_trace: Stack frames of the op's creation.
          - inputs: Specifications of all inputs to this op.
            Currently only immediate (one level of) inputs are provided.
            This is an array of length N_in, where N_in is the number of
            data inputs received by the op. Each element of the array is an
            object with the following fields:
              - op_name: Name of the op that provides the input tensor.
              - output_slot: 0-based output slot index from which the input
                tensor emits.
              - data: A recursive data structure of this same schema.
                This field is not populated (undefined) at the leaf nodes
                of this recursive data structure.
                In the rare case wherein the data for an input cannot be
                retrieved properly (e.g., special internal op types), this
                field will be unpopulated.
            This is an empty list for an op with no inputs.
          - consumers: Specifications for all the downstream consuming ops of
            this. Currently only immediate (one level of) consumers are provided.
            This is an array of length N_out, where N_out is the number of
            symbolic tensors output by this op.
            Each element of the array is an array of which the length equals
            the number of downstream ops that consume the corresponding symbolic
            tensor (only data edges are tracked).
            Each element of the array is an object with the following fields:
              - op_name: Name of the op that receives the output tensor as an
                input.
              - input_slot: 0-based input slot index at which the downstream
                op receives this output tensor.
              - data: A recursive data structure of this very schema.
                This field is not populated (undefined) at the leaf nodes
                of this recursive data structure.
                In the rare case wherein the data for a consumer op cannot be
                retrieved properly (e.g., special internal op types), this
                field will be unpopulated.
            If this op has no output tensors, this is an empty array.
            If one of the output tensors of this op has no consumers, the
            corresponding element is an empty array.
        """
        experiment = plugin_util.experiment_id(request.environ)
        run = request.args.get("run")
        if run is None:
            return _missing_run_error_response(request)
        graph_id = request.args.get("graph_id")
        op_name = request.args.get("op_name")
        run_tag_filter = debug_data_provider.graph_op_info_run_tag_filter(
            run, graph_id, op_name
        )
        blob_sequences = self._data_provider.read_blob_sequences(
            experiment_id=experiment,
            plugin_name=self.plugin_name,
            run_tag_filter=run_tag_filter,
        )
        tag = next(iter(run_tag_filter.tags))
        try:
            return http_util.Respond(
                request,
                self._data_provider.read_blob(
                    blob_key=blob_sequences[run][tag][0].blob_key
                ),
                "application/json",
            )
        except errors.NotFoundError as e:
            return _error_response(request, str(e))

    @wrappers.Request.application
    def serve_source_files_list(self, request):
        """Serves a list of all source files involved in the debugged program."""
        experiment = plugin_util.experiment_id(request.environ)
        run = request.args.get("run")
        if run is None:
            return _missing_run_error_response(request)
        run_tag_filter = debug_data_provider.source_file_list_run_tag_filter(
            run
        )
        blob_sequences = self._data_provider.read_blob_sequences(
            experiment_id=experiment,
            plugin_name=self.plugin_name,
            run_tag_filter=run_tag_filter,
        )
        tag = next(iter(run_tag_filter.tags))
        return http_util.Respond(
            request,
            self._data_provider.read_blob(
                blob_key=blob_sequences[run][tag][0].blob_key
            ),
            "application/json",
        )

    @wrappers.Request.application
    def serve_source_file(self, request):
        """Serves the content of a given source file.

        The source file is referred to by the index in the list of all source
        files involved in the execution of the debugged program, which is
        available via the `serve_source_files_list()`  serving route.

        Args:
          request: HTTP request.

        Returns:
          Response to the request.
        """
        experiment = plugin_util.experiment_id(request.environ)
        run = request.args.get("run")
        if run is None:
            return _missing_run_error_response(request)
        index = request.args.get("index")
        # TOOD(cais): When the need arises, support serving a subset of a
        # source file's lines.
        if index is None:
            return _error_response(
                request, "index is not provided for source file content"
            )
        index = int(index)
        run_tag_filter = debug_data_provider.source_file_run_tag_filter(
            run, index
        )
        blob_sequences = self._data_provider.read_blob_sequences(
            experiment_id=experiment,
            plugin_name=self.plugin_name,
            run_tag_filter=run_tag_filter,
        )
        tag = next(iter(run_tag_filter.tags))
        try:
            return http_util.Respond(
                request,
                self._data_provider.read_blob(
                    blob_key=blob_sequences[run][tag][0].blob_key
                ),
                "application/json",
            )
        except errors.NotFoundError as e:
            return _error_response(request, str(e))

    @wrappers.Request.application
    def serve_stack_frames(self, request):
        """Serves the content of stack frames.

        The source frames being requested are referred to be UUIDs for each of
        them, separated by commas.

        Args:
          request: HTTP request.

        Returns:
          Response to the request.
        """
        experiment = plugin_util.experiment_id(request.environ)
        run = request.args.get("run")
        if run is None:
            return _missing_run_error_response(request)
        stack_frame_ids = request.args.get("stack_frame_ids")
        if stack_frame_ids is None:
            return _error_response(request, "Missing stack_frame_ids parameter")
        if not stack_frame_ids:
            return _error_response(request, "Empty stack_frame_ids parameter")
        stack_frame_ids = stack_frame_ids.split(",")
        run_tag_filter = debug_data_provider.stack_frames_run_tag_filter(
            run, stack_frame_ids
        )
        blob_sequences = self._data_provider.read_blob_sequences(
            experiment_id=experiment,
            plugin_name=self.plugin_name,
            run_tag_filter=run_tag_filter,
        )
        tag = next(iter(run_tag_filter.tags))
        try:
            return http_util.Respond(
                request,
                self._data_provider.read_blob(
                    blob_key=blob_sequences[run][tag][0].blob_key
                ),
                "application/json",
            )
        except errors.NotFoundError as e:
            return _error_response(request, str(e))
