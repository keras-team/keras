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
"""A wrapper around DebugDataReader used for retrieving tfdbg v2 data."""


import threading

from tensorboard import errors

# Dummy run name for the debugger.
# Currently, the `DebuggerV2ExperimentMultiplexer` class is tied to a single
# logdir, which holds at most one DebugEvent file set in the tfdbg v2 (tfdbg2
# for short) format.
# TODO(cais): When tfdbg2 allows there to be multiple DebugEvent file sets in
# the same logdir, replace this magic string with actual run names.
DEFAULT_DEBUGGER_RUN_NAME = "__default_debugger_run__"

# Default number of alerts per monitor type.
# Limiting the number of alerts is based on the consideration that usually
# only the first few alerting events are the most critical and the subsequent
# ones are either repetitions of the earlier ones or caused by the earlier ones.
DEFAULT_PER_TYPE_ALERT_LIMIT = 1000

# Default interval between successive calls to `DebugDataReader.update()``.
DEFAULT_RELOAD_INTERVAL_SEC = 30


def run_repeatedly_in_background(target, interval_sec):
    """Run a target task repeatedly in the background.

    In the context of this module, `target` is the `update()` method of the
    underlying reader for tfdbg2-format data.
    This method is mocked by unit tests for deterministic behaviors during
    testing.

    Args:
      target: The target task to run in the background, a callable with no args.
      interval_sec: Time interval between repeats, in seconds.

    Returns:
      - A `threading.Event` object that can be used to interrupt an ongoing
          waiting interval between successive runs of `target`. To interrupt the
          interval, call the `set()` method of the object.
      - The `threading.Thread` object on which `target` is run repeatedly.
    """
    event = threading.Event()

    def _run_repeatedly():
        while True:
            target()
            event.wait(interval_sec)
            event.clear()

    # Use `daemon=True` to make sure the thread doesn't block program exit.
    thread = threading.Thread(target=_run_repeatedly, daemon=True)
    thread.start()
    return event, thread


def _alert_to_json(alert):
    # TODO(cais): Replace this with Alert.to_json() when supported by the
    # backend.
    from tensorflow.python.debug.lib import debug_events_monitors

    if isinstance(alert, debug_events_monitors.InfNanAlert):
        return {
            "alert_type": "InfNanAlert",
            "op_type": alert.op_type,
            "output_slot": alert.output_slot,
            # TODO(cais): Once supported by backend, add 'op_name' key
            # for intra-graph execution events.
            "size": alert.size,
            "num_neg_inf": alert.num_neg_inf,
            "num_pos_inf": alert.num_pos_inf,
            "num_nan": alert.num_nan,
            "execution_index": alert.execution_index,
            "graph_execution_trace_index": alert.graph_execution_trace_index,
        }
    else:
        raise TypeError("Unrecognized alert subtype: %s" % type(alert))


def parse_tensor_name(tensor_name):
    """Helper function that extracts op name and slot from tensor name."""
    output_slot = 0
    if ":" in tensor_name:
        op_name, output_slot = tensor_name.split(":")
        output_slot = int(output_slot)
    else:
        op_name = tensor_name
    return op_name, output_slot


class DebuggerV2EventMultiplexer:
    """A class used for accessing tfdbg v2 DebugEvent data on local filesystem.

    This class is a short-term hack, mirroring the EventMultiplexer for the main
    TensorBoard plugins (e.g., scalar, histogram and graphs.) As such, it only
    implements the methods relevant to the Debugger V2 pluggin.

    TODO(cais): Integrate it with EventMultiplexer and use the integrated class
    from MultiplexerDataProvider for a single path of accessing debugger and
    non-debugger data.
    """

    def __init__(self, logdir):
        """Constructor for the `DebugEventMultiplexer`.

        Args:
          logdir: Path to the directory to load the tfdbg v2 data from.
        """
        self._logdir = logdir
        self._reader = None
        self._reader_lock = threading.Lock()
        self._reload_needed_event = None
        # Create the reader for the tfdbg2 data in the lodir as soon as
        # the backend of the debugger-v2 plugin is created, so it doesn't need
        # to wait for the first request from the FE to start loading data.
        self._tryCreateReader()

    def _tryCreateReader(self):
        """Try creating reader for tfdbg2 data in the logdir.

        If the reader has already been created, a new one will not be created and
        this function is a no-op.

        If a reader has not been created, create it and start periodic calls to
        `update()` on a separate thread.
        """
        if self._reader:
            return
        with self._reader_lock:
            if not self._reader:
                try:
                    # TODO(cais): Avoid conditional imports and instead use
                    # plugin loader to gate the loading of this entire plugin.
                    from tensorflow.python.debug.lib import debug_events_reader
                    from tensorflow.python.debug.lib import (
                        debug_events_monitors,
                    )
                except ImportError:
                    # This ensures graceful behavior when tensorflow install is
                    # unavailable or when the installed tensorflow version does not
                    # contain the required modules.
                    return

                try:
                    self._reader = debug_events_reader.DebugDataReader(
                        self._logdir
                    )
                except AttributeError:
                    # Gracefully fail for users without the required API changes to
                    # debug_events_reader.DebugDataReader introduced in
                    # TF 2.1.0.dev20200103. This should be safe to remove when
                    # TF 2.2 is released.
                    return
                except ValueError:
                    # When no DebugEvent file set is found in the logdir, a
                    # `ValueError` is thrown.
                    return

                self._monitors = [
                    debug_events_monitors.InfNanMonitor(
                        self._reader, limit=DEFAULT_PER_TYPE_ALERT_LIMIT
                    )
                ]
                self._reload_needed_event, _ = run_repeatedly_in_background(
                    self._reader.update, DEFAULT_RELOAD_INTERVAL_SEC
                )

    def _reloadReader(self):
        """If a reader exists and has started period updating, unblock the update.

        The updates are performed periodically with a sleep interval between
        successive calls to the reader's update() method. Calling this method
        interrupts the sleep immediately if one is ongoing.
        """
        if self._reload_needed_event:
            self._reload_needed_event.set()

    def FirstEventTimestamp(self, run):
        """Return the timestamp of the first DebugEvent of the given run.

        This may perform I/O if no events have been loaded yet for the run.

        Args:
          run: A string name of the run for which the timestamp is retrieved.
            This currently must be hardcoded as `DEFAULT_DEBUGGER_RUN_NAME`,
            as each logdir contains at most one DebugEvent file set (i.e., a
            run of a tfdbg2-instrumented TensorFlow program.)

        Returns:
            The wall_time of the first event of the run, which will be in seconds
            since the epoch as a `float`.
        """
        if self._reader is None:
            raise ValueError("No tfdbg2 runs exists.")
        if run != DEFAULT_DEBUGGER_RUN_NAME:
            raise ValueError(
                "Expected run name to be %s, but got %s"
                % (DEFAULT_DEBUGGER_RUN_NAME, run)
            )
        return self._reader.starting_wall_time()

    def PluginRunToTagToContent(self, plugin_name):
        raise NotImplementedError(
            "DebugDataMultiplexer.PluginRunToTagToContent() has not been "
            "implemented yet."
        )

    def Runs(self):
        """Return all the tfdbg2 run names in the logdir watched by this instance.

        The `Run()` method of this class is specialized for the tfdbg2-format
        DebugEvent files.

        As a side effect, this method unblocks the underlying reader's period
        reloading if a reader exists. This lets the reader update at a higher
        frequency than the default one with 30-second sleeping period between
        reloading when data is being queried actively from this instance.
        Note that this `Runs()` method is used by all other public data-access
        methods of this class (e.g., `ExecutionData()`, `GraphExecutionData()`).
        Hence calls to those methods will lead to accelerated data reloading of
        the reader.

        Returns:
          If tfdbg2-format data exists in the `logdir` of this object, returns:
              ```
              {runName: { "debugger-v2": [tag1, tag2, tag3] } }
              ```
              where `runName` is the hard-coded string `DEFAULT_DEBUGGER_RUN_NAME`
              string. This is related to the fact that tfdbg2 currently contains
              at most one DebugEvent file set per directory.
          If no tfdbg2-format data exists in the `logdir`, an empty `dict`.
        """
        # Call `_tryCreateReader()` here to cover the possibility of tfdbg2
        # data start being written to the logdir after the tensorboard backend
        # starts.
        self._tryCreateReader()
        if self._reader:
            # If a _reader exists, unblock its reloading (on a separate thread)
            # immediately.
            self._reloadReader()
            return {
                DEFAULT_DEBUGGER_RUN_NAME: {
                    # TODO(cais): Add the semantically meaningful tag names such as
                    # 'execution_digests_book', 'alerts_book'
                    "debugger-v2": []
                }
            }
        else:
            return {}

    def _checkBeginEndIndices(self, begin, end, total_count):
        if begin < 0:
            raise errors.InvalidArgumentError(
                "Invalid begin index (%d)" % begin
            )
        if end > total_count:
            raise errors.InvalidArgumentError(
                "end index (%d) out of bounds (%d)" % (end, total_count)
            )
        if end >= 0 and end < begin:
            raise errors.InvalidArgumentError(
                "end index (%d) is unexpectedly less than begin index (%d)"
                % (end, begin)
            )
        if end < 0:  # This means all digests.
            end = total_count
        return end

    def Alerts(self, run, begin, end, alert_type_filter=None):
        """Get alerts from the debugged TensorFlow program.

        Args:
          run: The tfdbg2 run to get Alerts from.
          begin: Beginning alert index.
          end: Ending alert index.
          alert_type_filter: Optional filter string for alert type, used to
            restrict retrieved alerts data to a single type. If used,
            `begin` and `end` refer to the beginning and ending indices within
            the filtered alert type.
        """
        from tensorflow.python.debug.lib import debug_events_monitors

        runs = self.Runs()
        if run not in runs:
            # TODO(cais): This should generate a 400 response instead.
            return None
        alerts = []
        alerts_breakdown = dict()
        alerts_by_type = dict()
        for monitor in self._monitors:
            monitor_alerts = monitor.alerts()
            if not monitor_alerts:
                continue
            alerts.extend(monitor_alerts)
            # TODO(cais): Replace this with Alert.to_json() when
            # monitor.alert_type() is available.
            if isinstance(monitor, debug_events_monitors.InfNanMonitor):
                alert_type = "InfNanAlert"
            else:
                alert_type = "__MiscellaneousAlert__"
            alerts_breakdown[alert_type] = len(monitor_alerts)
            alerts_by_type[alert_type] = monitor_alerts
        num_alerts = len(alerts)
        if alert_type_filter is not None:
            if alert_type_filter not in alerts_breakdown:
                raise errors.InvalidArgumentError(
                    "Filtering of alerts failed: alert type %s does not exist"
                    % alert_type_filter
                )
            alerts = alerts_by_type[alert_type_filter]
        end = self._checkBeginEndIndices(begin, end, len(alerts))
        return {
            "begin": begin,
            "end": end,
            "alert_type": alert_type_filter,
            "num_alerts": num_alerts,
            "alerts_breakdown": alerts_breakdown,
            "per_type_alert_limit": DEFAULT_PER_TYPE_ALERT_LIMIT,
            "alerts": [_alert_to_json(alert) for alert in alerts[begin:end]],
        }

    def ExecutionDigests(self, run, begin, end):
        """Get ExecutionDigests.

        Args:
          run: The tfdbg2 run to get `ExecutionDigest`s from.
          begin: Beginning execution index.
          end: Ending execution index.

        Returns:
          A JSON-serializable object containing the `ExecutionDigest`s and
          related meta-information
        """
        runs = self.Runs()
        if run not in runs:
            return None
        # TODO(cais): For scalability, use begin and end kwargs when available in
        # `DebugDataReader.execution()`.`
        execution_digests = self._reader.executions(digest=True)
        end = self._checkBeginEndIndices(begin, end, len(execution_digests))
        return {
            "begin": begin,
            "end": end,
            "num_digests": len(execution_digests),
            "execution_digests": [
                digest.to_json() for digest in execution_digests[begin:end]
            ],
        }

    def ExecutionData(self, run, begin, end):
        """Get Execution data objects (Detailed, non-digest form).

        Args:
          run: The tfdbg2 run to get `ExecutionDigest`s from.
          begin: Beginning execution index.
          end: Ending execution index.

        Returns:
          A JSON-serializable object containing the `ExecutionDigest`s and
          related meta-information
        """
        runs = self.Runs()
        if run not in runs:
            return None
        execution_digests = self._reader.executions(digest=True)
        end = self._checkBeginEndIndices(begin, end, len(execution_digests))
        execution_digests = execution_digests[begin:end]
        executions = self._reader.executions(digest=False, begin=begin, end=end)
        return {
            "begin": begin,
            "end": end,
            "executions": [execution.to_json() for execution in executions],
        }

    def GraphExecutionDigests(self, run, begin, end, trace_id=None):
        """Get `GraphExecutionTraceDigest`s.

        Args:
          run: The tfdbg2 run to get `GraphExecutionTraceDigest`s from.
          begin: Beginning graph-execution index.
          end: Ending graph-execution index.

        Returns:
          A JSON-serializable object containing the `ExecutionDigest`s and
          related meta-information
        """
        runs = self.Runs()
        if run not in runs:
            return None
        # TODO(cais): Implement support for trace_id once the joining of eager
        # execution and intra-graph execution is supported by DebugDataReader.
        if trace_id is not None:
            raise NotImplementedError(
                "trace_id support for GraphExecutionTraceDigest is "
                "not implemented yet."
            )
        graph_exec_digests = self._reader.graph_execution_traces(digest=True)
        end = self._checkBeginEndIndices(begin, end, len(graph_exec_digests))
        return {
            "begin": begin,
            "end": end,
            "num_digests": len(graph_exec_digests),
            "graph_execution_digests": [
                digest.to_json() for digest in graph_exec_digests[begin:end]
            ],
        }

    def GraphExecutionData(self, run, begin, end, trace_id=None):
        """Get `GraphExecutionTrace`s.

        Args:
          run: The tfdbg2 run to get `GraphExecutionTrace`s from.
          begin: Beginning graph-execution index.
          end: Ending graph-execution index.

        Returns:
          A JSON-serializable object containing the `ExecutionDigest`s and
          related meta-information
        """
        runs = self.Runs()
        if run not in runs:
            return None
        # TODO(cais): Implement support for trace_id once the joining of eager
        # execution and intra-graph execution is supported by DebugDataReader.
        if trace_id is not None:
            raise NotImplementedError(
                "trace_id support for GraphExecutionTraceData is "
                "not implemented yet."
            )
        digests = self._reader.graph_execution_traces(digest=True)
        end = self._checkBeginEndIndices(begin, end, len(digests))
        graph_executions = self._reader.graph_execution_traces(
            digest=False, begin=begin, end=end
        )
        return {
            "begin": begin,
            "end": end,
            "graph_executions": [
                graph_exec.to_json() for graph_exec in graph_executions
            ],
        }

    def GraphInfo(self, run, graph_id):
        """Get the information regarding a TensorFlow graph.

        Args:
          run: Name of the run.
          graph_id: Debugger-generated ID of the graph in question.
            This information is available in the return values
            of `GraphOpInfo`, `GraphExecution`, etc.

        Returns:
          A JSON-serializable object containing the information regarding
            the TensorFlow graph.

        Raises:
          NotFoundError if the graph_id is not known to the debugger.
        """
        runs = self.Runs()
        if run not in runs:
            return None
        try:
            graph = self._reader.graph_by_id(graph_id)
        except KeyError:
            raise errors.NotFoundError(
                'There is no graph with ID "%s"' % graph_id
            )
        return graph.to_json()

    def GraphOpInfo(self, run, graph_id, op_name):
        """Get the information regarding a graph op's creation.

        Args:
          run: Name of the run.
          graph_id: Debugger-generated ID of the graph that contains
            the op in question. This ID is available from other methods
            of this class, e.g., the return value of `GraphExecutionDigests()`.
          op_name: Name of the op.

        Returns:
          A JSON-serializable object containing the information regarding
            the op's creation and its immediate inputs and consumers.

        Raises:
          NotFoundError if the graph_id or op_name does not exist.
        """
        runs = self.Runs()
        if run not in runs:
            return None
        try:
            graph = self._reader.graph_by_id(graph_id)
        except KeyError:
            raise errors.NotFoundError(
                'There is no graph with ID "%s"' % graph_id
            )
        try:
            op_creation_digest = graph.get_op_creation_digest(op_name)
        except KeyError:
            raise errors.NotFoundError(
                'There is no op named "%s" in graph with ID "%s"'
                % (op_name, graph_id)
            )
        data_object = self._opCreationDigestToDataObject(
            op_creation_digest, graph
        )
        # Populate data about immediate inputs.
        for input_spec in data_object["inputs"]:
            try:
                input_op_digest = graph.get_op_creation_digest(
                    input_spec["op_name"]
                )
            except KeyError:
                input_op_digest = None
            if input_op_digest:
                input_spec["data"] = self._opCreationDigestToDataObject(
                    input_op_digest, graph
                )
        # Populate data about immediate consuming ops.
        for slot_consumer_specs in data_object["consumers"]:
            for consumer_spec in slot_consumer_specs:
                try:
                    digest = graph.get_op_creation_digest(
                        consumer_spec["op_name"]
                    )
                except KeyError:
                    digest = None
                if digest:
                    consumer_spec["data"] = self._opCreationDigestToDataObject(
                        digest, graph
                    )
        return data_object

    def _opCreationDigestToDataObject(self, op_creation_digest, graph):
        if op_creation_digest is None:
            return None
        json_object = op_creation_digest.to_json()
        del json_object["graph_id"]
        json_object["graph_ids"] = self._getGraphStackIds(
            op_creation_digest.graph_id
        )
        # TODO(cais): "num_outputs" should be populated in to_json() instead.
        json_object["num_outputs"] = op_creation_digest.num_outputs
        del json_object["input_names"]

        json_object["inputs"] = []
        for input_tensor_name in op_creation_digest.input_names or []:
            input_op_name, output_slot = parse_tensor_name(input_tensor_name)
            json_object["inputs"].append(
                {"op_name": input_op_name, "output_slot": output_slot}
            )
        json_object["consumers"] = []
        for _ in range(json_object["num_outputs"]):
            json_object["consumers"].append([])
        for src_slot, consumer_op_name, dst_slot in graph.get_op_consumers(
            json_object["op_name"]
        ):
            json_object["consumers"][src_slot].append(
                {"op_name": consumer_op_name, "input_slot": dst_slot}
            )
        return json_object

    def _getGraphStackIds(self, graph_id):
        """Retrieve the IDs of all outer graphs of a graph.

        Args:
          graph_id: Id of the graph being queried with respect to its outer
            graphs context.

        Returns:
          A list of graph_ids, ordered from outermost to innermost, including
            the input `graph_id` argument as the last item.
        """
        graph_ids = [graph_id]
        graph = self._reader.graph_by_id(graph_id)
        while graph.outer_graph_id:
            graph_ids.insert(0, graph.outer_graph_id)
            graph = self._reader.graph_by_id(graph.outer_graph_id)
        return graph_ids

    def SourceFileList(self, run):
        runs = self.Runs()
        if run not in runs:
            return None
        return self._reader.source_file_list()

    def SourceLines(self, run, index):
        runs = self.Runs()
        if run not in runs:
            return None
        try:
            host_name, file_path = self._reader.source_file_list()[index]
        except IndexError:
            raise errors.NotFoundError(
                "There is no source-code file at index %d" % index
            )
        return {
            "host_name": host_name,
            "file_path": file_path,
            "lines": self._reader.source_lines(host_name, file_path),
        }

    def StackFrames(self, run, stack_frame_ids):
        runs = self.Runs()
        if run not in runs:
            return None
        stack_frames = []
        for stack_frame_id in stack_frame_ids:
            if stack_frame_id not in self._reader._stack_frame_by_id:
                raise errors.NotFoundError(
                    "Cannot find stack frame with ID %s" % stack_frame_id
                )
            # TODO(cais): Use public method (`stack_frame_by_id()`) when
            # available.
            # pylint: disable=protected-access
            stack_frames.append(self._reader._stack_frame_by_id[stack_frame_id])
            # pylint: enable=protected-access
        return {"stack_frames": stack_frames}
