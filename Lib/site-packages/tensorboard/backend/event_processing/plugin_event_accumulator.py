# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Takes a generator of values, and accumulates them for a frontend."""

import collections
import dataclasses
import threading

from typing import Optional

from tensorboard.backend.event_processing import directory_loader
from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.backend.event_processing import event_util
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.backend.event_processing import plugin_asset_util
from tensorboard.backend.event_processing import reservoir
from tensorboard.backend.event_processing import tag_types
from tensorboard.compat.proto import config_pb2
from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto import graph_pb2
from tensorboard.compat.proto import meta_graph_pb2
from tensorboard.compat.proto import tensor_pb2
from tensorboard.util import tb_logging


logger = tb_logging.get_logger()


# Legacy aliases
TENSORS = tag_types.TENSORS
GRAPH = tag_types.GRAPH
META_GRAPH = tag_types.META_GRAPH
RUN_METADATA = tag_types.RUN_METADATA

DEFAULT_SIZE_GUIDANCE = {
    TENSORS: 500,
}

STORE_EVERYTHING_SIZE_GUIDANCE = {
    TENSORS: 0,
}

_TENSOR_RESERVOIR_KEY = "."  # arbitrary


@dataclasses.dataclass(frozen=True)
class TensorEvent:
    """A tensor event.

    Attributes:
      wall_time: Timestamp of the event in seconds.
      step: Global step of the event.
      tensor_proto: A `TensorProto`.
    """

    wall_time: float
    step: int
    tensor_proto: tensor_pb2.TensorProto


class EventAccumulator:
    """An `EventAccumulator` takes an event generator, and accumulates the
    values.

    The `EventAccumulator` is intended to provide a convenient Python
    interface for loading Event data written during a TensorFlow run.
    TensorFlow writes out `Event` protobuf objects, which have a timestamp
    and step number, and often contain a `Summary`. Summaries can have
    different kinds of data stored as arbitrary tensors. The Summaries
    also have a tag, which we use to organize logically related data. The
    `EventAccumulator` supports retrieving the `Event` and `Summary` data
    by its tag.

    Calling `Tags()` gets a map from `tagType` (i.e., `tensors`) to the
    associated tags for those data types. Then, the functional endpoint
    (i.g., `Accumulator.Tensors(tag)`) allows for the retrieval of all
    data associated with that tag.

    The `Reload()` method synchronously loads all of the data written so far.

    Fields:
      most_recent_step: Step of last Event proto added. This should only
          be accessed from the thread that calls Reload. This is -1 if
          nothing has been loaded yet.
      most_recent_wall_time: Timestamp of last Event proto added. This is
          a float containing seconds from the UNIX epoch, or -1 if
          nothing has been loaded yet. This should only be accessed from
          the thread that calls Reload.
      path: A file path to a directory containing tf events files, or a single
          tf events file. The accumulator will load events from this path.
      tensors_by_tag: A dictionary mapping each tag name to a
        reservoir.Reservoir of tensor summaries. Each such reservoir will
        only use a single key, given by `_TENSOR_RESERVOIR_KEY`.

    @@Tensors
    """

    def __init__(
        self,
        path,
        size_guidance=None,
        tensor_size_guidance=None,
        purge_orphaned_data=True,
        event_file_active_filter=None,
        detect_file_replacement=None,
    ):
        """Construct the `EventAccumulator`.

        Args:
          path: A file path to a directory containing tf events files, or a single
            tf events file. The accumulator will load events from this path.
          size_guidance: Information on how much data the EventAccumulator should
            store in memory. The DEFAULT_SIZE_GUIDANCE tries not to store too much
            so as to avoid OOMing the client. The size_guidance should be a map
            from a `tagType` string to an integer representing the number of
            items to keep per tag for items of that `tagType`. If the size is 0,
            all events are stored.
          tensor_size_guidance: Like `size_guidance`, but allowing finer
            granularity for tensor summaries. Should be a map from the
            `plugin_name` field on the `PluginData` proto to an integer
            representing the number of items to keep per tag. Plugins for
            which there is no entry in this map will default to the value of
            `size_guidance[event_accumulator.TENSORS]`. Defaults to `{}`.
          purge_orphaned_data: Whether to discard any events that were "orphaned" by
            a TensorFlow restart.
          event_file_active_filter: Optional predicate for determining whether an
            event file latest load timestamp should be considered active. If passed,
            this will enable multifile directory loading.
          detect_file_replacement: Optional boolean; if True, event file loading
            will try to detect when a file has been replaced with a new version
            that contains additional data, by monitoring the file size.
        """
        size_guidance = dict(size_guidance or DEFAULT_SIZE_GUIDANCE)
        sizes = {}
        for key in DEFAULT_SIZE_GUIDANCE:
            if key in size_guidance:
                sizes[key] = size_guidance[key]
            else:
                sizes[key] = DEFAULT_SIZE_GUIDANCE[key]
        self._size_guidance = size_guidance
        self._tensor_size_guidance = dict(tensor_size_guidance or {})

        self._first_event_timestamp = None

        self._graph = None
        self._graph_from_metagraph = False
        self._meta_graph = None
        self._tagged_metadata = {}
        self.summary_metadata = {}
        self.tensors_by_tag = {}
        self._tensors_by_tag_lock = threading.Lock()

        # Keep a mapping from plugin name to a dict mapping from tag to plugin data
        # content obtained from the SummaryMetadata (metadata field of Value) for
        # that plugin (This is not the entire SummaryMetadata proto - only the
        # content for that plugin). The SummaryWriter only keeps the content on the
        # first event encountered per tag, so we must store that first instance of
        # content for each tag.
        self._plugin_to_tag_to_content = collections.defaultdict(dict)
        # Locks the dict `_plugin_to_tag_to_content` as well as the
        # dicts `_plugin_to_tag_to_content[p]` for each `p`.
        self._plugin_tag_lock = threading.Lock()

        self.path = path
        self._generator = _GeneratorFromPath(
            path, event_file_active_filter, detect_file_replacement
        )
        self._generator_mutex = threading.Lock()

        self.purge_orphaned_data = purge_orphaned_data
        self._seen_session_start = False

        self.most_recent_step = -1
        self.most_recent_wall_time = -1
        self.file_version = None

        # Name of the source writer that writes the event.
        self._source_writer = None

    def Reload(self):
        """Loads all events added since the last call to `Reload`.

        If `Reload` was never called, loads all events in the file.

        Returns:
          The `EventAccumulator`.
        """
        with self._generator_mutex:
            for event in self._generator.Load():
                self._ProcessEvent(event)
        return self

    def PluginAssets(self, plugin_name):
        """Return a list of all plugin assets for the given plugin.

        Args:
          plugin_name: The string name of a plugin to retrieve assets for.

        Returns:
          A list of string plugin asset names, or empty list if none are available.
          If the plugin was not registered, an empty list is returned.
        """
        return plugin_asset_util.ListAssets(self.path, plugin_name)

    def RetrievePluginAsset(self, plugin_name, asset_name):
        """Return the contents of a given plugin asset.

        Args:
          plugin_name: The string name of a plugin.
          asset_name: The string name of an asset.

        Returns:
          The string contents of the plugin asset.

        Raises:
          KeyError: If the asset is not available.
        """
        return plugin_asset_util.RetrieveAsset(
            self.path, plugin_name, asset_name
        )

    def FirstEventTimestamp(self):
        """Returns the timestamp in seconds of the first event.

        If the first event has been loaded (either by this method or by `Reload`,
        this returns immediately. Otherwise, it will load in the first event. Note
        that this means that calling `Reload` will cause this to block until
        `Reload` has finished.

        Returns:
          The timestamp in seconds of the first event that was loaded.

        Raises:
          ValueError: If no events have been loaded and there were no events found
          on disk.
        """
        if self._first_event_timestamp is not None:
            return self._first_event_timestamp
        with self._generator_mutex:
            try:
                event = next(self._generator.Load())
                self._ProcessEvent(event)
                return self._first_event_timestamp

            except StopIteration:
                raise ValueError("No event timestamp could be found")

    def GetSourceWriter(self) -> Optional[str]:
        """Returns the name of the event writer."""
        if self._source_writer is not None:
            return self._source_writer
        with self._generator_mutex:
            try:
                event = next(self._generator.Load())
                self._ProcessEvent(event)
                return self._source_writer
            except StopIteration:
                logger.info(
                    "End of file in %s, no source writer was found.", self.path
                )

    def PluginTagToContent(self, plugin_name):
        """Returns a dict mapping tags to content specific to that plugin.

        Args:
          plugin_name: The name of the plugin for which to fetch plugin-specific
            content.

        Raises:
          KeyError: if the plugin name is not found.

        Returns:
          A dict mapping tag names to bytestrings of plugin-specific content-- by
          convention, in the form of binary serialized protos.
        """
        with self._plugin_tag_lock:
            if plugin_name not in self._plugin_to_tag_to_content:
                raise KeyError("Plugin %r could not be found." % plugin_name)
            # Return a snapshot to avoid concurrent mutation and iteration issues.
            return dict(self._plugin_to_tag_to_content[plugin_name])

    def ActivePlugins(self):
        """Return a set of plugins with summary data.

        Returns:
          The distinct union of `plugin_data.plugin_name` fields from
          all the `SummaryMetadata` protos stored in this accumulator.
        """
        with self._plugin_tag_lock:
            return frozenset(self._plugin_to_tag_to_content)

    def SummaryMetadata(self, tag):
        """Given a summary tag name, return the associated metadata object.

        Args:
          tag: The name of a tag, as a string.

        Raises:
          KeyError: If the tag is not found.

        Returns:
          A `SummaryMetadata` protobuf.
        """
        return self.summary_metadata[tag]

    def AllSummaryMetadata(self):
        """Return summary metadata for all tags.

        Returns:
          A dict `d` such that `d[tag]` is a `SummaryMetadata` proto for
          the keyed tag.
        """
        return dict(self.summary_metadata)

    def _ProcessEvent(self, event):
        """Called whenever an event is loaded."""
        if self._first_event_timestamp is None:
            self._first_event_timestamp = event.wall_time

        if event.HasField("source_metadata"):
            new_source_writer = event_util.GetSourceWriter(
                event.source_metadata
            )
            if self._source_writer and self._source_writer != new_source_writer:
                logger.info(
                    (
                        "Found new source writer for event.proto. "
                        "Old: {0}, New: {1}"
                    ).format(self._source_writer, new_source_writer)
                )
            self._source_writer = new_source_writer

        if event.HasField("file_version"):
            new_file_version = event_util.ParseFileVersion(event.file_version)
            if self.file_version and self.file_version != new_file_version:
                ## This should not happen.
                logger.warning(
                    (
                        "Found new file_version for event.proto. This will "
                        "affect purging logic for TensorFlow restarts. "
                        "Old: {0} New: {1}"
                    ).format(self.file_version, new_file_version)
                )
            self.file_version = new_file_version

        self._MaybePurgeOrphanedData(event)

        ## Process the event.
        # GraphDef and MetaGraphDef are handled in a special way:
        # If no graph_def Event is available, but a meta_graph_def is, and it
        # contains a graph_def, then use the meta_graph_def.graph_def as our graph.
        # If a graph_def Event is available, always prefer it to the graph_def
        # inside the meta_graph_def.
        if event.HasField("graph_def"):
            if self._graph is not None:
                logger.warning(
                    (
                        "Found more than one graph event per run, or there was "
                        "a metagraph containing a graph_def, as well as one or "
                        "more graph events.  Overwriting the graph with the "
                        "newest event."
                    )
                )
            self._graph = event.graph_def
            self._graph_from_metagraph = False
        elif event.HasField("meta_graph_def"):
            if self._meta_graph is not None:
                logger.warning(
                    (
                        "Found more than one metagraph event per run. "
                        "Overwriting the metagraph with the newest event."
                    )
                )
            self._meta_graph = event.meta_graph_def
            if self._graph is None or self._graph_from_metagraph:
                # We may have a graph_def in the metagraph.  If so, and no
                # graph_def is directly available, use this one instead.
                meta_graph = meta_graph_pb2.MetaGraphDef()
                meta_graph.ParseFromString(self._meta_graph)
                if meta_graph.graph_def:
                    if self._graph is not None:
                        logger.warning(
                            (
                                "Found multiple metagraphs containing graph_defs,"
                                "but did not find any graph events.  Overwriting the "
                                "graph with the newest metagraph version."
                            )
                        )
                    self._graph_from_metagraph = True
                    self._graph = meta_graph.graph_def.SerializeToString()
        elif event.HasField("tagged_run_metadata"):
            tag = event.tagged_run_metadata.tag
            if tag in self._tagged_metadata:
                logger.warning(
                    'Found more than one "run metadata" event with tag '
                    + tag
                    + ". Overwriting it with the newest event."
                )
            self._tagged_metadata[tag] = event.tagged_run_metadata.run_metadata
        elif event.HasField("summary"):
            for value in event.summary.value:
                if value.HasField("metadata"):
                    tag = value.tag
                    # We only store the first instance of the metadata. This check
                    # is important: the `FileWriter` does strip metadata from all
                    # values except the first one per each tag, but a new
                    # `FileWriter` is created every time a training job stops and
                    # restarts. Hence, we must also ignore non-initial metadata in
                    # this logic.
                    if tag not in self.summary_metadata:
                        self.summary_metadata[tag] = value.metadata
                        plugin_data = value.metadata.plugin_data
                        if plugin_data.plugin_name:
                            with self._plugin_tag_lock:
                                self._plugin_to_tag_to_content[
                                    plugin_data.plugin_name
                                ][tag] = plugin_data.content
                        else:
                            logger.warning(
                                (
                                    "This summary with tag %r is oddly not associated with a "
                                    "plugin."
                                ),
                                tag,
                            )

                if value.HasField("tensor"):
                    datum = value.tensor
                    tag = value.tag
                    if not tag:
                        # This tensor summary was created using the old method that used
                        # plugin assets. We must still continue to support it.
                        tag = value.node_name
                    self._ProcessTensor(tag, event.wall_time, event.step, datum)

    def Tags(self):
        """Return all tags found in the value stream.

        Returns:
          A `{tagType: ['list', 'of', 'tags']}` dictionary.
        """
        return {
            TENSORS: list(self.tensors_by_tag.keys()),
            # Use a heuristic: if the metagraph is available, but
            # graph is not, then we assume the metagraph contains the graph.
            GRAPH: self._graph is not None,
            META_GRAPH: self._meta_graph is not None,
            RUN_METADATA: list(self._tagged_metadata.keys()),
        }

    def Graph(self):
        """Return the graph definition, if there is one.

        If the graph is stored directly, return that.  If no graph is stored
        directly but a metagraph is stored containing a graph, return that.

        Raises:
          ValueError: If there is no graph for this run.

        Returns:
          The `graph_def` proto.
        """
        graph = graph_pb2.GraphDef()
        if self._graph is not None:
            graph.ParseFromString(self._graph)
            return graph
        raise ValueError("There is no graph in this EventAccumulator")

    def SerializedGraph(self):
        """Return the graph definition in serialized form, if there is one."""
        return self._graph

    def MetaGraph(self):
        """Return the metagraph definition, if there is one.

        Raises:
          ValueError: If there is no metagraph for this run.

        Returns:
          The `meta_graph_def` proto.
        """
        if self._meta_graph is None:
            raise ValueError("There is no metagraph in this EventAccumulator")
        meta_graph = meta_graph_pb2.MetaGraphDef()
        meta_graph.ParseFromString(self._meta_graph)
        return meta_graph

    def RunMetadata(self, tag):
        """Given a tag, return the associated session.run() metadata.

        Args:
          tag: A string tag associated with the event.

        Raises:
          ValueError: If the tag is not found.

        Returns:
          The metadata in form of `RunMetadata` proto.
        """
        if tag not in self._tagged_metadata:
            raise ValueError("There is no run metadata with this tag name")

        run_metadata = config_pb2.RunMetadata()
        run_metadata.ParseFromString(self._tagged_metadata[tag])
        return run_metadata

    def Tensors(self, tag):
        """Given a summary tag, return all associated tensors.

        Args:
          tag: A string tag associated with the events.

        Raises:
          KeyError: If the tag is not found.

        Returns:
          An array of `TensorEvent`s.
        """
        return self.tensors_by_tag[tag].Items(_TENSOR_RESERVOIR_KEY)

    def _MaybePurgeOrphanedData(self, event):
        """Maybe purge orphaned data due to a TensorFlow crash.

        When TensorFlow crashes at step T+O and restarts at step T, any events
        written after step T are now "orphaned" and will be at best misleading if
        they are included in TensorBoard.

        This logic attempts to determine if there is orphaned data, and purge it
        if it is found.

        Args:
          event: The event to use as a reference, to determine if a purge is needed.
        """
        if not self.purge_orphaned_data:
            return
        ## Check if the event happened after a crash, and purge expired tags.
        if self.file_version and self.file_version >= 2:
            ## If the file_version is recent enough, use the SessionLog enum
            ## to check for restarts.
            self._CheckForRestartAndMaybePurge(event)
        else:
            ## If there is no file version, default to old logic of checking for
            ## out of order steps.
            self._CheckForOutOfOrderStepAndMaybePurge(event)
        # After checking, update the most recent summary step and wall time.
        if event.HasField("summary"):
            self.most_recent_step = event.step
            self.most_recent_wall_time = event.wall_time

    def _CheckForRestartAndMaybePurge(self, event):
        """Check and discard expired events using SessionLog.START.

        The first SessionLog.START event in a run indicates the start of a
        supervisor session. Subsequent SessionLog.START events indicate a
        *restart*, which may need to preempt old events. This method checks
        for a session restart event and purges all previously seen events whose
        step is larger than or equal to this event's step.

        Because of supervisor threading, it is possible that this logic will
        cause the first few event messages to be discarded since supervisor
        threading does not guarantee that the START message is deterministically
        written first.

        This method is preferred over _CheckForOutOfOrderStepAndMaybePurge which
        can inadvertently discard events due to supervisor threading.

        Args:
          event: The event to use as reference. If the event is a START event, all
            previously seen events with a greater event.step will be purged.
        """
        if event.session_log.status != event_pb2.SessionLog.START:
            return
        if not self._seen_session_start:
            # Initial start event: does not indicate a restart.
            self._seen_session_start = True
            return
        self._Purge(event, by_tags=False)

    def _CheckForOutOfOrderStepAndMaybePurge(self, event):
        """Check for out-of-order event.step and discard expired events for
        tags.

        Check if the event is out of order relative to the global most recent step.
        If it is, purge outdated summaries for tags that the event contains.

        Args:
          event: The event to use as reference. If the event is out-of-order, all
            events with the same tags, but with a greater event.step will be purged.
        """
        if event.step < self.most_recent_step and event.HasField("summary"):
            self._Purge(event, by_tags=True)

    def _ProcessTensor(self, tag, wall_time, step, tensor):
        tv = TensorEvent(wall_time=wall_time, step=step, tensor_proto=tensor)
        with self._tensors_by_tag_lock:
            if tag not in self.tensors_by_tag:
                reservoir_size = self._GetTensorReservoirSize(tag)
                self.tensors_by_tag[tag] = reservoir.Reservoir(reservoir_size)
        self.tensors_by_tag[tag].AddItem(_TENSOR_RESERVOIR_KEY, tv)

    def _GetTensorReservoirSize(self, tag):
        default = self._size_guidance[TENSORS]
        summary_metadata = self.summary_metadata.get(tag)
        if summary_metadata is None:
            return default
        return self._tensor_size_guidance.get(
            summary_metadata.plugin_data.plugin_name, default
        )

    def _Purge(self, event, by_tags):
        """Purge all events that have occurred after the given event.step.

        If by_tags is True, purge all events that occurred after the given
        event.step, but only for the tags that the event has. Non-sequential
        event.steps suggest that a TensorFlow restart occurred, and we discard
        the out-of-order events to display a consistent view in TensorBoard.

        Discarding by tags is the safer method, when we are unsure whether a restart
        has occurred, given that threading in supervisor can cause events of
        different tags to arrive with unsynchronized step values.

        If by_tags is False, then purge all events with event.step greater than the
        given event.step. This can be used when we are certain that a TensorFlow
        restart has occurred and these events can be discarded.

        Args:
          event: The event to use as reference for the purge. All events with
            the same tags, but with a greater event.step will be purged.
          by_tags: Bool to dictate whether to discard all out-of-order events or
            only those that are associated with the given reference event.
        """
        ## Keep data in reservoirs that has a step less than event.step
        _NotExpired = lambda x: x.step < event.step

        num_expired = 0
        if by_tags:
            for value in event.summary.value:
                if value.tag in self.tensors_by_tag:
                    tag_reservoir = self.tensors_by_tag[value.tag]
                    num_expired += tag_reservoir.FilterItems(
                        _NotExpired, _TENSOR_RESERVOIR_KEY
                    )
        else:
            for tag_reservoir in self.tensors_by_tag.values():
                num_expired += tag_reservoir.FilterItems(
                    _NotExpired, _TENSOR_RESERVOIR_KEY
                )
        if num_expired > 0:
            purge_msg = _GetPurgeMessage(
                self.most_recent_step,
                self.most_recent_wall_time,
                event.step,
                event.wall_time,
                num_expired,
            )
            logger.warning(purge_msg)


def _GetPurgeMessage(
    most_recent_step,
    most_recent_wall_time,
    event_step,
    event_wall_time,
    num_expired,
):
    """Return the string message associated with TensorBoard purges."""
    return (
        "Detected out of order event.step likely caused by a TensorFlow "
        "restart. Purging {} expired tensor events from Tensorboard display "
        "between the previous step: {} (timestamp: {}) and current step: {} "
        "(timestamp: {})."
    ).format(
        num_expired,
        most_recent_step,
        most_recent_wall_time,
        event_step,
        event_wall_time,
    )


def _GeneratorFromPath(
    path, event_file_active_filter=None, detect_file_replacement=None
):
    """Create an event generator for file or directory at given path string."""
    if not path:
        raise ValueError("path must be a valid string")
    if io_wrapper.IsSummaryEventsFile(path):
        return event_file_loader.EventFileLoader(path, detect_file_replacement)
    elif event_file_active_filter:
        loader_factory = (
            lambda path: event_file_loader.TimestampedEventFileLoader(
                path, detect_file_replacement
            )
        )
        return directory_loader.DirectoryLoader(
            path,
            loader_factory,
            path_filter=io_wrapper.IsSummaryEventsFile,
            active_filter=event_file_active_filter,
        )
    else:
        loader_factory = lambda path: event_file_loader.EventFileLoader(
            path, detect_file_replacement
        )
        return directory_watcher.DirectoryWatcher(
            path,
            loader_factory,
            io_wrapper.IsSummaryEventsFile,
        )
