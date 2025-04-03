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

from typing import Optional, Sequence, Tuple

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
from tensorboard.plugins.distribution import compressor
from tensorboard.util import tb_logging


logger = tb_logging.get_logger()


@dataclasses.dataclass(frozen=True)
class ScalarEvent:
    """Contains information of a scalar event.

    Attributes:
      wall_time: Timestamp of the event in seconds.
      step: Global step of the event.
      value: A float or int value of the scalar.
    """

    wall_time: float
    step: int
    value: float


@dataclasses.dataclass(frozen=True)
class CompressedHistogramEvent:
    """Contains information of a compressed histogram event.

    Attributes:
      wall_time: Timestamp of the event in seconds.
      step: Global step of the event.
      compressed_histogram_values: A sequence of tuples of basis points and
        associated values in a compressed histogram.
    """

    wall_time: float
    step: int
    compressed_histogram_values: Sequence[Tuple[float, float]]


@dataclasses.dataclass(frozen=True)
class HistogramValue:
    """Holds the information of the histogram values.

    Attributes:
      min: A float or int min value.
      max: A float or int max value.
      num: Total number of values.
      sum: Sum of all values.
      sum_squares: Sum of squares for all values.
      bucket_limit: Upper values per bucket.
      bucket: Numbers of values per bucket.
    """

    min: float
    max: float
    num: int
    sum: float
    sum_squares: float
    bucket_limit: Sequence[float]
    bucket: Sequence[int]


@dataclasses.dataclass(frozen=True)
class HistogramEvent:
    """Contains information of a histogram event.

    Attributes:
      wall_time: Timestamp of the event in seconds.
      step: Global step of the event.
      histogram_value: Information of the histogram values.
    """

    wall_time: float
    step: int
    histogram_value: HistogramValue


@dataclasses.dataclass(frozen=True)
class ImageEvent:
    """Contains information of an image event.

    Attributes:
      wall_time: Timestamp of the event in seconds.
      step: Global step of the event.
      encoded_image_string: Image content encoded in bytes.
      width: Width of the image.
      height: Height of the image.
    """

    wall_time: float
    step: int
    encoded_image_string: bytes
    width: int
    height: int


@dataclasses.dataclass(frozen=True)
class AudioEvent:
    """Contains information of an audio event.

    Attributes:
      wall_time: Timestamp of the event in seconds.
      step: Global step of the event.
      encoded_audio_string: Audio content encoded in bytes.
      content_type: A string describes the type of the audio content.
      sample_rate: Sample rate of the audio in Hz. Must be positive.
      length_frames: Length of the audio in frames (samples per channel).
    """

    wall_time: float
    step: int
    encoded_audio_string: bytes
    content_type: str
    sample_rate: float
    length_frames: int


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


## Different types of summary events handled by the event_accumulator
SUMMARY_TYPES = {
    "simple_value": "_ProcessScalar",
    "histo": "_ProcessHistogram",
    "image": "_ProcessImage",
    "audio": "_ProcessAudio",
    "tensor": "_ProcessTensor",
}

# Legacy aliases
COMPRESSED_HISTOGRAMS = tag_types.COMPRESSED_HISTOGRAMS
HISTOGRAMS = tag_types.HISTOGRAMS
IMAGES = tag_types.IMAGES
AUDIO = tag_types.AUDIO
SCALARS = tag_types.SCALARS
TENSORS = tag_types.TENSORS
GRAPH = tag_types.GRAPH
META_GRAPH = tag_types.META_GRAPH
RUN_METADATA = tag_types.RUN_METADATA

## Normal CDF for std_devs: (-Inf, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, Inf)
## naturally gives bands around median of width 1 std dev, 2 std dev, 3 std dev,
## and then the long tail.
NORMAL_HISTOGRAM_BPS = (0, 668, 1587, 3085, 5000, 6915, 8413, 9332, 10000)

DEFAULT_SIZE_GUIDANCE = {
    COMPRESSED_HISTOGRAMS: 500,
    IMAGES: 4,
    AUDIO: 4,
    SCALARS: 10000,
    HISTOGRAMS: 1,
    TENSORS: 10,
}

STORE_EVERYTHING_SIZE_GUIDANCE = {
    COMPRESSED_HISTOGRAMS: 0,
    IMAGES: 0,
    AUDIO: 0,
    SCALARS: 0,
    HISTOGRAMS: 0,
    TENSORS: 0,
}


class EventAccumulator:
    """An `EventAccumulator` takes an event generator, and accumulates the
    values.

    The `EventAccumulator` is intended to provide a convenient Python interface
    for loading Event data written during a TensorFlow run. TensorFlow writes out
    `Event` protobuf objects, which have a timestamp and step number, and often
    contain a `Summary`. Summaries can have different kinds of data like an image,
    a scalar value, or a histogram. The Summaries also have a tag, which we use to
    organize logically related data. The `EventAccumulator` supports retrieving
    the `Event` and `Summary` data by its tag.

    Calling `Tags()` gets a map from `tagType` (e.g. `'images'`,
    `'compressedHistograms'`, `'scalars'`, etc) to the associated tags for those
    data types. Then, various functional endpoints (eg
    `Accumulator.Scalars(tag)`) allow for the retrieval of all data
    associated with that tag.

    The `Reload()` method synchronously loads all of the data written so far.

    Histograms, audio, and images are very large, so storing all of them is not
    recommended.

    Fields:
      audios: A reservoir.Reservoir of audio summaries.
      compressed_histograms: A reservoir.Reservoir of compressed
          histogram summaries.
      histograms: A reservoir.Reservoir of histogram summaries.
      images: A reservoir.Reservoir of image summaries.
      most_recent_step: Step of last Event proto added. This should only
          be accessed from the thread that calls Reload. This is -1 if
          nothing has been loaded yet.
      most_recent_wall_time: Timestamp of last Event proto added. This is
          a float containing seconds from the UNIX epoch, or -1 if
          nothing has been loaded yet. This should only be accessed from
          the thread that calls Reload.
      path: A file path to a directory containing tf events files, or a single
          tf events file. The accumulator will load events from this path.
      scalars: A reservoir.Reservoir of scalar summaries.
      tensors: A reservoir.Reservoir of tensor summaries.

    @@Tensors
    """

    def __init__(
        self,
        path,
        size_guidance=None,
        compression_bps=NORMAL_HISTOGRAM_BPS,
        purge_orphaned_data=True,
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
          compression_bps: Information on how the `EventAccumulator` should compress
            histogram data for the `CompressedHistograms` tag (for details see
            `ProcessCompressedHistogram`).
          purge_orphaned_data: Whether to discard any events that were "orphaned" by
            a TensorFlow restart.
        """
        size_guidance = size_guidance or DEFAULT_SIZE_GUIDANCE
        sizes = {}
        for key in DEFAULT_SIZE_GUIDANCE:
            if key in size_guidance:
                sizes[key] = size_guidance[key]
            else:
                sizes[key] = DEFAULT_SIZE_GUIDANCE[key]

        self._first_event_timestamp = None
        self.scalars = reservoir.Reservoir(size=sizes[SCALARS])

        self._graph = None
        self._graph_from_metagraph = False
        self._meta_graph = None
        self._tagged_metadata = {}
        self.summary_metadata = {}
        self.histograms = reservoir.Reservoir(size=sizes[HISTOGRAMS])
        self.compressed_histograms = reservoir.Reservoir(
            size=sizes[COMPRESSED_HISTOGRAMS], always_keep_last=False
        )
        self.images = reservoir.Reservoir(size=sizes[IMAGES])
        self.audios = reservoir.Reservoir(size=sizes[AUDIO])
        self.tensors = reservoir.Reservoir(size=sizes[TENSORS])

        # Keep a mapping from plugin name to a dict mapping from tag to plugin data
        # content obtained from the SummaryMetadata (metadata field of Value) for
        # that plugin (This is not the entire SummaryMetadata proto - only the
        # content for that plugin). The SummaryWriter only keeps the content on the
        # first event encountered per tag, so we must store that first instance of
        # content for each tag.
        self._plugin_to_tag_to_content = collections.defaultdict(dict)

        self._generator_mutex = threading.Lock()
        self.path = path
        self._generator = _GeneratorFromPath(path)

        self._compression_bps = compression_bps
        self.purge_orphaned_data = purge_orphaned_data

        self.most_recent_step = -1
        self.most_recent_wall_time = -1
        self.file_version = None

        # Name of the source writer that writes the event.
        self._source_writer = None

        # The attributes that get built up by the accumulator
        self.accumulated_attrs = (
            "scalars",
            "histograms",
            "compressed_histograms",
            "images",
            "audios",
        )
        self._tensor_summaries = {}

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
        if plugin_name not in self._plugin_to_tag_to_content:
            raise KeyError("Plugin %r could not be found." % plugin_name)
        return self._plugin_to_tag_to_content[plugin_name]

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

                for summary_type, summary_func in SUMMARY_TYPES.items():
                    if value.HasField(summary_type):
                        datum = getattr(value, summary_type)
                        tag = value.tag
                        if summary_type == "tensor" and not tag:
                            # This tensor summary was created using the old method that used
                            # plugin assets. We must still continue to support it.
                            tag = value.node_name
                        getattr(self, summary_func)(
                            tag, event.wall_time, event.step, datum
                        )

    def Tags(self):
        """Return all tags found in the value stream.

        Returns:
          A `{tagType: ['list', 'of', 'tags']}` dictionary.
        """
        return {
            IMAGES: self.images.Keys(),
            AUDIO: self.audios.Keys(),
            HISTOGRAMS: self.histograms.Keys(),
            SCALARS: self.scalars.Keys(),
            COMPRESSED_HISTOGRAMS: self.compressed_histograms.Keys(),
            TENSORS: self.tensors.Keys(),
            # Use a heuristic: if the metagraph is available, but
            # graph is not, then we assume the metagraph contains the graph.
            GRAPH: self._graph is not None,
            META_GRAPH: self._meta_graph is not None,
            RUN_METADATA: list(self._tagged_metadata.keys()),
        }

    def Scalars(self, tag):
        """Given a summary tag, return all associated `ScalarEvent`s.

        Args:
          tag: A string tag associated with the events.

        Raises:
          KeyError: If the tag is not found.

        Returns:
          An array of `ScalarEvent`s.
        """
        return self.scalars.Items(tag)

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

    def Histograms(self, tag):
        """Given a summary tag, return all associated histograms.

        Args:
          tag: A string tag associated with the events.

        Raises:
          KeyError: If the tag is not found.

        Returns:
          An array of `HistogramEvent`s.
        """
        return self.histograms.Items(tag)

    def CompressedHistograms(self, tag):
        """Given a summary tag, return all associated compressed histograms.

        Args:
          tag: A string tag associated with the events.

        Raises:
          KeyError: If the tag is not found.

        Returns:
          An array of `CompressedHistogramEvent`s.
        """
        return self.compressed_histograms.Items(tag)

    def Images(self, tag):
        """Given a summary tag, return all associated images.

        Args:
          tag: A string tag associated with the events.

        Raises:
          KeyError: If the tag is not found.

        Returns:
          An array of `ImageEvent`s.
        """
        return self.images.Items(tag)

    def Audio(self, tag):
        """Given a summary tag, return all associated audio.

        Args:
          tag: A string tag associated with the events.

        Raises:
          KeyError: If the tag is not found.

        Returns:
          An array of `AudioEvent`s.
        """
        return self.audios.Items(tag)

    def Tensors(self, tag):
        """Given a summary tag, return all associated tensors.

        Args:
          tag: A string tag associated with the events.

        Raises:
          KeyError: If the tag is not found.

        Returns:
          An array of `TensorEvent`s.
        """
        return self.tensors.Items(tag)

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

    def _CheckForRestartAndMaybePurge(self, event):
        """Check and discard expired events using SessionLog.START.

        Check for a SessionLog.START event and purge all previously seen events
        with larger steps, because they are out of date. Because of supervisor
        threading, it is possible that this logic will cause the first few event
        messages to be discarded since supervisor threading does not guarantee
        that the START message is deterministically written first.

        This method is preferred over _CheckForOutOfOrderStepAndMaybePurge which
        can inadvertently discard events due to supervisor threading.

        Args:
          event: The event to use as reference. If the event is a START event, all
            previously seen events with a greater event.step will be purged.
        """
        if (
            event.HasField("session_log")
            and event.session_log.status == event_pb2.SessionLog.START
        ):
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
        else:
            self.most_recent_step = event.step
            self.most_recent_wall_time = event.wall_time

    def _ConvertHistogramProtoToPopo(self, histo):
        """Converts histogram proto to Python object."""
        return HistogramValue(
            min=histo.min,
            max=histo.max,
            num=histo.num,
            sum=histo.sum,
            sum_squares=histo.sum_squares,
            bucket_limit=list(histo.bucket_limit),
            bucket=list(histo.bucket),
        )

    def _ProcessHistogram(self, tag, wall_time, step, histo):
        """Processes a proto histogram by adding it to accumulated state."""
        histo = self._ConvertHistogramProtoToPopo(histo)
        histo_ev = HistogramEvent(wall_time, step, histo)
        self.histograms.AddItem(tag, histo_ev)
        self.compressed_histograms.AddItem(
            tag, histo_ev, self._CompressHistogram
        )

    def _CompressHistogram(self, histo_ev):
        """Callback for _ProcessHistogram."""
        return CompressedHistogramEvent(
            histo_ev.wall_time,
            histo_ev.step,
            compressor.compress_histogram_proto(
                histo_ev.histogram_value, self._compression_bps
            ),
        )

    def _ProcessImage(self, tag, wall_time, step, image):
        """Processes an image by adding it to accumulated state."""
        event = ImageEvent(
            wall_time=wall_time,
            step=step,
            encoded_image_string=image.encoded_image_string,
            width=image.width,
            height=image.height,
        )
        self.images.AddItem(tag, event)

    def _ProcessAudio(self, tag, wall_time, step, audio):
        """Processes a audio by adding it to accumulated state."""
        event = AudioEvent(
            wall_time=wall_time,
            step=step,
            encoded_audio_string=audio.encoded_audio_string,
            content_type=audio.content_type,
            sample_rate=audio.sample_rate,
            length_frames=audio.length_frames,
        )
        self.audios.AddItem(tag, event)

    def _ProcessScalar(self, tag, wall_time, step, scalar):
        """Processes a simple value by adding it to accumulated state."""
        sv = ScalarEvent(wall_time=wall_time, step=step, value=scalar)
        self.scalars.AddItem(tag, sv)

    def _ProcessTensor(self, tag, wall_time, step, tensor):
        tv = TensorEvent(wall_time=wall_time, step=step, tensor_proto=tensor)
        self.tensors.AddItem(tag, tv)

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

        if by_tags:

            def _ExpiredPerTag(value):
                return [
                    getattr(self, x).FilterItems(_NotExpired, value.tag)
                    for x in self.accumulated_attrs
                ]

            expired_per_tags = [
                _ExpiredPerTag(value) for value in event.summary.value
            ]
            expired_per_type = [sum(x) for x in zip(*expired_per_tags)]
        else:
            expired_per_type = [
                getattr(self, x).FilterItems(_NotExpired)
                for x in self.accumulated_attrs
            ]

        if sum(expired_per_type) > 0:
            purge_msg = _GetPurgeMessage(
                self.most_recent_step,
                self.most_recent_wall_time,
                event.step,
                event.wall_time,
                *expired_per_type,
            )
            logger.warning(purge_msg)


def _GetPurgeMessage(
    most_recent_step,
    most_recent_wall_time,
    event_step,
    event_wall_time,
    num_expired_scalars,
    num_expired_histos,
    num_expired_comp_histos,
    num_expired_images,
    num_expired_audio,
):
    """Return the string message associated with TensorBoard purges."""
    return (
        "Detected out of order event.step likely caused by "
        "a TensorFlow restart. Purging expired events from Tensorboard"
        " display between the previous step: {} (timestamp: {}) and "
        "current step: {} (timestamp: {}). Removing {} scalars, {} "
        "histograms, {} compressed histograms, {} images, "
        "and {} audio."
    ).format(
        most_recent_step,
        most_recent_wall_time,
        event_step,
        event_wall_time,
        num_expired_scalars,
        num_expired_histos,
        num_expired_comp_histos,
        num_expired_images,
        num_expired_audio,
    )


def _GeneratorFromPath(path):
    """Create an event generator for file or directory at given path string."""
    if not path:
        raise ValueError("path must be a valid string")
    if io_wrapper.IsSummaryEventsFile(path):
        return event_file_loader.LegacyEventFileLoader(path)
    else:
        return directory_watcher.DirectoryWatcher(
            path,
            event_file_loader.LegacyEventFileLoader,
            io_wrapper.IsSummaryEventsFile,
        )
