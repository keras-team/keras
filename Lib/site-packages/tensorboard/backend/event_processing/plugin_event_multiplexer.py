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
"""Provides an interface for working with multiple event files."""


import os
import queue
import threading

from typing import Optional

from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import (
    plugin_event_accumulator as event_accumulator,
)
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.util import tb_logging


logger = tb_logging.get_logger()


class EventMultiplexer:
    """An `EventMultiplexer` manages access to multiple `EventAccumulator`s.

    Each `EventAccumulator` is associated with a `run`, which is a self-contained
    TensorFlow execution. The `EventMultiplexer` provides methods for extracting
    information about events from multiple `run`s.

    Example usage for loading specific runs from files:

    ```python
    x = EventMultiplexer({'run1': 'path/to/run1', 'run2': 'path/to/run2'})
    x.Reload()
    ```

    Example usage for loading a directory where each subdirectory is a run

    ```python
    (eg:) /parent/directory/path/
          /parent/directory/path/run1/
          /parent/directory/path/run1/events.out.tfevents.1001
          /parent/directory/path/run1/events.out.tfevents.1002

          /parent/directory/path/run2/
          /parent/directory/path/run2/events.out.tfevents.9232

          /parent/directory/path/run3/
          /parent/directory/path/run3/events.out.tfevents.9232
    x = EventMultiplexer().AddRunsFromDirectory('/parent/directory/path')
    (which is equivalent to:)
    x = EventMultiplexer({'run1': '/parent/directory/path/run1', 'run2':...}
    ```

    If you would like to watch `/parent/directory/path`, wait for it to be created
      (if necessary) and then periodically pick up new runs, use
      `AutoloadingMultiplexer`
    @@Tensors
    """

    def __init__(
        self,
        run_path_map=None,
        size_guidance=None,
        tensor_size_guidance=None,
        purge_orphaned_data=True,
        max_reload_threads=None,
        event_file_active_filter=None,
        detect_file_replacement=None,
    ):
        """Constructor for the `EventMultiplexer`.

        Args:
          run_path_map: Dict `{run: path}` which specifies the
            name of a run, and the path to find the associated events. If it is
            None, then the EventMultiplexer initializes without any runs.
          size_guidance: A dictionary mapping from `tagType` to the number of items
            to store for each tag of that type. See
            `event_accumulator.EventAccumulator` for details.
          tensor_size_guidance: A dictionary mapping from `plugin_name` to
            the number of items to store for each tag of that type. See
            `event_accumulator.EventAccumulator` for details.
          purge_orphaned_data: Whether to discard any events that were "orphaned" by
            a TensorFlow restart.
          max_reload_threads: The max number of threads that TensorBoard can use
            to reload runs. Each thread reloads one run at a time. If not provided,
            reloads runs serially (one after another).
          event_file_active_filter: Optional predicate for determining whether an
            event file latest load timestamp should be considered active. If passed,
            this will enable multifile directory loading.
          detect_file_replacement: Optional boolean; if True, event file loading
            will try to detect when a file has been replaced with a new version
            that contains additional data, by monitoring the file size.
        """
        logger.info("Event Multiplexer initializing.")
        self._accumulators_mutex = threading.Lock()
        self._accumulators = {}
        self._paths = {}
        self._reload_called = False
        self._size_guidance = (
            size_guidance or event_accumulator.DEFAULT_SIZE_GUIDANCE
        )
        self._tensor_size_guidance = tensor_size_guidance
        self.purge_orphaned_data = purge_orphaned_data
        self._max_reload_threads = max_reload_threads or 1
        self._event_file_active_filter = event_file_active_filter
        self._detect_file_replacement = detect_file_replacement
        if run_path_map is not None:
            logger.info(
                "Event Multplexer doing initialization load for %s",
                run_path_map,
            )
            for run, path in run_path_map.items():
                self.AddRun(path, run)
        logger.info("Event Multiplexer done initializing")

    def AddRun(self, path, name=None):
        """Add a run to the multiplexer.

        If the name is not specified, it is the same as the path.

        If a run by that name exists, and we are already watching the right path,
          do nothing. If we are watching a different path, replace the event
          accumulator.

        If `Reload` has been called, it will `Reload` the newly created
        accumulators.

        Args:
          path: Path to the event files (or event directory) for given run.
          name: Name of the run to add. If not provided, is set to path.

        Returns:
          The `EventMultiplexer`.
        """
        name = name or path
        accumulator = None
        with self._accumulators_mutex:
            if name not in self._accumulators or self._paths[name] != path:
                if name in self._paths and self._paths[name] != path:
                    # TODO(@decentralion) - Make it impossible to overwrite an old path
                    # with a new path (just give the new path a distinct name)
                    logger.warning(
                        "Conflict for name %s: old path %s, new path %s",
                        name,
                        self._paths[name],
                        path,
                    )
                logger.info("Constructing EventAccumulator for %s", path)
                accumulator = event_accumulator.EventAccumulator(
                    path,
                    size_guidance=self._size_guidance,
                    tensor_size_guidance=self._tensor_size_guidance,
                    purge_orphaned_data=self.purge_orphaned_data,
                    event_file_active_filter=self._event_file_active_filter,
                    detect_file_replacement=self._detect_file_replacement,
                )
                self._accumulators[name] = accumulator
                self._paths[name] = path
        if accumulator:
            if self._reload_called:
                accumulator.Reload()
        return self

    def AddRunsFromDirectory(self, path, name=None):
        """Load runs from a directory; recursively walks subdirectories.

        If path doesn't exist, no-op. This ensures that it is safe to call
          `AddRunsFromDirectory` multiple times, even before the directory is made.

        If path is a directory, load event files in the directory (if any exist) and
          recursively call AddRunsFromDirectory on any subdirectories. This mean you
          can call AddRunsFromDirectory at the root of a tree of event logs and
          TensorBoard will load them all.

        If the `EventMultiplexer` is already loaded this will cause
        the newly created accumulators to `Reload()`.
        Args:
          path: A string path to a directory to load runs from.
          name: Optionally, what name to apply to the runs. If name is provided
            and the directory contains run subdirectories, the name of each subrun
            is the concatenation of the parent name and the subdirectory name. If
            name is provided and the directory contains event files, then a run
            is added called "name" and with the events from the path.

        Raises:
          ValueError: If the path exists and isn't a directory.

        Returns:
          The `EventMultiplexer`.
        """
        path = os.path.expanduser(path)
        logger.info("Starting AddRunsFromDirectory: %s", path)
        for subdir in io_wrapper.GetLogdirSubdirectories(path):
            logger.info("Adding run from directory %s", subdir)
            rpath = os.path.relpath(subdir, path)
            subname = os.path.join(name, rpath) if name else rpath
            self.AddRun(subdir, name=subname)
        logger.info("Done with AddRunsFromDirectory: %s", path)
        return self

    def Reload(self):
        """Call `Reload` on every `EventAccumulator`."""
        logger.info("Beginning EventMultiplexer.Reload()")
        self._reload_called = True
        # Build a list so we're safe even if the list of accumulators is modified
        # even while we're reloading.
        with self._accumulators_mutex:
            items = list(self._accumulators.items())
        items_queue = queue.Queue()
        for item in items:
            items_queue.put(item)

        # Methods of built-in python containers are thread-safe so long as the GIL
        # for the thread exists, but we might as well be careful.
        names_to_delete = set()
        names_to_delete_mutex = threading.Lock()

        def Worker():
            """Keeps reloading accumulators til none are left."""
            while True:
                try:
                    name, accumulator = items_queue.get(block=False)
                except queue.Empty:
                    # No more runs to reload.
                    break

                try:
                    accumulator.Reload()
                except (OSError, IOError) as e:
                    logger.error("Unable to reload accumulator %r: %s", name, e)
                except directory_watcher.DirectoryDeletedError:
                    with names_to_delete_mutex:
                        names_to_delete.add(name)
                finally:
                    items_queue.task_done()

        if self._max_reload_threads > 1:
            num_threads = min(self._max_reload_threads, len(items))
            logger.info("Starting %d threads to reload runs", num_threads)
            for i in range(num_threads):
                thread = threading.Thread(target=Worker, name="Reloader %d" % i)
                thread.daemon = True
                thread.start()
            items_queue.join()
        else:
            logger.info(
                "Reloading runs serially (one after another) on the main "
                "thread."
            )
            Worker()

        with self._accumulators_mutex:
            for name in names_to_delete:
                logger.warning("Deleting accumulator %r", name)
                del self._accumulators[name]
        logger.info("Finished with EventMultiplexer.Reload()")
        return self

    def PluginAssets(self, plugin_name):
        """Get index of runs and assets for a given plugin.

        Args:
          plugin_name: Name of the plugin we are checking for.

        Returns:
          A dictionary that maps from run_name to a list of plugin
            assets for that run.
        """
        with self._accumulators_mutex:
            # To avoid nested locks, we construct a copy of the run-accumulator map
            items = list(self._accumulators.items())

        return {run: accum.PluginAssets(plugin_name) for run, accum in items}

    def RetrievePluginAsset(self, run, plugin_name, asset_name):
        """Return the contents for a specific plugin asset from a run.

        Args:
          run: The string name of the run.
          plugin_name: The string name of a plugin.
          asset_name: The string name of an asset.

        Returns:
          The string contents of the plugin asset.

        Raises:
          KeyError: If the asset is not available.
        """
        accumulator = self.GetAccumulator(run)
        return accumulator.RetrievePluginAsset(plugin_name, asset_name)

    def FirstEventTimestamp(self, run):
        """Return the timestamp of the first event of the given run.

        This may perform I/O if no events have been loaded yet for the run.

        Args:
          run: A string name of the run for which the timestamp is retrieved.

        Returns:
          The wall_time of the first event of the run, which will typically be
          seconds since the epoch.

        Raises:
          KeyError: If the run is not found.
          ValueError: If the run has no events loaded and there are no events on
            disk to load.
        """
        accumulator = self.GetAccumulator(run)
        return accumulator.FirstEventTimestamp()

    def GetSourceWriter(self, run) -> Optional[str]:
        """Returns the source writer name from the first event of the given run.

        Assuming each run has only one source writer.

        Args:
          run: A string name of the run from which the event source information
            is retrieved.

        Returns:
          Name of the writer that wrote the events in the run.
        """
        accumulator = self.GetAccumulator(run)
        return accumulator.GetSourceWriter()

    def Graph(self, run):
        """Retrieve the graph associated with the provided run.

        Args:
          run: A string name of a run to load the graph for.

        Raises:
          KeyError: If the run is not found.
          ValueError: If the run does not have an associated graph.

        Returns:
          The `GraphDef` protobuf data structure.
        """
        accumulator = self.GetAccumulator(run)
        return accumulator.Graph()

    def SerializedGraph(self, run):
        """Retrieve the serialized graph associated with the provided run.

        Args:
          run: A string name of a run to load the graph for.

        Raises:
          KeyError: If the run is not found.
          ValueError: If the run does not have an associated graph.

        Returns:
          The serialized form of the `GraphDef` protobuf data structure.
        """
        accumulator = self.GetAccumulator(run)
        return accumulator.SerializedGraph()

    def MetaGraph(self, run):
        """Retrieve the metagraph associated with the provided run.

        Args:
          run: A string name of a run to load the graph for.

        Raises:
          KeyError: If the run is not found.
          ValueError: If the run does not have an associated graph.

        Returns:
          The `MetaGraphDef` protobuf data structure.
        """
        accumulator = self.GetAccumulator(run)
        return accumulator.MetaGraph()

    def RunMetadata(self, run, tag):
        """Get the session.run() metadata associated with a TensorFlow run and
        tag.

        Args:
          run: A string name of a TensorFlow run.
          tag: A string name of the tag associated with a particular session.run().

        Raises:
          KeyError: If the run is not found, or the tag is not available for the
            given run.

        Returns:
          The metadata in the form of `RunMetadata` protobuf data structure.
        """
        accumulator = self.GetAccumulator(run)
        return accumulator.RunMetadata(tag)

    def Tensors(self, run, tag):
        """Retrieve the tensor events associated with a run and tag.

        Args:
          run: A string name of the run for which values are retrieved.
          tag: A string name of the tag for which values are retrieved.

        Raises:
          KeyError: If the run is not found, or the tag is not available for
            the given run.

        Returns:
          An array of `event_accumulator.TensorEvent`s.
        """
        accumulator = self.GetAccumulator(run)
        return accumulator.Tensors(tag)

    def PluginRunToTagToContent(self, plugin_name):
        """Returns a 2-layer dictionary of the form {run: {tag: content}}.

        The `content` referred above is the content field of the PluginData proto
        for the specified plugin within a Summary.Value proto.

        Args:
          plugin_name: The name of the plugin for which to fetch content.

        Returns:
          A dictionary of the form {run: {tag: content}}.
        """
        mapping = {}
        for run in self.Runs():
            try:
                tag_to_content = self.GetAccumulator(run).PluginTagToContent(
                    plugin_name
                )
            except KeyError:
                # This run lacks content for the plugin. Try the next run.
                continue
            mapping[run] = tag_to_content
        return mapping

    def ActivePlugins(self):
        """Return a set of plugins with summary data.

        Returns:
          The distinct union of `plugin_data.plugin_name` fields from
          all the `SummaryMetadata` protos stored in any run known to
          this multiplexer.
        """
        with self._accumulators_mutex:
            accumulators = list(self._accumulators.values())
        return frozenset().union(*(a.ActivePlugins() for a in accumulators))

    def SummaryMetadata(self, run, tag):
        """Return the summary metadata for the given tag on the given run.

        Args:
          run: A string name of the run for which summary metadata is to be
            retrieved.
          tag: A string name of the tag whose summary metadata is to be
            retrieved.

        Raises:
          KeyError: If the run is not found, or the tag is not available for
            the given run.

        Returns:
          A `SummaryMetadata` protobuf.
        """
        accumulator = self.GetAccumulator(run)
        return accumulator.SummaryMetadata(tag)

    def AllSummaryMetadata(self):
        """Return summary metadata for all time series.

        Returns:
          A nested dict `d` such that `d[run][tag]` is a
          `SummaryMetadata` proto for the keyed time series.
        """
        with self._accumulators_mutex:
            # To avoid nested locks, we construct a copy of the run-accumulator map
            items = list(self._accumulators.items())
        return {
            run_name: accumulator.AllSummaryMetadata()
            for run_name, accumulator in items
        }

    def Runs(self):
        """Return all the run names in the `EventMultiplexer`.

        Returns:
        ```
          {runName: { scalarValues: [tagA, tagB, tagC],
                      graph: true, meta_graph: true}}
        ```
        """
        with self._accumulators_mutex:
            # To avoid nested locks, we construct a copy of the run-accumulator map
            items = list(self._accumulators.items())
        return {run_name: accumulator.Tags() for run_name, accumulator in items}

    def RunPaths(self):
        """Returns a dict mapping run names to event file paths."""
        return self._paths

    def GetAccumulator(self, run):
        """Returns EventAccumulator for a given run.

        Args:
          run: String name of run.

        Returns:
          An EventAccumulator object.

        Raises:
          KeyError: If run does not exist.
        """
        with self._accumulators_mutex:
            return self._accumulators[run]
