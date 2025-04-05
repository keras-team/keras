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

"""Implementation for a multi-file directory loader."""


from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.compat import tf
from tensorboard.util import tb_logging


logger = tb_logging.get_logger()


# Sentinel object for an inactive path.
_INACTIVE = object()


class DirectoryLoader:
    """Loader for an entire directory, maintaining multiple active file
    loaders.

    This class takes a directory, a factory for loaders, and optionally a
    path filter and watches all the paths inside that directory for new data.
    Each file loader created by the factory must read a path and produce an
    iterator of (timestamp, value) pairs.

    Unlike DirectoryWatcher, this class does not assume that only one file
    receives new data at a time; there can be arbitrarily many active files.
    However, any file whose maximum load timestamp fails an "active" predicate
    will be marked as inactive and no longer checked for new data.
    """

    def __init__(
        self,
        directory,
        loader_factory,
        path_filter=lambda x: True,
        active_filter=lambda timestamp: True,
    ):
        """Constructs a new MultiFileDirectoryLoader.

        Args:
          directory: The directory to load files from.
          loader_factory: A factory for creating loaders. The factory should take a
            path and return an object that has a Load method returning an iterator
            yielding (unix timestamp as float, value) pairs for any new data
          path_filter: If specified, only paths matching this filter are loaded.
          active_filter: If specified, any loader whose maximum load timestamp does
            not pass this filter will be marked as inactive and no longer read.

        Raises:
          ValueError: If directory or loader_factory are None.
        """
        if directory is None:
            raise ValueError("A directory is required")
        if loader_factory is None:
            raise ValueError("A loader factory is required")
        self._directory = directory
        self._loader_factory = loader_factory
        self._path_filter = path_filter
        self._active_filter = active_filter
        self._loaders = {}
        self._max_timestamps = {}

    def Load(self):
        """Loads new values from all active files.

        Yields:
          All values that have not been yielded yet.

        Raises:
          DirectoryDeletedError: If the directory has been permanently deleted
            (as opposed to being temporarily unavailable).
        """
        try:
            all_paths = io_wrapper.ListDirectoryAbsolute(self._directory)
            paths = sorted(p for p in all_paths if self._path_filter(p))
            for path in paths:
                for value in self._LoadPath(path):
                    yield value
        except tf.errors.OpError as e:
            if not tf.io.gfile.exists(self._directory):
                raise directory_watcher.DirectoryDeletedError(
                    "Directory %s has been permanently deleted"
                    % self._directory
                )
            else:
                logger.info("Ignoring error during file loading: %s" % e)

    def _LoadPath(self, path):
        """Generator for values from a single path's loader.

        Args:
          path: the path to load from

        Yields:
          All values from this path's loader that have not been yielded yet.
        """
        max_timestamp = self._max_timestamps.get(path, None)
        if max_timestamp is _INACTIVE or self._MarkIfInactive(
            path, max_timestamp
        ):
            logger.debug("Skipping inactive path %s", path)
            return
        loader = self._loaders.get(path, None)
        if loader is None:
            try:
                loader = self._loader_factory(path)
            except tf.errors.NotFoundError:
                # Happens if a file was removed after we listed the directory.
                logger.debug("Skipping nonexistent path %s", path)
                return
            self._loaders[path] = loader
        logger.info("Loading data from path %s", path)
        for timestamp, value in loader.Load():
            if max_timestamp is None or timestamp > max_timestamp:
                max_timestamp = timestamp
            yield value
        if not self._MarkIfInactive(path, max_timestamp):
            self._max_timestamps[path] = max_timestamp

    def _MarkIfInactive(self, path, max_timestamp):
        """If max_timestamp is inactive, returns True and marks the path as
        such."""
        logger.debug("Checking active status of %s at %s", path, max_timestamp)
        if max_timestamp is not None and not self._active_filter(max_timestamp):
            self._max_timestamps[path] = _INACTIVE
            del self._loaders[path]
            return True
        return False
