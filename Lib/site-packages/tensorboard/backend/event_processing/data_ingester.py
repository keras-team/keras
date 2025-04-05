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
"""Provides data ingestion logic backed by local event processing."""

import os
import re
import threading
import time


from tensorboard.backend.event_processing import data_provider
from tensorboard.backend.event_processing import plugin_event_multiplexer
from tensorboard.backend.event_processing import tag_types
from tensorboard.compat import tf
from tensorboard.data import ingester
from tensorboard.plugins.audio import metadata as audio_metadata
from tensorboard.plugins.histogram import metadata as histogram_metadata
from tensorboard.plugins.image import metadata as image_metadata
from tensorboard.plugins.pr_curve import metadata as pr_curve_metadata
from tensorboard.plugins.scalar import metadata as scalar_metadata
from tensorboard.util import tb_logging


DEFAULT_SIZE_GUIDANCE = {
    tag_types.TENSORS: 10,
}

# TODO(@wchargin): Replace with something that works for third-party plugins.
DEFAULT_TENSOR_SIZE_GUIDANCE = {
    scalar_metadata.PLUGIN_NAME: 1000,
    image_metadata.PLUGIN_NAME: 10,
    audio_metadata.PLUGIN_NAME: 10,
    histogram_metadata.PLUGIN_NAME: 500,
    pr_curve_metadata.PLUGIN_NAME: 100,
}

logger = tb_logging.get_logger()


class LocalDataIngester(ingester.DataIngester):
    """Data ingestion implementation to use when running locally."""

    def __init__(self, flags):
        """Initializes a `LocalDataIngester` from `flags`.

        Args:
          flags: An argparse.Namespace containing TensorBoard CLI flags.

        Returns:
          The new `LocalDataIngester`.
        """
        tensor_size_guidance = dict(DEFAULT_TENSOR_SIZE_GUIDANCE)
        tensor_size_guidance.update(flags.samples_per_plugin)
        self._multiplexer = plugin_event_multiplexer.EventMultiplexer(
            size_guidance=DEFAULT_SIZE_GUIDANCE,
            tensor_size_guidance=tensor_size_guidance,
            purge_orphaned_data=flags.purge_orphaned_data,
            max_reload_threads=flags.max_reload_threads,
            event_file_active_filter=_get_event_file_active_filter(flags),
            detect_file_replacement=flags.detect_file_replacement,
        )
        self._data_provider = data_provider.MultiplexerDataProvider(
            self._multiplexer, flags.logdir or flags.logdir_spec
        )
        self._reload_interval = flags.reload_interval
        self._reload_task = flags.reload_task
        if flags.logdir:
            self._path_to_run = {os.path.expanduser(flags.logdir): None}
        else:
            self._path_to_run = _parse_event_files_spec(flags.logdir_spec)

        # Conditionally import tensorflow_io.
        if getattr(tf, "__version__", "stub") != "stub":
            _check_filesystem_support(self._path_to_run.keys())

    @property
    def data_provider(self):
        return self._data_provider

    @property
    def deprecated_multiplexer(self):
        return self._multiplexer

    def start(self):
        """Starts ingesting data based on the ingester flag configuration."""

        def _reload():
            while True:
                start = time.time()
                logger.info("TensorBoard reload process beginning")
                for path, name in self._path_to_run.items():
                    self._multiplexer.AddRunsFromDirectory(path, name)
                logger.info(
                    "TensorBoard reload process: Reload the whole Multiplexer"
                )
                self._multiplexer.Reload()
                duration = time.time() - start
                logger.info(
                    "TensorBoard done reloading. Load took %0.3f secs", duration
                )
                if self._reload_interval == 0:
                    # Only load the multiplexer once. Do not continuously reload.
                    break
                time.sleep(self._reload_interval)

        if self._reload_task == "process":
            logger.info("Launching reload in a child process")
            import multiprocessing

            process = multiprocessing.Process(target=_reload, name="Reloader")
            # Best-effort cleanup; on exit, the main TB parent process will attempt to
            # kill all its daemonic children.
            process.daemon = True
            process.start()
        elif self._reload_task in ("thread", "auto"):
            logger.info("Launching reload in a daemon thread")
            thread = threading.Thread(target=_reload, name="Reloader")
            # Make this a daemon thread, which won't block TB from exiting.
            thread.daemon = True
            thread.start()
        elif self._reload_task == "blocking":
            if self._reload_interval != 0:
                raise ValueError(
                    "blocking reload only allowed with load_interval=0"
                )
            _reload()
        else:
            raise ValueError("unrecognized reload_task: %s" % self._reload_task)


def _get_event_file_active_filter(flags):
    """Returns a predicate for whether an event file load timestamp is active.

    Returns:
      A predicate function accepting a single UNIX timestamp float argument, or
      None if multi-file loading is not enabled.
    """
    if not flags.reload_multifile:
        return None
    inactive_secs = flags.reload_multifile_inactive_secs
    if inactive_secs == 0:
        return None
    if inactive_secs < 0:
        return lambda timestamp: True
    return lambda timestamp: timestamp + inactive_secs >= time.time()


def _parse_event_files_spec(logdir_spec):
    """Parses `logdir_spec` into a map from paths to run group names.

    The `--logdir_spec` flag format is a comma-separated list of path
    specifications. A path spec looks like 'group_name:/path/to/directory' or
    '/path/to/directory'; in the latter case, the group is unnamed. Group names
    cannot start with a forward slash: /foo:bar/baz will be interpreted as a spec
    with no name and path '/foo:bar/baz'.

    Globs are not supported.

    Args:
      logdir: A comma-separated list of run specifications.
    Returns:
      A dict mapping directory paths to names like {'/path/to/directory': 'name'}.
      Groups without an explicit name are named after their path. If logdir is
      None, returns an empty dict, which is helpful for testing things that don't
      require any valid runs.
    """
    files = {}
    if logdir_spec is None:
        return files
    # Make sure keeping consistent with ParseURI in core/lib/io/path.cc
    uri_pattern = re.compile("[a-zA-Z][0-9a-zA-Z.]*://.*")
    for specification in logdir_spec.split(","):
        # Check if the spec contains group. A spec start with xyz:// is regarded as
        # URI path spec instead of group spec. If the spec looks like /foo:bar/baz,
        # then we assume it's a path with a colon. If the spec looks like
        # [a-zA-z]:\foo then we assume its a Windows path and not a single letter
        # group
        if (
            uri_pattern.match(specification) is None
            and ":" in specification
            and specification[0] != "/"
            and not os.path.splitdrive(specification)[0]
        ):
            # We split at most once so run_name:/path:with/a/colon will work.
            run_name, _, path = specification.partition(":")
        else:
            run_name = None
            path = specification
        if uri_pattern.match(path) is None:
            path = os.path.realpath(os.path.expanduser(path))
        files[path] = run_name
    return files


def _get_filesystem_scheme(path):
    """Extracts filesystem scheme from a given path.

    The filesystem scheme is usually separated by `://` from the local filesystem
    path if given. For example, the scheme of `file://tmp/tf` is `file`.

    Args:
        path: A strings representing an input log directory.
    Returns:
        Filesystem scheme, None if the path doesn't contain one.
    """
    if "://" not in path:
        return None
    return path.split("://")[0]


def _check_filesystem_support(paths):
    """Examines the list of filesystems user requested.

    If TF I/O schemes are requested, try to import tensorflow_io module.

    Args:
        paths: A list of strings representing input log directories.
    """
    get_registered_schemes = getattr(
        tf.io.gfile, "get_registered_schemes", None
    )
    registered_schemes = (
        None if get_registered_schemes is None else get_registered_schemes()
    )

    # Only need to check one path for each scheme.
    scheme_to_path = {_get_filesystem_scheme(path): path for path in paths}
    missing_scheme = None
    for scheme, path in scheme_to_path.items():
        if scheme is None:
            continue
        # Use `tf.io.gfile.exists.get_registered_schemes` if possible.
        if registered_schemes is not None:
            if scheme not in registered_schemes:
                missing_scheme = scheme
                break
        else:
            # Fall back to `tf.io.gfile.exists`.
            try:
                tf.io.gfile.exists(path)
            except tf.errors.UnimplementedError:
                missing_scheme = scheme
                break
            except tf.errors.OpError:
                # Swallow other errors; we aren't concerned about them at this point.
                pass

    if missing_scheme:
        try:
            import tensorflow_io  # noqa: F401
        except ImportError as e:
            supported_schemes_msg = (
                " (supported schemes: {})".format(registered_schemes)
                if registered_schemes
                else ""
            )
            raise tf.errors.UnimplementedError(
                None,
                None,
                (
                    "Error: Unsupported filename scheme '{}'{}. For additional"
                    + " filesystem support, consider installing TensorFlow I/O"
                    + " (https://www.tensorflow.org/io) via `pip install tensorflow-io`."
                ).format(missing_scheme, supported_schemes_msg),
            ) from e
