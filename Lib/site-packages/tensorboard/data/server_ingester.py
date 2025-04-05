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
"""Provides data ingestion logic backed by a gRPC server."""

import errno
import logging
import os
import subprocess
import tempfile
import time

import grpc
import pkg_resources

from tensorboard.data import grpc_provider
from tensorboard.data import ingester
from tensorboard.data.proto import data_provider_pb2
from tensorboard.util import tb_logging


logger = tb_logging.get_logger()

# If this environment variable is non-empty, it will be used as the path to the
# data server binary rather than using a bundled version.
_ENV_DATA_SERVER_BINARY = "TENSORBOARD_DATA_SERVER_BINARY"


class ExistingServerDataIngester(ingester.DataIngester):
    """Connect to an already running gRPC server."""

    def __init__(self, address, *, channel_creds_type):
        """Initializes an ingester with the given configuration.

        Args:
          address: String, as passed to `--grpc_data_provider`.
          channel_creds_type: `grpc_util.ChannelCredsType`, as passed to
            `--grpc_creds_type`.
        """
        stub = _make_stub(address, channel_creds_type)
        self._data_provider = grpc_provider.GrpcDataProvider(address, stub)

    @property
    def data_provider(self):
        return self._data_provider

    def start(self):
        pass


class SubprocessServerDataIngester(ingester.DataIngester):
    """Start a new data server as a subprocess."""

    def __init__(
        self,
        server_binary,
        logdir,
        *,
        reload_interval,
        channel_creds_type,
        samples_per_plugin=None,
        extra_flags=None,
    ):
        """Initializes an ingester with the given configuration.

        Args:
          server_binary: `ServerBinary` to launch.
          logdir: String, as passed to `--logdir`.
          reload_interval: Number, as passed to `--reload_interval`.
          channel_creds_type: `grpc_util.ChannelCredsType`, as passed to
            `--grpc_creds_type`.
          samples_per_plugin: Dict[String, Int], as parsed from
            `--samples_per_plugin`.
          extra_flags: List of extra string flags to be passed to the
            data server without further interpretation.
        """
        self._server_binary = server_binary
        self._data_provider = None
        self._logdir = logdir
        self._reload_interval = reload_interval
        self._channel_creds_type = channel_creds_type
        self._samples_per_plugin = samples_per_plugin or {}
        self._extra_flags = list(extra_flags or [])

    @property
    def data_provider(self):
        if self._data_provider is None:
            raise RuntimeError("Must call `start` first")
        return self._data_provider

    def start(self):
        if self._data_provider:
            return

        tmpdir = tempfile.TemporaryDirectory(prefix="tensorboard_data_server_")
        port_file_path = os.path.join(tmpdir.name, "port")
        error_file_path = os.path.join(tmpdir.name, "startup_error")

        if self._reload_interval <= 0:
            reload = "once"
        else:
            reload = str(int(self._reload_interval))

        sample_hint_pairs = [
            "%s=%s" % (k, "all" if v == 0 else v)
            for k, v in self._samples_per_plugin.items()
        ]
        samples_per_plugin = ",".join(sample_hint_pairs)

        args = [
            self._server_binary.path,
            "--logdir=%s" % os.path.expanduser(self._logdir),
            "--reload=%s" % reload,
            "--samples-per-plugin=%s" % samples_per_plugin,
            "--port=0",
            "--port-file=%s" % (port_file_path,),
            "--die-after-stdin",
        ]
        if self._server_binary.at_least_version("0.5.0a0"):
            args.append("--error-file=%s" % (error_file_path,))
        if logger.isEnabledFor(logging.INFO):
            args.append("--verbose")
        if logger.isEnabledFor(logging.DEBUG):
            args.append("--verbose")  # Repeat arg to increase verbosity.
        args.extend(self._extra_flags)

        logger.info("Spawning data server: %r", args)
        popen = subprocess.Popen(args, stdin=subprocess.PIPE)
        # Stash stdin to avoid calling its destructor: on Windows, this
        # is a `subprocess.Handle` that closes itself in `__del__`,
        # which would cause the data server to shut down. (This is not
        # documented; you have to read CPython source to figure it out.)
        # We want that to happen at end of process, but not before.
        self._stdin_handle = popen.stdin  # stash to avoid stdin being closed

        port = None
        # The server only needs about 10 microseconds to spawn on my machine,
        # but give a few orders of magnitude of padding, and then poll.
        time.sleep(0.01)
        for i in range(20):
            if popen.poll() is not None:
                msg = (_maybe_read_file(error_file_path) or "").strip()
                if not msg:
                    msg = (
                        "exited with %d; check stderr for details"
                        % popen.poll()
                    )
                raise DataServerStartupError(msg)
            logger.info("Polling for data server port (attempt %d)", i)
            port_file_contents = _maybe_read_file(port_file_path)
            logger.info("Port file contents: %r", port_file_contents)
            if (port_file_contents or "").endswith("\n"):
                port = int(port_file_contents)
                break
            # Else, not done writing yet.
            time.sleep(0.5)
        if port is None:
            raise DataServerStartupError(
                "Timed out while waiting for data server to start. "
                "It may still be running as pid %d." % popen.pid
            )

        addr = "localhost:%d" % port
        stub = _make_stub(addr, self._channel_creds_type)
        logger.info(
            "Opened channel to data server at pid %d via %s",
            popen.pid,
            addr,
        )

        req = data_provider_pb2.GetExperimentRequest()
        try:
            stub.GetExperiment(req, timeout=5)  # should be near-instant
        except grpc.RpcError as e:
            msg = "Failed to communicate with data server at %s: %s" % (addr, e)
            logging.warning("%s", msg)
            raise DataServerStartupError(msg) from e
        logger.info("Got valid response from data server")
        self._data_provider = grpc_provider.GrpcDataProvider(addr, stub)


def _maybe_read_file(path):
    """Read a file, or return `None` on ENOENT specifically."""
    try:
        with open(path) as infile:
            return infile.read()
    except OSError as e:
        if e.errno == errno.ENOENT:
            return None
        raise


def _make_stub(addr, channel_creds_type):
    (creds, options) = channel_creds_type.channel_config()
    options.append(("grpc.max_receive_message_length", 1024 * 1024 * 256))
    channel = grpc.secure_channel(addr, creds, options=options)
    return grpc_provider.make_stub(channel)


class NoDataServerError(RuntimeError):
    pass


class DataServerStartupError(RuntimeError):
    pass


class ServerBinary:
    """Information about a data server binary."""

    def __init__(self, path, version):
        """Initializes a `ServerBinary`.

        Args:
          path: String path to executable on disk.
          version: PEP 396-compliant version string, or `None` if
            unknown or not applicable. Binaries at unknown versions are
            assumed to be bleeding-edge: if you bring your own binary,
            it's on you to make sure that it's up to date.
        """
        self._path = path
        self._version = (
            pkg_resources.parse_version(version)
            if version is not None
            else version
        )

    @property
    def path(self):
        return self._path

    def at_least_version(self, required_version):
        """Test whether the binary's version is at least the given one.

        Useful for gating features that are available in the latest data
        server builds from head, but not yet released to PyPI. For
        example, if v0.4.0 is the latest published version, you can
        check `at_least_version("0.5.0a0")` to include both prereleases
        at head and the eventual final release of v0.5.0.

        If this binary's version was set to `None` at construction time,
        this method always returns `True`.

        Args:
          required_version: PEP 396-compliant version string.

        Returns:
          Boolean.
        """
        if self._version is None:
            return True
        return self._version >= pkg_resources.parse_version(required_version)


def get_server_binary():
    """Get `ServerBinary` info or raise `NoDataServerError`."""
    env_result = os.environ.get(_ENV_DATA_SERVER_BINARY)
    if env_result:
        logging.info("Server binary (from env): %s", env_result)
        if not os.path.isfile(env_result):
            raise NoDataServerError(
                "Found environment variable %s=%s, but no such file exists."
                % (_ENV_DATA_SERVER_BINARY, env_result)
            )
        return ServerBinary(env_result, version=None)

    bundle_result = os.path.join(os.path.dirname(__file__), "server", "server")
    if os.path.exists(bundle_result):
        logging.info("Server binary (from bundle): %s", bundle_result)
        return ServerBinary(bundle_result, version=None)

    try:
        import tensorboard_data_server
    except ImportError:
        pass
    else:
        pkg_result = tensorboard_data_server.server_binary()
        version = tensorboard_data_server.__version__
        logging.info(
            "Server binary (from Python package v%s): %s", version, pkg_result
        )
        if pkg_result is None:
            raise NoDataServerError(
                "TensorBoard data server not supported on this platform."
            )
        return ServerBinary(pkg_result, version)

    raise NoDataServerError(
        "TensorBoard data server not found. This mode is experimental. "
        "If building from source, pass --define=link_data_server=true."
    )
