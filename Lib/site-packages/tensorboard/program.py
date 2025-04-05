# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for TensorBoard command line program.

This is a lightweight module for bringing up a TensorBoard HTTP server
or emulating the `tensorboard` shell command.

Those wishing to create custom builds of TensorBoard can use this module
by swapping out `tensorboard.main` with the custom definition that
modifies the set of plugins and static assets.

This module does not depend on first-party plugins or the default web
server assets. Those are defined in `tensorboard.default`.
"""


from abc import ABCMeta
from abc import abstractmethod
import argparse
import atexit
from collections import defaultdict
import errno
import logging
import mimetypes
import os
import shlex
import signal
import socket
import sys
import threading
import time
import urllib.parse

from absl import flags as absl_flags
from absl.flags import argparse_flags
from werkzeug import serving

from tensorboard import manager
from tensorboard import version
from tensorboard.backend import application
from tensorboard.backend.event_processing import data_ingester as local_ingester
from tensorboard.backend.event_processing import event_file_inspector as efi
from tensorboard.data import server_ingester
from tensorboard.plugins.core import core_plugin
from tensorboard.util import tb_logging


logger = tb_logging.get_logger()

# Default subcommand name. This is a user-facing CLI and should not change.
_SERVE_SUBCOMMAND_NAME = "serve"
# Internal flag name used to store which subcommand was invoked.
_SUBCOMMAND_FLAG = "__tensorboard_subcommand"

# Message printed when we actually use the data server, so that users are not
# caught terribly by surprise.
_DATA_SERVER_ADVISORY_MESSAGE = """
NOTE: Using experimental fast data loading logic. To disable, pass
    "--load_fast=false" and report issues on GitHub. More details:
    https://github.com/tensorflow/tensorboard/issues/4784

"""

# Message printed with `--load_fast=true` if the data server could not start up.
# To be formatted with one `DataServerStartupError` interpoland.
_DATA_SERVER_STARTUP_ERROR_MESSAGE_TEMPLATE = """\
Could not start data server: %s.
    Try with --load_fast=false and report issues on GitHub. Details:
    https://github.com/tensorflow/tensorboard/issues/4784
"""


class TensorBoard:
    """Class for running TensorBoard.

    Fields:
      plugin_loaders: Set from plugins passed to constructor.
      assets_zip_provider: Set by constructor.
      server_class: Set by constructor.
      flags: An argparse.Namespace set by the configure() method.
      cache_key: As `manager.cache_key`; set by the configure() method.
    """

    def __init__(
        self,
        plugins=None,
        assets_zip_provider=None,
        server_class=None,
        subcommands=None,
    ):
        """Creates new instance.

        Args:
          plugins: A list of TensorBoard plugins to load, as TBPlugin classes or
            TBLoader instances or classes. If not specified, defaults to first-party
            plugins.
          assets_zip_provider: A function that provides a zip file containing
            assets to the application. If `None`, the default TensorBoard web
            assets will be used. (If building from source, your binary must
            explicitly depend on `//tensorboard:assets_lib` if you pass `None`.)
          server_class: An optional factory for a `TensorBoardServer` to use
            for serving the TensorBoard WSGI app. If provided, its callable
            signature should match that of `TensorBoardServer.__init__`.
          subcommands: An optional list of TensorBoardSubcommand objects, which
            extend the functionality of the CLI.

        :type plugins:
          list[
            base_plugin.TBLoader | Type[base_plugin.TBLoader] |
            Type[base_plugin.TBPlugin]
          ]
        """
        if plugins is None:
            from tensorboard import default

            plugins = default.get_plugins()
        if assets_zip_provider is None:
            try:
                from tensorboard import assets
            except ImportError as e:
                # `tensorboard.assets` is not a strict Bazel dep; clients are
                # required to either depend on `//tensorboard:assets_lib` or
                # pass a valid assets provider.
                raise ImportError(
                    "No `assets_zip_provider` given, but `tensorboard.assets` "
                    "could not be imported to resolve defaults"
                ) from e
            assets_zip_provider = assets.get_default_assets_zip_provider()
        if server_class is None:
            server_class = create_port_scanning_werkzeug_server
        if subcommands is None:
            subcommands = []
        self.plugin_loaders = [
            application.make_plugin_loader(p) for p in plugins
        ]
        self.assets_zip_provider = assets_zip_provider
        self.server_class = server_class
        self.subcommands = {}
        for subcommand in subcommands:
            name = subcommand.name()
            if name in self.subcommands or name == _SERVE_SUBCOMMAND_NAME:
                raise ValueError("Duplicate subcommand name: %r" % name)
            self.subcommands[name] = subcommand
        self.flags = None

    def configure(self, argv=("",), **kwargs):
        """Configures TensorBoard behavior via flags.

        This method will populate the "flags" property with an argparse.Namespace
        representing flag values parsed from the provided argv list, overridden by
        explicit flags from remaining keyword arguments.

        Args:
          argv: Can be set to CLI args equivalent to sys.argv; the first arg is
            taken to be the name of the path being executed.
          kwargs: Additional arguments will override what was parsed from
            argv. They must be passed as Python data structures, e.g.
            `foo=1` rather than `foo="1"`.

        Returns:
          Either argv[:1] if argv was non-empty, or [''] otherwise, as a mechanism
          for absl.app.run() compatibility.

        Raises:
          ValueError: If flag values are invalid.
        """

        base_parser = argparse_flags.ArgumentParser(
            prog="tensorboard",
            description=(
                "TensorBoard is a suite of web applications for "
                "inspecting and understanding your TensorFlow runs "
                "and graphs. https://github.com/tensorflow/tensorboard "
            ),
        )
        subparsers = base_parser.add_subparsers(
            help="TensorBoard subcommand (defaults to %r)"
            % _SERVE_SUBCOMMAND_NAME
        )

        serve_subparser = subparsers.add_parser(
            _SERVE_SUBCOMMAND_NAME,
            help="start local TensorBoard server (default subcommand)",
        )
        serve_subparser.set_defaults(
            **{_SUBCOMMAND_FLAG: _SERVE_SUBCOMMAND_NAME}
        )

        if len(argv) < 2 or argv[1].startswith("-"):
            # This invocation, if valid, must not use any subcommands: we
            # don't permit flags before the subcommand name.
            serve_parser = base_parser
        else:
            # This invocation, if valid, must use a subcommand: we don't take
            # any positional arguments to `serve`.
            serve_parser = serve_subparser

        for name, subcommand in self.subcommands.items():
            subparser = subparsers.add_parser(
                name,
                help=subcommand.help(),
                description=subcommand.description(),
            )
            subparser.set_defaults(**{_SUBCOMMAND_FLAG: name})
            subcommand.define_flags(subparser)

        for loader in self.plugin_loaders:
            loader.define_flags(serve_parser)

        arg0 = argv[0] if argv else ""

        flags = base_parser.parse_args(argv[1:])  # Strip binary name from argv.
        if getattr(flags, _SUBCOMMAND_FLAG, None) is None:
            # Manually assign default value rather than using `set_defaults`
            # on the base parser to work around Python bug #9351 on old
            # versions of `argparse`: <https://bugs.python.org/issue9351>
            setattr(flags, _SUBCOMMAND_FLAG, _SERVE_SUBCOMMAND_NAME)

        self.cache_key = manager.cache_key(
            working_directory=os.getcwd(),
            arguments=argv[1:],
            configure_kwargs=kwargs,
        )
        if arg0:
            # Only expose main module Abseil flags as TensorBoard native flags.
            # This is the same logic Abseil's ArgumentParser uses for determining
            # which Abseil flags to include in the short helpstring.
            for flag in set(absl_flags.FLAGS.get_key_flags_for_module(arg0)):
                if hasattr(flags, flag.name):
                    raise ValueError("Conflicting Abseil flag: %s" % flag.name)
                setattr(flags, flag.name, flag.value)
        for k, v in kwargs.items():
            if not hasattr(flags, k):
                raise ValueError("Unknown TensorBoard flag: %s" % k)
            setattr(flags, k, v)
        if getattr(flags, _SUBCOMMAND_FLAG) == _SERVE_SUBCOMMAND_NAME:
            for loader in self.plugin_loaders:
                loader.fix_flags(flags)
        self.flags = flags
        return [arg0]

    def main(self, ignored_argv=("",)):
        """Blocking main function for TensorBoard.

        This method is called by `tensorboard.main.run_main`, which is the
        standard entrypoint for the tensorboard command line program. The
        configure() method must be called first.

        Args:
          ignored_argv: Do not pass. Required for Abseil compatibility.

        Returns:
          Process exit code, i.e. 0 if successful or non-zero on failure. In
          practice, an exception will most likely be raised instead of
          returning non-zero.

        :rtype: int
        """
        self._install_signal_handler(signal.SIGTERM, "SIGTERM")
        self._fix_mime_types()
        subcommand_name = getattr(self.flags, _SUBCOMMAND_FLAG)
        if subcommand_name == _SERVE_SUBCOMMAND_NAME:
            runner = self._run_serve_subcommand
        else:
            runner = self.subcommands[subcommand_name].run
        return runner(self.flags) or 0

    def _run_serve_subcommand(self, flags):
        # TODO(#2801): Make `--version` a flag on only the base parser, not `serve`.
        if flags.version_tb:
            print(version.VERSION)
            return 0
        if flags.inspect:
            # TODO(@wchargin): Convert `inspect` to a normal subcommand?
            logger.info(
                "Not bringing up TensorBoard, but inspecting event files."
            )
            event_file = os.path.expanduser(flags.event_file)
            efi.inspect(flags.logdir, event_file, flags.tag)
            return 0
        try:
            server = self._make_server()
            server.print_serving_message()
            self._register_info(server)
            server.serve_forever()
            return 0
        except TensorBoardServerException as e:
            logger.error(e.msg)
            sys.stderr.write("ERROR: %s\n" % e.msg)
            sys.stderr.flush()
            return -1

    def launch(self):
        """Python API for launching TensorBoard.

        This method is the same as main() except it launches TensorBoard in
        a separate permanent thread. The configure() method must be called
        first.

        Returns:
          The URL of the TensorBoard web server.

        :rtype: str
        """
        # Make it easy to run TensorBoard inside other programs, e.g. Colab.
        server = self._make_server()
        thread = threading.Thread(
            target=server.serve_forever, name="TensorBoard"
        )
        thread.daemon = True
        thread.start()
        return server.get_url()

    def _register_info(self, server):
        """Write a TensorBoardInfo file and arrange for its cleanup.

        Args:
          server: The result of `self._make_server()`.
        """
        server_url = urllib.parse.urlparse(server.get_url())
        info = manager.TensorBoardInfo(
            version=version.VERSION,
            start_time=int(time.time()),
            port=server_url.port,
            pid=os.getpid(),
            path_prefix=self.flags.path_prefix,
            logdir=self.flags.logdir or self.flags.logdir_spec,
            db=self.flags.db,
            cache_key=self.cache_key,
        )
        atexit.register(manager.remove_info_file)
        manager.write_info_file(info)

    def _install_signal_handler(self, signal_number, signal_name):
        """Set a signal handler to gracefully exit on the given signal.

        When this process receives the given signal, it will run `atexit`
        handlers and then exit with `0`.

        Args:
          signal_number: The numeric code for the signal to handle, like
            `signal.SIGTERM`.
          signal_name: The human-readable signal name.
        """
        # Note to maintainers: Google-internal code overrides this
        # method (cf. cl/334534610). Double-check changes before
        # modifying API.

        old_signal_handler = None  # set below

        def handler(handled_signal_number, frame):
            # In case we catch this signal again while running atexit
            # handlers, take the hint and actually die.
            signal.signal(signal_number, signal.SIG_DFL)
            sys.stderr.write(
                "TensorBoard caught %s; exiting...\n" % signal_name
            )
            # The main thread is the only non-daemon thread, so it suffices to
            # exit hence.
            if old_signal_handler not in (signal.SIG_IGN, signal.SIG_DFL):
                old_signal_handler(handled_signal_number, frame)
            sys.exit(0)

        old_signal_handler = signal.signal(signal_number, handler)

    def _fix_mime_types(self):
        """Fix incorrect entries in the `mimetypes` registry.

        On Windows, the Python standard library's `mimetypes` reads in
        mappings from file extension to MIME type from the Windows
        registry. Other applications can and do write incorrect values
        to this registry, which causes `mimetypes.guess_type` to return
        incorrect values, which causes TensorBoard to fail to render on
        the frontend.

        This method hard-codes the correct mappings for certain MIME
        types that are known to be either used by TensorBoard or
        problematic in general.
        """
        # Known to be problematic when Visual Studio is installed:
        # <https://github.com/tensorflow/tensorboard/issues/3120>
        mimetypes.add_type("text/javascript", ".js")
        # Not known to be problematic, but used by TensorBoard:
        mimetypes.add_type("font/woff2", ".woff2")
        mimetypes.add_type("text/html", ".html")

    def _start_subprocess_data_ingester(self):
        """Creates, starts, and returns a `SubprocessServerDataIngester`."""
        flags = self.flags
        server_binary = server_ingester.get_server_binary()
        ingester = server_ingester.SubprocessServerDataIngester(
            server_binary=server_binary,
            logdir=flags.logdir,
            reload_interval=flags.reload_interval,
            channel_creds_type=flags.grpc_creds_type,
            samples_per_plugin=flags.samples_per_plugin,
            extra_flags=shlex.split(flags.extra_data_server_flags),
        )
        ingester.start()
        return ingester

    def _make_data_ingester(self):
        """Determines the right data ingester, starts it, and returns it."""
        flags = self.flags
        if flags.grpc_data_provider:
            ingester = server_ingester.ExistingServerDataIngester(
                flags.grpc_data_provider,
                channel_creds_type=flags.grpc_creds_type,
            )
            ingester.start()
            return ingester

        if flags.load_fast == "true":
            try:
                return self._start_subprocess_data_ingester()
            except server_ingester.NoDataServerError as e:
                msg = "Option --load_fast=true not available: %s\n" % e
                sys.stderr.write(msg)
                sys.exit(1)
            except server_ingester.DataServerStartupError as e:
                msg = _DATA_SERVER_STARTUP_ERROR_MESSAGE_TEMPLATE % e
                sys.stderr.write(msg)
                sys.exit(1)

        if flags.load_fast == "auto" and _should_use_data_server(flags):
            try:
                ingester = self._start_subprocess_data_ingester()
                sys.stderr.write(_DATA_SERVER_ADVISORY_MESSAGE)
                sys.stderr.flush()
                return ingester
            except server_ingester.NoDataServerError as e:
                logger.info("No data server: %s", e)
            except server_ingester.DataServerStartupError as e:
                logger.info(
                    "Data server error: %s; falling back to multiplexer", e
                )

        ingester = local_ingester.LocalDataIngester(flags)
        ingester.start()
        return ingester

    def _make_data_provider(self):
        """Returns `(data_provider, deprecated_multiplexer)`."""
        ingester = self._make_data_ingester()
        # Stash ingester so that it can avoid GCing Windows file handles.
        # (See comment in `SubprocessServerDataIngester.start` for details.)
        self._ingester = ingester

        deprecated_multiplexer = None
        if isinstance(ingester, local_ingester.LocalDataIngester):
            deprecated_multiplexer = ingester.deprecated_multiplexer
        return (ingester.data_provider, deprecated_multiplexer)

    def _make_server(self):
        """Constructs the TensorBoard WSGI app and instantiates the server."""
        (data_provider, deprecated_multiplexer) = self._make_data_provider()
        app = application.TensorBoardWSGIApp(
            self.flags,
            self.plugin_loaders,
            data_provider,
            self.assets_zip_provider,
            deprecated_multiplexer,
        )
        return self.server_class(app, self.flags)


def _should_use_data_server(flags):
    if flags.logdir_spec and not flags.logdir:
        logger.info(
            "Note: --logdir_spec is not supported with --load_fast behavior; "
            "falling back to slower Python-only load path. To use the data "
            "server, replace --logdir_spec with --logdir."
        )
        return False
    if not flags.logdir:
        # Using some other legacy mode; not supported.
        return False
    if "://" in flags.logdir and not flags.logdir.startswith("gs://"):
        logger.info(
            "Note: --load_fast behavior only supports local and GCS (gs://) "
            "paths; falling back to slower Python-only load path."
        )
        return False
    if flags.detect_file_replacement is True:
        logger.info(
            "Note: --detect_file_replacement=true is not supported with "
            "--load_fast behavior; falling back to slower Python-only load "
            "path."
        )
        return False
    return True


class TensorBoardSubcommand(metaclass=ABCMeta):
    """Experimental private API for defining subcommands for tensorboard.

    The intended use is something like:

    `tensorboard <sub_cmd_name> <additional_args...>`

    Since our hosted service at http://tensorboard.dev has been shut down, this
    functionality is no longer used, but the support for subcommands remains,
    in case it is ever useful again.
    """

    @abstractmethod
    def name(self):
        """Name of this subcommand, as specified on the command line.

        This must be unique across all subcommands.

        Returns:
          A string.
        """
        pass

    @abstractmethod
    def define_flags(self, parser):
        """Configure an argument parser for this subcommand.

        Flags whose names start with two underscores (e.g., `__foo`) are
        reserved for use by the runtime and must not be defined by
        subcommands.

        Args:
          parser: An `argparse.ArgumentParser` scoped to this subcommand,
            which this function should mutate.
        """
        pass

    @abstractmethod
    def run(self, flags):
        """Execute this subcommand with user-provided flags.

        Args:
          flags: An `argparse.Namespace` object with all defined flags.

        Returns:
          An `int` exit code, or `None` as an alias for `0`.
        """
        pass

    def help(self):
        """Short, one-line help text to display on `tensorboard --help`."""
        return None

    def description(self):
        """Description to display on `tensorboard SUBCOMMAND --help`."""
        return None


class TensorBoardServer(metaclass=ABCMeta):
    """Class for customizing TensorBoard WSGI app serving."""

    @abstractmethod
    def __init__(self, wsgi_app, flags):
        """Create a flag-configured HTTP server for TensorBoard's WSGI app.

        Args:
          wsgi_app: The TensorBoard WSGI application to create a server for.
          flags: argparse.Namespace instance of TensorBoard flags.
        """
        raise NotImplementedError()

    @abstractmethod
    def serve_forever(self):
        """Blocking call to start serving the TensorBoard server."""
        raise NotImplementedError()

    @abstractmethod
    def get_url(self):
        """Returns a URL at which this server should be reachable."""
        raise NotImplementedError()

    def print_serving_message(self):
        """Prints a user-friendly message prior to server start.

        This will be called just before `serve_forever`.
        """
        sys.stderr.write(
            "TensorBoard %s at %s (Press CTRL+C to quit)\n"
            % (version.VERSION, self.get_url())
        )
        sys.stderr.flush()


class TensorBoardServerException(Exception):
    """Exception raised by TensorBoardServer for user-friendly errors.

    Subclasses of TensorBoardServer can raise this exception in order to
    generate a clean error message for the user rather than a
    stacktrace.
    """

    def __init__(self, msg):
        self.msg = msg


class TensorBoardPortInUseError(TensorBoardServerException):
    """Error raised when attempting to bind to a port that is in use.

    This should be raised when it is expected that binding to another
    similar port would succeed. It is used as a signal to indicate that
    automatic port searching should continue rather than abort.
    """

    pass


def with_port_scanning(cls):
    """Create a server factory that performs port scanning.

    This function returns a callable whose signature matches the
    specification of `TensorBoardServer.__init__`, using `cls` as an
    underlying implementation. It passes through `flags` unchanged except
    in the case that `flags.port is None`, in which case it repeatedly
    instantiates the underlying server with new port suggestions.

    Args:
      cls: A valid implementation of `TensorBoardServer`. This class's
        initializer should raise a `TensorBoardPortInUseError` upon
        failing to bind to a port when it is expected that binding to
        another nearby port might succeed.

        The initializer for `cls` will only ever be invoked with `flags`
        such that `flags.port is not None`.

    Returns:
      A function that implements the `__init__` contract of
      `TensorBoardServer`.
    """

    def init(wsgi_app, flags):
        # base_port: what's the first port to which we should try to bind?
        # should_scan: if that fails, shall we try additional ports?
        # max_attempts: how many ports shall we try?
        should_scan = flags.port is None
        base_port = (
            core_plugin.DEFAULT_PORT if flags.port is None else flags.port
        )

        if base_port > 0xFFFF:
            raise TensorBoardServerException(
                "TensorBoard cannot bind to port %d > %d" % (base_port, 0xFFFF)
            )
        max_attempts = 100 if should_scan else 1
        base_port = min(base_port + max_attempts, 0x10000) - max_attempts

        for port in range(base_port, base_port + max_attempts):
            subflags = argparse.Namespace(**vars(flags))
            subflags.port = port
            try:
                return cls(wsgi_app=wsgi_app, flags=subflags)
            except TensorBoardPortInUseError:
                if not should_scan:
                    raise
        # All attempts failed to bind.
        raise TensorBoardServerException(
            "TensorBoard could not bind to any port around %s "
            "(tried %d times)" % (base_port, max_attempts)
        )

    return init


class _WSGIRequestHandler(serving.WSGIRequestHandler):
    """Custom subclass of Werkzeug request handler to use HTTP/1.1."""

    # The default on the http.server is HTTP/1.0 for legacy reasons:
    # https://docs.python.org/3/library/http.server.html#http.server.BaseHTTPRequestHandler.protocol_version
    # Override here to use HTTP/1.1 to avoid needing a new TCP socket and Python
    # thread for each HTTP request. The tradeoff is we must always specify the
    # Content-Length header, or do chunked encoding for streaming.
    protocol_version = "HTTP/1.1"


class WerkzeugServer(serving.ThreadedWSGIServer, TensorBoardServer):
    """Implementation of TensorBoardServer using the Werkzeug dev server."""

    # ThreadedWSGIServer handles this in werkzeug 0.12+ but we allow 0.11.x.
    daemon_threads = True

    def __init__(self, wsgi_app, flags):
        self._flags = flags
        host = flags.host
        port = flags.port

        self._auto_wildcard = flags.bind_all
        if self._auto_wildcard:
            # Serve on all interfaces, and attempt to serve both IPv4 and IPv6
            # traffic through one socket.
            host = self._get_wildcard_address(port)
        elif host is None:
            host = "localhost"

        self._host = host
        self._url = None  # Will be set by get_url() below

        self._fix_werkzeug_logging()

        def is_port_in_use(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(("localhost", port)) == 0

        try:
            if is_port_in_use(port):
                raise TensorBoardPortInUseError(
                    "TensorBoard could not bind to port %d, it was already in use"
                    % port
                )
            super().__init__(host, port, wsgi_app, _WSGIRequestHandler)
        except socket.error as e:
            if hasattr(errno, "EACCES") and e.errno == errno.EACCES:
                raise TensorBoardServerException(
                    "TensorBoard must be run as superuser to bind to port %d"
                    % port
                )
            elif hasattr(errno, "EADDRINUSE") and e.errno == errno.EADDRINUSE:
                if port == 0:
                    raise TensorBoardServerException(
                        "TensorBoard unable to find any open port"
                    )
                else:
                    raise TensorBoardPortInUseError(
                        "TensorBoard could not bind to port %d, it was already in use"
                        % port
                    )
            elif (
                hasattr(errno, "EADDRNOTAVAIL")
                and e.errno == errno.EADDRNOTAVAIL
            ):
                raise TensorBoardServerException(
                    "TensorBoard could not bind to unavailable address %s"
                    % host
                )
            elif (
                hasattr(errno, "EAFNOSUPPORT") and e.errno == errno.EAFNOSUPPORT
            ):
                raise TensorBoardServerException(
                    "Tensorboard could not bind to unsupported address family %s"
                    % host
                )
            # Raise the raw exception if it wasn't identifiable as a user error.
            raise

    def _get_wildcard_address(self, port):
        """Returns a wildcard address for the port in question.

        This will attempt to follow the best practice of calling
        getaddrinfo() with a null host and AI_PASSIVE to request a
        server-side socket wildcard address. If that succeeds, this
        returns the first IPv6 address found, or if none, then returns
        the first IPv4 address. If that fails, then this returns the
        hardcoded address "::" if socket.has_ipv6 is True, else
        "0.0.0.0".
        """
        fallback_address = "::" if socket.has_ipv6 else "0.0.0.0"
        if hasattr(socket, "AI_PASSIVE"):
            try:
                addrinfos = socket.getaddrinfo(
                    None,
                    port,
                    socket.AF_UNSPEC,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    socket.AI_PASSIVE,
                )
            except socket.gaierror as e:
                logger.warning(
                    "Failed to auto-detect wildcard address, assuming %s: %s",
                    fallback_address,
                    str(e),
                )
                return fallback_address
            addrs_by_family = defaultdict(list)
            for family, _, _, _, sockaddr in addrinfos:
                # Format of the "sockaddr" socket address varies by address family,
                # but [0] is always the IP address portion.
                addrs_by_family[family].append(sockaddr[0])
            if hasattr(socket, "AF_INET6") and addrs_by_family[socket.AF_INET6]:
                return addrs_by_family[socket.AF_INET6][0]
            if hasattr(socket, "AF_INET") and addrs_by_family[socket.AF_INET]:
                return addrs_by_family[socket.AF_INET][0]
        logger.warning(
            "Failed to auto-detect wildcard address, assuming %s",
            fallback_address,
        )
        return fallback_address

    def server_bind(self):
        """Override to set custom options on the socket."""
        if self._flags.reuse_port:
            try:
                socket.SO_REUSEPORT
            except AttributeError:
                raise TensorBoardServerException(
                    "TensorBoard --reuse_port option is not supported on this platform"
                )
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

        # Enable IPV4 mapping for IPV6 sockets when desired.
        # The main use case for this is so that when no host is specified,
        # TensorBoard can listen on all interfaces for both IPv4 and IPv6
        # connections, rather than having to choose v4 or v6 and hope the
        # browser didn't choose the other one.
        socket_is_v6 = (
            hasattr(socket, "AF_INET6")
            and self.socket.family == socket.AF_INET6
        )
        has_v6only_option = hasattr(socket, "IPPROTO_IPV6") and hasattr(
            socket, "IPV6_V6ONLY"
        )
        if self._auto_wildcard and socket_is_v6 and has_v6only_option:
            try:
                self.socket.setsockopt(
                    socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0
                )
            except socket.error as e:
                # Log a warning on failure to dual-bind, except for EAFNOSUPPORT
                # since that's expected if IPv4 isn't supported at all (IPv6-only).
                if (
                    hasattr(errno, "EAFNOSUPPORT")
                    and e.errno != errno.EAFNOSUPPORT
                ):
                    logger.warning(
                        "Failed to dual-bind to IPv4 wildcard: %s", str(e)
                    )
        super().server_bind()

    def handle_error(self, request, client_address):
        """Override to get rid of noisy EPIPE errors."""
        del request  # unused
        # Kludge to override a SocketServer.py method so we can get rid of noisy
        # EPIPE errors. They're kind of a red herring as far as errors go. For
        # example, `curl -N http://localhost:6006/ | head` will cause an EPIPE.
        exc_info = sys.exc_info()
        e = exc_info[1]
        if isinstance(e, IOError) and e.errno == errno.EPIPE:
            logger.warning(
                "EPIPE caused by %s in HTTP serving" % str(client_address)
            )
        else:
            logger.error("HTTP serving error", exc_info=exc_info)

    def get_url(self):
        if not self._url:
            if self._auto_wildcard:
                display_host = socket.getfqdn()
                # Confirm that the connection is open, otherwise change to `localhost`
                try:
                    socket.create_connection(
                        (display_host, self.server_port), timeout=1
                    )
                except socket.error as e:
                    display_host = "localhost"

            else:
                host = self._host
                display_host = (
                    "[%s]" % host
                    if ":" in host and not host.startswith("[")
                    else host
                )
            self._url = "http://%s:%d%s/" % (
                display_host,
                self.server_port,
                self._flags.path_prefix.rstrip("/"),
            )
        return self._url

    def print_serving_message(self):
        if self._flags.host is None and not self._flags.bind_all:
            sys.stderr.write(
                "Serving TensorBoard on localhost; to expose to the network, "
                "use a proxy or pass --bind_all\n"
            )
            sys.stderr.flush()
        super().print_serving_message()

    def _fix_werkzeug_logging(self):
        """Fix werkzeug logging setup so it inherits TensorBoard's log level.

        This addresses a change in werkzeug 0.15.0+ [1] that causes it set its own
        log level to INFO regardless of the root logger configuration. We instead
        want werkzeug to inherit TensorBoard's root logger log level (set via absl
        to WARNING by default).

        [1]: https://github.com/pallets/werkzeug/commit/4cf77d25858ff46ac7e9d64ade054bf05b41ce12
        """
        # Log once at DEBUG to force werkzeug to initialize its singleton logger,
        # which sets the logger level to INFO it if is unset, and then access that
        # object via logging.getLogger('werkzeug') to durably revert the level to
        # unset (and thus make messages logged to it inherit the root logger level).
        self.log(
            "debug", "Fixing werkzeug logger to inherit TensorBoard log level"
        )
        logging.getLogger("werkzeug").setLevel(logging.NOTSET)


create_port_scanning_werkzeug_server = with_port_scanning(WerkzeugServer)
