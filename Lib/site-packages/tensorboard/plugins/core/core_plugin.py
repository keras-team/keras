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
"""TensorBoard core plugin package."""


import argparse
import functools
import gzip
import io
import mimetypes
import posixpath
import zipfile

from werkzeug import utils
from werkzeug import wrappers

from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin
from tensorboard.util import grpc_util
from tensorboard.util import tb_logging
from tensorboard import version

logger = tb_logging.get_logger()


# If no port is specified, try to bind to this port. See help for --port
# for more details.
DEFAULT_PORT = 6006
# Valid javascript mimetypes that we have seen configured, in practice.
# Historically (~2020-2022) we saw "application/javascript" exclusively but with
# RFC 9239 (https://www.rfc-editor.org/rfc/rfc9239) we saw some systems quickly
# transition to 'text/javascript'.
JS_MIMETYPES = ["text/javascript", "application/javascript"]
JS_CACHE_EXPIRATION_IN_SECS = 86400


class CorePlugin(base_plugin.TBPlugin):
    """Core plugin for TensorBoard.

    This plugin serves runs, configuration data, and static assets. This
    plugin should always be present in a TensorBoard WSGI application.
    """

    plugin_name = "core"

    def __init__(self, context, include_debug_info=None):
        """Instantiates CorePlugin.

        Args:
          context: A base_plugin.TBContext instance.
          include_debug_info: If true, `/data/environment` will include some
            basic information like the TensorBoard server version. Disabled by
            default to prevent surprising information leaks in custom builds of
            TensorBoard.
        """
        self._flags = context.flags
        logdir_spec = context.flags.logdir_spec if context.flags else ""
        self._logdir = context.logdir or logdir_spec
        self._window_title = context.window_title
        self._path_prefix = context.flags.path_prefix if context.flags else None
        self._assets_zip_provider = context.assets_zip_provider
        self._data_provider = context.data_provider
        self._include_debug_info = bool(include_debug_info)

    def is_active(self):
        return True

    def get_plugin_apps(self):
        apps = {
            "/___rPc_sWiTcH___": self._send_404_without_logging,
            "/audio": self._redirect_to_index,
            "/data/environment": self._serve_environment,
            "/data/logdir": self._serve_logdir,
            "/data/runs": self._serve_runs,
            "/data/experiments": self._serve_experiments,
            "/data/experiment_runs": self._serve_experiment_runs,
            "/data/notifications": self._serve_notifications,
            "/data/window_properties": self._serve_window_properties,
            "/events": self._redirect_to_index,
            "/favicon.ico": self._send_404_without_logging,
            "/graphs": self._redirect_to_index,
            "/histograms": self._redirect_to_index,
            "/images": self._redirect_to_index,
        }
        apps.update(self.get_resource_apps())
        return apps

    def get_resource_apps(self):
        apps = {}
        if not self._assets_zip_provider:
            return apps

        with self._assets_zip_provider() as fp:
            with zipfile.ZipFile(fp) as zip_:
                for path in zip_.namelist():
                    content = zip_.read(path)
                    # Opt out of gzipping index.html
                    if path == "index.html":
                        apps["/" + path] = functools.partial(
                            self._serve_index, content
                        )
                        continue

                    gzipped_asset_bytes = _gzip(content)
                    wsgi_app = functools.partial(
                        self._serve_asset, path, gzipped_asset_bytes
                    )
                    apps["/" + path] = wsgi_app
        apps["/"] = apps["/index.html"]
        return apps

    @wrappers.Request.application
    def _send_404_without_logging(self, request):
        return http_util.Respond(request, "Not found", "text/plain", code=404)

    @wrappers.Request.application
    def _redirect_to_index(self, unused_request):
        return utils.redirect("/")

    @wrappers.Request.application
    def _serve_asset(self, path, gzipped_asset_bytes, request):
        """Serves a pre-gzipped static asset from the zip file."""
        mimetype = mimetypes.guess_type(path)[0] or "application/octet-stream"

        # Cache JS resources while keep others do not cache.
        expires = (
            JS_CACHE_EXPIRATION_IN_SECS
            if request.args.get("_file_hash") and mimetype in JS_MIMETYPES
            else 0
        )

        return http_util.Respond(
            request,
            gzipped_asset_bytes,
            mimetype,
            content_encoding="gzip",
            expires=expires,
        )

    @wrappers.Request.application
    def _serve_index(self, index_asset_bytes, request):
        """Serves index.html content.

        Note that we opt out of gzipping index.html to write preamble before the
        resource content. This inflates the resource size from 2x kiB to 1xx
        kiB, but we require an ability to flush preamble with the HTML content.
        """
        relpath = (
            posixpath.relpath(self._path_prefix, request.script_root)
            if self._path_prefix
            else "."
        )
        meta_header = (
            '<!doctype html><meta name="tb-relative-root" content="%s/">'
            % relpath
        )
        content = meta_header.encode("utf-8") + index_asset_bytes
        # By passing content_encoding, disallow gzipping. Bloats the content
        # from ~25 kiB to ~120 kiB but reduces CPU usage and avoid 3ms worth of
        # gzipping.
        return http_util.Respond(
            request, content, "text/html", content_encoding="identity"
        )

    @wrappers.Request.application
    def _serve_environment(self, request):
        """Serve a JSON object describing the TensorBoard parameters."""
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        md = self._data_provider.experiment_metadata(
            ctx, experiment_id=experiment
        )

        environment = {
            "version": version.VERSION,
            "data_location": md.data_location,
            "window_title": self._window_title,
            "experiment_name": md.experiment_name,
            "experiment_description": md.experiment_description,
            "creation_time": md.creation_time,
        }
        if self._include_debug_info:
            environment["debug"] = {
                "data_provider": str(self._data_provider),
                "flags": self._render_flags(),
            }
        return http_util.Respond(
            request,
            environment,
            "application/json",
        )

    def _render_flags(self):
        """Return a JSON-and-human-friendly version of `self._flags`.

        Like `json.loads(json.dumps(self._flags, default=str))` but
        without the wasteful serialization overhead.
        """
        if self._flags is None:
            return None

        def go(x):
            if isinstance(x, (type(None), str, int, float)):
                return x
            if isinstance(x, (list, tuple)):
                return [go(v) for v in x]
            if isinstance(x, dict):
                return {str(k): go(v) for (k, v) in x.items()}
            return str(x)

        return go(vars(self._flags))

    @wrappers.Request.application
    def _serve_logdir(self, request):
        """Respond with a JSON object containing this TensorBoard's logdir."""
        # TODO(chihuahua): Remove this method once the frontend instead uses the
        # /data/environment route (and no deps throughout Google use the
        # /data/logdir route).
        return http_util.Respond(
            request, {"logdir": self._logdir}, "application/json"
        )

    @wrappers.Request.application
    def _serve_window_properties(self, request):
        """Serve a JSON object containing this TensorBoard's window
        properties."""
        # TODO(chihuahua): Remove this method once the frontend instead uses the
        # /data/environment route.
        return http_util.Respond(
            request, {"window_title": self._window_title}, "application/json"
        )

    @wrappers.Request.application
    def _serve_runs(self, request):
        """Serve a JSON array of run names, ordered by run started time.

        Sort order is by started time (aka first event time) with empty
        times sorted last, and then ties are broken by sorting on the
        run name.
        """
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        runs = sorted(
            self._data_provider.list_runs(ctx, experiment_id=experiment),
            key=lambda run: (
                run.start_time if run.start_time is not None else float("inf"),
                run.run_name,
            ),
        )
        run_names = [run.run_name for run in runs]
        return http_util.Respond(request, run_names, "application/json")

    @wrappers.Request.application
    def _serve_experiments(self, request):
        """Serve a JSON array of experiments.

        Experiments are ordered by experiment started time (aka first
        event time) with empty times sorted last, and then ties are
        broken by sorting on the experiment name.
        """
        results = self.list_experiments_impl()
        return http_util.Respond(request, results, "application/json")

    def list_experiments_impl(self):
        return []

    @wrappers.Request.application
    def _serve_experiment_runs(self, request):
        """Serve a JSON runs of an experiment, specified with query param
        `experiment`, with their nested data, tag, populated.

        Runs returned are ordered by started time (aka first event time)
        with empty times sorted last, and then ties are broken by
        sorting on the run name. Tags are sorted by its name,
        displayName, and lastly, inserted time.
        """
        results = []
        return http_util.Respond(request, results, "application/json")

    @wrappers.Request.application
    def _serve_notifications(self, request):
        """Serve JSON payload of notifications to show in the UI."""
        response = utils.redirect("../notifications_note.json")
        # Disable Werkzeug's automatic Location header correction routine, which
        # absolutizes relative paths "to be RFC conformant" [1], but this is
        # based on an outdated HTTP/1.1 RFC; the current one allows them:
        # https://tools.ietf.org/html/rfc7231#section-7.1.2
        response.autocorrect_location_header = False
        return response


class CorePluginLoader(base_plugin.TBLoader):
    """CorePlugin factory."""

    def __init__(self, include_debug_info=None):
        self._include_debug_info = include_debug_info

    def define_flags(self, parser):
        """Adds standard TensorBoard CLI flags to parser."""
        parser.add_argument(
            "--logdir",
            metavar="PATH",
            type=str,
            default="",
            help="""\
Directory where TensorBoard will look to find TensorFlow event files
that it can display. TensorBoard will recursively walk the directory
structure rooted at logdir, looking for .*tfevents.* files.

A leading tilde will be expanded with the semantics of Python's
os.expanduser function.
""",
        )

        parser.add_argument(
            "--logdir_spec",
            metavar="PATH_SPEC",
            type=str,
            default="",
            help="""\
Like `--logdir`, but with special interpretation for commas and colons:
commas separate multiple runs, where a colon specifies a new name for a
run. For example:
`tensorboard --logdir_spec=name1:/path/to/logs/1,name2:/path/to/logs/2`.

This flag is discouraged and can usually be avoided. TensorBoard walks
log directories recursively; for finer-grained control, prefer using a
symlink tree. Some features may not work when using `--logdir_spec`
instead of `--logdir`.
""",
        )

        parser.add_argument(
            "--host",
            metavar="ADDR",
            type=str,
            default=None,  # like localhost, but prints a note about `--bind_all`
            help="""\
What host to listen to (default: localhost). To serve to the entire local
network on both IPv4 and IPv6, see `--bind_all`, with which this option is
mutually exclusive.
""",
        )

        parser.add_argument(
            "--bind_all",
            action="store_true",
            help="""\
Serve on all public interfaces. This will expose your TensorBoard instance to
the network on both IPv4 and IPv6 (where available). Mutually exclusive with
`--host`.
""",
        )

        parser.add_argument(
            "--port",
            metavar="PORT",
            type=lambda s: (None if s == "default" else int(s)),
            default="default",
            help="""\
Port to serve TensorBoard on. Pass 0 to request an unused port selected
by the operating system, or pass "default" to try to bind to the default
port (%s) but search for a nearby free port if the default port is
unavailable. (default: "default").\
"""
            % DEFAULT_PORT,
        )

        parser.add_argument(
            "--reuse_port",
            metavar="BOOL",
            # Custom str-to-bool converter since regular bool() doesn't work.
            type=lambda v: {"true": True, "false": False}.get(v.lower(), v),
            choices=[True, False],
            default=False,
            help="""\
Enables the SO_REUSEPORT option on the socket opened by TensorBoard's HTTP
server, for platforms that support it. This is useful in cases when a parent
process has obtained the port already and wants to delegate access to the
port to TensorBoard as a subprocess.(default: %(default)s).\
""",
        )

        parser.add_argument(
            "--load_fast",
            type=str,
            default="auto",
            choices=["false", "auto", "true"],
            help="""\
Use alternate mechanism to load data. Typically 100x faster or more, but only
available on some platforms and invocations. Defaults to "auto" to use this new
mode only if available, otherwise falling back to the legacy loading path. Set
to "true" to suppress the advisory note and hard-fail if the fast codepath is
not available. Set to "false" to always fall back. Feedback/issues:
https://github.com/tensorflow/tensorboard/issues/4784
(default: %(default)s)
""",
        )

        parser.add_argument(
            "--extra_data_server_flags",
            type=str,
            default="",
            help="""\
Experimental. With `--load_fast`, pass these additional command-line flags to
the data server. Subject to POSIX word splitting per `shlex.split`. Meant for
debugging; not officially supported.
""",
        )

        parser.add_argument(
            "--grpc_creds_type",
            type=grpc_util.ChannelCredsType,
            default=grpc_util.ChannelCredsType.LOCAL,
            choices=grpc_util.ChannelCredsType.choices(),
            help="""\
Experimental. The type of credentials to use to connect to the data server.
(default: %(default)s)
""",
        )

        parser.add_argument(
            "--grpc_data_provider",
            metavar="PORT",
            type=str,
            default="",
            help="""\
Experimental. Address of a gRPC server exposing a data provider. Set to empty
string to disable. (default: %(default)s)
""",
        )

        parser.add_argument(
            "--purge_orphaned_data",
            metavar="BOOL",
            # Custom str-to-bool converter since regular bool() doesn't work.
            type=lambda v: {"true": True, "false": False}.get(v.lower(), v),
            choices=[True, False],
            default=True,
            help="""\
Whether to purge data that may have been orphaned due to TensorBoard
restarts. Setting --purge_orphaned_data=False can be used to debug data
disappearance. (default: %(default)s)\
""",
        )

        parser.add_argument(
            "--db",
            metavar="URI",
            type=str,
            default="",
            help="""\
[experimental] sets SQL database URI and enables DB backend mode, which is
read-only unless --db_import is also passed.\
""",
        )

        parser.add_argument(
            "--db_import",
            action="store_true",
            help="""\
[experimental] enables DB read-and-import mode, which in combination with
--logdir imports event files into a DB backend on the fly. The backing DB is
temporary unless --db is also passed to specify a DB path to use.\
""",
        )

        parser.add_argument(
            "--inspect",
            action="store_true",
            help="""\
Prints digests of event files to command line.

This is useful when no data is shown on TensorBoard, or the data shown
looks weird.

Must specify one of `logdir` or `event_file` flag.

Example usage:
  `tensorboard --inspect --logdir mylogdir --tag loss`

See tensorboard/backend/event_processing/event_file_inspector.py for more info.\
""",
        )

        # This flag has a "_tb" suffix to avoid conflicting with an internal flag
        # named --version.  Note that due to argparse auto-expansion of unambiguous
        # flag prefixes, you can still invoke this as `tensorboard --version`.
        parser.add_argument(
            "--version_tb",
            action="store_true",
            help="Prints the version of Tensorboard",
        )

        parser.add_argument(
            "--tag",
            metavar="TAG",
            type=str,
            default="",
            help="tag to query for; used with --inspect",
        )

        parser.add_argument(
            "--event_file",
            metavar="PATH",
            type=str,
            default="",
            help="""\
The particular event file to query for. Only used if --inspect is
present and --logdir is not specified.\
""",
        )

        parser.add_argument(
            "--path_prefix",
            metavar="PATH",
            type=str,
            default="",
            help="""\
An optional, relative prefix to the path, e.g. "/path/to/tensorboard".
resulting in the new base url being located at
localhost:6006/path/to/tensorboard under default settings. A leading
slash is required when specifying the path_prefix. A trailing slash is
optional and has no effect. The path_prefix can be leveraged for path
based routing of an ELB when the website base_url is not available e.g.
"example.site.com/path/to/tensorboard/".\
""",
        )

        parser.add_argument(
            "--window_title",
            metavar="TEXT",
            type=str,
            default="",
            help="changes title of browser window",
        )

        parser.add_argument(
            "--max_reload_threads",
            metavar="COUNT",
            type=int,
            default=1,
            help="""\
The max number of threads that TensorBoard can use to reload runs. Not
relevant for db read-only mode. Each thread reloads one run at a time.
(default: %(default)s)\
""",
        )

        parser.add_argument(
            "--reload_interval",
            metavar="SECONDS",
            type=_nonnegative_float,
            default=5.0,
            help="""\
How often the backend should load more data, in seconds. Set to 0 to
load just once at startup. Must be non-negative. (default: %(default)s)\
""",
        )

        parser.add_argument(
            "--reload_task",
            metavar="TYPE",
            type=str,
            default="auto",
            choices=["auto", "thread", "process", "blocking"],
            help="""\
[experimental] The mechanism to use for the background data reload task.
The default "auto" option will conditionally use threads for legacy reloading
and a child process for DB import reloading. The "process" option is only
useful with DB import mode. The "blocking" option will block startup until
reload finishes, and requires --load_interval=0. (default: %(default)s)\
""",
        )

        parser.add_argument(
            "--reload_multifile",
            metavar="BOOL",
            # Custom str-to-bool converter since regular bool() doesn't work.
            type=lambda v: {"true": True, "false": False}.get(v.lower(), v),
            choices=[True, False],
            default=None,
            help="""\
[experimental] If true, this enables experimental support for continuously
polling multiple event files in each run directory for newly appended data
(rather than only polling the last event file). Event files will only be
polled as long as their most recently read data is newer than the threshold
defined by --reload_multifile_inactive_secs, to limit resource usage. Beware
of running out of memory if the logdir contains many active event files.
(default: false)\
""",
        )

        parser.add_argument(
            "--reload_multifile_inactive_secs",
            metavar="SECONDS",
            type=int,
            default=86400,
            help="""\
[experimental] Configures the age threshold in seconds at which an event file
that has no event wall time more recent than that will be considered an
inactive file and no longer polled (to limit resource usage). If set to -1,
no maximum age will be enforced, but beware of running out of memory and
heavier filesystem read traffic. If set to 0, this reverts to the older
last-file-only polling strategy (akin to --reload_multifile=false).
(default: %(default)s - intended to ensure an event file remains active if
it receives new data at least once per 24 hour period)\
""",
        )

        parser.add_argument(
            "--generic_data",
            metavar="TYPE",
            type=str,
            default="auto",
            choices=["false", "auto", "true"],
            help="""\
[experimental] Hints whether plugins should read from generic data
provider infrastructure. For plugins that support only the legacy
multiplexer APIs or only the generic data APIs, this option has no
effect. The "auto" option enables this only for plugins that are
considered to have stable support for generic data providers. (default:
%(default)s)\
""",
        )

        parser.add_argument(
            "--samples_per_plugin",
            type=_parse_samples_per_plugin,
            default="",
            help="""\
An optional comma separated list of plugin_name=num_samples pairs to
explicitly specify how many samples to keep per tag for that plugin. For
unspecified plugins, TensorBoard randomly downsamples logged summaries
to reasonable values to prevent out-of-memory errors for long running
jobs. This flag allows fine control over that downsampling. Note that if a
plugin is not specified in this list, a plugin-specific default number of
samples will be enforced. (for example, 10 for images, 500 for histograms,
and 1000 for scalars). Most users should not need to set this flag.\
""",
        )

        parser.add_argument(
            "--detect_file_replacement",
            metavar="BOOL",
            # Custom str-to-bool converter since regular bool() doesn't work.
            type=lambda v: {"true": True, "false": False}.get(v.lower(), v),
            choices=[True, False],
            default=None,
            help="""\
[experimental] If true, this enables experimental support for detecting when
event files are replaced with new versions that contain additional data. This is
not needed in the normal case where new data is either appended to an existing
file or written to a brand new file, but it arises, for example, when using
rsync without the --inplace option, in which new versions of the original file
are first written to a temporary file, then swapped into the final location.

This option is currently incompatible with --load_fast=true, and if passed will
disable fast-loading mode. (default: false)\
""",
        )

    def fix_flags(self, flags):
        """Fixes standard TensorBoard CLI flags to parser."""
        FlagsError = base_plugin.FlagsError
        if flags.version_tb:
            pass
        elif flags.inspect:
            if flags.logdir_spec:
                raise FlagsError(
                    "--logdir_spec is not supported with --inspect."
                )
            if flags.logdir and flags.event_file:
                raise FlagsError(
                    "Must specify either --logdir or --event_file, but not both."
                )
            if not (flags.logdir or flags.event_file):
                raise FlagsError(
                    "Must specify either --logdir or --event_file."
                )
        elif flags.logdir and flags.logdir_spec:
            raise FlagsError("May not specify both --logdir and --logdir_spec")
        elif (
            not flags.db
            and not flags.logdir
            and not flags.logdir_spec
            and not flags.grpc_data_provider
        ):
            raise FlagsError(
                "A logdir or db must be specified. "
                "For example `tensorboard --logdir mylogdir` "
                "or `tensorboard --db sqlite:~/.tensorboard.db`. "
                "Run `tensorboard --helpfull` for details and examples."
            )
        elif flags.host is not None and flags.bind_all:
            raise FlagsError("Must not specify both --host and --bind_all.")
        elif (
            flags.load_fast == "true" and flags.detect_file_replacement is True
        ):
            raise FlagsError(
                "Must not specify both --load_fast=true and"
                "--detect_file_replacement=true"
            )

        flags.path_prefix = flags.path_prefix.rstrip("/")
        if flags.path_prefix and not flags.path_prefix.startswith("/"):
            raise FlagsError(
                "Path prefix must start with slash, but got: %r."
                % flags.path_prefix
            )

    def load(self, context):
        """Creates CorePlugin instance."""
        return CorePlugin(context, include_debug_info=self._include_debug_info)


def _gzip(bytestring):
    out = io.BytesIO()
    # Set mtime to zero for deterministic results across TensorBoard launches.
    with gzip.GzipFile(fileobj=out, mode="wb", compresslevel=3, mtime=0) as f:
        f.write(bytestring)
    return out.getvalue()


def _parse_samples_per_plugin(value):
    """Parses `value` as a string-to-int dict in the form `foo=12,bar=34`."""
    result = {}
    for token in value.split(","):
        if token:
            k, v = token.strip().split("=")
            result[k] = int(v)
    return result


def _nonnegative_float(v):
    try:
        v = float(v)
    except ValueError:
        raise argparse.ArgumentTypeError("invalid float: %r" % v)
    if not (v >= 0):  # no NaNs, please
        raise argparse.ArgumentTypeError("must be non-negative: %r" % v)
    return v
