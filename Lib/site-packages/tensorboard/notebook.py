# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for using TensorBoard in notebook contexts, like Colab.

These APIs are experimental and subject to change.
"""


import datetime
import errno
import html
import json
import os
import random
import shlex
import textwrap
import time

from tensorboard import manager


# Return values for `_get_context` (see that function's docs for
# details).
_CONTEXT_COLAB = "_CONTEXT_COLAB"
_CONTEXT_IPYTHON = "_CONTEXT_IPYTHON"
_CONTEXT_NONE = "_CONTEXT_NONE"


def _get_context():
    """Determine the most specific context that we're in.

    Returns:
      _CONTEXT_COLAB: If in Colab with an IPython notebook context.
      _CONTEXT_IPYTHON: If not in Colab, but we are in an IPython notebook
        context (e.g., from running `jupyter notebook` at the command
        line).
      _CONTEXT_NONE: Otherwise (e.g., by running a Python script at the
        command-line or using the `ipython` interactive shell).
    """
    # In Colab, the `google.colab` module is available, but the shell
    # returned by `IPython.get_ipython` does not have a `get_trait`
    # method.
    try:
        import google.colab  # noqa: F401
        import IPython
    except ImportError:
        pass
    else:
        if IPython.get_ipython() is not None:
            # We'll assume that we're in a Colab notebook context.
            return _CONTEXT_COLAB

    # In an IPython command line shell or Jupyter notebook, we can
    # directly query whether we're in a notebook context.
    try:
        import IPython
    except ImportError:
        pass
    else:
        ipython = IPython.get_ipython()
        if ipython is not None and ipython.has_trait("kernel"):
            return _CONTEXT_IPYTHON

    # Otherwise, we're not in a known notebook context.
    return _CONTEXT_NONE


def load_ipython_extension(ipython):
    """Deprecated: use `%load_ext tensorboard` instead.

    Raises:
      RuntimeError: Always.
    """
    raise RuntimeError(
        "Use '%load_ext tensorboard' instead of '%load_ext tensorboard.notebook'."
    )


def _load_ipython_extension(ipython):
    """Load the TensorBoard notebook extension.

    Intended to be called from `%load_ext tensorboard`. Do not invoke this
    directly.

    Args:
      ipython: An `IPython.InteractiveShell` instance.
    """
    _register_magics(ipython)


def _register_magics(ipython):
    """Register IPython line/cell magics.

    Args:
      ipython: An `InteractiveShell` instance.
    """
    ipython.register_magic_function(
        _start_magic,
        magic_kind="line",
        magic_name="tensorboard",
    )


def _start_magic(line):
    """Implementation of the `%tensorboard` line magic."""
    return start(line)


def start(args_string):
    """Launch and display a TensorBoard instance as if at the command line.

    Args:
      args_string: Command-line arguments to TensorBoard, to be
        interpreted by `shlex.split`: e.g., "--logdir ./logs --port 0".
        Shell metacharacters are not supported: e.g., "--logdir 2>&1" will
        point the logdir at the literal directory named "2>&1".
    """
    context = _get_context()
    try:
        import IPython
        import IPython.display
    except ImportError:
        IPython = None

    if context == _CONTEXT_NONE:
        handle = None
        print("Launching TensorBoard...")
    else:
        handle = IPython.display.display(
            IPython.display.Pretty("Launching TensorBoard..."),
            display_id=True,
        )

    def print_or_update(message):
        if handle is None:
            print(message)
        else:
            handle.update(IPython.display.Pretty(message))

    parsed_args = shlex.split(args_string, comments=True, posix=True)
    start_result = manager.start(parsed_args)

    if isinstance(start_result, manager.StartLaunched):
        _display(
            port=start_result.info.port,
            print_message=False,
            display_handle=handle,
        )

    elif isinstance(start_result, manager.StartReused):
        template = (
            "Reusing TensorBoard on port {port} (pid {pid}), started {delta} ago. "
            "(Use '!kill {pid}' to kill it.)"
        )
        message = template.format(
            port=start_result.info.port,
            pid=start_result.info.pid,
            delta=_time_delta_from_info(start_result.info),
        )
        print_or_update(message)
        _display(
            port=start_result.info.port,
            print_message=False,
            display_handle=None,
        )

    elif isinstance(start_result, manager.StartFailed):

        def format_stream(name, value):
            if value == "":
                return ""
            elif value is None:
                return "\n<could not read %s>" % name
            else:
                return "\nContents of %s:\n%s" % (name, value.strip())

        message = (
            "ERROR: Failed to launch TensorBoard (exited with %d).%s%s"
            % (
                start_result.exit_code,
                format_stream("stderr", start_result.stderr),
                format_stream("stdout", start_result.stdout),
            )
        )
        print_or_update(message)

    elif isinstance(start_result, manager.StartExecFailed):
        the_tensorboard_binary = (
            "%r (set by the `TENSORBOARD_BINARY` environment variable)"
            % (start_result.explicit_binary,)
            if start_result.explicit_binary is not None
            else "`tensorboard`"
        )
        if start_result.os_error.errno == errno.ENOENT:
            message = (
                "ERROR: Could not find %s. Please ensure that your PATH contains "
                "an executable `tensorboard` program, or explicitly specify the path "
                "to a TensorBoard binary by setting the `TENSORBOARD_BINARY` "
                "environment variable." % (the_tensorboard_binary,)
            )
        else:
            message = "ERROR: Failed to start %s: %s" % (
                the_tensorboard_binary,
                start_result.os_error,
            )
        print_or_update(textwrap.fill(message))

    elif isinstance(start_result, manager.StartTimedOut):
        message = (
            "ERROR: Timed out waiting for TensorBoard to start. "
            "It may still be running as pid %d." % start_result.pid
        )
        print_or_update(message)

    else:
        raise TypeError(
            "Unexpected result from `manager.start`: %r.\n"
            "This is a TensorBoard bug; please report it." % start_result
        )


def _time_delta_from_info(info):
    """Format the elapsed time for the given TensorBoardInfo.

    Args:
      info: A TensorBoardInfo value.

    Returns:
      A human-readable string describing the time since the server
      described by `info` started: e.g., "2 days, 0:48:58".
    """
    delta_seconds = int(time.time()) - info.start_time
    return str(datetime.timedelta(seconds=delta_seconds))


def display(port=None, height=None):
    """Display a TensorBoard instance already running on this machine.

    Args:
      port: The port on which the TensorBoard server is listening, as an
        `int`, or `None` to automatically select the most recently
        launched TensorBoard.
      height: The height of the frame into which to render the TensorBoard
        UI, as an `int` number of pixels, or `None` to use a default value
        (currently 800).
    """
    _display(port=port, height=height, print_message=True, display_handle=None)


def _display(port=None, height=None, print_message=False, display_handle=None):
    """Internal version of `display`.

    Args:
      port: As with `display`.
      height: As with `display`.
      print_message: True to print which TensorBoard instance was selected
        for display (if applicable), or False otherwise.
      display_handle: If not None, an IPython display handle into which to
        render TensorBoard.
    """
    if height is None:
        height = 800

    if port is None:
        infos = manager.get_all()
        if not infos:
            raise ValueError(
                "Can't display TensorBoard: no known instances running."
            )
        else:
            info = max(manager.get_all(), key=lambda x: x.start_time)
            port = info.port
    else:
        infos = [i for i in manager.get_all() if i.port == port]
        info = max(infos, key=lambda x: x.start_time) if infos else None

    if print_message:
        if info is not None:
            message = (
                "Selecting TensorBoard with {data_source} "
                "(started {delta} ago; port {port}, pid {pid})."
            ).format(
                data_source=manager.data_source_from_info(info),
                delta=_time_delta_from_info(info),
                port=info.port,
                pid=info.pid,
            )
            print(message)
        else:
            # The user explicitly provided a port, and we don't have any
            # additional information. There's nothing useful to say.
            pass

    fn = {
        _CONTEXT_COLAB: _display_colab,
        _CONTEXT_IPYTHON: _display_ipython,
        _CONTEXT_NONE: _display_cli,
    }[_get_context()]
    return fn(port=port, height=height, display_handle=display_handle)


def _display_colab(port, height, display_handle):
    """Display a TensorBoard instance in a Colab output frame.

    The Colab VM is not directly exposed to the network, so the Colab
    runtime provides a service worker tunnel to proxy requests from the
    end user's browser through to servers running on the Colab VM: the
    output frame may issue requests to https://localhost:<port> (HTTPS
    only), which will be forwarded to the specified port on the VM.

    It does not suffice to create an `iframe` and let the service worker
    redirect its traffic (`<iframe src="https://localhost:6006">`),
    because for security reasons service workers cannot intercept iframe
    traffic. Instead, we manually fetch the TensorBoard index page with an
    XHR in the output frame, and inject the raw HTML into `document.body`.

    By default, the TensorBoard web app requests resources against
    relative paths, like `./data/logdir`. Within the output frame, these
    requests must instead hit `https://localhost:<port>/data/logdir`. To
    redirect them, we change the document base URI, which transparently
    affects all requests (XHRs and resources alike).
    """
    import IPython.display

    shell = """
        (async () => {
            const url = new URL(await google.colab.kernel.proxyPort(%PORT%, {'cache': true}));
            url.searchParams.set('tensorboardColab', 'true');
            const iframe = document.createElement('iframe');
            iframe.src = url;
            iframe.setAttribute('width', '100%');
            iframe.setAttribute('height', '%HEIGHT%');
            iframe.setAttribute('frameborder', 0);
            document.body.appendChild(iframe);
        })();
    """
    replacements = [
        ("%PORT%", "%d" % port),
        ("%HEIGHT%", "%d" % height),
    ]
    for k, v in replacements:
        shell = shell.replace(k, v)
    script = IPython.display.Javascript(shell)

    if display_handle:
        display_handle.update(script)
    else:
        IPython.display.display(script)


def _display_ipython(port, height, display_handle):
    import IPython.display

    frame_id = "tensorboard-frame-{:08x}".format(random.getrandbits(64))
    shell = """
      <iframe id="%HTML_ID%" width="100%" height="%HEIGHT%" frameborder="0">
      </iframe>
      <script>
        (function() {
          const frame = document.getElementById(%JSON_ID%);
          const url = new URL(%URL%, window.location);
          const port = %PORT%;
          if (port) {
            url.port = port;
          }
          frame.src = url;
        })();
      </script>
    """
    proxy_url = os.environ.get("TENSORBOARD_PROXY_URL")
    if proxy_url is not None:
        # Allow %PORT% in $TENSORBOARD_PROXY_URL
        proxy_url = proxy_url.replace("%PORT%", "%d" % port)
        replacements = [
            ("%HTML_ID%", html.escape(frame_id, quote=True)),
            ("%JSON_ID%", json.dumps(frame_id)),
            ("%HEIGHT%", "%d" % height),
            ("%PORT%", "0"),
            ("%URL%", json.dumps(proxy_url)),
        ]
    else:
        replacements = [
            ("%HTML_ID%", html.escape(frame_id, quote=True)),
            ("%JSON_ID%", json.dumps(frame_id)),
            ("%HEIGHT%", "%d" % height),
            ("%PORT%", "%d" % port),
            ("%URL%", json.dumps("/")),
        ]

    for k, v in replacements:
        shell = shell.replace(k, v)
    iframe = IPython.display.HTML(shell)
    if display_handle:
        display_handle.update(iframe)
    else:
        IPython.display.display(iframe)


def _display_cli(port, height, display_handle):
    del height  # unused
    del display_handle  # unused
    message = "Please visit http://localhost:%d in a web browser." % port
    print(message)


def list():
    """Print a listing of known running TensorBoard instances.

    TensorBoard instances that were killed uncleanly (e.g., with SIGKILL
    or SIGQUIT) may appear in this list even if they are no longer
    running. Conversely, this list may be missing some entries if your
    operating system's temporary directory has been cleared since a
    still-running TensorBoard instance started.
    """
    infos = manager.get_all()
    if not infos:
        print("No known TensorBoard instances running.")
        return

    print("Known TensorBoard instances:")
    for info in infos:
        template = (
            "  - port {port}: {data_source} (started {delta} ago; pid {pid})"
        )
        print(
            template.format(
                port=info.port,
                data_source=manager.data_source_from_info(info),
                delta=_time_delta_from_info(info),
                pid=info.pid,
            )
        )
