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
"""Internal path prefix support for TensorBoard.

Using a path prefix of `/foo/bar` enables TensorBoard to serve from
`http://localhost:6006/foo/bar/` rather than `http://localhost:6006/`.
See the `--path_prefix` flag docs for more details.
"""


from tensorboard import errors


class PathPrefixMiddleware:
    """WSGI middleware for path prefixes.

    All requests to this middleware must begin with the specified path
    prefix (otherwise, a 404 will be returned immediately). Requests
    will be forwarded to the underlying application with the path prefix
    stripped and appended to `SCRIPT_NAME` (see the WSGI spec, PEP 3333,
    for details).
    """

    def __init__(self, application, path_prefix):
        """Initializes this middleware.

        Args:
          application: The WSGI application to wrap (see PEP 3333).
          path_prefix: A string path prefix to be stripped from incoming
            requests. If empty, this middleware is a no-op. If non-empty,
            the path prefix must start with a slash and not end with one
            (e.g., "/tensorboard").
        """
        if path_prefix.endswith("/"):
            raise ValueError(
                "Path prefix must not end with slash: %r" % path_prefix
            )
        if path_prefix and not path_prefix.startswith("/"):
            raise ValueError(
                "Non-empty path prefix must start with slash: %r" % path_prefix
            )
        self._application = application
        self._path_prefix = path_prefix
        self._strict_prefix = self._path_prefix + "/"

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        if path != self._path_prefix and not path.startswith(
            self._strict_prefix
        ):
            raise errors.NotFoundError()
        environ["PATH_INFO"] = path[len(self._path_prefix) :]
        environ["SCRIPT_NAME"] = (
            environ.get("SCRIPT_NAME", "") + self._path_prefix
        )
        return self._application(environ, start_response)
