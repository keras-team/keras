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
"""Redirect from an empty path to the virtual application root.

Sometimes, middleware transformations will make the path empty: for
example, navigating to "/foo" (no trailing slash) when the path prefix
is exactly "/foo". In such cases, relative links on the frontend would
break. Instead of handling this special case in each relevant
middleware, we install a top-level redirect handler from "" to "/".

This middleware respects `SCRIPT_NAME` as described by the WSGI spec. If
`SCRIPT_NAME` is set to "/foo", then an empty `PATH_INFO` corresponds to
the actual path "/foo", and so will be redirected to "/foo/".
"""


class EmptyPathRedirectMiddleware:
    """WSGI middleware to redirect from "" to "/"."""

    def __init__(self, application):
        """Initializes this middleware.

        Args:
          application: The WSGI application to wrap (see PEP 3333).
        """
        self._application = application

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        if path:
            return self._application(environ, start_response)
        location = environ.get("SCRIPT_NAME", "") + "/"
        start_response("301 Moved Permanently", [("Location", location)])
        return []
