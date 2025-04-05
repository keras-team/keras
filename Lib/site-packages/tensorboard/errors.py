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
"""Error codes (experimental module).

These types represent a subset of the Google error codes [1], which are
also used by TensorFlow, gRPC, et al.

When an HTTP handler raises one of these errors, the TensorBoard core
application will catch it and automatically serve a properly formatted
response with the error message.

[1]: https://github.com/googleapis/googleapis/blob/master/google/rpc/code.proto
"""


class PublicError(RuntimeError):
    """An error whose text does not contain sensitive information.

    Fields:
      http_code: Integer between 400 and 599 inclusive (e.g., 404).
      headers: List of additional key-value pairs to include in the
        response body, like `[("Allow", "GET")]` for HTTP 405 or
        `[("WWW-Authenticate", "Digest")]` for HTTP 401. May be empty.
    """

    http_code = 500  # default; subclasses should override

    def __init__(self, details):
        super().__init__(details)
        self.headers = []


class InvalidArgumentError(PublicError):
    """Client specified an invalid argument.

    The text of this error is assumed not to contain sensitive data,
    and so may appear in (e.g.) the response body of a failed HTTP
    request.

    Corresponds to HTTP 400 Bad Request or Google error code `INVALID_ARGUMENT`.
    """

    http_code = 400

    def __init__(self, details=None):
        msg = _format_message("Invalid argument", details)
        super().__init__(msg)


class NotFoundError(PublicError):
    """Some requested entity (e.g., file or directory) was not found.

    The text of this error is assumed not to contain sensitive data,
    and so may appear in (e.g.) the response body of a failed HTTP
    request.

    Corresponds to HTTP 404 Not Found or Google error code `NOT_FOUND`.
    """

    http_code = 404

    def __init__(self, details=None):
        msg = _format_message("Not found", details)
        super().__init__(msg)


class UnauthenticatedError(PublicError):
    """Request does not have valid authentication credentials for the operation.

    The text of this error is assumed not to contain sensitive data,
    and so may appear in (e.g.) the response body of a failed HTTP
    request.

    Corresponds to HTTP 401 Unauthorized (despite the name) or Google
    error code `UNAUTHENTICATED`. HTTP 401 responses are required to
    contain a `WWW-Authenticate` challenge, so `UnauthenticatedError`
    values are, too.
    """

    http_code = 401

    def __init__(self, details=None, *, challenge):
        """Initialize an `UnauthenticatedError`.

        Args;
          details: Optional public, user-facing error message, as a
            string or any value that can be converted to string, or
            `None` to omit details.
          challenge: String value of the `WWW-Authenticate` HTTP header
            as described in RFC 7235.
        """
        msg = _format_message("Unauthenticated", details)
        super().__init__(msg)
        self.headers.append(("WWW-Authenticate", challenge))


class PermissionDeniedError(PublicError):
    """The caller does not have permission to execute the specified operation.

    The text of this error is assumed not to contain sensitive data,
    and so may appear in (e.g.) the response body of a failed HTTP
    request.

    Corresponds to HTTP 403 Forbidden or Google error code `PERMISSION_DENIED`.
    """

    http_code = 403

    def __init__(self, details=None):
        msg = _format_message("Permission denied", details)
        super().__init__(msg)


def _format_message(code_name, details):
    if details is None:
        return code_name
    else:
        return "%s: %s" % (code_name, details)
