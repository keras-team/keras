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
"""Validates responses and their security features."""


import dataclasses

from typing import Collection

from werkzeug.datastructures import Headers
from werkzeug import http

from tensorboard.util import tb_logging

logger = tb_logging.get_logger()

_HTML_MIME_TYPE = "text/html"
_CSP_DEFAULT_SRC = "default-src"
# Whitelist of allowed CSP violations.
_CSP_IGNORE = {
    # Polymer-based code uses unsafe-inline.
    "style-src": ["'unsafe-inline'", "data:"],
    # Used in canvas
    "img-src": ["blob:", "data:"],
    # Used by numericjs.
    # TODO(stephanwlee): remove it eventually.
    "script-src": ["'unsafe-eval'"],
    "font-src": ["data:"],
}


@dataclasses.dataclass(frozen=True)
class Directive:
    """Content security policy directive.

    Loosely follow vocabulary from https://www.w3.org/TR/CSP/#framework-directives.

    Attributes:
      name: A non-empty string.
      value: A collection of non-empty strings.
    """

    name: str
    value: Collection[str]


def _maybe_raise_value_error(error_msg):
    logger.warning("In 3.0, this warning will become an error:\n%s" % error_msg)
    # TODO(3.x): raise a value error.


class SecurityValidatorMiddleware:
    """WSGI middleware validating security on response.

    It validates:
    - responses have Content-Type
    - responses have X-Content-Type-Options: nosniff
    - text/html responses have CSP header. It also validates whether the CSP
      headers pass basic requirement. e.g., default-src should be present, cannot
      use "*" directive, and others. For more complete list, please refer to
      _validate_csp_policies.

    Instances of this class are WSGI applications (see PEP 3333).
    """

    def __init__(self, application):
        """Initializes an `SecurityValidatorMiddleware`.

        Args:
          application: The WSGI application to wrap (see PEP 3333).
        """
        self._application = application

    def __call__(self, environ, start_response):
        def start_response_proxy(status, headers, exc_info=None):
            self._validate_headers(headers)
            return start_response(status, headers, exc_info)

        return self._application(environ, start_response_proxy)

    def _validate_headers(self, headers_list):
        headers = Headers(headers_list)
        self._validate_content_type(headers)
        self._validate_x_content_type_options(headers)
        self._validate_csp_headers(headers)

    def _validate_content_type(self, headers):
        if headers.get("Content-Type"):
            return

        _maybe_raise_value_error("Content-Type is required on a Response")

    def _validate_x_content_type_options(self, headers):
        option = headers.get("X-Content-Type-Options")
        if option == "nosniff":
            return

        _maybe_raise_value_error(
            'X-Content-Type-Options is required to be "nosniff"'
        )

    def _validate_csp_headers(self, headers):
        mime_type, _ = http.parse_options_header(headers.get("Content-Type"))
        if mime_type != _HTML_MIME_TYPE:
            return

        csp_texts = headers.get_all("Content-Security-Policy")
        policies = []

        for csp_text in csp_texts:
            policies += self._parse_serialized_csp(csp_text)

        self._validate_csp_policies(policies)

    def _validate_csp_policies(self, policies):
        has_default_src = False
        violations = []

        for directive in policies:
            name = directive.name
            for value in directive.value:
                has_default_src = has_default_src or name == _CSP_DEFAULT_SRC

                if value in _CSP_IGNORE.get(name, []):
                    # There are cases where certain directives are legitimate.
                    continue

                # TensorBoard follows principle of least privilege. However, to make it
                # easier to conform to the security policy for plugin authors,
                # TensorBoard trusts request and resources originating its server. Also,
                # it can selectively trust domains as long as they use https protocol.
                # Lastly, it can allow 'none' directive.
                # TODO(stephanwlee): allow configuration for whitelist of domains for
                # stricter enforcement.
                # TODO(stephanwlee): deprecate the sha-based whitelisting.
                if (
                    value == "'self'"
                    or value == "'none'"
                    or value.startswith("https:")
                    or value.startswith("'sha256-")
                ):
                    continue

                msg = "Illegal Content-Security-Policy for {name}: {value}".format(
                    name=name, value=value
                )
                violations.append(msg)

        if not has_default_src:
            violations.append(
                "Requires default-src for Content-Security-Policy"
            )

        if violations:
            _maybe_raise_value_error("\n".join(violations))

    def _parse_serialized_csp(self, csp_text):
        # See https://www.w3.org/TR/CSP/#parse-serialized-policy.
        # Below Steps are based on the algorithm stated in above spec.
        # Deviations:
        # - it does not warn and ignore duplicative directive (Step 2.5)

        # Step 2
        csp_srcs = csp_text.split(";")

        policy = []
        for token in csp_srcs:
            # Step 2.1
            token = token.strip()

            if not token:
                # Step 2.2
                continue

            # Step 2.3
            token_frag = token.split(None, 1)
            name = token_frag[0]

            values = token_frag[1] if len(token_frag) == 2 else ""

            # Step 2.4
            name = name.lower()

            # Step 2.6
            value = values.split()
            # Step 2.7
            directive = Directive(name=name, value=value)
            # Step 2.8
            policy.append(directive)

        return policy
