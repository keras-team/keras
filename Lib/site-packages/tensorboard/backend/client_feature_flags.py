# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Middleware for injecting client-side feature flags into the Context."""

import json
import urllib.parse

from tensorboard import context
from tensorboard import errors


class ClientFeatureFlagsMiddleware:
    """Middleware for injecting client-side feature flags into the Context.

    The client webapp is expected to include a json-serialized version of its
    FeatureFlags in the `X-TensorBoard-Feature-Flags` header or the
    `tensorBoardFeatureFlags` query parameter. This middleware extracts the
    header or query parameter value and converts it into the client_feature_flags
    property for the DataProvider's Context object, where client_feature_flags
    is a Dict of string keys and arbitrary value types.

    In the event that both the header and query parameter are specified, the
    values from the header will take precedence.
    """

    def __init__(self, application):
        """Initializes this middleware.

        Args:
          application: The WSGI application to wrap (see PEP 3333).
        """
        self._application = application

    def __call__(self, environ, start_response):
        header_feature_flags = self._parse_potential_header_param_flags(
            environ.get("HTTP_X_TENSORBOARD_FEATURE_FLAGS")
        )
        query_string_feature_flags = self._parse_potential_query_param_flags(
            environ.get("QUERY_STRING")
        )

        if not header_feature_flags and not query_string_feature_flags:
            return self._application(environ, start_response)

        # header flags take precedence
        for flag, value in header_feature_flags.items():
            query_string_feature_flags[flag] = value

        ctx = context.from_environ(environ).replace(
            client_feature_flags=query_string_feature_flags
        )
        context.set_in_environ(environ, ctx)

        return self._application(environ, start_response)

    def _parse_potential_header_param_flags(self, header_string):
        if not header_string:
            return {}

        try:
            header_feature_flags = json.loads(header_string)
        except json.JSONDecodeError:
            raise errors.InvalidArgumentError(
                "X-TensorBoard-Feature-Flags cannot be JSON decoded."
            )

        if not isinstance(header_feature_flags, dict):
            raise errors.InvalidArgumentError(
                "X-TensorBoard-Feature-Flags cannot be decoded to a dict."
            )

        return header_feature_flags

    def _parse_potential_query_param_flags(self, query_string):
        if not query_string:
            return {}

        try:
            query_string_json = urllib.parse.parse_qs(query_string)
        except ValueError:
            return {}

        # parse_qs returns the dictionary values as lists for each name.
        potential_feature_flags = query_string_json.get(
            "tensorBoardFeatureFlags", []
        )
        if not potential_feature_flags:
            return {}
        try:
            client_feature_flags = json.loads(potential_feature_flags[0])
        except json.JSONDecodeError:
            raise errors.InvalidArgumentError(
                "tensorBoardFeatureFlags cannot be JSON decoded."
            )

        if not isinstance(client_feature_flags, dict):
            raise errors.InvalidArgumentError(
                "tensorBoardFeatureFlags cannot be decoded to a dict."
            )

        return client_feature_flags
