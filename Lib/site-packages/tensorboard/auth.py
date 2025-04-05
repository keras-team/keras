# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Experimental framework for authentication in TensorBoard."""

import abc


class AuthProvider(metaclass=abc.ABCMeta):
    """Authentication provider for a specific kind of credential."""

    def authenticate(self, environ):
        """Produce an opaque auth token from a WSGI request environment.

        Args:
          environ: A WSGI environment `dict`; see PEP 3333.

        Returns:
          A Python object representing an auth token. The representation
          and semantics depend on the particular `AuthProvider`
          implementation.

        Raises:
          Exception: Any error, usually `tensorboard.errors.PublicError`
            subclasses (like `PermissionDenied`) but also possibly a
            custom error type that should propagate to a WSGI middleware
            for effecting a redirect-driven auth flow.
        """
        pass


class AuthContext:
    """Authentication context within the scope of a single request.

    Auth providers are keyed within an `AuthContext` by arbitrary
    unique keys. It may often make sense for the key used for an
    auth provider to simply be that provider's type object.
    """

    def __init__(self, providers, environ):
        """Create an auth context.

        Args:
          providers: A mapping from provider keys (opaque values) to
            `AuthProvider` implementations.
          environ: A WSGI environment (see PEP 3333).
        """
        self._environ = environ
        self._providers = providers
        self._cache = {}

    @classmethod
    def empty(cls):
        """Create an auth context with no registered providers.

        Returns:
          A new `AuthContext` value for which any call to `get` will
          fail with a `KeyError`.
        """
        # Use an empty dict for the environ. This is not a valid WSGI
        # environment, but it doesn't matter because it's never used.
        return cls({}, {})

    def get(self, provider_key):
        """Get an auth token from the auth provider with the given key.

        If successful, the result will be cached on this auth context.
        If unsuccessful, nothing will be cached, so a future call will
        invoke the underlying `AuthProvider.authenticate` method again.

        This method is not thread-safe. If multiple threads share an
        auth context for a single request, then they must synchronize
        externally when calling this method.

        Returns:
          The result of `provider.authenticate(...)` for the auth
          provider specified by `provider_key`.

        Raises:
          KeyError: If the given `provider_key` does not correspond to
            any registered `AuthProvider`.
          Exception: As raised by the underlying `AuthProvider`.
        """
        provider = self._providers[provider_key]
        sentinel = object()
        value = self._cache.get(provider_key, sentinel)
        if value is not sentinel:
            return value
        value = provider.authenticate(self._environ)
        self._cache[provider_key] = value
        return value
