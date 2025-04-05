# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
from tensorboard import auth
from tensorboard import context


class AuthContextMiddleware:
    """WSGI middleware to inject an AuthContext into the RequestContext."""

    def __init__(self, application, auth_providers):
        """Initializes this middleware.

        Args:
          application: A WSGI application to delegate to.
          auth_providers: The auth_providers to provide to the AuthContext.
        """
        self._application = application
        self._auth_providers = auth_providers

    def __call__(self, environ, start_response):
        """Invoke this WSGI application."""
        environ = dict(environ)
        auth_ctx = auth.AuthContext(self._auth_providers, environ)
        ctx = context.from_environ(environ).replace(auth=auth_ctx)
        context.set_in_environ(environ, ctx)
        return self._application(environ, start_response)
