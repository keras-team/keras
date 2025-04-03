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
"""Application-level experiment ID support."""


import re


# Value of the first path component that signals that the second path
# component represents an experiment ID.
_EXPERIMENT_PATH_COMPONENT = "experiment"

# Key into the WSGI environment used for the experiment ID.
WSGI_ENVIRON_KEY = "HTTP_TENSORBOARD_EXPERIMENT_ID"


class ExperimentIdMiddleware:
    """WSGI middleware extracting experiment IDs from URL to environment.

    Any request whose path matches `/experiment/SOME_EID[/...]` will have
    its first two path components stripped, and its experiment ID stored
    onto the WSGI environment with key taken from the `WSGI_ENVIRON_KEY`
    constant. All other requests will have paths unchanged and the
    experiment ID set to the empty string. It noops if the key taken from
    the `WSGI_ENVIRON_KEY` is already present in the environment.

    Instances of this class are WSGI applications (see PEP 3333).
    """

    def __init__(self, application):
        """Initializes an `ExperimentIdMiddleware`.

        Args:
          application: The WSGI application to wrap (see PEP 3333).
        """
        self._application = application
        # Regular expression that matches the whole `/experiment/EID` prefix
        # (without any trailing slash) and captures the experiment ID.
        self._pat = re.compile(
            r"/%s/([^/]*)" % re.escape(_EXPERIMENT_PATH_COMPONENT)
        )

    def __call__(self, environ, start_response):
        # Skip ExperimentIdMiddleware was already called.
        if WSGI_ENVIRON_KEY in environ:
            return self._application(environ, start_response)

        path = environ.get("PATH_INFO", "")
        m = self._pat.match(path)
        if m:
            eid = m.group(1)
            new_path = path[m.end(0) :]
            root = m.group(0)
        else:
            eid = ""
            new_path = path
            root = ""
        environ[WSGI_ENVIRON_KEY] = eid
        environ["PATH_INFO"] = new_path
        environ["SCRIPT_NAME"] = environ.get("SCRIPT_NAME", "") + root
        return self._application(environ, start_response)
