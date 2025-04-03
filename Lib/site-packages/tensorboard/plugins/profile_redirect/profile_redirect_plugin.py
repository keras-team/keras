# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Plugin that only displays a message with installation instructions."""


from tensorboard.plugins import base_plugin


class ProfileRedirectPluginLoader(base_plugin.TBLoader):
    """Load the redirect notice iff the dynamic plugin is unavailable."""

    def load(self, context):
        try:
            import tensorboard_plugin_profile  # noqa: F401

            # If we successfully load the dynamic plugin, don't show
            # this redirect plugin at all.
            return None
        except ImportError:
            return _ProfileRedirectPlugin(context)


class _ProfileRedirectPlugin(base_plugin.TBPlugin):
    """Redirect notice pointing users to the new dynamic profile plugin."""

    plugin_name = "profile_redirect"

    def get_plugin_apps(self):
        return {}

    def is_active(self):
        return False

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(
            element_name="tf-profile-redirect-dashboard",
            tab_name="Profile",
        )
