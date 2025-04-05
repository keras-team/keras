# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Bindings for TensorBoard frontend assets."""

import os

from tensorboard.util import tb_logging

logger = tb_logging.get_logger()


def get_default_assets_zip_provider():
    """Try to get a function to provide frontend assets.

    Returns:
      Either (a) a callable that takes no arguments and returns an open
      file handle to a Zip archive of frontend assets, or (b) `None`, if
      the frontend assets cannot be found.
    """
    path = os.path.join(os.path.dirname(__file__), "webfiles.zip")
    if not os.path.exists(path):
        logger.warning("webfiles.zip static assets not found: %s", path)
        return None
    return lambda: open(path, "rb")
