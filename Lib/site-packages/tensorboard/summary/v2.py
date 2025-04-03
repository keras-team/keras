# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Central API entry point for v2 versions of summary operations.

This module exposes v2 summary ops for the standard TensorBoard plugins.
"""

# pylint: disable=unused-import
from tensorboard.plugins.audio.summary_v2 import audio  # noqa: F401
from tensorboard.plugins.histogram.summary_v2 import histogram  # noqa: F401
from tensorboard.plugins.image.summary_v2 import image  # noqa: F401
from tensorboard.plugins.scalar.summary_v2 import scalar  # noqa: F401
from tensorboard.plugins.text.summary_v2 import text  # noqa: F401
