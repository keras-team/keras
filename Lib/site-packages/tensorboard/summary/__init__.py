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
"""Central API entry point for summary operations.

This module exposes summary ops for the standard TensorBoard plugins.
"""

# If the V1 summary API is accessible, load and re-export it here.
try:
    from tensorboard.summary import v1  # noqa: F401
except ImportError:
    pass

# Load the V2 summary API if accessible.
try:
    from tensorboard.summary import v2  # noqa: F401
    from tensorboard.summary.v2 import *  # noqa: F401
except ImportError:
    pass

from tensorboard.summary._output import DirectoryOutput  # noqa: F401
from tensorboard.summary._output import Output  # noqa: F401
from tensorboard.summary._writer import Writer  # noqa: F401
