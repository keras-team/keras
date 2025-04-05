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
"""Helpers for TensorBoard main module."""

import os
import sys

import absl.logging

from tensorboard.compat import tf


_TBDEV_SHUTDOWN_MESSAGE = """\
======================================================================
ERROR: The `tensorboard dev` command is no longer available.

TensorBoard.dev has been shut down. For further information,
see the FAQ at <https://tensorboard.dev/>.
======================================================================
"""


def global_init():
    """Modifies the global environment for running TensorBoard as main.

    This functions changes global state in the Python process, so it should
    not be called from library routines.
    """
    # TF versions prior to 1.15.0 included default GCS filesystem caching logic
    # that interacted pathologically with the pattern of reads used by TensorBoard
    # for logdirs. See: https://github.com/tensorflow/tensorboard/issues/1225
    # The problematic behavior was fixed in 1.15.0 by
    # https://github.com/tensorflow/tensorflow/commit/e43b94649d3e1ac5d538e4eca9166b899511d681
    # but for older versions of TF, we avoid a regression by setting this env var to
    # disable the cache, which must be done before the first import of tensorflow.
    os.environ["GCS_READ_CACHE_DISABLED"] = "1"

    if getattr(tf, "__version__", "stub") == "stub":
        print(
            "TensorFlow installation not found - running with reduced feature set.",
            file=sys.stderr,
        )

    # Only emit log messages at WARNING and above by default to reduce spam.
    absl.logging.set_verbosity(absl.logging.WARNING)

    # Intercept attempts to invoke `tensorboard dev` and print turndown message.
    if sys.argv[1:] and sys.argv[1] == "dev":
        sys.stderr.write(_TBDEV_SHUTDOWN_MESSAGE)
        sys.stderr.flush()
        sys.exit(1)
