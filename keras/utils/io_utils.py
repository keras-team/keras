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

"""Utilities related to disk I/O."""

import os
import sys
import threading

from absl import logging

from keras.utils import keras_logging

# isort: off
from tensorflow.python.util.tf_export import keras_export

INTERACTIVE_LOGGING = threading.local()
INTERACTIVE_LOGGING.enable = keras_logging.INTERACTIVE_LOGGING_DEFAULT


@keras_export("keras.utils.enable_interactive_logging")
def enable_interactive_logging():
    """Turn on interactive logging.

    When interactive logging is enabled, Keras displays logs via stdout.
    This provides the best experience when using Keras in an interactive
    environment such as a shell or a notebook.
    """
    INTERACTIVE_LOGGING.enable = True


@keras_export("keras.utils.disable_interactive_logging")
def disable_interactive_logging():
    """Turn off interactive logging.

    When interactive logging is disabled, Keras sends logs to `absl.logging`.
    This is the best option when using Keras in a non-interactive
    way, such as running a training or inference job on a server.
    """
    INTERACTIVE_LOGGING.enable = False


@keras_export("keras.utils.is_interactive_logging_enabled")
def is_interactive_logging_enabled():
    """Check if interactive logging is enabled.

    To switch between writing logs to stdout and `absl.logging`, you may use
    `keras.utils.enable_interactive_logging()` and
    `keras.utils.disable_interactie_logging()`.

    Returns:
      Boolean (True if interactive logging is enabled and False otherwise).
    """
    # Use `getattr` in case `INTERACTIVE_LOGGING`
    # does not have the `enable` attribute.
    return getattr(
        INTERACTIVE_LOGGING, "enable", keras_logging.INTERACTIVE_LOGGING_DEFAULT
    )


def print_msg(message, line_break=True):
    """Print the message to absl logging or stdout."""
    if is_interactive_logging_enabled():
        if line_break:
            sys.stdout.write(message + "\n")
        else:
            sys.stdout.write(message)
        sys.stdout.flush()
    else:
        logging.info(message)


def path_to_string(path):
    """Convert `PathLike` objects to their string representation.

    If given a non-string typed path object, converts it to its string
    representation.

    If the object passed to `path` is not among the above, then it is
    returned unchanged. This allows e.g. passthrough of file objects
    through this function.

    Args:
      path: `PathLike` object that represents a path

    Returns:
      A string representation of the path argument, if Python support exists.
    """
    if isinstance(path, os.PathLike):
        return os.fspath(path)
    return path


def ask_to_proceed_with_overwrite(filepath):
    """Produces a prompt asking about overwriting a file.

    Args:
        filepath: the path to the file to be overwritten.

    Returns:
        True if we can proceed with overwrite, False otherwise.
    """
    overwrite = (
        input(f"[WARNING] {filepath} already exists - overwrite? [y/n]")
        .strip()
        .lower()
    )
    while overwrite not in ("y", "n"):
        overwrite = (
            input('Enter "y" (overwrite) or "n" (cancel).').strip().lower()
        )
    if overwrite == "n":
        return False
    print_msg("[TIP] Next time specify overwrite=True!")
    return True
