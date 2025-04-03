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

"""Functionality for processing events."""

from typing import Optional

from tensorboard.compat.proto import event_pb2
from tensorboard.util import tb_logging

logger = tb_logging.get_logger()

# Maxmimum length for event writer name.
_MAX_WRITER_NAME_LEN = 128


def ParseFileVersion(file_version: str) -> float:
    """Convert the string file_version in event.proto into a float.

    Args:
      file_version: String file_version from event.proto

    Returns:
      Version number as a float.
    """
    tokens = file_version.split("brain.Event:")
    try:
        return float(tokens[-1])
    except ValueError:
        ## This should never happen according to the definition of file_version
        ## specified in event.proto.
        logger.warning(
            (
                "Invalid event.proto file_version. Defaulting to use of "
                "out-of-order event.step logic for purging expired events."
            )
        )
        return -1


def GetSourceWriter(
    source_metadata: event_pb2.SourceMetadata,
) -> Optional[str]:
    """Gets the source writer name from the source metadata proto."""
    writer_name = source_metadata.writer
    if not writer_name:
        return None
    # Checks the length of the writer name.
    if len(writer_name) > _MAX_WRITER_NAME_LEN:
        logger.error(
            "Source writer name `%s` is too long, maximum allowed length is %d.",
            writer_name,
            _MAX_WRITER_NAME_LEN,
        )
        return None
    return writer_name
