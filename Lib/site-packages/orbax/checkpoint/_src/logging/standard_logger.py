# Copyright 2024 The Orbax Authors.
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

"""Standard logging implementation for checkpointing."""

from typing import Any

from absl import logging
from orbax.checkpoint._src.logging import abstract_logger


class StandardLogger(abstract_logger.AbstractLogger):
  """Logging implementation utilizing an absl logging API."""

  def log_entry(self, msg: Any, *args: Any, **kwargs: Any):
    """Logs an informational message.

    Args:
        msg (Any): The main log message.
        *args(Any):  Additional data to be included in the log record.
        **kwargs: Additional structured data to be included in the log record.
    """
    logging.info(msg, *args, **kwargs)
