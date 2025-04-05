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

"""Composite logging implementation for checkpointing."""

from typing import Any
from absl import logging

from orbax.checkpoint._src.logging import abstract_logger


def is_valid_logger(logger: abstract_logger.AbstractLogger):
  if not isinstance(logger, abstract_logger.AbstractLogger):
    logging.warning(
        'CompositeLogger only supports AbstractLogger instances.'
    )
    return False
  if isinstance(logger, CompositeLogger):
    logging.warning(
        'CompositeLogger doesn\'t support instance of itself.'
    )
    return False
  return True


class CompositeLogger(abstract_logger.AbstractLogger):
  """Composite logger implementation."""

  def __init__(
      self,
      *loggers: abstract_logger.AbstractLogger,
  ):
    """CompositeLogger constructor.

    Args:
      *loggers: List of loggers to be used.
    """
    self._known_loggers: list[abstract_logger.AbstractLogger] = []
    for logger in loggers:
      if is_valid_logger(logger):
        self._known_loggers.append(logger)

  def log_entry(self, entry: Any):
    """Logs an informational message.

    Args:
        entry: Additional structured data to be included in the log record.
    """
    for logger in self._known_loggers:
      logger.log_entry(entry)
