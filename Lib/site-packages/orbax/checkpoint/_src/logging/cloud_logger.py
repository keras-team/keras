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

"""Cloud logging implementation for checkpointing."""

import dataclasses
from typing import Any, Optional

import google.cloud.logging as google_cloud_logging
from orbax.checkpoint._src.logging import abstract_logger

_JOB_NAME = 'checkpoint_job'
_LOGGER_NAME = 'checkpoint_logger'


@dataclasses.dataclass
class CloudLoggerOptions:
  """Logger options for Checkpoint loggers."""

  job_name: str = _JOB_NAME
  logger_name: str = _LOGGER_NAME
  client: Optional[google_cloud_logging.Client] = None
  checkpoint_metadata: Optional[Any] = None


def _get_severity(severity: str) -> str:
  """Gets the severity of the log message.

  Args:
    severity: The severity of the log message.

  Returns:
    The severity of the log message.
  """
  if severity == 'DEBUG':
    return 'DEBUG'
  elif severity == 'INFO':
    return 'INFO'
  elif severity == 'WARNING':
    return 'WARNING'
  elif severity == 'ERROR':
    return 'ERROR'
  elif severity == 'CRITICAL':
    return 'CRITICAL'
  else:
    return 'INFO'


class CloudLogger(abstract_logger.AbstractLogger):
  """Logging implementation utilizing a cloud logging API."""

  def __init__(
      self,
      options: CloudLoggerOptions = CloudLoggerOptions()
  ):
    """CloudLogger constructor.

    Args:
      options: Options for the logger.
    """
    self.job_name = options.job_name
    self.log_name = options.logger_name
    self.options = options
    if options.client is None:
      self.logging_client = google_cloud_logging.Client()
    else:
      self.logging_client = options.client
    self._logger = self.logging_client.logger(self.log_name)

  def log_entry(
      self, entry: dict[str, Any], severity: Optional[str] = 'INFO'
  ) -> None:
    """Logs a structured message at the given severity.

    Args:
      entry: Dictionary to be logged.
      severity: The severity of the log message.
    """
    entry['job_name'] = self.job_name
    self._logger.log_struct(
        entry,
        severity=_get_severity(severity),
    )
