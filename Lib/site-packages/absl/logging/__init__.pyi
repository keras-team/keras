# Copyright 2017 The Abseil Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Any, Callable, Dict, IO, NoReturn, Optional, Tuple, TypeVar, Union

from absl import flags

# Logging levels.
FATAL: int
ERROR: int
WARNING: int
WARN: int  # Deprecated name.
INFO: int
DEBUG: int

ABSL_LOGGING_PREFIX_REGEX: str

LOGTOSTDERR: flags.FlagHolder[bool]
ALSOLOGTOSTDERR: flags.FlagHolder[bool]
LOG_DIR: flags.FlagHolder[str]
VERBOSITY: flags.FlagHolder[int]
LOGGER_LEVELS: flags.FlagHolder[Dict[str, str]]
STDERRTHRESHOLD: flags.FlagHolder[str]
SHOWPREFIXFORINFO: flags.FlagHolder[bool]


def get_verbosity() -> int:
  ...


def set_verbosity(v: Union[int, str]) -> None:
  ...


def set_stderrthreshold(s: Union[int, str]) -> None:
  ...


# TODO(b/277607978): Provide actual args+kwargs shadowing stdlib's logging functions.
def fatal(msg: Any, *args: Any, **kwargs: Any) -> NoReturn:
  ...


def error(msg: Any, *args: Any, **kwargs: Any) -> None:
  ...


def warning(msg: Any, *args: Any, **kwargs: Any) -> None:
  ...


def warn(msg: Any, *args: Any, **kwargs: Any) -> None:
  ...


def info(msg: Any, *args: Any, **kwargs: Any) -> None:
  ...


def debug(msg: Any, *args: Any, **kwargs: Any) -> None:
  ...


def exception(msg: Any, *args: Any, **kwargs: Any) -> None:
  ...


def log_every_n(
    level: int, msg: Any, n: int, *args: Any, use_call_stack: bool = ...
) -> None:
  ...


def log_every_n_seconds(
    level: int,
    msg: Any,
    n_seconds: float,
    *args: Any,
    use_call_stack: bool = ...
) -> None:
  ...


def log_first_n(
    level: int, msg: Any, n: int, *args: Any, use_call_stack: bool = ...
) -> None:
  ...


def log_if(level: int, msg: Any, condition: Any, *args: Any) -> None:
  ...


def log(level: int, msg: Any, *args: Any, **kwargs: Any) -> None:
  ...


def vlog(level: int, msg: Any, *args: Any, **kwargs: Any) -> None:
  ...


def vlog_is_on(level: int) -> bool:
  ...


def flush() -> None:
  ...


def level_debug() -> bool:
  ...


def level_info() -> bool:
  ...


def level_warning() -> bool:
  ...


level_warn = level_warning  # Deprecated function.


def level_error() -> bool:
  ...


def get_log_file_name(level: int = ...) -> str:
  ...


def find_log_dir_and_names(
    program_name: Optional[str] = ..., log_dir: Optional[str] = ...
) -> Tuple[str, str, str]:
  ...


def find_log_dir(log_dir: Optional[str] = ...) -> str:
  ...


def get_absl_log_prefix(record: logging.LogRecord) -> str:
  ...


_SkipLogT = TypeVar('_SkipLogT', str, Callable[..., Any])

def skip_log_prefix(func: _SkipLogT) -> _SkipLogT:
  ...


_StreamT = TypeVar('_StreamT')


class PythonHandler(logging.StreamHandler[_StreamT]):  # type: ignore[type-var]

  def __init__(
      self,
      stream: Optional[_StreamT] = ...,
      formatter: Optional[logging.Formatter] = ...,
  ) -> None:
    ...

  def start_logging_to_file(
      self, program_name: Optional[str] = ..., log_dir: Optional[str] = ...
  ) -> None:
    ...

  def use_absl_log_file(
      self, program_name: Optional[str] = ..., log_dir: Optional[str] = ...
  ) -> None:
    ...

  def flush(self) -> None:
    ...

  def emit(self, record: logging.LogRecord) -> None:
    ...

  def close(self) -> None:
    ...


class ABSLHandler(logging.Handler):

  def __init__(self, python_logging_formatter: PythonFormatter) -> None:
    ...

  def format(self, record: logging.LogRecord) -> str:
    ...

  def setFormatter(self, fmt) -> None:
    ...

  def emit(self, record: logging.LogRecord) -> None:
    ...

  def flush(self) -> None:
    ...

  def close(self) -> None:
    ...

  def handle(self, record: logging.LogRecord) -> bool:
    ...

  @property
  def python_handler(self) -> PythonHandler:
    ...

  def activate_python_handler(self) -> None:
    ...

  def use_absl_log_file(
      self, program_name: Optional[str] = ..., log_dir: Optional[str] = ...
  ) -> None:
    ...

  def start_logging_to_file(self, program_name=None, log_dir=None) -> None:
    ...


class PythonFormatter(logging.Formatter):

  def format(self, record: logging.LogRecord) -> str:
    ...


class ABSLLogger(logging.Logger):

  def findCaller(
      self, stack_info: bool = ..., stacklevel: int = ...
  ) -> Tuple[str, int, str, Optional[str]]:
    ...

  def critical(self, msg: Any, *args: Any, **kwargs: Any) -> None:
    ...

  def fatal(self, msg: Any, *args: Any, **kwargs: Any) -> NoReturn:  # type: ignore[override]
    ...

  def error(self, msg: Any, *args: Any, **kwargs: Any) -> None:
    ...

  def warn(self, msg: Any, *args: Any, **kwargs: Any) -> None:
    ...

  def warning(self, msg: Any, *args: Any, **kwargs: Any) -> None:
    ...

  def info(self, msg: Any, *args: Any, **kwargs: Any) -> None:
    ...

  def debug(self, msg: Any, *args: Any, **kwargs: Any) -> None:
    ...

  def log(self, level: int, msg: Any, *args: Any, **kwargs: Any) -> None:
    ...

  def handle(self, record: logging.LogRecord) -> None:
    ...

  @classmethod
  def register_frame_to_skip(
      cls, file_name: str, function_name: str, line_number: Optional[int] = ...
  ) -> None:
    ...


# NOTE: Returns None before _initialize called but shouldn't occur after import.
def get_absl_logger() -> ABSLLogger:
  ...


# NOTE: Returns None before _initialize called but shouldn't occur after import.
def get_absl_handler() -> ABSLHandler:
  ...


def use_python_logging(quiet: bool = ...) -> None:
  ...


def use_absl_handler() -> None:
  ...
