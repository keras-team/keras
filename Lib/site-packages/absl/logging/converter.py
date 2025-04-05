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

"""Module to convert log levels between Abseil Python, C++, and Python standard.

This converter has to convert (best effort) between three different
logging level schemes:

  * **cpp**: The C++ logging level scheme used in Abseil C++.
  * **absl**: The absl.logging level scheme used in Abseil Python.
  * **standard**: The python standard library logging level scheme.

Here is a handy ascii chart for easy mental mapping::

    LEVEL    | cpp |  absl  | standard |
    ---------+-----+--------+----------+
    DEBUG    |  0  |    1   |    10    |
    INFO     |  0  |    0   |    20    |
    WARNING  |  1  |   -1   |    30    |
    ERROR    |  2  |   -2   |    40    |
    CRITICAL |  3  |   -3   |    50    |
    FATAL    |  3  |   -3   |    50    |

Note: standard logging ``CRITICAL`` is mapped to absl/cpp ``FATAL``.
However, only ``CRITICAL`` logs from the absl logger (or absl.logging.fatal)
will terminate the program. ``CRITICAL`` logs from non-absl loggers are treated
as error logs with a message prefix ``"CRITICAL - "``.

Converting from standard to absl or cpp is a lossy conversion.
Converting back to standard will lose granularity.  For this reason,
users should always try to convert to standard, the richest
representation, before manipulating the levels, and then only to cpp
or absl if those level schemes are absolutely necessary.
"""

import logging

STANDARD_CRITICAL = logging.CRITICAL
STANDARD_ERROR = logging.ERROR
STANDARD_WARNING = logging.WARNING
STANDARD_INFO = logging.INFO
STANDARD_DEBUG = logging.DEBUG

# These levels are also used to define the constants
# FATAL, ERROR, WARNING, INFO, and DEBUG in the
# absl.logging module.
ABSL_FATAL = -3
ABSL_ERROR = -2
ABSL_WARNING = -1
ABSL_WARN = -1  # Deprecated name.
ABSL_INFO = 0
ABSL_DEBUG = 1

ABSL_LEVELS = {ABSL_FATAL: 'FATAL',
               ABSL_ERROR: 'ERROR',
               ABSL_WARNING: 'WARNING',
               ABSL_INFO: 'INFO',
               ABSL_DEBUG: 'DEBUG'}

# Inverts the ABSL_LEVELS dictionary
ABSL_NAMES = {'FATAL': ABSL_FATAL,
              'ERROR': ABSL_ERROR,
              'WARNING': ABSL_WARNING,
              'WARN': ABSL_WARNING,  # Deprecated name.
              'INFO': ABSL_INFO,
              'DEBUG': ABSL_DEBUG}

ABSL_TO_STANDARD = {ABSL_FATAL: STANDARD_CRITICAL,
                    ABSL_ERROR: STANDARD_ERROR,
                    ABSL_WARNING: STANDARD_WARNING,
                    ABSL_INFO: STANDARD_INFO,
                    ABSL_DEBUG: STANDARD_DEBUG}

# Inverts the ABSL_TO_STANDARD
STANDARD_TO_ABSL = {v: k for (k, v) in ABSL_TO_STANDARD.items()}


def get_initial_for_level(level):
  """Gets the initial that should start the log line for the given level.

  It returns:

  * ``'I'`` when: ``level < STANDARD_WARNING``.
  * ``'W'`` when: ``STANDARD_WARNING <= level < STANDARD_ERROR``.
  * ``'E'`` when: ``STANDARD_ERROR <= level < STANDARD_CRITICAL``.
  * ``'F'`` when: ``level >= STANDARD_CRITICAL``.

  Args:
    level: int, a Python standard logging level.

  Returns:
    The first initial as it would be logged by the C++ logging module.
  """
  if level < STANDARD_WARNING:
    return 'I'
  elif level < STANDARD_ERROR:
    return 'W'
  elif level < STANDARD_CRITICAL:
    return 'E'
  else:
    return 'F'


def absl_to_cpp(level):
  """Converts an absl log level to a cpp log level.

  Args:
    level: int, an absl.logging level.

  Raises:
    TypeError: Raised when level is not an integer.

  Returns:
    The corresponding integer level for use in Abseil C++.
  """
  if not isinstance(level, int):
    raise TypeError(f'Expect an int level, found {type(level)}')
  if level >= 0:
    # C++ log levels must be >= 0
    return 0
  else:
    return -level


def absl_to_standard(level):
  """Converts an integer level from the absl value to the standard value.

  Args:
    level: int, an absl.logging level.

  Raises:
    TypeError: Raised when level is not an integer.

  Returns:
    The corresponding integer level for use in standard logging.
  """
  if not isinstance(level, int):
    raise TypeError(f'Expect an int level, found {type(level)}')
  if level < ABSL_FATAL:
    level = ABSL_FATAL
  if level <= ABSL_DEBUG:
    return ABSL_TO_STANDARD[level]
  # Maps to vlog levels.
  return STANDARD_DEBUG - level + 1


def string_to_standard(level):
  """Converts a string level to standard logging level value.

  Args:
    level: str, case-insensitive ``'debug'``, ``'info'``, ``'warning'``,
        ``'error'``, ``'fatal'``.

  Returns:
    The corresponding integer level for use in standard logging.
  """
  return absl_to_standard(ABSL_NAMES.get(level.upper()))


def standard_to_absl(level):
  """Converts an integer level from the standard value to the absl value.

  Args:
    level: int, a Python standard logging level.

  Raises:
    TypeError: Raised when level is not an integer.

  Returns:
    The corresponding integer level for use in absl logging.
  """
  if not isinstance(level, int):
    raise TypeError(f'Expect an int level, found {type(level)}')
  if level < 0:
    level = 0
  if level < STANDARD_DEBUG:
    # Maps to vlog levels.
    return STANDARD_DEBUG - level + 1
  elif level < STANDARD_INFO:
    return ABSL_DEBUG
  elif level < STANDARD_WARNING:
    return ABSL_INFO
  elif level < STANDARD_ERROR:
    return ABSL_WARNING
  elif level < STANDARD_CRITICAL:
    return ABSL_ERROR
  else:
    return ABSL_FATAL


def standard_to_cpp(level):
  """Converts an integer level from the standard value to the cpp value.

  Args:
    level: int, a Python standard logging level.

  Raises:
    TypeError: Raised when level is not an integer.

  Returns:
    The corresponding integer level for use in cpp logging.
  """
  return absl_to_cpp(standard_to_absl(level))
