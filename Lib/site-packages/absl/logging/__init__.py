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

"""Abseil Python logging module implemented on top of standard logging.

Simple usage::

    from absl import logging

    logging.info('Interesting Stuff')
    logging.info('Interesting Stuff with Arguments: %d', 42)

    logging.set_verbosity(logging.INFO)
    logging.log(logging.DEBUG, 'This will *not* be printed')
    logging.set_verbosity(logging.DEBUG)
    logging.log(logging.DEBUG, 'This will be printed')

    logging.warning('Worrying Stuff')
    logging.error('Alarming Stuff')
    logging.fatal('AAAAHHHHH!!!!')  # Process exits.

Usage note: Do not pre-format the strings in your program code.
Instead, let the logging module perform argument interpolation.
This saves cycles because strings that don't need to be printed
are never formatted.  Note that this module does not attempt to
interpolate arguments when no arguments are given.  In other words::

    logging.info('Interesting Stuff: %s')

does not raise an exception because logging.info() has only one
argument, the message string.

"Lazy" evaluation for debugging
-------------------------------

If you do something like this::

    logging.debug('Thing: %s', thing.ExpensiveOp())

then the ExpensiveOp will be evaluated even if nothing
is printed to the log. To avoid this, use the level_debug() function::

  if logging.level_debug():
    logging.debug('Thing: %s', thing.ExpensiveOp())

Per file level logging is supported by logging.vlog() and
logging.vlog_is_on(). For example::

    if logging.vlog_is_on(2):
      logging.vlog(2, very_expensive_debug_message())

Notes on Unicode
----------------

The log output is encoded as UTF-8.  Don't pass data in other encodings in
bytes() instances -- instead pass unicode string instances when you need to
(for both the format string and arguments).

Note on critical and fatal:
Standard logging module defines fatal as an alias to critical, but it's not
documented, and it does NOT actually terminate the program.
This module only defines fatal but not critical, and it DOES terminate the
program.

The differences in behavior are historical and unfortunate.
"""

import collections
from collections import abc
import getpass
import io
import inspect
import itertools
import logging
import os
import socket
import struct
import sys
import tempfile
import threading
import time
import timeit
import traceback
import types
import warnings

from absl import flags
from absl.logging import converter

# pylint: disable=g-import-not-at-top
try:
  from typing import NoReturn
except ImportError:
  pass

# pylint: enable=g-import-not-at-top

FLAGS = flags.FLAGS


# Logging levels.
FATAL = converter.ABSL_FATAL
ERROR = converter.ABSL_ERROR
WARNING = converter.ABSL_WARNING
WARN = converter.ABSL_WARNING  # Deprecated name.
INFO = converter.ABSL_INFO
DEBUG = converter.ABSL_DEBUG

# Regex to match/parse log line prefixes.
ABSL_LOGGING_PREFIX_REGEX = (
    r'^(?P<severity>[IWEF])'
    r'(?P<month>\d\d)(?P<day>\d\d) '
    r'(?P<hour>\d\d):(?P<minute>\d\d):(?P<second>\d\d)'
    r'\.(?P<microsecond>\d\d\d\d\d\d) +'
    r'(?P<thread_id>-?\d+) '
    r'(?P<filename>[a-zA-Z<][\w._<>-]+):(?P<line>\d+)')


# Mask to convert integer thread ids to unsigned quantities for logging purposes
_THREAD_ID_MASK = 2 ** (struct.calcsize('L') * 8) - 1

# Extra property set on the LogRecord created by ABSLLogger when its level is
# CRITICAL/FATAL.
_ABSL_LOG_FATAL = '_absl_log_fatal'
# Extra prefix added to the log message when a non-absl logger logs a
# CRITICAL/FATAL message.
_CRITICAL_PREFIX = 'CRITICAL - '

# Used by findCaller to skip callers from */logging/__init__.py.
_LOGGING_FILE_PREFIX = os.path.join('logging', '__init__.')

# The ABSL logger instance, initialized in _initialize().
_absl_logger = None
# The ABSL handler instance, initialized in _initialize().
_absl_handler = None


_CPP_NAME_TO_LEVELS = {
    'debug': '0',  # Abseil C++ has no DEBUG level, mapping it to INFO here.
    'info': '0',
    'warning': '1',
    'warn': '1',
    'error': '2',
    'fatal': '3'
}

_CPP_LEVEL_TO_NAMES = {
    '0': 'info',
    '1': 'warning',
    '2': 'error',
    '3': 'fatal',
}


class _VerbosityFlag(flags.Flag):
  """Flag class for -v/--verbosity."""

  def __init__(self, *args, **kwargs):
    super().__init__(
        flags.IntegerParser(), flags.ArgumentSerializer(), *args, **kwargs
    )

  @property
  def value(self):
    return self._value

  @value.setter
  def value(self, v):
    self._value = v
    self._update_logging_levels()

  def _update_logging_levels(self):
    """Updates absl logging levels to the current verbosity.

    Visibility: module-private
    """
    if not _absl_logger:
      return

    if self._value <= converter.ABSL_DEBUG:
      standard_verbosity = converter.absl_to_standard(self._value)
    else:
      # --verbosity is set to higher than 1 for vlog.
      standard_verbosity = logging.DEBUG - (self._value - 1)

    # Also update root level when absl_handler is used.
    if _absl_handler in logging.root.handlers:
      # Make absl logger inherit from the root logger. absl logger might have
      # a non-NOTSET value if logging.set_verbosity() is called at import time.
      _absl_logger.setLevel(logging.NOTSET)
      logging.root.setLevel(standard_verbosity)
    else:
      _absl_logger.setLevel(standard_verbosity)


class _LoggerLevelsFlag(flags.Flag):
  """Flag class for --logger_levels."""

  def __init__(self, *args, **kwargs):
    super().__init__(
        _LoggerLevelsParser(), _LoggerLevelsSerializer(), *args, **kwargs
    )

  @property
  def value(self):
    # For lack of an immutable type, be defensive and return a copy.
    # Modifications to the dict aren't supported and won't have any affect.
    # While Py3 could use MappingProxyType, that isn't deepcopy friendly, so
    # just return a copy.
    return self._value.copy()

  @value.setter
  def value(self, v):
    self._value = {} if v is None else v
    self._update_logger_levels()

  def _update_logger_levels(self):
    # Visibility: module-private.
    # This is called by absl.app.run() during initialization.
    for name, level in self._value.items():
      logging.getLogger(name).setLevel(level)


class _LoggerLevelsParser(flags.ArgumentParser):
  """Parser for --logger_levels flag."""

  def parse(self, value):
    if isinstance(value, abc.Mapping):
      return value

    pairs = [pair.strip() for pair in value.split(',') if pair.strip()]

    # Preserve the order so that serialization is deterministic.
    levels = collections.OrderedDict()
    for name_level in pairs:
      name, level = name_level.split(':', 1)
      name = name.strip()
      level = level.strip()
      levels[name] = level
    return levels


class _LoggerLevelsSerializer:
  """Serializer for --logger_levels flag."""

  def serialize(self, value):
    if isinstance(value, str):
      return value
    return ','.join(f'{name}:{level}' for name, level in value.items())


class _StderrthresholdFlag(flags.Flag):
  """Flag class for --stderrthreshold."""

  def __init__(self, *args, **kwargs):
    super().__init__(
        flags.ArgumentParser(), flags.ArgumentSerializer(), *args, **kwargs
    )

  @property
  def value(self):
    return self._value

  @value.setter
  def value(self, v):
    if v in _CPP_LEVEL_TO_NAMES:
      # --stderrthreshold also accepts numeric strings whose values are
      # Abseil C++ log levels.
      cpp_value = int(v)
      v = _CPP_LEVEL_TO_NAMES[v]  # Normalize to strings.
    elif v.lower() in _CPP_NAME_TO_LEVELS:
      v = v.lower()
      if v == 'warn':
        v = 'warning'  # Use 'warning' as the canonical name.
      cpp_value = int(_CPP_NAME_TO_LEVELS[v])
    else:
      raise ValueError(
          '--stderrthreshold must be one of (case-insensitive) '
          "'debug', 'info', 'warning', 'error', 'fatal', "
          "or '0', '1', '2', '3', not '%s'" % v)

    self._value = v


LOGTOSTDERR = flags.DEFINE_boolean(
    'logtostderr',
    False,
    'Should only log to stderr?',
    allow_override_cpp=True,
)
ALSOLOGTOSTDERR = flags.DEFINE_boolean(
    'alsologtostderr',
    False,
    'also log to stderr?',
    allow_override_cpp=True,
)
LOG_DIR = flags.DEFINE_string(
    'log_dir',
    os.getenv('TEST_TMPDIR', ''),
    'directory to write logfiles into',
    allow_override_cpp=True,
)
VERBOSITY = flags.DEFINE_flag(
    _VerbosityFlag(
        'verbosity',
        -1,
        (
            'Logging verbosity level. Messages logged at this level or lower'
            ' will be included. Set to 1 for debug logging. If the flag was not'
            ' set or supplied, the value will be changed from the default of -1'
            ' (warning) to 0 (info) after flags are parsed.'
        ),
        short_name='v',
        allow_hide_cpp=True,
    )
)
LOGGER_LEVELS = flags.DEFINE_flag(
    _LoggerLevelsFlag(
        'logger_levels',
        {},
        (
            'Specify log level of loggers. The format is a CSV list of '
            '`name:level`. Where `name` is the logger name used with '
            '`logging.getLogger()`, and `level` is a level name  (INFO, DEBUG, '
            'etc). e.g. `myapp.foo:INFO,other.logger:DEBUG`'
        ),
    )
)
STDERRTHRESHOLD = flags.DEFINE_flag(
    _StderrthresholdFlag(
        'stderrthreshold',
        'fatal',
        (
            'log messages at this level, or more severe, to stderr in '
            'addition to the logfile.  Possible values are '
            "'debug', 'info', 'warning', 'error', and 'fatal'.  "
            'Obsoletes --alsologtostderr. Using --alsologtostderr '
            'cancels the effect of this flag. Please also note that '
            'this flag is subject to --verbosity and requires logfile '
            'not be stderr.'
        ),
        allow_hide_cpp=True,
    )
)
SHOWPREFIXFORINFO = flags.DEFINE_boolean(
    'showprefixforinfo',
    True,
    (
        'If False, do not prepend prefix to info messages '
        "when it's logged to stderr, "
        '--verbosity is set to INFO level, '
        'and python logging is used.'
    ),
)


def get_verbosity():
  """Returns the logging verbosity."""
  return FLAGS['verbosity'].value


def set_verbosity(v):
  """Sets the logging verbosity.

  Causes all messages of level <= v to be logged,
  and all messages of level > v to be silently discarded.

  Args:
    v: int|str, the verbosity level as an integer or string. Legal string values
        are those that can be coerced to an integer as well as case-insensitive
        'debug', 'info', 'warning', 'error', and 'fatal'.
  """
  try:
    new_level = int(v)
  except ValueError:
    new_level = converter.ABSL_NAMES[v.upper()]
  FLAGS.verbosity = new_level


def set_stderrthreshold(s):
  """Sets the stderr threshold to the value passed in.

  Args:
    s: str|int, valid strings values are case-insensitive 'debug',
        'info', 'warning', 'error', and 'fatal'; valid integer values are
        logging.DEBUG|INFO|WARNING|ERROR|FATAL.

  Raises:
      ValueError: Raised when s is an invalid value.
  """
  if s in converter.ABSL_LEVELS:
    FLAGS.stderrthreshold = converter.ABSL_LEVELS[s]
  elif isinstance(s, str) and s.upper() in converter.ABSL_NAMES:
    FLAGS.stderrthreshold = s
  else:
    raise ValueError(
        'set_stderrthreshold only accepts integer absl logging level '
        'from -3 to 1, or case-insensitive string values '
        "'debug', 'info', 'warning', 'error', and 'fatal'. "
        'But found "{}" ({}).'.format(s, type(s)))


def fatal(msg, *args, **kwargs):
  # type: (Any, Any, Any) -> NoReturn
  """Logs a fatal message."""
  log(FATAL, msg, *args, **kwargs)


def error(msg, *args, **kwargs):
  """Logs an error message."""
  log(ERROR, msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
  """Logs a warning message."""
  log(WARNING, msg, *args, **kwargs)


def warn(msg, *args, **kwargs):
  """Deprecated, use 'warning' instead."""
  warnings.warn("The 'warn' function is deprecated, use 'warning' instead",
                DeprecationWarning, 2)
  log(WARNING, msg, *args, **kwargs)


def info(msg, *args, **kwargs):
  """Logs an info message."""
  log(INFO, msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
  """Logs a debug message."""
  log(DEBUG, msg, *args, **kwargs)


def exception(msg, *args, exc_info=True, **kwargs):
  """Logs an exception, with traceback and message."""
  error(msg, *args, exc_info=exc_info, **kwargs)


def _fast_stack_trace():
  """A fast stack trace that gets us the minimal information we need.

  Compared to using `get_absl_logger().findCaller(stack_info=True)`, this
  function is ~100x faster.

  Returns:
    A tuple of tuples of (filename, line_number, last_instruction_offset).
  """
  cur_stack = inspect.currentframe()
  if cur_stack is None or cur_stack.f_back is None:
    return tuple()
  # We drop the first frame, which is this function itself.
  cur_stack = cur_stack.f_back
  call_stack = []
  while cur_stack.f_back:
    cur_stack = cur_stack.f_back
    call_stack.append(
        (cur_stack.f_code.co_filename, cur_stack.f_lineno, cur_stack.f_lasti)
    )
  return tuple(call_stack)


# Counter to keep track of number of log entries per token.
_log_counter_per_token = {}


def _get_next_log_count_per_token(token):
  """Wrapper for _log_counter_per_token. Thread-safe.

  Args:
    token: The token for which to look up the count.

  Returns:
    The number of times this function has been called with
    *token* as an argument (starting at 0).
  """
  # Can't use a defaultdict because defaultdict isn't atomic, whereas
  # setdefault is.
  return next(_log_counter_per_token.setdefault(token, itertools.count()))


def log_every_n(level, msg, n, *args, use_call_stack=False):
  """Logs ``msg % args`` at level 'level' once per 'n' times.

  Logs the 1st call, (N+1)st call, (2N+1)st call,  etc.
  Not threadsafe.

  Args:
    level: int, the absl logging level at which to log.
    msg: str, the message to be logged.
    n: int, the number of times this should be called before it is logged.
    *args: The args to be substituted into the msg.
    use_call_stack: bool, whether to include the call stack when counting the
      number of times the message is logged.
  """
  caller_info = get_absl_logger().findCaller()
  if use_call_stack:
    # To reduce storage costs, we hash the call stack.
    caller_info = (*caller_info[0:3], hash(_fast_stack_trace()))
  count = _get_next_log_count_per_token(caller_info)
  log_if(level, msg, not (count % n), *args)


# Keeps track of the last log time of the given token.
# Note: must be a dict since set/get is atomic in CPython.
# Note: entries are never released as their number is expected to be low.
_log_timer_per_token = {}


def _seconds_have_elapsed(token, num_seconds):
  """Tests if 'num_seconds' have passed since 'token' was requested.

  Not strictly thread-safe - may log with the wrong frequency if called
  concurrently from multiple threads. Accuracy depends on resolution of
  'timeit.default_timer()'.

  Always returns True on the first call for a given 'token'.

  Args:
    token: The token for which to look up the count.
    num_seconds: The number of seconds to test for.

  Returns:
    Whether it has been >= 'num_seconds' since 'token' was last requested.
  """
  now = timeit.default_timer()
  then = _log_timer_per_token.get(token, None)
  if then is None or (now - then) >= num_seconds:
    _log_timer_per_token[token] = now
    return True
  else:
    return False


def log_every_n_seconds(level, msg, n_seconds, *args, use_call_stack=False):
  """Logs ``msg % args`` at level ``level`` iff ``n_seconds`` elapsed since last call.

  Logs the first call, logs subsequent calls if 'n' seconds have elapsed since
  the last logging call from the same call site (file + line). Not thread-safe.

  Args:
    level: int, the absl logging level at which to log.
    msg: str, the message to be logged.
    n_seconds: float or int, seconds which should elapse before logging again.
    *args: The args to be substituted into the msg.
    use_call_stack: bool, whether to include the call stack when counting the
      number of times the message is logged.
  """
  caller_info = get_absl_logger().findCaller()
  if use_call_stack:
    # To reduce storage costs, we hash the call stack.
    caller_info = (*caller_info[0:3], hash(_fast_stack_trace()))
  should_log = _seconds_have_elapsed(caller_info, n_seconds)
  log_if(level, msg, should_log, *args)


def log_first_n(level, msg, n, *args, use_call_stack=False):
  """Logs ``msg % args`` at level ``level`` only first ``n`` times.

  Not threadsafe.

  Args:
    level: int, the absl logging level at which to log.
    msg: str, the message to be logged.
    n: int, the maximal number of times the message is logged.
    *args: The args to be substituted into the msg.
    use_call_stack: bool, whether to include the call stack when counting the
      number of times the message is logged.
  """
  caller_info = get_absl_logger().findCaller()
  if use_call_stack:
    # To reduce storage costs, we hash the call stack.
    caller_info = (*caller_info[0:3], hash(_fast_stack_trace()))
  count = _get_next_log_count_per_token(caller_info)
  log_if(level, msg, count < n, *args)


def log_if(level, msg, condition, *args):
  """Logs ``msg % args`` at level ``level`` only if condition is fulfilled."""
  if condition:
    log(level, msg, *args)


def log(level, msg, *args, **kwargs):
  """Logs ``msg % args`` at absl logging level ``level``.

  If no args are given just print msg, ignoring any interpolation specifiers.

  Args:
    level: int, the absl logging level at which to log the message
        (logging.DEBUG|INFO|WARNING|ERROR|FATAL). While some C++ verbose logging
        level constants are also supported, callers should prefer explicit
        logging.vlog() calls for such purpose.

    msg: str, the message to be logged.
    *args: The args to be substituted into the msg.
    **kwargs: May contain exc_info to add exception traceback to message.
  """
  if level > converter.ABSL_DEBUG:
    # Even though this function supports level that is greater than 1, users
    # should use logging.vlog instead for such cases.
    # Treat this as vlog, 1 is equivalent to DEBUG.
    standard_level = converter.STANDARD_DEBUG - (level - 1)
  else:
    if level < converter.ABSL_FATAL:
      level = converter.ABSL_FATAL
    standard_level = converter.absl_to_standard(level)

  # Match standard logging's behavior. Before use_absl_handler() and
  # logging is configured, there is no handler attached on _absl_logger nor
  # logging.root. So logs go no where.
  if not logging.root.handlers:
    logging.basicConfig()

  _absl_logger.log(standard_level, msg, *args, **kwargs)


def vlog(level, msg, *args, **kwargs):
  """Log ``msg % args`` at C++ vlog level ``level``.

  Args:
    level: int, the C++ verbose logging level at which to log the message,
        e.g. 1, 2, 3, 4... While absl level constants are also supported,
        callers should prefer logging.log|debug|info|... calls for such purpose.
    msg: str, the message to be logged.
    *args: The args to be substituted into the msg.
    **kwargs: May contain exc_info to add exception traceback to message.
  """
  log(level, msg, *args, **kwargs)


def vlog_is_on(level):
  """Checks if vlog is enabled for the given level in caller's source file.

  Args:
    level: int, the C++ verbose logging level at which to log the message,
        e.g. 1, 2, 3, 4... While absl level constants are also supported,
        callers should prefer level_debug|level_info|... calls for
        checking those.

  Returns:
    True if logging is turned on for that level.
  """

  if level > converter.ABSL_DEBUG:
    # Even though this function supports level that is greater than 1, users
    # should use logging.vlog instead for such cases.
    # Treat this as vlog, 1 is equivalent to DEBUG.
    standard_level = converter.STANDARD_DEBUG - (level - 1)
  else:
    if level < converter.ABSL_FATAL:
      level = converter.ABSL_FATAL
    standard_level = converter.absl_to_standard(level)
  return _absl_logger.isEnabledFor(standard_level)


def flush():
  """Flushes all log files."""
  get_absl_handler().flush()


def level_debug():
  """Returns True if debug logging is turned on."""
  return get_verbosity() >= DEBUG


def level_info():
  """Returns True if info logging is turned on."""
  return get_verbosity() >= INFO


def level_warning():
  """Returns True if warning logging is turned on."""
  return get_verbosity() >= WARNING


level_warn = level_warning  # Deprecated function.


def level_error():
  """Returns True if error logging is turned on."""
  return get_verbosity() >= ERROR


def get_log_file_name(level=INFO):
  """Returns the name of the log file.

  For Python logging, only one file is used and level is ignored. And it returns
  empty string if it logs to stderr/stdout or the log stream has no `name`
  attribute.

  Args:
    level: int, the absl.logging level.

  Raises:
    ValueError: Raised when `level` has an invalid value.
  """
  if level not in converter.ABSL_LEVELS:
    raise ValueError(f'Invalid absl.logging level {level}')
  stream = get_absl_handler().python_handler.stream
  if (stream == sys.stderr or stream == sys.stdout or
      not hasattr(stream, 'name')):
    return ''
  else:
    return stream.name


def find_log_dir_and_names(program_name=None, log_dir=None):
  """Computes the directory and filename prefix for log file.

  Args:
    program_name: str|None, the filename part of the path to the program that
        is running without its extension.  e.g: if your program is called
        ``usr/bin/foobar.py`` this method should probably be called with
        ``program_name='foobar`` However, this is just a convention, you can
        pass in any string you want, and it will be used as part of the
        log filename. If you don't pass in anything, the default behavior
        is as described in the example.  In python standard logging mode,
        the program_name will be prepended with ``py_`` if it is the
        ``program_name`` argument is omitted.
    log_dir: str|None, the desired log directory.

  Returns:
    (log_dir, file_prefix, symlink_prefix)

  Raises:
    FileNotFoundError: raised in Python 3 when it cannot find a log directory.
    OSError: raised in Python 2 when it cannot find a log directory.
  """
  if not program_name:
    # Strip the extension (foobar.par becomes foobar, and
    # fubar.py becomes fubar). We do this so that the log
    # file names are similar to C++ log file names.
    program_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

    # Prepend py_ to files so that python code gets a unique file, and
    # so that C++ libraries do not try to write to the same log files as us.
    program_name = 'py_%s' % program_name

  actual_log_dir = find_log_dir(log_dir=log_dir)

  try:
    username = getpass.getuser()
  except KeyError:
    # This can happen, e.g. when running under docker w/o passwd file.
    if hasattr(os, 'getuid'):
      # Windows doesn't have os.getuid
      username = str(os.getuid())
    else:
      username = 'unknown'
  hostname = socket.gethostname()
  file_prefix = '%s.%s.%s.log' % (program_name, hostname, username)

  return actual_log_dir, file_prefix, program_name


def find_log_dir(log_dir=None):
  """Returns the most suitable directory to put log files into.

  Args:
    log_dir: str|None, if specified, the logfile(s) will be created in that
        directory.  Otherwise if the --log_dir command-line flag is provided,
        the logfile will be created in that directory.  Otherwise the logfile
        will be created in a standard location.

  Raises:
    FileNotFoundError: raised in Python 3 when it cannot find a log directory.
    OSError: raised in Python 2 when it cannot find a log directory.
  """
  # Get a list of possible log dirs (will try to use them in order).
  # NOTE: Google's internal implementation has a special handling for Google
  # machines, which uses a list of directories. Hence the following uses `dirs`
  # instead of a single directory.
  if log_dir:
    # log_dir was explicitly specified as an arg, so use it and it alone.
    dirs = [log_dir]
  elif FLAGS['log_dir'].value:
    # log_dir flag was provided, so use it and it alone (this mimics the
    # behavior of the same flag in logging.cc).
    dirs = [FLAGS['log_dir'].value]
  else:
    dirs = [tempfile.gettempdir()]

  # Find the first usable log dir.
  for d in dirs:
    if os.path.isdir(d) and os.access(d, os.W_OK):
      return d
  raise FileNotFoundError(
      "Can't find a writable directory for logs, tried %s" % dirs)


def get_absl_log_prefix(record):
  """Returns the absl log prefix for the log record.

  Args:
    record: logging.LogRecord, the record to get prefix for.
  """
  created_tuple = time.localtime(record.created)
  created_microsecond = int(record.created % 1.0 * 1e6)

  critical_prefix = ''
  level = record.levelno
  if _is_non_absl_fatal_record(record):
    # When the level is FATAL, but not logged from absl, lower the level so
    # it's treated as ERROR.
    level = logging.ERROR
    critical_prefix = _CRITICAL_PREFIX
  severity = converter.get_initial_for_level(level)

  return '%c%02d%02d %02d:%02d:%02d.%06d %5d %s:%d] %s' % (
      severity,
      created_tuple.tm_mon,
      created_tuple.tm_mday,
      created_tuple.tm_hour,
      created_tuple.tm_min,
      created_tuple.tm_sec,
      created_microsecond,
      _get_thread_id(),
      record.filename,
      record.lineno,
      critical_prefix)


def skip_log_prefix(func):
  """Skips reporting the prefix of a given function or name by :class:`~absl.logging.ABSLLogger`.

  This is a convenience wrapper function / decorator for
  :meth:`~absl.logging.ABSLLogger.register_frame_to_skip`.

  If a callable function is provided, only that function will be skipped.
  If a function name is provided, all functions with the same name in the
  file that this is called in will be skipped.

  This can be used as a decorator of the intended function to be skipped.

  Args:
    func: Callable function or its name as a string.

  Returns:
    func (the input, unchanged).

  Raises:
    ValueError: The input is callable but does not have a function code object.
    TypeError: The input is neither callable nor a string.
  """
  if callable(func):
    func_code = getattr(func, '__code__', None)
    if func_code is None:
      raise ValueError('Input callable does not have a function code object.')
    file_name = func_code.co_filename
    func_name = func_code.co_name
    func_lineno = func_code.co_firstlineno
  elif isinstance(func, str):
    file_name = get_absl_logger().findCaller()[0]
    func_name = func
    func_lineno = None
  else:
    raise TypeError('Input is neither callable nor a string.')
  ABSLLogger.register_frame_to_skip(file_name, func_name, func_lineno)
  return func


def _is_non_absl_fatal_record(log_record):
  return (log_record.levelno >= logging.FATAL and
          not log_record.__dict__.get(_ABSL_LOG_FATAL, False))


def _is_absl_fatal_record(log_record):
  return (log_record.levelno >= logging.FATAL and
          log_record.__dict__.get(_ABSL_LOG_FATAL, False))


# Indicates if we still need to warn about pre-init logs going to stderr.
_warn_preinit_stderr = True


class PythonHandler(logging.StreamHandler):
  """The handler class used by Abseil Python logging implementation."""

  def __init__(self, stream=None, formatter=None):
    super().__init__(stream)
    self.setFormatter(formatter or PythonFormatter())

  def start_logging_to_file(self, program_name=None, log_dir=None):
    """Starts logging messages to files instead of standard error."""
    FLAGS.logtostderr = False

    actual_log_dir, file_prefix, symlink_prefix = find_log_dir_and_names(
        program_name=program_name, log_dir=log_dir)

    basename = '%s.INFO.%s.%d' % (
        file_prefix,
        time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time())),
        os.getpid())
    filename = os.path.join(actual_log_dir, basename)

    self.stream = open(filename, 'a', encoding='utf-8')

    # os.symlink is not available on Windows Python 2.
    if getattr(os, 'symlink', None):
      # Create a symlink to the log file with a canonical name.
      symlink = os.path.join(actual_log_dir, symlink_prefix + '.INFO')
      try:
        if os.path.islink(symlink):
          os.unlink(symlink)
        os.symlink(os.path.basename(filename), symlink)
      except OSError:
        # If it fails, we're sad but it's no error.  Commonly, this
        # fails because the symlink was created by another user and so
        # we can't modify it
        pass

  def use_absl_log_file(self, program_name=None, log_dir=None):
    """Conditionally logs to files, based on --logtostderr."""
    if FLAGS['logtostderr'].value:
      self.stream = sys.stderr
    else:
      self.start_logging_to_file(program_name=program_name, log_dir=log_dir)

  def flush(self):
    """Flushes all log files."""
    self.acquire()
    try:
      if self.stream and hasattr(self.stream, 'flush'):
        self.stream.flush()
    except (OSError, ValueError):
      # A ValueError is thrown if we try to flush a closed file.
      pass
    finally:
      self.release()

  def _log_to_stderr(self, record):
    """Emits the record to stderr.

    This temporarily sets the handler stream to stderr, calls
    StreamHandler.emit, then reverts the stream back.

    Args:
      record: logging.LogRecord, the record to log.
    """
    # emit() is protected by a lock in logging.Handler, so we don't need to
    # protect here again.
    old_stream = self.stream
    self.stream = sys.stderr
    try:
      super().emit(record)
    finally:
      self.stream = old_stream

  def emit(self, record):
    """Prints a record out to some streams.

    1. If ``FLAGS.logtostderr`` is set, it will print to ``sys.stderr`` ONLY.
    2. If ``FLAGS.alsologtostderr`` is set, it will print to ``sys.stderr``.
    3. If ``FLAGS.logtostderr`` is not set, it will log to the stream
        associated with the current thread.

    Args:
      record: :class:`logging.LogRecord`, the record to emit.
    """
    # People occasionally call logging functions at import time before
    # our flags may have even been defined yet, let alone even parsed, as we
    # rely on the C++ side to define some flags for us and app init to
    # deal with parsing.  Match the C++ library behavior of notify and emit
    # such messages to stderr.  It encourages people to clean-up and does
    # not hide the message.
    level = record.levelno
    if not FLAGS.is_parsed():  # Also implies "before flag has been defined".
      global _warn_preinit_stderr
      if _warn_preinit_stderr:
        sys.stderr.write(
            'WARNING: Logging before flag parsing goes to stderr.\n')
        _warn_preinit_stderr = False
      self._log_to_stderr(record)
    elif FLAGS['logtostderr'].value:
      self._log_to_stderr(record)
    else:
      super().emit(record)
      stderr_threshold = converter.string_to_standard(
          FLAGS['stderrthreshold'].value)
      if ((FLAGS['alsologtostderr'].value or level >= stderr_threshold) and
          self.stream != sys.stderr):
        self._log_to_stderr(record)
    # Die when the record is created from ABSLLogger and level is FATAL.
    if _is_absl_fatal_record(record):
      self.flush()  # Flush the log before dying.

      # In threaded python, sys.exit() from a non-main thread only
      # exits the thread in question.
      os.abort()

  def close(self):
    """Closes the stream to which we are writing."""
    self.acquire()
    try:
      self.flush()
      try:
        # Do not close the stream if it's sys.stderr|stdout. They may be
        # redirected or overridden to files, which should be managed by users
        # explicitly.
        user_managed = sys.stderr, sys.stdout, sys.__stderr__, sys.__stdout__
        if self.stream not in user_managed and (
            not hasattr(self.stream, 'isatty') or not self.stream.isatty()):
          self.stream.close()
      except ValueError:
        # A ValueError is thrown if we try to run isatty() on a closed file.
        pass
      super().close()
    finally:
      self.release()


class ABSLHandler(logging.Handler):
  """Abseil Python logging module's log handler."""

  def __init__(self, python_logging_formatter):
    super().__init__()

    self._python_handler = PythonHandler(formatter=python_logging_formatter)
    self.activate_python_handler()

  def format(self, record):
    return self._current_handler.format(record)

  def setFormatter(self, fmt):
    self._current_handler.setFormatter(fmt)

  def emit(self, record):
    self._current_handler.emit(record)

  def flush(self):
    self._current_handler.flush()

  def close(self):
    super().close()
    self._current_handler.close()

  def handle(self, record):
    rv = self.filter(record)
    if rv:
      return self._current_handler.handle(record)
    return rv

  @property
  def python_handler(self):
    return self._python_handler

  def activate_python_handler(self):
    """Uses the Python logging handler as the current logging handler."""
    self._current_handler = self._python_handler

  def use_absl_log_file(self, program_name=None, log_dir=None):
    self._current_handler.use_absl_log_file(program_name, log_dir)

  def start_logging_to_file(self, program_name=None, log_dir=None):
    self._current_handler.start_logging_to_file(program_name, log_dir)


class PythonFormatter(logging.Formatter):
  """Formatter class used by :class:`~absl.logging.PythonHandler`."""

  def format(self, record):
    """Appends the message from the record to the results of the prefix.

    Args:
      record: logging.LogRecord, the record to be formatted.

    Returns:
      The formatted string representing the record.
    """
    if (not FLAGS['showprefixforinfo'].value and
        FLAGS['verbosity'].value == converter.ABSL_INFO and
        record.levelno == logging.INFO and
        _absl_handler.python_handler.stream == sys.stderr):
      prefix = ''
    else:
      prefix = get_absl_log_prefix(record)
    return prefix + super().format(record)


class ABSLLogger(logging.getLoggerClass()):
  """A logger that will create LogRecords while skipping some stack frames.

  This class maintains an internal list of filenames and method names
  for use when determining who called the currently executing stack
  frame.  Any method names from specific source files are skipped when
  walking backwards through the stack.

  Client code should use the register_frame_to_skip method to let the
  ABSLLogger know which method from which file should be
  excluded from the walk backwards through the stack.
  """
  _frames_to_skip = set()

  def findCaller(self, stack_info=False, stacklevel=1):
    """Finds the frame of the calling method on the stack.

    This method skips any frames registered with the
    ABSLLogger and any methods from this file, and whatever
    method is currently being used to generate the prefix for the log
    line.  Then it returns the file name, line number, and method name
    of the calling method.  An optional fourth item may be returned,
    callers who only need things from the first three are advised to
    always slice or index the result rather than using direct unpacking
    assignment.

    Args:
      stack_info: bool, when True, include the stack trace as a fourth item
        returned.  On Python 3 there are always four items returned - the fourth
        will be None when this is False.  On Python 2 the stdlib base class API
        only returns three items.  We do the same when this new parameter is
        unspecified or False for compatibility.
      stacklevel: int, if greater than 1, that number of frames will be skipped.

    Returns:
      (filename, lineno, methodname[, sinfo]) of the calling method.
    """
    f_to_skip = ABSLLogger._frames_to_skip
    # Use sys._getframe(2) instead of logging.currentframe(), it's slightly
    # faster because there is one less frame to traverse.
    frame = sys._getframe(2)  # pylint: disable=protected-access
    frame_to_return = None

    while frame:
      code = frame.f_code
      if (_LOGGING_FILE_PREFIX not in code.co_filename and
          (code.co_filename, code.co_name,
           code.co_firstlineno) not in f_to_skip and
          (code.co_filename, code.co_name) not in f_to_skip):
        frame_to_return = frame
        stacklevel -= 1
        if stacklevel <= 0:
          break
      frame = frame.f_back

    if frame_to_return is not None:
      sinfo = None
      if stack_info:
        out = io.StringIO()
        out.write('Stack (most recent call last):\n')
        traceback.print_stack(frame, file=out)
        sinfo = out.getvalue().rstrip('\n')
      return (
          frame_to_return.f_code.co_filename,
          frame_to_return.f_lineno,
          frame_to_return.f_code.co_name,
          sinfo,
      )

    return None

  def critical(self, msg, *args, **kwargs):
    """Logs ``msg % args`` with severity ``CRITICAL``."""
    self.log(logging.CRITICAL, msg, *args, **kwargs)

  def fatal(self, msg, *args, **kwargs):
    """Logs ``msg % args`` with severity ``FATAL``."""
    self.log(logging.FATAL, msg, *args, **kwargs)

  def error(self, msg, *args, **kwargs):
    """Logs ``msg % args`` with severity ``ERROR``."""
    self.log(logging.ERROR, msg, *args, **kwargs)

  def warn(self, msg, *args, **kwargs):
    """Logs ``msg % args`` with severity ``WARN``."""
    warnings.warn("The 'warn' method is deprecated, use 'warning' instead",
                  DeprecationWarning, 2)
    self.log(logging.WARN, msg, *args, **kwargs)

  def warning(self, msg, *args, **kwargs):
    """Logs ``msg % args`` with severity ``WARNING``."""
    self.log(logging.WARNING, msg, *args, **kwargs)

  def info(self, msg, *args, **kwargs):
    """Logs ``msg % args`` with severity ``INFO``."""
    self.log(logging.INFO, msg, *args, **kwargs)

  def debug(self, msg, *args, **kwargs):
    """Logs ``msg % args`` with severity ``DEBUG``."""
    self.log(logging.DEBUG, msg, *args, **kwargs)

  def log(self, level, msg, *args, **kwargs):
    """Logs a message at a cetain level substituting in the supplied arguments.

    This method behaves differently in python and c++ modes.

    Args:
      level: int, the standard logging level at which to log the message.
      msg: str, the text of the message to log.
      *args: The arguments to substitute in the message.
      **kwargs: The keyword arguments to substitute in the message.
    """
    if level >= logging.FATAL:
      # Add property to the LogRecord created by this logger.
      # This will be used by the ABSLHandler to determine whether it should
      # treat CRITICAL/FATAL logs as really FATAL.
      extra = kwargs.setdefault('extra', {})
      extra[_ABSL_LOG_FATAL] = True
    super().log(level, msg, *args, **kwargs)

  def handle(self, record):
    """Calls handlers without checking ``Logger.disabled``.

    Non-root loggers are set to disabled after setup with :func:`logging.config`
    if it's not explicitly specified. Historically, absl logging will not be
    disabled by that. To maintaining this behavior, this function skips
    checking the ``Logger.disabled`` bit.

    This logger can still be disabled by adding a filter that filters out
    everything.

    Args:
      record: logging.LogRecord, the record to handle.
    """
    if self.filter(record):
      self.callHandlers(record)

  @classmethod
  def register_frame_to_skip(cls, file_name, function_name, line_number=None):
    """Registers a function name to skip when walking the stack.

    The :class:`~absl.logging.ABSLLogger` sometimes skips method calls on the
    stack to make the log messages meaningful in their appropriate context.
    This method registers a function from a particular file as one
    which should be skipped.

    Args:
      file_name: str, the name of the file that contains the function.
      function_name: str, the name of the function to skip.
      line_number: int, if provided, only the function with this starting line
          number will be skipped. Otherwise, all functions with the same name
          in the file will be skipped.
    """
    if line_number is not None:
      cls._frames_to_skip.add((file_name, function_name, line_number))
    else:
      cls._frames_to_skip.add((file_name, function_name))


def _get_thread_id():
  """Gets id of current thread, suitable for logging as an unsigned quantity.

  If pywrapbase is linked, returns GetTID() for the thread ID to be
  consistent with C++ logging.  Otherwise, returns the numeric thread id.
  The quantities are made unsigned by masking with 2*sys.maxint + 1.

  Returns:
    Thread ID unique to this process (unsigned)
  """
  thread_id = threading.get_ident()
  return thread_id & _THREAD_ID_MASK


def get_absl_logger():
  """Returns the absl logger instance."""
  assert _absl_logger is not None
  return _absl_logger


def get_absl_handler():
  """Returns the absl handler instance."""
  assert _absl_handler is not None
  return _absl_handler


def use_python_logging(quiet=False):
  """Uses the python implementation of the logging code.

  Args:
    quiet: No logging message about switching logging type.
  """
  get_absl_handler().activate_python_handler()
  if not quiet:
    info('Restoring pure python logging')


_attempted_to_remove_stderr_stream_handlers = False


def use_absl_handler():
  """Uses the ABSL logging handler for logging.

  This method is called in :func:`app.run()<absl.app.run>` so the absl handler
  is used in absl apps.
  """
  global _attempted_to_remove_stderr_stream_handlers
  if not _attempted_to_remove_stderr_stream_handlers:
    # The absl handler logs to stderr by default. To prevent double logging to
    # stderr, the following code tries its best to remove other handlers that
    # emit to stderr. Those handlers are most commonly added when
    # logging.info/debug is called before calling use_absl_handler().
    handlers = [
        h for h in logging.root.handlers
        if isinstance(h, logging.StreamHandler) and h.stream == sys.stderr]
    for h in handlers:
      logging.root.removeHandler(h)
    _attempted_to_remove_stderr_stream_handlers = True

  absl_handler = get_absl_handler()
  if absl_handler not in logging.root.handlers:
    logging.root.addHandler(absl_handler)
    FLAGS['verbosity']._update_logging_levels()  # pylint: disable=protected-access
    FLAGS['logger_levels']._update_logger_levels()  # pylint: disable=protected-access


def _initialize():
  """Initializes loggers and handlers."""
  global _absl_logger, _absl_handler

  if _absl_logger:
    return

  original_logger_class = logging.getLoggerClass()
  logging.setLoggerClass(ABSLLogger)
  _absl_logger = logging.getLogger('absl')
  logging.setLoggerClass(original_logger_class)

  python_logging_formatter = PythonFormatter()
  _absl_handler = ABSLHandler(python_logging_formatter)


_initialize()
