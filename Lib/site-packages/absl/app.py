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

"""Generic entry point for Abseil Python applications.

To use this module, define a ``main`` function with a single ``argv`` argument
and call ``app.run(main)``. For example::

    def main(argv):
      if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    if __name__ == '__main__':
      app.run(main)
"""

import collections
import errno
import os
import pdb
import sys
import textwrap
import traceback

from absl import command_name
from absl import flags
from absl import logging

try:
  import faulthandler
except ImportError:
  faulthandler = None

FLAGS = flags.FLAGS

flags.DEFINE_boolean('run_with_pdb', False, 'Set to true for PDB debug mode')
flags.DEFINE_boolean('pdb_post_mortem', False,
                     'Set to true to handle uncaught exceptions with PDB '
                     'post mortem.')
flags.DEFINE_alias('pdb', 'pdb_post_mortem')
flags.DEFINE_boolean('run_with_profiling', False,
                     'Set to true for profiling the script. '
                     'Execution will be slower, and the output format might '
                     'change over time.')
flags.DEFINE_string('profile_file', None,
                    'Dump profile information to a file (for python -m '
                    'pstats). Implies --run_with_profiling.')
flags.DEFINE_boolean('use_cprofile_for_profiling', True,
                     'Use cProfile instead of the profile module for '
                     'profiling. This has no effect unless '
                     '--run_with_profiling is set.')
flags.DEFINE_boolean('only_check_args', False,
                     'Set to true to validate args and exit.',
                     allow_hide_cpp=True)


# If main() exits via an abnormal exception, call into these
# handlers before exiting.
EXCEPTION_HANDLERS = []


class Error(Exception):
  pass


class UsageError(Error):
  """Exception raised when the arguments supplied by the user are invalid.

  Raise this when the arguments supplied are invalid from the point of
  view of the application. For example when two mutually exclusive
  flags have been supplied or when there are not enough non-flag
  arguments. It is distinct from flags.Error which covers the lower
  level of parsing and validating individual flags.
  """

  def __init__(self, message, exitcode=1):
    super().__init__(message)
    self.exitcode = exitcode


class HelpFlag(flags.BooleanFlag):
  """Special boolean flag that displays usage and raises SystemExit."""
  NAME = 'help'
  SHORT_NAME = '?'

  def __init__(self):
    super().__init__(
        self.NAME,
        False,
        'show this help',
        short_name=self.SHORT_NAME,
        allow_hide_cpp=True,
    )

  def parse(self, arg):
    if self._parse(arg):
      usage(shorthelp=True, writeto_stdout=True)
      # Advertise --helpfull on stdout, since usage() was on stdout.
      print()
      print('Try --helpfull to get a list of all flags.')
      sys.exit(1)


class HelpshortFlag(HelpFlag):
  """--helpshort is an alias for --help."""
  NAME = 'helpshort'
  SHORT_NAME = None


class HelpfullFlag(flags.BooleanFlag):
  """Display help for flags in the main module and all dependent modules."""

  def __init__(self):
    super().__init__('helpfull', False, 'show full help', allow_hide_cpp=True)

  def parse(self, arg):
    if self._parse(arg):
      usage(writeto_stdout=True)
      sys.exit(1)


class HelpXMLFlag(flags.BooleanFlag):
  """Similar to HelpfullFlag, but generates output in XML format."""

  def __init__(self):
    super().__init__(
        'helpxml',
        False,
        'like --helpfull, but generates XML output',
        allow_hide_cpp=True,
    )

  def parse(self, arg):
    if self._parse(arg):
      flags.FLAGS.write_help_in_xml_format(sys.stdout)
      sys.exit(1)


def parse_flags_with_usage(args):
  """Tries to parse the flags, print usage, and exit if unparsable.

  Args:
    args: [str], a non-empty list of the command line arguments including
        program name.

  Returns:
    [str], a non-empty list of remaining command line arguments after parsing
    flags, including program name.
  """
  try:
    return FLAGS(args)
  except flags.Error as error:
    message = str(error)
    if '\n' in message:
      final_message = 'FATAL Flags parsing error:\n%s\n' % textwrap.indent(
          message, '  ')
    else:
      final_message = 'FATAL Flags parsing error: %s\n' % message
    sys.stderr.write(final_message)
    sys.stderr.write('Pass --helpshort or --helpfull to see help on flags.\n')
    sys.exit(1)


_define_help_flags_called = False


def define_help_flags():
  """Registers help flags. Idempotent."""
  # Use a global to ensure idempotence.
  global _define_help_flags_called

  if not _define_help_flags_called:
    flags.DEFINE_flag(HelpFlag())
    flags.DEFINE_flag(HelpshortFlag())  # alias for --help
    flags.DEFINE_flag(HelpfullFlag())
    flags.DEFINE_flag(HelpXMLFlag())
    _define_help_flags_called = True


def _register_and_parse_flags_with_usage(
    argv=None,
    flags_parser=parse_flags_with_usage,
):
  """Registers help flags, parses arguments and shows usage if appropriate.

  This also calls sys.exit(0) if flag --only_check_args is True.

  Args:
    argv: [str], a non-empty list of the command line arguments including
      program name, sys.argv is used if None.
    flags_parser: Callable[[List[str]], Any], the function used to parse flags.
      The return value of this function is passed to `main` untouched. It must
      guarantee FLAGS is parsed after this function is called.

  Returns:
    The return value of `flags_parser`. When using the default `flags_parser`,
    it returns the following:
    [str], a non-empty list of remaining command line arguments after parsing
    flags, including program name.

  Raises:
    Error: Raised when flags_parser is called, but FLAGS is not parsed.
    SystemError: Raised when it's called more than once.
  """
  # fmt: on
  if _register_and_parse_flags_with_usage.done:
    raise SystemError('Flag registration can be done only once.')

  define_help_flags()

  original_argv = sys.argv if argv is None else argv
  args_to_main = flags_parser(original_argv)
  if not FLAGS.is_parsed():
    raise Error('FLAGS must be parsed after flags_parser is called.')

  # Exit when told so.
  if FLAGS.only_check_args:
    sys.exit(0)
  # Immediately after flags are parsed, bump verbosity to INFO if the flag has
  # not been set.
  if FLAGS['verbosity'].using_default_value:
    FLAGS.verbosity = 0
  _register_and_parse_flags_with_usage.done = True

  return args_to_main

_register_and_parse_flags_with_usage.done = False


def _run_main(main, argv):
  """Calls main, optionally with pdb or profiler."""
  if FLAGS.run_with_pdb:
    sys.exit(pdb.runcall(main, argv))
  elif FLAGS.run_with_profiling or FLAGS.profile_file:
    # Avoid import overhead since most apps (including performance-sensitive
    # ones) won't be run with profiling.
    # pylint: disable=g-import-not-at-top
    import atexit
    if FLAGS.use_cprofile_for_profiling:
      import cProfile as profile
    else:
      import profile
    profiler = profile.Profile()
    if FLAGS.profile_file:
      atexit.register(profiler.dump_stats, FLAGS.profile_file)
    else:
      atexit.register(profiler.print_stats)
    sys.exit(profiler.runcall(main, argv))
  else:
    sys.exit(main(argv))


def _call_exception_handlers(exception):
  """Calls any installed exception handlers."""
  for handler in EXCEPTION_HANDLERS:
    try:
      if handler.wants(exception):
        handler.handle(exception)
    except:  # pylint: disable=bare-except
      try:
        # We don't want to stop for exceptions in the exception handlers but
        # we shouldn't hide them either.
        logging.error(traceback.format_exc())
      except:  # pylint: disable=bare-except
        # In case even the logging statement fails, ignore.
        pass


def run(
    main,
    argv=None,
    flags_parser=parse_flags_with_usage,
):
  """Begins executing the program.

  Args:
    main: The main function to execute. It takes an single argument "argv",
        which is a list of command line arguments with parsed flags removed.
        The return value is passed to `sys.exit`, and so for example
        a return value of 0 or None results in a successful termination, whereas
        a return value of 1 results in abnormal termination.
        For more details, see https://docs.python.org/3/library/sys#sys.exit
    argv: A non-empty list of the command line arguments including program name,
        sys.argv is used if None.
    flags_parser: Callable[[List[str]], Any], the function used to parse flags.
        The return value of this function is passed to `main` untouched.
        It must guarantee FLAGS is parsed after this function is called.
        Should be passed as a keyword-only arg which will become mandatory in a
        future release.
  - Parses command line flags with the flag module.
  - If there are any errors, prints usage().
  - Calls main() with the remaining arguments.
  - If main() raises a UsageError, prints usage and the error message.
  """
  # fmt: on
  try:
    args = _run_init(
        sys.argv if argv is None else argv,
        flags_parser,
    )
    while _init_callbacks:
      callback = _init_callbacks.popleft()
      callback()
    try:
      _run_main(main, args)
    except UsageError as error:
      usage(shorthelp=True, detailed_error=error, exitcode=error.exitcode)
    except:
      exc = sys.exc_info()[1]
      # Don't try to post-mortem debug successful SystemExits, since those
      # mean there wasn't actually an error. In particular, the test framework
      # raises SystemExit(False) even if all tests passed.
      if isinstance(exc, SystemExit) and not exc.code:
        raise

      # Check the tty so that we don't hang waiting for input in an
      # non-interactive scenario.
      if FLAGS.pdb_post_mortem and sys.stdout.isatty():
        traceback.print_exc()
        print()
        print(' *** Entering post-mortem debugging ***')
        print()
        pdb.post_mortem()
      raise
  except Exception as e:
    _call_exception_handlers(e)
    raise

# Callbacks which have been deferred until after _run_init has been called.
_init_callbacks = collections.deque()


def call_after_init(callback):
  """Calls the given callback only once ABSL has finished initialization.

  If ABSL has already finished initialization when ``call_after_init`` is
  called then the callback is executed immediately, otherwise `callback` is
  stored to be executed after ``app.run`` has finished initializing (aka. just
  before the main function is called).

  If called after ``app.run``, this is equivalent to calling ``callback()`` in
  the caller thread. If called before ``app.run``, callbacks are run
  sequentially (in an undefined order) in the same thread as ``app.run``.

  Args:
    callback: a callable to be called once ABSL has finished initialization.
      This may be immediate if initialization has already finished. It
      takes no arguments and returns nothing.
  """
  if _run_init.done:
    callback()
  else:
    _init_callbacks.append(callback)


def _run_init(
    argv,
    flags_parser,
):
  """Does one-time initialization and re-parses flags on rerun."""
  if _run_init.done:
    return flags_parser(argv)
  command_name.make_process_name_useful()
  # Set up absl logging handler.
  logging.use_absl_handler()
  args = _register_and_parse_flags_with_usage(
      argv=argv,
      flags_parser=flags_parser,
  )
  if faulthandler:
    try:
      faulthandler.enable()
    except Exception:  # pylint: disable=broad-except
      # Some tests verify stderr output very closely, so don't print anything.
      # Disabled faulthandler is a low-impact error.
      pass
  _run_init.done = True
  return args


_run_init.done = False


def usage(shorthelp=False, writeto_stdout=False, detailed_error=None,
          exitcode=None):
  """Writes __main__'s docstring to stderr with some help text.

  Args:
    shorthelp: bool, if True, prints only flags from the main module,
        rather than all flags.
    writeto_stdout: bool, if True, writes help message to stdout,
        rather than to stderr.
    detailed_error: str, additional detail about why usage info was presented.
    exitcode: optional integer, if set, exits with this status code after
        writing help.
  """
  if writeto_stdout:
    stdfile = sys.stdout
  else:
    stdfile = sys.stderr

  doc = sys.modules['__main__'].__doc__
  if not doc:
    doc = '\nUSAGE: %s [flags]\n' % sys.argv[0]
    doc = flags.text_wrap(doc, indent='       ', firstline_indent='')
  else:
    # Replace all '%s' with sys.argv[0], and all '%%' with '%'.
    num_specifiers = doc.count('%') - 2 * doc.count('%%')
    try:
      doc %= (sys.argv[0],) * num_specifiers
    except (OverflowError, TypeError, ValueError):
      # Just display the docstring as-is.
      pass
  if shorthelp:
    flag_str = FLAGS.main_module_help()
  else:
    flag_str = FLAGS.get_help()
  try:
    stdfile.write(doc)
    if flag_str:
      stdfile.write('\nflags:\n')
      stdfile.write(flag_str)
    stdfile.write('\n')
    if detailed_error is not None:
      stdfile.write('\n%s\n' % detailed_error)
  except OSError as e:
    # We avoid printing a huge backtrace if we get EPIPE, because
    # "foo.par --help | less" is a frequent use case.
    if e.errno != errno.EPIPE:
      raise
  if exitcode is not None:
    sys.exit(exitcode)


class ExceptionHandler:
  """Base exception handler from which other may inherit."""

  def wants(self, exc):
    """Returns whether this handler wants to handle the exception or not.

    This base class returns True for all exceptions by default. Override in
    subclass if it wants to be more selective.

    Args:
      exc: Exception, the current exception.
    """
    del exc  # Unused.
    return True

  def handle(self, exc):
    """Do something with the current exception.

    Args:
      exc: Exception, the current exception

    This method must be overridden.
    """
    raise NotImplementedError()


def install_exception_handler(handler):
  """Installs an exception handler.

  Args:
    handler: ExceptionHandler, the exception handler to install.

  Raises:
    TypeError: Raised when the handler was not of the correct type.

  All installed exception handlers will be called if main() exits via
  an abnormal exception, i.e. not one of SystemExit, KeyboardInterrupt,
  FlagsError or UsageError.
  """
  if not isinstance(handler, ExceptionHandler):
    raise TypeError('handler of type %s does not inherit from ExceptionHandler'
                    % type(handler))
  EXCEPTION_HANDLERS.append(handler)
