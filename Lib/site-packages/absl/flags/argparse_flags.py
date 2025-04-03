# Copyright 2018 The Abseil Authors.
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

"""This module provides argparse integration with absl.flags.

``argparse_flags.ArgumentParser`` is a drop-in replacement for
:class:`argparse.ArgumentParser`. It takes care of collecting and defining absl
flags in :mod:`argparse`.

Here is a simple example::

    # Assume the following absl.flags is defined in another module:
    #
    #     from absl import flags
    #     flags.DEFINE_string('echo', None, 'The echo message.')
    #
    parser = argparse_flags.ArgumentParser(
        description='A demo of absl.flags and argparse integration.')
    parser.add_argument('--header', help='Header message to print.')

    # The parser will also accept the absl flag `--echo`.
    # The `header` value is available as `args.header` just like a regular
    # argparse flag. The absl flag `--echo` continues to be available via
    # `absl.flags.FLAGS` if you want to access it.
    args = parser.parse_args()

    # Example usages:
    # ./program --echo='A message.' --header='A header'
    # ./program --header 'A header' --echo 'A message.'


Here is another example demonstrates subparsers::

    parser = argparse_flags.ArgumentParser(description='A subcommands demo.')
    parser.add_argument('--header', help='The header message to print.')

    subparsers = parser.add_subparsers(help='The command to execute.')

    roll_dice_parser = subparsers.add_parser(
        'roll_dice', help='Roll a dice.',
        # By default, absl flags can also be specified after the sub-command.
        # To only allow them before sub-command, pass
        # `inherited_absl_flags=None`.
        inherited_absl_flags=None)
    roll_dice_parser.add_argument('--num_faces', type=int, default=6)
    roll_dice_parser.set_defaults(command=roll_dice)

    shuffle_parser = subparsers.add_parser('shuffle', help='Shuffle inputs.')
    shuffle_parser.add_argument(
        'inputs', metavar='I', nargs='+', help='Inputs to shuffle.')
    shuffle_parser.set_defaults(command=shuffle)

    args = parser.parse_args(argv[1:])
    args.command(args)

    # Example usages:
    # ./program --echo='A message.' roll_dice --num_faces=6
    # ./program shuffle --echo='A message.' 1 2 3 4


There are several differences between :mod:`absl.flags` and
:mod:`~absl.flags.argparse_flags`:

1. Flags defined with absl.flags are parsed differently when using the
   argparse parser. Notably:

   1) absl.flags allows both single-dash and double-dash for any flag, and
      doesn't distinguish them; argparse_flags only allows double-dash for
      flag's regular name, and single-dash for flag's ``short_name``.
   2) Boolean flags in absl.flags can be specified with ``--bool``,
      ``--nobool``, as well as ``--bool=true/false`` (though not recommended);
      in argparse_flags, it only allows ``--bool``, ``--nobool``.

2. Help related flag differences:

   1) absl.flags does not define help flags, absl.app does that; argparse_flags
      defines help flags unless passed with ``add_help=False``.
   2) absl.app supports ``--helpxml``; argparse_flags does not.
   3) argparse_flags supports ``-h``; absl.app does not.
"""

import argparse
import sys

from absl import flags


_BUILT_IN_FLAGS = frozenset({
    'help',
    'helpshort',
    'helpfull',
    'helpxml',
    'flagfile',
    'undefok',
})


class ArgumentParser(argparse.ArgumentParser):
  """Custom ArgumentParser class to support special absl flags."""

  def __init__(self, **kwargs):
    """Initializes ArgumentParser.

    Args:
      **kwargs: same as argparse.ArgumentParser, except:
          1. It also accepts `inherited_absl_flags`: the absl flags to inherit.
             The default is the global absl.flags.FLAGS instance. Pass None to
             ignore absl flags.
          2. The `prefix_chars` argument must be the default value '-'.

    Raises:
      ValueError: Raised when prefix_chars is not '-'.
    """
    prefix_chars = kwargs.get('prefix_chars', '-')
    if prefix_chars != '-':
      raise ValueError(
          'argparse_flags.ArgumentParser only supports "-" as the prefix '
          'character, found "{}".'.format(prefix_chars))

    # Remove inherited_absl_flags before calling super.
    self._inherited_absl_flags = kwargs.pop('inherited_absl_flags', flags.FLAGS)
    # Now call super to initialize argparse.ArgumentParser before calling
    # add_argument in _define_absl_flags.
    super().__init__(**kwargs)

    if self.add_help:
      # -h and --help are defined in super.
      # Also add the --helpshort and --helpfull flags.
      self.add_argument(
          # Action 'help' defines a similar flag to -h/--help.
          '--helpshort', action='help',
          default=argparse.SUPPRESS, help=argparse.SUPPRESS)
      self.add_argument(
          '--helpfull', action=_HelpFullAction,
          default=argparse.SUPPRESS, help='show full help message and exit')

    if self._inherited_absl_flags is not None:
      self.add_argument(
          '--undefok', default=argparse.SUPPRESS, help=argparse.SUPPRESS)
      self._define_absl_flags(self._inherited_absl_flags)

  def parse_known_args(self, args=None, namespace=None):
    if args is None:
      args = sys.argv[1:]
    if self._inherited_absl_flags is not None:
      # Handle --flagfile.
      # Explicitly specify force_gnu=True, since argparse behaves like
      # gnu_getopt: flags can be specified after positional arguments.
      args = self._inherited_absl_flags.read_flags_from_files(
          args, force_gnu=True)

    undefok_missing = object()
    undefok = getattr(namespace, 'undefok', undefok_missing)

    namespace, args = super().parse_known_args(args, namespace)

    # For Python <= 2.7.8: https://bugs.python.org/issue9351, a bug where
    # sub-parsers don't preserve existing namespace attributes.
    # Restore the undefok attribute if a sub-parser dropped it.
    if undefok is not undefok_missing:
      namespace.undefok = undefok

    if self._inherited_absl_flags is not None:
      # Handle --undefok. At this point, `args` only contains unknown flags,
      # so it won't strip defined flags that are also specified with --undefok.
      # For Python <= 2.7.8: https://bugs.python.org/issue9351, a bug where
      # sub-parsers don't preserve existing namespace attributes. The undefok
      # attribute might not exist because a subparser dropped it.
      if hasattr(namespace, 'undefok'):
        args = _strip_undefok_args(namespace.undefok, args)
        # absl flags are not exposed in the Namespace object. See Namespace:
        # https://docs.python.org/3/library/argparse.html#argparse.Namespace.
        del namespace.undefok
      self._inherited_absl_flags.mark_as_parsed()
      try:
        self._inherited_absl_flags.validate_all_flags()
      except flags.IllegalFlagValueError as e:
        self.error(str(e))

    return namespace, args

  def _define_absl_flags(self, absl_flags):
    """Defines flags from absl_flags."""
    key_flags = set(absl_flags.get_key_flags_for_module(sys.argv[0]))
    for name in absl_flags:
      if name in _BUILT_IN_FLAGS:
        # Do not inherit built-in flags.
        continue
      flag_instance = absl_flags[name]
      # Each flags with short_name appears in FLAGS twice, so only define
      # when the dictionary key is equal to the regular name.
      if name == flag_instance.name:
        # Suppress the flag in the help short message if it's not a main
        # module's key flag.
        suppress = flag_instance not in key_flags
        self._define_absl_flag(flag_instance, suppress)

  def _define_absl_flag(self, flag_instance, suppress):
    """Defines a flag from the flag_instance."""
    flag_name = flag_instance.name
    short_name = flag_instance.short_name
    argument_names = ['--' + flag_name]
    if short_name:
      argument_names.insert(0, '-' + short_name)
    if suppress:
      helptext = argparse.SUPPRESS
    else:
      # argparse help string uses %-formatting. Escape the literal %'s.
      helptext = flag_instance.help.replace('%', '%%')
    if flag_instance.boolean:
      # Only add the `no` form to the long name.
      argument_names.append('--no' + flag_name)
      self.add_argument(
          *argument_names, action=_BooleanFlagAction, help=helptext,
          metavar=flag_instance.name.upper(),
          flag_instance=flag_instance)
    else:
      self.add_argument(
          *argument_names, action=_FlagAction, help=helptext,
          metavar=flag_instance.name.upper(),
          flag_instance=flag_instance)


class _FlagAction(argparse.Action):
  """Action class for Abseil non-boolean flags."""

  def __init__(
      self,
      option_strings,
      dest,
      help,  # pylint: disable=redefined-builtin
      metavar,
      flag_instance,
      default=argparse.SUPPRESS):
    """Initializes _FlagAction.

    Args:
      option_strings: See argparse.Action.
      dest: Ignored. The flag is always defined with dest=argparse.SUPPRESS.
      help: See argparse.Action.
      metavar: See argparse.Action.
      flag_instance: absl.flags.Flag, the absl flag instance.
      default: Ignored. The flag always uses dest=argparse.SUPPRESS so it
          doesn't affect the parsing result.
    """
    del dest
    self._flag_instance = flag_instance
    super().__init__(
        option_strings=option_strings,
        dest=argparse.SUPPRESS,
        help=help,
        metavar=metavar,
    )

  def __call__(self, parser, namespace, values, option_string=None):
    """See https://docs.python.org/3/library/argparse.html#action-classes."""
    self._flag_instance.parse(values)
    self._flag_instance.using_default_value = False


class _BooleanFlagAction(argparse.Action):
  """Action class for Abseil boolean flags."""

  def __init__(
      self,
      option_strings,
      dest,
      help,  # pylint: disable=redefined-builtin
      metavar,
      flag_instance,
      default=argparse.SUPPRESS):
    """Initializes _BooleanFlagAction.

    Args:
      option_strings: See argparse.Action.
      dest: Ignored. The flag is always defined with dest=argparse.SUPPRESS.
      help: See argparse.Action.
      metavar: See argparse.Action.
      flag_instance: absl.flags.Flag, the absl flag instance.
      default: Ignored. The flag always uses dest=argparse.SUPPRESS so it
          doesn't affect the parsing result.
    """
    del dest, default
    self._flag_instance = flag_instance
    flag_names = [self._flag_instance.name]
    if self._flag_instance.short_name:
      flag_names.append(self._flag_instance.short_name)
    self._flag_names = frozenset(flag_names)
    super().__init__(
        option_strings=option_strings,
        dest=argparse.SUPPRESS,
        nargs=0,  # Does not accept values, only `--bool` or `--nobool`.
        help=help,
        metavar=metavar,
    )

  def __call__(self, parser, namespace, values, option_string=None):
    """See https://docs.python.org/3/library/argparse.html#action-classes."""
    if not isinstance(values, list) or values:
      raise ValueError('values must be an empty list.')
    if option_string.startswith('--'):
      option = option_string[2:]
    else:
      option = option_string[1:]
    if option in self._flag_names:
      self._flag_instance.parse('true')
    else:
      if not option.startswith('no') or option[2:] not in self._flag_names:
        raise ValueError('invalid option_string: ' + option_string)
      self._flag_instance.parse('false')
    self._flag_instance.using_default_value = False


class _HelpFullAction(argparse.Action):
  """Action class for --helpfull flag."""

  def __init__(self, option_strings, dest, default, help):  # pylint: disable=redefined-builtin
    """Initializes _HelpFullAction.

    Args:
      option_strings: See argparse.Action.
      dest: Ignored. The flag is always defined with dest=argparse.SUPPRESS.
      default: Ignored.
      help: See argparse.Action.
    """
    del dest, default
    super().__init__(
        option_strings=option_strings,
        dest=argparse.SUPPRESS,
        default=argparse.SUPPRESS,
        nargs=0,
        help=help,
    )

  def __call__(self, parser, namespace, values, option_string=None):
    """See https://docs.python.org/3/library/argparse.html#action-classes."""
    # This only prints flags when help is not argparse.SUPPRESS.
    # It includes user defined argparse flags, as well as main module's
    # key absl flags. Other absl flags use argparse.SUPPRESS, so they aren't
    # printed here.
    parser.print_help()

    absl_flags = parser._inherited_absl_flags  # pylint: disable=protected-access
    if absl_flags is not None:
      modules = sorted(absl_flags.flags_by_module_dict())
      main_module = sys.argv[0]
      if main_module in modules:
        # The main module flags are already printed in parser.print_help().
        modules.remove(main_module)
      print(absl_flags._get_help_for_modules(  # pylint: disable=protected-access
          modules, prefix='', include_special_flags=True))
    parser.exit()


def _strip_undefok_args(undefok, args):
  """Returns a new list of args after removing flags in --undefok."""
  if undefok:
    undefok_names = {name.strip() for name in undefok.split(',')}
    undefok_names |= {'no' + name for name in undefok_names}
    # Remove undefok flags.
    args = [arg for arg in args if not _is_undefok(arg, undefok_names)]
  return args


def _is_undefok(arg, undefok_names):
  """Returns whether we can ignore arg based on a set of undefok flag names."""
  if not arg.startswith('-'):
    return False
  if arg.startswith('--'):
    arg_without_dash = arg[2:]
  else:
    arg_without_dash = arg[1:]
  if '=' in arg_without_dash:
    name, _ = arg_without_dash.split('=', 1)
  else:
    name = arg_without_dash
  if name in undefok_names:
    return True
  return False
