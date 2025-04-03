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

"""Exception classes in ABSL flags library.

Do NOT import this module directly. Import the flags package and use the
aliases defined at the package level instead.
"""

import sys

from absl.flags import _helpers


_helpers.disclaim_module_ids.add(id(sys.modules[__name__]))


class Error(Exception):
  """The base class for all flags errors."""


class CantOpenFlagFileError(Error):
  """Raised when flagfile fails to open.

  E.g. the file doesn't exist, or has wrong permissions.
  """


class DuplicateFlagError(Error):
  """Raised if there is a flag naming conflict."""

  @classmethod
  def from_flag(cls, flagname, flag_values, other_flag_values=None):
    """Creates a DuplicateFlagError by providing flag name and values.

    Args:
      flagname: str, the name of the flag being redefined.
      flag_values: :class:`FlagValues`, the FlagValues instance containing the
        first definition of flagname.
      other_flag_values: :class:`FlagValues`, if it is not None, it should be
        the FlagValues object where the second definition of flagname occurs.
        If it is None, we assume that we're being called when attempting to
        create the flag a second time, and we use the module calling this one
        as the source of the second definition.

    Returns:
      An instance of DuplicateFlagError.
    """
    first_module = flag_values.find_module_defining_flag(
        flagname, default='<unknown>')
    if other_flag_values is None:
      second_module = _helpers.get_calling_module()
    else:
      second_module = other_flag_values.find_module_defining_flag(
          flagname, default='<unknown>')
    flag_summary = flag_values[flagname].help
    msg = ("The flag '%s' is defined twice. First from %s, Second from %s.  "
           "Description from first occurrence: %s") % (
               flagname, first_module, second_module, flag_summary)
    return cls(msg)


class IllegalFlagValueError(Error):
  """Raised when the flag command line argument is illegal."""


class UnrecognizedFlagError(Error):
  """Raised when a flag is unrecognized.

  Attributes:
    flagname: str, the name of the unrecognized flag.
    flagvalue: The value of the flag, empty if the flag is not defined.
  """

  def __init__(self, flagname, flagvalue='', suggestions=None):
    self.flagname = flagname
    self.flagvalue = flagvalue
    if suggestions:
      # Space before the question mark is intentional to not include it in the
      # selection when copy-pasting the suggestion from (some) terminals.
      tip = '. Did you mean: %s ?' % ', '.join(suggestions)
    else:
      tip = ''
    super().__init__("Unknown command line flag '%s'%s" % (flagname, tip))


class UnparsedFlagAccessError(Error):
  """Raised when accessing the flag value from unparsed :class:`FlagValues`."""


class ValidationError(Error):
  """Raised when flag validator constraint is not satisfied."""


class FlagNameConflictsWithMethodError(Error):
  """Raised when a flag name conflicts with :class:`FlagValues` methods."""
