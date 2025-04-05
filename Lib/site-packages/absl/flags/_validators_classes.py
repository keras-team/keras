# Copyright 2021 The Abseil Authors.
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

"""Defines *private* classes used for flag validators.

Do NOT import this module. DO NOT use anything from this module. They are
private APIs.
"""

from absl.flags import _exceptions


class Validator:
  """Base class for flags validators.

  Users should NOT overload these classes, and use flags.Register...
  methods instead.
  """

  # Used to assign each validator an unique insertion_index
  validators_count = 0

  def __init__(self, checker, message):
    """Constructor to create all validators.

    Args:
      checker: function to verify the constraint.
          Input of this method varies, see SingleFlagValidator and
          multi_flags_validator for a detailed description.
      message: str, error message to be shown to the user.
    """
    self.checker = checker
    self.message = message
    Validator.validators_count += 1
    # Used to assert validators in the order they were registered.
    self.insertion_index = Validator.validators_count

  def verify(self, flag_values):
    """Verifies that constraint is satisfied.

    flags library calls this method to verify Validator's constraint.

    Args:
      flag_values: flags.FlagValues, the FlagValues instance to get flags from.
    Raises:
      Error: Raised if constraint is not satisfied.
    """
    param = self._get_input_to_checker_function(flag_values)
    if not self.checker(param):
      raise _exceptions.ValidationError(self.message)

  def get_flags_names(self):
    """Returns the names of the flags checked by this validator.

    Returns:
      [string], names of the flags.
    """
    raise NotImplementedError('This method should be overloaded')

  def print_flags_with_values(self, flag_values):
    raise NotImplementedError('This method should be overloaded')

  def _get_input_to_checker_function(self, flag_values):
    """Given flag values, returns the input to be given to checker.

    Args:
      flag_values: flags.FlagValues, containing all flags.
    Returns:
      The input to be given to checker. The return type depends on the specific
      validator.
    """
    raise NotImplementedError('This method should be overloaded')


class SingleFlagValidator(Validator):
  """Validator behind register_validator() method.

  Validates that a single flag passes its checker function. The checker function
  takes the flag value and returns True (if value looks fine) or, if flag value
  is not valid, either returns False or raises an Exception.
  """

  def __init__(self, flag_name, checker, message):
    """Constructor.

    Args:
      flag_name: string, name of the flag.
      checker: function to verify the validator.
          input  - value of the corresponding flag (string, boolean, etc).
          output - bool, True if validator constraint is satisfied.
              If constraint is not satisfied, it should either return False or
              raise flags.ValidationError(desired_error_message).
      message: str, error message to be shown to the user if validator's
          condition is not satisfied.
    """
    super().__init__(checker, message)
    self.flag_name = flag_name

  def get_flags_names(self):
    return [self.flag_name]

  def print_flags_with_values(self, flag_values):
    return 'flag --%s=%s' % (self.flag_name, flag_values[self.flag_name].value)

  def _get_input_to_checker_function(self, flag_values):
    """Given flag values, returns the input to be given to checker.

    Args:
      flag_values: flags.FlagValues, the FlagValues instance to get flags from.
    Returns:
      object, the input to be given to checker.
    """
    return flag_values[self.flag_name].value


class MultiFlagsValidator(Validator):
  """Validator behind register_multi_flags_validator method.

  Validates that flag values pass their common checker function. The checker
  function takes flag values and returns True (if values look fine) or,
  if values are not valid, either returns False or raises an Exception.
  """

  def __init__(self, flag_names, checker, message):
    """Constructor.

    Args:
      flag_names: [str], containing names of the flags used by checker.
      checker: function to verify the validator.
          input  - dict, with keys() being flag_names, and value for each
              key being the value of the corresponding flag (string, boolean,
              etc).
          output - bool, True if validator constraint is satisfied.
              If constraint is not satisfied, it should either return False or
              raise flags.ValidationError(desired_error_message).
      message: str, error message to be shown to the user if validator's
          condition is not satisfied
    """
    super().__init__(checker, message)
    self.flag_names = flag_names

  def _get_input_to_checker_function(self, flag_values):
    """Given flag values, returns the input to be given to checker.

    Args:
      flag_values: flags.FlagValues, the FlagValues instance to get flags from.
    Returns:
      dict, with keys() being self.flag_names, and value for each key
      being the value of the corresponding flag (string, boolean, etc).
    """
    return {key: flag_values[key].value for key in self.flag_names}

  def print_flags_with_values(self, flag_values):
    prefix = 'flags '
    flags_with_values = []
    for key in self.flag_names:
      flags_with_values.append('%s=%s' % (key, flag_values[key].value))
    return prefix + ', '.join(flags_with_values)

  def get_flags_names(self):
    return self.flag_names
