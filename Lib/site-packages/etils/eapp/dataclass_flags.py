# Copyright 2024 The etils Authors.
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

"""Wrapper around `simple_parsing` for absl compatibility."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional, TypeVar

import __main__
from etils.epy import _internal

with _internal.check_missing_deps():
  # pylint: disable=g-import-not-at-top
  from absl import flags
  import simple_parsing
  # pylint: enable=g-import-not-at-top

_DataclassT = TypeVar('_DataclassT')

FLAGS = flags.FLAGS


def make_flags_parser(
    cls: _DataclassT,
    *,
    prog: Optional[str] = None,
    description: Optional[str] = None,
    **extra_kwargs,
) -> Callable[[list[str]], _DataclassT]:
  """Dataclass flag parser for absl.

  Allow to define CLI flags through dataclasses.

  Usage:

  ```python
  @dataclasses.dataclass
  class Args:
    user: str
    verbose: bool = False


  def main(args: Args):
    if args.verbose:
      print(args.user)


  if __name__ == '__main__':
    app.run(main, flags_parser=eapp.make_flags_parser(Args))
  ```

  Allow to call your program with `my_program --user=$USER --verbose`

  This is a wrapper around `simple_parsing`
  (https://github.com/lebrice/SimpleParsing). See documentation for details.

  Args:
    cls: Dataclass containing the arguments to be parsed
    prog: Program name. Forwarded to `argparse.ArgumentParser`
    description: Description (auto-extracted from the `__main__` docstring)
      Forwarded to `argparse.ArgumentParser`
    **extra_kwargs: Extra arguments to be forwarded to `argparse.ArgumentParser`

  Returns:
    flags_parser function, for `app.run(main, flags_parser=...)`.
  """

  if not description and __main__.__doc__:
    description = __main__.__doc__.split('\n', 1)[0]

  def _flag_parser(argv: list[str]) -> _DataclassT:
    parser = simple_parsing.ArgumentParser(
        prog=prog,
        description=description,
        **extra_kwargs,
    )
    parser.add_arguments(cls, dest='args')

    namespace, remaining_argv = parser.parse_known_args(argv[1:])

    # Parse absl.flags
    # For consistency with argparse, we could catch
    # `flags.IllegalFlagValueError` and exit through sys.exit(),
    # like absl.flags.argparse_flags
    FLAGS([''] + remaining_argv)

    # Forward the parsed args to `main`
    return namespace.args

  return _flag_parser
