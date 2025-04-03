# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/nedbat/coveragepy/blob/master/NOTICE.txt

"""Determine facts about the environment."""

from __future__ import annotations

import os
import platform
import sys

from typing import Any, Final
from collections.abc import Iterable

# debug_info() at the bottom wants to show all the globals, but not imports.
# Grab the global names here to know which names to not show. Nothing defined
# above this line will be in the output.
_UNINTERESTING_GLOBALS = list(globals())
# These names also shouldn't be shown.
_UNINTERESTING_GLOBALS += ["PYBEHAVIOR", "debug_info"]

# Operating systems.
WINDOWS = sys.platform == "win32"
LINUX = sys.platform.startswith("linux")
MACOS = sys.platform == "darwin"

# Python implementations.
CPYTHON = (platform.python_implementation() == "CPython")
PYPY = (platform.python_implementation() == "PyPy")

# Python versions. We amend version_info with one more value, a zero if an
# official version, or 1 if built from source beyond an official version.
# Only use sys.version_info directly where tools like mypy need it to understand
# version-specfic code, otherwise use PYVERSION.
PYVERSION = sys.version_info + (int(platform.python_version()[-1] == "+"),)

if PYPY:
    # Minimum now is 7.3.16
    PYPYVERSION = sys.pypy_version_info         # type: ignore[attr-defined]
else:
    PYPYVERSION = (0,)

# Python behavior.
class PYBEHAVIOR:
    """Flags indicating this Python's behavior."""

    # Does Python conform to PEP626, Precise line numbers for debugging and other tools.
    # https://www.python.org/dev/peps/pep-0626
    pep626 = (PYVERSION > (3, 10, 0, "alpha", 4))

    # Is "if __debug__" optimized away?
    optimize_if_debug = not pep626

    # Is "if not __debug__" optimized away? The exact details have changed
    # across versions.
    optimize_if_not_debug = 1 if pep626 else 2

    # 3.7 changed how functions with only docstrings are numbered.
    docstring_only_function = (not PYPY) and (PYVERSION <= (3, 10))

    # Lines after break/continue/return/raise are no longer compiled into the
    # bytecode.  They used to be marked as missing, now they aren't executable.
    omit_after_jump = pep626 or PYPY

    # PyPy has always omitted statements after return.
    omit_after_return = omit_after_jump or PYPY

    # Optimize away unreachable try-else clauses.
    optimize_unreachable_try_else = pep626

    # Modules used to have firstlineno equal to the line number of the first
    # real line of code.  Now they always start at 1.
    module_firstline_1 = pep626

    # Are "if 0:" lines (and similar) kept in the compiled code?
    keep_constant_test = pep626

    # When leaving a with-block, do we visit the with-line again for the exit?
    # For example, wwith.py:
    #
    #    with open("/tmp/test", "w") as f1:
    #        a = 2
    #        with open("/tmp/test2", "w") as f3:
    #            print(4)
    #
    # % python3.9 -m trace -t wwith.py | grep wwith
    #  --- modulename: wwith, funcname: <module>
    # wwith.py(1): with open("/tmp/test", "w") as f1:
    # wwith.py(2):     a = 2
    # wwith.py(3):     with open("/tmp/test2", "w") as f3:
    # wwith.py(4):         print(4)
    #
    # % python3.10 -m trace -t wwith.py | grep wwith
    #  --- modulename: wwith, funcname: <module>
    # wwith.py(1): with open("/tmp/test", "w") as f1:
    # wwith.py(2):     a = 2
    # wwith.py(3):     with open("/tmp/test2", "w") as f3:
    # wwith.py(4):         print(4)
    # wwith.py(3):     with open("/tmp/test2", "w") as f3:
    # wwith.py(1): with open("/tmp/test", "w") as f1:
    #
    exit_through_with = (PYVERSION >= (3, 10, 0, "beta"))

    # When leaving a with-block, do we visit the with-line exactly,
    # or the context managers in inner-out order?
    #
    # mwith.py:
    #    with (
    #        open("/tmp/one", "w") as f2,
    #        open("/tmp/two", "w") as f3,
    #        open("/tmp/three", "w") as f4,
    #    ):
    #        print("hello 6")
    #
    # % python3.11 -m trace -t mwith.py | grep mwith
    #  --- modulename: mwith, funcname: <module>
    # mwith.py(2):     open("/tmp/one", "w") as f2,
    # mwith.py(1): with (
    # mwith.py(2):     open("/tmp/one", "w") as f2,
    # mwith.py(3):     open("/tmp/two", "w") as f3,
    # mwith.py(1): with (
    # mwith.py(3):     open("/tmp/two", "w") as f3,
    # mwith.py(4):     open("/tmp/three", "w") as f4,
    # mwith.py(1): with (
    # mwith.py(4):     open("/tmp/three", "w") as f4,
    # mwith.py(6):     print("hello 6")
    # mwith.py(1): with (
    #
    # % python3.12 -m trace -t mwith.py | grep mwith
    #  --- modulename: mwith, funcname: <module>
    # mwith.py(2):      open("/tmp/one", "w") as f2,
    # mwith.py(3):      open("/tmp/two", "w") as f3,
    # mwith.py(4):      open("/tmp/three", "w") as f4,
    # mwith.py(6):      print("hello 6")
    # mwith.py(4):      open("/tmp/three", "w") as f4,
    # mwith.py(3):      open("/tmp/two", "w") as f3,
    # mwith.py(2):      open("/tmp/one", "w") as f2,

    exit_with_through_ctxmgr = (PYVERSION >= (3, 12, 6))

    # Match-case construct.
    match_case = (PYVERSION >= (3, 10))

    # Some words are keywords in some places, identifiers in other places.
    soft_keywords = (PYVERSION >= (3, 10))

    # PEP669 Low Impact Monitoring: https://peps.python.org/pep-0669/
    pep669: Final[bool] = bool(getattr(sys, "monitoring", None))

    # Where does frame.f_lasti point when yielding from a generator?
    # It used to point at the YIELD, in 3.13 it points at the RESUME,
    # then it went back to the YIELD.
    # https://github.com/python/cpython/issues/113728
    lasti_is_yield = (PYVERSION[:2] != (3, 13))

    # PEP649 and PEP749: Deferred annotations
    deferred_annotations = (PYVERSION >= (3, 14))

    # Does sys.monitoring support BRANCH_RIGHT and BRANCH_LEFT?  The names
    # were added in early 3.14 alphas, but didn't work entirely correctly until
    # after 3.14.0a5.
    branch_right_left = (pep669 and (PYVERSION > (3, 14, 0, "alpha", 5, 0)))


# Coverage.py specifics, about testing scenarios. See tests/testenv.py also.

# Are we coverage-measuring ourselves?
METACOV = os.getenv("COVERAGE_COVERAGE") is not None

# Are we running our test suite?
# Even when running tests, you can use COVERAGE_TESTING=0 to disable the
# test-specific behavior like AST checking.
TESTING = os.getenv("COVERAGE_TESTING") == "True"


def debug_info() -> Iterable[tuple[str, Any]]:
    """Return a list of (name, value) pairs for printing debug information."""
    info = [
        (name, value) for name, value in globals().items()
        if not name.startswith("_") and name not in _UNINTERESTING_GLOBALS
    ]
    info += [
        (name, value) for name, value in PYBEHAVIOR.__dict__.items()
        if not name.startswith("_")
    ]
    return sorted(info)
