# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/nedbat/coveragepy/blob/master/NOTICE.txt

"""Monkey-patching to add multiprocessing support for coverage.py"""

from __future__ import annotations

import multiprocessing
import multiprocessing.process
import os
import os.path
import sys
import traceback

from typing import Any

from coverage.debug import DebugControl

# An attribute that will be set on the module to indicate that it has been
# monkey-patched.
PATCHED_MARKER = "_coverage$patched"


OriginalProcess = multiprocessing.process.BaseProcess
original_bootstrap = OriginalProcess._bootstrap     # type: ignore[attr-defined]

class ProcessWithCoverage(OriginalProcess):         # pylint: disable=abstract-method
    """A replacement for multiprocess.Process that starts coverage."""

    def _bootstrap(self, *args, **kwargs):          # type: ignore[no-untyped-def]
        """Wrapper around _bootstrap to start coverage."""
        debug: DebugControl | None = None
        try:
            from coverage import Coverage       # avoid circular import
            cov = Coverage(data_suffix=True, auto_data=True)
            cov._warn_preimported_source = False
            cov.start()
            _debug = cov._debug
            assert _debug is not None
            if _debug.should("multiproc"):
                debug = _debug
            if debug:
                debug.write("Calling multiprocessing bootstrap")
        except Exception:
            print("Exception during multiprocessing bootstrap init:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            raise
        try:
            return original_bootstrap(self, *args, **kwargs)
        finally:
            if debug:
                debug.write("Finished multiprocessing bootstrap")
            try:
                cov.stop()
                cov.save()
            except Exception as exc:
                if debug:
                    debug.write("Exception during multiprocessing bootstrap cleanup", exc=exc)
                raise
            if debug:
                debug.write("Saved multiprocessing data")

class Stowaway:
    """An object to pickle, so when it is unpickled, it can apply the monkey-patch."""
    def __init__(self, rcfile: str) -> None:
        self.rcfile = rcfile

    def __getstate__(self) -> dict[str, str]:
        return {"rcfile": self.rcfile}

    def __setstate__(self, state: dict[str, str]) -> None:
        patch_multiprocessing(state["rcfile"])


def patch_multiprocessing(rcfile: str) -> None:
    """Monkey-patch the multiprocessing module.

    This enables coverage measurement of processes started by multiprocessing.
    This involves aggressive monkey-patching.

    `rcfile` is the path to the rcfile being used.

    """

    if hasattr(multiprocessing, PATCHED_MARKER):
        return

    OriginalProcess._bootstrap = ProcessWithCoverage._bootstrap     # type: ignore[attr-defined]

    # Set the value in ProcessWithCoverage that will be pickled into the child
    # process.
    os.environ["COVERAGE_RCFILE"] = os.path.abspath(rcfile)

    # When spawning processes rather than forking them, we have no state in the
    # new process.  We sneak in there with a Stowaway: we stuff one of our own
    # objects into the data that gets pickled and sent to the subprocess. When
    # the Stowaway is unpickled, its __setstate__ method is called, which
    # re-applies the monkey-patch.
    # Windows only spawns, so this is needed to keep Windows working.
    try:
        from multiprocessing import spawn
        original_get_preparation_data = spawn.get_preparation_data
    except (ImportError, AttributeError):
        pass
    else:
        def get_preparation_data_with_stowaway(name: str) -> dict[str, Any]:
            """Get the original preparation data, and also insert our stowaway."""
            d = original_get_preparation_data(name)
            d["stowaway"] = Stowaway(rcfile)
            return d

        spawn.get_preparation_data = get_preparation_data_with_stowaway

    setattr(multiprocessing, PATCHED_MARKER, True)
