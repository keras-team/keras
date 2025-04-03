# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/nedbat/coveragepy/blob/master/NOTICE.txt

"""Management of core choices."""

from __future__ import annotations

import os
import sys
from typing import Any

from coverage import env
from coverage.config import CoverageConfig
from coverage.disposition import FileDisposition
from coverage.exceptions import ConfigError
from coverage.misc import isolate_module
from coverage.pytracer import PyTracer
from coverage.sysmon import SysMonitor
from coverage.types import (
    TFileDisposition,
    TWarnFn,
    Tracer,
)


os = isolate_module(os)

try:
    # Use the C extension code when we can, for speed.
    from coverage.tracer import CTracer, CFileDisposition
    HAS_CTRACER = True
except ImportError:
    # Couldn't import the C extension, maybe it isn't built.
    if os.getenv("COVERAGE_CORE") == "ctrace":      # pragma: part covered
        # During testing, we use the COVERAGE_CORE environment variable
        # to indicate that we've fiddled with the environment to test this
        # fallback code.  If we thought we had a C tracer, but couldn't import
        # it, then exit quickly and clearly instead of dribbling confusing
        # errors. I'm using sys.exit here instead of an exception because an
        # exception here causes all sorts of other noise in unittest.
        sys.stderr.write("*** COVERAGE_CORE is 'ctrace' but can't import CTracer!\n")
        sys.exit(1)
    HAS_CTRACER = False


class Core:
    """Information about the central technology enabling execution measurement."""

    tracer_class: type[Tracer]
    tracer_kwargs: dict[str, Any]
    file_disposition_class: type[TFileDisposition]
    supports_plugins: bool
    packed_arcs: bool
    systrace: bool

    def __init__(
        self,
        warn: TWarnFn,
        config: CoverageConfig,
        dynamic_contexts: bool,
        metacov: bool,
    ) -> None:
        # Check the conditions that preclude us from using sys.monitoring.
        reason_no_sysmon = ""
        if not env.PYBEHAVIOR.pep669:
            reason_no_sysmon = "isn't available in this version"
        elif config.branch and not env.PYBEHAVIOR.branch_right_left:
            reason_no_sysmon = "can't measure branches in this version"
        elif dynamic_contexts:
            reason_no_sysmon = "doesn't yet support dynamic contexts"

        core_name: str | None = None
        if config.timid:
            core_name = "pytrace"

        if core_name is None:
            core_name = os.getenv("COVERAGE_CORE")

        if core_name == "sysmon" and reason_no_sysmon:
            warn(f"sys.monitoring {reason_no_sysmon}, using default core", slug="no-sysmon")
            core_name = None

        if core_name is None:
            # Someday we will default to sysmon, but it's still experimental:
            #   if not reason_no_sysmon:
            #       core_name = "sysmon"
            if HAS_CTRACER:
                core_name = "ctrace"
            else:
                core_name = "pytrace"

        self.tracer_kwargs = {}

        if core_name == "sysmon":
            self.tracer_class = SysMonitor
            self.tracer_kwargs["tool_id"] = 3 if metacov else 1
            self.file_disposition_class = FileDisposition
            self.supports_plugins = False
            self.packed_arcs = False
            self.systrace = False
        elif core_name == "ctrace":
            self.tracer_class = CTracer
            self.file_disposition_class = CFileDisposition
            self.supports_plugins = True
            self.packed_arcs = True
            self.systrace = True
        elif core_name == "pytrace":
            self.tracer_class = PyTracer
            self.file_disposition_class = FileDisposition
            self.supports_plugins = False
            self.packed_arcs = False
            self.systrace = True
        else:
            raise ConfigError(f"Unknown core value: {core_name!r}")
