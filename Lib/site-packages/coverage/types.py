# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/nedbat/coveragepy/blob/master/NOTICE.txt

"""
Types for use throughout coverage.py.
"""

from __future__ import annotations

import os
import pathlib

from collections.abc import Iterable, Mapping
from types import FrameType, ModuleType
from typing import (
    Any, Callable, Optional, Protocol,
    Union, TYPE_CHECKING,
)

if TYPE_CHECKING:
    from coverage.plugin import FileTracer


AnyCallable = Callable[..., Any]

## File paths

# For arguments that are file paths:
if TYPE_CHECKING:
    FilePath = Union[str, os.PathLike[str]]
else:
    # PathLike < python3.9 doesn't support subscription
    FilePath = Union[str, os.PathLike]
# For testing FilePath arguments
FilePathClasses = [str, pathlib.Path]
FilePathType = Union[type[str], type[pathlib.Path]]

## Python tracing

class TTraceFn(Protocol):
    """A Python trace function."""
    def __call__(
        self,
        frame: FrameType,
        event: str,
        arg: Any,
        lineno: TLineNo | None = None,  # Our own twist, see collector.py
    ) -> TTraceFn | None:
        ...

## Coverage.py tracing

# Line numbers are pervasive enough that they deserve their own type.
TLineNo = int

# Bytecode offsets are pervasive enough that they deserve their own type.
TOffset = int

TArc = tuple[TLineNo, TLineNo]

class TFileDisposition(Protocol):
    """A simple value type for recording what to do with a file."""

    original_filename: str
    canonical_filename: str
    source_filename: str | None
    trace: bool
    reason: str
    file_tracer: FileTracer | None
    has_dynamic_filename: bool


# When collecting data, we use a dictionary with a few possible shapes. The
# keys are always file names.
# - If measuring line coverage, the values are sets of line numbers.
# - If measuring arcs in the Python tracer, the values are sets of arcs (pairs
#   of line numbers).
# - If measuring arcs in the C tracer, the values are sets of packed arcs (two
#   line numbers combined into one integer).

TTraceFileData = Union[set[TLineNo], set[TArc], set[int]]

TTraceData = dict[str, TTraceFileData]

# Functions passed into collectors.
TShouldTraceFn = Callable[[str, FrameType], TFileDisposition]
TCheckIncludeFn = Callable[[str, FrameType], bool]
TShouldStartContextFn = Callable[[FrameType], Union[str, None]]

class Tracer(Protocol):
    """Anything that can report on Python execution."""

    data: TTraceData
    trace_arcs: bool
    should_trace: TShouldTraceFn
    should_trace_cache: Mapping[str, TFileDisposition | None]
    should_start_context: TShouldStartContextFn | None
    switch_context: Callable[[str | None], None] | None
    lock_data: Callable[[], None]
    unlock_data: Callable[[], None]
    warn: TWarnFn

    def __init__(self) -> None:
        ...

    def start(self) -> TTraceFn | None:
        """Start this tracer, return a trace function if based on sys.settrace."""

    def stop(self) -> None:
        """Stop this tracer."""

    def activity(self) -> bool:
        """Has there been any activity?"""

    def reset_activity(self) -> None:
        """Reset the activity() flag."""

    def get_stats(self) -> dict[str, int] | None:
        """Return a dictionary of statistics, or None."""


## Coverage

# Many places use kwargs as Coverage kwargs.
TCovKwargs = Any


## Configuration

# One value read from a config file.
TConfigValueIn = Optional[Union[bool, int, float, str, Iterable[str], Mapping[str, Iterable[str]]]]
TConfigValueOut = Optional[Union[bool, int, float, str, list[str], dict[str, list[str]]]]
# An entire config section, mapping option names to values.
TConfigSectionIn = Mapping[str, TConfigValueIn]
TConfigSectionOut = Mapping[str, TConfigValueOut]

class TConfigurable(Protocol):
    """Something that can proxy to the coverage configuration settings."""

    def get_option(self, option_name: str) -> TConfigValueOut | None:
        """Get an option from the configuration.

        `option_name` is a colon-separated string indicating the section and
        option name.  For example, the ``branch`` option in the ``[run]``
        section of the config file would be indicated with `"run:branch"`.

        Returns the value of the option.

        """

    def set_option(self, option_name: str, value: TConfigValueIn | TConfigSectionIn) -> None:
        """Set an option in the configuration.

        `option_name` is a colon-separated string indicating the section and
        option name.  For example, the ``branch`` option in the ``[run]``
        section of the config file would be indicated with `"run:branch"`.

        `value` is the new value for the option.

        """

class TPluginConfig(Protocol):
    """Something that can provide options to a plugin."""

    def get_plugin_options(self, plugin: str) -> TConfigSectionOut:
        """Get the options for a plugin."""


## Parsing

TMorf = Union[ModuleType, str]

TSourceTokenLines = Iterable[list[tuple[str, str]]]


## Plugins

class TPlugin(Protocol):
    """What all plugins have in common."""
    _coverage_plugin_name: str
    _coverage_enabled: bool


## Debugging

class TWarnFn(Protocol):
    """A callable warn() function."""
    def __call__(self, msg: str, slug: str | None = None, once: bool = False) -> None:
        ...


class TDebugCtl(Protocol):
    """A DebugControl object, or something like it."""

    def should(self, option: str) -> bool:
        """Decide whether to output debug information in category `option`."""

    def write(self, msg: str) -> None:
        """Write a line of debug output."""


class TWritable(Protocol):
    """Anything that can be written to."""

    def write(self, msg: str) -> None:
        """Write a message."""
