# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/nedbat/coveragepy/blob/master/NOTICE.txt

"""Callback functions and support for sys.monitoring data collection."""

from __future__ import annotations

import functools
import inspect
import os
import os.path
import sys
import threading
import traceback

from dataclasses import dataclass
from types import CodeType
from typing import (
    Any,
    Callable,
    NewType,
    Optional,
    cast,
)

from coverage import env
from coverage.bytecode import TBranchTrails, branch_trails
from coverage.debug import short_filename, short_stack
from coverage.misc import isolate_module
from coverage.types import (
    AnyCallable,
    TFileDisposition,
    TLineNo,
    TOffset,
    TShouldStartContextFn,
    TShouldTraceFn,
    TTraceData,
    TTraceFileData,
    Tracer,
    TWarnFn,
)

os = isolate_module(os)

# pylint: disable=unused-argument

# $set_env.py: COVERAGE_SYSMON_LOG - Log sys.monitoring activity
LOG = bool(int(os.getenv("COVERAGE_SYSMON_LOG", 0)))

# $set_env.py: COVERAGE_SYSMON_STATS - Collect sys.monitoring stats
COLLECT_STATS = bool(int(os.getenv("COVERAGE_SYSMON_STATS", 0)))

# This module will be imported in all versions of Python, but only used in 3.12+
# It will be type-checked for 3.12, but not for earlier versions.
sys_monitoring = getattr(sys, "monitoring", None)

DISABLE_TYPE = NewType("DISABLE_TYPE", object)
MonitorReturn = Optional[DISABLE_TYPE]
DISABLE = cast(MonitorReturn, getattr(sys_monitoring, "DISABLE", None))


if LOG:  # pragma: debugging

    class LoggingWrapper:
        """Wrap a namespace to log all its functions."""

        def __init__(self, wrapped: Any, namespace: str) -> None:
            self.wrapped = wrapped
            self.namespace = namespace

        def __getattr__(self, name: str) -> Callable[..., Any]:
            def _wrapped(*args: Any, **kwargs: Any) -> Any:
                log(f"{self.namespace}.{name}{args}{kwargs}")
                return getattr(self.wrapped, name)(*args, **kwargs)

            return _wrapped

    sys_monitoring = LoggingWrapper(sys_monitoring, "sys.monitoring")
    assert sys_monitoring is not None

    short_stack = functools.partial(
        short_stack,
        full=True,
        short_filenames=True,
        frame_ids=True,
    )
    seen_threads: set[int] = set()

    def log(msg: str) -> None:
        """Write a message to our detailed debugging log(s)."""
        # Thread ids are reused across processes?
        # Make a shorter number more likely to be unique.
        pid = os.getpid()
        tid = cast(int, threading.current_thread().ident)
        tslug = f"{(pid * tid) % 9_999_991:07d}"
        if tid not in seen_threads:
            seen_threads.add(tid)
            log(f"New thread {tid} {tslug}:\n{short_stack()}")
        # log_seq = int(os.getenv("PANSEQ", "0"))
        # root = f"/tmp/pan.{log_seq:03d}"
        for filename in [
            "/tmp/foo.out",
            # f"{root}.out",
            # f"{root}-{pid}.out",
            # f"{root}-{pid}-{tslug}.out",
        ]:
            with open(filename, "a") as f:
                try:
                    print(f"{pid}:{tslug}: {msg}", file=f, flush=True)
                except UnicodeError:
                    print(f"{pid}:{tslug}: {ascii(msg)}", file=f, flush=True)

    def arg_repr(arg: Any) -> str:
        """Make a customized repr for logged values."""
        if isinstance(arg, CodeType):
            return (
                f"<code @{id(arg):#x}"
                + f" name={arg.co_name},"
                + f" file={short_filename(arg.co_filename)!r}#{arg.co_firstlineno}>"
            )
        return repr(arg)

    def panopticon(*names: str | None) -> AnyCallable:
        """Decorate a function to log its calls."""

        def _decorator(method: AnyCallable) -> AnyCallable:
            @functools.wraps(method)
            def _wrapped(self: Any, *args: Any) -> Any:
                try:
                    # log(f"{method.__name__}() stack:\n{short_stack()}")
                    args_reprs = []
                    for name, arg in zip(names, args):
                        if name is None:
                            continue
                        args_reprs.append(f"{name}={arg_repr(arg)}")
                    log(f"{id(self):#x}:{method.__name__}({', '.join(args_reprs)})")
                    ret = method(self, *args)
                    # log(f" end {id(self):#x}:{method.__name__}({', '.join(args_reprs)})")
                    return ret
                except Exception as exc:
                    log(f"!!{exc.__class__.__name__}: {exc}")
                    if 1:
                        # pylint: disable=no-value-for-parameter
                        log("".join(traceback.format_exception(exc)))
                    try:
                        assert sys_monitoring is not None
                        sys_monitoring.set_events(sys.monitoring.COVERAGE_ID, 0)
                    except ValueError:
                        # We might have already shut off monitoring.
                        log("oops, shutting off events with disabled tool id")
                    raise

            return _wrapped

        return _decorator

else:

    def log(msg: str) -> None:
        """Write a message to our detailed debugging log(s), but not really."""

    def panopticon(*names: str | None) -> AnyCallable:
        """Decorate a function to log its calls, but not really."""

        def _decorator(meth: AnyCallable) -> AnyCallable:
            return meth

        return _decorator


@dataclass
class CodeInfo:
    """The information we want about each code object."""

    tracing: bool
    file_data: TTraceFileData | None
    byte_to_line: dict[TOffset, TLineNo] | None

    # Keys are start instruction offsets for branches.
    # Values are lists:
    #   [
    #       ([offset, offset, ...], (from_line, to_line)),
    #       ([offset, offset, ...], (from_line, to_line)),
    #   ]
    #   Two possible trails from the branch point, left and right.
    branch_trails: TBranchTrails


def bytes_to_lines(code: CodeType) -> dict[TOffset, TLineNo]:
    """Make a dict mapping byte code offsets to line numbers."""
    b2l = {}
    for bstart, bend, lineno in code.co_lines():
        if lineno is not None:
            for boffset in range(bstart, bend, 2):
                b2l[boffset] = lineno
    return b2l


class SysMonitor(Tracer):
    """Python implementation of the raw data tracer for PEP669 implementations."""

    # One of these will be used across threads. Be careful.

    def __init__(self, tool_id: int) -> None:
        # Attributes set from the collector:
        self.data: TTraceData
        self.trace_arcs = False
        self.should_trace: TShouldTraceFn
        self.should_trace_cache: dict[str, TFileDisposition | None]
        # TODO: should_start_context and switch_context are unused!
        # Change tests/testenv.py:DYN_CONTEXTS when this is updated.
        self.should_start_context: TShouldStartContextFn | None = None
        self.switch_context: Callable[[str | None], None] | None = None
        self.lock_data: Callable[[], None]
        self.unlock_data: Callable[[], None]
        # TODO: warn is unused.
        self.warn: TWarnFn

        self.myid = tool_id

        # Map id(code_object) -> CodeInfo
        self.code_infos: dict[int, CodeInfo] = {}
        # A list of code_objects, just to keep them alive so that id's are
        # useful as identity.
        self.code_objects: list[CodeType] = []
        self.sysmon_on = False
        self.lock = threading.Lock()

        self.stats: dict[str, int] | None = None
        if COLLECT_STATS:
            self.stats = dict.fromkeys(
                "starts start_tracing returns line_lines line_arcs branches branch_trails".split(),
                0,
            )

        self.stopped = False
        self._activity = False

    def __repr__(self) -> str:
        points = sum(len(v) for v in self.data.values())
        files = len(self.data)
        return f"<SysMonitor at {id(self):#x}: {points} data points in {files} files>"

    @panopticon()
    def start(self) -> None:
        """Start this Tracer."""
        self.stopped = False

        assert sys_monitoring is not None
        sys_monitoring.use_tool_id(self.myid, "coverage.py")
        register = functools.partial(sys_monitoring.register_callback, self.myid)
        events = sys.monitoring.events

        sys_monitoring.set_events(self.myid, events.PY_START)
        register(events.PY_START, self.sysmon_py_start)
        if self.trace_arcs:
            register(events.PY_RETURN, self.sysmon_py_return)
            register(events.LINE, self.sysmon_line_arcs)
            if env.PYBEHAVIOR.branch_right_left:
                register(
                    events.BRANCH_RIGHT,  # type:ignore[attr-defined]
                    self.sysmon_branch_either,
                )
                register(
                    events.BRANCH_LEFT,  # type:ignore[attr-defined]
                    self.sysmon_branch_either,
                )
        else:
            register(events.LINE, self.sysmon_line_lines)
        sys_monitoring.restart_events()
        self.sysmon_on = True

    @panopticon()
    def stop(self) -> None:
        """Stop this Tracer."""
        if not self.sysmon_on:
            # In forking situations, we might try to stop when we are not
            # started.  Do nothing in that case.
            return
        assert sys_monitoring is not None
        sys_monitoring.set_events(self.myid, 0)
        self.sysmon_on = False
        sys_monitoring.free_tool_id(self.myid)

    @panopticon()
    def post_fork(self) -> None:
        """The process has forked, clean up as needed."""
        self.stop()

    def activity(self) -> bool:
        """Has there been any activity?"""
        return self._activity

    def reset_activity(self) -> None:
        """Reset the activity() flag."""
        self._activity = False

    def get_stats(self) -> dict[str, int] | None:
        """Return a dictionary of statistics, or None."""
        return self.stats

    @panopticon("code", "@")
    def sysmon_py_start(
        self, code: CodeType, instruction_offset: TOffset
    ) -> MonitorReturn:
        """Handle sys.monitoring.events.PY_START events."""
        # Entering a new frame.  Decide if we should trace in this file.
        self._activity = True
        if self.stats is not None:
            self.stats["starts"] += 1

        code_info = self.code_infos.get(id(code))
        tracing_code: bool | None = None
        file_data: TTraceFileData | None = None
        if code_info is not None:
            tracing_code = code_info.tracing
            file_data = code_info.file_data

        if tracing_code is None:
            filename = code.co_filename
            disp = self.should_trace_cache.get(filename)
            if disp is None:
                frame = inspect.currentframe().f_back  # type: ignore[union-attr]
                if LOG:
                    # @panopticon adds a frame.
                    frame = frame.f_back  # type: ignore[union-attr]
                disp = self.should_trace(filename, frame)  # type: ignore[arg-type]
                self.should_trace_cache[filename] = disp

            tracing_code = disp.trace
            if tracing_code:
                tracename = disp.source_filename
                assert tracename is not None
                self.lock_data()
                try:
                    if tracename not in self.data:
                        self.data[tracename] = set()
                finally:
                    self.unlock_data()
                file_data = self.data[tracename]
                b2l = bytes_to_lines(code)
            else:
                file_data = None
                b2l = None

            code_info = CodeInfo(
                tracing=tracing_code,
                file_data=file_data,
                byte_to_line=b2l,
                branch_trails={},
            )
            self.code_infos[id(code)] = code_info
            self.code_objects.append(code)

            if tracing_code:
                if self.stats is not None:
                    self.stats["start_tracing"] += 1
                events = sys.monitoring.events
                with self.lock:
                    if self.sysmon_on:
                        assert sys_monitoring is not None
                        local_events = events.PY_RETURN | events.PY_RESUME | events.LINE
                        if self.trace_arcs:
                            assert env.PYBEHAVIOR.branch_right_left
                            local_events |= (
                                events.BRANCH_RIGHT  # type:ignore[attr-defined]
                                | events.BRANCH_LEFT  # type:ignore[attr-defined]
                            )
                        sys_monitoring.set_local_events(self.myid, code, local_events)

        return DISABLE

    @panopticon("code", "@", None)
    def sysmon_py_return(
        self,
        code: CodeType,
        instruction_offset: TOffset,
        retval: object,
    ) -> MonitorReturn:
        """Handle sys.monitoring.events.PY_RETURN events for branch coverage."""
        if self.stats is not None:
            self.stats["returns"] += 1
        code_info = self.code_infos.get(id(code))
        # code_info is not None and code_info.file_data is not None, since we
        # wouldn't have enabled this event if they were.
        last_line = code_info.byte_to_line[instruction_offset]  # type: ignore
        if last_line is not None:
            arc = (last_line, -code.co_firstlineno)
            code_info.file_data.add(arc)  # type: ignore
            # log(f"adding {arc=}")
        return DISABLE

    @panopticon("code", "line")
    def sysmon_line_lines(self, code: CodeType, line_number: TLineNo) -> MonitorReturn:
        """Handle sys.monitoring.events.LINE events for line coverage."""
        if self.stats is not None:
            self.stats["line_lines"] += 1
        code_info = self.code_infos.get(id(code))
        # It should be true that code_info is not None and code_info.file_data
        # is not None, since we wouldn't have enabled this event if they were.
        # But somehow code_info can be None here, so we have to check.
        if code_info is not None and code_info.file_data is not None:
            code_info.file_data.add(line_number)  # type: ignore
        # log(f"adding {line_number=}")
        return DISABLE

    @panopticon("code", "line")
    def sysmon_line_arcs(self, code: CodeType, line_number: TLineNo) -> MonitorReturn:
        """Handle sys.monitoring.events.LINE events for branch coverage."""
        if self.stats is not None:
            self.stats["line_arcs"] += 1
        code_info = self.code_infos[id(code)]
        # code_info is not None and code_info.file_data is not None, since we
        # wouldn't have enabled this event if they were.
        arc = (line_number, line_number)
        code_info.file_data.add(arc)  # type: ignore
        # log(f"adding {arc=}")
        return DISABLE

    @panopticon("code", "@", "@")
    def sysmon_branch_either(
        self, code: CodeType, instruction_offset: TOffset, destination_offset: TOffset
    ) -> MonitorReturn:
        """Handle BRANCH_RIGHT and BRANCH_LEFT events."""
        if self.stats is not None:
            self.stats["branches"] += 1
        code_info = self.code_infos[id(code)]
        # code_info is not None and code_info.file_data is not None, since we
        # wouldn't have enabled this event if they were.
        if not code_info.branch_trails:
            if self.stats is not None:
                self.stats["branch_trails"] += 1
            code_info.branch_trails = branch_trails(code)
            # log(f"branch_trails for {code}:\n    {code_info.branch_trails}")
        added_arc = False
        dest_info = code_info.branch_trails.get(instruction_offset)
        # log(f"{dest_info = }")
        if dest_info is not None:
            for offsets, arc in dest_info:
                if arc is None:
                    continue
                if destination_offset in offsets:
                    code_info.file_data.add(arc)  # type: ignore
                    # log(f"adding {arc=}")
                    added_arc = True
                    break

        if not added_arc:
            # This could be an exception jumping from line to line.
            assert code_info.byte_to_line is not None
            l1 = code_info.byte_to_line[instruction_offset]
            l2 = code_info.byte_to_line[destination_offset]
            if l1 != l2:
                arc = (l1, l2)
                code_info.file_data.add(arc)  # type: ignore
                # log(f"adding unforeseen {arc=}")

        return DISABLE
