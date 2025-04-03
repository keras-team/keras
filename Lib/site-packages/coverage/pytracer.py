# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/nedbat/coveragepy/blob/master/NOTICE.txt

"""Raw data collector for coverage.py."""

from __future__ import annotations

import atexit
import dis
import itertools
import sys
import threading

from types import FrameType, ModuleType
from typing import Any, Callable, cast

from coverage import env
from coverage.types import (
    TArc,
    TFileDisposition,
    TLineNo,
    TShouldStartContextFn,
    TShouldTraceFn,
    TTraceData,
    TTraceFileData,
    TTraceFn,
    TWarnFn,
    Tracer,
)


# I don't understand why, but if we use `cast(set[TLineNo], ...)` inside
# the _trace() function, we get some strange behavior on PyPy 3.10.
# Assigning these names here and using them below fixes the problem.
# See https://github.com/nedbat/coveragepy/issues/1902
set_TLineNo = set[TLineNo]
set_TArc = set[TArc]


# We need the YIELD_VALUE opcode below, in a comparison-friendly form.
# PYVERSIONS: RESUME is new in Python3.11
RESUME = dis.opmap.get("RESUME")
RETURN_VALUE = dis.opmap["RETURN_VALUE"]
if RESUME is None:
    YIELD_VALUE = dis.opmap["YIELD_VALUE"]
    YIELD_FROM = dis.opmap["YIELD_FROM"]
    YIELD_FROM_OFFSET = 0 if env.PYPY else 2
else:
    YIELD_VALUE = YIELD_FROM = YIELD_FROM_OFFSET = -1

# When running meta-coverage, this file can try to trace itself, which confuses
# everything.  Don't trace ourselves.

THIS_FILE = __file__.rstrip("co")

class PyTracer(Tracer):
    """Python implementation of the raw data tracer."""

    # Because of poor implementations of trace-function-manipulating tools,
    # the Python trace function must be kept very simple.  In particular, there
    # must be only one function ever set as the trace function, both through
    # sys.settrace, and as the return value from the trace function.  Put
    # another way, the trace function must always return itself.  It cannot
    # swap in other functions, or return None to avoid tracing a particular
    # frame.
    #
    # The trace manipulator that introduced this restriction is DecoratorTools,
    # which sets a trace function, and then later restores the pre-existing one
    # by calling sys.settrace with a function it found in the current frame.
    #
    # Systems that use DecoratorTools (or similar trace manipulations) must use
    # PyTracer to get accurate results.  The command-line --timid argument is
    # used to force the use of this tracer.

    tracer_ids = itertools.count()

    def __init__(self) -> None:
        # Which tracer are we?
        self.id = next(self.tracer_ids)

        # Attributes set from the collector:
        self.data: TTraceData
        self.trace_arcs = False
        self.should_trace: TShouldTraceFn
        self.should_trace_cache: dict[str, TFileDisposition | None]
        self.should_start_context: TShouldStartContextFn | None = None
        self.switch_context: Callable[[str | None], None] | None = None
        self.lock_data: Callable[[], None]
        self.unlock_data: Callable[[], None]
        self.warn: TWarnFn

        # The threading module to use, if any.
        self.threading: ModuleType | None = None

        self.cur_file_data: TTraceFileData | None = None
        self.last_line: TLineNo = 0
        self.cur_file_name: str | None = None
        self.context: str | None = None
        self.started_context = False

        # The data_stack parallels the Python call stack. Each entry is
        # information about an active frame, a four-element tuple:
        #   [0] The TTraceData for this frame's file. Could be None if we
        #           aren't tracing this frame.
        #   [1] The current file name for the frame. None if we aren't tracing
        #           this frame.
        #   [2] The last line number executed in this frame.
        #   [3] Boolean: did this frame start a new context?
        self.data_stack: list[tuple[TTraceFileData | None, str | None, TLineNo, bool]] = []
        self.thread: threading.Thread | None = None
        self.stopped = False
        self._activity = False

        self.in_atexit = False
        # On exit, self.in_atexit = True
        atexit.register(setattr, self, "in_atexit", True)

        # Cache a bound method on the instance, so that we don't have to
        # re-create a bound method object all the time.
        self._cached_bound_method_trace: TTraceFn = self._trace

    def __repr__(self) -> str:
        points = sum(len(v) for v in self.data.values())
        files = len(self.data)
        return f"<PyTracer at {id(self):#x}: {points} data points in {files} files>"

    def log(self, marker: str, *args: Any) -> None:
        """For hard-core logging of what this tracer is doing."""
        with open("/tmp/debug_trace.txt", "a") as f:
            f.write(f"{marker} {self.id}[{len(self.data_stack)}]")
            if 0:   # if you want thread ids..
                f.write(".{:x}.{:x}".format(                    # type: ignore[unreachable]
                    self.thread.ident,
                    self.threading.current_thread().ident,
                ))
            f.write(" {}".format(" ".join(map(str, args))))
            if 0:   # if you want callers..
                f.write(" | ")                                  # type: ignore[unreachable]
                stack = " / ".join(
                    (fname or "???").rpartition("/")[-1]
                    for _, fname, _, _ in self.data_stack
                )
                f.write(stack)
            f.write("\n")

    def _trace(
        self,
        frame: FrameType,
        event: str,
        arg: Any,                               # pylint: disable=unused-argument
        lineno: TLineNo | None = None,       # pylint: disable=unused-argument
    ) -> TTraceFn | None:
        """The trace function passed to sys.settrace."""

        if THIS_FILE in frame.f_code.co_filename:
            return None

        # f = frame; code = f.f_code
        # self.log(":", f"{code.co_filename} {f.f_lineno} {code.co_name}()", event)

        if (self.stopped and sys.gettrace() == self._cached_bound_method_trace):    # pylint: disable=comparison-with-callable
            # The PyTrace.stop() method has been called, possibly by another
            # thread, let's deactivate ourselves now.
            if 0:
                f = frame                           # type: ignore[unreachable]
                self.log("---\nX", f.f_code.co_filename, f.f_lineno)
                while f:
                    self.log(">", f.f_code.co_filename, f.f_lineno, f.f_code.co_name, f.f_trace)
                    f = f.f_back
            sys.settrace(None)
            try:
                self.cur_file_data, self.cur_file_name, self.last_line, self.started_context = (
                    self.data_stack.pop()
                )
            except IndexError:
                self.log(
                    "Empty stack!",
                    frame.f_code.co_filename,
                    frame.f_lineno,
                    frame.f_code.co_name,
                )
            return None

        # if event != "call" and frame.f_code.co_filename != self.cur_file_name:
        #     self.log("---\n*", frame.f_code.co_filename, self.cur_file_name, frame.f_lineno)

        if event == "call":
            # Should we start a new context?
            if self.should_start_context and self.context is None:
                context_maybe = self.should_start_context(frame)    # pylint: disable=not-callable
                if context_maybe is not None:
                    self.context = context_maybe
                    started_context = True
                    assert self.switch_context is not None
                    self.switch_context(self.context)   # pylint: disable=not-callable
                else:
                    started_context = False
            else:
                started_context = False
            self.started_context = started_context

            # Entering a new frame.  Decide if we should trace in this file.
            self._activity = True
            self.data_stack.append(
                (
                    self.cur_file_data,
                    self.cur_file_name,
                    self.last_line,
                    started_context,
                ),
            )

            # Improve tracing performance: when calling a function, both caller
            # and callee are often within the same file. if that's the case, we
            # don't have to re-check whether to trace the corresponding
            # function (which is a little bit expensive since it involves
            # dictionary lookups). This optimization is only correct if we
            # didn't start a context.
            filename = frame.f_code.co_filename
            if filename != self.cur_file_name or started_context:
                self.cur_file_name = filename
                disp = self.should_trace_cache.get(filename)
                if disp is None:
                    disp = self.should_trace(filename, frame)
                    self.should_trace_cache[filename] = disp

                self.cur_file_data = None
                if disp.trace:
                    tracename = disp.source_filename
                    assert tracename is not None
                    self.lock_data()
                    try:
                        if tracename not in self.data:
                            self.data[tracename] = set()
                    finally:
                        self.unlock_data()
                    self.cur_file_data = self.data[tracename]
                else:
                    frame.f_trace_lines = False
            elif not self.cur_file_data:
                frame.f_trace_lines = False

            # The call event is really a "start frame" event, and happens for
            # function calls and re-entering generators.  The f_lasti field is
            # -1 for calls, and a real offset for generators.  Use <0 as the
            # line number for calls, and the real line number for generators.
            if RESUME is not None:
                # The current opcode is guaranteed to be RESUME. The argument
                # determines what kind of resume it is.
                oparg = frame.f_code.co_code[frame.f_lasti + 1]
                real_call = (oparg == 0)
            else:
                real_call = (getattr(frame, "f_lasti", -1) < 0)
            if real_call:
                self.last_line = -frame.f_code.co_firstlineno
            else:
                self.last_line = frame.f_lineno

        elif event == "line":
            # Record an executed line.
            if self.cur_file_data is not None:
                flineno: TLineNo = frame.f_lineno

                if self.trace_arcs:
                    cast(set_TArc, self.cur_file_data).add((self.last_line, flineno))
                else:
                    cast(set_TLineNo, self.cur_file_data).add(flineno)
                self.last_line = flineno

        elif event == "return":
            if self.trace_arcs and self.cur_file_data:
                # Record an arc leaving the function, but beware that a
                # "return" event might just mean yielding from a generator.
                code = frame.f_code.co_code
                lasti = frame.f_lasti
                if RESUME is not None:
                    if len(code) == lasti + 2:
                        # A return from the end of a code object is a real return.
                        real_return = True
                    else:
                        # It is a real return if we aren't going to resume next.
                        if env.PYBEHAVIOR.lasti_is_yield:
                            lasti += 2
                        real_return = (code[lasti] != RESUME)
                else:
                    if code[lasti] == RETURN_VALUE:
                        real_return = True
                    elif code[lasti] == YIELD_VALUE:
                        real_return = False
                    elif len(code) <= lasti + YIELD_FROM_OFFSET:
                        real_return = True
                    elif code[lasti + YIELD_FROM_OFFSET] == YIELD_FROM:
                        real_return = False
                    else:
                        real_return = True
                if real_return:
                    first = frame.f_code.co_firstlineno
                    cast(set_TArc, self.cur_file_data).add((self.last_line, -first))

            # Leaving this function, pop the filename stack.
            self.cur_file_data, self.cur_file_name, self.last_line, self.started_context = (
                self.data_stack.pop()
            )
            # Leaving a context?
            if self.started_context:
                assert self.switch_context is not None
                self.context = None
                self.switch_context(None)   # pylint: disable=not-callable
        return self._cached_bound_method_trace

    def start(self) -> TTraceFn:
        """Start this Tracer.

        Return a Python function suitable for use with sys.settrace().

        """
        self.stopped = False
        if self.threading:
            if self.thread is None:
                self.thread = self.threading.current_thread()

        sys.settrace(self._cached_bound_method_trace)
        return self._cached_bound_method_trace

    def stop(self) -> None:
        """Stop this Tracer."""
        # Get the active tracer callback before setting the stop flag to be
        # able to detect if the tracer was changed prior to stopping it.
        tf = sys.gettrace()

        # Set the stop flag. The actual call to sys.settrace(None) will happen
        # in the self._trace callback itself to make sure to call it from the
        # right thread.
        self.stopped = True

        if self.threading:
            assert self.thread is not None
            if self.thread.ident != self.threading.current_thread().ident:
                # Called on a different thread than started us: we can't unhook
                # ourselves, but we've set the flag that we should stop, so we
                # won't do any more tracing.
                #self.log("~", "stopping on different threads")
                return

        # PyPy clears the trace function before running atexit functions,
        # so don't warn if we are in atexit on PyPy and the trace function
        # has changed to None.  Metacoverage also messes this up, so don't
        # warn if we are measuring ourselves.
        suppress_warning = (
            (env.PYPY and self.in_atexit and tf is None)
            or env.METACOV
        )
        if self.warn and not suppress_warning:
            if tf != self._cached_bound_method_trace:   # pylint: disable=comparison-with-callable
                self.warn(
                    "Trace function changed, data is likely wrong: " +
                    f"{tf!r} != {self._cached_bound_method_trace!r}",
                    slug="trace-changed",
                )

    def activity(self) -> bool:
        """Has there been any activity?"""
        return self._activity

    def reset_activity(self) -> None:
        """Reset the activity() flag."""
        self._activity = False

    def get_stats(self) -> dict[str, int] | None:
        """Return a dictionary of statistics, or None."""
        return None
