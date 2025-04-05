# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/nedbat/coveragepy/blob/master/NOTICE.txt

"""Raw data collector for coverage.py."""

from __future__ import annotations

import contextlib
import functools
import os
import sys

from collections.abc import Mapping
from types import FrameType
from typing import cast, Any, Callable, TypeVar

from coverage import env
from coverage.config import CoverageConfig
from coverage.core import Core
from coverage.data import CoverageData
from coverage.debug import short_stack
from coverage.exceptions import ConfigError
from coverage.misc import human_sorted_items, isolate_module
from coverage.plugin import CoveragePlugin
from coverage.types import (
    TArc,
    TCheckIncludeFn,
    TFileDisposition,
    TShouldStartContextFn,
    TShouldTraceFn,
    TTraceData,
    TTraceFn,
    Tracer,
    TWarnFn,
)

os = isolate_module(os)


T = TypeVar("T")


class Collector:
    """Collects trace data.

    Creates a Tracer object for each thread, since they track stack
    information.  Each Tracer points to the same shared data, contributing
    traced data points.

    When the Collector is started, it creates a Tracer for the current thread,
    and installs a function to create Tracers for each new thread started.
    When the Collector is stopped, all active Tracers are stopped.

    Threads started while the Collector is stopped will never have Tracers
    associated with them.

    """

    # The stack of active Collectors.  Collectors are added here when started,
    # and popped when stopped.  Collectors on the stack are paused when not
    # the top, and resumed when they become the top again.
    _collectors: list[Collector] = []

    # The concurrency settings we support here.
    LIGHT_THREADS = {"greenlet", "eventlet", "gevent"}

    def __init__(
        self,
        core: Core,
        should_trace: TShouldTraceFn,
        check_include: TCheckIncludeFn,
        should_start_context: TShouldStartContextFn | None,
        file_mapper: Callable[[str], str],
        branch: bool,
        warn: TWarnFn,
        concurrency: list[str],
    ) -> None:
        """Create a collector.

        `should_trace` is a function, taking a file name and a frame, and
        returning a `coverage.FileDisposition object`.

        `check_include` is a function taking a file name and a frame. It returns
        a boolean: True if the file should be traced, False if not.

        `should_start_context` is a function taking a frame, and returning a
        string. If the frame should be the start of a new context, the string
        is the new context. If the frame should not be the start of a new
        context, return None.

        `file_mapper` is a function taking a filename, and returning a Unicode
        filename.  The result is the name that will be recorded in the data
        file.

        If `branch` is true, then branches will be measured.  This involves
        collecting data on which statements followed each other (arcs).  Use
        `get_arc_data` to get the arc data.

        `warn` is a warning function, taking a single string message argument
        and an optional slug argument which will be a string or None, to be
        used if a warning needs to be issued.

        `concurrency` is a list of strings indicating the concurrency libraries
        in use.  Valid values are "greenlet", "eventlet", "gevent", or "thread"
        (the default).  "thread" can be combined with one of the other three.
        Other values are ignored.

        """
        self.core = core
        self.should_trace = should_trace
        self.check_include = check_include
        self.should_start_context = should_start_context
        self.file_mapper = file_mapper
        self.branch = branch
        self.warn = warn
        self.concurrency = concurrency
        assert isinstance(self.concurrency, list), f"Expected a list: {self.concurrency!r}"

        self.pid = os.getpid()

        self.covdata: CoverageData
        self.threading = None
        self.static_context: str | None = None

        self.origin = short_stack()

        self.concur_id_func = None

        # We can handle a few concurrency options here, but only one at a time.
        concurrencies = set(self.concurrency)
        unknown = concurrencies - CoverageConfig.CONCURRENCY_CHOICES
        if unknown:
            show = ", ".join(sorted(unknown))
            raise ConfigError(f"Unknown concurrency choices: {show}")
        light_threads = concurrencies & self.LIGHT_THREADS
        if len(light_threads) > 1:
            show = ", ".join(sorted(light_threads))
            raise ConfigError(f"Conflicting concurrency settings: {show}")
        do_threading = False

        tried = "nothing"  # to satisfy pylint
        try:
            if "greenlet" in concurrencies:
                tried = "greenlet"
                import greenlet
                self.concur_id_func = greenlet.getcurrent
            elif "eventlet" in concurrencies:
                tried = "eventlet"
                import eventlet.greenthread     # pylint: disable=import-error,useless-suppression
                self.concur_id_func = eventlet.greenthread.getcurrent
            elif "gevent" in concurrencies:
                tried = "gevent"
                import gevent                   # pylint: disable=import-error,useless-suppression
                self.concur_id_func = gevent.getcurrent

            if "thread" in concurrencies:
                do_threading = True
        except ImportError as ex:
            msg = f"Couldn't trace with concurrency={tried}, the module isn't installed."
            raise ConfigError(msg) from ex

        if self.concur_id_func and not hasattr(core.tracer_class, "concur_id_func"):
            raise ConfigError(
                "Can't support concurrency={} with {}, only threads are supported.".format(
                    tried, self.tracer_name(),
                ),
            )

        if do_threading or not concurrencies:
            # It's important to import threading only if we need it.  If
            # it's imported early, and the program being measured uses
            # gevent, then gevent's monkey-patching won't work properly.
            import threading
            self.threading = threading

        self.reset()

    def __repr__(self) -> str:
        return f"<Collector at {id(self):#x}: {self.tracer_name()}>"

    def use_data(self, covdata: CoverageData, context: str | None) -> None:
        """Use `covdata` for recording data."""
        self.covdata = covdata
        self.static_context = context
        self.covdata.set_context(self.static_context)

    def tracer_name(self) -> str:
        """Return the class name of the tracer we're using."""
        return self.core.tracer_class.__name__

    def _clear_data(self) -> None:
        """Clear out existing data, but stay ready for more collection."""
        # We used to use self.data.clear(), but that would remove filename
        # keys and data values that were still in use higher up the stack
        # when we are called as part of switch_context.
        with self.data_lock or contextlib.nullcontext():
            for d in self.data.values():
                d.clear()

        for tracer in self.tracers:
            tracer.reset_activity()

    def reset(self) -> None:
        """Clear collected data, and prepare to collect more."""
        self.data_lock = self.threading.Lock() if self.threading else None

        # The trace data we are collecting.
        self.data: TTraceData = {}

        # A dictionary mapping file names to file tracer plugin names that will
        # handle them.
        self.file_tracers: dict[str, str] = {}

        self.disabled_plugins: set[str] = set()

        # The .should_trace_cache attribute is a cache from file names to
        # coverage.FileDisposition objects, or None.  When a file is first
        # considered for tracing, a FileDisposition is obtained from
        # Coverage.should_trace.  Its .trace attribute indicates whether the
        # file should be traced or not.  If it should be, a plugin with dynamic
        # file names can decide not to trace it based on the dynamic file name
        # being excluded by the inclusion rules, in which case the
        # FileDisposition will be replaced by None in the cache.
        if env.PYPY:
            import __pypy__                     # pylint: disable=import-error
            # Alex Gaynor said:
            # should_trace_cache is a strictly growing key: once a key is in
            # it, it never changes.  Further, the keys used to access it are
            # generally constant, given sufficient context. That is to say, at
            # any given point _trace() is called, pypy is able to know the key.
            # This is because the key is determined by the physical source code
            # line, and that's invariant with the call site.
            #
            # This property of a dict with immutable keys, combined with
            # call-site-constant keys is a match for PyPy's module dict,
            # which is optimized for such workloads.
            #
            # This gives a 20% benefit on the workload described at
            # https://bitbucket.org/pypy/pypy/issue/1871/10x-slower-than-cpython-under-coverage
            self.should_trace_cache = __pypy__.newdict("module")
        else:
            self.should_trace_cache = {}

        # Our active Tracers.
        self.tracers: list[Tracer] = []

        self._clear_data()

    def lock_data(self) -> None:
        """Lock self.data_lock, for use by the C tracer."""
        if self.data_lock is not None:
            self.data_lock.acquire()

    def unlock_data(self) -> None:
        """Unlock self.data_lock, for use by the C tracer."""
        if self.data_lock is not None:
            self.data_lock.release()

    def _start_tracer(self) -> TTraceFn | None:
        """Start a new Tracer object, and store it in self.tracers."""
        tracer = self.core.tracer_class(**self.core.tracer_kwargs)
        tracer.data = self.data
        tracer.lock_data = self.lock_data
        tracer.unlock_data = self.unlock_data
        tracer.trace_arcs = self.branch
        tracer.should_trace = self.should_trace
        tracer.should_trace_cache = self.should_trace_cache
        tracer.warn = self.warn

        if hasattr(tracer, 'concur_id_func'):
            tracer.concur_id_func = self.concur_id_func
        if hasattr(tracer, 'file_tracers'):
            tracer.file_tracers = self.file_tracers
        if hasattr(tracer, 'threading'):
            tracer.threading = self.threading
        if hasattr(tracer, 'check_include'):
            tracer.check_include = self.check_include
        if hasattr(tracer, 'should_start_context'):
            tracer.should_start_context = self.should_start_context
        if hasattr(tracer, 'switch_context'):
            tracer.switch_context = self.switch_context
        if hasattr(tracer, 'disable_plugin'):
            tracer.disable_plugin = self.disable_plugin

        fn = tracer.start()
        self.tracers.append(tracer)

        return fn

    # The trace function has to be set individually on each thread before
    # execution begins.  Ironically, the only support the threading module has
    # for running code before the thread main is the tracing function.  So we
    # install this as a trace function, and the first time it's called, it does
    # the real trace installation.
    #
    # New in 3.12: threading.settrace_all_threads: https://github.com/python/cpython/pull/96681

    def _installation_trace(self, frame: FrameType, event: str, arg: Any) -> TTraceFn | None:
        """Called on new threads, installs the real tracer."""
        # Remove ourselves as the trace function.
        sys.settrace(None)
        # Install the real tracer.
        fn: TTraceFn | None = self._start_tracer()
        # Invoke the real trace function with the current event, to be sure
        # not to lose an event.
        if fn:
            fn = fn(frame, event, arg)
        # Return the new trace function to continue tracing in this scope.
        return fn

    def start(self) -> None:
        """Start collecting trace information."""
        # We may be a new collector in a forked process.  The old process'
        # collectors will be in self._collectors, but they won't be usable.
        # Find them and discard them.
        keep_collectors = []
        for c in self._collectors:
            if c.pid == self.pid:
                keep_collectors.append(c)
            else:
                c.post_fork()
        self._collectors[:] = keep_collectors

        if self._collectors:
            self._collectors[-1].pause()

        self.tracers = []

        try:
            # Install the tracer on this thread.
            self._start_tracer()
        except:
            if self._collectors:
                self._collectors[-1].resume()
            raise

        # If _start_tracer succeeded, then we add ourselves to the global
        # stack of collectors.
        self._collectors.append(self)

        # Install our installation tracer in threading, to jump-start other
        # threads.
        if self.core.systrace and self.threading:
            self.threading.settrace(self._installation_trace)

    def stop(self) -> None:
        """Stop collecting trace information."""
        assert self._collectors
        if self._collectors[-1] is not self:
            print("self._collectors:")
            for c in self._collectors:
                print(f"  {c!r}\n{c.origin}")
        assert self._collectors[-1] is self, (
            f"Expected current collector to be {self!r}, but it's {self._collectors[-1]!r}"
        )

        self.pause()

        # Remove this Collector from the stack, and resume the one underneath (if any).
        self._collectors.pop()
        if self._collectors:
            self._collectors[-1].resume()

    def pause(self) -> None:
        """Pause tracing, but be prepared to `resume`."""
        for tracer in self.tracers:
            tracer.stop()
            stats = tracer.get_stats()
            if stats:
                print(f"\nCoverage.py {tracer.__class__.__name__} stats:")
                for k, v in human_sorted_items(stats.items()):
                    print(f"{k:>20}: {v}")
        if self.threading:
            self.threading.settrace(None)

    def resume(self) -> None:
        """Resume tracing after a `pause`."""
        for tracer in self.tracers:
            tracer.start()
        if self.core.systrace:
            if self.threading:
                self.threading.settrace(self._installation_trace)
            else:
                self._start_tracer()

    def post_fork(self) -> None:
        """After a fork, tracers might need to adjust."""
        for tracer in self.tracers:
            if hasattr(tracer, "post_fork"):
                tracer.post_fork()

    def _activity(self) -> bool:
        """Has any activity been traced?

        Returns a boolean, True if any trace function was invoked.

        """
        return any(tracer.activity() for tracer in self.tracers)

    def switch_context(self, new_context: str | None) -> None:
        """Switch to a new dynamic context."""
        context: str | None
        self.flush_data()
        if self.static_context:
            context = self.static_context
            if new_context:
                context += "|" + new_context
        else:
            context = new_context
        self.covdata.set_context(context)

    def disable_plugin(self, disposition: TFileDisposition) -> None:
        """Disable the plugin mentioned in `disposition`."""
        file_tracer = disposition.file_tracer
        assert file_tracer is not None
        plugin = file_tracer._coverage_plugin
        plugin_name = plugin._coverage_plugin_name
        self.warn(f"Disabling plug-in {plugin_name!r} due to previous exception")
        plugin._coverage_enabled = False
        disposition.trace = False

    @functools.cache          # pylint: disable=method-cache-max-size-none
    def cached_mapped_file(self, filename: str) -> str:
        """A locally cached version of file names mapped through file_mapper."""
        return self.file_mapper(filename)

    def mapped_file_dict(self, d: Mapping[str, T]) -> dict[str, T]:
        """Return a dict like d, but with keys modified by file_mapper."""
        # The call to list(items()) ensures that the GIL protects the dictionary
        # iterator against concurrent modifications by tracers running
        # in other threads. We try three times in case of concurrent
        # access, hoping to get a clean copy.
        runtime_err = None
        for _ in range(3):                      # pragma: part covered
            try:
                items = list(d.items())
            except RuntimeError as ex:          # pragma: cant happen
                runtime_err = ex
            else:
                break
        else:                                   # pragma: cant happen
            assert isinstance(runtime_err, Exception)
            raise runtime_err

        return {self.cached_mapped_file(k): v for k, v in items if v}

    def plugin_was_disabled(self, plugin: CoveragePlugin) -> None:
        """Record that `plugin` was disabled during the run."""
        self.disabled_plugins.add(plugin._coverage_plugin_name)

    def flush_data(self) -> bool:
        """Save the collected data to our associated `CoverageData`.

        Data may have also been saved along the way. This forces the
        last of the data to be saved.

        Returns True if there was data to save, False if not.
        """
        if not self._activity():
            return False

        if self.branch:
            if self.core.packed_arcs:
                # Unpack the line number pairs packed into integers.  See
                # tracer.c:CTracer_record_pair for the C code that creates
                # these packed ints.
                arc_data: dict[str, list[TArc]] = {}
                packed_data = cast(dict[str, set[int]], self.data)

                # The list() here and in the inner loop are to get a clean copy
                # even as tracers are continuing to add data.
                for fname, packeds in list(packed_data.items()):
                    tuples = []
                    for packed in list(packeds):
                        l1 = packed & 0xFFFFF
                        l2 = (packed & (0xFFFFF << 20)) >> 20
                        if packed & (1 << 40):
                            l1 *= -1
                        if packed & (1 << 41):
                            l2 *= -1
                        tuples.append((l1, l2))
                    arc_data[fname] = tuples
            else:
                arc_data = cast(dict[str, list[TArc]], self.data)
            self.covdata.add_arcs(self.mapped_file_dict(arc_data))
        else:
            line_data = cast(dict[str, set[int]], self.data)
            self.covdata.add_lines(self.mapped_file_dict(line_data))

        file_tracers = {
            k: v for k, v in self.file_tracers.items()
            if v not in self.disabled_plugins
        }
        self.covdata.add_file_tracers(self.mapped_file_dict(file_tracers))

        self._clear_data()
        return True
