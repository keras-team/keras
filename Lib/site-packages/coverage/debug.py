# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/nedbat/coveragepy/blob/master/NOTICE.txt

"""Control of and utilities for debugging."""

from __future__ import annotations

import atexit
import contextlib
import functools
import inspect
import itertools
import os
import pprint
import re
import reprlib
import sys
import traceback
import types
import _thread

from typing import (
    overload,
    Any, Callable, Final, IO,
)
from collections.abc import Iterable, Iterator, Mapping

from coverage.misc import human_sorted_items, isolate_module
from coverage.types import AnyCallable, TWritable

os = isolate_module(os)


# When debugging, it can be helpful to force some options, especially when
# debugging the configuration mechanisms you usually use to control debugging!
# This is a list of forced debugging options.
FORCED_DEBUG: list[str] = []
FORCED_DEBUG_FILE = None


class DebugControl:
    """Control and output for debugging."""

    show_repr_attr = False      # For auto_repr

    def __init__(
        self,
        options: Iterable[str],
        output: IO[str] | None,
        file_name: str | None = None,
    ) -> None:
        """Configure the options and output file for debugging."""
        self.options = list(options) + FORCED_DEBUG
        self.suppress_callers = False

        filters = []
        if self.should("process"):
            filters.append(CwdTracker().filter)
            filters.append(ProcessTracker().filter)
        if self.should("pytest"):
            filters.append(PytestTracker().filter)
        if self.should("pid"):
            filters.append(add_pid_and_tid)

        self.output = DebugOutputFile.get_one(
            output,
            file_name=file_name,
            filters=filters,
        )
        self.raw_output = self.output.outfile

    def __repr__(self) -> str:
        return f"<DebugControl options={self.options!r} raw_output={self.raw_output!r}>"

    def should(self, option: str) -> bool:
        """Decide whether to output debug information in category `option`."""
        if option == "callers" and self.suppress_callers:
            return False
        return (option in self.options)

    @contextlib.contextmanager
    def without_callers(self) -> Iterator[None]:
        """A context manager to prevent call stacks from being logged."""
        old = self.suppress_callers
        self.suppress_callers = True
        try:
            yield
        finally:
            self.suppress_callers = old

    def write(self, msg: str, *, exc: BaseException | None = None) -> None:
        """Write a line of debug output.

        `msg` is the line to write. A newline will be appended.

        If `exc` is provided, a stack trace of the exception will be written
        after the message.

        """
        self.output.write(msg + "\n")
        if exc is not None:
            self.output.write("".join(traceback.format_exception(None, exc, exc.__traceback__)))
        if self.should("self"):
            caller_self = inspect.stack()[1][0].f_locals.get("self")
            if caller_self is not None:
                self.output.write(f"self: {caller_self!r}\n")
        if self.should("callers"):
            dump_stack_frames(out=self.output, skip=1)
        self.output.flush()


class NoDebugging(DebugControl):
    """A replacement for DebugControl that will never try to do anything."""
    def __init__(self) -> None:
        # pylint: disable=super-init-not-called
        ...

    def should(self, option: str) -> bool:
        """Should we write debug messages?  Never."""
        return False

    @contextlib.contextmanager
    def without_callers(self) -> Iterator[None]:
        """A dummy context manager to satisfy the api."""
        yield

    def write(self, msg: str, *, exc: BaseException | None = None) -> None:
        """This will never be called."""
        raise AssertionError("NoDebugging.write should never be called.")


def info_header(label: str) -> str:
    """Make a nice header string."""
    return "--{:-<60s}".format(" "+label+" ")


def info_formatter(info: Iterable[tuple[str, Any]]) -> Iterator[str]:
    """Produce a sequence of formatted lines from info.

    `info` is a sequence of pairs (label, data).  The produced lines are
    nicely formatted, ready to print.

    """
    info = list(info)
    if not info:
        return
    label_len = 30
    assert all(len(l) < label_len for l, _ in info)
    for label, data in info:
        if data == []:
            data = "-none-"
        if isinstance(data, tuple) and len(repr(tuple(data))) < 30:
            # Convert to tuple to scrub namedtuples.
            yield "%*s: %r" % (label_len, label, tuple(data))
        elif isinstance(data, (list, set, tuple)):
            prefix = "%*s:" % (label_len, label)
            for e in data:
                yield "%*s %s" % (label_len+1, prefix, e)
                prefix = ""
        else:
            yield "%*s: %s" % (label_len, label, data)


def write_formatted_info(
    write: Callable[[str], None],
    header: str,
    info: Iterable[tuple[str, Any]],
) -> None:
    """Write a sequence of (label,data) pairs nicely.

    `write` is a function write(str) that accepts each line of output.
    `header` is a string to start the section.  `info` is a sequence of
    (label, data) pairs, where label is a str, and data can be a single
    value, or a list/set/tuple.

    """
    write(info_header(header))
    for line in info_formatter(info):
        write(f" {line}")


def exc_one_line(exc: Exception) -> str:
    """Get a one-line summary of an exception, including class name and message."""
    lines = traceback.format_exception_only(type(exc), exc)
    return "|".join(l.rstrip() for l in lines)


_FILENAME_REGEXES: list[tuple[str, str]] = [
    (r".*[/\\]pytest-of-.*[/\\]pytest-\d+([/\\]popen-gw\d+)?", "tmp:"),
]
_FILENAME_SUBS: list[tuple[str, str]] = []

@overload
def short_filename(filename: str) -> str:
    pass

@overload
def short_filename(filename: None) -> None:
    pass

def short_filename(filename: str | None) -> str | None:
    """Shorten a file name. Directories are replaced by prefixes like 'syspath:'"""
    if not _FILENAME_SUBS:
        for pathdir in sys.path:
            _FILENAME_SUBS.append((pathdir, "syspath:"))
        import coverage
        _FILENAME_SUBS.append((os.path.dirname(coverage.__file__), "cov:"))
        _FILENAME_SUBS.sort(key=(lambda pair: len(pair[0])), reverse=True)
    if filename is not None:
        for pat, sub in _FILENAME_REGEXES:
            filename = re.sub(pat, sub, filename)
        for before, after in _FILENAME_SUBS:
            filename = filename.replace(before, after)
    return filename


def short_stack(
    skip: int = 0,
    full: bool = False,
    frame_ids: bool = False,
    short_filenames: bool = False,
) -> str:
    """Return a string summarizing the call stack.

    The string is multi-line, with one line per stack frame. Each line shows
    the function name, the file name, and the line number:

        ...
        start_import_stop : /Users/ned/coverage/trunk/tests/coveragetest.py:95
        import_local_file : /Users/ned/coverage/trunk/tests/coveragetest.py:81
        import_local_file : /Users/ned/coverage/trunk/coverage/backward.py:159
        ...

    `skip` is the number of closest immediate frames to skip, so that debugging
    functions can call this and not be included in the result.

    If `full` is true, then include all frames.  Otherwise, initial "boring"
    frames (ones in site-packages and earlier) are omitted.

    `short_filenames` will shorten filenames using `short_filename`, to reduce
    the amount of repetitive noise in stack traces.

    """
    # Regexes in initial frames that we don't care about.
    BORING_PRELUDE = [
        "<string>",             # pytest-xdist has string execution.
        r"\bigor.py$",          # Our test runner.
        r"\bsite-packages\b",   # pytest etc getting to our tests.
    ]

    stack: Iterable[inspect.FrameInfo] = inspect.stack()[:skip:-1]
    if not full:
        for pat in BORING_PRELUDE:
            stack = itertools.dropwhile(
                (lambda fi, pat=pat: re.search(pat, fi.filename)),  # type: ignore[misc]
                stack,
            )
    lines = []
    for frame_info in stack:
        line = f"{frame_info.function:>30s} : "
        if frame_ids:
            line += f"{id(frame_info.frame):#x} "
        filename = frame_info.filename
        if short_filenames:
            filename = short_filename(filename)
        line += f"{filename}:{frame_info.lineno}"
        lines.append(line)
    return "\n".join(lines)


def dump_stack_frames(out: TWritable, skip: int = 0) -> None:
    """Print a summary of the stack to `out`."""
    out.write(short_stack(skip=skip+1) + "\n")


def clipped_repr(text: str, numchars: int = 50) -> str:
    """`repr(text)`, but limited to `numchars`."""
    r = reprlib.Repr()
    r.maxstring = numchars
    return r.repr(text)


def short_id(id64: int) -> int:
    """Given a 64-bit id, make a shorter 16-bit one."""
    id16 = 0
    for offset in range(0, 64, 16):
        id16 ^= id64 >> offset
    return id16 & 0xFFFF


def add_pid_and_tid(text: str) -> str:
    """A filter to add pid and tid to debug messages."""
    # Thread ids are useful, but too long. Make a shorter one.
    tid = f"{short_id(_thread.get_ident()):04x}"
    text = f"{os.getpid():5d}.{tid}: {text}"
    return text


AUTO_REPR_IGNORE = {"$coverage.object_id"}

def auto_repr(self: Any) -> str:
    """A function implementing an automatic __repr__ for debugging."""
    show_attrs = (
        (k, v) for k, v in self.__dict__.items()
        if getattr(v, "show_repr_attr", True)
        and not inspect.ismethod(v)
        and k not in AUTO_REPR_IGNORE
    )
    return "<{klass} @{id:#x}{attrs}>".format(
        klass=self.__class__.__name__,
        id=id(self),
        attrs="".join(f" {k}={v!r}" for k, v in show_attrs),
    )


def simplify(v: Any) -> Any:                                # pragma: debugging
    """Turn things which are nearly dict/list/etc into dict/list/etc."""
    if isinstance(v, dict):
        return {k:simplify(vv) for k, vv in v.items()}
    elif isinstance(v, (list, tuple)):
        return type(v)(simplify(vv) for vv in v)
    elif hasattr(v, "__dict__"):
        return simplify({"."+k: v for k, v in v.__dict__.items()})
    else:
        return v


def pp(v: Any) -> None:                                     # pragma: debugging
    """Debug helper to pretty-print data, including SimpleNamespace objects."""
    # Might not be needed in 3.9+
    pprint.pprint(simplify(v))


def filter_text(text: str, filters: Iterable[Callable[[str], str]]) -> str:
    """Run `text` through a series of filters.

    `filters` is a list of functions. Each takes a string and returns a
    string.  Each is run in turn. After each filter, the text is split into
    lines, and each line is passed through the next filter.

    Returns: the final string that results after all of the filters have
    run.

    """
    clean_text = text.rstrip()
    ending = text[len(clean_text):]
    text = clean_text
    for filter_fn in filters:
        lines = []
        for line in text.splitlines():
            lines.extend(filter_fn(line).splitlines())
        text = "\n".join(lines)
    return text + ending


class CwdTracker:
    """A class to add cwd info to debug messages."""
    def __init__(self) -> None:
        self.cwd: str | None = None

    def filter(self, text: str) -> str:
        """Add a cwd message for each new cwd."""
        cwd = os.getcwd()
        if cwd != self.cwd:
            text = f"cwd is now {cwd!r}\n{text}"
            self.cwd = cwd
        return text


class ProcessTracker:
    """Track process creation for debug logging."""
    def __init__(self) -> None:
        self.pid: int = os.getpid()
        self.did_welcome = False

    def filter(self, text: str) -> str:
        """Add a message about how new processes came to be."""
        welcome = ""
        pid = os.getpid()
        if self.pid != pid:
            welcome = f"New process: forked {self.pid} -> {pid}\n"
            self.pid = pid
        elif not self.did_welcome:
            argv = getattr(sys, "argv", None)
            welcome = (
                f"New process: {pid=}, executable: {sys.executable!r}\n"
                + f"New process: cmd: {argv!r}\n"
                + f"New process parent pid: {os.getppid()!r}\n"
            )

        if welcome:
            self.did_welcome = True
            return welcome + text
        else:
            return text


class PytestTracker:
    """Track the current pytest test name to add to debug messages."""
    def __init__(self) -> None:
        self.test_name: str | None = None

    def filter(self, text: str) -> str:
        """Add a message when the pytest test changes."""
        test_name = os.getenv("PYTEST_CURRENT_TEST")
        if test_name != self.test_name:
            text = f"Pytest context: {test_name}\n{text}"
            self.test_name = test_name
        return text


class DebugOutputFile:
    """A file-like object that includes pid and cwd information."""
    def __init__(
        self,
        outfile: IO[str] | None,
        filters: Iterable[Callable[[str], str]],
    ):
        self.outfile = outfile
        self.filters = list(filters)
        self.pid = os.getpid()

    @classmethod
    def get_one(
        cls,
        fileobj: IO[str] | None = None,
        file_name: str | None = None,
        filters: Iterable[Callable[[str], str]] = (),
        interim: bool = False,
    ) -> DebugOutputFile:
        """Get a DebugOutputFile.

        If `fileobj` is provided, then a new DebugOutputFile is made with it.

        If `fileobj` isn't provided, then a file is chosen (`file_name` if
        provided, or COVERAGE_DEBUG_FILE, or stderr), and a process-wide
        singleton DebugOutputFile is made.

        `filters` are the text filters to apply to the stream to annotate with
        pids, etc.

        If `interim` is true, then a future `get_one` can replace this one.

        """
        if fileobj is not None:
            # Make DebugOutputFile around the fileobj passed.
            return cls(fileobj, filters)

        the_one, is_interim = cls._get_singleton_data()
        if the_one is None or is_interim:
            if file_name is not None:
                fileobj = open(file_name, "a", encoding="utf-8")
            else:
                # $set_env.py: COVERAGE_DEBUG_FILE - Where to write debug output
                file_name = os.getenv("COVERAGE_DEBUG_FILE", FORCED_DEBUG_FILE)
                if file_name in ("stdout", "stderr"):
                    fileobj = getattr(sys, file_name)
                elif file_name:
                    fileobj = open(file_name, "a", encoding="utf-8")
                    atexit.register(fileobj.close)
                else:
                    fileobj = sys.stderr
            the_one = cls(fileobj, filters)
            cls._set_singleton_data(the_one, interim)

        if not(the_one.filters):
            the_one.filters = list(filters)
        return the_one

    # Because of the way igor.py deletes and re-imports modules,
    # this class can be defined more than once. But we really want
    # a process-wide singleton. So stash it in sys.modules instead of
    # on a class attribute. Yes, this is aggressively gross.

    SYS_MOD_NAME: Final[str] = "$coverage.debug.DebugOutputFile.the_one"
    SINGLETON_ATTR: Final[str] = "the_one_and_is_interim"

    @classmethod
    def _set_singleton_data(cls, the_one: DebugOutputFile, interim: bool) -> None:
        """Set the one DebugOutputFile to rule them all."""
        singleton_module = types.ModuleType(cls.SYS_MOD_NAME)
        setattr(singleton_module, cls.SINGLETON_ATTR, (the_one, interim))
        sys.modules[cls.SYS_MOD_NAME] = singleton_module

    @classmethod
    def _get_singleton_data(cls) -> tuple[DebugOutputFile | None, bool]:
        """Get the one DebugOutputFile."""
        singleton_module = sys.modules.get(cls.SYS_MOD_NAME)
        return getattr(singleton_module, cls.SINGLETON_ATTR, (None, True))

    @classmethod
    def _del_singleton_data(cls) -> None:
        """Delete the one DebugOutputFile, just for tests to use."""
        if cls.SYS_MOD_NAME in sys.modules:
            del sys.modules[cls.SYS_MOD_NAME]

    def write(self, text: str) -> None:
        """Just like file.write, but filter through all our filters."""
        assert self.outfile is not None
        self.outfile.write(filter_text(text, self.filters))
        self.outfile.flush()

    def flush(self) -> None:
        """Flush our file."""
        assert self.outfile is not None
        self.outfile.flush()


def log(msg: str, stack: bool = False) -> None:             # pragma: debugging
    """Write a log message as forcefully as possible."""
    out = DebugOutputFile.get_one(interim=True)
    out.write(msg+"\n")
    if stack:
        dump_stack_frames(out=out, skip=1)


def decorate_methods(
    decorator: Callable[..., Any],
    butnot: Iterable[str] = (),
    private: bool = False,
) -> Callable[..., Any]:                                    # pragma: debugging
    """A class decorator to apply a decorator to methods."""
    def _decorator(cls):                                    # type: ignore[no-untyped-def]
        for name, meth in inspect.getmembers(cls, inspect.isroutine):
            if name not in cls.__dict__:
                continue
            if name != "__init__":
                if not private and name.startswith("_"):
                    continue
            if name in butnot:
                continue
            setattr(cls, name, decorator(meth))
        return cls
    return _decorator


def break_in_pudb(func: AnyCallable) -> AnyCallable:  # pragma: debugging
    """A function decorator to stop in the debugger for each call."""
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        import pudb
        sys.stdout = sys.__stdout__
        pudb.set_trace()
        return func(*args, **kwargs)
    return _wrapper


OBJ_IDS = itertools.count()
CALLS = itertools.count()
OBJ_ID_ATTR = "$coverage.object_id"

def show_calls(
    show_args: bool = True,
    show_stack: bool = False,
    show_return: bool = False,
) -> Callable[..., Any]:                                    # pragma: debugging
    """A method decorator to debug-log each call to the function."""
    def _decorator(func: AnyCallable) -> AnyCallable:
        @functools.wraps(func)
        def _wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            oid = getattr(self, OBJ_ID_ATTR, None)
            if oid is None:
                oid = f"{os.getpid():08d} {next(OBJ_IDS):04d}"
                setattr(self, OBJ_ID_ATTR, oid)
            extra = ""
            if show_args:
                eargs = ", ".join(map(repr, args))
                ekwargs = ", ".join("{}={!r}".format(*item) for item in kwargs.items())
                extra += "("
                extra += eargs
                if eargs and ekwargs:
                    extra += ", "
                extra += ekwargs
                extra += ")"
            if show_stack:
                extra += " @ "
                extra += "; ".join(short_stack(short_filenames=True).splitlines())
            callid = next(CALLS)
            msg = f"{oid} {callid:04d} {func.__name__}{extra}\n"
            DebugOutputFile.get_one(interim=True).write(msg)
            ret = func(self, *args, **kwargs)
            if show_return:
                msg = f"{oid} {callid:04d} {func.__name__} return {ret!r}\n"
                DebugOutputFile.get_one(interim=True).write(msg)
            return ret
        return _wrapper
    return _decorator


def relevant_environment_display(env: Mapping[str, str]) -> list[tuple[str, str]]:
    """Filter environment variables for a debug display.

    Select variables to display (with COV or PY in the name, or HOME, TEMP, or
    TMP), and also cloak sensitive values with asterisks.

    Arguments:
        env: a dict of environment variable names and values.

    Returns:
        A list of pairs (name, value) to show.

    """
    slugs = {"COV", "PY"}
    include = {"HOME", "TEMP", "TMP"}
    cloak = {"API", "TOKEN", "KEY", "SECRET", "PASS", "SIGNATURE"}

    to_show = []
    for name, val in env.items():
        keep = False
        if name in include:
            keep = True
        elif any(slug in name for slug in slugs):
            keep = True
        if keep:
            if any(slug in name for slug in cloak):
                val = re.sub(r"\w", "*", val)
            to_show.append((name, val))
    return human_sorted_items(to_show)
