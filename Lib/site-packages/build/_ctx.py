from __future__ import annotations

import contextvars
import logging
import subprocess
import typing

from collections.abc import Mapping, Sequence
from functools import partial

from ._types import StrPath


class _Logger(typing.Protocol):  # pragma: no cover
    def __call__(self, message: str, *, origin: tuple[str, ...] | None = None) -> None: ...


_package_name = __spec__.parent  # type: ignore[name-defined]
_default_logger = logging.getLogger(_package_name)


def _log_default(message: str, *, origin: tuple[str, ...] | None = None) -> None:
    if origin is None:
        _default_logger.log(logging.INFO, message, stacklevel=2)


LOGGER = contextvars.ContextVar('LOGGER', default=_log_default)
VERBOSITY = contextvars.ContextVar('VERBOSITY', default=0)


def log_subprocess_error(error: subprocess.CalledProcessError) -> None:
    log = LOGGER.get()

    log(subprocess.list2cmdline(error.cmd), origin=('subprocess', 'cmd'))

    for stream_name in ('stdout', 'stderr'):
        stream = getattr(error, stream_name)
        if stream:
            log(stream.decode() if isinstance(stream, bytes) else stream, origin=('subprocess', stream_name))


def run_subprocess(cmd: Sequence[StrPath], env: Mapping[str, str] | None = None) -> None:
    verbosity = VERBOSITY.get()

    if verbosity:
        import concurrent.futures

        log = LOGGER.get()

        def log_stream(stream_name: str, stream: typing.IO[str]) -> None:
            for line in stream:
                log(line, origin=('subprocess', stream_name))

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor, subprocess.Popen(
            cmd, encoding='utf-8', env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ) as process:
            log(subprocess.list2cmdline(cmd), origin=('subprocess', 'cmd'))

            # Logging in sub-thread to more-or-less ensure order of stdout and stderr whilst also
            # being able to distinguish between the two.
            concurrent.futures.wait(
                [executor.submit(partial(log_stream, n, getattr(process, n))) for n in ('stdout', 'stderr')]
            )

            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, process.args)

    else:
        try:
            subprocess.run(cmd, capture_output=True, check=True, env=env)
        except subprocess.CalledProcessError as error:
            log_subprocess_error(error)
            raise


if typing.TYPE_CHECKING:
    log: _Logger
    verbosity: bool

else:

    def __getattr__(name):
        if name == 'log':
            return LOGGER.get()
        elif name == 'verbosity':
            return VERBOSITY.get()
        raise AttributeError(name)  # pragma: no cover


__all__ = [
    'log_subprocess_error',
    'log',
    'run_subprocess',
    'LOGGER',
    'verbosity',
    'VERBOSITY',
]
