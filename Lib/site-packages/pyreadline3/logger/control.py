# -*- coding: utf-8 -*-
# *****************************************************************************
#     Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#     Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************

import os
from logging import FileHandler, Formatter, StreamHandler
from logging.handlers import DEFAULT_TCP_LOGGING_PORT
from typing import Optional

from .logger import LOGGER
from .socket_stream import SocketStream

_default_formatter_str = os.environ.get("PYREADLINE_FORMATTER", "%(message)s")

SOCKET_HANDLER: Optional["StreamHandler[SocketStream]"] = None
FILE_HANDLER: Optional[FileHandler] = None


def start_socket_log(
    host: str = "localhost",
    port: int = DEFAULT_TCP_LOGGING_PORT,
    formatter_str: str = _default_formatter_str,
) -> None:
    global SOCKET_HANDLER

    if SOCKET_HANDLER is not None:
        return

    SOCKET_HANDLER = StreamHandler(SocketStream(host, port))
    SOCKET_HANDLER.setFormatter(Formatter(formatter_str))

    LOGGER.addHandler(SOCKET_HANDLER)


def stop_socket_log() -> None:
    global SOCKET_HANDLER

    if SOCKET_HANDLER is None:
        return

    LOGGER.removeHandler(SOCKET_HANDLER)

    SOCKET_HANDLER = None


def start_file_log(filename: str) -> None:
    global FILE_HANDLER

    if FILE_HANDLER is not None:
        return

    FILE_HANDLER = FileHandler(filename, "w")
    LOGGER.addHandler(FILE_HANDLER)


def stop_file_log() -> None:
    global FILE_HANDLER

    if FILE_HANDLER is None:
        return

    LOGGER.removeHandler(FILE_HANDLER)

    FILE_HANDLER.close()
    FILE_HANDLER = None


def stop_logging() -> None:
    stop_file_log()
    stop_socket_log()
