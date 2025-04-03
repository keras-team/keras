# -*- coding: utf-8 -*-
# *****************************************************************************
#     Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#     Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************

from socket import AF_INET, SOCK_DGRAM, socket

from pyreadline3.unicode_helper import ensure_str


class SocketStream:
    def __init__(
        self,
        host: str,
        port: int,
    ) -> None:
        self.__host = host
        self.__port = port
        self.__socket = socket(AF_INET, SOCK_DGRAM)

    def write(self, record: str) -> None:
        self.__socket.sendto(
            ensure_str(record),
            (self.__host, self.__port),
        )

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass
