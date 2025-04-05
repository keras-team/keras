# -*- coding: utf-8 -*-
# *****************************************************************************
#     Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#     Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************

GLOBAL_CLIPBOARD_BUFFER = ""


def get_clipboard_text() -> str:
    return GLOBAL_CLIPBOARD_BUFFER


def set_clipboard_text(text: str) -> None:
    global GLOBAL_CLIPBOARD_BUFFER
    GLOBAL_CLIPBOARD_BUFFER = text
