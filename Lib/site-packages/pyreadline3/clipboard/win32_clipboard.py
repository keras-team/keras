# -*- coding: utf-8 -*-
# *****************************************************************************
#     Copyright (C) 2003-2006 Jack Trainor.
#     Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#     Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************
###################################
#
# Based on recipe posted to ctypes-users
# see archive
# http://aspn.activestate.com/ASPN/Mail/Message/ctypes-users/1771866
#
#

##########################################################################
#
# The Python win32clipboard lib functions work well enough ... except that they
# can only cut and paste items from within one application, not across
# applications or processes.
#
# I've written a number of Python text filters I like to run on the contents of
# the clipboard so I need to call the Windows clipboard API with global memory
# for my filters to work properly.
#
# Here's some sample code solving this problem using ctypes.
#
# This is my first work with ctypes.  It's powerful stuff, but passing
# arguments in and out of functions is tricky.  More sample code would have
# been helpful, hence this contribution.
#
##########################################################################

import ctypes
import ctypes.wintypes as wintypes
from ctypes import (
    addressof,
    c_buffer,
    c_char_p,
    c_int,
    c_size_t,
    c_void_p,
    c_wchar_p,
    cast,
    create_unicode_buffer,
    sizeof,
    windll,
    wstring_at,
)
from typing import Union

from pyreadline3.keysyms.winconstants import CF_UNICODETEXT, GHND
from pyreadline3.unicode_helper import ensure_unicode

OpenClipboard = windll.user32.OpenClipboard
OpenClipboard.argtypes = [wintypes.HWND]
OpenClipboard.restype = wintypes.BOOL

EmptyClipboard = windll.user32.EmptyClipboard

GetClipboardData = windll.user32.GetClipboardData
GetClipboardData.argtypes = [wintypes.UINT]
GetClipboardData.restype = wintypes.HANDLE

GetClipboardFormatName = windll.user32.GetClipboardFormatNameA
GetClipboardFormatName.argtypes = [wintypes.UINT, c_char_p, c_int]

SetClipboardData = windll.user32.SetClipboardData
SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
SetClipboardData.restype = wintypes.HANDLE

EnumClipboardFormats = windll.user32.EnumClipboardFormats
EnumClipboardFormats.argtypes = [c_int]

CloseClipboard = windll.user32.CloseClipboard
CloseClipboard.argtypes = []


GlobalAlloc = windll.kernel32.GlobalAlloc
GlobalAlloc.argtypes = [wintypes.UINT, c_size_t]
GlobalAlloc.restype = wintypes.HGLOBAL
GlobalLock = windll.kernel32.GlobalLock
GlobalLock.argtypes = [wintypes.HGLOBAL]
GlobalLock.restype = c_void_p
GlobalUnlock = windll.kernel32.GlobalUnlock
GlobalUnlock.argtypes = [c_int]

_strncpy = ctypes.windll.kernel32.lstrcpynW
_strncpy.restype = c_wchar_p
_strncpy.argtypes = [c_wchar_p, c_wchar_p, c_size_t]


def _enum() -> None:
    OpenClipboard(0)

    q = EnumClipboardFormats(0)
    while q:
        q = EnumClipboardFormats(q)

    CloseClipboard()


def _get_format_name(format_str: str) -> bytes:
    buffer = c_buffer(100)
    bufferSize = sizeof(buffer)

    OpenClipboard(0)

    GetClipboardFormatName(format_str, buffer, bufferSize)

    CloseClipboard()

    return buffer.value


def get_clipboard_text() -> str:
    text = ""

    if OpenClipboard(0):
        h_clip_mem = GetClipboardData(CF_UNICODETEXT)

        if h_clip_mem:
            text = wstring_at(GlobalLock(h_clip_mem))
            GlobalUnlock(h_clip_mem)

        CloseClipboard()

    return text


def set_clipboard_text(text: Union[str, bytes]) -> None:
    buffer = create_unicode_buffer(ensure_unicode(text))
    buffer_size = sizeof(buffer)

    h_global_mem = GlobalAlloc(GHND, c_size_t(buffer_size))
    GlobalLock.restype = c_void_p
    lp_global_mem = GlobalLock(h_global_mem)

    _strncpy(
        cast(lp_global_mem, c_wchar_p),
        cast(addressof(buffer), c_wchar_p),
        c_size_t(buffer_size),
    )

    GlobalUnlock(c_int(h_global_mem))

    if OpenClipboard(0):
        EmptyClipboard()
        SetClipboardData(CF_UNICODETEXT, h_global_mem)
        CloseClipboard()


if __name__ == "__main__":
    txt = get_clipboard_text()  # display last text clipped
    print(txt)
