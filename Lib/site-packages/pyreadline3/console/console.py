# -*- coding: utf-8 -*-
# *****************************************************************************
#       Copyright (C) 2003-2006 Gary Bishop.
#       Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#       Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************


from .event import Event

"""Cursor control and color for the Windows console.

This was modeled after the C extension of the same name by Fredrik Lundh.
"""

# primitive debug printing that won't interfere with the screen

import os
import re
import sys
import traceback

import pyreadline3.unicode_helper as unicode_helper
from pyreadline3.console.ansi import AnsiState, AnsiWriter
from pyreadline3.keysyms import KeyPress, make_KeyPress
from pyreadline3.logger import log
from pyreadline3.unicode_helper import ensure_str, ensure_unicode

try:
    import ctypes.util
    from ctypes import *
    from ctypes.wintypes import *

    from _ctypes import call_function
except ImportError:
    raise ImportError("You need ctypes to run this code")


def nolog(string):
    pass


log = nolog


# some constants we need
STD_INPUT_HANDLE = -10
STD_OUTPUT_HANDLE = -11
ENABLE_WINDOW_INPUT = 0x0008
ENABLE_MOUSE_INPUT = 0x0010
ENABLE_PROCESSED_INPUT = 0x0001
WHITE = 0x7
BLACK = 0
MENU_EVENT = 0x0008
KEY_EVENT = 0x0001
MOUSE_MOVED = 0x0001
MOUSE_EVENT = 0x0002
WINDOW_BUFFER_SIZE_EVENT = 0x0004
FOCUS_EVENT = 0x0010
MENU_EVENT = 0x0008
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12
GENERIC_READ = int(0x80000000)
GENERIC_WRITE = 0x40000000

# Windows structures we'll need later


class COORD(Structure):
    _fields_ = [("X", c_short), ("Y", c_short)]


class SMALL_RECT(Structure):
    _fields_ = [
        ("Left", c_short),
        ("Top", c_short),
        ("Right", c_short),
        ("Bottom", c_short),
    ]


class CONSOLE_SCREEN_BUFFER_INFO(Structure):
    _fields_ = [
        ("dwSize", COORD),
        ("dwCursorPosition", COORD),
        ("wAttributes", c_short),
        ("srWindow", SMALL_RECT),
        ("dwMaximumWindowSize", COORD),
    ]


class CHAR_UNION(Union):
    _fields_ = [("UnicodeChar", c_wchar), ("AsciiChar", c_char)]


class CHAR_INFO(Structure):
    _fields_ = [("Char", CHAR_UNION), ("Attributes", c_short)]


class KEY_EVENT_RECORD(Structure):
    _fields_ = [
        ("bKeyDown", c_byte),
        ("pad2", c_byte),
        ("pad1", c_short),
        ("wRepeatCount", c_short),
        ("wVirtualKeyCode", c_short),
        ("wVirtualScanCode", c_short),
        ("uChar", CHAR_UNION),
        ("dwControlKeyState", c_int),
    ]


class MOUSE_EVENT_RECORD(Structure):
    _fields_ = [
        ("dwMousePosition", COORD),
        ("dwButtonState", c_int),
        ("dwControlKeyState", c_int),
        ("dwEventFlags", c_int),
    ]


class WINDOW_BUFFER_SIZE_RECORD(Structure):
    _fields_ = [("dwSize", COORD)]


class MENU_EVENT_RECORD(Structure):
    _fields_ = [("dwCommandId", c_uint)]


class FOCUS_EVENT_RECORD(Structure):
    _fields_ = [("bSetFocus", c_byte)]


class INPUT_UNION(Union):
    _fields_ = [
        ("KeyEvent", KEY_EVENT_RECORD),
        ("MouseEvent", MOUSE_EVENT_RECORD),
        ("WindowBufferSizeEvent", WINDOW_BUFFER_SIZE_RECORD),
        ("MenuEvent", MENU_EVENT_RECORD),
        ("FocusEvent", FOCUS_EVENT_RECORD),
    ]


class INPUT_RECORD(Structure):
    _fields_ = [("EventType", c_short), ("Event", INPUT_UNION)]


class CONSOLE_CURSOR_INFO(Structure):
    _fields_ = [("dwSize", c_int), ("bVisible", c_byte)]


# I didn't want to have to individually import these so I made a list, they are
# added to the Console class later in this file.

funcs = [
    "AllocConsole",
    "CreateConsoleScreenBuffer",
    "FillConsoleOutputAttribute",
    "FillConsoleOutputCharacterW",
    "FreeConsole",
    "GetConsoleCursorInfo",
    "GetConsoleMode",
    "GetConsoleScreenBufferInfo",
    "GetConsoleTitleW",
    "GetProcAddress",
    "GetStdHandle",
    "PeekConsoleInputW",
    "ReadConsoleInputW",
    "ScrollConsoleScreenBufferW",
    "SetConsoleActiveScreenBuffer",
    "SetConsoleCursorInfo",
    "SetConsoleCursorPosition",
    "SetConsoleMode",
    "SetConsoleScreenBufferSize",
    "SetConsoleTextAttribute",
    "SetConsoleTitleW",
    "SetConsoleWindowInfo",
    "WriteConsoleW",
    "WriteConsoleOutputCharacterW",
    "WriteFile",
]

# I don't want events for these keys, they are just a bother for my application
key_modifiers = {
    VK_SHIFT: 1,
    VK_CONTROL: 1,
    VK_MENU: 1,  # alt key
    0x5B: 1,  # windows key
}


def split_block(text, size=1000):
    return [text[start : start + size] for start in range(0, len(text), size)]


class Console(object):
    """Console driver for Windows."""

    def __init__(self, newbuffer=0):
        """Initialize the Console object.

        newbuffer=1 will allocate a new buffer so the old content will be restored
        on exit.
        """
        # Do I need the following line? It causes a console to be created whenever
        # readline is imported into a pythonw application which seems wrong. Things
        # seem to work without it...
        # self.AllocConsole()

        if newbuffer:
            self.hout = self.CreateConsoleScreenBuffer(
                GENERIC_READ | GENERIC_WRITE, 0, None, 1, None
            )
            self.SetConsoleActiveScreenBuffer(self.hout)
        else:
            self.hout = self.GetStdHandle(STD_OUTPUT_HANDLE)

        self.hin = self.GetStdHandle(STD_INPUT_HANDLE)
        self.inmode = DWORD(0)
        self.GetConsoleMode(self.hin, byref(self.inmode))
        self.SetConsoleMode(self.hin, 0xF)
        info = CONSOLE_SCREEN_BUFFER_INFO()
        self.GetConsoleScreenBufferInfo(self.hout, byref(info))
        self.attr = info.wAttributes
        self.saveattr = info.wAttributes  # remember the initial colors
        self.defaultstate = AnsiState()
        self.defaultstate.winattr = info.wAttributes
        self.ansiwriter = AnsiWriter(self.defaultstate)

        background = self.attr & 0xF0
        for escape in self.escape_to_color:
            if self.escape_to_color[escape] is not None:
                self.escape_to_color[escape] |= background
        log("initial attr=%x" % self.attr)
        self.softspace = 0  # this is for using it as a file-like object
        self.serial = 0

        self.pythondll = ctypes.pythonapi
        self.inputHookPtr = c_void_p.from_address(
            addressof(self.pythondll.PyOS_InputHook)
        ).value

        self.pythondll.PyMem_RawMalloc.restype = c_size_t
        self.pythondll.PyMem_RawMalloc.argtypes = [c_size_t]
        setattr(Console, "PyMem_Malloc", self.pythondll.PyMem_RawMalloc)

    def __del__(self):
        """Cleanup the console when finished."""
        # I don't think this ever gets called
        self.SetConsoleTextAttribute(self.hout, self.saveattr)
        self.SetConsoleMode(self.hin, self.inmode)
        self.FreeConsole()

    def _get_top_bot(self):
        info = CONSOLE_SCREEN_BUFFER_INFO()
        self.GetConsoleScreenBufferInfo(self.hout, byref(info))
        rect = info.srWindow
        top = rect.Top
        bot = rect.Bottom
        return top, bot

    def fixcoord(self, x, y):
        """Return a long with x and y packed inside,
        also handle negative x and y."""
        if x < 0 or y < 0:
            info = CONSOLE_SCREEN_BUFFER_INFO()
            self.GetConsoleScreenBufferInfo(self.hout, byref(info))
            if x < 0:
                x = info.srWindow.Right - x
                y = info.srWindow.Bottom + y

        # this is a hack! ctypes won't pass structures but COORD is
        # just like a long, so this works.
        return c_int(y << 16 | x)

    def pos(self, x=None, y=None):
        """Move or query the window cursor."""
        if x is None:
            info = CONSOLE_SCREEN_BUFFER_INFO()
            self.GetConsoleScreenBufferInfo(self.hout, byref(info))
            return (info.dwCursorPosition.X, info.dwCursorPosition.Y)
        else:
            return self.SetConsoleCursorPosition(self.hout, self.fixcoord(x, y))

    def home(self):
        """Move to home."""
        self.pos(0, 0)

    # Map ANSI color escape sequences into Windows Console Attributes

    terminal_escape = re.compile("(\001?\033\\[[0-9;]+m\002?)")
    escape_parts = re.compile("\001?\033\\[([0-9;]+)m\002?")
    escape_to_color = {
        "0;30": 0x0,  # black
        "0;31": 0x4,  # red
        "0;32": 0x2,  # green
        "0;33": 0x4 + 0x2,  # brown?
        "0;34": 0x1,  # blue
        "0;35": 0x1 + 0x4,  # purple
        "0;36": 0x2 + 0x4,  # cyan
        "0;37": 0x1 + 0x2 + 0x4,  # grey
        "1;30": 0x1 + 0x2 + 0x4,  # dark gray
        "1;31": 0x4 + 0x8,  # red
        "1;32": 0x2 + 0x8,  # light green
        "1;33": 0x4 + 0x2 + 0x8,  # yellow
        "1;34": 0x1 + 0x8,  # light blue
        "1;35": 0x1 + 0x4 + 0x8,  # light purple
        "1;36": 0x1 + 0x2 + 0x8,  # light cyan
        "1;37": 0x1 + 0x2 + 0x4 + 0x8,  # white
        "0": None,
    }

    # This pattern should match all characters that change the cursor position differently
    # than a normal character.
    motion_char_re = re.compile("([\n\r\t\010\007])")

    def write_scrolling(self, text, attr=None):
        """write text at current cursor position while watching for scrolling.

        If the window scrolls because you are at the bottom of the screen
        buffer, all positions that you are storing will be shifted by the
        scroll amount. For example, I remember the cursor position of the
        prompt so that I can redraw the line but if the window scrolls,
        the remembered position is off.

        This variant of write tries to keep track of the cursor position
        so that it will know when the screen buffer is scrolled. It
        returns the number of lines that the buffer scrolled.

        """
        text = ensure_unicode(text)
        x, y = self.pos()
        w, h = self.size()
        scroll = 0  # the result
        # split the string into ordinary characters and funny characters
        chunks = self.motion_char_re.split(text)
        for chunk in chunks:
            n = self.write_color(chunk, attr)
            if len(chunk) == 1:  # the funny characters will be alone
                if chunk[0] == "\n":  # newline
                    x = 0
                    y += 1
                elif chunk[0] == "\r":  # carriage return
                    x = 0
                elif chunk[0] == "\t":  # tab
                    x = 8 * (int(x / 8) + 1)
                    if x > w:  # newline
                        x -= w
                        y += 1
                elif chunk[0] == "\007":  # bell
                    pass
                elif chunk[0] == "\010":
                    x -= 1
                    if x < 0:
                        y -= 1  # backed up 1 line
                else:  # ordinary character
                    x += 1
                if x == w:  # wrap
                    x = 0
                    y += 1
                if y == h:  # scroll
                    scroll += 1
                    y = h - 1
            else:  # chunk of ordinary characters
                x += n
                l = int(x / w)  # lines we advanced
                x = x % w  # new x value
                y += l
                if y >= h:  # scroll
                    scroll += y - h + 1
                    y = h - 1
        return scroll

    def write_color(self, text, attr=None):
        text = ensure_unicode(text)
        n, res = self.ansiwriter.write_color(text, attr)
        junk = DWORD(0)
        for attr, chunk in res:
            log("console.attr:%s" % (attr))
            log("console.chunk:%s" % (chunk))
            self.SetConsoleTextAttribute(self.hout, attr.winattr)
            for short_chunk in split_block(chunk):
                self.WriteConsoleW(
                    self.hout, short_chunk, len(short_chunk), byref(junk), None
                )
        return n

    def write_plain(self, text, attr=None):
        """write text at current cursor position."""
        text = ensure_unicode(text)
        log('write("%s", %s)' % (text, attr))
        if attr is None:
            attr = self.attr
        junk = DWORD(0)
        self.SetConsoleTextAttribute(self.hout, attr)
        for short_chunk in split_block(chunk):
            self.WriteConsoleW(
                self.hout,
                ensure_unicode(short_chunk),
                len(short_chunk),
                byref(junk),
                None,
            )
        return len(text)

    # This function must be used to ensure functioning with EMACS
    # Emacs sets the EMACS environment variable
    if "EMACS" in os.environ:

        def write_color(self, text, attr=None):
            text = ensure_str(text)
            junk = DWORD(0)
            self.WriteFile(self.hout, text, len(text), byref(junk), None)
            return len(text)

        write_plain = write_color

    # make this class look like a file object
    def write(self, text):
        text = ensure_unicode(text)
        log('write("%s")' % text)
        return self.write_color(text)

    # write = write_scrolling

    def isatty(self):
        return True

    def flush(self):
        pass

    def page(self, attr=None, fill=" "):
        """Fill the entire screen."""
        if attr is None:
            attr = self.attr
        if len(fill) != 1:
            raise ValueError
        info = CONSOLE_SCREEN_BUFFER_INFO()
        self.GetConsoleScreenBufferInfo(self.hout, byref(info))
        if info.dwCursorPosition.X != 0 or info.dwCursorPosition.Y != 0:
            self.SetConsoleCursorPosition(self.hout, self.fixcoord(0, 0))

        w = info.dwSize.X
        n = DWORD(0)
        for y in range(info.dwSize.Y):
            self.FillConsoleOutputAttribute(
                self.hout, attr, w, self.fixcoord(0, y), byref(n)
            )
            self.FillConsoleOutputCharacterW(
                self.hout, ord(fill[0]), w, self.fixcoord(0, y), byref(n)
            )

        self.attr = attr

    def text(self, x, y, text, attr=None):
        """Write text at the given position."""
        if attr is None:
            attr = self.attr

        pos = self.fixcoord(x, y)
        n = DWORD(0)
        self.WriteConsoleOutputCharacterW(self.hout, text, len(text), pos, byref(n))
        self.FillConsoleOutputAttribute(self.hout, attr, n, pos, byref(n))

    def clear_to_end_of_window(self):
        top, bot = self._get_top_bot()
        pos = self.pos()
        w, h = self.size()
        self.rectangle((pos[0], pos[1], w, pos[1] + 1))
        if pos[1] < bot:
            self.rectangle((0, pos[1] + 1, w, bot + 1))

    def rectangle(self, rect, attr=None, fill=" "):
        """Fill Rectangle."""
        x0, y0, x1, y1 = rect
        n = DWORD(0)
        if attr is None:
            attr = self.attr
        for y in range(y0, y1):
            pos = self.fixcoord(x0, y)
            self.FillConsoleOutputAttribute(self.hout, attr, x1 - x0, pos, byref(n))
            self.FillConsoleOutputCharacterW(
                self.hout, ord(fill[0]), x1 - x0, pos, byref(n)
            )

    def scroll(self, rect, dx, dy, attr=None, fill=" "):
        """Scroll a rectangle."""
        if attr is None:
            attr = self.attr
        x0, y0, x1, y1 = rect
        source = SMALL_RECT(x0, y0, x1 - 1, y1 - 1)
        dest = self.fixcoord(x0 + dx, y0 + dy)
        style = CHAR_INFO()
        style.Char.AsciiChar = ensure_str(fill[0])
        style.Attributes = attr

        return self.ScrollConsoleScreenBufferW(
            self.hout, byref(source), byref(source), dest, byref(style)
        )

    def scroll_window(self, lines):
        """Scroll the window by the indicated number of lines."""
        info = CONSOLE_SCREEN_BUFFER_INFO()
        self.GetConsoleScreenBufferInfo(self.hout, byref(info))
        rect = info.srWindow
        log("sw: rtop=%d rbot=%d" % (rect.Top, rect.Bottom))
        top = rect.Top + lines
        bot = rect.Bottom + lines
        h = bot - top
        maxbot = info.dwSize.Y - 1
        if top < 0:
            top = 0
            bot = h
        if bot > maxbot:
            bot = maxbot
            top = bot - h

        nrect = SMALL_RECT()
        nrect.Top = top
        nrect.Bottom = bot
        nrect.Left = rect.Left
        nrect.Right = rect.Right
        log("sn: top=%d bot=%d" % (top, bot))
        r = self.SetConsoleWindowInfo(self.hout, True, byref(nrect))
        log("r=%d" % r)

    def get(self):
        """Get next event from queue."""
        inputHookFunc = c_void_p.from_address(self.inputHookPtr).value

        Cevent = INPUT_RECORD()
        count = DWORD(0)
        while True:
            if inputHookFunc:
                call_function(inputHookFunc, ())
            status = self.ReadConsoleInputW(self.hin, byref(Cevent), 1, byref(count))
            if status and count.value == 1:
                e = event(self, Cevent)
                return e

    def getkeypress(self):
        """Return next key press event from the queue, ignoring others."""
        while True:
            e = self.get()
            if e.type == "KeyPress" and e.keycode not in key_modifiers:
                log("console.getkeypress %s" % e)
                if e.keyinfo.keyname == "next":
                    self.scroll_window(12)
                elif e.keyinfo.keyname == "prior":
                    self.scroll_window(-12)
                else:
                    return e
            elif (e.type == "KeyRelease") and (
                e.keyinfo == KeyPress("S", False, True, False, "S")
                or e.keyinfo == KeyPress("C", False, True, False, "C")
            ):
                log("getKeypress:%s,%s,%s" % (e.keyinfo, e.keycode, e.type))
                return e

    def getchar(self):
        """Get next character from queue."""

        Cevent = INPUT_RECORD()
        count = DWORD(0)
        while True:
            status = self.ReadConsoleInputW(self.hin, byref(Cevent), 1, byref(count))
            if (
                status
                and (count.value == 1)
                and (Cevent.EventType == 1)
                and Cevent.Event.KeyEvent.bKeyDown
            ):
                sym = keysym(Cevent.Event.KeyEvent.wVirtualKeyCode)
                if len(sym) == 0:
                    sym = Cevent.Event.KeyEvent.uChar.AsciiChar
                return sym

    def peek(self):
        """Check event queue."""
        Cevent = INPUT_RECORD()
        count = DWORD(0)
        status = self.PeekConsoleInputW(self.hin, byref(Cevent), 1, byref(count))
        if status and count == 1:
            return event(self, Cevent)

    def title(self, txt=None):
        """Set/get title."""
        if txt:
            self.SetConsoleTitleW(txt)
        else:
            buffer = create_unicode_buffer(200)
            n = self.GetConsoleTitleW(buffer, 200)
            if n > 0:
                return buffer.value[:n]

    def size(self, width=None, height=None):
        """Set/get window size."""
        info = CONSOLE_SCREEN_BUFFER_INFO()
        status = self.GetConsoleScreenBufferInfo(self.hout, byref(info))
        if not status:
            return None
        if width is not None and height is not None:
            wmin = info.srWindow.Right - info.srWindow.Left + 1
            hmin = info.srWindow.Bottom - info.srWindow.Top + 1
            # print wmin, hmin
            width = max(width, wmin)
            height = max(height, hmin)
            # print width, height
            self.SetConsoleScreenBufferSize(self.hout, self.fixcoord(width, height))
        else:
            return (info.dwSize.X, info.dwSize.Y)

    def cursor(self, visible=None, size=None):
        """Set cursor on or off."""
        info = CONSOLE_CURSOR_INFO()
        if self.GetConsoleCursorInfo(self.hout, byref(info)):
            if visible is not None:
                info.bVisible = visible
            if size is not None:
                info.dwSize = size
            self.SetConsoleCursorInfo(self.hout, byref(info))

    def bell(self):
        self.write("\007")

    def next_serial(self):
        """Get next event serial number."""
        self.serial += 1
        return self.serial


# add the functions from the dll to the class
for func in funcs:
    setattr(Console, func, getattr(windll.kernel32, func))


_strncpy = ctypes.windll.kernel32.lstrcpynA
_strncpy.restype = c_char_p
_strncpy.argtypes = [c_char_p, c_char_p, c_size_t]


LPVOID = c_void_p
LPCVOID = c_void_p
FARPROC = c_void_p
LPDWORD = POINTER(DWORD)

Console.AllocConsole.restype = BOOL
Console.AllocConsole.argtypes = []  # void
Console.CreateConsoleScreenBuffer.restype = HANDLE
Console.CreateConsoleScreenBuffer.argtypes = [
    DWORD,
    DWORD,
    c_void_p,
    DWORD,
    LPVOID,
]  # DWORD, DWORD, SECURITY_ATTRIBUTES*, DWORD, LPVOID
Console.FillConsoleOutputAttribute.restype = BOOL
Console.FillConsoleOutputAttribute.argtypes = [
    HANDLE,
    WORD,
    DWORD,
    c_int,
    LPDWORD,
]  # HANDLE, WORD, DWORD, COORD, LPDWORD
Console.FillConsoleOutputCharacterW.restype = BOOL
Console.FillConsoleOutputCharacterW.argtypes = [
    HANDLE,
    c_ushort,
    DWORD,
    c_int,
    LPDWORD,
]  # HANDLE, TCHAR, DWORD, COORD, LPDWORD
Console.FreeConsole.restype = BOOL
Console.FreeConsole.argtypes = []  # void
Console.GetConsoleCursorInfo.restype = BOOL
Console.GetConsoleCursorInfo.argtypes = [
    HANDLE,
    c_void_p,
]  # HANDLE, PCONSOLE_CURSOR_INFO
Console.GetConsoleMode.restype = BOOL
Console.GetConsoleMode.argtypes = [HANDLE, LPDWORD]  # HANDLE, LPDWORD
Console.GetConsoleScreenBufferInfo.restype = BOOL
Console.GetConsoleScreenBufferInfo.argtypes = [
    HANDLE,
    c_void_p,
]  # HANDLE, PCONSOLE_SCREEN_BUFFER_INFO
Console.GetConsoleTitleW.restype = DWORD
Console.GetConsoleTitleW.argtypes = [c_wchar_p, DWORD]  # LPTSTR , DWORD
Console.GetProcAddress.restype = FARPROC
Console.GetProcAddress.argtypes = [HMODULE, c_char_p]  # HMODULE , LPCSTR
Console.GetStdHandle.restype = HANDLE
Console.GetStdHandle.argtypes = [DWORD]
Console.PeekConsoleInputW.restype = BOOL
# HANDLE, PINPUT_RECORD, DWORD, LPDWORD
Console.PeekConsoleInputW.argtypes = [HANDLE, c_void_p, DWORD, LPDWORD]
Console.ReadConsoleInputW.restype = BOOL
# HANDLE, PINPUT_RECORD, DWORD, LPDWORD
Console.ReadConsoleInputW.argtypes = [HANDLE, c_void_p, DWORD, LPDWORD]
Console.ScrollConsoleScreenBufferW.restype = BOOL
Console.ScrollConsoleScreenBufferW.argtypes = [
    HANDLE,
    c_void_p,
    c_void_p,
    c_int,
    c_void_p,
]  # HANDLE, SMALL_RECT*, SMALL_RECT*, COORD, LPDWORD
Console.SetConsoleActiveScreenBuffer.restype = BOOL
Console.SetConsoleActiveScreenBuffer.argtypes = [HANDLE]  # HANDLE
Console.SetConsoleCursorInfo.restype = BOOL
Console.SetConsoleCursorInfo.argtypes = [
    HANDLE,
    c_void_p,
]  # HANDLE, CONSOLE_CURSOR_INFO*
Console.SetConsoleCursorPosition.restype = BOOL
Console.SetConsoleCursorPosition.argtypes = [HANDLE, c_int]  # HANDLE, COORD
Console.SetConsoleMode.restype = BOOL
Console.SetConsoleMode.argtypes = [HANDLE, DWORD]  # HANDLE, DWORD
Console.SetConsoleScreenBufferSize.restype = BOOL
Console.SetConsoleScreenBufferSize.argtypes = [HANDLE, c_int]  # HANDLE, COORD
Console.SetConsoleTextAttribute.restype = BOOL
Console.SetConsoleTextAttribute.argtypes = [HANDLE, WORD]  # HANDLE, WORD
Console.SetConsoleTitleW.restype = BOOL
Console.SetConsoleTitleW.argtypes = [c_wchar_p]  # LPCTSTR
Console.SetConsoleWindowInfo.restype = BOOL
Console.SetConsoleWindowInfo.argtypes = [
    HANDLE,
    BOOL,
    c_void_p,
]  # HANDLE, BOOL, SMALL_RECT*
Console.WriteConsoleW.restype = BOOL
# HANDLE, VOID*, DWORD, LPDWORD, LPVOID
Console.WriteConsoleW.argtypes = [HANDLE, c_void_p, DWORD, LPDWORD, LPVOID]
Console.WriteConsoleOutputCharacterW.restype = BOOL
Console.WriteConsoleOutputCharacterW.argtypes = [
    HANDLE,
    c_wchar_p,
    DWORD,
    c_int,
    LPDWORD,
]  # HANDLE, LPCTSTR, DWORD, COORD, LPDWORD
Console.WriteFile.restype = BOOL
# HANDLE, LPCVOID , DWORD, LPDWORD , LPOVERLAPPED
Console.WriteFile.argtypes = [HANDLE, LPCVOID, DWORD, LPDWORD, c_void_p]


VkKeyScan = windll.user32.VkKeyScanA


class event(Event):
    """Represent events from the console."""

    def __init__(self, console, input):
        """Initialize an event from the Windows input structure."""
        self.type = "??"
        self.serial = console.next_serial()
        self.width = 0
        self.height = 0
        self.x = 0
        self.y = 0
        self.char = ""
        self.keycode = 0
        self.keysym = "??"
        # a tuple with (control, meta, shift, keycode) for dispatch
        self.keyinfo = None
        self.width = None

        if input.EventType == KEY_EVENT:
            if input.Event.KeyEvent.bKeyDown:
                self.type = "KeyPress"
            else:
                self.type = "KeyRelease"
            self.char = input.Event.KeyEvent.uChar.UnicodeChar
            self.keycode = input.Event.KeyEvent.wVirtualKeyCode
            self.state = input.Event.KeyEvent.dwControlKeyState
            self.keyinfo = make_KeyPress(self.char, self.state, self.keycode)

        elif input.EventType == MOUSE_EVENT:
            if input.Event.MouseEvent.dwEventFlags & MOUSE_MOVED:
                self.type = "Motion"
            else:
                self.type = "Button"
            self.x = input.Event.MouseEvent.dwMousePosition.X
            self.y = input.Event.MouseEvent.dwMousePosition.Y
            self.state = input.Event.MouseEvent.dwButtonState
        elif input.EventType == WINDOW_BUFFER_SIZE_EVENT:
            self.type = "Configure"
            self.width = input.Event.WindowBufferSizeEvent.dwSize.X
            self.height = input.Event.WindowBufferSizeEvent.dwSize.Y
        elif input.EventType == FOCUS_EVENT:
            if input.Event.FocusEvent.bSetFocus:
                self.type = "FocusIn"
            else:
                self.type = "FocusOut"
        elif input.EventType == MENU_EVENT:
            self.type = "Menu"
            self.state = input.Event.MenuEvent.dwCommandId


def getconsole(buffer=1):
    """Get a console handle.

    If buffer is non-zero, a new console buffer is allocated and
    installed.  Otherwise, this returns a handle to the current
    console buffer"""

    c = Console(buffer)

    return c


# The following code uses ctypes to allow a Python callable to
# substitute for GNU readline within the Python interpreter. Calling
# raw_input or other functions that do input, inside your callable
# might be a bad idea, then again, it might work.

# The Python callable can raise EOFError or KeyboardInterrupt and
# these will be translated into the appropriate outputs from readline
# so that they will then be translated back!

# If the Python callable raises any other exception, a traceback will
# be printed and readline will appear to return an empty line.

# I use ctypes to create a C-callable from a Python wrapper that
# handles the exceptions and gets the result into the right form.


# the type for our C-callable wrapper
HOOKFUNC23 = CFUNCTYPE(c_char_p, c_void_p, c_void_p, c_char_p)

readline_hook = None  # the python hook goes here
readline_ref = None  # reference to the c-callable to keep it alive


def hook_wrapper_23(stdin, stdout, prompt):
    """Wrap a Python readline so it behaves like GNU readline."""
    try:
        # call the Python hook
        res = ensure_str(readline_hook(prompt))
        # make sure it returned the right sort of thing
        if res and not isinstance(res, bytes):
            raise TypeError("readline must return a string.")
    except KeyboardInterrupt:
        # GNU readline returns 0 on keyboard interrupt
        return 0
    except EOFError:
        # It returns an empty string on EOF
        res = ensure_str("")
    except BaseException:
        print("Readline internal error", file=sys.stderr)
        traceback.print_exc()
        res = ensure_str("\n")
    # we have to make a copy because the caller expects to free the result
    n = len(res)
    p = Console.PyMem_Malloc(n + 1)
    _strncpy(cast(p, c_char_p), res, n + 1)
    return p


def install_readline(hook):
    """Set up things for the interpreter to call
    our function like GNU readline."""
    global readline_hook, readline_ref
    # save the hook so the wrapper can call it
    readline_hook = hook
    # get the address of PyOS_ReadlineFunctionPointer so we can update it
    PyOS_RFP = c_void_p.from_address(
        Console.GetProcAddress(
            sys.dllhandle, "PyOS_ReadlineFunctionPointer".encode("ascii")
        )
    )
    # save a reference to the generated C-callable so it doesn't go away
    readline_ref = HOOKFUNC23(hook_wrapper_23)
    # get the address of the function
    func_start = c_void_p.from_address(addressof(readline_ref)).value
    # write the function address into PyOS_ReadlineFunctionPointer
    PyOS_RFP.value = func_start


if __name__ == "__main__":
    import sys
    import time

    def p(char):
        return chr(VkKeyScan(ord(char)) & 0xFF)

    c = Console(0)
    sys.stdout = c
    sys.stderr = c
    c.page()
    print(p("d"), p("D"))
    c.pos(5, 10)
    c.write("hi there")
    print("some printed output")
    for i in range(10):
        q = c.getkeypress()
        print(q)
    del c
