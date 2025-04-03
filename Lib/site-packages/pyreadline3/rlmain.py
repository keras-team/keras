# -*- coding: utf-8 -*-
# *****************************************************************************
#       Copyright (C) 2003-2006 Gary Bishop.
#       Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#       Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************


import os
import re
import sys
import time
from glob import glob

import pyreadline3
import pyreadline3.clipboard as clipboard
import pyreadline3.console as console
import pyreadline3.lineeditor.history as history
import pyreadline3.lineeditor.lineobj as lineobj
import pyreadline3.logger as logger
from pyreadline3.keysyms.common import make_KeyPress_from_keydescr
from pyreadline3.py3k_compat import is_ironpython
from pyreadline3.unicode_helper import ensure_str, ensure_unicode

from .error import GetSetError, ReadlineError
from .logger import log
from .modes import editingmodes
from .py3k_compat import execfile, is_callable

# an attempt to implement readline for Python in Python using ctypes


if is_ironpython:  # ironpython does not provide a prompt string to readline
    import System

    default_prompt = ">>> "
else:
    default_prompt = ""


class MockConsoleError(Exception):
    pass


class MockConsole(object):
    """object used during refactoring. Should raise errors when someone tries
    to use it.
    """

    def __setattr__(self, _name, _value):
        raise MockConsoleError("Should not try to get attributes from MockConsole")

    def cursor(self, size=50):
        pass


class BaseReadline(object):
    def __init__(self):
        self.allow_ctrl_c = False
        self.ctrl_c_tap_time_interval = 0.3

        self.debug = False
        self.bell_style = "none"
        self.mark = -1
        self.console = MockConsole()
        self.disable_readline = False
        # this code needs to follow l_buffer and history creation
        self.editingmodes = [mode(self) for mode in editingmodes]
        for mode in self.editingmodes:
            mode.init_editing_mode(None)
        self.mode = self.editingmodes[0]

        self.read_inputrc()
        log("\n".join(self.mode.rl_settings_to_string()))

        self.callback = None

    def parse_and_bind(self, string):
        """Parse and execute single line of a readline init file."""
        try:
            log('parse_and_bind("%s")' % string)
            if string.startswith("#"):
                return
            if string.startswith("set"):
                m = re.compile(r"set\s+([-a-zA-Z0-9]+)\s+(.+)\s*$").match(string)
                if m:
                    var_name = m.group(1)
                    val = m.group(2)
                    try:
                        setattr(self.mode, var_name.replace("-", "_"), val)
                    except AttributeError:
                        log('unknown var="%s" val="%s"' % (var_name, val))
                else:
                    log('bad set "%s"' % string)
                return
            m = re.compile(r"\s*(\S+)\s*:\s*([-a-zA-Z]+)\s*$").match(string)
            if m:
                key = m.group(1)
                func_name = m.group(2)
                py_name = func_name.replace("-", "_")
                try:
                    func = getattr(self.mode, py_name)
                except AttributeError:
                    log('unknown func key="%s" func="%s"' % (key, func_name))
                    if self.debug:
                        print(
                            "pyreadline3 parse_and_bind error, unknown "
                            'function to bind: "%s"' % func_name
                        )
                    return
                self.mode._bind_key(key, func)
        except BaseException:
            log("error")
            raise

    def _set_prompt(self, prompt):
        self.mode.prompt = prompt

    def _get_prompt(self):
        return self.mode.prompt

    prompt = property(_get_prompt, _set_prompt)

    def get_line_buffer(self):
        """Return the current contents of the line buffer."""
        return self.mode.l_buffer.get_line_text()

    def insert_text(self, string):
        """Insert text into the command line."""
        self.mode.insert_text(string)

    def read_init_file(self, filename=None):
        """Parse a readline initialization file. The default filename is the last filename used."""
        log('read_init_file("%s")' % filename)

    # History file book keeping methods (non-bindable)

    def add_history(self, line):
        """Append a line to the history buffer, as if it was the last line typed."""
        self.mode._history.add_history(line)

    def get_current_history_length(self):
        """Return the number of lines currently in the history.
        (This is different from get_history_length(), which returns
        the maximum number of lines that will be written to a history file.)"""
        return self.mode._history.get_current_history_length()

    def get_history_length(self):
        """Return the desired length of the history file.

        Negative values imply unlimited history file size."""
        return self.mode._history.get_history_length()

    def set_history_length(self, length):
        """Set the number of lines to save in the history file.

        write_history_file() uses this value to truncate the history file
        when saving. Negative values imply unlimited history file size.
        """
        self.mode._history.set_history_length(length)

    def get_history_item(self, index):
        """Return the current contents of history item at index."""
        return self.mode._history.get_history_item(index)

    def clear_history(self):
        """Clear readline history"""
        self.mode._history.clear_history()

    def read_history_file(self, filename=None):
        """Load a readline history file. The default filename is ~/.history."""
        if filename is None:
            filename = self.mode._history.history_filename
        log("read_history_file from %s" % ensure_unicode(filename))
        self.mode._history.read_history_file(filename)

    def write_history_file(self, filename=None):
        """Save a readline history file. The default filename is ~/.history."""
        self.mode._history.write_history_file(filename)

    # Completer functions

    def set_completer(self, function=None):
        """Set or remove the completer function.

        If function is specified, it will be used as the new completer
        function; if omitted or None, any completer function already
        installed is removed. The completer function is called as
        function(text, state), for state in 0, 1, 2, ..., until it returns a
        non-string value. It should return the next possible completion
        starting with text.
        """
        log("set_completer")
        self.mode.completer = function

    def get_completer(self):
        """Get the completer function."""
        log("get_completer")
        return self.mode.completer

    def get_begidx(self):
        """Get the beginning index of the readline tab-completion scope."""
        return self.mode.begidx

    def get_endidx(self):
        """Get the ending index of the readline tab-completion scope."""
        return self.mode.endidx

    def set_completer_delims(self, string):
        """Set the readline word delimiters for tab-completion."""
        self.mode.completer_delims = string

    def get_completer_delims(self):
        """Get the readline word delimiters for tab-completion."""
        return self.mode.completer_delims

    def set_startup_hook(self, function=None):
        """Set or remove the startup_hook function.

        If function is specified, it will be used as the new startup_hook
        function; if omitted or None, any hook function already installed is
        removed. The startup_hook function is called with no arguments just
        before readline prints the first prompt.

        """
        self.mode.startup_hook = function

    def set_pre_input_hook(self, function=None):
        """Set or remove the pre_input_hook function.

        If function is specified, it will be used as the new pre_input_hook
        function; if omitted or None, any hook function already installed is
        removed. The pre_input_hook function is called with no arguments
        after the first prompt has been printed and just before readline
        starts reading input characters.

        """
        self.mode.pre_input_hook = function

    # Functions that are not relevant for all Readlines but should at least
    # have a NOP

    def _bell(self):
        pass

    #
    # Standard call, not available for all implementations
    #

    def readline(self, prompt=""):
        raise NotImplementedError

    #
    # Callback interface
    #
    def process_keyevent(self, keyinfo):
        return self.mode.process_keyevent(keyinfo)

    def readline_setup(self, prompt=""):
        return self.mode.readline_setup(prompt)

    def keyboard_poll(self):
        return self.mode._readline_from_keyboard_poll()

    def callback_handler_install(self, prompt, callback):
        """bool readline_callback_handler_install ( string prompt, callback callback)
        Initializes the readline callback interface and terminal, prints the prompt and returns immediately
        """
        self.callback = callback
        self.readline_setup(prompt)

    def callback_handler_remove(self):
        """Removes a previously installed callback handler and restores terminal settings"""
        self.callback = None

    def callback_read_char(self):
        """Reads a character and informs the readline callback interface when a line is received"""
        if self.keyboard_poll():
            line = self.get_line_buffer() + "\n"
            # however there is another newline added by
            # self.mode.readline_setup(prompt) which is called by callback_handler_install
            # this differs from GNU readline
            self.add_history(self.mode.l_buffer)
            # TADA:
            self.callback(line)

    def read_inputrc(
        self,  # in 2.4 we cannot call expanduser with unicode string
        inputrcpath=os.path.expanduser(ensure_str("~/pyreadlineconfig.ini")),
    ):
        modes = dict([(x.mode, x) for x in self.editingmodes])
        mode = self.editingmodes[0].mode

        def setmode(name):
            self.mode = modes[name]

        def bind_key(key, name):
            import types

            if is_callable(name):
                modes[mode]._bind_key(key, types.MethodType(name, modes[mode]))
            elif hasattr(modes[mode], name):
                modes[mode]._bind_key(key, getattr(modes[mode], name))
            else:
                print("Trying to bind unknown command '%s' to key '%s'" % (name, key))

        def un_bind_key(key):
            keyinfo = make_KeyPress_from_keydescr(key).tuple()
            if keyinfo in modes[mode].key_dispatch:
                del modes[mode].key_dispatch[keyinfo]

        def bind_exit_key(key):
            modes[mode]._bind_exit_key(key)

        def un_bind_exit_key(key):
            keyinfo = make_KeyPress_from_keydescr(key).tuple()
            if keyinfo in modes[mode].exit_dispatch:
                del modes[mode].exit_dispatch[keyinfo]

        def setkill_ring_to_clipboard(killring):
            import pyreadline3.lineeditor.lineobj

            pyreadline3.lineeditor.lineobj.kill_ring_to_clipboard = killring

        def sethistoryfilename(filename):
            self.mode._history.history_filename = os.path.expanduser(
                ensure_str(filename)
            )

        def setbellstyle(mode):
            self.bell_style = mode

        def disable_readline(mode):
            self.disable_readline = mode

        def sethistorylength(length):
            self.mode._history.history_length = int(length)

        def allow_ctrl_c(mode):
            log("allow_ctrl_c:%s:%s" % (self.allow_ctrl_c, mode))
            self.allow_ctrl_c = mode

        def setbellstyle(mode):
            self.bell_style = mode

        def show_all_if_ambiguous(mode):
            self.mode.show_all_if_ambiguous = mode

        def ctrl_c_tap_time_interval(mode):
            self.ctrl_c_tap_time_interval = mode

        def mark_directories(mode):
            self.mode.mark_directories = mode

        def completer_delims(delims):
            self.mode.completer_delims = delims

        def complete_filesystem(delims):
            self.mode.complete_filesystem = delims.lower()

        def enable_ipython_paste_for_paths(boolean):
            self.mode.enable_ipython_paste_for_paths = boolean

        def debug_output(
            on, filename="pyreadline_debug_log.txt"
        ):  # Not implemented yet
            if on in ["on", "on_nologfile"]:
                self.debug = True

            if on == "on":
                logger.start_file_log(filename)
                logger.start_socket_log()
                logger.log("STARTING LOG")
            elif on == "on_nologfile":
                logger.start_socket_log()
                logger.log("STARTING LOG")
            else:
                logger.log("STOPING LOG")
                logger.stop_file_log()
                logger.stop_socket_log()

        _color_trtable = {
            "black": 0,
            "darkred": 4,
            "darkgreen": 2,
            "darkyellow": 6,
            "darkblue": 1,
            "darkmagenta": 5,
            "darkcyan": 3,
            "gray": 7,
            "red": 4 + 8,
            "green": 2 + 8,
            "yellow": 6 + 8,
            "blue": 1 + 8,
            "magenta": 5 + 8,
            "cyan": 3 + 8,
            "white": 7 + 8,
        }

        def set_prompt_color(color):
            self.prompt_color = self._color_trtable.get(color.lower(), 7)

        def set_input_color(color):
            self.command_color = self._color_trtable.get(color.lower(), 7)

        loc = {
            "version": pyreadline3.__version__,
            "mode": mode,
            "modes": modes,
            "set_mode": setmode,
            "bind_key": bind_key,
            "disable_readline": disable_readline,
            "bind_exit_key": bind_exit_key,
            "un_bind_key": un_bind_key,
            "un_bind_exit_key": un_bind_exit_key,
            "bell_style": setbellstyle,
            "mark_directories": mark_directories,
            "show_all_if_ambiguous": show_all_if_ambiguous,
            "completer_delims": completer_delims,
            "complete_filesystem": complete_filesystem,
            "debug_output": debug_output,
            "history_filename": sethistoryfilename,
            "history_length": sethistorylength,
            "set_prompt_color": set_prompt_color,
            "set_input_color": set_input_color,
            "allow_ctrl_c": allow_ctrl_c,
            "ctrl_c_tap_time_interval": ctrl_c_tap_time_interval,
            "kill_ring_to_clipboard": setkill_ring_to_clipboard,
            "enable_ipython_paste_for_paths": enable_ipython_paste_for_paths,
        }
        if os.path.isfile(inputrcpath):
            try:
                execfile(inputrcpath, loc, loc)
            except Exception as x:
                raise
                import traceback

                print("Error reading .pyinputrc", file=sys.stderr)
                filepath, lineno = traceback.extract_tb(sys.exc_info()[2])[1][:2]
                print("Line: %s in file %s" % (lineno, filepath), file=sys.stderr)
                print(x, file=sys.stderr)
                raise ReadlineError("Error reading .pyinputrc")

    def redisplay(self):
        pass


class Readline(BaseReadline):
    """Baseclass for readline based on a console"""

    def __init__(self):
        super().__init__()

        self.console = console.Console()
        self.selection_color = self.console.saveattr << 4
        self.command_color = None
        self.prompt_color = None
        self.size = self.console.size()

        # variables you can control with parse_and_bind

    #  To export as readline interface

    # Internal functions

    def _bell(self):
        """ring the bell if requested."""
        if self.bell_style == "none":
            pass
        elif self.bell_style == "visible":
            raise NotImplementedError("Bellstyle visible is not implemented yet.")
        elif self.bell_style == "audible":
            self.console.bell()
        else:
            raise ReadlineError("Bellstyle %s unknown." % self.bell_style)

    def _clear_after(self):
        c = self.console
        x, y = c.pos()
        w, h = c.size()
        c.rectangle((x, y, w + 1, y + 1))
        c.rectangle((0, y + 1, w, min(y + 3, h)))

    def _set_cursor(self):
        c = self.console
        xc, yc = self.prompt_end_pos
        w, h = c.size()
        xc += self.mode.l_buffer.visible_line_width()
        while xc >= w:
            xc -= w
            yc += 1
        c.pos(xc, yc)

    def _print_prompt(self):
        c = self.console
        x, y = c.pos()

        n = c.write_scrolling(self.prompt, self.prompt_color)
        self.prompt_begin_pos = (x, y - n)
        self.prompt_end_pos = c.pos()
        self.size = c.size()

    def _update_prompt_pos(self, n):
        if n != 0:
            bx, by = self.prompt_begin_pos
            ex, ey = self.prompt_end_pos
            self.prompt_begin_pos = (bx, by - n)
            self.prompt_end_pos = (ex, ey - n)

    def _update_line(self):
        c = self.console
        l_buffer = self.mode.l_buffer
        c.cursor(0)  # Hide cursor avoiding flicking
        c.pos(*self.prompt_begin_pos)
        self._print_prompt()
        ltext = l_buffer.quoted_text()
        if l_buffer.enable_selection and (l_buffer.selection_mark >= 0):
            start = len(l_buffer[: l_buffer.selection_mark].quoted_text())
            stop = len(l_buffer[: l_buffer.point].quoted_text())
            if start > stop:
                stop, start = start, stop
            n = c.write_scrolling(ltext[:start], self.command_color)
            n = c.write_scrolling(ltext[start:stop], self.selection_color)
            n = c.write_scrolling(ltext[stop:], self.command_color)
        else:
            n = c.write_scrolling(ltext, self.command_color)

        x, y = c.pos()  # Preserve one line for Asian IME(Input Method Editor) statusbar
        w, h = c.size()
        if (y >= h - 1) or (n > 0):
            c.scroll_window(-1)
            c.scroll((0, 0, w, h), 0, -1)
            n += 1

        self._update_prompt_pos(n)
        if hasattr(
            c, "clear_to_end_of_window"
        ):  # Work around function for ironpython due
            c.clear_to_end_of_window()  # to System.Console's lack of FillFunction
        else:
            self._clear_after()

        # Show cursor, set size vi mode changes size in insert/overwrite mode
        c.cursor(1, size=self.mode.cursor_size)
        self._set_cursor()

    def callback_read_char(self):
        # Override base to get automatic newline
        """Reads a character and informs the readline callback interface when a line is received"""
        if self.keyboard_poll():
            line = self.get_line_buffer() + "\n"
            self.console.write("\r\n")
            # however there is another newline added by
            # self.mode.readline_setup(prompt) which is called by callback_handler_install
            # this differs from GNU readline
            self.add_history(self.mode.l_buffer)
            # TADA:
            self.callback(line)

    def event_available(self):
        return self.console.peek() or (len(self.paste_line_buffer) > 0)

    def _readline_from_keyboard(self):
        while True:
            if self._readline_from_keyboard_poll():
                break

    def _readline_from_keyboard_poll(self):
        pastebuffer = self.mode.paste_line_buffer
        if len(pastebuffer) > 0:
            # paste first line in multiline paste buffer
            self.l_buffer = lineobj.ReadLineTextBuffer(pastebuffer[0])
            self._update_line()
            self.mode.paste_line_buffer = pastebuffer[1:]
            return True

        c = self.console

        def nop(e):
            pass

        try:
            event = c.getkeypress()
        except KeyboardInterrupt:
            event = self.handle_ctrl_c()
        try:
            result = self.mode.process_keyevent(event.keyinfo)
        except EOFError:
            logger.stop_logging()
            raise
        self._update_line()
        return result

    def readline_setup(self, prompt=""):
        BaseReadline.readline_setup(self, prompt)
        self._print_prompt()
        self._update_line()

    def readline(self, prompt=""):
        self.readline_setup(prompt)
        self.ctrl_c_timeout = time.time()
        self._readline_from_keyboard()
        self.console.write("\r\n")
        log("returning(%s)" % self.get_line_buffer())
        return self.get_line_buffer() + "\n"

    def handle_ctrl_c(self):
        from pyreadline3.console.event import Event
        from pyreadline3.keysyms.common import KeyPress

        log("KBDIRQ")
        event = Event(0, 0)
        event.char = "c"
        event.keyinfo = KeyPress(
            "c", shift=False, control=True, meta=False, keyname=None
        )
        if self.allow_ctrl_c:
            now = time.time()
            if (now - self.ctrl_c_timeout) < self.ctrl_c_tap_time_interval:
                log("Raise KeyboardInterrupt")
                raise KeyboardInterrupt
            else:
                self.ctrl_c_timeout = now
        else:
            raise KeyboardInterrupt
        return event

    def redisplay(self):
        BaseReadline.redisplay(self)
