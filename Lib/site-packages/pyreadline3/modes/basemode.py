# -*- coding: utf-8 -*-
# *****************************************************************************
#       Copyright (C) 2003-2006 Gary Bishop.
#       Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#       Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************


import glob
import math
import os
import re
import sys

import pyreadline3.clipboard as clipboard
import pyreadline3.lineeditor.history as history
import pyreadline3.lineeditor.lineobj as lineobj
from pyreadline3.error import ReadlineError
from pyreadline3.keysyms.common import make_KeyPress_from_keydescr
from pyreadline3.logger import log
from pyreadline3.py3k_compat import is_callable, is_ironpython
from pyreadline3.unicode_helper import ensure_str, ensure_unicode


class BaseMode(object):
    mode = "base"

    def __init__(self, rlobj):
        self.argument = 0
        self.rlobj = rlobj
        self.exit_dispatch = {}
        self.key_dispatch = {}
        self.argument = 1
        self.prevargument = None
        self.l_buffer = lineobj.ReadLineTextBuffer("")
        self._history = history.LineHistory()
        self.completer_delims = " \t\n\"\\'`@$><=;|&{("
        self.show_all_if_ambiguous = "on"
        self.mark_directories = "on"
        self.complete_filesystem = "off"
        self.completer = None
        self.begidx = 0
        self.endidx = 0
        self.tabstop = 4
        self.startup_hook = None
        self.pre_input_hook = None
        self.first_prompt = True
        self.cursor_size = 25

        self.prompt = ">>> "

        # Paste settings
        # assumes data on clipboard is path if shorter than 300 characters and doesn't contain \t or \n
        # and replace \ with / for easier use in ipython
        self.enable_ipython_paste_for_paths = True

        # automatically convert tabseparated data to list of lists or array
        # constructors
        self.enable_ipython_paste_list_of_lists = True
        self.enable_win32_clipboard = True

        self.paste_line_buffer = []

        self._sub_modes = []

    def __repr__(self):
        return "<BaseMode>"

    def _gs(x):
        def g(self):
            return getattr(self.rlobj, x)

        def s(self, q):
            setattr(self.rlobj, x, q)

        return g, s

    def _g(x):
        def g(self):
            return getattr(self.rlobj, x)

        return g

    def _argreset(self):
        val = self.argument
        self.argument = 0
        if val == 0:
            val = 1
        return val

    argument_reset = property(_argreset)

    # used in readline
    ctrl_c_tap_time_interval = property(*_gs("ctrl_c_tap_time_interval"))
    allow_ctrl_c = property(*_gs("allow_ctrl_c"))
    _print_prompt = property(_g("_print_prompt"))
    _update_line = property(_g("_update_line"))
    console = property(_g("console"))
    prompt_begin_pos = property(_g("prompt_begin_pos"))
    prompt_end_pos = property(_g("prompt_end_pos"))

    # used in completer _completions
    # completer_delims=property(*_gs("completer_delims"))
    _bell = property(_g("_bell"))
    bell_style = property(_g("bell_style"))

    # used in emacs
    _clear_after = property(_g("_clear_after"))
    _update_prompt_pos = property(_g("_update_prompt_pos"))

    # not used in basemode or emacs

    def process_keyevent(self, keyinfo):
        raise NotImplementedError

    def readline_setup(self, prompt=""):
        self.l_buffer.selection_mark = -1
        if self.first_prompt:
            self.first_prompt = False
            if self.startup_hook:
                try:
                    self.startup_hook()
                except BaseException:
                    print("startup hook failed")
                    traceback.print_exc()

        self.l_buffer.reset_line()
        self.prompt = prompt

        if self.pre_input_hook:
            try:
                self.pre_input_hook()
            except BaseException:
                print("pre_input_hook failed")
                traceback.print_exc()
                self.pre_input_hook = None

    # ###################################

    def finalize(self):
        """Every bindable command should call this function for cleanup.
        Except those that want to set argument to a non-zero value.
        """
        self.argument = 0

    def add_history(self, text):
        self._history.add_history(lineobj.ReadLineTextBuffer(text))

    # Create key bindings:

    def rl_settings_to_string(self):
        out = ["%-20s: %s" % ("show all if ambigous", self.show_all_if_ambiguous)]
        out.append("%-20s: %s" % ("mark_directories", self.mark_directories))
        out.append("%-20s: %s" % ("bell_style", self.bell_style))
        out.append("------------- key bindings ------------")
        tablepat = "%-7s %-7s %-7s %-15s %-15s "
        out.append(tablepat % ("Control", "Meta", "Shift", "Keycode/char", "Function"))
        bindings = sorted(
            [(k[0], k[1], k[2], k[3], v.__name__) for k, v in self.key_dispatch.items()]
        )
        for key in bindings:
            out.append(tablepat % (key))
        return out

    def _bind_key(self, key, func):
        """setup the mapping from key to call the function."""
        if not is_callable(func):
            print("Trying to bind non method to keystroke:%s,%s" % (key, func))
            raise ReadlineError(
                "Trying to bind non method to keystroke:%s,%s,%s,%s"
                % (key, func, type(func), type(self._bind_key))
            )
        keyinfo = make_KeyPress_from_keydescr(key.lower()).tuple()
        log(">>>%s -> %s<<<" % (keyinfo, func.__name__))
        self.key_dispatch[keyinfo] = func

    def _bind_exit_key(self, key):
        """setup the mapping from key to call the function."""
        keyinfo = make_KeyPress_from_keydescr(key.lower()).tuple()
        self.exit_dispatch[keyinfo] = None

    def init_editing_mode(self, e):  # (C-e)
        """When in vi command mode, this causes a switch to emacs editing
        mode."""

        raise NotImplementedError

    # completion commands

    def _get_completions(self):
        """Return a list of possible completions for the string ending at the point.
        Also set begidx and endidx in the process."""
        completions = []
        self.begidx = self.l_buffer.point
        self.endidx = self.l_buffer.point
        buf = self.l_buffer.line_buffer
        if self.completer:
            # get the string to complete
            while self.begidx > 0:
                self.begidx -= 1
                if buf[self.begidx] in self.completer_delims:
                    self.begidx += 1
                    break
            text = ensure_str("".join(buf[self.begidx : self.endidx]))
            log('complete text="%s"' % ensure_unicode(text))
            i = 0
            while True:
                try:
                    r = self.completer(ensure_unicode(text), i)
                except IndexError:
                    break
                i += 1
                if r is None:
                    break
                elif r and r not in completions:
                    completions.append(r)
                else:
                    pass
            log("text completions=<%s>" % list(map(ensure_unicode, completions)))
        if (self.complete_filesystem == "on") and not completions:
            # get the filename to complete
            while self.begidx > 0:
                self.begidx -= 1
                if buf[self.begidx] in " \t\n":
                    self.begidx += 1
                    break
            text = ensure_str("".join(buf[self.begidx : self.endidx]))
            log('file complete text="%s"' % ensure_unicode(text))
            completions = list(
                map(
                    ensure_unicode,
                    glob.glob(os.path.expanduser(text) + "*".encode("ascii")),
                )
            )
            if self.mark_directories == "on":
                mc = []
                for f in completions:
                    if os.path.isdir(f):
                        mc.append(f + os.sep)
                    else:
                        mc.append(f)
                completions = mc
            log("fnames=<%s>" % list(map(ensure_unicode, completions)))
        return completions

    def _display_completions(self, completions):
        if not completions:
            return
        self.console.write("\n")
        wmax = max(map(len, completions))
        w, h = self.console.size()
        cols = max(1, int((w - 1) / (wmax + 1)))
        rows = int(math.ceil(float(len(completions)) / cols))
        for row in range(rows):
            s = ""
            for col in range(cols):
                i = col * rows + row
                if i < len(completions):
                    self.console.write(completions[i].ljust(wmax + 1))
            self.console.write("\n")
        if is_ironpython:
            self.prompt = sys.ps1
        self._print_prompt()

    def complete(self, e):  # (TAB)
        """Attempt to perform completion on the text before point. The
        actual completion performed is application-specific. The default is
        filename completion."""
        completions = self._get_completions()
        if completions:
            cprefix = commonprefix(completions)
            if len(cprefix) > 0:
                rep = [c for c in cprefix]
                point = self.l_buffer.point
                self.l_buffer[self.begidx : self.endidx] = rep
                self.l_buffer.point = point + len(rep) - (self.endidx - self.begidx)
            if len(completions) > 1:
                if self.show_all_if_ambiguous == "on":
                    self._display_completions(completions)
                else:
                    self._bell()
        else:
            self._bell()
        self.finalize()

    def possible_completions(self, e):  # (M-?)
        """List the possible completions of the text before point."""
        completions = self._get_completions()
        self._display_completions(completions)
        self.finalize()

    def insert_completions(self, e):  # (M-*)
        """Insert all completions of the text before point that would have
        been generated by possible-completions."""
        completions = self._get_completions()
        b = self.begidx
        e = self.endidx
        for comp in completions:
            rep = [c for c in comp]
            rep.append(" ")
            self.l_buffer[b:e] = rep
            b += len(rep)
            e = b
        self.line_cursor = b
        self.finalize()

    def menu_complete(self, e):  # ()
        """Similar to complete, but replaces the word to be completed with a
        single match from the list of possible completions. Repeated
        execution of menu-complete steps through the list of possible
        completions, inserting each match in turn. At the end of the list of
        completions, the bell is rung (subject to the setting of bell-style)
        and the original text is restored. An argument of n moves n
        positions forward in the list of matches; a negative argument may be
        used to move backward through the list. This command is intended to
        be bound to TAB, but is unbound by default."""
        self.finalize()

    # Methods below here are bindable emacs functions

    def insert_text(self, string):
        """Insert text into the command line."""
        self.l_buffer.insert_text(string, self.argument_reset)
        self.finalize()

    def beginning_of_line(self, e):  # (C-a)
        """Move to the start of the current line."""
        self.l_buffer.beginning_of_line()
        self.finalize()

    def end_of_line(self, e):  # (C-e)
        """Move to the end of the line."""
        self.l_buffer.end_of_line()
        self.finalize()

    def forward_char(self, e):  # (C-f)
        """Move forward a character."""
        self.l_buffer.forward_char(self.argument_reset)
        self.finalize()

    def backward_char(self, e):  # (C-b)
        """Move back a character."""
        self.l_buffer.backward_char(self.argument_reset)
        self.finalize()

    def forward_word(self, e):  # (M-f)
        """Move forward to the end of the next word. Words are composed of
        letters and digits."""
        self.l_buffer.forward_word(self.argument_reset)
        self.finalize()

    def backward_word(self, e):  # (M-b)
        """Move back to the start of the current or previous word. Words are
        composed of letters and digits."""
        self.l_buffer.backward_word(self.argument_reset)
        self.finalize()

    def forward_word_end(self, e):  # ()
        """Move forward to the end of the next word. Words are composed of
        letters and digits."""
        self.l_buffer.forward_word_end(self.argument_reset)
        self.finalize()

    def backward_word_end(self, e):  # ()
        """Move forward to the end of the next word. Words are composed of
        letters and digits."""
        self.l_buffer.backward_word_end(self.argument_reset)
        self.finalize()

    # Movement with extend selection
    def beginning_of_line_extend_selection(self, e):
        """Move to the start of the current line."""
        self.l_buffer.beginning_of_line_extend_selection()
        self.finalize()

    def end_of_line_extend_selection(self, e):
        """Move to the end of the line."""
        self.l_buffer.end_of_line_extend_selection()
        self.finalize()

    def forward_char_extend_selection(self, e):
        """Move forward a character."""
        self.l_buffer.forward_char_extend_selection(self.argument_reset)
        self.finalize()

    def backward_char_extend_selection(self, e):
        """Move back a character."""
        self.l_buffer.backward_char_extend_selection(self.argument_reset)
        self.finalize()

    def forward_word_extend_selection(self, e):
        """Move forward to the end of the next word. Words are composed of
        letters and digits."""
        self.l_buffer.forward_word_extend_selection(self.argument_reset)
        self.finalize()

    def backward_word_extend_selection(self, e):
        """Move back to the start of the current or previous word. Words are
        composed of letters and digits."""
        self.l_buffer.backward_word_extend_selection(self.argument_reset)
        self.finalize()

    def forward_word_end_extend_selection(self, e):
        """Move forward to the end of the next word. Words are composed of
        letters and digits."""
        self.l_buffer.forward_word_end_extend_selection(self.argument_reset)
        self.finalize()

    def backward_word_end_extend_selection(self, e):
        """Move forward to the end of the next word. Words are composed of
        letters and digits."""
        self.l_buffer.forward_word_end_extend_selection(self.argument_reset)
        self.finalize()

    # Change case

    def upcase_word(self, e):  # (M-u)
        """Uppercase the current (or following) word. With a negative
        argument, uppercase the previous word, but do not move the cursor."""
        self.l_buffer.upcase_word()
        self.finalize()

    def downcase_word(self, e):  # (M-l)
        """Lowercase the current (or following) word. With a negative
        argument, lowercase the previous word, but do not move the cursor."""
        self.l_buffer.downcase_word()
        self.finalize()

    def capitalize_word(self, e):  # (M-c)
        """Capitalize the current (or following) word. With a negative
        argument, capitalize the previous word, but do not move the cursor."""
        self.l_buffer.capitalize_word()
        self.finalize()

    # #######

    def clear_screen(self, e):  # (C-l)
        """Clear the screen and redraw the current line, leaving the current
        line at the top of the screen."""
        self.console.page()
        self.finalize()

    def redraw_current_line(self, e):  # ()
        """Refresh the current line. By default, this is unbound."""
        self.finalize()

    def accept_line(self, e):  # (Newline or Return)
        """Accept the line regardless of where the cursor is. If this line
        is non-empty, it may be added to the history list for future recall
        with add_history(). If this line is a modified history line, the
        history line is restored to its original state."""
        self.finalize()
        return True

    def delete_char(self, e):  # (C-d)
        """Delete the character at point. If point is at the beginning of
        the line, there are no characters in the line, and the last
        character typed was not bound to delete-char, then return EOF."""
        self.l_buffer.delete_char(self.argument_reset)
        self.finalize()

    def backward_delete_char(self, e):  # (Rubout)
        """Delete the character behind the cursor. A numeric argument means
        to kill the characters instead of deleting them."""
        self.l_buffer.backward_delete_char(self.argument_reset)
        self.finalize()

    def backward_delete_word(self, e):  # (Control-Rubout)
        """Delete the character behind the cursor. A numeric argument means
        to kill the characters instead of deleting them."""
        self.l_buffer.backward_delete_word(self.argument_reset)
        self.finalize()

    def forward_delete_word(self, e):  # (Control-Delete)
        """Delete the character behind the cursor. A numeric argument means
        to kill the characters instead of deleting them."""
        self.l_buffer.forward_delete_word(self.argument_reset)
        self.finalize()

    def delete_horizontal_space(self, e):  # ()
        """Delete all spaces and tabs around point. By default, this is unbound."""
        self.l_buffer.delete_horizontal_space()
        self.finalize()

    def self_insert(self, e):  # (a, b, A, 1, !, ...)
        """Insert yourself."""
        if (
            e.char and ord(e.char) != 0
        ):  # don't insert null character in buffer, can happen with dead keys.
            self.insert_text(e.char)
        self.finalize()

    # Paste from clipboard

    def paste(self, e):
        """Paste windows clipboard.
        Assume single line strip other lines and end of line markers and trailing spaces
        """  # (Control-v)
        if self.enable_win32_clipboard:
            txt = clipboard.get_clipboard_text_and_convert(False)
            txt = txt.split("\n")[0].strip("\r").strip("\n")
            log("paste: >%s<" % list(map(ord, txt)))
            self.insert_text(txt)
        self.finalize()

    def paste_mulitline_code(self, e):
        """Paste windows clipboard as multiline code.
        Removes any empty lines in the code"""
        reg = re.compile("\r?\n")
        if self.enable_win32_clipboard:
            txt = clipboard.get_clipboard_text_and_convert(False)
            t = reg.split(txt)
            t = [row for row in t if row.strip() != ""]  # remove empty lines
            if t != [""]:
                self.insert_text(t[0])
                self.add_history(self.l_buffer.copy())
                self.paste_line_buffer = t[1:]
                log("multi: >%s<" % self.paste_line_buffer)
                return True
            else:
                return False
        self.finalize()

    def ipython_paste(self, e):
        """Paste windows clipboard. If enable_ipython_paste_list_of_lists is
        True then try to convert tabseparated data to repr of list of lists or
        repr of array.
        If enable_ipython_paste_for_paths==True then change \\ to / and spaces
        to \\space"""
        if self.enable_win32_clipboard:
            txt = clipboard.get_clipboard_text_and_convert(
                self.enable_ipython_paste_list_of_lists
            )
            if self.enable_ipython_paste_for_paths:
                if len(txt) < 300 and ("\t" not in txt) and ("\n" not in txt):
                    txt = txt.replace("\\", "/").replace(" ", r"\ ")
            self.insert_text(txt)
        self.finalize()

    def copy_region_to_clipboard(self, e):  # ()
        """Copy the text in the region to the windows clipboard."""
        self.l_buffer.copy_region_to_clipboard()
        self.finalize()

    def copy_selection_to_clipboard(self, e):  # ()
        """Copy the text in the region to the windows clipboard."""
        self.l_buffer.copy_selection_to_clipboard()
        self.finalize()

    def cut_selection_to_clipboard(self, e):  # ()
        """Copy the text in the region to the windows clipboard."""
        self.l_buffer.cut_selection_to_clipboard()
        self.finalize()

    def dump_functions(self, e):  # ()
        """Print all of the functions and their key bindings to the Readline
        output stream. If a numeric argument is supplied, the output is
        formatted in such a way that it can be made part of an inputrc
        file. This command is unbound by default."""
        print()
        txt = "\n".join(self.rl_settings_to_string())
        print(txt)
        self._print_prompt()
        self.finalize()


def commonprefix(m):
    "Given a list of pathnames, returns the longest common leading component"
    if not m:
        return ""
    prefix = m[0]
    for item in m:
        for i in range(len(prefix)):
            if prefix[: i + 1].lower() != item[: i + 1].lower():
                prefix = prefix[:i]
                if i == 0:
                    return ""
                break
    return prefix
