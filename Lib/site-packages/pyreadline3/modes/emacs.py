# -*- coding: utf-8 -*-
# *****************************************************************************
#       Copyright (C) 2003-2006 Gary Bishop.
#       Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#       Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************


import pyreadline3.lineeditor.lineobj as lineobj
from pyreadline3.lineeditor.lineobj import Point
from pyreadline3.logger import log

from . import basemode


class IncrementalSearchPromptMode(object):
    def __init__(self, rlobj):
        pass

    def _process_incremental_search_keyevent(self, keyinfo):
        log("_process_incremental_search_keyevent")
        keytuple = keyinfo.tuple()
        # dispatch_func = self.key_dispatch.get(keytuple, default)
        revtuples = []
        fwdtuples = []
        for ktuple, func in self.key_dispatch.items():
            if func == self.reverse_search_history:
                revtuples.append(ktuple)
            elif func == self.forward_search_history:
                fwdtuples.append(ktuple)

        log("IncrementalSearchPromptMode %s %s" % (keyinfo, keytuple))
        if keyinfo.keyname == "backspace":
            self.subsearch_query = self.subsearch_query[:-1]
            if len(self.subsearch_query) > 0:
                self.line = self.subsearch_fun(self.subsearch_query)
            else:
                self._bell()
                self.line = ""  # empty query means no search result
        elif keyinfo.keyname in ["return", "escape"]:
            self._bell()
            self.prompt = self.subsearch_oldprompt
            self.process_keyevent_queue = self.process_keyevent_queue[:-1]
            self._history.history_cursor = len(self._history.history)
            if keyinfo.keyname == "escape":
                self.l_buffer.set_line(self.subsearch_old_line)
            return True
        elif keyinfo.keyname:
            pass
        elif keytuple in revtuples:
            self.subsearch_fun = self._history.reverse_search_history
            self.subsearch_prompt = "reverse-i-search%d`%s': "
            self.line = self.subsearch_fun(self.subsearch_query)
        elif keytuple in fwdtuples:
            self.subsearch_fun = self._history.forward_search_history
            self.subsearch_prompt = "forward-i-search%d`%s': "
            self.line = self.subsearch_fun(self.subsearch_query)
        elif keyinfo.control == False and keyinfo.meta == False:
            self.subsearch_query += keyinfo.char
            self.line = self.subsearch_fun(self.subsearch_query)
        else:
            pass
        self.prompt = self.subsearch_prompt % (
            self._history.history_cursor,
            self.subsearch_query,
        )
        self.l_buffer.set_line(self.line)

    def _init_incremental_search(self, searchfun, init_event):
        """Initialize search prompt"""
        log("init_incremental_search")
        self.subsearch_query = ""
        self.subsearch_fun = searchfun
        self.subsearch_old_line = self.l_buffer.get_line_text()

        queue = self.process_keyevent_queue
        queue.append(self._process_incremental_search_keyevent)

        self.subsearch_oldprompt = self.prompt

        if (
            self.previous_func != self.reverse_search_history
            and self.previous_func != self.forward_search_history
        ):
            self.subsearch_query = self.l_buffer[0:Point].get_line_text()

        if self.subsearch_fun == self.reverse_search_history:
            self.subsearch_prompt = "reverse-i-search%d`%s': "
        else:
            self.subsearch_prompt = "forward-i-search%d`%s': "

        self.prompt = self.subsearch_prompt % (self._history.history_cursor, "")

        if self.subsearch_query:
            self.line = self._process_incremental_search_keyevent(init_event)
        else:
            self.line = ""


class SearchPromptMode(object):
    def __init__(self, rlobj):
        pass

    def _process_non_incremental_search_keyevent(self, keyinfo):
        keytuple = keyinfo.tuple()
        log("SearchPromptMode %s %s" % (keyinfo, keytuple))
        history = self._history

        if keyinfo.keyname == "backspace":
            self.non_inc_query = self.non_inc_query[:-1]
        elif keyinfo.keyname in ["return", "escape"]:
            if self.non_inc_query:
                if self.non_inc_direction == -1:
                    res = history.reverse_search_history(self.non_inc_query)
                else:
                    res = history.forward_search_history(self.non_inc_query)

            self._bell()
            self.prompt = self.non_inc_oldprompt
            self.process_keyevent_queue = self.process_keyevent_queue[:-1]
            self._history.history_cursor = len(self._history.history)
            if keyinfo.keyname == "escape":
                self.l_buffer = self.non_inc_oldline
            else:
                self.l_buffer.set_line(res)
            return False
        elif keyinfo.keyname:
            pass
        elif keyinfo.control == False and keyinfo.meta == False:
            self.non_inc_query += keyinfo.char
        else:
            pass
        self.prompt = self.non_inc_oldprompt + ":" + self.non_inc_query

    def _init_non_i_search(self, direction):
        self.non_inc_direction = direction
        self.non_inc_query = ""
        self.non_inc_oldprompt = self.prompt
        self.non_inc_oldline = self.l_buffer.copy()
        self.l_buffer.reset_line()
        self.prompt = self.non_inc_oldprompt + ":"
        queue = self.process_keyevent_queue
        queue.append(self._process_non_incremental_search_keyevent)

    def non_incremental_reverse_search_history(self, e):  # (M-p)
        """Search backward starting at the current line and moving up
        through the history as necessary using a non-incremental search for
        a string supplied by the user."""
        return self._init_non_i_search(-1)

    def non_incremental_forward_search_history(self, e):  # (M-n)
        """Search forward starting at the current line and moving down
        through the the history as necessary using a non-incremental search
        for a string supplied by the user."""
        return self._init_non_i_search(1)


class LeaveModeTryNext(Exception):
    pass


class DigitArgumentMode(object):
    def __init__(self, rlobj):
        pass

    def _process_digit_argument_keyevent(self, keyinfo):
        log("DigitArgumentMode.keyinfo %s" % keyinfo)
        keytuple = keyinfo.tuple()
        log("DigitArgumentMode.keytuple %s %s" % (keyinfo, keytuple))
        if keyinfo.keyname in ["return"]:
            self.prompt = self._digit_argument_oldprompt
            self.process_keyevent_queue = self.process_keyevent_queue[:-1]
            return True
        elif keyinfo.keyname:
            pass
        elif (
            keyinfo.char in "0123456789"
            and keyinfo.control == False
            and keyinfo.meta == False
        ):
            log("arg %s %s" % (self.argument, keyinfo.char))
            self.argument = self.argument * 10 + int(keyinfo.char)
        else:
            self.prompt = self._digit_argument_oldprompt
            raise LeaveModeTryNext
        self.prompt = "(arg: %s) " % self.argument

    def _init_digit_argument(self, keyinfo):
        """Initialize search prompt"""
        c = self.console
        line = self.l_buffer.get_line_text()
        self._digit_argument_oldprompt = self.prompt
        queue = self.process_keyevent_queue
        queue = self.process_keyevent_queue
        queue.append(self._process_digit_argument_keyevent)

        if keyinfo.char == "-":
            self.argument = -1
        elif keyinfo.char in "0123456789":
            self.argument = int(keyinfo.char)
        log("<%s> %s" % (self.argument, type(self.argument)))
        self.prompt = "(arg: %s) " % self.argument
        log("arg-init %s %s" % (self.argument, keyinfo.char))


class EmacsMode(
    DigitArgumentMode, IncrementalSearchPromptMode, SearchPromptMode, basemode.BaseMode
):
    mode = "emacs"

    def __init__(self, rlobj):
        basemode.BaseMode.__init__(self, rlobj)
        IncrementalSearchPromptMode.__init__(self, rlobj)
        SearchPromptMode.__init__(self, rlobj)
        DigitArgumentMode.__init__(self, rlobj)
        self._keylog = lambda x, y: None
        self.previous_func = None
        self.prompt = ">>> "
        self._insert_verbatim = False
        self.next_meta = False  # True to force meta on next character

        self.process_keyevent_queue = [self._process_keyevent]

    def __repr__(self):
        return "<EmacsMode>"

    def add_key_logger(self, logfun):
        """logfun should be function that takes disp_fun and line_""" """buffer object """
        self._keylog = logfun

    def process_keyevent(self, keyinfo):
        try:
            r = self.process_keyevent_queue[-1](keyinfo)
        except LeaveModeTryNext:
            self.process_keyevent_queue = self.process_keyevent_queue[:-1]
            r = self.process_keyevent(keyinfo)
        if r:
            self.add_history(self.l_buffer.copy())
            return True
        return False

    def _process_keyevent(self, keyinfo):
        """return True when line is final"""
        # Process exit keys. Only exit on empty line
        log("_process_keyevent <%s>" % keyinfo)

        def nop(e):
            pass

        if self.next_meta:
            self.next_meta = False
            keyinfo.meta = True
        keytuple = keyinfo.tuple()

        if self._insert_verbatim:
            self.insert_text(keyinfo)
            self._insert_verbatim = False
            self.argument = 0
            return False

        if keytuple in self.exit_dispatch:
            pars = (self.l_buffer, lineobj.EndOfLine(self.l_buffer))
            log("exit_dispatch:<%s, %s>" % pars)
            if lineobj.EndOfLine(self.l_buffer) == 0:
                raise EOFError
        if keyinfo.keyname or keyinfo.control or keyinfo.meta:
            default = nop
        else:
            default = self.self_insert
        dispatch_func = self.key_dispatch.get(keytuple, default)

        log("readline from keyboard:<%s,%s>" % (keytuple, dispatch_func))

        r = None
        if dispatch_func:
            r = dispatch_func(keyinfo)
            self._keylog(dispatch_func, self.l_buffer)
            self.l_buffer.push_undo()

        self.previous_func = dispatch_func
        return r

    # History commands
    def previous_history(self, e):  # (C-p)
        """Move back through the history list, fetching the previous
        command."""
        self._history.previous_history(self.l_buffer)
        self.l_buffer.point = lineobj.EndOfLine
        self.finalize()

    def next_history(self, e):  # (C-n)
        """Move forward through the history list, fetching the next
        command."""
        self._history.next_history(self.l_buffer)
        self.finalize()

    def beginning_of_history(self, e):  # (M-<)
        """Move to the first line in the history."""
        self._history.beginning_of_history()
        self.finalize()

    def end_of_history(self, e):  # (M->)
        """Move to the end of the input history, i.e., the line currently
        being entered."""
        self._history.end_of_history(self.l_buffer)
        self.finalize()

    def reverse_search_history(self, e):  # (C-r)
        """Search backward starting at the current line and moving up
        through the history as necessary. This is an incremental search."""
        log("rev_search_history")
        self._init_incremental_search(self._history.reverse_search_history, e)
        self.finalize()

    def forward_search_history(self, e):  # (C-s)
        """Search forward starting at the current line and moving down
        through the the history as necessary. This is an incremental
        search."""
        log("fwd_search_history")
        self._init_incremental_search(self._history.forward_search_history, e)
        self.finalize()

    def history_search_forward(self, e):  # ()
        """Search forward through the history for the string of characters
        between the start of the current line and the point. This is a
        non-incremental search. By default, this command is unbound."""
        if self.previous_func and hasattr(self._history, self.previous_func.__name__):
            self._history.lastcommand = getattr(
                self._history, self.previous_func.__name__
            )
        else:
            self._history.lastcommand = None
        q = self._history.history_search_forward(self.l_buffer)
        self.l_buffer = q
        self.l_buffer.point = q.point
        self.finalize()

    def history_search_backward(self, e):  # ()
        """Search backward through the history for the string of characters
        between the start of the current line and the point. This is a
        non-incremental search. By default, this command is unbound."""
        if self.previous_func and hasattr(self._history, self.previous_func.__name__):
            self._history.lastcommand = getattr(
                self._history, self.previous_func.__name__
            )
        else:
            self._history.lastcommand = None
        q = self._history.history_search_backward(self.l_buffer)
        self.l_buffer = q
        self.l_buffer.point = q.point
        self.finalize()

    def yank_nth_arg(self, e):  # (M-C-y)
        """Insert the first argument to the previous command (usually the
        second word on the previous line) at point. With an argument n,
        insert the nth word from the previous command (the words in the
        previous command begin with word 0). A negative argument inserts the
        nth word from the end of the previous command."""
        self.finalize()

    def yank_last_arg(self, e):  # (M-. or M-_)
        """Insert last argument to the previous command (the last word of
        the previous history entry). With an argument, behave exactly like
        yank-nth-arg. Successive calls to yank-last-arg move back through
        the history list, inserting the last argument of each line in turn."""
        self.finalize()

    def forward_backward_delete_char(self, e):  # ()
        """Delete the character under the cursor, unless the cursor is at
        the end of the line, in which case the character behind the cursor
        is deleted. By default, this is not bound to a key."""
        self.finalize()

    def quoted_insert(self, e):  # (C-q or C-v)
        """Add the next character typed to the line verbatim. This is how to
        insert key sequences like C-q, for example."""
        self._insert_verbatim = True
        self.finalize()

    def tab_insert(self, e):  # (M-TAB)
        """Insert a tab character."""
        cursor = min(self.l_buffer.point, len(self.l_buffer.line_buffer))
        ws = " " * (self.tabstop - (cursor % self.tabstop))
        self.insert_text(ws)
        self.finalize()

    def transpose_chars(self, e):  # (C-t)
        """Drag the character before the cursor forward over the character
        at the cursor, moving the cursor forward as well. If the insertion
        point is at the end of the line, then this transposes the last two
        characters of the line. Negative arguments have no effect."""
        self.l_buffer.transpose_chars()
        self.finalize()

    def transpose_words(self, e):  # (M-t)
        """Drag the word before point past the word after point, moving
        point past that word as well. If the insertion point is at the end
        of the line, this transposes the last two words on the line."""
        self.l_buffer.transpose_words()
        self.finalize()

    def overwrite_mode(self, e):  # ()
        """Toggle overwrite mode. With an explicit positive numeric
        argument, switches to overwrite mode. With an explicit non-positive
        numeric argument, switches to insert mode. This command affects only
        emacs mode; vi mode does overwrite differently. Each call to
        readline() starts in insert mode. In overwrite mode, characters
        bound to self-insert replace the text at point rather than pushing
        the text to the right. Characters bound to backward-delete-char
        replace the character before point with a space."""
        self.finalize()

    def kill_line(self, e):  # (C-k)
        """Kill the text from point to the end of the line."""
        self.l_buffer.kill_line()
        self.finalize()

    def backward_kill_line(self, e):  # (C-x Rubout)
        """Kill backward to the beginning of the line."""
        self.l_buffer.backward_kill_line()
        self.finalize()

    def unix_line_discard(self, e):  # (C-u)
        """Kill backward from the cursor to the beginning of the current
        line."""
        # how is this different from backward_kill_line?
        self.l_buffer.unix_line_discard()
        self.finalize()

    def kill_whole_line(self, e):  # ()
        """Kill all characters on the current line, no matter where point
        is. By default, this is unbound."""
        self.l_buffer.kill_whole_line()
        self.finalize()

    def kill_word(self, e):  # (M-d)
        """Kill from point to the end of the current word, or if between
        words, to the end of the next word. Word boundaries are the same as
        forward-word."""
        self.l_buffer.kill_word()
        self.finalize()

    forward_kill_word = kill_word

    def backward_kill_word(self, e):  # (M-DEL)
        """Kill the word behind point. Word boundaries are the same as
        backward-word."""
        self.l_buffer.backward_kill_word()
        self.finalize()

    def unix_word_rubout(self, e):  # (C-w)
        """Kill the word behind point, using white space as a word
        boundary. The killed text is saved on the kill-ring."""
        self.l_buffer.unix_word_rubout()
        self.finalize()

    def kill_region(self, e):  # ()
        """Kill the text in the current region. By default, this command is
        unbound."""
        self.finalize()

    def copy_region_as_kill(self, e):  # ()
        """Copy the text in the region to the kill buffer, so it can be
        yanked right away. By default, this command is unbound."""
        self.finalize()

    def copy_backward_word(self, e):  # ()
        """Copy the word before point to the kill buffer. The word
        boundaries are the same as backward-word. By default, this command
        is unbound."""
        self.finalize()

    def copy_forward_word(self, e):  # ()
        """Copy the word following point to the kill buffer. The word
        boundaries are the same as forward-word. By default, this command is
        unbound."""
        self.finalize()

    def yank(self, e):  # (C-y)
        """Yank the top of the kill ring into the buffer at point."""
        self.l_buffer.yank()
        self.finalize()

    def yank_pop(self, e):  # (M-y)
        """Rotate the kill-ring, and yank the new top. You can only do this
        if the prior command is yank or yank-pop."""
        self.l_buffer.yank_pop()
        self.finalize()

    def delete_char_or_list(self, e):  # ()
        """Deletes the character under the cursor if not at the beginning or
        end of the line (like delete-char). If at the end of the line,
        behaves identically to possible-completions. This command is unbound
        by default."""
        self.finalize()

    def start_kbd_macro(self, e):  # (C-x ()
        """Begin saving the characters typed into the current keyboard
        macro."""
        self.finalize()

    def end_kbd_macro(self, e):  # (C-x ))
        """Stop saving the characters typed into the current keyboard macro
        and save the definition."""
        self.finalize()

    def call_last_kbd_macro(self, e):  # (C-x e)
        """Re-execute the last keyboard macro defined, by making the
        characters in the macro appear as if typed at the keyboard."""
        self.finalize()

    def re_read_init_file(self, e):  # (C-x C-r)
        """Read in the contents of the inputrc file, and incorporate any
        bindings or variable assignments found there."""
        self.finalize()

    def abort(self, e):  # (C-g)
        """Abort the current editing command and ring the terminals bell
        (subject to the setting of bell-style)."""
        self._bell()
        self.finalize()

    def do_uppercase_version(self, e):  # (M-a, M-b, M-x, ...)
        """If the metafied character x is lowercase, run the command that is
        bound to the corresponding uppercase character."""
        self.finalize()

    def prefix_meta(self, e):  # (ESC)
        """Metafy the next character typed. This is for keyboards without a
        meta key. Typing ESC f is equivalent to typing M-f."""
        self.next_meta = True
        self.finalize()

    def undo(self, e):  # (C-_ or C-x C-u)
        """Incremental undo, separately remembered for each line."""
        self.l_buffer.pop_undo()
        self.finalize()

    def revert_line(self, e):  # (M-r)
        """Undo all changes made to this line. This is like executing the
        undo command enough times to get back to the beginning."""
        self.finalize()

    def tilde_expand(self, e):  # (M-~)
        """Perform tilde expansion on the current word."""
        self.finalize()

    def set_mark(self, e):  # (C-@)
        """Set the mark to the point. If a numeric argument is supplied, the
        mark is set to that position."""
        self.l_buffer.set_mark()
        self.finalize()

    def exchange_point_and_mark(self, e):  # (C-x C-x)
        """Swap the point with the mark. The current cursor position is set
        to the saved position, and the old cursor position is saved as the
        mark."""
        self.finalize()

    def character_search(self, e):  # (C-])
        """A character is read and point is moved to the next occurrence of
        that character. A negative count searches for previous occurrences."""
        self.finalize()

    def character_search_backward(self, e):  # (M-C-])
        """A character is read and point is moved to the previous occurrence
        of that character. A negative count searches for subsequent
        occurrences."""
        self.finalize()

    def insert_comment(self, e):  # (M-#)
        """Without a numeric argument, the value of the comment-begin
        variable is inserted at the beginning of the current line. If a
        numeric argument is supplied, this command acts as a toggle: if the
        characters at the beginning of the line do not match the value of
        comment-begin, the value is inserted, otherwise the characters in
        comment-begin are deleted from the beginning of the line. In either
        case, the line is accepted as if a newline had been typed."""
        self.finalize()

    def dump_variables(self, e):  # ()
        """Print all of the settable variables and their values to the
        Readline output stream. If a numeric argument is supplied, the
        output is formatted in such a way that it can be made part of an
        inputrc file. This command is unbound by default."""
        self.finalize()

    def dump_macros(self, e):  # ()
        """Print all of the Readline key sequences bound to macros and the
        strings they output. If a numeric argument is supplied, the output
        is formatted in such a way that it can be made part of an inputrc
        file. This command is unbound by default."""
        self.finalize()

    def digit_argument(self, e):  # (M-0, M-1, ... M--)
        """Add this digit to the argument already accumulating, or start a
        new argument. M-- starts a negative argument."""
        self._init_digit_argument(e)
        # Should not finalize

    def universal_argument(self, e):  # ()
        """This is another way to specify an argument. If this command is
        followed by one or more digits, optionally with a leading minus
        sign, those digits define the argument. If the command is followed
        by digits, executing universal-argument again ends the numeric
        argument, but is otherwise ignored. As a special case, if this
        command is immediately followed by a character that is neither a
        digit or minus sign, the argument count for the next command is
        multiplied by four. The argument count is initially one, so
        executing this function the first time makes the argument count
        four, a second time makes the argument count sixteen, and so on. By
        default, this is not bound to a key."""
        # Should not finalize

    # Create key bindings:
    def init_editing_mode(self, e):  # (C-e)
        """When in vi command mode, this causes a switch to emacs editing
        mode."""
        self._bind_exit_key("Control-d")
        self._bind_exit_key("Control-z")

        # I often accidentally hold the shift or control while typing space
        self._bind_key("space", self.self_insert)
        self._bind_key("Shift-space", self.self_insert)
        self._bind_key("Control-space", self.self_insert)
        self._bind_key("Return", self.accept_line)
        self._bind_key("Left", self.backward_char)
        self._bind_key("Control-b", self.backward_char)
        self._bind_key("Right", self.forward_char)
        self._bind_key("Control-f", self.forward_char)
        self._bind_key("Control-h", self.backward_delete_char)
        self._bind_key("BackSpace", self.backward_delete_char)
        self._bind_key("Control-BackSpace", self.backward_delete_word)

        self._bind_key("Home", self.beginning_of_line)
        self._bind_key("End", self.end_of_line)
        self._bind_key("Delete", self.delete_char)
        self._bind_key("Control-d", self.delete_char)
        self._bind_key("Clear", self.clear_screen)
        self._bind_key("Alt-f", self.forward_word)
        self._bind_key("Alt-b", self.backward_word)
        self._bind_key("Control-l", self.clear_screen)
        self._bind_key("Control-p", self.previous_history)
        self._bind_key("Up", self.history_search_backward)
        self._bind_key("Control-n", self.next_history)
        self._bind_key("Down", self.history_search_forward)
        self._bind_key("Control-a", self.beginning_of_line)
        self._bind_key("Control-e", self.end_of_line)
        self._bind_key("Alt-<", self.beginning_of_history)
        self._bind_key("Alt->", self.end_of_history)
        self._bind_key("Control-r", self.reverse_search_history)
        self._bind_key("Control-s", self.forward_search_history)
        self._bind_key("Control-Shift-r", self.forward_search_history)
        self._bind_key("Alt-p", self.non_incremental_reverse_search_history)
        self._bind_key("Alt-n", self.non_incremental_forward_search_history)
        self._bind_key("Control-z", self.undo)
        self._bind_key("Control-_", self.undo)
        self._bind_key("Escape", self.kill_whole_line)
        self._bind_key("Meta-d", self.kill_word)
        self._bind_key("Control-Delete", self.forward_delete_word)
        self._bind_key("Control-w", self.unix_word_rubout)
        # self._bind_key('Control-Shift-v',   self.quoted_insert)
        self._bind_key("Control-v", self.paste)
        self._bind_key("Alt-v", self.ipython_paste)
        self._bind_key("Control-y", self.yank)
        self._bind_key("Control-k", self.kill_line)
        self._bind_key("Control-m", self.set_mark)
        self._bind_key("Control-q", self.copy_region_to_clipboard)
        #        self._bind_key('Control-shift-k',  self.kill_whole_line)
        self._bind_key("Control-Shift-v", self.paste_mulitline_code)
        self._bind_key("Control-Right", self.forward_word_end)
        self._bind_key("Control-Left", self.backward_word)
        self._bind_key("Shift-Right", self.forward_char_extend_selection)
        self._bind_key("Shift-Left", self.backward_char_extend_selection)
        self._bind_key("Shift-Control-Right", self.forward_word_end_extend_selection)
        self._bind_key("Shift-Control-Left", self.backward_word_extend_selection)
        self._bind_key("Shift-Home", self.beginning_of_line_extend_selection)
        self._bind_key("Shift-End", self.end_of_line_extend_selection)
        self._bind_key("numpad0", self.self_insert)
        self._bind_key("numpad1", self.self_insert)
        self._bind_key("numpad2", self.self_insert)
        self._bind_key("numpad3", self.self_insert)
        self._bind_key("numpad4", self.self_insert)
        self._bind_key("numpad5", self.self_insert)
        self._bind_key("numpad6", self.self_insert)
        self._bind_key("numpad7", self.self_insert)
        self._bind_key("numpad8", self.self_insert)
        self._bind_key("numpad9", self.self_insert)
        self._bind_key("add", self.self_insert)
        self._bind_key("subtract", self.self_insert)
        self._bind_key("multiply", self.self_insert)
        self._bind_key("divide", self.self_insert)
        self._bind_key("vk_decimal", self.self_insert)
        log("RUNNING INIT EMACS")
        for i in range(0, 10):
            self._bind_key("alt-%d" % i, self.digit_argument)
        self._bind_key("alt--", self.digit_argument)


# make it case insensitive
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
