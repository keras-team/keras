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

import pyreadline3.lineeditor.history as history
import pyreadline3.lineeditor.lineobj as lineobj
import pyreadline3.logger as logger
from pyreadline3.logger import log

from . import basemode


class NotEmacsMode(basemode.BaseMode):
    mode = "notemacs"

    def __init__(self, rlobj):
        super().__init__(rlobj)

    def __repr__(self):
        return "<NotEmacsMode>"

    def _readline_from_keyboard(self):
        c = self.console
        while True:
            self._update_line()
            event = c.getkeypress()
            if self.next_meta:
                self.next_meta = False
                control, meta, shift, code = event.keyinfo
                event.keyinfo = (control, True, shift, code)

            # Process exit keys. Only exit on empty line
            if event.keyinfo in self.exit_dispatch:
                if lineobj.EndOfLine(self.l_buffer) == 0:
                    raise EOFError

            dispatch_func = self.key_dispatch.get(event.keyinfo, self.self_insert)
            log("readline from keyboard:%s" % (event.keyinfo,))
            r = None
            if dispatch_func:
                r = dispatch_func(event)
                self.l_buffer.push_undo()

            self.previous_func = dispatch_func
            if r:
                self._update_line()
                break

    def readline(self, prompt=""):
        """Try to act like GNU readline."""
        # handle startup_hook
        if self.first_prompt:
            self.first_prompt = False
            if self.startup_hook:
                try:
                    self.startup_hook()
                except BaseException:
                    print("startup hook failed")
                    traceback.print_exc()

        c = self.console
        self.l_buffer.reset_line()
        self.prompt = prompt
        self._print_prompt()

        if self.pre_input_hook:
            try:
                self.pre_input_hook()
            except BaseException:
                print("pre_input_hook failed")
                traceback.print_exc()
                self.pre_input_hook = None

        log("in readline: %s" % self.paste_line_buffer)
        if len(self.paste_line_buffer) > 0:
            self.l_buffer = lineobj.ReadlineTextBuffer(self.paste_line_buffer[0])
            self._update_line()
            self.paste_line_buffer = self.paste_line_buffer[1:]
            c.write("\r\n")
        else:
            self._readline_from_keyboard()
            c.write("\r\n")

        self.add_history(self.l_buffer.copy())

        log("returning(%s)" % self.l_buffer.get_line_text())
        return self.l_buffer.get_line_text() + "\n"

    # Methods below here are bindable emacs functions

    def beginning_of_line(self, e):  # (C-a)
        """Move to the start of the current line."""
        self.l_buffer.beginning_of_line()

    def end_of_line(self, e):  # (C-e)
        """Move to the end of the line."""
        self.l_buffer.end_of_line()

    def forward_char(self, e):  # (C-f)
        """Move forward a character."""
        self.l_buffer.forward_char()

    def backward_char(self, e):  # (C-b)
        """Move back a character."""
        self.l_buffer.backward_char()

    def forward_word(self, e):  # (M-f)
        """Move forward to the end of the next word. Words are composed of
        letters and digits."""
        self.l_buffer.forward_word()

    def backward_word(self, e):  # (M-b)
        """Move back to the start of the current or previous word. Words are
        composed of letters and digits."""
        self.l_buffer.backward_word()

    def clear_screen(self, e):  # (C-l)
        """Clear the screen and redraw the current line, leaving the current
        line at the top of the screen."""
        self.console.page()

    def redraw_current_line(self, e):  # ()
        """Refresh the current line. By default, this is unbound."""
        pass

    def accept_line(self, e):  # (Newline or Return)
        """Accept the line regardless of where the cursor is. If this line
        is non-empty, it may be added to the history list for future recall
        with add_history(). If this line is a modified history line, the
        history line is restored to its original state."""
        return True

    # History commands
    def previous_history(self, e):  # (C-p)
        """Move back through the history list, fetching the previous command."""
        self._history.previous_history(self.l_buffer)

    def next_history(self, e):  # (C-n)
        """Move forward through the history list, fetching the next command."""
        self._history.next_history(self.l_buffer)

    def beginning_of_history(self, e):  # (M-<)
        """Move to the first line in the history."""
        self._history.beginning_of_history()

    def end_of_history(self, e):  # (M->)
        """Move to the end of the input history, i.e., the line currently
        being entered."""
        self._history.end_of_history(self.l_buffer)

    def _i_search(self, searchfun, direction, init_event):
        c = self.console
        line = self.get_line_buffer()
        query = ""
        hc_start = self._history.history_cursor  # + direction
        while True:
            x, y = self.prompt_end_pos
            c.pos(0, y)
            if direction < 0:
                prompt = "reverse-i-search"
            else:
                prompt = "forward-i-search"

            scroll = c.write_scrolling("%s`%s': %s" % (prompt, query, line))
            self._update_prompt_pos(scroll)
            self._clear_after()

            event = c.getkeypress()
            if event.keysym == "BackSpace":
                if len(query) > 0:
                    query = query[:-1]
                    self._history.history_cursor = hc_start
                else:
                    self._bell()
            elif (
                event.char in string.letters + string.digits + string.punctuation + " "
            ):
                self._history.history_cursor = hc_start
                query += event.char
            elif event.keyinfo == init_event.keyinfo:
                self._history.history_cursor += direction
                line = searchfun(query)
                pass
            else:
                if event.keysym != "Return":
                    self._bell()
                break
            line = searchfun(query)

        px, py = self.prompt_begin_pos
        c.pos(0, py)
        self.l_buffer.set_line(line)
        self._print_prompt()
        self._history.history_cursor = len(self._history.history)

    def reverse_search_history(self, e):  # (C-r)
        """Search backward starting at the current line and moving up
        through the history as necessary. This is an incremental search."""
        #        print("HEJ")
        #        self.console.bell()
        self._i_search(self._history.reverse_search_history, -1, e)

    def forward_search_history(self, e):  # (C-s)
        """Search forward starting at the current line and moving down
        through the the history as necessary. This is an incremental search."""
        #        print("HEJ")
        #        self.console.bell()
        self._i_search(self._history.forward_search_history, 1, e)

    def non_incremental_reverse_search_history(self, e):  # (M-p)
        """Search backward starting at the current line and moving up
        through the history as necessary using a non-incremental search for
        a string supplied by the user."""
        self._history.non_incremental_reverse_search_history(self.l_buffer)

    def non_incremental_forward_search_history(self, e):  # (M-n)
        """Search forward starting at the current line and moving down
        through the the history as necessary using a non-incremental search
        for a string supplied by the user."""
        self._history.non_incremental_reverse_search_history(self.l_buffer)

    def history_search_forward(self, e):  # ()
        """Search forward through the history for the string of characters
        between the start of the current line and the point. This is a
        non-incremental search. By default, this command is unbound."""
        self.l_buffer = self._history.history_search_forward(self.l_buffer)

    def history_search_backward(self, e):  # ()
        """Search backward through the history for the string of characters
        between the start of the current line and the point. This is a
        non-incremental search. By default, this command is unbound."""
        self.l_buffer = self._history.history_search_backward(self.l_buffer)

    def yank_nth_arg(self, e):  # (M-C-y)
        """Insert the first argument to the previous command (usually the
        second word on the previous line) at point. With an argument n,
        insert the nth word from the previous command (the words in the
        previous command begin with word 0). A negative argument inserts the
        nth word from the end of the previous command."""
        pass

    def yank_last_arg(self, e):  # (M-. or M-_)
        """Insert last argument to the previous command (the last word of
        the previous history entry). With an argument, behave exactly like
        yank-nth-arg. Successive calls to yank-last-arg move back through
        the history list, inserting the last argument of each line in turn."""
        pass

    def delete_char(self, e):  # (C-d)
        """Delete the character at point. If point is at the beginning of
        the line, there are no characters in the line, and the last
        character typed was not bound to delete-char, then return EOF."""
        self.l_buffer.delete_char()

    def backward_delete_char(self, e):  # (Rubout)
        """Delete the character behind the cursor. A numeric argument means
        to kill the characters instead of deleting them."""
        self.l_buffer.backward_delete_char()

    def forward_backward_delete_char(self, e):  # ()
        """Delete the character under the cursor, unless the cursor is at
        the end of the line, in which case the character behind the cursor
        is deleted. By default, this is not bound to a key."""
        pass

    def quoted_insert(self, e):  # (C-q or C-v)
        """Add the next character typed to the line verbatim. This is how to
        insert key sequences like C-q, for example."""
        e = self.console.getkeypress()
        self.insert_text(e.char)

    def tab_insert(self, e):  # (M-TAB)
        """Insert a tab character."""
        cursor = min(self.l_buffer.point, len(self.l_buffer.line_buffer))
        ws = " " * (self.tabstop - (cursor % self.tabstop))
        self.insert_text(ws)

    def self_insert(self, e):  # (a, b, A, 1, !, ...)
        """Insert yourself."""
        if (
            ord(e.char) != 0
        ):  # don't insert null character in buffer, can happen with dead keys.
            self.insert_text(e.char)

    def transpose_chars(self, e):  # (C-t)
        """Drag the character before the cursor forward over the character
        at the cursor, moving the cursor forward as well. If the insertion
        point is at the end of the line, then this transposes the last two
        characters of the line. Negative arguments have no effect."""
        self.l_buffer.transpose_chars()

    def transpose_words(self, e):  # (M-t)
        """Drag the word before point past the word after point, moving
        point past that word as well. If the insertion point is at the end
        of the line, this transposes the last two words on the line."""
        self.l_buffer.transpose_words()

    def upcase_word(self, e):  # (M-u)
        """Uppercase the current (or following) word. With a negative
        argument, uppercase the previous word, but do not move the cursor."""
        self.l_buffer.upcase_word()

    def downcase_word(self, e):  # (M-l)
        """Lowercase the current (or following) word. With a negative
        argument, lowercase the previous word, but do not move the cursor."""
        self.l_buffer.downcase_word()

    def capitalize_word(self, e):  # (M-c)
        """Capitalize the current (or following) word. With a negative
        argument, capitalize the previous word, but do not move the cursor."""
        self.l_buffer.capitalize_word()

    def overwrite_mode(self, e):  # ()
        """Toggle overwrite mode. With an explicit positive numeric
        argument, switches to overwrite mode. With an explicit non-positive
        numeric argument, switches to insert mode. This command affects only
        emacs mode; vi mode does overwrite differently. Each call to
        readline() starts in insert mode. In overwrite mode, characters
        bound to self-insert replace the text at point rather than pushing
        the text to the right. Characters bound to backward-delete-char
        replace the character before point with a space."""
        pass

    def kill_line(self, e):  # (C-k)
        """Kill the text from point to the end of the line."""
        self.l_buffer.kill_line()

    def backward_kill_line(self, e):  # (C-x Rubout)
        """Kill backward to the beginning of the line."""
        self.l_buffer.backward_kill_line()

    def unix_line_discard(self, e):  # (C-u)
        """Kill backward from the cursor to the beginning of the current line."""
        # how is this different from backward_kill_line?
        self.l_buffer.unix_line_discard()

    def kill_whole_line(self, e):  # ()
        """Kill all characters on the current line, no matter where point
        is. By default, this is unbound."""
        self.l_buffer.kill_whole_line()

    def kill_word(self, e):  # (M-d)
        """Kill from point to the end of the current word, or if between
        words, to the end of the next word. Word boundaries are the same as
        forward-word."""
        self.l_buffer.kill_word()

    def backward_kill_word(self, e):  # (M-DEL)
        """Kill the word behind point. Word boundaries are the same as
        backward-word."""
        self.l_buffer.backward_kill_word()

    def unix_word_rubout(self, e):  # (C-w)
        """Kill the word behind point, using white space as a word
        boundary. The killed text is saved on the kill-ring."""
        self.l_buffer.unix_word_rubout()

    def delete_horizontal_space(self, e):  # ()
        """Delete all spaces and tabs around point. By default, this is unbound."""
        pass

    def kill_region(self, e):  # ()
        """Kill the text in the current region. By default, this command is unbound."""
        pass

    def copy_region_as_kill(self, e):  # ()
        """Copy the text in the region to the kill buffer, so it can be
        yanked right away. By default, this command is unbound."""
        pass

    def copy_region_to_clipboard(self, e):  # ()
        """Copy the text in the region to the windows clipboard."""
        if self.enable_win32_clipboard:
            mark = min(self.l_buffer.mark, len(self.l_buffer.line_buffer))
            cursor = min(self.l_buffer.point, len(self.l_buffer.line_buffer))
            if self.l_buffer.mark == -1:
                return
            begin = min(cursor, mark)
            end = max(cursor, mark)
            toclipboard = "".join(self.l_buffer.line_buffer[begin:end])
            clipboard.set_clipboard_text(str(toclipboard))

    def copy_backward_word(self, e):  # ()
        """Copy the word before point to the kill buffer. The word
        boundaries are the same as backward-word. By default, this command
        is unbound."""
        pass

    def copy_forward_word(self, e):  # ()
        """Copy the word following point to the kill buffer. The word
        boundaries are the same as forward-word. By default, this command is
        unbound."""
        pass

    def paste(self, e):
        """Paste windows clipboard"""
        if self.enable_win32_clipboard:
            txt = clipboard.get_clipboard_text_and_convert(False)
            self.insert_text(txt)

    def paste_mulitline_code(self, e):
        """Paste windows clipboard"""
        reg = re.compile("\r?\n")
        if self.enable_win32_clipboard:
            txt = clipboard.get_clipboard_text_and_convert(False)
            t = reg.split(txt)
            t = [row for row in t if row.strip() != ""]  # remove empty lines
            if t != [""]:
                self.insert_text(t[0])
                self.add_history(self.l_buffer.copy())
                self.paste_line_buffer = t[1:]
                log("multi: %s" % self.paste_line_buffer)
                return True
            else:
                return False

    def ipython_paste(self, e):
        """Paste windows clipboard. If enable_ipython_paste_list_of_lists is
        True then try to convert tabseparated data to repr of list of lists or
        repr of array"""
        if self.enable_win32_clipboard:
            txt = clipboard.get_clipboard_text_and_convert(
                self.enable_ipython_paste_list_of_lists
            )
            if self.enable_ipython_paste_for_paths:
                if len(txt) < 300 and ("\t" not in txt) and ("\n" not in txt):
                    txt = txt.replace("\\", "/").replace(" ", r"\ ")
            self.insert_text(txt)

    def yank(self, e):  # (C-y)
        """Yank the top of the kill ring into the buffer at point."""
        pass

    def yank_pop(self, e):  # (M-y)
        """Rotate the kill-ring, and yank the new top. You can only do this
        if the prior command is yank or yank-pop."""
        pass

    def digit_argument(self, e):  # (M-0, M-1, ... M--)
        """Add this digit to the argument already accumulating, or start a
        new argument. M-- starts a negative argument."""
        pass

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
        pass

    def delete_char_or_list(self, e):  # ()
        """Deletes the character under the cursor if not at the beginning or
        end of the line (like delete-char). If at the end of the line,
        behaves identically to possible-completions. This command is unbound
        by default."""
        pass

    def start_kbd_macro(self, e):  # (C-x ()
        """Begin saving the characters typed into the current keyboard macro."""
        pass

    def end_kbd_macro(self, e):  # (C-x ))
        """Stop saving the characters typed into the current keyboard macro
        and save the definition."""
        pass

    def call_last_kbd_macro(self, e):  # (C-x e)
        """Re-execute the last keyboard macro defined, by making the
        characters in the macro appear as if typed at the keyboard."""
        pass

    def re_read_init_file(self, e):  # (C-x C-r)
        """Read in the contents of the inputrc file, and incorporate any
        bindings or variable assignments found there."""
        pass

    def abort(self, e):  # (C-g)
        """Abort the current editing command and ring the terminals bell
        (subject to the setting of bell-style)."""
        self._bell()

    def do_uppercase_version(self, e):  # (M-a, M-b, M-x, ...)
        """If the metafied character x is lowercase, run the command that is
        bound to the corresponding uppercase character."""
        pass

    def prefix_meta(self, e):  # (ESC)
        """Metafy the next character typed. This is for keyboards without a
        meta key. Typing ESC f is equivalent to typing M-f."""
        self.next_meta = True

    def undo(self, e):  # (C-_ or C-x C-u)
        """Incremental undo, separately remembered for each line."""
        self.l_buffer.pop_undo()

    def revert_line(self, e):  # (M-r)
        """Undo all changes made to this line. This is like executing the
        undo command enough times to get back to the beginning."""
        pass

    def tilde_expand(self, e):  # (M-~)
        """Perform tilde expansion on the current word."""
        pass

    def set_mark(self, e):  # (C-@)
        """Set the mark to the point. If a numeric argument is supplied, the
        mark is set to that position."""
        self.l_buffer.set_mark()

    def exchange_point_and_mark(self, e):  # (C-x C-x)
        """Swap the point with the mark. The current cursor position is set
        to the saved position, and the old cursor position is saved as the
        mark."""
        pass

    def character_search(self, e):  # (C-])
        """A character is read and point is moved to the next occurrence of
        that character. A negative count searches for previous occurrences."""
        pass

    def character_search_backward(self, e):  # (M-C-])
        """A character is read and point is moved to the previous occurrence
        of that character. A negative count searches for subsequent
        occurrences."""
        pass

    def insert_comment(self, e):  # (M-#)
        """Without a numeric argument, the value of the comment-begin
        variable is inserted at the beginning of the current line. If a
        numeric argument is supplied, this command acts as a toggle: if the
        characters at the beginning of the line do not match the value of
        comment-begin, the value is inserted, otherwise the characters in
        comment-begin are deleted from the beginning of the line. In either
        case, the line is accepted as if a newline had been typed."""
        pass

    def dump_functions(self, e):  # ()
        """Print all of the functions and their key bindings to the Readline
        output stream. If a numeric argument is supplied, the output is
        formatted in such a way that it can be made part of an inputrc
        file. This command is unbound by default."""
        pass

    def dump_variables(self, e):  # ()
        """Print all of the settable variables and their values to the
        Readline output stream. If a numeric argument is supplied, the
        output is formatted in such a way that it can be made part of an
        inputrc file. This command is unbound by default."""
        pass

    def dump_macros(self, e):  # ()
        """Print all of the Readline key sequences bound to macros and the
        strings they output. If a numeric argument is supplied, the output
        is formatted in such a way that it can be made part of an inputrc
        file. This command is unbound by default."""
        pass

    # Create key bindings:

    def init_editing_mode(self, e):  # (C-e)
        """When in vi command mode, this causes a switch to emacs editing
        mode."""

        self._bind_exit_key("Control-d")
        self._bind_exit_key("Control-z")

        # I often accidentally hold the shift or control while typing space
        self._bind_key("Shift-space", self.self_insert)
        self._bind_key("Control-space", self.self_insert)
        self._bind_key("Return", self.accept_line)
        self._bind_key("Left", self.backward_char)
        self._bind_key("Control-b", self.backward_char)
        self._bind_key("Right", self.forward_char)
        self._bind_key("Control-f", self.forward_char)
        self._bind_key("BackSpace", self.backward_delete_char)
        self._bind_key("Home", self.beginning_of_line)
        self._bind_key("End", self.end_of_line)
        self._bind_key("Delete", self.delete_char)
        self._bind_key("Control-d", self.delete_char)
        self._bind_key("Clear", self.clear_screen)


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
