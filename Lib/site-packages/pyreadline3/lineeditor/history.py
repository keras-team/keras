# -*- coding: utf-8 -*-
# *****************************************************************************
#       Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#       Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************

import re, operator, string, sys, os, io

from pyreadline3.logger import log
from pyreadline3.unicode_helper import ensure_str, ensure_unicode

from . import lineobj


class EscapeHistory(Exception):
    pass


class LineHistory(object):
    def __init__(self):
        self.history = []
        self._history_length = 100
        self._history_cursor = 0
        # Cannot expand unicode strings correctly on python2.4
        self.history_filename = os.path.expanduser(ensure_str("~/.history"))
        self.lastcommand = None
        self.query = ""
        self.last_search_for = ""

    def get_current_history_length(self):
        """Return the number of lines currently in the history.
        (This is different from get_history_length(), which returns
        the maximum number of lines that will be written to a history file.)"""
        value = len(self.history)
        log("get_current_history_length:%d" % value)
        return value

    def get_history_length(self):
        """Return the desired length of the history file. Negative values imply
        unlimited history file size."""
        value = self._history_length
        log("get_history_length:%d" % value)
        return value

    def get_history_item(self, index):
        """Return the current contents of history item at index (starts with index 1)."""
        item = self.history[index - 1]
        log("get_history_item: index:%d item:%r" % (index, item))
        return item.get_line_text()

    def set_history_length(self, value):
        log("set_history_length: old:%d new:%d" % (self._history_length, value))
        self._history_length = value

    def get_history_cursor(self):
        value = self._history_cursor
        log("get_history_cursor:%d" % value)
        return value

    def set_history_cursor(self, value):
        log("set_history_cursor: old:%d new:%d" % (self._history_cursor, value))
        self._history_cursor = value

    history_length = property(get_history_length, set_history_length)
    history_cursor = property(get_history_cursor, set_history_cursor)

    def clear_history(self):
        """Clear readline history."""
        self.history[:] = []
        self.history_cursor = 0

    def read_history_file(self, filename=None):
        """Load a readline history file."""
        if filename is None:
            filename = self.history_filename
        try:
            with io.open(filename, "rt", errors="replace") as fd:
                for line in fd:
                    self.add_history(lineobj.ReadLineTextBuffer(line.rstrip()))
        except IOError:
            self.history = []
            self.history_cursor = 0

    def write_history_file(self, filename=None):
        """Save a readline history file."""
        if filename is None:
            filename = self.history_filename
        with io.open(filename, "wt", errors="replace") as fp:
            fp.writelines(
                tuple(
                    line.get_line_text()+"\n"
                    for line in self.history[-self.history_length :]
                )
            )

    def add_history(self, line):
        """Append a line to the history buffer, as if it was the last line typed."""
        line = ensure_unicode(line)
        if not hasattr(line, "get_line_text"):
            line = lineobj.ReadLineTextBuffer(line)
        if not line.get_line_text():
            pass
        elif (
            len(self.history) > 0
            and self.history[-1].get_line_text() == line.get_line_text()
        ):
            pass
        else:
            self.history.append(line)
        self.history_cursor = len(self.history)

    def previous_history(self, current):  # (C-p)
        """Move back through the history list, fetching the previous command."""
        if self.history_cursor == len(self.history):
            # do not use add_history since we do not want to increment cursor
            self.history.append(current.copy())

        if self.history_cursor > 0:
            self.history_cursor -= 1
            current.set_line(self.history[self.history_cursor].get_line_text())
            current.point = lineobj.EndOfLine

    def next_history(self, current):  # (C-n)
        """Move forward through the history list, fetching the next command."""
        if self.history_cursor < len(self.history) - 1:
            self.history_cursor += 1
            current.set_line(self.history[self.history_cursor].get_line_text())

    def beginning_of_history(self):  # (M-<)
        """Move to the first line in the history."""
        self.history_cursor = 0
        if len(self.history) > 0:
            self.l_buffer = self.history[0]

    def end_of_history(self, current):  # (M->)
        """Move to the end of the input history, i.e., the line currently
        being entered."""
        self.history_cursor = len(self.history)
        current.set_line(self.history[-1].get_line_text())

    def reverse_search_history(self, searchfor, startpos=None):
        if startpos is None:
            startpos = self.history_cursor
        origpos = startpos

        result = lineobj.ReadLineTextBuffer("")

        for idx, line in list(enumerate(self.history))[startpos:0:-1]:
            if searchfor in line:
                startpos = idx
                break

        # If we get a new search without change in search term it means
        # someone pushed ctrl-r and we should find the next match
        if self.last_search_for == searchfor and startpos > 0:
            startpos -= 1
            for idx, line in list(enumerate(self.history))[startpos:0:-1]:
                if searchfor in line:
                    startpos = idx
                    break

        if self.history:
            result = self.history[startpos].get_line_text()
        else:
            result = ""
        self.history_cursor = startpos
        self.last_search_for = searchfor
        log(
            "reverse_search_history: old:%d new:%d result:%r"
            % (origpos, self.history_cursor, result)
        )
        return result

    def forward_search_history(self, searchfor, startpos=None):
        if startpos is None:
            startpos = min(
                self.history_cursor, max(0, self.get_current_history_length() - 1)
            )
        # origpos = startpos

        result = lineobj.ReadLineTextBuffer("")

        for idx, line in list(enumerate(self.history))[startpos:]:
            if searchfor in line:
                startpos = idx
                break

        # If we get a new search without change in search term it means
        # someone pushed ctrl-r and we should find the next match
        if (
            self.last_search_for == searchfor
            and startpos < self.get_current_history_length() - 1
        ):
            startpos += 1
            for idx, line in list(enumerate(self.history))[startpos:]:
                if searchfor in line:
                    startpos = idx
                    break

        if self.history:
            result = self.history[startpos].get_line_text()
        else:
            result = ""
        self.history_cursor = startpos
        self.last_search_for = searchfor
        return result

    def _search(self, direction, partial):
        try:
            if (
                self.lastcommand != self.history_search_forward
                and self.lastcommand != self.history_search_backward
            ):
                self.query = "".join(partial[0 : partial.point].get_line_text())
            hcstart = max(self.history_cursor, 0)
            hc = self.history_cursor + direction
            while (direction < 0 and hc >= 0) or (
                direction > 0 and hc < len(self.history)
            ):
                h = self.history[hc]
                if not self.query:
                    self.history_cursor = hc
                    result = lineobj.ReadLineTextBuffer(h, point=len(h.get_line_text()))
                    return result
                elif h.get_line_text().startswith(self.query) and (
                    h != partial.get_line_text()
                ):
                    self.history_cursor = hc
                    result = lineobj.ReadLineTextBuffer(h, point=partial.point)
                    return result
                hc += direction
            else:
                if len(self.history) == 0:
                    pass
                elif hc >= len(self.history) and not self.query:
                    self.history_cursor = len(self.history)
                    return lineobj.ReadLineTextBuffer("", point=0)
                elif (
                    self.history[max(min(hcstart, len(self.history) - 1), 0)]
                    .get_line_text()
                    .startswith(self.query)
                    and self.query
                ):
                    return lineobj.ReadLineTextBuffer(
                        self.history[max(min(hcstart, len(self.history) - 1), 0)],
                        point=partial.point,
                    )
                else:
                    return lineobj.ReadLineTextBuffer(partial, point=partial.point)
                return lineobj.ReadLineTextBuffer(
                    self.query, point=min(len(self.query), partial.point)
                )
        except IndexError:
            raise

    def history_search_forward(self, partial):  # ()
        """Search forward through the history for the string of characters
        between the start of the current line and the point. This is a
        non-incremental search. By default, this command is unbound."""
        return self._search(1, partial)

    def history_search_backward(self, partial):  # ()
        """Search backward through the history for the string of characters
        between the start of the current line and the point. This is a
        non-incremental search. By default, this command is unbound."""

        return self._search(-1, partial)


if __name__ == "__main__":
    q = LineHistory()
    r = LineHistory()
    s = LineHistory()
    RL = lineobj.ReadLineTextBuffer
    q.add_history(RL("aaaa"))
    q.add_history(RL("aaba"))
    q.add_history(RL("aaca"))
    q.add_history(RL("akca"))
    q.add_history(RL("bbb"))
    q.add_history(RL("ako"))
    r.add_history(RL("ako"))
