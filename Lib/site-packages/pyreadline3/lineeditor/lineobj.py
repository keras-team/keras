# -*- coding: utf-8 -*-
# *****************************************************************************
#       Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#       Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************


import pyreadline3.clipboard as clipboard
from pyreadline3.unicode_helper import biter, ensure_unicode

from . import wordmatcher

# set to true to copy every addition to kill ring to clipboard
kill_ring_to_clipboard = False


class NotAWordError(IndexError):
    pass


def quote_char(c):
    if ord(c) > 0:
        return c


# ############# Line positioner ########################


class LinePositioner(object):
    def __call__(self, line):
        NotImplementedError("Base class !!!")


class NextChar(LinePositioner):
    def __call__(self, line):
        if line.point < len(line.line_buffer):
            return line.point + 1
        else:
            return line.point


NextChar = NextChar()


class PrevChar(LinePositioner):
    def __call__(self, line):
        if line.point > 0:
            return line.point - 1
        else:
            return line.point


PrevChar = PrevChar()


class NextWordStart(LinePositioner):
    def __call__(self, line):
        return line.next_start_segment(line.line_buffer, line.is_word_token)[line.point]


NextWordStart = NextWordStart()


class NextWordEnd(LinePositioner):
    def __call__(self, line):
        return line.next_end_segment(line.line_buffer, line.is_word_token)[line.point]


NextWordEnd = NextWordEnd()


class PrevWordStart(LinePositioner):
    def __call__(self, line):
        return line.prev_start_segment(line.line_buffer, line.is_word_token)[line.point]


PrevWordStart = PrevWordStart()


class WordStart(LinePositioner):
    def __call__(self, line):
        if line.is_word_token(line.get_line_text()[Point(line) : Point(line) + 1]):
            if Point(line) > 0 and line.is_word_token(
                line.get_line_text()[Point(line) - 1 : Point(line)]
            ):
                return PrevWordStart(line)
            else:
                return line.point
        else:
            raise NotAWordError("Point is not in a word")


WordStart = WordStart()


class WordEnd(LinePositioner):
    def __call__(self, line):
        if line.is_word_token(line.get_line_text()[Point(line) : Point(line) + 1]):
            if line.is_word_token(
                line.get_line_text()[Point(line) + 1 : Point(line) + 2]
            ):
                return NextWordEnd(line)
            else:
                return line.point
        else:
            raise NotAWordError("Point is not in a word")


WordEnd = WordEnd()


class PrevWordEnd(LinePositioner):
    def __call__(self, line):
        return line.prev_end_segment(line.line_buffer, line.is_word_token)[line.point]


PrevWordEnd = PrevWordEnd()


class PrevSpace(LinePositioner):
    def __call__(self, line):
        point = line.point
        if line[point - 1 : point].get_line_text() == " ":
            while point > 0 and line[point - 1 : point].get_line_text() == " ":
                point -= 1
        while point > 0 and line[point - 1 : point].get_line_text() != " ":
            point -= 1
        return point


PrevSpace = PrevSpace()


class StartOfLine(LinePositioner):
    def __call__(self, line):
        return 0


StartOfLine = StartOfLine()


class EndOfLine(LinePositioner):
    def __call__(self, line):
        return len(line.line_buffer)


EndOfLine = EndOfLine()


class Point(LinePositioner):
    def __call__(self, line):
        return line.point


Point = Point()


class Mark(LinePositioner):
    def __call__(self, line):
        return line.mark


k = Mark()

all_positioners = sorted(
    [
        (value.__class__.__name__, value)
        for key, value in globals().items()
        if isinstance(value, LinePositioner)
    ]
)

# ############## LineSlice #################


class LineSlice(object):
    def __call__(self, line):
        NotImplementedError("Base class !!!")


class CurrentWord(LineSlice):
    def __call__(self, line):
        return slice(WordStart(line), WordEnd(line), None)


CurrentWord = CurrentWord()


class NextWord(LineSlice):
    def __call__(self, line):
        work = TextLine(line)
        work.point = NextWordStart
        start = work.point
        stop = NextWordEnd(work)
        return slice(start, stop)


NextWord = NextWord()


class PrevWord(LineSlice):
    def __call__(self, line):
        work = TextLine(line)
        work.point = PrevWordEnd
        stop = work.point
        start = PrevWordStart(work)
        return slice(start, stop)


PrevWord = PrevWord()


class PointSlice(LineSlice):
    def __call__(self, line):
        return slice(Point(line), Point(line) + 1, None)


PointSlice = PointSlice()


# ##############  TextLine  ######################


class TextLine(object):
    def __init__(self, txtstr, point=None, mark=None):
        self.line_buffer = []
        self._point = 0
        self.mark = -1
        self.undo_stack = []
        self.overwrite = False
        if isinstance(txtstr, TextLine):  # copy
            self.line_buffer = txtstr.line_buffer[:]
            if point is None:
                self.point = txtstr.point
            else:
                self.point = point
            if mark is None:
                self.mark = txtstr.mark
            else:
                self.mark = mark
        else:
            self._insert_text(txtstr)
            if point is None:
                self.point = 0
            else:
                self.point = point
            if mark is None:
                self.mark = -1
            else:
                self.mark = mark

        self.is_word_token = wordmatcher.is_word_token
        self.next_start_segment = wordmatcher.next_start_segment
        self.next_end_segment = wordmatcher.next_end_segment
        self.prev_start_segment = wordmatcher.prev_start_segment
        self.prev_end_segment = wordmatcher.prev_end_segment

    def push_undo(self):
        l_text = self.get_line_text()
        if self.undo_stack and l_text == self.undo_stack[-1].get_line_text():
            self.undo_stack[-1].point = self.point
        else:
            self.undo_stack.append(self.copy())

    def pop_undo(self):
        if len(self.undo_stack) >= 2:
            self.undo_stack.pop()
            self.set_top_undo()
            self.undo_stack.pop()
        else:
            self.reset_line()
            self.undo_stack = []

    def set_top_undo(self):
        if self.undo_stack:
            undo = self.undo_stack[-1]
            self.line_buffer = undo.line_buffer
            self.point = undo.point
            self.mark = undo.mark
        else:
            pass

    def __repr__(self):
        return 'TextLine("%s",point=%s,mark=%s)' % (
            self.line_buffer,
            self.point,
            self.mark,
        )

    def copy(self):
        return self.__class__(self)

    def set_point(self, value):
        if isinstance(value, LinePositioner):
            value = value(self)
        assert value <= len(self.line_buffer)
        if value > len(self.line_buffer):
            value = len(self.line_buffer)
        self._point = value

    def get_point(self):
        return self._point

    point = property(get_point, set_point)

    def visible_line_width(self, position=Point):
        """Return the visible width of the text up to position."""
        extra_char_width = len(
            [None for c in self[:position].line_buffer if 0x2013 <= ord(c) <= 0xFFFD]
        )
        return (
            len(self[:position].quoted_text())
            + self[:position].line_buffer.count("\t") * 7
            + extra_char_width
        )

    def quoted_text(self):
        quoted = [quote_char(c) for c in self.line_buffer]
        return "".join(map(ensure_unicode, quoted))

    def get_line_text(self):
        buf = self.line_buffer
        buf = list(map(ensure_unicode, buf))
        return "".join(buf)

    def set_line(self, text, cursor=None):
        self.line_buffer = [c for c in str(text)]
        if cursor is None:
            self.point = len(self.line_buffer)
        else:
            self.point = cursor

    def reset_line(self):
        self.line_buffer = []
        self.point = 0

    def end_of_line(self):
        self.point = len(self.line_buffer)

    def _insert_text(self, text, argument=1):
        text = text * argument
        if self.overwrite:
            for c in biter(text):
                # if self.point:
                self.line_buffer[self.point] = c
                self.point += 1
        else:
            for c in biter(text):
                self.line_buffer.insert(self.point, c)
                self.point += 1

    def __getitem__(self, key):
        # Check if key is LineSlice, convert to regular slice
        # and continue processing
        if isinstance(key, LineSlice):
            key = key(self)
        if isinstance(key, slice):
            if key.step is None:
                pass
            else:
                raise RuntimeError('step is not "None"')
            if key.start is None:
                start = StartOfLine(self)
            elif isinstance(key.start, LinePositioner):
                start = key.start(self)
            else:
                start = key.start
            if key.stop is None:
                stop = EndOfLine(self)
            elif isinstance(key.stop, LinePositioner):
                stop = key.stop(self)
            else:
                stop = key.stop
            return self.__class__(self.line_buffer[start:stop], point=0)
        elif isinstance(key, LinePositioner):
            return self.line_buffer[key(self)]
        elif isinstance(key, tuple):
            # Multiple slice not allowed
            raise IndexError("Cannot use step in line buffer indexing")
        else:
            # return TextLine(self.line_buffer[key])
            return self.line_buffer[key]

    def __delitem__(self, key):
        point = self.point
        if isinstance(key, LineSlice):
            key = key(self)
        if isinstance(key, slice):
            start = key.start
            stop = key.stop
            if isinstance(start, LinePositioner):
                start = start(self)
            elif start is None:
                start = 0
            if isinstance(stop, LinePositioner):
                stop = stop(self)
            elif stop is None:
                stop = EndOfLine(self)
        elif isinstance(key, LinePositioner):
            start = key(self)
            stop = start + 1
        else:
            start = key
            stop = key + 1
        prev = self.line_buffer[:start]
        rest = self.line_buffer[stop:]
        self.line_buffer = prev + rest
        if point > stop:
            self.point = point - (stop - start)
        elif point >= start and point <= stop:
            self.point = start

    def __setitem__(self, key, value):
        if isinstance(key, LineSlice):
            key = key(self)
        if isinstance(key, slice):
            start = key.start
            stop = key.stop
        elif isinstance(key, LinePositioner):
            start = key(self)
            stop = start + 1
        else:
            start = key
            stop = key + 1
        prev = self.line_buffer[:start]
        value = self.__class__(value).line_buffer
        rest = self.line_buffer[stop:]
        out = prev + value + rest
        if len(out) >= len(self):
            self.point = len(self)
        self.line_buffer = out

    def __len__(self):
        return len(self.line_buffer)

    def upper(self):
        self.line_buffer = [x.upper() for x in self.line_buffer]
        return self

    def lower(self):
        self.line_buffer = [x.lower() for x in self.line_buffer]
        return self

    def capitalize(self):
        self.set_line(self.get_line_text().capitalize(), self.point)
        return self

    def startswith(self, txt):
        return self.get_line_text().startswith(txt)

    def endswith(self, txt):
        return self.get_line_text().endswith(txt)

    def __contains__(self, txt):
        return txt in self.get_line_text()


class ReadLineTextBuffer(TextLine):
    def __init__(self, txtstr, point=None, mark=None):
        super().__init__(txtstr, point, mark)

        self.enable_win32_clipboard = True
        self.selection_mark = -1
        self.enable_selection = True
        self.kill_ring = []

    def __repr__(self):
        return "ReadLineTextBuffer" '("%s",point=%s,mark=%s,selection_mark=%s)' % (
            self.line_buffer,
            self.point,
            self.mark,
            self.selection_mark,
        )

    def insert_text(self, char, argument=1):
        self.delete_selection()
        self.selection_mark = -1
        self._insert_text(char, argument)

    def to_clipboard(self):
        if self.enable_win32_clipboard:
            clipboard.set_clipboard_text(self.get_line_text())

    # Movement

    def beginning_of_line(self):
        self.selection_mark = -1
        self.point = StartOfLine

    def end_of_line(self):
        self.selection_mark = -1
        self.point = EndOfLine

    def forward_char(self, argument=1):
        if argument < 0:
            self.backward_char(-argument)
        self.selection_mark = -1
        for _ in range(argument):
            self.point = NextChar

    def backward_char(self, argument=1):
        if argument < 0:
            self.forward_char(-argument)
        self.selection_mark = -1
        for _ in range(argument):
            self.point = PrevChar

    def forward_word(self, argument=1):
        if argument < 0:
            self.backward_word(-argument)
        self.selection_mark = -1
        for _ in range(argument):
            self.point = NextWordStart

    def backward_word(self, argument=1):
        if argument < 0:
            self.forward_word(-argument)
        self.selection_mark = -1
        for _ in range(argument):
            self.point = PrevWordStart

    def forward_word_end(self, argument=1):
        if argument < 0:
            self.backward_word_end(-argument)
        self.selection_mark = -1
        for _ in range(argument):
            self.point = NextWordEnd

    def backward_word_end(self, argument=1):
        if argument < 0:
            self.forward_word_end(-argument)
        self.selection_mark = -1
        for _ in range(argument):
            self.point = NextWordEnd

    # Movement select
    def beginning_of_line_extend_selection(self):
        if self.enable_selection and self.selection_mark < 0:
            self.selection_mark = self.point
        self.point = StartOfLine

    def end_of_line_extend_selection(self):
        if self.enable_selection and self.selection_mark < 0:
            self.selection_mark = self.point
        self.point = EndOfLine

    def forward_char_extend_selection(self, argument=1):
        if argument < 0:
            self.backward_char_extend_selection(-argument)
        if self.enable_selection and self.selection_mark < 0:
            self.selection_mark = self.point
        for _ in range(argument):
            self.point = NextChar

    def backward_char_extend_selection(self, argument=1):
        if argument < 0:
            self.forward_char_extend_selection(-argument)
        if self.enable_selection and self.selection_mark < 0:
            self.selection_mark = self.point
        for _ in range(argument):
            self.point = PrevChar

    def forward_word_extend_selection(self, argument=1):
        if argument < 0:
            self.backward_word_extend_selection(-argument)
        if self.enable_selection and self.selection_mark < 0:
            self.selection_mark = self.point
        for _ in range(argument):
            self.point = NextWordStart

    def backward_word_extend_selection(self, argument=1):
        if argument < 0:
            self.forward_word_extend_selection(-argument)
        if self.enable_selection and self.selection_mark < 0:
            self.selection_mark = self.point
        for _ in range(argument):
            self.point = PrevWordStart

    def forward_word_end_extend_selection(self, argument=1):
        if argument < 0:
            self.backward_word_end_extend_selection(-argument)
        if self.enable_selection and self.selection_mark < 0:
            self.selection_mark = self.point
        for _ in range(argument):
            self.point = NextWordEnd

    def backward_word_end_extend_selection(self, argument=1):
        if argument < 0:
            self.forward_word_end_extend_selection(-argument)
        if self.enable_selection and self.selection_mark < 0:
            self.selection_mark = self.point
        for _ in range(argument):
            self.point = PrevWordEnd

    # delete

    def delete_selection(self):
        if self.enable_selection and self.selection_mark >= 0:
            if self.selection_mark < self.point:
                del self[self.selection_mark : self.point]
                self.selection_mark = -1
            else:
                del self[self.point : self.selection_mark]
                self.selection_mark = -1
            return True
        else:
            self.selection_mark = -1
            return False

    def delete_char(self, argument=1):
        if argument < 0:
            self.backward_delete_char(-argument)
        if self.delete_selection():
            argument -= 1
        for _ in range(argument):
            del self[Point]

    def backward_delete_char(self, argument=1):
        if argument < 0:
            self.delete_char(-argument)
        if self.delete_selection():
            argument -= 1
        for _ in range(argument):
            if self.point > 0:
                self.backward_char()
                self.delete_char()

    def forward_delete_word(self, argument=1):
        if argument < 0:
            self.backward_delete_word(-argument)
        if self.delete_selection():
            argument -= 1
        for _ in range(argument):
            del self[Point:NextWordStart]

    def backward_delete_word(self, argument=1):
        if argument < 0:
            self.forward_delete_word(-argument)
        if self.delete_selection():
            argument -= 1
        for _ in range(argument):
            del self[PrevWordStart:Point]

    def delete_current_word(self):
        if not self.delete_selection():
            del self[CurrentWord]
        self.selection_mark = -1

    def delete_horizontal_space(self):
        if self[Point] in " \t":
            del self[PrevWordEnd:NextWordStart]
        self.selection_mark = -1

    # Case

    def upcase_word(self):
        p = self.point
        try:
            self[CurrentWord] = self[CurrentWord].upper()
            self.point = p
        except NotAWordError:
            pass

    def downcase_word(self):
        p = self.point
        try:
            self[CurrentWord] = self[CurrentWord].lower()
            self.point = p
        except NotAWordError:
            pass

    def capitalize_word(self):
        p = self.point
        try:
            self[CurrentWord] = self[CurrentWord].capitalize()
            self.point = p
        except NotAWordError:
            pass

    # Transpose

    def transpose_chars(self):
        p2 = Point(self)
        if p2 == 0:
            return
        elif p2 == len(self):
            p2 = p2 - 1
        p1 = p2 - 1
        self[p2], self[p1] = self[p1], self[p2]
        self.point = p2 + 1

    def transpose_words(self):
        word1 = TextLine(self)
        word2 = TextLine(self)
        if self.point == len(self):
            word2.point = PrevWordStart
            word1.point = PrevWordStart(word2)
        else:
            word1.point = PrevWordStart
            word2.point = NextWordStart
        stop1 = NextWordEnd(word1)
        stop2 = NextWordEnd(word2)
        start1 = word1.point
        start2 = word2.point
        self[start2:stop2] = word1[Point:NextWordEnd]
        self[start1:stop1] = word2[Point:NextWordEnd]
        self.point = stop2

    # Kill

    def kill_line(self):
        self.add_to_kill_ring(self[self.point :])
        del self.line_buffer[self.point :]

    def kill_whole_line(self):
        self.add_to_kill_ring(self[:])
        del self[:]

    def backward_kill_line(self):
        del self[StartOfLine:Point]

    def unix_line_discard(self):
        del self[StartOfLine:Point]

    def kill_word(self):
        """Kills to next word ending"""
        del self[Point:NextWordEnd]

    def backward_kill_word(self):
        """Kills to next word ending"""
        if not self.delete_selection():
            del self[PrevWordStart:Point]
        self.selection_mark = -1

    def forward_kill_word(self):
        """Kills to next word ending"""
        if not self.delete_selection():
            del self[Point:NextWordEnd]
        self.selection_mark = -1

    def unix_word_rubout(self):
        if not self.delete_selection():
            del self[PrevSpace:Point]
        self.selection_mark = -1

    def kill_region(self):
        pass

    def copy_region_as_kill(self):
        pass

    def copy_backward_word(self):
        pass

    def copy_forward_word(self):
        pass

    def yank(self):
        self.paste_from_kill_ring()

    def yank_pop(self):
        pass

    # Mark

    def set_mark(self):
        self.mark = self.point

    def exchange_point_and_mark(self):
        pass

    def copy_region_to_clipboard(self):  # ()
        """Copy the text in the region to the windows clipboard."""
        if self.enable_win32_clipboard:
            mark = min(self.mark, len(self.line_buffer))
            cursor = min(self.point, len(self.line_buffer))
            if self.mark == -1:
                return
            begin = min(cursor, mark)
            end = max(cursor, mark)
            toclipboard = "".join(self.line_buffer[begin:end])
            clipboard.set_clipboard_text(toclipboard)

    def copy_selection_to_clipboard(self):  # ()
        """Copy the text in the region to the windows clipboard."""
        if (
            self.enable_win32_clipboard
            and self.enable_selection
            and self.selection_mark >= 0
        ):
            selection_mark = min(self.selection_mark, len(self.line_buffer))
            cursor = min(self.point, len(self.line_buffer))
            if self.selection_mark == -1:
                return
            begin = min(cursor, selection_mark)
            end = max(cursor, selection_mark)
            toclipboard = "".join(self.line_buffer[begin:end])
            clipboard.set_clipboard_text(toclipboard)

    def cut_selection_to_clipboard(self):  # ()
        self.copy_selection_to_clipboard()
        self.delete_selection()

    # Paste

    # Kill ring

    def add_to_kill_ring(self, txt):
        self.kill_ring = [txt]
        if kill_ring_to_clipboard:
            clipboard.set_clipboard_text(txt.get_line_text())

    def paste_from_kill_ring(self):
        if self.kill_ring:
            self.insert_text(self.kill_ring[0])


##################################################################
q = ReadLineTextBuffer("asff asFArw  ewrWErhg", point=8)
q = TextLine("asff asFArw  ewrWErhg", point=8)


def show_pos(buff, pos, chr="."):
    l_n = len(buff.line_buffer)

    def choice(is_bool):
        if is_bool:
            return chr
        else:
            return " "

    return "".join([choice(pos == idx) for idx in range(l_n + 1)])


def test_positioner(buff, points, positioner):
    print((" %s " % positioner.__class__.__name__).center(40, "-"))
    buffstr = buff.line_buffer

    print('"%s"' % (buffstr))
    for point in points:
        b = TextLine(buff, point=point)
        out = [" "] * (len(buffstr) + 1)
        pos = positioner(b)
        if pos == point:
            out[pos] = "&"
        else:
            out[point] = "."
            out[pos] = "^"
        print('"%s"' % ("".join(out)))


if __name__ == "__main__":

    print('%-15s "%s"' % ("Position", q.get_line_text()))
    print('%-15s "%s"' % ("Point", show_pos(q, q.point)))

    for name, positioner_q in all_positioners:
        pos_q = positioner_q(q)

        print('%-15s "%s"' % (name, show_pos(q, pos_q, "^")))

    l_t = ReadLineTextBuffer("kjjk asads   asad")
    l_t.point = EndOfLine
