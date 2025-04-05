# Python Markdown

# A Python implementation of John Gruber's Markdown.

# Documentation: https://python-markdown.github.io/
# GitHub: https://github.com/Python-Markdown/markdown/
# PyPI: https://pypi.org/project/Markdown/

# Started by Manfred Stienstra (http://www.dwerg.net/).
# Maintained for a few years by Yuri Takhteyev (http://www.freewisdom.org).
# Currently maintained by Waylan Limberg (https://github.com/waylan),
# Dmitry Shachnev (https://github.com/mitya57) and Isaac Muse (https://github.com/facelessuser).

# Copyright 2007-2023 The Python Markdown Project (v. 1.7 and later)
# Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
# Copyright 2004 Manfred Stienstra (the original version)

# License: BSD (see LICENSE.md for details).

"""
In version 3.0, a new, more flexible inline processor was added, [`markdown.inlinepatterns.InlineProcessor`][].   The
original inline patterns, which inherit from [`markdown.inlinepatterns.Pattern`][] or one of its children are still
supported, though users are encouraged to migrate.

The new `InlineProcessor` provides two major enhancements to `Patterns`:

1. Inline Processors no longer need to match the entire block, so regular expressions no longer need to start with
  `r'^(.*?)'` and end with `r'(.*?)%'`. This runs faster. The returned [`Match`][re.Match] object will only contain
   what is explicitly matched in the pattern, and extension pattern groups now start with `m.group(1)`.

2.  The `handleMatch` method now takes an additional input called `data`, which is the entire block under analysis,
    not just what is matched with the specified pattern. The method now returns the element *and* the indexes relative
    to `data` that the return element is replacing (usually `m.start(0)` and `m.end(0)`).  If the boundaries are
    returned as `None`, it is assumed that the match did not take place, and nothing will be altered in `data`.

    This allows handling of more complex constructs than regular expressions can handle, e.g., matching nested
    brackets, and explicit control of the span "consumed" by the processor.

"""

from __future__ import annotations

from . import util
from typing import TYPE_CHECKING, Any, Collection, NamedTuple
import re
import xml.etree.ElementTree as etree
from html import entities

if TYPE_CHECKING:  # pragma: no cover
    from markdown import Markdown


def build_inlinepatterns(md: Markdown, **kwargs: Any) -> util.Registry[InlineProcessor]:
    """
    Build the default set of inline patterns for Markdown.

    The order in which processors and/or patterns are applied is very important - e.g. if we first replace
    `http://.../` links with `<a>` tags and _then_ try to replace inline HTML, we would end up with a mess. So, we
    apply the expressions in the following order:

    * backticks and escaped characters have to be handled before everything else so that we can preempt any markdown
      patterns by escaping them;

    * then we handle the various types of links (auto-links must be handled before inline HTML);

    * then we handle inline HTML.  At this point we will simply replace all inline HTML strings with a placeholder
      and add the actual HTML to a stash;

    * finally we apply strong, emphasis, etc.

    """
    inlinePatterns = util.Registry()
    inlinePatterns.register(BacktickInlineProcessor(BACKTICK_RE), 'backtick', 190)
    inlinePatterns.register(EscapeInlineProcessor(ESCAPE_RE, md), 'escape', 180)
    inlinePatterns.register(ReferenceInlineProcessor(REFERENCE_RE, md), 'reference', 170)
    inlinePatterns.register(LinkInlineProcessor(LINK_RE, md), 'link', 160)
    inlinePatterns.register(ImageInlineProcessor(IMAGE_LINK_RE, md), 'image_link', 150)
    inlinePatterns.register(
        ImageReferenceInlineProcessor(IMAGE_REFERENCE_RE, md), 'image_reference', 140
    )
    inlinePatterns.register(
        ShortReferenceInlineProcessor(REFERENCE_RE, md), 'short_reference', 130
    )
    inlinePatterns.register(
        ShortImageReferenceInlineProcessor(IMAGE_REFERENCE_RE, md), 'short_image_ref', 125
    )
    inlinePatterns.register(AutolinkInlineProcessor(AUTOLINK_RE, md), 'autolink', 120)
    inlinePatterns.register(AutomailInlineProcessor(AUTOMAIL_RE, md), 'automail', 110)
    inlinePatterns.register(SubstituteTagInlineProcessor(LINE_BREAK_RE, 'br'), 'linebreak', 100)
    inlinePatterns.register(HtmlInlineProcessor(HTML_RE, md), 'html', 90)
    inlinePatterns.register(HtmlInlineProcessor(ENTITY_RE, md), 'entity', 80)
    inlinePatterns.register(SimpleTextInlineProcessor(NOT_STRONG_RE), 'not_strong', 70)
    inlinePatterns.register(AsteriskProcessor(r'\*'), 'em_strong', 60)
    inlinePatterns.register(UnderscoreProcessor(r'_'), 'em_strong2', 50)
    return inlinePatterns


# The actual regular expressions for patterns
# -----------------------------------------------------------------------------

NOIMG = r'(?<!\!)'
""" Match not an image. Partial regular expression which matches if not preceded by `!`. """

BACKTICK_RE = r'(?:(?<!\\)((?:\\{2})+)(?=`+)|(?<!\\)(`+)(.+?)(?<!`)\2(?!`))'
""" Match backtick quoted string (`` `e=f()` `` or ``` ``e=f("`")`` ```). """

ESCAPE_RE = r'\\(.)'
""" Match a backslash escaped character (`\\<` or `\\*`). """

EMPHASIS_RE = r'(\*)([^\*]+)\1'
""" Match emphasis with an asterisk (`*emphasis*`). """

STRONG_RE = r'(\*{2})(.+?)\1'
""" Match strong with an asterisk (`**strong**`). """

SMART_STRONG_RE = r'(?<!\w)(_{2})(?!_)(.+?)(?<!_)\1(?!\w)'
""" Match strong with underscore while ignoring middle word underscores (`__smart__strong__`). """

SMART_EMPHASIS_RE = r'(?<!\w)(_)(?!_)(.+?)(?<!_)\1(?!\w)'
""" Match emphasis with underscore while ignoring middle word underscores (`_smart_emphasis_`). """

SMART_STRONG_EM_RE = r'(?<!\w)(\_)\1(?!\1)(.+?)(?<!\w)\1(?!\1)(.+?)\1{3}(?!\w)'
""" Match strong emphasis with underscores (`__strong _em__`). """

EM_STRONG_RE = r'(\*)\1{2}(.+?)\1(.*?)\1{2}'
""" Match emphasis strong with asterisk (`***strongem***` or `***em*strong**`). """

EM_STRONG2_RE = r'(_)\1{2}(.+?)\1(.*?)\1{2}'
""" Match emphasis strong with underscores (`___emstrong___` or `___em_strong__`). """

STRONG_EM_RE = r'(\*)\1{2}(.+?)\1{2}(.*?)\1'
""" Match strong emphasis with asterisk (`***strong**em*`). """

STRONG_EM2_RE = r'(_)\1{2}(.+?)\1{2}(.*?)\1'
""" Match strong emphasis with underscores (`___strong__em_`). """

STRONG_EM3_RE = r'(\*)\1(?!\1)([^*]+?)\1(?!\1)(.+?)\1{3}'
""" Match strong emphasis with asterisk (`**strong*em***`). """

LINK_RE = NOIMG + r'\['
""" Match start of in-line link (`[text](url)` or `[text](<url>)` or `[text](url "title")`). """

IMAGE_LINK_RE = r'\!\['
""" Match start of in-line image link (`![alttxt](url)` or `![alttxt](<url>)`). """

REFERENCE_RE = LINK_RE
""" Match start of reference link (`[Label][3]`). """

IMAGE_REFERENCE_RE = IMAGE_LINK_RE
""" Match start of image reference (`![alt text][2]`). """

NOT_STRONG_RE = r'((^|(?<=\s))(\*{1,3}|_{1,3})(?=\s|$))'
""" Match a stand-alone `*` or `_`. """

AUTOLINK_RE = r'<((?:[Ff]|[Hh][Tt])[Tt][Pp][Ss]?://[^<>]*)>'
""" Match an automatic link (`<http://www.example.com>`). """

AUTOMAIL_RE = r'<([^<> !]+@[^@<> ]+)>'
""" Match an automatic email link (`<me@example.com>`). """

HTML_RE = r'(<(\/?[a-zA-Z][^<>@ ]*( [^<>]*)?|!--(?:(?!<!--|-->).)*--)>)'
""" Match an HTML tag (`<...>`). """

ENTITY_RE = r'(&(?:\#[0-9]+|\#x[0-9a-fA-F]+|[a-zA-Z0-9]+);)'
""" Match an HTML entity (`&#38;` (decimal) or `&#x26;` (hex) or `&amp;` (named)). """

LINE_BREAK_RE = r'  \n'
""" Match two spaces at end of line. """


def dequote(string: str) -> str:
    """Remove quotes from around a string."""
    if ((string.startswith('"') and string.endswith('"')) or
       (string.startswith("'") and string.endswith("'"))):
        return string[1:-1]
    else:
        return string


class EmStrongItem(NamedTuple):
    """Emphasis/strong pattern item."""
    pattern: re.Pattern[str]
    builder: str
    tags: str


# The pattern classes
# -----------------------------------------------------------------------------


class Pattern:  # pragma: no cover
    """
    Base class that inline patterns subclass.

    Inline patterns are handled by means of `Pattern` subclasses, one per regular expression.
    Each pattern object uses a single regular expression and must support the following methods:
    [`getCompiledRegExp`][markdown.inlinepatterns.Pattern.getCompiledRegExp] and
    [`handleMatch`][markdown.inlinepatterns.Pattern.handleMatch].

    All the regular expressions used by `Pattern` subclasses must capture the whole block.  For this
    reason, they all start with `^(.*)` and end with `(.*)!`.  When passing a regular expression on
    class initialization, the `^(.*)` and `(.*)!` are added automatically and the regular expression
    is pre-compiled.

    It is strongly suggested that the newer style [`markdown.inlinepatterns.InlineProcessor`][] that
    use a more efficient and flexible search approach be used instead. However, the older style
    `Pattern` remains for backward compatibility with many existing third-party extensions.

    """

    ANCESTOR_EXCLUDES: Collection[str] = tuple()
    """
    A collection of elements which are undesirable ancestors. The processor will be skipped if it
    would cause the content to be a descendant of one of the listed tag names.
    """

    compiled_re: re.Pattern[str]
    md: Markdown | None

    def __init__(self, pattern: str, md: Markdown | None = None):
        """
        Create an instant of an inline pattern.

        Arguments:
            pattern: A regular expression that matches a pattern.
            md: An optional pointer to the instance of `markdown.Markdown` and is available as
                `self.md` on the class instance.


        """
        self.pattern = pattern
        self.compiled_re = re.compile(r"^(.*?)%s(.*)$" % pattern,
                                      re.DOTALL | re.UNICODE)

        self.md = md

    def getCompiledRegExp(self) -> re.Pattern:
        """ Return a compiled regular expression. """
        return self.compiled_re

    def handleMatch(self, m: re.Match[str]) -> etree.Element | str:
        """Return a ElementTree element from the given match.

        Subclasses should override this method.

        Arguments:
            m: A match object containing a match of the pattern.

        Returns: An ElementTree Element object.

        """
        pass  # pragma: no cover

    def type(self) -> str:
        """ Return class name, to define pattern type """
        return self.__class__.__name__

    def unescape(self, text: str) -> str:
        """ Return unescaped text given text with an inline placeholder. """
        try:
            stash = self.md.treeprocessors['inline'].stashed_nodes
        except KeyError:  # pragma: no cover
            return text

        def get_stash(m):
            id = m.group(1)
            if id in stash:
                value = stash.get(id)
                if isinstance(value, str):
                    return value
                else:
                    # An `etree` Element - return text content only
                    return ''.join(value.itertext())
        return util.INLINE_PLACEHOLDER_RE.sub(get_stash, text)


class InlineProcessor(Pattern):
    """
    Base class that inline processors subclass.

    This is the newer style inline processor that uses a more
    efficient and flexible search approach.

    """

    def __init__(self, pattern: str, md: Markdown | None = None):
        """
        Create an instant of an inline processor.

        Arguments:
            pattern: A regular expression that matches a pattern.
            md: An optional pointer to the instance of `markdown.Markdown` and is available as
                `self.md` on the class instance.

        """
        self.pattern = pattern
        self.compiled_re = re.compile(pattern, re.DOTALL | re.UNICODE)

        # API for Markdown to pass `safe_mode` into instance
        self.safe_mode = False
        self.md = md

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element | str | None, int | None, int | None]:
        """Return a ElementTree element from the given match and the
        start and end index of the matched text.

        If `start` and/or `end` are returned as `None`, it will be
        assumed that the processor did not find a valid region of text.

        Subclasses should override this method.

        Arguments:
            m: A re match object containing a match of the pattern.
            data: The buffer currently under analysis.

        Returns:
            el: The ElementTree element, text or None.
            start: The start of the region that has been matched or None.
            end: The end of the region that has been matched or None.

        """
        pass  # pragma: no cover


class SimpleTextPattern(Pattern):  # pragma: no cover
    """ Return a simple text of `group(2)` of a Pattern. """
    def handleMatch(self, m: re.Match[str]) -> str:
        """ Return string content of `group(2)` of a matching pattern. """
        return m.group(2)


class SimpleTextInlineProcessor(InlineProcessor):
    """ Return a simple text of `group(1)` of a Pattern. """
    def handleMatch(self, m: re.Match[str], data: str) -> tuple[str, int, int]:
        """ Return string content of `group(1)` of a matching pattern. """
        return m.group(1), m.start(0), m.end(0)


class EscapeInlineProcessor(InlineProcessor):
    """ Return an escaped character. """

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[str | None, int, int]:
        """
        If the character matched by `group(1)` of a pattern is in [`ESCAPED_CHARS`][markdown.Markdown.ESCAPED_CHARS]
        then return the integer representing the character's Unicode code point (as returned by [`ord`][]) wrapped
        in [`util.STX`][markdown.util.STX] and [`util.ETX`][markdown.util.ETX].

        If the matched character is not in [`ESCAPED_CHARS`][markdown.Markdown.ESCAPED_CHARS], then return `None`.
        """

        char = m.group(1)
        if char in self.md.ESCAPED_CHARS:
            return '{}{}{}'.format(util.STX, ord(char), util.ETX), m.start(0), m.end(0)
        else:
            return None, m.start(0), m.end(0)


class SimpleTagPattern(Pattern):  # pragma: no cover
    """
    Return element of type `tag` with a text attribute of `group(3)`
    of a Pattern.

    """
    def __init__(self, pattern: str, tag: str):
        """
        Create an instant of an simple tag pattern.

        Arguments:
            pattern: A regular expression that matches a pattern.
            tag: Tag of element.

        """
        Pattern.__init__(self, pattern)
        self.tag = tag
        """ The tag of the rendered element. """

    def handleMatch(self, m: re.Match[str]) -> etree.Element:
        """
        Return [`Element`][xml.etree.ElementTree.Element] of type `tag` with the string in `group(3)` of a
        matching pattern as the Element's text.
        """
        el = etree.Element(self.tag)
        el.text = m.group(3)
        return el


class SimpleTagInlineProcessor(InlineProcessor):
    """
    Return element of type `tag` with a text attribute of `group(2)`
    of a Pattern.

    """
    def __init__(self, pattern: str, tag: str):
        """
        Create an instant of an simple tag processor.

        Arguments:
            pattern: A regular expression that matches a pattern.
            tag: Tag of element.

        """
        InlineProcessor.__init__(self, pattern)
        self.tag = tag
        """ The tag of the rendered element. """

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element, int, int]:  # pragma: no cover
        """
        Return [`Element`][xml.etree.ElementTree.Element] of type `tag` with the string in `group(2)` of a
        matching pattern as the Element's text.
        """
        el = etree.Element(self.tag)
        el.text = m.group(2)
        return el, m.start(0), m.end(0)


class SubstituteTagPattern(SimpleTagPattern):  # pragma: no cover
    """ Return an element of type `tag` with no children. """
    def handleMatch(self, m: re.Match[str]) -> etree.Element:
        """ Return empty [`Element`][xml.etree.ElementTree.Element] of type `tag`. """
        return etree.Element(self.tag)


class SubstituteTagInlineProcessor(SimpleTagInlineProcessor):
    """ Return an element of type `tag` with no children. """
    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element, int, int]:
        """ Return empty [`Element`][xml.etree.ElementTree.Element] of type `tag`. """
        return etree.Element(self.tag), m.start(0), m.end(0)


class BacktickInlineProcessor(InlineProcessor):
    """ Return a `<code>` element containing the escaped matching text. """
    def __init__(self, pattern: str):
        InlineProcessor.__init__(self, pattern)
        self.ESCAPED_BSLASH = '{}{}{}'.format(util.STX, ord('\\'), util.ETX)
        self.tag = 'code'
        """ The tag of the rendered element. """

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element | str, int, int]:
        """
        If the match contains `group(3)` of a pattern, then return a `code`
        [`Element`][xml.etree.ElementTree.Element] which contains HTML escaped text (with
        [`code_escape`][markdown.util.code_escape]) as an [`AtomicString`][markdown.util.AtomicString].

        If the match does not contain `group(3)` then return the text of `group(1)` backslash escaped.

        """
        if m.group(3):
            el = etree.Element(self.tag)
            el.text = util.AtomicString(util.code_escape(m.group(3).strip()))
            return el, m.start(0), m.end(0)
        else:
            return m.group(1).replace('\\\\', self.ESCAPED_BSLASH), m.start(0), m.end(0)


class DoubleTagPattern(SimpleTagPattern):  # pragma: no cover
    """Return a ElementTree element nested in tag2 nested in tag1.

    Useful for strong emphasis etc.

    """
    def handleMatch(self, m: re.Match[str]) -> etree.Element:
        """
        Return [`Element`][xml.etree.ElementTree.Element] in following format:
        `<tag1><tag2>group(3)</tag2>group(4)</tag2>` where `group(4)` is optional.

        """
        tag1, tag2 = self.tag.split(",")
        el1 = etree.Element(tag1)
        el2 = etree.SubElement(el1, tag2)
        el2.text = m.group(3)
        if len(m.groups()) == 5:
            el2.tail = m.group(4)
        return el1


class DoubleTagInlineProcessor(SimpleTagInlineProcessor):
    """Return a ElementTree element nested in tag2 nested in tag1.

    Useful for strong emphasis etc.

    """
    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element, int, int]:  # pragma: no cover
        """
        Return [`Element`][xml.etree.ElementTree.Element] in following format:
        `<tag1><tag2>group(2)</tag2>group(3)</tag2>` where `group(3)` is optional.

        """
        tag1, tag2 = self.tag.split(",")
        el1 = etree.Element(tag1)
        el2 = etree.SubElement(el1, tag2)
        el2.text = m.group(2)
        if len(m.groups()) == 3:
            el2.tail = m.group(3)
        return el1, m.start(0), m.end(0)


class HtmlInlineProcessor(InlineProcessor):
    """ Store raw inline html and return a placeholder. """
    def handleMatch(self, m: re.Match[str], data: str) -> tuple[str, int, int]:
        """ Store the text of `group(1)` of a pattern and return a placeholder string. """
        rawhtml = self.backslash_unescape(self.unescape(m.group(1)))
        place_holder = self.md.htmlStash.store(rawhtml)
        return place_holder, m.start(0), m.end(0)

    def unescape(self, text: str) -> str:
        """ Return unescaped text given text with an inline placeholder. """
        try:
            stash = self.md.treeprocessors['inline'].stashed_nodes
        except KeyError:  # pragma: no cover
            return text

        def get_stash(m: re.Match[str]) -> str:
            id = m.group(1)
            value = stash.get(id)
            if value is not None:
                try:
                    return self.md.serializer(value)
                except Exception:
                    return r'\%s' % value

        return util.INLINE_PLACEHOLDER_RE.sub(get_stash, text)

    def backslash_unescape(self, text: str) -> str:
        """ Return text with backslash escapes undone (backslashes are restored). """
        try:
            RE = self.md.treeprocessors['unescape'].RE
        except KeyError:  # pragma: no cover
            return text

        def _unescape(m: re.Match[str]) -> str:
            return chr(int(m.group(1)))

        return RE.sub(_unescape, text)


class AsteriskProcessor(InlineProcessor):
    """Emphasis processor for handling strong and em matches inside asterisks."""

    PATTERNS = [
        EmStrongItem(re.compile(EM_STRONG_RE, re.DOTALL | re.UNICODE), 'double', 'strong,em'),
        EmStrongItem(re.compile(STRONG_EM_RE, re.DOTALL | re.UNICODE), 'double', 'em,strong'),
        EmStrongItem(re.compile(STRONG_EM3_RE, re.DOTALL | re.UNICODE), 'double2', 'strong,em'),
        EmStrongItem(re.compile(STRONG_RE, re.DOTALL | re.UNICODE), 'single', 'strong'),
        EmStrongItem(re.compile(EMPHASIS_RE, re.DOTALL | re.UNICODE), 'single', 'em')
    ]
    """ The various strong and emphasis patterns handled by this processor. """

    def build_single(self, m: re.Match[str], tag: str, idx: int) -> etree.Element:
        """Return single tag."""
        el1 = etree.Element(tag)
        text = m.group(2)
        self.parse_sub_patterns(text, el1, None, idx)
        return el1

    def build_double(self, m: re.Match[str], tags: str, idx: int) -> etree.Element:
        """Return double tag."""

        tag1, tag2 = tags.split(",")
        el1 = etree.Element(tag1)
        el2 = etree.Element(tag2)
        text = m.group(2)
        self.parse_sub_patterns(text, el2, None, idx)
        el1.append(el2)
        if len(m.groups()) == 3:
            text = m.group(3)
            self.parse_sub_patterns(text, el1, el2, idx)
        return el1

    def build_double2(self, m: re.Match[str], tags: str, idx: int) -> etree.Element:
        """Return double tags (variant 2): `<strong>text <em>text</em></strong>`."""

        tag1, tag2 = tags.split(",")
        el1 = etree.Element(tag1)
        el2 = etree.Element(tag2)
        text = m.group(2)
        self.parse_sub_patterns(text, el1, None, idx)
        text = m.group(3)
        el1.append(el2)
        self.parse_sub_patterns(text, el2, None, idx)
        return el1

    def parse_sub_patterns(
        self, data: str, parent: etree.Element, last: etree.Element | None, idx: int
    ) -> None:
        """
        Parses sub patterns.

        `data`: text to evaluate.

        `parent`: Parent to attach text and sub elements to.

        `last`: Last appended child to parent. Can also be None if parent has no children.

        `idx`: Current pattern index that was used to evaluate the parent.
        """

        offset = 0
        pos = 0

        length = len(data)
        while pos < length:
            # Find the start of potential emphasis or strong tokens
            if self.compiled_re.match(data, pos):
                matched = False
                # See if the we can match an emphasis/strong pattern
                for index, item in enumerate(self.PATTERNS):
                    # Only evaluate patterns that are after what was used on the parent
                    if index <= idx:
                        continue
                    m = item.pattern.match(data, pos)
                    if m:
                        # Append child nodes to parent
                        # Text nodes should be appended to the last
                        # child if present, and if not, it should
                        # be added as the parent's text node.
                        text = data[offset:m.start(0)]
                        if text:
                            if last is not None:
                                last.tail = text
                            else:
                                parent.text = text
                        el = self.build_element(m, item.builder, item.tags, index)
                        parent.append(el)
                        last = el
                        # Move our position past the matched hunk
                        offset = pos = m.end(0)
                        matched = True
                if not matched:
                    # We matched nothing, move on to the next character
                    pos += 1
            else:
                # Increment position as no potential emphasis start was found.
                pos += 1

        # Append any leftover text as a text node.
        text = data[offset:]
        if text:
            if last is not None:
                last.tail = text
            else:
                parent.text = text

    def build_element(self, m: re.Match[str], builder: str, tags: str, index: int) -> etree.Element:
        """Element builder."""

        if builder == 'double2':
            return self.build_double2(m, tags, index)
        elif builder == 'double':
            return self.build_double(m, tags, index)
        else:
            return self.build_single(m, tags, index)

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element | None, int | None, int | None]:
        """Parse patterns."""

        el = None
        start = None
        end = None

        for index, item in enumerate(self.PATTERNS):
            m1 = item.pattern.match(data, m.start(0))
            if m1:
                start = m1.start(0)
                end = m1.end(0)
                el = self.build_element(m1, item.builder, item.tags, index)
                break
        return el, start, end


class UnderscoreProcessor(AsteriskProcessor):
    """Emphasis processor for handling strong and em matches inside underscores."""

    PATTERNS = [
        EmStrongItem(re.compile(EM_STRONG2_RE, re.DOTALL | re.UNICODE), 'double', 'strong,em'),
        EmStrongItem(re.compile(STRONG_EM2_RE, re.DOTALL | re.UNICODE), 'double', 'em,strong'),
        EmStrongItem(re.compile(SMART_STRONG_EM_RE, re.DOTALL | re.UNICODE), 'double2', 'strong,em'),
        EmStrongItem(re.compile(SMART_STRONG_RE, re.DOTALL | re.UNICODE), 'single', 'strong'),
        EmStrongItem(re.compile(SMART_EMPHASIS_RE, re.DOTALL | re.UNICODE), 'single', 'em')
    ]
    """ The various strong and emphasis patterns handled by this processor. """


class LinkInlineProcessor(InlineProcessor):
    """ Return a link element from the given match. """
    RE_LINK = re.compile(r'''\(\s*(?:(<[^<>]*>)\s*(?:('[^']*'|"[^"]*")\s*)?\))?''', re.DOTALL | re.UNICODE)
    RE_TITLE_CLEAN = re.compile(r'\s')

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element | None, int | None, int | None]:
        """ Return an `a` [`Element`][xml.etree.ElementTree.Element] or `(None, None, None)`. """
        text, index, handled = self.getText(data, m.end(0))

        if not handled:
            return None, None, None

        href, title, index, handled = self.getLink(data, index)
        if not handled:
            return None, None, None

        el = etree.Element("a")
        el.text = text

        el.set("href", href)

        if title is not None:
            el.set("title", title)

        return el, m.start(0), index

    def getLink(self, data: str, index: int) -> tuple[str, str | None, int, bool]:
        """Parse data between `()` of `[Text]()` allowing recursive `()`. """

        href = ''
        title: str | None = None
        handled = False

        m = self.RE_LINK.match(data, pos=index)
        if m and m.group(1):
            # Matches [Text](<link> "title")
            href = m.group(1)[1:-1].strip()
            if m.group(2):
                title = m.group(2)[1:-1]
            index = m.end(0)
            handled = True
        elif m:
            # Track bracket nesting and index in string
            bracket_count = 1
            backtrack_count = 1
            start_index = m.end()
            index = start_index
            last_bracket = -1

            # Primary (first found) quote tracking.
            quote: str | None = None
            start_quote = -1
            exit_quote = -1
            ignore_matches = False

            # Secondary (second found) quote tracking.
            alt_quote = None
            start_alt_quote = -1
            exit_alt_quote = -1

            # Track last character
            last = ''

            for pos in range(index, len(data)):
                c = data[pos]
                if c == '(':
                    # Count nested (
                    # Don't increment the bracket count if we are sure we're in a title.
                    if not ignore_matches:
                        bracket_count += 1
                    elif backtrack_count > 0:
                        backtrack_count -= 1
                elif c == ')':
                    # Match nested ) to (
                    # Don't decrement if we are sure we are in a title that is unclosed.
                    if ((exit_quote != -1 and quote == last) or (exit_alt_quote != -1 and alt_quote == last)):
                        bracket_count = 0
                    elif not ignore_matches:
                        bracket_count -= 1
                    elif backtrack_count > 0:
                        backtrack_count -= 1
                        # We've found our backup end location if the title doesn't resolve.
                        if backtrack_count == 0:
                            last_bracket = index + 1

                elif c in ("'", '"'):
                    # Quote has started
                    if not quote:
                        # We'll assume we are now in a title.
                        # Brackets are quoted, so no need to match them (except for the final one).
                        ignore_matches = True
                        backtrack_count = bracket_count
                        bracket_count = 1
                        start_quote = index + 1
                        quote = c
                    # Secondary quote (in case the first doesn't resolve): [text](link'"title")
                    elif c != quote and not alt_quote:
                        start_alt_quote = index + 1
                        alt_quote = c
                    # Update primary quote match
                    elif c == quote:
                        exit_quote = index + 1
                    # Update secondary quote match
                    elif alt_quote and c == alt_quote:
                        exit_alt_quote = index + 1

                index += 1

                # Link is closed, so let's break out of the loop
                if bracket_count == 0:
                    # Get the title if we closed a title string right before link closed
                    if exit_quote >= 0 and quote == last:
                        href = data[start_index:start_quote - 1]
                        title = ''.join(data[start_quote:exit_quote - 1])
                    elif exit_alt_quote >= 0 and alt_quote == last:
                        href = data[start_index:start_alt_quote - 1]
                        title = ''.join(data[start_alt_quote:exit_alt_quote - 1])
                    else:
                        href = data[start_index:index - 1]
                    break

                if c != ' ':
                    last = c

            # We have a scenario: `[test](link"notitle)`
            # When we enter a string, we stop tracking bracket resolution in the main counter,
            # but we do keep a backup counter up until we discover where we might resolve all brackets
            # if the title string fails to resolve.
            if bracket_count != 0 and backtrack_count == 0:
                href = data[start_index:last_bracket - 1]
                index = last_bracket
                bracket_count = 0

            handled = bracket_count == 0

        if title is not None:
            title = self.RE_TITLE_CLEAN.sub(' ', dequote(self.unescape(title.strip())))

        href = self.unescape(href).strip()

        return href, title, index, handled

    def getText(self, data: str, index: int) -> tuple[str, int, bool]:
        """Parse the content between `[]` of the start of an image or link
        resolving nested square brackets.

        """
        bracket_count = 1
        text = []
        for pos in range(index, len(data)):
            c = data[pos]
            if c == ']':
                bracket_count -= 1
            elif c == '[':
                bracket_count += 1
            index += 1
            if bracket_count == 0:
                break
            text.append(c)
        return ''.join(text), index, bracket_count == 0


class ImageInlineProcessor(LinkInlineProcessor):
    """ Return a `img` element from the given match. """

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element | None, int | None, int | None]:
        """ Return an `img` [`Element`][xml.etree.ElementTree.Element] or `(None, None, None)`. """
        text, index, handled = self.getText(data, m.end(0))
        if not handled:
            return None, None, None

        src, title, index, handled = self.getLink(data, index)
        if not handled:
            return None, None, None

        el = etree.Element("img")

        el.set("src", src)

        if title is not None:
            el.set("title", title)

        el.set('alt', self.unescape(text))
        return el, m.start(0), index


class ReferenceInlineProcessor(LinkInlineProcessor):
    """ Match to a stored reference and return link element. """
    NEWLINE_CLEANUP_RE = re.compile(r'\s+', re.MULTILINE)

    RE_LINK = re.compile(r'\s?\[([^\]]*)\]', re.DOTALL | re.UNICODE)

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element | None, int | None, int | None]:
        """
        Return [`Element`][xml.etree.ElementTree.Element] returned by `makeTag` method or `(None, None, None)`.

        """
        text, index, handled = self.getText(data, m.end(0))
        if not handled:
            return None, None, None

        id, end, handled = self.evalId(data, index, text)
        if not handled:
            return None, None, None

        # Clean up line breaks in id
        id = self.NEWLINE_CLEANUP_RE.sub(' ', id)
        if id not in self.md.references:  # ignore undefined refs
            return None, m.start(0), end

        href, title = self.md.references[id]

        return self.makeTag(href, title, text), m.start(0), end

    def evalId(self, data: str, index: int, text: str) -> tuple[str | None, int, bool]:
        """
        Evaluate the id portion of `[ref][id]`.

        If `[ref][]` use `[ref]`.
        """
        m = self.RE_LINK.match(data, pos=index)
        if not m:
            return None, index, False
        else:
            id = m.group(1).lower()
            end = m.end(0)
            if not id:
                id = text.lower()
        return id, end, True

    def makeTag(self, href: str, title: str, text: str) -> etree.Element:
        """ Return an `a` [`Element`][xml.etree.ElementTree.Element]. """
        el = etree.Element('a')

        el.set('href', href)
        if title:
            el.set('title', title)

        el.text = text
        return el


class ShortReferenceInlineProcessor(ReferenceInlineProcessor):
    """Short form of reference: `[google]`. """
    def evalId(self, data: str, index: int, text: str) -> tuple[str, int, bool]:
        """Evaluate the id of `[ref]`.  """

        return text.lower(), index, True


class ImageReferenceInlineProcessor(ReferenceInlineProcessor):
    """ Match to a stored reference and return `img` element. """
    def makeTag(self, href: str, title: str, text: str) -> etree.Element:
        """ Return an `img` [`Element`][xml.etree.ElementTree.Element]. """
        el = etree.Element("img")
        el.set("src", href)
        if title:
            el.set("title", title)
        el.set("alt", self.unescape(text))
        return el


class ShortImageReferenceInlineProcessor(ImageReferenceInlineProcessor):
    """ Short form of image reference: `![ref]`. """
    def evalId(self, data: str, index: int, text: str) -> tuple[str, int, bool]:
        """Evaluate the id of `[ref]`.  """

        return text.lower(), index, True


class AutolinkInlineProcessor(InlineProcessor):
    """ Return a link Element given an auto-link (`<http://example/com>`). """
    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element, int, int]:
        """ Return an `a` [`Element`][xml.etree.ElementTree.Element] of `group(1)`. """
        el = etree.Element("a")
        el.set('href', self.unescape(m.group(1)))
        el.text = util.AtomicString(m.group(1))
        return el, m.start(0), m.end(0)


class AutomailInlineProcessor(InlineProcessor):
    """
    Return a `mailto` link Element given an auto-mail link (`<foo@example.com>`).
    """
    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element, int, int]:
        """ Return an [`Element`][xml.etree.ElementTree.Element] containing a `mailto` link  of `group(1)`. """
        el = etree.Element('a')
        email = self.unescape(m.group(1))
        if email.startswith("mailto:"):
            email = email[len("mailto:"):]

        def codepoint2name(code: int) -> str:
            """Return entity definition by code, or the code if not defined."""
            entity = entities.codepoint2name.get(code)
            if entity:
                return "{}{};".format(util.AMP_SUBSTITUTE, entity)
            else:
                return "%s#%d;" % (util.AMP_SUBSTITUTE, code)

        letters = [codepoint2name(ord(letter)) for letter in email]
        el.text = util.AtomicString(''.join(letters))

        mailto = "mailto:" + email
        mailto = "".join([util.AMP_SUBSTITUTE + '#%d;' %
                          ord(letter) for letter in mailto])
        el.set('href', mailto)
        return el, m.start(0), m.end(0)
