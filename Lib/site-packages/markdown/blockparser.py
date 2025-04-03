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
The block parser handles basic parsing of Markdown blocks.  It doesn't concern
itself with inline elements such as `**bold**` or `*italics*`, but rather just
catches blocks, lists, quotes, etc.

The `BlockParser` is made up of a bunch of `BlockProcessors`, each handling a
different type of block. Extensions may add/replace/remove `BlockProcessors`
as they need to alter how Markdown blocks are parsed.
"""

from __future__ import annotations

import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Iterable, Any
from . import util

if TYPE_CHECKING:  # pragma: no cover
    from markdown import Markdown
    from .blockprocessors import BlockProcessor


class State(list):
    """ Track the current and nested state of the parser.

    This utility class is used to track the state of the `BlockParser` and
    support multiple levels if nesting. It's just a simple API wrapped around
    a list. Each time a state is set, that state is appended to the end of the
    list. Each time a state is reset, that state is removed from the end of
    the list.

    Therefore, each time a state is set for a nested block, that state must be
    reset when we back out of that level of nesting or the state could be
    corrupted.

    While all the methods of a list object are available, only the three
    defined below need be used.

    """

    def set(self, state: Any):
        """ Set a new state. """
        self.append(state)

    def reset(self) -> None:
        """ Step back one step in nested state. """
        self.pop()

    def isstate(self, state: Any) -> bool:
        """ Test that top (current) level is of given state. """
        if len(self):
            return self[-1] == state
        else:
            return False


class BlockParser:
    """ Parse Markdown blocks into an `ElementTree` object.

    A wrapper class that stitches the various `BlockProcessors` together,
    looping through them and creating an `ElementTree` object.

    """

    def __init__(self, md: Markdown):
        """ Initialize the block parser.

        Arguments:
            md: A Markdown instance.

        Attributes:
            BlockParser.md (Markdown): A Markdown instance.
            BlockParser.state (State): Tracks the nesting level of current location in document being parsed.
            BlockParser.blockprocessors (util.Registry): A collection of
                [`blockprocessors`][markdown.blockprocessors].

        """
        self.blockprocessors: util.Registry[BlockProcessor] = util.Registry()
        self.state = State()
        self.md = md

    def parseDocument(self, lines: Iterable[str]) -> etree.ElementTree:
        """ Parse a Markdown document into an `ElementTree`.

        Given a list of lines, an `ElementTree` object (not just a parent
        `Element`) is created and the root element is passed to the parser
        as the parent. The `ElementTree` object is returned.

        This should only be called on an entire document, not pieces.

        Arguments:
            lines: A list of lines (strings).

        Returns:
            An element tree.
        """
        # Create an `ElementTree` from the lines
        self.root = etree.Element(self.md.doc_tag)
        self.parseChunk(self.root, '\n'.join(lines))
        return etree.ElementTree(self.root)

    def parseChunk(self, parent: etree.Element, text: str) -> None:
        """ Parse a chunk of Markdown text and attach to given `etree` node.

        While the `text` argument is generally assumed to contain multiple
        blocks which will be split on blank lines, it could contain only one
        block. Generally, this method would be called by extensions when
        block parsing is required.

        The `parent` `etree` Element passed in is altered in place.
        Nothing is returned.

        Arguments:
            parent: The parent element.
            text: The text to parse.

        """
        self.parseBlocks(parent, text.split('\n\n'))

    def parseBlocks(self, parent: etree.Element, blocks: list[str]) -> None:
        """ Process blocks of Markdown text and attach to given `etree` node.

        Given a list of `blocks`, each `blockprocessor` is stepped through
        until there are no blocks left. While an extension could potentially
        call this method directly, it's generally expected to be used
        internally.

        This is a public method as an extension may need to add/alter
        additional `BlockProcessors` which call this method to recursively
        parse a nested block.

        Arguments:
            parent: The parent element.
            blocks: The blocks of text to parse.

        """
        while blocks:
            for processor in self.blockprocessors:
                if processor.test(parent, blocks[0]):
                    if processor.run(parent, blocks) is not False:
                        # run returns True or None
                        break
