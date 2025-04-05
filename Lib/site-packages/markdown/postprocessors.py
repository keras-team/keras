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

Post-processors run on the text of the entire document after is has been serialized into a string.
Postprocessors should be used to work with the text just before output. Usually, they are used add
back sections that were extracted in a preprocessor, fix up outgoing encodings, or wrap the whole
document.

"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any
from . import util
import re

if TYPE_CHECKING:  # pragma: no cover
    from markdown import Markdown


def build_postprocessors(md: Markdown, **kwargs: Any) -> util.Registry[Postprocessor]:
    """ Build the default postprocessors for Markdown. """
    postprocessors = util.Registry()
    postprocessors.register(RawHtmlPostprocessor(md), 'raw_html', 30)
    postprocessors.register(AndSubstitutePostprocessor(), 'amp_substitute', 20)
    return postprocessors


class Postprocessor(util.Processor):
    """
    Postprocessors are run after the ElementTree it converted back into text.

    Each Postprocessor implements a `run` method that takes a pointer to a
    text string, modifies it as necessary and returns a text string.

    Postprocessors must extend `Postprocessor`.

    """

    def run(self, text: str) -> str:
        """
        Subclasses of `Postprocessor` should implement a `run` method, which
        takes the html document as a single text string and returns a
        (possibly modified) string.

        """
        pass  # pragma: no cover


class RawHtmlPostprocessor(Postprocessor):
    """ Restore raw html to the document. """

    BLOCK_LEVEL_REGEX = re.compile(r'^\<\/?([^ >]+)')

    def run(self, text: str) -> str:
        """ Iterate over html stash and restore html. """
        replacements = OrderedDict()
        for i in range(self.md.htmlStash.html_counter):
            html = self.stash_to_string(self.md.htmlStash.rawHtmlBlocks[i])
            if self.isblocklevel(html):
                replacements["<p>{}</p>".format(
                    self.md.htmlStash.get_placeholder(i))] = html
            replacements[self.md.htmlStash.get_placeholder(i)] = html

        def substitute_match(m: re.Match[str]) -> str:
            key = m.group(0)

            if key not in replacements:
                if key[3:-4] in replacements:
                    return f'<p>{ replacements[key[3:-4]] }</p>'
                else:
                    return key

            return replacements[key]

        if replacements:
            base_placeholder = util.HTML_PLACEHOLDER % r'([0-9]+)'
            pattern = re.compile(f'<p>{ base_placeholder }</p>|{ base_placeholder }')
            processed_text = pattern.sub(substitute_match, text)
        else:
            return text

        if processed_text == text:
            return processed_text
        else:
            return self.run(processed_text)

    def isblocklevel(self, html: str) -> bool:
        """ Check is block of HTML is block-level. """
        m = self.BLOCK_LEVEL_REGEX.match(html)
        if m:
            if m.group(1)[0] in ('!', '?', '@', '%'):
                # Comment, PHP etc...
                return True
            return self.md.is_block_level(m.group(1))
        return False

    def stash_to_string(self, text: str) -> str:
        """ Convert a stashed object to a string. """
        return str(text)


class AndSubstitutePostprocessor(Postprocessor):
    """ Restore valid entities """

    def run(self, text: str) -> str:
        text = text.replace(util.AMP_SUBSTITUTE, "&")
        return text


@util.deprecated(
    "This class is deprecated and will be removed in the future; "
    "use [`UnescapeTreeprocessor`][markdown.treeprocessors.UnescapeTreeprocessor] instead."
)
class UnescapePostprocessor(Postprocessor):
    """ Restore escaped chars. """

    RE = re.compile(r'{}(\d+){}'.format(util.STX, util.ETX))

    def unescape(self, m: re.Match[str]) -> str:
        return chr(int(m.group(1)))

    def run(self, text: str) -> str:
        return self.RE.sub(self.unescape, text)
