# Python Markdown

# A Python implementation of John Gruber's Markdown.

# - Documentation: https://python-markdown.github.io/
# - GitHub: https://github.com/Python-Markdown/markdown/
# - PyPI: https://pypi.org/project/Markdown/

# Started by Manfred Stienstra (http://www.dwerg.net/).
# Maintained for a few years by Yuri Takhteyev (http://www.freewisdom.org).
# Currently maintained by Waylan Limberg (https://github.com/waylan),
# Dmitry Shachnev (https://github.com/mitya57) and Isaac Muse (https://github.com/facelessuser).

# - Copyright 2007-2023 The Python Markdown Project (v. 1.7 and later)
# - Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
# - Copyright 2004 Manfred Stienstra (the original version)

# License: BSD (see LICENSE.md for details).

"""
Python-Markdown provides two public functions ([`markdown.markdown`][] and [`markdown.markdownFromFile`][])
both of which wrap the public class [`markdown.Markdown`][]. All submodules support these public functions
and class and/or provide extension support.

Modules:
    core: Core functionality.
    preprocessors: Pre-processors.
    blockparser: Core Markdown block parser.
    blockprocessors: Block processors.
    treeprocessors: Tree processors.
    inlinepatterns: Inline patterns.
    postprocessors: Post-processors.
    serializers: Serializers.
    util: Utility functions.
    htmlparser: HTML parser.
    test_tools: Testing utilities.
    extensions: Markdown extensions.
"""

from __future__ import annotations

from .core import Markdown, markdown, markdownFromFile
from .__meta__ import __version__, __version_info__  # noqa

# For backward compatibility as some extensions expect it...
from .extensions import Extension  # noqa

__all__ = ['Markdown', 'markdown', 'markdownFromFile']
