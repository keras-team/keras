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
This module contains various contacts, classes and functions which get referenced and used
throughout the code base.
"""

from __future__ import annotations

import re
import sys
import warnings
from functools import wraps, lru_cache
from itertools import count
from typing import TYPE_CHECKING, Generic, Iterator, NamedTuple, TypeVar, TypedDict, overload

if TYPE_CHECKING:  # pragma: no cover
    from markdown import Markdown
    import xml.etree.ElementTree as etree

_T = TypeVar('_T')


"""
Constants you might want to modify
-----------------------------------------------------------------------------
"""


BLOCK_LEVEL_ELEMENTS: list[str] = [
    # Elements which are invalid to wrap in a `<p>` tag.
    # See https://w3c.github.io/html/grouping-content.html#the-p-element
    'address', 'article', 'aside', 'blockquote', 'details', 'div', 'dl',
    'fieldset', 'figcaption', 'figure', 'footer', 'form', 'h1', 'h2', 'h3',
    'h4', 'h5', 'h6', 'header', 'hgroup', 'hr', 'main', 'menu', 'nav', 'ol',
    'p', 'pre', 'section', 'table', 'ul',
    # Other elements which Markdown should not be mucking up the contents of.
    'canvas', 'colgroup', 'dd', 'body', 'dt', 'group', 'html', 'iframe', 'li', 'legend',
    'math', 'map', 'noscript', 'output', 'object', 'option', 'progress', 'script',
    'style', 'summary', 'tbody', 'td', 'textarea', 'tfoot', 'th', 'thead', 'tr', 'video'
]
"""
List of HTML tags which get treated as block-level elements. Same as the `block_level_elements`
attribute of the [`Markdown`][markdown.Markdown] class. Generally one should use the
attribute on the class. This remains for compatibility with older extensions.
"""

# Placeholders
STX = '\u0002'
""" "Start of Text" marker for placeholder templates. """
ETX = '\u0003'
""" "End of Text" marker for placeholder templates. """
INLINE_PLACEHOLDER_PREFIX = STX+"klzzwxh:"
""" Prefix for inline placeholder template. """
INLINE_PLACEHOLDER = INLINE_PLACEHOLDER_PREFIX + "%s" + ETX
""" Placeholder template for stashed inline text. """
INLINE_PLACEHOLDER_RE = re.compile(INLINE_PLACEHOLDER % r'([0-9]+)')
""" Regular Expression which matches inline placeholders. """
AMP_SUBSTITUTE = STX+"amp"+ETX
""" Placeholder template for HTML entities. """
HTML_PLACEHOLDER = STX + "wzxhzdk:%s" + ETX
""" Placeholder template for raw HTML. """
HTML_PLACEHOLDER_RE = re.compile(HTML_PLACEHOLDER % r'([0-9]+)')
""" Regular expression which matches HTML placeholders. """
TAG_PLACEHOLDER = STX + "hzzhzkh:%s" + ETX
""" Placeholder template for tags. """


# Constants you probably do not need to change
# -----------------------------------------------------------------------------

RTL_BIDI_RANGES = (
    ('\u0590', '\u07FF'),
    # Hebrew (0590-05FF), Arabic (0600-06FF),
    # Syriac (0700-074F), Arabic supplement (0750-077F),
    # Thaana (0780-07BF), Nko (07C0-07FF).
    ('\u2D30', '\u2D7F')  # Tifinagh
)


# AUXILIARY GLOBAL FUNCTIONS
# =============================================================================


@lru_cache(maxsize=None)
def get_installed_extensions():
    """ Return all entry_points in the `markdown.extensions` group. """
    if sys.version_info >= (3, 10):
        from importlib import metadata
    else:  # `<PY310` use backport
        import importlib_metadata as metadata
    # Only load extension entry_points once.
    return metadata.entry_points(group='markdown.extensions')


def deprecated(message: str, stacklevel: int = 2):
    """
    Raise a [`DeprecationWarning`][] when wrapped function/method is called.

    Usage:

    ```python
    @deprecated("This method will be removed in version X; use Y instead.")
    def some_method():
        pass
    ```
    """
    def wrapper(func):
        @wraps(func)
        def deprecated_func(*args, **kwargs):
            warnings.warn(
                f"'{func.__name__}' is deprecated. {message}",
                category=DeprecationWarning,
                stacklevel=stacklevel
            )
            return func(*args, **kwargs)
        return deprecated_func
    return wrapper


def parseBoolValue(value: str | None, fail_on_errors: bool = True, preserve_none: bool = False) -> bool | None:
    """Parses a string representing a boolean value. If parsing was successful,
       returns `True` or `False`. If `preserve_none=True`, returns `True`, `False`,
       or `None`. If parsing was not successful, raises `ValueError`, or, if
       `fail_on_errors=False`, returns `None`."""
    if not isinstance(value, str):
        if preserve_none and value is None:
            return value
        return bool(value)
    elif preserve_none and value.lower() == 'none':
        return None
    elif value.lower() in ('true', 'yes', 'y', 'on', '1'):
        return True
    elif value.lower() in ('false', 'no', 'n', 'off', '0', 'none'):
        return False
    elif fail_on_errors:
        raise ValueError('Cannot parse bool value: %r' % value)


def code_escape(text: str) -> str:
    """HTML escape a string of code."""
    if "&" in text:
        text = text.replace("&", "&amp;")
    if "<" in text:
        text = text.replace("<", "&lt;")
    if ">" in text:
        text = text.replace(">", "&gt;")
    return text


def _get_stack_depth(size: int = 2) -> int:
    """Get current stack depth, performantly.
    """
    frame = sys._getframe(size)

    for size in count(size):
        frame = frame.f_back
        if not frame:
            return size


def nearing_recursion_limit() -> bool:
    """Return true if current stack depth is within 100 of maximum limit."""
    return sys.getrecursionlimit() - _get_stack_depth() < 100


# MISC AUXILIARY CLASSES
# =============================================================================


class AtomicString(str):
    """A string which should not be further processed."""
    pass


class Processor:
    """ The base class for all processors.

    Attributes:
        Processor.md: The `Markdown` instance passed in an initialization.

    Arguments:
        md: The `Markdown` instance this processor is a part of.

    """
    def __init__(self, md: Markdown | None = None):
        self.md = md


if TYPE_CHECKING:  # pragma: no cover
    class TagData(TypedDict):
        tag: str
        attrs: dict[str, str]
        left_index: int
        right_index: int


class HtmlStash:
    """
    This class is used for stashing HTML objects that we extract
    in the beginning and replace with place-holders.
    """

    def __init__(self):
        """ Create an `HtmlStash`. """
        self.html_counter = 0  # for counting inline html segments
        self.rawHtmlBlocks: list[str | etree.Element] = []
        self.tag_counter = 0
        self.tag_data: list[TagData] = []  # list of dictionaries in the order tags appear

    def store(self, html: str | etree.Element) -> str:
        """
        Saves an HTML segment for later reinsertion.  Returns a
        placeholder string that needs to be inserted into the
        document.

        Keyword arguments:
            html: An html segment.

        Returns:
            A placeholder string.

        """
        self.rawHtmlBlocks.append(html)
        placeholder = self.get_placeholder(self.html_counter)
        self.html_counter += 1
        return placeholder

    def reset(self) -> None:
        """ Clear the stash. """
        self.html_counter = 0
        self.rawHtmlBlocks = []

    def get_placeholder(self, key: int) -> str:
        return HTML_PLACEHOLDER % key

    def store_tag(self, tag: str, attrs: dict[str, str], left_index: int, right_index: int) -> str:
        """Store tag data and return a placeholder."""
        self.tag_data.append({'tag': tag, 'attrs': attrs,
                              'left_index': left_index,
                              'right_index': right_index})
        placeholder = TAG_PLACEHOLDER % str(self.tag_counter)
        self.tag_counter += 1  # equal to the tag's index in `self.tag_data`
        return placeholder


# Used internally by `Registry` for each item in its sorted list.
# Provides an easier to read API when editing the code later.
# For example, `item.name` is more clear than `item[0]`.
class _PriorityItem(NamedTuple):
    name: str
    priority: float


class Registry(Generic[_T]):
    """
    A priority sorted registry.

    A `Registry` instance provides two public methods to alter the data of the
    registry: `register` and `deregister`. Use `register` to add items and
    `deregister` to remove items. See each method for specifics.

    When registering an item, a "name" and a "priority" must be provided. All
    items are automatically sorted by "priority" from highest to lowest. The
    "name" is used to remove ("deregister") and get items.

    A `Registry` instance it like a list (which maintains order) when reading
    data. You may iterate over the items, get an item and get a count (length)
    of all items. You may also check that the registry contains an item.

    When getting an item you may use either the index of the item or the
    string-based "name". For example:

        registry = Registry()
        registry.register(SomeItem(), 'itemname', 20)
        # Get the item by index
        item = registry[0]
        # Get the item by name
        item = registry['itemname']

    When checking that the registry contains an item, you may use either the
    string-based "name", or a reference to the actual item. For example:

        someitem = SomeItem()
        registry.register(someitem, 'itemname', 20)
        # Contains the name
        assert 'itemname' in registry
        # Contains the item instance
        assert someitem in registry

    The method `get_index_for_name` is also available to obtain the index of
    an item using that item's assigned "name".
    """

    def __init__(self):
        self._data: dict[str, _T] = {}
        self._priority: list[_PriorityItem] = []
        self._is_sorted = False

    def __contains__(self, item: str | _T) -> bool:
        if isinstance(item, str):
            # Check if an item exists by this name.
            return item in self._data.keys()
        # Check if this instance exists.
        return item in self._data.values()

    def __iter__(self) -> Iterator[_T]:
        self._sort()
        return iter([self._data[k] for k, p in self._priority])

    @overload
    def __getitem__(self, key: str | int) -> _T:  # pragma: no cover
        ...

    @overload
    def __getitem__(self, key: slice) -> Registry[_T]:  # pragma: no cover
        ...

    def __getitem__(self, key: str | int | slice) -> _T | Registry[_T]:
        self._sort()
        if isinstance(key, slice):
            data: Registry[_T] = Registry()
            for k, p in self._priority[key]:
                data.register(self._data[k], k, p)
            return data
        if isinstance(key, int):
            return self._data[self._priority[key].name]
        return self._data[key]

    def __len__(self) -> int:
        return len(self._priority)

    def __repr__(self):
        return '<{}({})>'.format(self.__class__.__name__, list(self))

    def get_index_for_name(self, name: str) -> int:
        """
        Return the index of the given name.
        """
        if name in self:
            self._sort()
            return self._priority.index(
                [x for x in self._priority if x.name == name][0]
            )
        raise ValueError('No item named "{}" exists.'.format(name))

    def register(self, item: _T, name: str, priority: float) -> None:
        """
        Add an item to the registry with the given name and priority.

        Arguments:
            item: The item being registered.
            name: A string used to reference the item.
            priority: An integer or float used to sort against all items.

        If an item is registered with a "name" which already exists, the
        existing item is replaced with the new item. Treat carefully as the
        old item is lost with no way to recover it. The new item will be
        sorted according to its priority and will **not** retain the position
        of the old item.
        """
        if name in self:
            # Remove existing item of same name first
            self.deregister(name)
        self._is_sorted = False
        self._data[name] = item
        self._priority.append(_PriorityItem(name, priority))

    def deregister(self, name: str, strict: bool = True) -> None:
        """
        Remove an item from the registry.

        Set `strict=False` to fail silently. Otherwise a [`ValueError`][] is raised for an unknown `name`.
        """
        try:
            index = self.get_index_for_name(name)
            del self._priority[index]
            del self._data[name]
        except ValueError:
            if strict:
                raise

    def _sort(self) -> None:
        """
        Sort the registry by priority from highest to lowest.

        This method is called internally and should never be explicitly called.
        """
        if not self._is_sorted:
            self._priority.sort(key=lambda item: item.priority, reverse=True)
            self._is_sorted = True
