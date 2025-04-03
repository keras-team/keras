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
Markdown accepts an [`Extension`][markdown.extensions.Extension] instance for each extension. Therefore, each extension
must to define a class that extends [`Extension`][markdown.extensions.Extension] and over-rides the
[`extendMarkdown`][markdown.extensions.Extension.extendMarkdown] method. Within this class one can manage configuration
options for their extension and attach the various processors and patterns which make up an extension to the
[`Markdown`][markdown.Markdown] instance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Mapping
from ..util import parseBoolValue

if TYPE_CHECKING:  # pragma: no cover
    from markdown import Markdown


class Extension:
    """ Base class for extensions to subclass. """

    config: Mapping[str, list] = {}
    """
    Default configuration for an extension.

    This attribute is to be defined in a subclass and must be of the following format:

    ``` python
    config = {
        'key': ['value', 'description']
    }
    ```

    Note that [`setConfig`][markdown.extensions.Extension.setConfig] will raise a [`KeyError`][]
    if a default is not set for each option.
    """

    def __init__(self, **kwargs):
        """ Initiate Extension and set up configs. """
        self.setConfigs(kwargs)

    def getConfig(self, key: str, default: Any = '') -> Any:
        """
        Return a single configuration option value.

        Arguments:
            key: The configuration option name.
            default: Default value to return if key is not set.

        Returns:
            Value of stored configuration option.
        """
        if key in self.config:
            return self.config[key][0]
        else:
            return default

    def getConfigs(self) -> dict[str, Any]:
        """
        Return all configuration options.

        Returns:
            All configuration options.
        """
        return {key: self.getConfig(key) for key in self.config.keys()}

    def getConfigInfo(self) -> list[tuple[str, str]]:
        """
        Return descriptions of all configuration options.

        Returns:
            All descriptions of configuration options.
        """
        return [(key, self.config[key][1]) for key in self.config.keys()]

    def setConfig(self, key: str, value: Any) -> None:
        """
        Set a configuration option.

        If the corresponding default value set in [`config`][markdown.extensions.Extension.config]
        is a `bool` value or `None`, then `value` is passed through
        [`parseBoolValue`][markdown.util.parseBoolValue] before being stored.

        Arguments:
            key: Name of configuration option to set.
            value: Value to assign to option.

        Raises:
            KeyError: If `key` is not known.
        """
        if isinstance(self.config[key][0], bool):
            value = parseBoolValue(value)
        if self.config[key][0] is None:
            value = parseBoolValue(value, preserve_none=True)
        self.config[key][0] = value

    def setConfigs(self, items: Mapping[str, Any] | Iterable[tuple[str, Any]]) -> None:
        """
        Loop through a collection of configuration options, passing each to
        [`setConfig`][markdown.extensions.Extension.setConfig].

        Arguments:
            items: Collection of configuration options.

        Raises:
            KeyError: for any unknown key.
        """
        if hasattr(items, 'items'):
            # it's a dict
            items = items.items()
        for key, value in items:
            self.setConfig(key, value)

    def extendMarkdown(self, md: Markdown) -> None:
        """
        Add the various processors and patterns to the Markdown Instance.

        This method must be overridden by every extension.

        Arguments:
            md: The Markdown instance.

        """
        raise NotImplementedError(
            'Extension "%s.%s" must define an "extendMarkdown"'
            'method.' % (self.__class__.__module__, self.__class__.__name__)
        )
