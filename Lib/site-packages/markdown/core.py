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

from __future__ import annotations

import codecs
import sys
import logging
import importlib
from typing import TYPE_CHECKING, Any, BinaryIO, Callable, ClassVar, Mapping, Sequence
from . import util
from .preprocessors import build_preprocessors
from .blockprocessors import build_block_parser
from .treeprocessors import build_treeprocessors
from .inlinepatterns import build_inlinepatterns
from .postprocessors import build_postprocessors
from .extensions import Extension
from .serializers import to_html_string, to_xhtml_string
from .util import BLOCK_LEVEL_ELEMENTS

if TYPE_CHECKING:  # pragma: no cover
    from xml.etree.ElementTree import Element

__all__ = ['Markdown', 'markdown', 'markdownFromFile']


logger = logging.getLogger('MARKDOWN')


class Markdown:
    """
    A parser which converts Markdown to HTML.

    Attributes:
        Markdown.tab_length (int): The number of spaces which correspond to a single tab. Default: `4`.
        Markdown.ESCAPED_CHARS (list[str]): List of characters which get the backslash escape treatment.
        Markdown.block_level_elements (list[str]): List of HTML tags which get treated as block-level elements.
            See [`markdown.util.BLOCK_LEVEL_ELEMENTS`][] for the full list of elements.
        Markdown.registeredExtensions (list[Extension]): List of extensions which have called
            [`registerExtension`][markdown.Markdown.registerExtension] during setup.
        Markdown.doc_tag (str): Element used to wrap document. Default: `div`.
        Markdown.stripTopLevelTags (bool): Indicates whether the `doc_tag` should be removed. Default: 'True'.
        Markdown.references (dict[str, tuple[str, str]]): A mapping of link references found in a parsed document
             where the key is the reference name and the value is a tuple of the URL and title.
        Markdown.htmlStash (util.HtmlStash): The instance of the `HtmlStash` used by an instance of this class.
        Markdown.output_formats (dict[str, Callable[xml.etree.ElementTree.Element]]): A mapping of known output
             formats by name and their respective serializers. Each serializer must be a callable which accepts an
            [`Element`][xml.etree.ElementTree.Element] and returns a `str`.
        Markdown.output_format (str): The output format set by
            [`set_output_format`][markdown.Markdown.set_output_format].
        Markdown.serializer (Callable[xml.etree.ElementTree.Element]): The serializer set by
            [`set_output_format`][markdown.Markdown.set_output_format].
        Markdown.preprocessors (util.Registry): A collection of [`preprocessors`][markdown.preprocessors].
        Markdown.parser (blockparser.BlockParser): A collection of [`blockprocessors`][markdown.blockprocessors].
        Markdown.inlinePatterns (util.Registry): A collection of [`inlinepatterns`][markdown.inlinepatterns].
        Markdown.treeprocessors (util.Registry): A collection of [`treeprocessors`][markdown.treeprocessors].
        Markdown.postprocessors (util.Registry): A collection of [`postprocessors`][markdown.postprocessors].

    """

    doc_tag = "div"     # Element used to wrap document - later removed

    output_formats: ClassVar[dict[str, Callable[[Element], str]]] = {
        'html':   to_html_string,
        'xhtml':  to_xhtml_string,
    }
    """
    A mapping of known output formats by name and their respective serializers. Each serializer must be a
    callable which accepts an [`Element`][xml.etree.ElementTree.Element] and returns a `str`.
    """

    def __init__(self, **kwargs):
        """
        Creates a new Markdown instance.

        Keyword Arguments:
            extensions (list[Extension | str]): A list of extensions.

                If an item is an instance of a subclass of [`markdown.extensions.Extension`][],
                the instance will be used as-is. If an item is of type `str`, it is passed
                to [`build_extension`][markdown.Markdown.build_extension] with its corresponding
                `extension_configs` and the returned instance  of [`markdown.extensions.Extension`][]
                is used.
            extension_configs (dict[str, dict[str, Any]]): Configuration settings for extensions.
            output_format (str): Format of output. Supported formats are:

                * `xhtml`: Outputs XHTML style tags. Default.
                * `html`: Outputs HTML style tags.
            tab_length (int): Length of tabs in the source. Default: `4`

        """

        self.tab_length: int = kwargs.get('tab_length', 4)

        self.ESCAPED_CHARS: list[str] = [
            '\\', '`', '*', '_', '{', '}', '[', ']', '(', ')', '>', '#', '+', '-', '.', '!'
        ]
        """ List of characters which get the backslash escape treatment. """

        self.block_level_elements: list[str] = BLOCK_LEVEL_ELEMENTS.copy()

        self.registeredExtensions: list[Extension] = []
        self.docType = ""  # TODO: Maybe delete this. It does not appear to be used anymore.
        self.stripTopLevelTags: bool = True

        self.build_parser()

        self.references: dict[str, tuple[str, str]] = {}
        self.htmlStash: util.HtmlStash = util.HtmlStash()
        self.registerExtensions(extensions=kwargs.get('extensions', []),
                                configs=kwargs.get('extension_configs', {}))
        self.set_output_format(kwargs.get('output_format', 'xhtml'))
        self.reset()

    def build_parser(self) -> Markdown:
        """
        Build the parser from the various parts.

        Assigns a value to each of the following attributes on the class instance:

        * **`Markdown.preprocessors`** ([`Registry`][markdown.util.Registry]) -- A collection of
          [`preprocessors`][markdown.preprocessors].
        * **`Markdown.parser`** ([`BlockParser`][markdown.blockparser.BlockParser]) -- A collection of
          [`blockprocessors`][markdown.blockprocessors].
        * **`Markdown.inlinePatterns`** ([`Registry`][markdown.util.Registry]) -- A collection of
          [`inlinepatterns`][markdown.inlinepatterns].
        * **`Markdown.treeprocessors`** ([`Registry`][markdown.util.Registry]) -- A collection of
          [`treeprocessors`][markdown.treeprocessors].
        * **`Markdown.postprocessors`** ([`Registry`][markdown.util.Registry]) -- A collection of
          [`postprocessors`][markdown.postprocessors].

        This method could be redefined in a subclass to build a custom parser which is made up of a different
        combination of processors and patterns.

        """
        self.preprocessors = build_preprocessors(self)
        self.parser = build_block_parser(self)
        self.inlinePatterns = build_inlinepatterns(self)
        self.treeprocessors = build_treeprocessors(self)
        self.postprocessors = build_postprocessors(self)
        return self

    def registerExtensions(
        self,
        extensions: Sequence[Extension | str],
        configs: Mapping[str, dict[str, Any]]
    ) -> Markdown:
        """
        Load a list of extensions into an instance of the `Markdown` class.

        Arguments:
            extensions (list[Extension | str]): A list of extensions.

                If an item is an instance of a subclass of [`markdown.extensions.Extension`][],
                the instance will be used as-is. If an item is of type `str`, it is passed
                to [`build_extension`][markdown.Markdown.build_extension] with its corresponding `configs` and the
                returned instance  of [`markdown.extensions.Extension`][] is used.
            configs (dict[str, dict[str, Any]]): Configuration settings for extensions.

        """
        for ext in extensions:
            if isinstance(ext, str):
                ext = self.build_extension(ext, configs.get(ext, {}))
            if isinstance(ext, Extension):
                ext.extendMarkdown(self)
                logger.debug(
                    'Successfully loaded extension "%s.%s".'
                    % (ext.__class__.__module__, ext.__class__.__name__)
                )
            elif ext is not None:
                raise TypeError(
                    'Extension "{}.{}" must be of type: "{}.{}"'.format(
                        ext.__class__.__module__, ext.__class__.__name__,
                        Extension.__module__, Extension.__name__
                    )
                )
        return self

    def build_extension(self, ext_name: str, configs: Mapping[str, Any]) -> Extension:
        """
        Build extension from a string name, then return an instance using the given `configs`.

        Arguments:
            ext_name: Name of extension as a string.
            configs: Configuration settings for extension.

        Returns:
            An instance of the extension with the given configuration settings.

        First attempt to load an entry point. The string name must be registered as an entry point in the
        `markdown.extensions` group which points to a subclass of the [`markdown.extensions.Extension`][] class.
        If multiple distributions have registered the same name, the first one found is returned.

        If no entry point is found, assume dot notation (`path.to.module:ClassName`). Load the specified class and
        return an instance. If no class is specified, import the module and call a `makeExtension` function and return
        the [`markdown.extensions.Extension`][] instance returned by that function.
        """
        configs = dict(configs)

        entry_points = [ep for ep in util.get_installed_extensions() if ep.name == ext_name]
        if entry_points:
            ext = entry_points[0].load()
            return ext(**configs)

        # Get class name (if provided): `path.to.module:ClassName`
        ext_name, class_name = ext_name.split(':', 1) if ':' in ext_name else (ext_name, '')

        try:
            module = importlib.import_module(ext_name)
            logger.debug(
                'Successfully imported extension module "%s".' % ext_name
            )
        except ImportError as e:
            message = 'Failed loading extension "%s".' % ext_name
            e.args = (message,) + e.args[1:]
            raise

        if class_name:
            # Load given class name from module.
            return getattr(module, class_name)(**configs)
        else:
            # Expect  `makeExtension()` function to return a class.
            try:
                return module.makeExtension(**configs)
            except AttributeError as e:
                message = e.args[0]
                message = "Failed to initiate extension " \
                          "'%s': %s" % (ext_name, message)
                e.args = (message,) + e.args[1:]
                raise

    def registerExtension(self, extension: Extension) -> Markdown:
        """
        Register an extension as having a resettable state.

        Arguments:
            extension: An instance of the extension to register.

        This should get called once by an extension during setup. A "registered" extension's
        `reset` method is called by [`Markdown.reset()`][markdown.Markdown.reset]. Not all extensions have or need a
        resettable state, and so it should not be assumed that all extensions are "registered."

        """
        self.registeredExtensions.append(extension)
        return self

    def reset(self) -> Markdown:
        """
        Resets all state variables to prepare the parser instance for new input.

        Called once upon creation of a class instance. Should be called manually between calls
        to [`Markdown.convert`][markdown.Markdown.convert].
        """
        self.htmlStash.reset()
        self.references.clear()

        for extension in self.registeredExtensions:
            if hasattr(extension, 'reset'):
                extension.reset()

        return self

    def set_output_format(self, format: str) -> Markdown:
        """
        Set the output format for the class instance.

        Arguments:
            format: Must be a known value in `Markdown.output_formats`.

        """
        self.output_format = format.lower().rstrip('145')  # ignore number
        try:
            self.serializer = self.output_formats[self.output_format]
        except KeyError as e:
            valid_formats = list(self.output_formats.keys())
            valid_formats.sort()
            message = 'Invalid Output Format: "%s". Use one of %s.' \
                % (self.output_format,
                   '"' + '", "'.join(valid_formats) + '"')
            e.args = (message,) + e.args[1:]
            raise
        return self

    # Note: the `tag` argument is type annotated `Any` as ElementTree uses many various objects as tags.
    # As there is no standardization in ElementTree, the type of a given tag is unpredictable.
    def is_block_level(self, tag: Any) -> bool:
        """
        Check if the given `tag` is a block level HTML tag.

        Returns `True` for any string listed in `Markdown.block_level_elements`. A `tag` which is
        not a string always returns `False`.

        """
        if isinstance(tag, str):
            return tag.lower().rstrip('/') in self.block_level_elements
        # Some ElementTree tags are not strings, so return False.
        return False

    def convert(self, source: str) -> str:
        """
        Convert a Markdown string to a string in the specified output format.

        Arguments:
            source: Markdown formatted text as Unicode or ASCII string.

        Returns:
            A string in the specified output format.

        Markdown parsing takes place in five steps:

        1. A bunch of [`preprocessors`][markdown.preprocessors] munge the input text.
        2. A [`BlockParser`][markdown.blockparser.BlockParser] parses the high-level structural elements of the
           pre-processed text into an [`ElementTree`][xml.etree.ElementTree.ElementTree] object.
        3. A bunch of [`treeprocessors`][markdown.treeprocessors] are run against the
           [`ElementTree`][xml.etree.ElementTree.ElementTree] object. One such `treeprocessor`
           ([`markdown.treeprocessors.InlineProcessor`][]) runs [`inlinepatterns`][markdown.inlinepatterns]
           against the [`ElementTree`][xml.etree.ElementTree.ElementTree] object, parsing inline markup.
        4. Some [`postprocessors`][markdown.postprocessors] are run against the text after the
           [`ElementTree`][xml.etree.ElementTree.ElementTree] object has been serialized into text.
        5. The output is returned as a string.

        """

        # Fix up the source text
        if not source.strip():
            return ''  # a blank Unicode string

        try:
            source = str(source)
        except UnicodeDecodeError as e:  # pragma: no cover
            # Customize error message while maintaining original traceback
            e.reason += '. -- Note: Markdown only accepts Unicode input!'
            raise

        # Split into lines and run the line preprocessors.
        self.lines = source.split("\n")
        for prep in self.preprocessors:
            self.lines = prep.run(self.lines)

        # Parse the high-level elements.
        root = self.parser.parseDocument(self.lines).getroot()

        # Run the tree-processors
        for treeprocessor in self.treeprocessors:
            newRoot = treeprocessor.run(root)
            if newRoot is not None:
                root = newRoot

        # Serialize _properly_.  Strip top-level tags.
        output = self.serializer(root)
        if self.stripTopLevelTags:
            try:
                start = output.index(
                    '<%s>' % self.doc_tag) + len(self.doc_tag) + 2
                end = output.rindex('</%s>' % self.doc_tag)
                output = output[start:end].strip()
            except ValueError as e:  # pragma: no cover
                if output.strip().endswith('<%s />' % self.doc_tag):
                    # We have an empty document
                    output = ''
                else:
                    # We have a serious problem
                    raise ValueError('Markdown failed to strip top-level '
                                     'tags. Document=%r' % output.strip()) from e

        # Run the text post-processors
        for pp in self.postprocessors:
            output = pp.run(output)

        return output.strip()

    def convertFile(
        self,
        input: str | BinaryIO | None = None,
        output: str | BinaryIO | None = None,
        encoding: str | None = None,
    ) -> Markdown:
        """
        Converts a Markdown file and returns the HTML as a Unicode string.

        Decodes the file using the provided encoding (defaults to `utf-8`),
        passes the file content to markdown, and outputs the HTML to either
        the provided stream or the file with provided name, using the same
        encoding as the source file. The
        [`xmlcharrefreplace`](https://docs.python.org/3/library/codecs.html#error-handlers)
        error handler is used when encoding the output.

        **Note:** This is the only place that decoding and encoding of Unicode
        takes place in Python-Markdown.  (All other code is Unicode-in /
        Unicode-out.)

        Arguments:
            input: File object or path. Reads from `stdin` if `None`.
            output: File object or path. Writes to `stdout` if `None`.
            encoding: Encoding of input and output files. Defaults to `utf-8`.

        """

        encoding = encoding or "utf-8"

        # Read the source
        if input:
            if isinstance(input, str):
                input_file = codecs.open(input, mode="r", encoding=encoding)
            else:
                input_file = codecs.getreader(encoding)(input)
            text = input_file.read()
            input_file.close()
        else:
            text = sys.stdin.read()

        text = text.lstrip('\ufeff')  # remove the byte-order mark

        # Convert
        html = self.convert(text)

        # Write to file or stdout
        if output:
            if isinstance(output, str):
                output_file = codecs.open(output, "w",
                                          encoding=encoding,
                                          errors="xmlcharrefreplace")
                output_file.write(html)
                output_file.close()
            else:
                writer = codecs.getwriter(encoding)
                output_file = writer(output, errors="xmlcharrefreplace")
                output_file.write(html)
                # Don't close here. User may want to write more.
        else:
            # Encode manually and write bytes to stdout.
            html = html.encode(encoding, "xmlcharrefreplace")
            sys.stdout.buffer.write(html)

        return self


"""
EXPORTED FUNCTIONS
=============================================================================

Those are the two functions we really mean to export: `markdown()` and
`markdownFromFile()`.
"""


def markdown(text: str, **kwargs: Any) -> str:
    """
    Convert a markdown string to HTML and return HTML as a Unicode string.

    This is a shortcut function for [`Markdown`][markdown.Markdown] class to cover the most
    basic use case.  It initializes an instance of [`Markdown`][markdown.Markdown], loads the
    necessary extensions and runs the parser on the given text.

    Arguments:
        text: Markdown formatted text as Unicode or ASCII string.

    Keyword arguments:
        **kwargs: Any arguments accepted by the Markdown class.

    Returns:
        A string in the specified output format.

    """
    md = Markdown(**kwargs)
    return md.convert(text)


def markdownFromFile(**kwargs: Any):
    """
    Read Markdown text from a file and write output to a file or a stream.

    This is a shortcut function which initializes an instance of [`Markdown`][markdown.Markdown],
    and calls the [`convertFile`][markdown.Markdown.convertFile] method rather than
    [`convert`][markdown.Markdown.convert].

    Keyword arguments:
        input (str | BinaryIO): A file name or readable object.
        output (str | BinaryIO): A file name or writable object.
        encoding (str): Encoding of input and output.
        **kwargs: Any arguments accepted by the `Markdown` class.

    """
    md = Markdown(**kwargs)
    md.convertFile(kwargs.get('input', None),
                   kwargs.get('output', None),
                   kwargs.get('encoding', None))
