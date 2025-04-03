# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from tensorboard._vendor.bleach.linkifier import (
    DEFAULT_CALLBACKS,
    Linker,
    LinkifyFilter,
)
from tensorboard._vendor.bleach.sanitizer import (
    ALLOWED_ATTRIBUTES,
    ALLOWED_PROTOCOLS,
    ALLOWED_STYLES,
    ALLOWED_TAGS,
    BleachSanitizerFilter,
    Cleaner,
)
from tensorboard._vendor.bleach.version import __version__, VERSION # flake8: noqa

__all__ = ['clean', 'linkify']


def clean(text, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRIBUTES,
          styles=ALLOWED_STYLES, protocols=ALLOWED_PROTOCOLS, strip=False,
          strip_comments=True):
    """Clean an HTML fragment of malicious content and return it

    This function is a security-focused function whose sole purpose is to
    remove malicious content from a string such that it can be displayed as
    content in a web page.

    This function is not designed to use to transform content to be used in
    non-web-page contexts.

    Example::

        import bleach

        better_text = bleach.clean(yucky_text)


    .. Note::

       If you're cleaning a lot of text and passing the same argument values or
       you want more configurability, consider using a
       :py:class:`bleach.sanitizer.Cleaner` instance.

    :arg str text: the text to clean

    :arg list tags: allowed list of tags; defaults to
        ``bleach.sanitizer.ALLOWED_TAGS``

    :arg dict attributes: allowed attributes; can be a callable, list or dict;
        defaults to ``bleach.sanitizer.ALLOWED_ATTRIBUTES``

    :arg list styles: allowed list of css styles; defaults to
        ``bleach.sanitizer.ALLOWED_STYLES``

    :arg list protocols: allowed list of protocols for links; defaults
        to ``bleach.sanitizer.ALLOWED_PROTOCOLS``

    :arg bool strip: whether or not to strip disallowed elements

    :arg bool strip_comments: whether or not to strip HTML comments

    :returns: cleaned text as unicode

    """
    cleaner = Cleaner(
        tags=tags,
        attributes=attributes,
        styles=styles,
        protocols=protocols,
        strip=strip,
        strip_comments=strip_comments,
    )
    return cleaner.clean(text)


def linkify(text, callbacks=DEFAULT_CALLBACKS, skip_tags=None, parse_email=False):
    """Convert URL-like strings in an HTML fragment to links

    This function converts strings that look like URLs, domain names and email
    addresses in text that may be an HTML fragment to links, while preserving:

    1. links already in the string
    2. urls found in attributes
    3. email addresses

    linkify does a best-effort approach and tries to recover from bad
    situations due to crazy text.

    .. Note::

       If you're linking a lot of text and passing the same argument values or
       you want more configurability, consider using a
       :py:class:`bleach.linkifier.Linker` instance.

    .. Note::

       If you have text that you want to clean and then linkify, consider using
       the :py:class:`bleach.linkifier.LinkifyFilter` as a filter in the clean
       pass. That way you're not parsing the HTML twice.

    :arg str text: the text to linkify

    :arg list callbacks: list of callbacks to run when adjusting tag attributes;
        defaults to ``bleach.linkifier.DEFAULT_CALLBACKS``

    :arg list skip_tags: list of tags that you don't want to linkify the
        contents of; for example, you could set this to ``['pre']`` to skip
        linkifying contents of ``pre`` tags

    :arg bool parse_email: whether or not to linkify email addresses

    :returns: linkified text as unicode

    """
    linker = Linker(
        callbacks=callbacks,
        skip_tags=skip_tags,
        parse_email=parse_email
    )
    return linker.linkify(text)
