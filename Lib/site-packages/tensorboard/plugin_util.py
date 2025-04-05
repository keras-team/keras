# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides utilities that may be especially useful to plugins."""

from google.protobuf import json_format
from importlib import metadata
from packaging import version
import threading

from tensorboard._vendor.bleach.sanitizer import Cleaner
import markdown

from tensorboard import context as _context
from tensorboard.backend import experiment_id as _experiment_id
from tensorboard.util import tb_logging


logger = tb_logging.get_logger()

_ALLOWED_ATTRIBUTES = {
    "a": ["href", "title"],
    "img": ["src", "title", "alt"],
}

_ALLOWED_TAGS = [
    "ul",
    "ol",
    "li",
    "p",
    "pre",
    "code",
    "blockquote",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
    "br",
    "strong",
    "em",
    "a",
    "img",
    "table",
    "thead",
    "tbody",
    "td",
    "tr",
    "th",
]


# Cache Markdown converter to avoid expensive initialization at each
# call to `markdown_to_safe_html`. Cache a different instance per thread.
class _MarkdownStore(threading.local):
    def __init__(self):
        self.markdown = markdown.Markdown(
            extensions=[
                "markdown.extensions.tables",
                "markdown.extensions.fenced_code",
            ]
        )


_MARKDOWN_STORE = _MarkdownStore()


# Cache Cleaner to avoid expensive initialization at each call to `clean`.
# Cache a different instance per thread.
class _CleanerStore(threading.local):
    def __init__(self):
        self.cleaner = Cleaner(
            tags=_ALLOWED_TAGS, attributes=_ALLOWED_ATTRIBUTES
        )


_CLEANER_STORE = _CleanerStore()


def safe_html(unsafe_string):
    """Return the input as a str, sanitized for insertion into the DOM.

    Arguments:
      unsafe_string: A Unicode string or UTF-8--encoded bytestring
        possibly containing unsafe HTML markup.

    Returns:
      A string containing safe HTML.
    """
    total_null_bytes = 0
    if isinstance(unsafe_string, bytes):
        unsafe_string = unsafe_string.decode("utf-8")
    return _CLEANER_STORE.cleaner.clean(unsafe_string)


def markdown_to_safe_html(markdown_string):
    """Convert Markdown to HTML that's safe to splice into the DOM.

    Arguments:
      markdown_string: A Unicode string or UTF-8--encoded bytestring
        containing Markdown source. Markdown tables are supported.

    Returns:
      A string containing safe HTML.
    """
    return markdowns_to_safe_html([markdown_string], lambda xs: xs[0])


def markdowns_to_safe_html(markdown_strings, combine):
    """Convert multiple Markdown documents to one safe HTML document.

    One could also achieve this by calling `markdown_to_safe_html`
    multiple times and combining the results. Compared to that approach,
    this function may be faster, because HTML sanitization (which can be
    expensive) is performed only once rather than once per input. It may
    also be less precise: if one of the input documents has unsafe HTML
    that is sanitized away, that sanitization might affect other
    documents, even if those documents are safe.

    Args:
      markdown_strings: List of Markdown source strings to convert, as
        Unicode strings or UTF-8--encoded bytestrings. Markdown tables
        are supported.
      combine: Callback function that takes a list of unsafe HTML
        strings of the same shape as `markdown_strings` and combines
        them into a single unsafe HTML string, which will be sanitized
        and returned.

    Returns:
      A string containing safe HTML.
    """
    unsafe_htmls = []
    total_null_bytes = 0

    for source in markdown_strings:
        # Convert to utf-8 whenever we have a binary input.
        if isinstance(source, bytes):
            source_decoded = source.decode("utf-8")
            # Remove null bytes and warn if there were any, since it probably means
            # we were given a bad encoding.
            source = source_decoded.replace("\x00", "")
            total_null_bytes += len(source_decoded) - len(source)
        unsafe_html = _MARKDOWN_STORE.markdown.convert(source)
        unsafe_htmls.append(unsafe_html)

    unsafe_combined = combine(unsafe_htmls)
    sanitized_combined = _CLEANER_STORE.cleaner.clean(unsafe_combined)

    warning = ""
    if total_null_bytes:
        warning = (
            "<!-- WARNING: discarded %d null bytes in markdown string "
            "after UTF-8 decoding -->\n"
        ) % total_null_bytes

    return warning + sanitized_combined


def context(environ):
    """Get a TensorBoard `RequestContext` from a WSGI environment.

    Returns:
      A `RequestContext` value.
    """
    return _context.from_environ(environ)


def experiment_id(environ):
    """Determine the experiment ID associated with a WSGI request.

    Each request to TensorBoard has an associated experiment ID, which is
    always a string and may be empty. This experiment ID should be passed
    to data providers.

    Args:
      environ: A WSGI environment `dict`. For a Werkzeug request, this is
        `request.environ`.

    Returns:
      A experiment ID, as a possibly-empty `str`.
    """
    return environ.get(_experiment_id.WSGI_ENVIRON_KEY, "")


def proto_to_json(proto):
    """Utility method to convert proto to JSON, accounting for different version support.

    Args:
      proto: The proto to convert to JSON.
    """
    # Fallback for internal usage, since non third-party code doesn't really
    # have the concept of "versions" in a monorepo. The package version chosen
    # below is the minimum value to choose the non-deprecated kwarg to
    # `MessageToJson`.
    try:
        current_version = metadata.version("protobuf")
    except metadata.PackageNotFoundError:
        current_version = "5.0.0"
    if version.parse(current_version) >= version.parse("5.0.0"):
        return json_format.MessageToJson(
            proto,
            always_print_fields_with_no_presence=True,
        )
    return json_format.MessageToJson(
        proto,
        including_default_value_fields=True,
    )


class _MetadataVersionChecker:
    """TensorBoard-internal utility for warning when data is too new.

    Specify a maximum known `version` number as stored in summary
    metadata, and automatically reject and warn on data from newer
    versions. This keeps a (single) bit of internal state to handle
    logging a warning to the user at most once.

    This should only be used by plugins bundled with TensorBoard, since
    it may instruct users to upgrade their copy of TensorBoard.
    """

    def __init__(self, data_kind, latest_known_version):
        """Initialize a `_MetadataVersionChecker`.

        Args:
          data_kind: A human-readable description of the kind of data
            being read, like "scalar" or "histogram" or "PR curve".
          latest_known_version: Highest tolerated value of `version`,
            like `0`.
        """
        self._data_kind = data_kind
        self._latest_known_version = latest_known_version
        self._warned = False

    def ok(self, version, run, tag):
        """Test whether `version` is permitted, else complain."""
        if 0 <= version <= self._latest_known_version:
            return True
        self._maybe_warn(version, run, tag)
        return False

    def _maybe_warn(self, version, run, tag):
        if self._warned:
            return
        self._warned = True
        logger.warning(
            "Some %s data is too new to be read by this version of TensorBoard. "
            "Upgrading TensorBoard may fix this. "
            "(sample: run %r, tag %r, data version %r)",
            self._data_kind,
            run,
            tag,
            version,
        )
