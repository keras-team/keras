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
"""The TensorBoard Text plugin."""


import textwrap

# pylint: disable=g-bad-import-order
# Necessary for an internal test with special behavior for numpy.
import numpy as np

# pylint: enable=g-bad-import-order

from werkzeug import wrappers

from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.text import metadata

# HTTP routes
TAGS_ROUTE = "/tags"
TEXT_ROUTE = "/text"


WARNING_TEMPLATE = textwrap.dedent(
    """\
  **Warning:** This text summary contained data of dimensionality %d, but only \
  2d tables are supported. Showing a 2d slice of the data instead."""
)

_DEFAULT_DOWNSAMPLING = 100  # text tensors per time series


def make_table_row(contents, tag="td"):
    """Given an iterable of string contents, make a table row.

    Args:
      contents: An iterable yielding strings.
      tag: The tag to place contents in. Defaults to 'td', you might want 'th'.

    Returns:
      A string containing the content strings, organized into a table row.

    Example: make_table_row(['one', 'two', 'three']) == '''
    <tr>
    <td>one</td>
    <td>two</td>
    <td>three</td>
    </tr>'''
    """
    columns = ("<%s>%s</%s>\n" % (tag, s, tag) for s in contents)
    return "<tr>\n" + "".join(columns) + "</tr>\n"


def make_table(contents, headers=None):
    """Given a numpy ndarray of strings, concatenate them into a html table.

    Args:
      contents: A np.ndarray of strings. May be 1d or 2d. In the 1d case, the
        table is laid out vertically (i.e. row-major).
      headers: A np.ndarray or list of string header names for the table.

    Returns:
      A string containing all of the content strings, organized into a table.

    Raises:
      ValueError: If contents is not a np.ndarray.
      ValueError: If contents is not 1d or 2d.
      ValueError: If contents is empty.
      ValueError: If headers is present and not a list, tuple, or ndarray.
      ValueError: If headers is not 1d.
      ValueError: If number of elements in headers does not correspond to number
        of columns in contents.
    """
    if not isinstance(contents, np.ndarray):
        raise ValueError("make_table contents must be a numpy ndarray")

    if contents.ndim not in [1, 2]:
        raise ValueError(
            "make_table requires a 1d or 2d numpy array, was %dd"
            % contents.ndim
        )

    if headers:
        if isinstance(headers, (list, tuple)):
            headers = np.array(headers)
        if not isinstance(headers, np.ndarray):
            raise ValueError(
                "Could not convert headers %s into np.ndarray" % headers
            )
        if headers.ndim != 1:
            raise ValueError("Headers must be 1d, is %dd" % headers.ndim)
        expected_n_columns = contents.shape[1] if contents.ndim == 2 else 1
        if headers.shape[0] != expected_n_columns:
            raise ValueError(
                "Number of headers %d must match number of columns %d"
                % (headers.shape[0], expected_n_columns)
            )
        header = "<thead>\n%s</thead>\n" % make_table_row(headers, tag="th")
    else:
        header = ""

    n_rows = contents.shape[0]
    if contents.ndim == 1:
        # If it's a vector, we need to wrap each element in a new list, otherwise
        # we would turn the string itself into a row (see test code)
        rows = (make_table_row([contents[i]]) for i in range(n_rows))
    else:
        rows = (make_table_row(contents[i, :]) for i in range(n_rows))

    return "<table>\n%s<tbody>\n%s</tbody>\n</table>" % (header, "".join(rows))


def reduce_to_2d(arr):
    """Given a np.npdarray with nDims > 2, reduce it to 2d.

    It does this by selecting the zeroth coordinate for every dimension greater
    than two.

    Args:
      arr: a numpy ndarray of dimension at least 2.

    Returns:
      A two-dimensional subarray from the input array.

    Raises:
      ValueError: If the argument is not a numpy ndarray, or the dimensionality
        is too low.
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("reduce_to_2d requires a numpy.ndarray")

    ndims = len(arr.shape)
    if ndims < 2:
        raise ValueError("reduce_to_2d requires an array of dimensionality >=2")
    # slice(None) is equivalent to `:`, so we take arr[0,0,...0,:,:]
    slices = ([0] * (ndims - 2)) + [slice(None), slice(None)]
    return arr[tuple(slices)]


def text_array_to_html(text_arr, enable_markdown):
    """Take a numpy.ndarray containing strings, and convert it into html.

    If the ndarray contains a single scalar string, that string is converted to
    html via our sanitized markdown parser. If it contains an array of strings,
    the strings are individually converted to html and then composed into a table
    using make_table. If the array contains dimensionality greater than 2,
    all but two of the dimensions are removed, and a warning message is prefixed
    to the table.

    Args:
      text_arr: A numpy.ndarray containing strings.
      enable_markdown: boolean, whether to enable Markdown

    Returns:
      The array converted to html.
    """
    if not text_arr.shape:
        # It is a scalar. No need to put it in a table.
        if enable_markdown:
            return plugin_util.markdown_to_safe_html(text_arr.item())
        else:
            return plugin_util.safe_html(text_arr.item())
    warning = ""
    if len(text_arr.shape) > 2:
        warning = plugin_util.markdown_to_safe_html(
            WARNING_TEMPLATE % len(text_arr.shape)
        )
        text_arr = reduce_to_2d(text_arr)
    if enable_markdown:
        table = plugin_util.markdowns_to_safe_html(
            text_arr.reshape(-1),
            lambda xs: make_table(np.array(xs).reshape(text_arr.shape)),
        )
    else:
        # Convert utf-8 bytes to str. The built-in np.char.decode doesn't work on
        # object arrays, and converting to an numpy chararray is lossy.
        decode = lambda bs: bs.decode("utf-8") if isinstance(bs, bytes) else bs
        text_arr_str = np.array(
            [decode(bs) for bs in text_arr.reshape(-1)]
        ).reshape(text_arr.shape)
        table = plugin_util.safe_html(make_table(text_arr_str))
    return warning + table


def process_event(wall_time, step, string_ndarray, enable_markdown):
    """Convert a text event into a JSON-compatible response."""
    html = text_array_to_html(string_ndarray, enable_markdown)
    return {
        "wall_time": wall_time,
        "step": step,
        "text": html,
    }


class TextPlugin(base_plugin.TBPlugin):
    """Text Plugin for TensorBoard."""

    plugin_name = metadata.PLUGIN_NAME

    def __init__(self, context):
        """Instantiates TextPlugin via TensorBoard core.

        Args:
          context: A base_plugin.TBContext instance.
        """
        self._downsample_to = (context.sampling_hints or {}).get(
            self.plugin_name, _DEFAULT_DOWNSAMPLING
        )
        self._data_provider = context.data_provider
        self._version_checker = plugin_util._MetadataVersionChecker(
            data_kind="text",
            latest_known_version=0,
        )

    def is_active(self):
        return False  # `list_plugins` as called by TB core suffices

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(element_name="tf-text-dashboard")

    def index_impl(self, ctx, experiment):
        mapping = self._data_provider.list_tensors(
            ctx,
            experiment_id=experiment,
            plugin_name=metadata.PLUGIN_NAME,
        )
        result = {run: [] for run in mapping}
        for run, tag_to_content in mapping.items():
            for tag, metadatum in tag_to_content.items():
                md = metadata.parse_plugin_metadata(metadatum.plugin_content)
                if not self._version_checker.ok(md.version, run, tag):
                    continue
                result[run].append(tag)
        return result

    @wrappers.Request.application
    def tags_route(self, request):
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        index = self.index_impl(ctx, experiment)
        return http_util.Respond(request, index, "application/json")

    def text_impl(self, ctx, run, tag, experiment, enable_markdown):
        all_text = self._data_provider.read_tensors(
            ctx,
            experiment_id=experiment,
            plugin_name=metadata.PLUGIN_NAME,
            downsample=self._downsample_to,
            run_tag_filter=provider.RunTagFilter(runs=[run], tags=[tag]),
        )
        text = all_text.get(run, {}).get(tag, None)
        if text is None:
            return []
        return [
            process_event(d.wall_time, d.step, d.numpy, enable_markdown)
            for d in text
        ]

    @wrappers.Request.application
    def text_route(self, request):
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        run = request.args.get("run")
        tag = request.args.get("tag")
        markdown_arg = request.args.get("markdown")
        enable_markdown = markdown_arg != "false"  # Default to enabled.
        response = self.text_impl(ctx, run, tag, experiment, enable_markdown)
        return http_util.Respond(request, response, "application/json")

    def get_plugin_apps(self):
        return {
            TAGS_ROUTE: self.tags_route,
            TEXT_ROUTE: self.text_route,
        }
