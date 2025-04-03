# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Bridge from event multiplexer storage to generic data APIs."""


import base64
import collections
import json
import random

from tensorboard import errors
from tensorboard.compat.proto import summary_pb2
from tensorboard.data import provider
from tensorboard.util import tb_logging
from tensorboard.util import tensor_util

logger = tb_logging.get_logger()


class MultiplexerDataProvider(provider.DataProvider):
    def __init__(self, multiplexer, logdir):
        """Trivial initializer.

        Args:
          multiplexer: A `plugin_event_multiplexer.EventMultiplexer` (note:
            not a boring old `event_multiplexer.EventMultiplexer`).
          logdir: The log directory from which data is being read. Only used
            cosmetically. Should be a `str`.
        """
        self._multiplexer = multiplexer
        self._logdir = logdir

    def __str__(self):
        return "MultiplexerDataProvider(logdir=%r)" % self._logdir

    def _validate_context(self, ctx):
        if type(ctx).__name__ != "RequestContext":
            raise TypeError("ctx must be a RequestContext; got: %r" % (ctx,))

    def _validate_experiment_id(self, experiment_id):
        # This data provider doesn't consume the experiment ID at all, but
        # as a courtesy to callers we require that it be a valid string, to
        # help catch usage errors.
        if not isinstance(experiment_id, str):
            raise TypeError(
                "experiment_id must be %r, but got %r: %r"
                % (str, type(experiment_id), experiment_id)
            )

    def _validate_downsample(self, downsample):
        if downsample is None:
            raise TypeError("`downsample` required but not given")
        if isinstance(downsample, int):
            return  # OK
        raise TypeError(
            "`downsample` must be an int, but got %r: %r"
            % (type(downsample), downsample)
        )

    def _test_run_tag(self, run_tag_filter, run, tag):
        runs = run_tag_filter.runs
        if runs is not None and run not in runs:
            return False
        tags = run_tag_filter.tags
        if tags is not None and tag not in tags:
            return False
        return True

    def _get_first_event_timestamp(self, run_name):
        try:
            return self._multiplexer.FirstEventTimestamp(run_name)
        except ValueError as e:
            return None

    def experiment_metadata(self, ctx=None, *, experiment_id):
        self._validate_context(ctx)
        self._validate_experiment_id(experiment_id)
        return provider.ExperimentMetadata(data_location=self._logdir)

    def list_plugins(self, ctx=None, *, experiment_id):
        self._validate_context(ctx)
        self._validate_experiment_id(experiment_id)
        # Note: This result may include plugins that only have time
        # series with `DATA_CLASS_UNKNOWN`, which will not actually be
        # accessible via `list_*` or read_*`. This is inconsistent with
        # the specification for `list_plugins`, but the bug should be
        # mostly harmless.
        return self._multiplexer.ActivePlugins()

    def list_runs(self, ctx=None, *, experiment_id):
        self._validate_context(ctx)
        self._validate_experiment_id(experiment_id)
        return [
            provider.Run(
                run_id=run,  # use names as IDs
                run_name=run,
                start_time=self._get_first_event_timestamp(run),
            )
            for run in self._multiplexer.Runs()
        ]

    def list_scalars(
        self, ctx=None, *, experiment_id, plugin_name, run_tag_filter=None
    ):
        self._validate_context(ctx)
        self._validate_experiment_id(experiment_id)
        index = self._index(
            plugin_name, run_tag_filter, summary_pb2.DATA_CLASS_SCALAR
        )
        return self._list(provider.ScalarTimeSeries, index)

    def read_scalars(
        self,
        ctx=None,
        *,
        experiment_id,
        plugin_name,
        downsample=None,
        run_tag_filter=None,
    ):
        self._validate_context(ctx)
        self._validate_experiment_id(experiment_id)
        self._validate_downsample(downsample)
        index = self._index(
            plugin_name, run_tag_filter, summary_pb2.DATA_CLASS_SCALAR
        )
        return self._read(_convert_scalar_event, index, downsample)

    def read_last_scalars(
        self,
        ctx=None,
        *,
        experiment_id,
        plugin_name,
        run_tag_filter=None,
    ):
        self._validate_context(ctx)
        self._validate_experiment_id(experiment_id)
        index = self._index(
            plugin_name, run_tag_filter, summary_pb2.DATA_CLASS_SCALAR
        )
        run_tag_to_last_scalar_datum = collections.defaultdict(dict)
        for run, tags_for_run in index.items():
            for tag, metadata in tags_for_run.items():
                events = self._multiplexer.Tensors(run, tag)
                if events:
                    run_tag_to_last_scalar_datum[run][tag] = (
                        _convert_scalar_event(events[-1])
                    )

        return run_tag_to_last_scalar_datum

    def list_tensors(
        self, ctx=None, *, experiment_id, plugin_name, run_tag_filter=None
    ):
        self._validate_context(ctx)
        self._validate_experiment_id(experiment_id)
        index = self._index(
            plugin_name, run_tag_filter, summary_pb2.DATA_CLASS_TENSOR
        )
        return self._list(provider.TensorTimeSeries, index)

    def read_tensors(
        self,
        ctx=None,
        *,
        experiment_id,
        plugin_name,
        downsample=None,
        run_tag_filter=None,
    ):
        self._validate_context(ctx)
        self._validate_experiment_id(experiment_id)
        self._validate_downsample(downsample)
        index = self._index(
            plugin_name, run_tag_filter, summary_pb2.DATA_CLASS_TENSOR
        )
        return self._read(_convert_tensor_event, index, downsample)

    def _index(self, plugin_name, run_tag_filter, data_class_filter):
        """List time series and metadata matching the given filters.

        This is like `_list`, but doesn't traverse `Tensors(...)` to
        compute metadata that's not always needed.

        Args:
          plugin_name: A string plugin name filter (required).
          run_tag_filter: An `provider.RunTagFilter`, or `None`.
          data_class_filter: A `summary_pb2.DataClass` filter (required).

        Returns:
          A nested dict `d` such that `d[run][tag]` is a
          `SummaryMetadata` proto.
        """
        if run_tag_filter is None:
            run_tag_filter = provider.RunTagFilter(runs=None, tags=None)
        runs = run_tag_filter.runs
        tags = run_tag_filter.tags

        # Optimization for a common case, reading a single time series.
        if runs and len(runs) == 1 and tags and len(tags) == 1:
            (run,) = runs
            (tag,) = tags
            try:
                metadata = self._multiplexer.SummaryMetadata(run, tag)
            except KeyError:
                return {}
            all_metadata = {run: {tag: metadata}}
        else:
            all_metadata = self._multiplexer.AllSummaryMetadata()

        result = {}
        for run, tag_to_metadata in all_metadata.items():
            if runs is not None and run not in runs:
                continue
            result_for_run = {}
            for tag, metadata in tag_to_metadata.items():
                if tags is not None and tag not in tags:
                    continue
                if metadata.data_class != data_class_filter:
                    continue
                if metadata.plugin_data.plugin_name != plugin_name:
                    continue
                result[run] = result_for_run
                result_for_run[tag] = metadata

        return result

    def _list(self, construct_time_series, index):
        """Helper to list scalar or tensor time series.

        Args:
          construct_time_series: `ScalarTimeSeries` or `TensorTimeSeries`.
          index: The result of `self._index(...)`.

        Returns:
          A list of objects of type given by `construct_time_series`,
          suitable to be returned from `list_scalars` or `list_tensors`.
        """
        result = {}
        for run, tag_to_metadata in index.items():
            result_for_run = {}
            result[run] = result_for_run
            for tag, summary_metadata in tag_to_metadata.items():
                max_step = None
                max_wall_time = None
                for event in self._multiplexer.Tensors(run, tag):
                    if max_step is None or max_step < event.step:
                        max_step = event.step
                    if max_wall_time is None or max_wall_time < event.wall_time:
                        max_wall_time = event.wall_time
                summary_metadata = self._multiplexer.SummaryMetadata(run, tag)
                result_for_run[tag] = construct_time_series(
                    max_step=max_step,
                    max_wall_time=max_wall_time,
                    plugin_content=summary_metadata.plugin_data.content,
                    description=summary_metadata.summary_description,
                    display_name=summary_metadata.display_name,
                )
        return result

    def _read(self, convert_event, index, downsample):
        """Helper to read scalar or tensor data from the multiplexer.

        Args:
          convert_event: Takes `plugin_event_accumulator.TensorEvent` to
            either `provider.ScalarDatum` or `provider.TensorDatum`.
          index: The result of `self._index(...)`.
          downsample: Non-negative `int`; how many samples to return per
            time series.

        Returns:
          A dict of dicts of values returned by `convert_event` calls,
          suitable to be returned from `read_scalars` or `read_tensors`.
        """
        result = {}
        for run, tags_for_run in index.items():
            result_for_run = {}
            result[run] = result_for_run
            for tag, metadata in tags_for_run.items():
                events = self._multiplexer.Tensors(run, tag)
                data = [convert_event(e) for e in events]
                result_for_run[tag] = _downsample(data, downsample)
        return result

    def list_blob_sequences(
        self, ctx=None, *, experiment_id, plugin_name, run_tag_filter=None
    ):
        self._validate_context(ctx)
        self._validate_experiment_id(experiment_id)
        index = self._index(
            plugin_name, run_tag_filter, summary_pb2.DATA_CLASS_BLOB_SEQUENCE
        )
        result = {}
        for run, tag_to_metadata in index.items():
            result_for_run = {}
            result[run] = result_for_run
            for tag, metadata in tag_to_metadata.items():
                max_step = None
                max_wall_time = None
                max_length = None
                for event in self._multiplexer.Tensors(run, tag):
                    if max_step is None or max_step < event.step:
                        max_step = event.step
                    if max_wall_time is None or max_wall_time < event.wall_time:
                        max_wall_time = event.wall_time
                    length = _tensor_size(event.tensor_proto)
                    if max_length is None or length > max_length:
                        max_length = length
                result_for_run[tag] = provider.BlobSequenceTimeSeries(
                    max_step=max_step,
                    max_wall_time=max_wall_time,
                    max_length=max_length,
                    plugin_content=metadata.plugin_data.content,
                    description=metadata.summary_description,
                    display_name=metadata.display_name,
                )
        return result

    def read_blob_sequences(
        self,
        ctx=None,
        *,
        experiment_id,
        plugin_name,
        downsample=None,
        run_tag_filter=None,
    ):
        self._validate_context(ctx)
        self._validate_experiment_id(experiment_id)
        self._validate_downsample(downsample)
        index = self._index(
            plugin_name, run_tag_filter, summary_pb2.DATA_CLASS_BLOB_SEQUENCE
        )
        result = {}
        for run, tags in index.items():
            result_for_run = {}
            result[run] = result_for_run
            for tag in tags:
                events = self._multiplexer.Tensors(run, tag)
                data_by_step = {}
                for event in events:
                    if event.step in data_by_step:
                        continue
                    data_by_step[event.step] = _convert_blob_sequence_event(
                        experiment_id, plugin_name, run, tag, event
                    )
                data = [datum for (step, datum) in sorted(data_by_step.items())]
                result_for_run[tag] = _downsample(data, downsample)
        return result

    def read_blob(self, ctx=None, *, blob_key):
        self._validate_context(ctx)
        (
            unused_experiment_id,
            plugin_name,
            run,
            tag,
            step,
            index,
        ) = _decode_blob_key(blob_key)

        summary_metadata = self._multiplexer.SummaryMetadata(run, tag)
        if summary_metadata.data_class != summary_pb2.DATA_CLASS_BLOB_SEQUENCE:
            raise errors.NotFoundError(blob_key)
        tensor_events = self._multiplexer.Tensors(run, tag)
        # In case of multiple events at this step, take first (arbitrary).
        matching_step = next((e for e in tensor_events if e.step == step), None)
        if not matching_step:
            raise errors.NotFoundError("%s: no such step %r" % (blob_key, step))
        tensor = tensor_util.make_ndarray(matching_step.tensor_proto)
        return tensor[index]


# TODO(davidsoergel): deduplicate with other implementations
def _encode_blob_key(experiment_id, plugin_name, run, tag, step, index):
    """Generate a blob key: a short, URL-safe string identifying a blob.

    A blob can be located using a set of integer and string fields; here we
    serialize these to allow passing the data through a URL.  Specifically, we
    1) construct a tuple of the arguments in order; 2) represent that as an
    ascii-encoded JSON string (without whitespace); and 3) take the URL-safe
    base64 encoding of that, with no padding.  For example:

        1)  Tuple: ("some_id", "graphs", "train", "graph_def", 2, 0)
        2)   JSON: ["some_id","graphs","train","graph_def",2,0]
        3) base64: WyJzb21lX2lkIiwiZ3JhcGhzIiwidHJhaW4iLCJncmFwaF9kZWYiLDIsMF0K

    Args:
      experiment_id: a string ID identifying an experiment.
      plugin_name: string
      run: string
      tag: string
      step: int
      index: int

    Returns:
      A URL-safe base64-encoded string representing the provided arguments.
    """
    # Encodes the blob key as a URL-safe string, as required by the
    # `BlobReference` API in `tensorboard/data/provider.py`, because these keys
    # may be used to construct URLs for retrieving blobs.
    stringified = json.dumps(
        (experiment_id, plugin_name, run, tag, step, index),
        separators=(",", ":"),
    )
    bytesified = stringified.encode("ascii")
    encoded = base64.urlsafe_b64encode(bytesified)
    return encoded.decode("ascii").rstrip("=")


# Any changes to this function need not be backward-compatible, even though
# the current encoding was used to generate URLs.  The reason is that the
# generated URLs are not considered permalinks: they need to be valid only
# within the context of the session that created them (via the matching
# `_encode_blob_key` function above).
def _decode_blob_key(key):
    """Decode a blob key produced by `_encode_blob_key` into component fields.

    Args:
      key: a blob key, as generated by `_encode_blob_key`.

    Returns:
      A tuple of `(experiment_id, plugin_name, run, tag, step, index)`, with types
      matching the arguments of `_encode_blob_key`.
    """
    decoded = base64.urlsafe_b64decode(key + "==")  # pad past a multiple of 4.
    stringified = decoded.decode("ascii")
    (experiment_id, plugin_name, run, tag, step, index) = json.loads(
        stringified
    )
    return (experiment_id, plugin_name, run, tag, step, index)


def _convert_scalar_event(event):
    """Helper for `read_scalars`."""
    return provider.ScalarDatum(
        step=event.step,
        wall_time=event.wall_time,
        value=tensor_util.make_ndarray(event.tensor_proto).item(),
    )


def _convert_tensor_event(event):
    """Helper for `read_tensors`."""
    return provider.TensorDatum(
        step=event.step,
        wall_time=event.wall_time,
        numpy=tensor_util.make_ndarray(event.tensor_proto),
    )


def _convert_blob_sequence_event(experiment_id, plugin_name, run, tag, event):
    """Helper for `read_blob_sequences`."""
    num_blobs = _tensor_size(event.tensor_proto)
    values = tuple(
        provider.BlobReference(
            _encode_blob_key(
                experiment_id,
                plugin_name,
                run,
                tag,
                event.step,
                idx,
            )
        )
        for idx in range(num_blobs)
    )
    return provider.BlobSequenceDatum(
        wall_time=event.wall_time,
        step=event.step,
        values=values,
    )


def _tensor_size(tensor_proto):
    """Compute the number of elements in a tensor.

    This does not deserialize the full tensor contents.

    Args:
      tensor_proto: A `tensorboard.compat.proto.tensor_pb2.TensorProto`.

    Returns:
      A non-negative `int`.
    """
    # This is the same logic that `tensor_util.make_ndarray` uses to
    # compute the size, but without the actual buffer copies.
    result = 1
    for dim in tensor_proto.tensor_shape.dim:
        result *= dim.size
    return result


def _downsample(xs, k):
    """Downsample `xs` to at most `k` elements.

    If `k` is larger than `xs`, then the contents of `xs` itself will be
    returned. If `k` is smaller than `xs`, the last element of `xs` will
    always be included (unless `k` is `0`) and the preceding elements
    will be selected uniformly at random.

    This differs from `random.sample` in that it returns a subsequence
    (i.e., order is preserved) and that it permits `k > len(xs)`.

    The random number generator will always be `random.Random(0)`, so
    this function is deterministic (within a Python process).

    Args:
      xs: A sequence (`collections.abc.Sequence`).
      k: A non-negative integer.

    Returns:
      A new list whose elements are a subsequence of `xs` of length
      `min(k, len(xs))` and that is guaranteed to include the last
      element of `xs`, uniformly selected among such subsequences.
    """

    if k > len(xs):
        return list(xs)
    if k == 0:
        return []
    indices = random.Random(0).sample(range(len(xs) - 1), k - 1)
    indices.sort()
    indices += [len(xs) - 1]
    return [xs[i] for i in indices]
