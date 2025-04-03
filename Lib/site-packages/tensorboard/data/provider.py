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
"""Experimental framework for generic TensorBoard data providers."""


from typing import Collection, Sequence, Tuple, Union
import abc
import dataclasses
import enum

import numpy as np


class DataProvider(metaclass=abc.ABCMeta):
    """Interface for reading TensorBoard scalar, tensor, and blob data.

    These APIs are under development and subject to change. For instance,
    providers may be asked to implement more filtering mechanisms, such as
    downsampling strategies or domain restriction by step or wall time.

    The data provider interface specifies three *data classes*: scalars,
    tensors, and blob sequences. All data is stored in *time series* for
    one of these data classes. A time series is identified by run name and
    tag name (each a non-empty text string), as well as an experiment ID
    and plugin name (see below). Points in a time series are uniquely
    indexed by *step*, an arbitrary non-negative integer. Each point in a
    time series also has an associated wall time, plus its actual value,
    which is drawn from the corresponding data class.

    Each point in a scalar time series contains a single scalar value, as
    a 64-bit floating point number. Scalars are "privileged" rather than
    being subsumed under tensors because there are useful operations on
    scalars that don't make sense in the general tensor case: e.g., "list
    all scalar time series with tag name `accuracy` whose exponentially
    weighted moving average is at least 0.999".

    Each point in a tensor time series contains a tensor of arbitrary
    dtype (including byte strings and text strings) and shape (including
    rank-0 tensors, a.k.a. scalars). Each tensor is expected to be
    "reasonably small" to accommodate common database cell size limits.
    For instance, a histogram with a bounded number of buckets (say, 30)
    occupies about 500 bytes, and a PR curve with a bounded number of
    thresholds (say, 201) occupies about 5000 bytes. These are both well
    within typical database tolerances (Google Cloud Spanner: 10 MiB;
    MySQL: 64 KiB), and would be appropriate to store as tensors. By
    contrast, image, audio, or model graph data may easily be multiple
    megabytes in size, and so should be stored as blobs instead. The
    tensors at each step in a time series need not have the same dtype or
    shape.

    Each point in a blob sequence time series contains an ordered sequence
    of zero or more blobs, which are arbitrary data with no tensor
    structure. These might represent PNG-encoded image data, protobuf wire
    encodings of TensorFlow graphs, or PLY-format 3D mesh data, for some
    examples. This data class provides blob *sequences* rather than just
    blobs because it's common to want to take multiple homogeneous samples
    of a given time series: say, "show me the bounding box classifications
    for 3 random inputs from this batch". A single blob can of course be
    represented as a blob sequence that always has exactly one element.

    When reading time series, *downsampling* refers to selecting a
    subset of the points in each time series. Downsampling only occurs
    across the step axis (rather than, e.g., the blobs in a single blob
    sequence datum), and occurs individually within each time series.
    When downsampling, the latest datum should always be included in the
    sample, so that clients have a view of metrics that is maximally up
    to date. Implementations may choose to force the first (oldest)
    datum to be included in each sample as well, but this is not
    required; clients should not make assumptions either way. The
    remainder of the points in the sample should be selected uniformly
    at random from available points. Downsampling should be
    deterministic within a time series. It is also useful for the
    downsampling behavior to depend only on the set of step values
    within a time series, such that two "parallel" time series with data
    at exactly the same steps also retain the same steps after
    downsampling.

    Every time series belongs to a specific experiment and is owned by a
    specific plugin. (Thus, the "primary key" for a time series has four
    components: experiment, plugin, run, tag.) The experiment ID is an
    arbitrary URL-safe non-empty text string, whose interpretation is at
    the discretion of the data provider. As a special case, the empty
    string as an experiment ID denotes that no experiment was given. Data
    providers may or may not fully support an empty experiment ID. The
    plugin name should correspond to the `plugin_data.plugin_name` field
    of the `SummaryMetadata` proto passed to `tf.summary.write`.

    Additionally, the data provider interface specifies one *hyperparameter*
    class, which is metadata about the parameters used to generate the data for
    one or more runs within one or more experiments. Each hyperparameter has a
    value type -- one of string, bool, and float. Each one also has a domain,
    which describes the set of known values for that hyperparameter across the
    given set of experiments.

    There is a corresponding *hyperparameter value* class, which describes an
    actual value of a hyperparameter that was logged during experiment
    execution.

    Each run within an experiment may specify its own value for a
    hyperparameter. Runs that were logically executed together with the same set
    of hyperparameter values form a hyperparameter `session`. Sessions that
    include the same hyperparameter values can be grouped together in a
    hyperparameter `session group`. Often a session group will contain only a
    single session. However, in some scenarios, the same hyperparameters will be
    used to execute multiple jobs with the idea to aggregate the metrics across
    those jobs and analyze non-deterministic factors. In that case, a session
    group will contain multiple sessions. The result will group runs by
    hyperparameter session group and provide one set of hyperparameter values
    for each group.

    All methods on this class take a `RequestContext` parameter as the
    first positional argument. This argument is temporarily optional to
    facilitate migration, but will be required in the future.

    Unless otherwise noted, any methods on this class may raise errors
    defined in `tensorboard.errors`, like `tensorboard.errors.NotFoundError`.

    If not implemented, optional methods may return `None`.
    """

    def experiment_metadata(self, ctx=None, *, experiment_id):
        """Retrieve metadata of a given experiment.

        The metadata may include fields such as name and description
        of the experiment, as well as a timestamp for the experiment.

        Args:
          ctx: A TensorBoard `RequestContext` value.
          experiment_id:  ID of the experiment in question.

        Returns:
          An `ExperimentMetadata` object containing metadata about the
          experiment.
        """
        return ExperimentMetadata()

    def list_plugins(self, ctx=None, *, experiment_id):
        """List all plugins that own data in a given experiment.

        This should be the set of all plugin names `p` such that calling
        `list_scalars`, `list_tensors`, or `list_blob_sequences` for the
        given `experiment_id` and plugin name `p` gives a non-empty
        result.

        This operation is optional, but may later become required.

        Args:
          ctx: A TensorBoard `RequestContext` value.
          experiment_id: ID of enclosing experiment.

        Returns:
          A collection of strings representing plugin names, or `None`
          if this operation is not supported by this data provider.
        """
        return None

    @abc.abstractmethod
    def list_runs(self, ctx=None, *, experiment_id):
        """List all runs within an experiment.

        Args:
          ctx: A TensorBoard `RequestContext` value.
          experiment_id: ID of enclosing experiment.

        Returns:
          A collection of `Run` values.

        Raises:
          tensorboard.errors.PublicError: See `DataProvider` class docstring.
        """
        pass

    @abc.abstractmethod
    def list_scalars(
        self, ctx=None, *, experiment_id, plugin_name, run_tag_filter=None
    ):
        """List metadata about scalar time series.

        Args:
          ctx: A TensorBoard `RequestContext` value.
          experiment_id: ID of enclosing experiment.
          plugin_name: String name of the TensorBoard plugin that created
            the data to be queried. Required.
          run_tag_filter: Optional `RunTagFilter` value. If omitted, all
            runs and tags will be included.

        The result will only contain keys for run-tag combinations that
        actually exist, which may not include all entries in the
        `run_tag_filter`.

        Returns:
          A nested map `d` such that `d[run][tag]` is a `ScalarTimeSeries`
          value.

        Raises:
          tensorboard.errors.PublicError: See `DataProvider` class docstring.
        """
        pass

    @abc.abstractmethod
    def read_scalars(
        self,
        ctx=None,
        *,
        experiment_id,
        plugin_name,
        downsample=None,
        run_tag_filter=None,
    ):
        """Read values from scalar time series.

        Args:
          ctx: A TensorBoard `RequestContext` value.
          experiment_id: ID of enclosing experiment.
          plugin_name: String name of the TensorBoard plugin that created
            the data to be queried. Required.
          downsample: Integer number of steps to which to downsample the
            results (e.g., `1000`). The most recent datum (last scalar)
            should always be included. See `DataProvider` class docstring
            for details about this parameter. Required.
          run_tag_filter: Optional `RunTagFilter` value. If provided, a time
            series will only be included in the result if its run and tag
            both pass this filter. If `None`, all time series will be
            included.

        The result will only contain keys for run-tag combinations that
        actually exist, which may not include all entries in the
        `run_tag_filter`.

        Returns:
          A nested map `d` such that `d[run][tag]` is a list of
          `ScalarDatum` values sorted by step.

        Raises:
          tensorboard.errors.PublicError: See `DataProvider` class docstring.
        """
        pass

    @abc.abstractmethod
    def read_last_scalars(
        self,
        ctx=None,
        *,
        experiment_id,
        plugin_name,
        run_tag_filter=None,
    ):
        """Read the most recent values from scalar time series.

        The most recent scalar value for each tag under each run is retrieved
        from the latest event (at the latest timestamp). Note that this is
        different from the sorting used in `read_scalars`, which is by step.
        This was an accidental misalignment that would need considerable effort
        to change across our implementations, so we're leaving it as is for now.
        In most cases this should not matter, but if the same log dir is used
        for multiple runs, this might not match the last data point returned by
        the `read_scalars`.

        Args:
          ctx: A TensorBoard `RequestContext` value.
          experiment_id: ID of enclosing experiment.
          plugin_name: String name of the TensorBoard plugin that created
            the data to be queried. Required.
          run_tag_filter: Optional `RunTagFilter` value. If provided, a datum
            series will only be included in the result if its run and tag
            both pass this filter. If `None`, all time series will be
            included.

        The result will only contain keys for run-tag combinations that
        actually exist, which may not include all entries in the
        `run_tag_filter`.

        Returns:
          A nested map `d` such that `d[run][tag]` is a `ScalarDatum`
          representing the latest scalar in the time series.

        Raises:
          tensorboard.errors.PublicError: See `DataProvider` class docstring.
        """
        pass

    def list_tensors(
        self, ctx=None, *, experiment_id, plugin_name, run_tag_filter=None
    ):
        """List metadata about tensor time series.

        Args:
          ctx: A TensorBoard `RequestContext` value.
          experiment_id: ID of enclosing experiment.
          plugin_name: String name of the TensorBoard plugin that created
            the data to be queried. Required.
          run_tag_filter: Optional `RunTagFilter` value. If omitted, all
            runs and tags will be included.

        The result will only contain keys for run-tag combinations that
        actually exist, which may not include all entries in the
        `run_tag_filter`.

        Returns:
          A nested map `d` such that `d[run][tag]` is a `TensorTimeSeries`
          value.

        Raises:
          tensorboard.errors.PublicError: See `DataProvider` class docstring.
        """
        pass

    def read_tensors(
        self,
        ctx=None,
        *,
        experiment_id,
        plugin_name,
        downsample=None,
        run_tag_filter=None,
    ):
        """Read values from tensor time series.

        Args:
          ctx: A TensorBoard `RequestContext` value.
          experiment_id: ID of enclosing experiment.
          plugin_name: String name of the TensorBoard plugin that created
            the data to be queried. Required.
          downsample: Integer number of steps to which to downsample the
            results (e.g., `1000`). See `DataProvider` class docstring
            for details about this parameter. Required.
          run_tag_filter: Optional `RunTagFilter` value. If provided, a time
            series will only be included in the result if its run and tag
            both pass this filter. If `None`, all time series will be
            included.

        The result will only contain keys for run-tag combinations that
        actually exist, which may not include all entries in the
        `run_tag_filter`.

        Returns:
          A nested map `d` such that `d[run][tag]` is a list of
          `TensorDatum` values sorted by step.

        Raises:
          tensorboard.errors.PublicError: See `DataProvider` class docstring.
        """
        pass

    def list_blob_sequences(
        self, ctx=None, *, experiment_id, plugin_name, run_tag_filter=None
    ):
        """List metadata about blob sequence time series.

        Args:
          ctx: A TensorBoard `RequestContext` value.
          experiment_id: ID of enclosing experiment.
          plugin_name: String name of the TensorBoard plugin that created the data
            to be queried. Required.
          run_tag_filter: Optional `RunTagFilter` value. If omitted, all runs and
            tags will be included. The result will only contain keys for run-tag
            combinations that actually exist, which may not include all entries in
            the `run_tag_filter`.

        Returns:
          A nested map `d` such that `d[run][tag]` is a `BlobSequenceTimeSeries`
          value.

        Raises:
          tensorboard.errors.PublicError: See `DataProvider` class docstring.
        """
        pass

    def read_blob_sequences(
        self,
        ctx=None,
        *,
        experiment_id,
        plugin_name,
        downsample=None,
        run_tag_filter=None,
    ):
        """Read values from blob sequence time series.

        Args:
          ctx: A TensorBoard `RequestContext` value.
          experiment_id: ID of enclosing experiment.
          plugin_name: String name of the TensorBoard plugin that created the data
            to be queried. Required.
          downsample: Integer number of steps to which to downsample the
            results (e.g., `1000`). See `DataProvider` class docstring
            for details about this parameter. Required.
          run_tag_filter: Optional `RunTagFilter` value. If provided, a time series
            will only be included in the result if its run and tag both pass this
            filter. If `None`, all time series will be included. The result will
            only contain keys for run-tag combinations that actually exist, which
            may not include all entries in the `run_tag_filter`.

        Returns:
          A nested map `d` such that `d[run][tag]` is a list of
          `BlobSequenceDatum` values sorted by step.

        Raises:
          tensorboard.errors.PublicError: See `DataProvider` class docstring.
        """
        pass

    def read_blob(self, ctx=None, *, blob_key):
        """Read data for a single blob.

        Args:
          ctx: A TensorBoard `RequestContext` value.
          blob_key: A key identifying the desired blob, as provided by
            `read_blob_sequences(...)`.

        Returns:
          Raw binary data as `bytes`.

        Raises:
          tensorboard.errors.PublicError: See `DataProvider` class docstring.
        """
        pass

    def list_hyperparameters(self, ctx=None, *, experiment_ids, limit=None):
        """List hyperparameters metadata.

        Args:
          ctx: A TensorBoard `RequestContext` value.
          experiment_ids: A Collection[string] of IDs of the enclosing
            experiments.
          limit: Optional number of hyperparameter metadata to include in the
            result. If unset or zero, all metadata will be included.

        Returns:
          A ListHyperparametersResult describing the hyperparameter-related
          metadata for the experiments.

        Raises:
          tensorboard.errors.PublicError: See `DataProvider` class docstring.
        """
        return ListHyperparametersResult(hyperparameters=[], session_groups=[])

    def read_hyperparameters(
        self,
        ctx=None,
        *,
        experiment_ids,
        filters=None,
        sort=None,
        hparams_to_include=None,
    ):
        """Read hyperparameter values.

        Args:
          ctx: A TensorBoard `RequestContext` value.
          experiment_ids: A Collection[string] of IDs of the enclosing
            experiments.
          filters: A Collection[HyperparameterFilter] that constrain the
            returned session groups based on hyperparameter value.
          sort: A Sequence[HyperparameterSort] that specify how the results
            should be sorted.
          hparams_to_include: An optional Collection[str] of the full names of
            hyperparameters to include in the results. This collection will be
            augmented to include all the hyperparameters specified in `filters`
            and `sort`. If None, all hyperparameters will be returned.

        Returns:
          A Sequence[HyperparameterSessionGroup] describing the groups and
          their hyperparameter values.

        Raises:
          tensorboard.errors.PublicError: See `DataProvider` class docstring.
        """
        return []


class ExperimentMetadata:
    """Metadata about an experiment.

    All fields have default values: i.e., they will always be present on
    the object, but may be omitted in a constructor call.

    Attributes:
      data_location: A human-readable description of the data source, such as a
        path to a directory on disk.
      experiment_name: A user-facing name for the experiment (as a `str`).
      experiment_description: A user-facing description for the experiment
        (as a `str`).
      creation_time: A timestamp for the creation of the experiment, as `float`
        seconds since the epoch.
    """

    def __init__(
        self,
        *,
        data_location="",
        experiment_name="",
        experiment_description="",
        creation_time=0,
    ):
        self._data_location = data_location
        self._experiment_name = experiment_name
        self._experiment_description = experiment_description
        self._creation_time = creation_time

    @property
    def data_location(self):
        return self._data_location

    @property
    def experiment_name(self):
        return self._experiment_name

    @property
    def experiment_description(self):
        return self._experiment_description

    @property
    def creation_time(self):
        return self._creation_time

    def _as_tuple(self):
        """Helper for `__eq__` and `__hash__`."""
        return (
            self._data_location,
            self._experiment_name,
            self._experiment_description,
            self._creation_time,
        )

    def __eq__(self, other):
        if not isinstance(other, ExperimentMetadata):
            return False
        return self._as_tuple() == other._as_tuple()

    def __hash__(self):
        return hash(self._as_tuple())

    def __repr__(self):
        return "ExperimentMetadata(%s)" % ", ".join(
            (
                "data_location=%r" % (self.data_location,),
                "experiment_name=%r" % (self._experiment_name,),
                "experiment_description=%r" % (self._experiment_description,),
                "creation_time=%r" % (self._creation_time,),
            )
        )


class Run:
    """Metadata about a run.

    Attributes:
      run_id: A unique opaque string identifier for this run.
      run_name: A user-facing name for this run (as a `str`).
      start_time: The wall time of the earliest recorded event in this
        run, as `float` seconds since epoch, or `None` if this run has no
        recorded events.
    """

    __slots__ = ("_run_id", "_run_name", "_start_time")

    def __init__(self, run_id, run_name, start_time):
        self._run_id = run_id
        self._run_name = run_name
        self._start_time = start_time

    @property
    def run_id(self):
        return self._run_id

    @property
    def run_name(self):
        return self._run_name

    @property
    def start_time(self):
        return self._start_time

    def __eq__(self, other):
        if not isinstance(other, Run):
            return False
        if self._run_id != other._run_id:
            return False
        if self._run_name != other._run_name:
            return False
        if self._start_time != other._start_time:
            return False
        return True

    def __hash__(self):
        return hash((self._run_id, self._run_name, self._start_time))

    def __repr__(self):
        return "Run(%s)" % ", ".join(
            (
                "run_id=%r" % (self._run_id,),
                "run_name=%r" % (self._run_name,),
                "start_time=%r" % (self._start_time,),
            )
        )


class HyperparameterDomainType(enum.Enum):
    """Describes how to represent the set of known values for a hyperparameter."""

    # A range of numeric values. Normally represented as Tuple[float, float].
    INTERVAL = "interval"
    # A finite set of numeric values. Normally represented as Collection[float].
    DISCRETE_FLOAT = "discrete_float"
    # A finite set of string values. Normally represented as Collection[string].
    DISCRETE_STRING = "discrete_string"
    # A finite set of bool values. Normally represented as Collection[bool].
    DISCRETE_BOOL = "discrete_bool"


@dataclasses.dataclass(frozen=True)
class Hyperparameter:
    """Metadata about a hyperparameter.

    Attributes:
      hyperparameter_name: A string identifier for the hyperparameter that
        should be unique in any result set of Hyperparameter objects.
      hyperparameter_display_name: A displayable name for the hyperparameter.
        Unlike hyperparameter_name, there is no uniqueness constraint.
      domain_type: A HyperparameterDomainType describing how we represent the
        set of known values in the `domain` attribute.
      domain: A representation of the set of known values for the
        hyperparameter.

        If domain_type is INTERVAL, a Tuple[float, float] describing the
          range of numeric values.
        If domain_type is DISCRETE_FLOAT, a Collection[float] describing the
          finite set of numeric values.
        If domain_type is DISCRETE_STRING, a Collection[string] describing the
          finite set of string values.
        If domain_type is DISCRETE_BOOL, a Collection[bool] describing the
          finite set of bool values.

      differs: Describes whether there are two or more known values for the
        hyperparameter for the set of experiments specified in the
        list_hyperparameters() request. Hyperparameters for which this is
        true are made more prominent or easier to discover in the UI.
    """

    hyperparameter_name: str
    hyperparameter_display_name: str
    domain_type: Union[HyperparameterDomainType, None] = None
    domain: Union[
        Tuple[float, float],
        Collection[float],
        Collection[str],
        Collection[bool],
        None,
    ] = None
    differs: bool = False


@dataclasses.dataclass(frozen=True)
class HyperparameterValue:
    """A hyperparameter value.

    Attributes:
      hyperparameter_name: A string identifier for the hyperparameters. It
        corresponds to the hyperparameter_name field in the Hyperparameter
        class.
      domain_type: A HyperparameterDomainType describing how we represent the
        set of known values in the `domain` attribute.
      value: The value of the hyperparameter.

        If domain_type is INTERVAL or DISCRETE_FLOAT, value is a float.
        If domain_type is DISCRETE_STRING, value is a str.
        If domain_type is DISCRETE_BOOL, value is a bool.
        If domain_type is unknown (None), value is None.
    """

    hyperparameter_name: str
    domain_type: Union[HyperparameterDomainType, None] = None
    value: Union[float, str, bool, None] = None


@dataclasses.dataclass(frozen=True)
class HyperparameterSessionRun:
    """A single run in a HyperparameterSessionGroup.

    Attributes:
      experiment_id: The id of the experiment to which the run belongs.
      run: The name of the run.
    """

    experiment_id: str
    run: str


@dataclasses.dataclass(frozen=True)
class HyperparameterSessionGroup:
    """A group of sessions logically executed together with the same hparam values.

    A `session` generally represents a particular execution of a job with a given
    set of hyperparameter values. A session may contain multiple related runs
    executed together to train and/or validate a model.

    Often a `session group` will contain only a single session. However, in some
    scenarios, the same hyperparameters will be used to execute multiple jobs
    with the idea to aggregate the metrics across those jobs and analyze
    non-deterministic factors. In that case, a session group will contain multiple
    sessions.

    Attributes:
      root: A descriptor of the common ancestor of all sessions in this
        group.

        In the case where the group contains all runs in the experiment, this
        would just be a HyperparameterSessionRun with the experiment_id property
        set to the experiment's id but run property set to empty.

        In the case where the group contains a subset of runs in the experiment,
        this would be a HyperparameterSessionRun with the experiment_id property
        set and the run property set to the largest common prefix for runs.

        The root might correspond to a session within the group but it is not
        necessary.
      sessions: A sequence of all sessions in this group.
      hyperparameter_values: A collection of all hyperparameter values in this
        group.
    """

    root: HyperparameterSessionRun
    sessions: Sequence[HyperparameterSessionRun]
    hyperparameter_values: Collection[HyperparameterValue]


class HyperparameterFilterType(enum.Enum):
    """Describes how to represent filter values."""

    # A regular expression string. Normally represented as str.
    REGEX = "regex"
    # A range of numeric values. Normally represented as Tuple[float, float].
    INTERVAL = "interval"
    # A finite set of values. Normally represented as Collection[float|str|bool].
    DISCRETE = "discrete"


@dataclasses.dataclass(frozen=True)
class HyperparameterFilter:
    """A constraint based on hyperparameter value.

    Attributes:
      hyperparameter_name: A string identifier for the hyperparameter to use for
        the filter. It corresponds to the hyperparameter_name field in the
        Hyperparameter class.
      filter_type: A HyperparameterFilterType describing how we represent the
        filter values in the 'filter' attribute.
      filter: A representation of the set of the filter values.

        If filter_type is REGEX, a str containing the regular expression.
        If filter_type is INTERVAL, a Tuple[float, float] describing the min and
          max values of the filter interval.
        If filter_type is DISCRETE a Collection[float|str|bool] describing the
          finite set of filter values.
    """

    hyperparameter_name: str
    filter_type: HyperparameterFilterType
    filter: Union[
        str,
        Tuple[float, float],
        Collection[Union[float, str, bool]],
    ]


class HyperparameterSortDirection(enum.Enum):
    """Describes which direction to sort a value."""

    # Sort values ascending.
    ASCENDING = "ascending"
    # Sort values descending.
    DESCENDING = "descending"


@dataclasses.dataclass(frozen=True)
class HyperparameterSort:
    """A sort criterium based on hyperparameter value.

    Attributes:
      hyperparameter_name: A string identifier for the hyperparameter to use for
        the sort. It corresponds to the hyperparameter_name field in the
        Hyperparameter class.
      sort_direction: The direction to sort.
    """

    hyperparameter_name: str
    sort_direction: HyperparameterSortDirection


@dataclasses.dataclass(frozen=True)
class ListHyperparametersResult:
    """The result from calling list_hyperparameters().

    Attributes:
      hyperparameters: The hyperparameteres belonging to the experiments in the
        request.
      session_groups: The session groups present in the experiments in the
        request.
    """

    hyperparameters: Collection[Hyperparameter]
    session_groups: Collection[HyperparameterSessionGroup]


class _TimeSeries:
    """Metadata about time series data for a particular run and tag.

    Superclass of `ScalarTimeSeries`, `TensorTimeSeries`, and
    `BlobSequenceTimeSeries`.
    """

    __slots__ = (
        "_max_step",
        "_max_wall_time",
        "_plugin_content",
        "_description",
        "_display_name",
        "_last_value",
    )

    def __init__(
        self,
        *,
        max_step,
        max_wall_time,
        plugin_content,
        description,
        display_name,
        last_value=None,
    ):
        self._max_step = max_step
        self._max_wall_time = max_wall_time
        self._plugin_content = plugin_content
        self._description = description
        self._display_name = display_name
        self._last_value = last_value

    @property
    def max_step(self):
        return self._max_step

    @property
    def max_wall_time(self):
        return self._max_wall_time

    @property
    def plugin_content(self):
        return self._plugin_content

    @property
    def description(self):
        return self._description

    @property
    def display_name(self):
        return self._display_name

    @property
    def last_value(self):
        return self._last_value


class ScalarTimeSeries(_TimeSeries):
    """Metadata about a scalar time series for a particular run and tag.

    Attributes:
      max_step: The largest step value of any datum in this scalar time series; a
        nonnegative integer.
      max_wall_time: The largest wall time of any datum in this time series, as
        `float` seconds since epoch.
      plugin_content: A bytestring of arbitrary plugin-specific metadata for this
        time series, as provided to `tf.summary.write` in the
        `plugin_data.content` field of the `metadata` argument.
      description: An optional long-form Markdown description, as a `str` that is
        empty if no description was specified.
      display_name: An optional long-form Markdown description, as a `str` that is
        empty if no description was specified. Deprecated; may be removed soon.
      last_value: An optional value for the latest scalar in the time series,
        corresponding to the scalar at `max_step`. Note that this field might NOT
        be populated by all data provider implementations.
    """

    def __eq__(self, other):
        if not isinstance(other, ScalarTimeSeries):
            return False
        if self._max_step != other._max_step:
            return False
        if self._max_wall_time != other._max_wall_time:
            return False
        if self._plugin_content != other._plugin_content:
            return False
        if self._description != other._description:
            return False
        if self._display_name != other._display_name:
            return False
        if self._last_value != other._last_value:
            return False
        return True

    def __hash__(self):
        return hash(
            (
                self._max_step,
                self._max_wall_time,
                self._plugin_content,
                self._description,
                self._display_name,
                self._last_value,
            )
        )

    def __repr__(self):
        return "ScalarTimeSeries(%s)" % ", ".join(
            (
                "max_step=%r" % (self._max_step,),
                "max_wall_time=%r" % (self._max_wall_time,),
                "plugin_content=%r" % (self._plugin_content,),
                "description=%r" % (self._description,),
                "display_name=%r" % (self._display_name,),
                "last_value=%r" % (self._last_value,),
            )
        )


class ScalarDatum:
    """A single datum in a scalar time series for a run and tag.

    Attributes:
      step: The global step at which this datum occurred; an integer. This
        is a unique key among data of this time series.
      wall_time: The real-world time at which this datum occurred, as
        `float` seconds since epoch.
      value: The scalar value for this datum; a `float`.
    """

    __slots__ = ("_step", "_wall_time", "_value")

    def __init__(self, step, wall_time, value):
        self._step = step
        self._wall_time = wall_time
        self._value = value

    @property
    def step(self):
        return self._step

    @property
    def wall_time(self):
        return self._wall_time

    @property
    def value(self):
        return self._value

    def __eq__(self, other):
        if not isinstance(other, ScalarDatum):
            return False
        if self._step != other._step:
            return False
        if self._wall_time != other._wall_time:
            return False
        if self._value != other._value:
            return False
        return True

    def __hash__(self):
        return hash((self._step, self._wall_time, self._value))

    def __repr__(self):
        return "ScalarDatum(%s)" % ", ".join(
            (
                "step=%r" % (self._step,),
                "wall_time=%r" % (self._wall_time,),
                "value=%r" % (self._value,),
            )
        )


class TensorTimeSeries(_TimeSeries):
    """Metadata about a tensor time series for a particular run and tag.

    Attributes:
      max_step: The largest step value of any datum in this tensor time series; a
        nonnegative integer.
      max_wall_time: The largest wall time of any datum in this time series, as
        `float` seconds since epoch.
      plugin_content: A bytestring of arbitrary plugin-specific metadata for this
        time series, as provided to `tf.summary.write` in the
        `plugin_data.content` field of the `metadata` argument.
      description: An optional long-form Markdown description, as a `str` that is
        empty if no description was specified.
      display_name: An optional long-form Markdown description, as a `str` that is
        empty if no description was specified. Deprecated; may be removed soon.
    """

    def __eq__(self, other):
        if not isinstance(other, TensorTimeSeries):
            return False
        if self._max_step != other._max_step:
            return False
        if self._max_wall_time != other._max_wall_time:
            return False
        if self._plugin_content != other._plugin_content:
            return False
        if self._description != other._description:
            return False
        if self._display_name != other._display_name:
            return False
        return True

    def __hash__(self):
        return hash(
            (
                self._max_step,
                self._max_wall_time,
                self._plugin_content,
                self._description,
                self._display_name,
            )
        )

    def __repr__(self):
        return "TensorTimeSeries(%s)" % ", ".join(
            (
                "max_step=%r" % (self._max_step,),
                "max_wall_time=%r" % (self._max_wall_time,),
                "plugin_content=%r" % (self._plugin_content,),
                "description=%r" % (self._description,),
                "display_name=%r" % (self._display_name,),
            )
        )


class TensorDatum:
    """A single datum in a tensor time series for a run and tag.

    Attributes:
      step: The global step at which this datum occurred; an integer. This
        is a unique key among data of this time series.
      wall_time: The real-world time at which this datum occurred, as
        `float` seconds since epoch.
      numpy: The `numpy.ndarray` value with the tensor contents of this
        datum.
    """

    __slots__ = ("_step", "_wall_time", "_numpy")

    def __init__(self, step, wall_time, numpy):
        self._step = step
        self._wall_time = wall_time
        self._numpy = numpy

    @property
    def step(self):
        return self._step

    @property
    def wall_time(self):
        return self._wall_time

    @property
    def numpy(self):
        return self._numpy

    def __eq__(self, other):
        if not isinstance(other, TensorDatum):
            return False
        if self._step != other._step:
            return False
        if self._wall_time != other._wall_time:
            return False
        if not np.array_equal(self._numpy, other._numpy):
            return False
        return True

    # Unhashable type: numpy arrays are mutable.
    __hash__ = None

    def __repr__(self):
        return "TensorDatum(%s)" % ", ".join(
            (
                "step=%r" % (self._step,),
                "wall_time=%r" % (self._wall_time,),
                "numpy=%r" % (self._numpy,),
            )
        )


class BlobSequenceTimeSeries(_TimeSeries):
    """Metadata about a blob sequence time series for a particular run and tag.

    Attributes:
      max_step: The largest step value of any datum in this scalar time series; a
        nonnegative integer.
      max_wall_time: The largest wall time of any datum in this time series, as
        `float` seconds since epoch.
      max_length: The largest length (number of blobs) of any datum in
        this scalar time series, or `None` if this time series is empty.
      plugin_content: A bytestring of arbitrary plugin-specific metadata for this
        time series, as provided to `tf.summary.write` in the
        `plugin_data.content` field of the `metadata` argument.
      description: An optional long-form Markdown description, as a `str` that is
        empty if no description was specified.
      display_name: An optional long-form Markdown description, as a `str` that is
        empty if no description was specified. Deprecated; may be removed soon.
    """

    __slots__ = ("_max_length",)

    def __init__(
        self,
        *,
        max_step,
        max_wall_time,
        max_length,
        plugin_content,
        description,
        display_name,
    ):
        super().__init__(
            max_step=max_step,
            max_wall_time=max_wall_time,
            plugin_content=plugin_content,
            description=description,
            display_name=display_name,
        )
        self._max_length = max_length

    @property
    def max_length(self):
        return self._max_length

    def __eq__(self, other):
        if not isinstance(other, BlobSequenceTimeSeries):
            return False
        if self._max_step != other._max_step:
            return False
        if self._max_wall_time != other._max_wall_time:
            return False
        if self._max_length != other._max_length:
            return False
        if self._plugin_content != other._plugin_content:
            return False
        if self._description != other._description:
            return False
        if self._display_name != other._display_name:
            return False
        return True

    def __hash__(self):
        return hash(
            (
                self._max_step,
                self._max_wall_time,
                self._max_length,
                self._plugin_content,
                self._description,
                self._display_name,
            )
        )

    def __repr__(self):
        return "BlobSequenceTimeSeries(%s)" % ", ".join(
            (
                "max_step=%r" % (self._max_step,),
                "max_wall_time=%r" % (self._max_wall_time,),
                "max_length=%r" % (self._max_length,),
                "plugin_content=%r" % (self._plugin_content,),
                "description=%r" % (self._description,),
                "display_name=%r" % (self._display_name,),
            )
        )


class BlobReference:
    """A reference to a blob.

    Attributes:
      blob_key: A string containing a key uniquely identifying a blob, which
        may be dereferenced via `provider.read_blob(blob_key)`.

        These keys must be constructed such that they can be included directly in
        a URL, with no further encoding. Concretely, this means that they consist
        exclusively of "unreserved characters" per RFC 3986, namely
        [a-zA-Z0-9._~-]. These keys are case-sensitive; it may be wise for
        implementations to normalize case to reduce confusion. The empty string
        is not a valid key.

        Blob keys must not contain information that should be kept secret.
        Privacy-sensitive applications should use random keys (e.g. UUIDs), or
        encrypt keys containing secret fields.
      url: (optional) A string containing a URL from which the blob data may be
        fetched directly, bypassing the data provider. URLs may be a vector
        for data leaks (e.g. via browser history, web proxies, etc.), so these
        URLs should not expose secret information.
    """

    __slots__ = ("_url", "_blob_key")

    def __init__(self, blob_key, url=None):
        self._blob_key = blob_key
        self._url = url

    @property
    def blob_key(self):
        """Provide a key uniquely identifying a blob.

        Callers should consider these keys to be opaque-- i.e., to have
        no intrinsic meaning. Some data providers may use random IDs;
        but others may encode information into the key, in which case
        callers must make no attempt to decode it.
        """
        return self._blob_key

    @property
    def url(self):
        """Provide the direct-access URL for this blob, if available.

        Note that this method is *not* expected to construct a URL to
        the data-loading endpoint provided by TensorBoard. If this
        method returns None, then the caller should proceed to use
        `blob_key()` to build the URL, as needed.
        """
        return self._url

    def __eq__(self, other):
        if not isinstance(other, BlobReference):
            return False
        if self._blob_key != other._blob_key:
            return False
        if self._url != other._url:
            return False
        return True

    def __hash__(self):
        return hash((self._blob_key, self._url))

    def __repr__(self):
        return "BlobReference(%s)" % ", ".join(
            ("blob_key=%r" % (self._blob_key,), "url=%r" % (self._url,))
        )


class BlobSequenceDatum:
    """A single datum in a blob sequence time series for a run and tag.

    Attributes:
      step: The global step at which this datum occurred; an integer. This is a
        unique key among data of this time series.
      wall_time: The real-world time at which this datum occurred, as `float`
        seconds since epoch.
      values: A tuple of `BlobReference` objects, providing access to elements of
        this sequence.
    """

    __slots__ = ("_step", "_wall_time", "_values")

    def __init__(self, step, wall_time, values):
        self._step = step
        self._wall_time = wall_time
        self._values = values

    @property
    def step(self):
        return self._step

    @property
    def wall_time(self):
        return self._wall_time

    @property
    def values(self):
        return self._values

    def __eq__(self, other):
        if not isinstance(other, BlobSequenceDatum):
            return False
        if self._step != other._step:
            return False
        if self._wall_time != other._wall_time:
            return False
        if self._values != other._values:
            return False
        return True

    def __hash__(self):
        return hash((self._step, self._wall_time, self._values))

    def __repr__(self):
        return "BlobSequenceDatum(%s)" % ", ".join(
            (
                "step=%r" % (self._step,),
                "wall_time=%r" % (self._wall_time,),
                "values=%r" % (self._values,),
            )
        )


class RunTagFilter:
    """Filters data by run and tag names."""

    def __init__(self, runs=None, tags=None):
        """Construct a `RunTagFilter`.

        A time series passes this filter if both its run *and* its tag are
        included in the corresponding whitelists.

        Order and multiplicity are ignored; `runs` and `tags` are treated as
        sets.

        Args:
          runs: Collection of run names, as strings, or `None` to admit all
            runs.
          tags: Collection of tag names, as strings, or `None` to admit all
            tags.
        """
        self._runs = self._parse_optional_string_set("runs", runs)
        self._tags = self._parse_optional_string_set("tags", tags)

    def _parse_optional_string_set(self, name, value):
        if value is None:
            return None
        if isinstance(value, str):
            # Prevent confusion: strings _are_ iterable, but as
            # sequences of characters, so this likely signals an error.
            raise TypeError(
                "%s: expected `None` or collection of strings; got %r: %r"
                % (name, type(value), value)
            )
        value = frozenset(value)
        for item in value:
            if not isinstance(item, str):
                raise TypeError(
                    "%s: expected `None` or collection of strings; "
                    "got item of type %r: %r" % (name, type(item), item)
                )
        return value

    @property
    def runs(self):
        return self._runs

    @property
    def tags(self):
        return self._tags

    def __repr__(self):
        return "RunTagFilter(%s)" % ", ".join(
            (
                "runs=%r" % (self._runs,),
                "tags=%r" % (self._tags,),
            )
        )
