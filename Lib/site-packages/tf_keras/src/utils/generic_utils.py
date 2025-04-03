# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Python utilities required by TF-Keras."""

import binascii
import codecs
import importlib
import marshal
import os
import re
import sys
import time
import types as python_types

import numpy as np
import tensorflow.compat.v2 as tf

from tf_keras.src.utils import io_utils
from tf_keras.src.utils import tf_inspect

# isort: off
from tensorflow.python.util.tf_export import keras_export


def func_dump(func):
    """Serializes a user defined function.

    Args:
        func: the function to serialize.

    Returns:
        A tuple `(code, defaults, closure)`.
    """
    if os.name == "nt":
        raw_code = marshal.dumps(func.__code__).replace(b"\\", b"/")
        code = codecs.encode(raw_code, "base64").decode("ascii")
    else:
        raw_code = marshal.dumps(func.__code__)
        code = codecs.encode(raw_code, "base64").decode("ascii")
    defaults = func.__defaults__
    if func.__closure__:
        closure = tuple(c.cell_contents for c in func.__closure__)
    else:
        closure = None
    return code, defaults, closure


def func_load(code, defaults=None, closure=None, globs=None):
    """Deserializes a user defined function.

    Args:
        code: bytecode of the function.
        defaults: defaults of the function.
        closure: closure of the function.
        globs: dictionary of global objects.

    Returns:
        A function object.
    """
    if isinstance(code, (tuple, list)):  # unpack previous dump
        code, defaults, closure = code
        if isinstance(defaults, list):
            defaults = tuple(defaults)

    def ensure_value_to_cell(value):
        """Ensures that a value is converted to a python cell object.

        Args:
            value: Any value that needs to be casted to the cell type

        Returns:
            A value wrapped as a cell object (see function "func_load")
        """

        def dummy_fn():
            value  # just access it so it gets captured in .__closure__

        cell_value = dummy_fn.__closure__[0]
        if not isinstance(value, type(cell_value)):
            return cell_value
        return value

    if closure is not None:
        closure = tuple(ensure_value_to_cell(_) for _ in closure)
    try:
        raw_code = codecs.decode(code.encode("ascii"), "base64")
    except (UnicodeEncodeError, binascii.Error):
        raw_code = code.encode("raw_unicode_escape")
    code = marshal.loads(raw_code)
    if globs is None:
        globs = globals()
    return python_types.FunctionType(
        code, globs, name=code.co_name, argdefs=defaults, closure=closure
    )


def has_arg(fn, name, accept_all=False):
    """Checks if a callable accepts a given keyword argument.

    Args:
        fn: Callable to inspect.
        name: Check if `fn` can be called with `name` as a keyword argument.
        accept_all: What to return if there is no parameter called `name` but
          the function accepts a `**kwargs` argument.

    Returns:
        bool, whether `fn` accepts a `name` keyword argument.
    """
    arg_spec = tf_inspect.getfullargspec(fn)
    if accept_all and arg_spec.varkw is not None:
        return True
    return name in arg_spec.args or name in arg_spec.kwonlyargs


@keras_export("keras.utils.Progbar")
class Progbar:
    """Displays a progress bar.

    Args:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that should *not*
          be averaged over time. Metrics in this list will be displayed as-is.
          All others will be averaged by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
        unit_name: Display name for step counts (usually "step" or "sample").
    """

    def __init__(
        self,
        target,
        width=30,
        verbose=1,
        interval=0.05,
        stateful_metrics=None,
        unit_name="step",
    ):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        self.unit_name = unit_name
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = (
            (hasattr(sys.stdout, "isatty") and sys.stdout.isatty())
            or "ipykernel" in sys.modules
            or "posix" in sys.modules
            or "PYCHARM_HOSTED" in os.environ
        )
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0
        self._time_at_epoch_start = self._start
        self._time_at_epoch_end = None
        self._time_after_first_step = None

    def update(self, current, values=None, finalize=None):
        """Updates the progress bar.

        Args:
            current: Index of current step.
            values: List of tuples: `(name, value_for_last_step)`. If `name` is
              in `stateful_metrics`, `value_for_last_step` will be displayed
              as-is. Else, an average of the metric over time will be
              displayed.
            finalize: Whether this is the last update for the progress bar. If
              `None`, uses `current >= self.target`. Defaults to `None`.
        """
        if finalize is None:
            if self.target is None:
                finalize = False
            else:
                finalize = current >= self.target

        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                # In the case that progress bar doesn't have a target value in
                # the first epoch, both on_batch_end and on_epoch_end will be
                # called, which will cause 'current' and 'self._seen_so_far' to
                # have the same value. Force the minimal value to 1 here,
                # otherwise stateful_metric will be 0s.
                value_base = max(current - self._seen_so_far, 1)
                if k not in self._values:
                    self._values[k] = [v * value_base, value_base]
                else:
                    self._values[k][0] += v * value_base
                    self._values[k][1] += value_base
            else:
                # Stateful metrics output a numeric value. This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]
        self._seen_so_far = current

        message = ""
        now = time.time()
        info = f" - {now - self._start:.0f}s"
        if current == self.target:
            self._time_at_epoch_end = now
        if self.verbose == 1:
            if now - self._last_update < self.interval and not finalize:
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                message += "\b" * prev_total_width
                message += "\r"
            else:
                message += "\n"

            if self.target is not None:
                numdigits = int(np.log10(self.target)) + 1
                bar = ("%" + str(numdigits) + "d/%d [") % (current, self.target)
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += "=" * (prog_width - 1)
                    if current < self.target:
                        bar += ">"
                    else:
                        bar += "="
                bar += "." * (self.width - prog_width)
                bar += "]"
            else:
                bar = "%7d/Unknown" % current

            self._total_width = len(bar)
            message += bar

            time_per_unit = self._estimate_step_duration(current, now)

            if self.target is None or finalize:
                info += self._format_time(time_per_unit, self.unit_name)
            else:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = "%d:%02d:%02d" % (
                        eta // 3600,
                        (eta % 3600) // 60,
                        eta % 60,
                    )
                elif eta > 60:
                    eta_format = "%d:%02d" % (eta // 60, eta % 60)
                else:
                    eta_format = "%ds" % eta

                info = f" - ETA: {eta_format}"

            for k in self._values_order:
                info += f" - {k}:"
                if isinstance(self._values[k], list):
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1])
                    )
                    if abs(avg) > 1e-3:
                        info += f" {avg:.4f}"
                    else:
                        info += f" {avg:.4e}"
                else:
                    info += f" {self._values[k]}"

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += " " * (prev_total_width - self._total_width)

            if finalize:
                info += "\n"

            message += info
            io_utils.print_msg(message, line_break=False)
            message = ""

        elif self.verbose == 2:
            if finalize:
                numdigits = int(np.log10(self.target)) + 1
                count = ("%" + str(numdigits) + "d/%d") % (current, self.target)
                info = count + info
                for k in self._values_order:
                    info += f" - {k}:"
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1])
                    )
                    if avg > 1e-3:
                        info += f" {avg:.4f}"
                    else:
                        info += f" {avg:.4e}"
                if self._time_at_epoch_end:
                    time_per_epoch = (
                        self._time_at_epoch_end - self._time_at_epoch_start
                    )
                    avg_time_per_step = time_per_epoch / self.target
                    self._time_at_epoch_start = now
                    self._time_at_epoch_end = None
                    info += " -" + self._format_time(time_per_epoch, "epoch")
                    info += " -" + self._format_time(
                        avg_time_per_step, self.unit_name
                    )
                    info += "\n"
                message += info
                io_utils.print_msg(message, line_break=False)
                message = ""

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)

    def _format_time(self, time_per_unit, unit_name):
        """format a given duration to display to the user.

        Given the duration, this function formats it in either milliseconds
        or seconds and displays the unit (i.e. ms/step or s/epoch)
        Args:
          time_per_unit: the duration to display
          unit_name: the name of the unit to display
        Returns:
          a string with the correctly formatted duration and units
        """
        formatted = ""
        if time_per_unit >= 1 or time_per_unit == 0:
            formatted += f" {time_per_unit:.0f}s/{unit_name}"
        elif time_per_unit >= 1e-3:
            formatted += f" {time_per_unit * 1000.0:.0f}ms/{unit_name}"
        else:
            formatted += f" {time_per_unit * 1000000.0:.0f}us/{unit_name}"
        return formatted

    def _estimate_step_duration(self, current, now):
        """Estimate the duration of a single step.

        Given the step number `current` and the corresponding time `now` this
        function returns an estimate for how long a single step takes. If this
        is called before one step has been completed (i.e. `current == 0`) then
        zero is given as an estimate. The duration estimate ignores the duration
        of the (assumed to be non-representative) first step for estimates when
        more steps are available (i.e. `current>1`).

        Args:
          current: Index of current step.
          now: The current time.

        Returns: Estimate of the duration of a single step.
        """
        if current:
            # there are a few special scenarios here:
            # 1) somebody is calling the progress bar without ever supplying
            #    step 1
            # 2) somebody is calling the progress bar and supplies step one
            #    multiple times, e.g. as part of a finalizing call
            # in these cases, we just fall back to the simple calculation
            if self._time_after_first_step is not None and current > 1:
                time_per_unit = (now - self._time_after_first_step) / (
                    current - 1
                )
            else:
                time_per_unit = (now - self._start) / current

            if current == 1:
                self._time_after_first_step = now
            return time_per_unit
        else:
            return 0

    def _update_stateful_metrics(self, stateful_metrics):
        self.stateful_metrics = self.stateful_metrics.union(stateful_metrics)


def make_batches(size, batch_size):
    """Returns a list of batch indices (tuples of indices).

    Args:
        size: Integer, total size of the data to slice into batches.
        batch_size: Integer, batch size.

    Returns:
        A list of tuples of array indices.
    """
    num_batches = int(np.ceil(size / float(batch_size)))
    return [
        (i * batch_size, min(size, (i + 1) * batch_size))
        for i in range(0, num_batches)
    ]


def slice_arrays(arrays, start=None, stop=None):
    """Slice an array or list of arrays.

    This takes an array-like, or a list of
    array-likes, and outputs:
        - arrays[start:stop] if `arrays` is an array-like
        - [x[start:stop] for x in arrays] if `arrays` is a list

    Can also work on list/array of indices: `slice_arrays(x, indices)`

    Args:
        arrays: Single array or list of arrays.
        start: can be an integer index (start index) or a list/array of indices
        stop: integer (stop index); should be None if `start` was a list.

    Returns:
        A slice of the array(s).

    Raises:
        ValueError: If the value of start is a list and stop is not None.
    """
    if arrays is None:
        return [None]
    if isinstance(start, list) and stop is not None:
        raise ValueError(
            "The stop argument has to be None if the value of start "
            f"is a list. Received start={start}, stop={stop}"
        )
    elif isinstance(arrays, list):
        if hasattr(start, "__len__"):
            # hdf5 datasets only support list objects as indices
            if hasattr(start, "shape"):
                start = start.tolist()
            return [None if x is None else x[start] for x in arrays]
        return [
            None
            if x is None
            else None
            if not hasattr(x, "__getitem__")
            else x[start:stop]
            for x in arrays
        ]
    else:
        if hasattr(start, "__len__"):
            if hasattr(start, "shape"):
                start = start.tolist()
            return arrays[start]
        if hasattr(start, "__getitem__"):
            return arrays[start:stop]
        return [None]


def to_list(x):
    """Normalizes a list/tensor into a list.

    If a tensor is passed, we return
    a list of size 1 containing the tensor.

    Args:
        x: target object to be normalized.

    Returns:
        A list.
    """
    if isinstance(x, list):
        return x
    return [x]


def to_snake_case(name):
    intermediate = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    insecure = re.sub("([a-z])([A-Z])", r"\1_\2", intermediate).lower()
    # If the class is private the name starts with "_" which is not secure
    # for creating scopes. We prefix the name with "private" in this case.
    if insecure[0] != "_":
        return insecure
    return "private" + insecure


def is_all_none(structure):
    iterable = tf.nest.flatten(structure)
    # We cannot use Python's `any` because the iterable may return Tensors.
    for element in iterable:
        if element is not None:
            return False
    return True


def check_for_unexpected_keys(name, input_dict, expected_values):
    unknown = set(input_dict.keys()).difference(expected_values)
    if unknown:
        raise ValueError(
            f"Unknown entries in {name} dictionary: {list(unknown)}. "
            f"Only expected following keys: {expected_values}"
        )


def validate_kwargs(
    kwargs, allowed_kwargs, error_message="Keyword argument not understood:"
):
    """Checks that all keyword arguments are in the set of allowed keys."""
    for kwarg in kwargs:
        if kwarg not in allowed_kwargs:
            raise TypeError(error_message, kwarg)


def default(method):
    """Decorates a method to detect overrides in subclasses."""
    method._is_default = True
    return method


def is_default(method):
    """Check if a method is decorated with the `default` wrapper."""
    return getattr(method, "_is_default", False)


def populate_dict_with_module_objects(target_dict, modules, obj_filter):
    for module in modules:
        for name in dir(module):
            obj = getattr(module, name)
            if obj_filter(obj):
                target_dict[name] = obj


class LazyLoader(python_types.ModuleType):
    """Lazily import a module, mainly to avoid pulling in large dependencies."""

    def __init__(self, local_name, parent_module_globals, name):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        super().__init__(name)

    def _load(self):
        """Load the module and insert it into the parent's globals."""
        # Import the target module and insert it into the parent's namespace
        module = importlib.import_module(self.__name__)
        self._parent_module_globals[self._local_name] = module
        # Update this object's dict so that if someone keeps a reference to the
        # LazyLoader, lookups are efficient (__getattr__ is only called on
        # lookups that fail).
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)

