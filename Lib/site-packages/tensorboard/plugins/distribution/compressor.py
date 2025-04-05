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
"""Package for histogram compression."""


import dataclasses
import numpy as np

from typing import Tuple

# Normal CDF for std_devs: (-Inf, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, Inf)
# naturally gives bands around median of width 1 std dev, 2 std dev, 3 std dev,
# and then the long tail.
NORMAL_HISTOGRAM_BPS = (0, 668, 1587, 3085, 5000, 6915, 8413, 9332, 10000)


@dataclasses.dataclass(frozen=True)
class CompressedHistogramValue:
    """Represents a value in a compressed histogram.

    Attributes:
      basis_point: Compression point represented in basis point, 1/100th of a
        percent.
      value: Cumulative weight at the basis point.
    """

    basis_point: float
    value: float

    def as_tuple(self) -> Tuple[float, float]:
        """Returns the basis point and the value as a tuple."""
        return (self.basis_point, self.value)


# TODO(@jart): Unfork these methods.
def compress_histogram_proto(histo, bps=NORMAL_HISTOGRAM_BPS):
    """Creates fixed size histogram by adding compression to accumulated state.

    This routine transforms a histogram at a particular step by interpolating its
    variable number of buckets to represent their cumulative weight at a constant
    number of compression points. This significantly reduces the size of the
    histogram and makes it suitable for a two-dimensional area plot where the
    output of this routine constitutes the ranges for a single x coordinate.

    Args:
      histo: A HistogramProto object.
      bps: Compression points represented in basis points, 1/100ths of a percent.
          Defaults to normal distribution.

    Returns:
      List of values for each basis point.
    """
    # See also: Histogram::Percentile() in core/lib/histogram/histogram.cc
    if not histo.num:
        return [CompressedHistogramValue(b, 0.0).as_tuple() for b in bps]
    bucket = np.array(histo.bucket)
    bucket_limit = list(histo.bucket_limit)
    weights = (bucket * bps[-1] / (bucket.sum() or 1.0)).cumsum()
    values = []
    j = 0
    while j < len(bps):
        i = np.searchsorted(weights, bps[j], side="right")
        while i < len(weights):
            cumsum = weights[i]
            cumsum_prev = weights[i - 1] if i > 0 else 0.0
            if cumsum == cumsum_prev:  # prevent lerp divide by zero
                i += 1
                continue
            if not i or not cumsum_prev:
                lhs = histo.min
            else:
                lhs = max(bucket_limit[i - 1], histo.min)
            rhs = min(bucket_limit[i], histo.max)
            weight = _lerp(bps[j], cumsum_prev, cumsum, lhs, rhs)
            values.append(CompressedHistogramValue(bps[j], weight).as_tuple())
            j += 1
            break
        else:
            break
    while j < len(bps):
        values.append(CompressedHistogramValue(bps[j], histo.max).as_tuple())
        j += 1
    return values


def compress_histogram(buckets, bps=NORMAL_HISTOGRAM_BPS):
    """Creates fixed size histogram by adding compression to accumulated state.

    This routine transforms a histogram at a particular step by linearly
    interpolating its variable number of buckets to represent their cumulative
    weight at a constant number of compression points. This significantly reduces
    the size of the histogram and makes it suitable for a two-dimensional area
    plot where the output of this routine constitutes the ranges for a single x
    coordinate.

    Args:
      buckets: A list of buckets, each of which is a 3-tuple of the form
        `(min, max, count)`.
      bps: Compression points represented in basis points, 1/100ths of a percent.
          Defaults to normal distribution.

    Returns:
      List of values for each basis point.
    """
    # See also: Histogram::Percentile() in core/lib/histogram/histogram.cc
    buckets = np.array(buckets)
    if not buckets.size:
        return [CompressedHistogramValue(b, 0.0).as_tuple() for b in bps]
    (minmin, maxmax) = (buckets[0][0], buckets[-1][1])
    counts = buckets[:, 2]
    right_edges = list(buckets[:, 1])
    weights = (counts * bps[-1] / (counts.sum() or 1.0)).cumsum()

    result = []
    bp_index = 0
    while bp_index < len(bps):
        i = np.searchsorted(weights, bps[bp_index], side="right")
        while i < len(weights):
            cumsum = weights[i]
            cumsum_prev = weights[i - 1] if i > 0 else 0.0
            if cumsum == cumsum_prev:  # prevent division-by-zero in `_lerp`
                i += 1
                continue
            if not i or not cumsum_prev:
                lhs = minmin
            else:
                lhs = max(right_edges[i - 1], minmin)
            rhs = min(right_edges[i], maxmax)
            weight = _lerp(bps[bp_index], cumsum_prev, cumsum, lhs, rhs)
            result.append(
                CompressedHistogramValue(bps[bp_index], weight).as_tuple()
            )
            bp_index += 1
            break
        else:
            break
    while bp_index < len(bps):
        result.append(
            CompressedHistogramValue(bps[bp_index], maxmax).as_tuple()
        )
        bp_index += 1
    return result


def _lerp(x, x0, x1, y0, y1):
    """Affinely map from [x0, x1] onto [y0, y1]."""
    return y0 + (x - x0) * float(y1 - y0) / (x1 - x0)
