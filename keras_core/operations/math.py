"""
segment_sum
top_k
"""

from keras_core import backend
from keras_core.backend import any_symbolic_tensors
from keras_core.operations.operation import Operation


class SegmentSum(Operation):
    def call(self, x, segment_ids, num_segments=None, sorted=False):
        return backend.math.segment_sum(x, segment_ids, num_segments, sorted)


def segment_sum(x, segment_ids, num_segments=None, sorted=False):
    if any_symbolic_tensors((x,)):
        return SegmentSum().symbolic_call(x, segment_ids, num_segments, sorted)
    return backend.math.segment_sum(x, segment_ids, num_segments, sorted)


class TopK(Operation):
    def call(self, x, k, sorted=True):
        return backend.math.top_k(x, k, sorted)


def top_k(x, k, sorted=True):
    if any_symbolic_tensors((x,)):
        return TopK().symbolic_call(x, k, sorted)
    return backend.math.top_k(x, k, sorted)
