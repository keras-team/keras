"""
segment_sum
top_k
in_top_k
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


class InTopK(Operation):
    def call(self, targets, predictions, k):
        return backend.math.in_top_k(targets, predictions, k)


def in_top_k(targets, predictions, k):
    if any_symbolic_tensors((targets, predictions)):
        return InTopK().symbolic_call(targets, predictions, k)
    return backend.math.in_top_k(targets, predictions, k)
