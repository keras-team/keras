"""Utilities for preprocessing sequence data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_preprocessing import sequence

pad_sequences = sequence.pad_sequences
make_sampling_table = sequence.make_sampling_table
skipgrams = sequence.skipgrams
_remove_long_seq = sequence._remove_long_seq  # TODO: make it public?
TimeseriesGenerator = sequence.TimeseriesGenerator
