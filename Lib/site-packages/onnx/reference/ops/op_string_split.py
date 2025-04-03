# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun

_acceptable_str_dtypes = ("U", "O")


def pad_empty_string(
    split_lists: list | np.ndarray, padding_requirement: list | int
) -> list:
    if isinstance(split_lists, list):
        return split_lists + ["" for _ in range(padding_requirement)]
    if isinstance(split_lists, np.ndarray):
        return list(map(pad_empty_string, split_lists, padding_requirement))
    raise TypeError(f"Invalid array type '{type(split_lists)}'")


def split_with_padding(x, separator=None, maxsplit=None):
    split_lists = np.char.split(x.astype(np.str_), separator, maxsplit)
    # Find the maximum length after splitting
    num_splits = np.vectorize(len, otypes=[np.int64])(split_lists)
    padding_requirement = (np.max(num_splits, initial=0) - num_splits).tolist()
    # Add padding to lists that are shorter than the maximum length
    split_lists_padded = np.array(
        pad_empty_string(split_lists, padding_requirement), dtype=object
    )
    if x.size == 0:
        split_lists_padded = split_lists_padded.reshape(*x.shape, 0)
    return split_lists_padded, num_splits


class StringSplit(OpRun):
    def _run(self, x, delimiter=None, maxsplit=None):
        if delimiter == "":
            delimiter = None

        if x.dtype.kind not in _acceptable_str_dtypes:
            raise TypeError(f"Inputs must be string tensors, received dtype {x.dtype}")
        return split_with_padding(x, delimiter, maxsplit)
