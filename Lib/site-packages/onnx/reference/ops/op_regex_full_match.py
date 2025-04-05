# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun

_acceptable_str_dtypes = ("U", "O")


class RegexFullMatch(OpRun):
    def _run(self, x, pattern=None):
        try:
            import re2
        except ImportError as e:
            raise ImportError(
                "re2 must be installed to use the reference implementation of the RegexFullMatch operator"
            ) from e

            # As per onnx/mapping.py, object numpy dtype corresponds to TensorProto.STRING
        if x.dtype.kind not in _acceptable_str_dtypes:
            raise TypeError(f"Input must be string tensor, received dtype {x.dtype}")
        try:
            regex = re2.compile(pattern)
        except re2.error as e:
            raise ValueError(f"Invalid regex pattern {pattern!r}") from e

        fullmatch_func = np.vectorize(
            lambda x: regex.fullmatch(x) is not None, otypes=[np.bool_]
        )
        return (fullmatch_func(x),)
