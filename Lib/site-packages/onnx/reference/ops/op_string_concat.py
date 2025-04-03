# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun

_acceptable_str_dtypes = ("U", "O")


class StringConcat(OpRun):
    def _run(self, x, y):
        if (
            x.dtype.kind not in _acceptable_str_dtypes
            or y.dtype.kind not in _acceptable_str_dtypes
        ):
            raise TypeError(
                f"Inputs must be string tensors, received dtype {x.dtype} and {y.dtype}"
            )
        # As per onnx/mapping.py, object numpy dtype corresponds to TensorProto.STRING
        return (np.char.add(x.astype(np.str_), y.astype(np.str_)).astype(object),)
