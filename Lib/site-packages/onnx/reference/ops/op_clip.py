# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class Clip_6(OpRun):
    def _run(self, data, min=None, max=None):  # type: ignore  # noqa: A002
        amin = min
        amax = max
        res = data if amin is amax is None else np.clip(data, amin, amax)  # type: ignore
        return (res,) if res.dtype == data.dtype else (res.astype(data.dtype),)


class Clip_11(OpRun):
    def _run(self, data, *minmax):  # type: ignore
        le = len(minmax)
        amin = minmax[0] if le > 0 else None
        amax = minmax[1] if le > 1 else None
        res = data if amin is amax is None else np.clip(data, amin, amax)
        res = (res,) if res.dtype == data.dtype else (res.astype(data.dtype),)
        return res
