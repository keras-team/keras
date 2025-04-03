# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.op_run import OpRun


class Identity(OpRun):
    def _run(self, a):  # type: ignore
        if a is None:
            return (None,)
        return (a.copy(),)
