# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunBinary


class And(OpRunBinary):
    def _run(self, x, y):  # type: ignore
        return (np.logical_and(x, y),)
