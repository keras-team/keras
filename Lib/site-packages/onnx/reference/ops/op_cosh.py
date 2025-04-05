# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


class Cosh(OpRunUnaryNum):
    def _run(self, x):  # type: ignore
        return (np.cosh(x),)
