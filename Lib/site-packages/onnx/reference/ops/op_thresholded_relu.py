# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


class ThresholdedRelu(OpRunUnaryNum):
    def _run(self, x, alpha=None):  # type: ignore
        alpha = alpha or self.alpha  # type: ignore
        return (np.where(x > alpha, x, 0).astype(x.dtype),)  # type: ignore
