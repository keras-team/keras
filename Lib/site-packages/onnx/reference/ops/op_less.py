# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunBinaryComparison


class Less(OpRunBinaryComparison):
    def _run(self, a, b):  # type: ignore
        return (np.less(a, b),)
