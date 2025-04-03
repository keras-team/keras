# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class Det(OpRun):
    def _run(self, x):  # type: ignore
        return (np.array(np.linalg.det(x)),)
