# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.op_run import OpRun


class SequenceConstruct(OpRun):
    def _run(self, *data):  # type: ignore
        return (list(data),)
