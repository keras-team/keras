# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.op_run import OpRun


class OpRunAiOnnxMl(OpRun):
    op_domain = "ai.onnx.ml"
