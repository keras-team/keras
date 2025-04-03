# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

import onnx


@dataclass
class TestCase:
    name: str
    model_name: str
    url: str | None
    model_dir: str | None
    model: onnx.ModelProto | None
    data_sets: Sequence[tuple[Sequence[np.ndarray], Sequence[np.ndarray]]] | None
    kind: str
    rtol: float
    atol: float
    # Tell PyTest this isn't a real test.
    __test__: bool = False
