# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np


def abs(input: np.ndarray) -> np.ndarray:  # noqa: A001
    return np.abs(input)  # type: ignore[no-any-return]
