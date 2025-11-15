from typing import Any

import numpy as np

from keras import ops


def _maybe_convert_to_int(x: Any) -> Any:
    if isinstance(x, int):
        return x
    if isinstance(x, (tuple, list)):
        try:
            return tuple(int(v) for v in x)
        except Exception:
            return x

    try:
        np_val = ops.convert_to_numpy(x)
    except Exception:
        return x

    if np.isscalar(np_val):
        try:
            return int(np_val)
        except Exception:
            return x

    arr = np.asarray(np_val).ravel()
    if arr.size == 0:
        return x
    if arr.size == 1:
        return int(arr[0])
    try:
        return tuple(int(v) for v in arr.tolist())
    except Exception:
        return x
