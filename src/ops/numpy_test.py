# src/ops/numpy_test.py

import numpy as np
from decompositions.openvino.numpy_ops import logspace, _dummy_evaluate


def test_logspace_basic():
    """Test logspace returns expected numpy array (placeholder)."""
    node = logspace(0, 2, num=3, dtype=np.float32)
    result = _dummy_evaluate(node)
    expected = np.array([1.0, 10.0, 100.0], dtype=np.float32)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
