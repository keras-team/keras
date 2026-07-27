"""Tests for PyTorch backend nn utilities."""

import numpy as np
import pytest
import torch

from keras.src import backend
from keras.src import ops
from keras.src import testing


@pytest.mark.skipif(
    backend.backend() != "torch",
    reason="This test is only applicable to the PyTorch backend.",
)
class DotProductAttentionCompileTest(testing.TestCase):
    def _qkv(self, batch=2, seq=16, heads=4, head_dim=8):
        rng = np.random.default_rng(0)
        shape = (batch, seq, heads, head_dim)
        return tuple(
            ops.convert_to_tensor(rng.standard_normal(shape).astype("float32"))
            for _ in range(3)
        )

    def test_compiles_without_graph_break(self):
        # `fullgraph=True` turns a graph break into an error, which is what
        # makes this a regression test rather than a smoke test.
        query, key, value = self._qkv()

        def attend():
            return ops.dot_product_attention(query, key, value, is_causal=True)

        compiled = torch.compile(attend, fullgraph=True)
        self.assertAllClose(compiled(), attend())

    def test_compiled_matches_eager(self):
        query, key, value = self._qkv()
        rng = np.random.default_rng(1)
        mask = ops.convert_to_tensor(
            rng.integers(0, 2, (2, 1, 16, 16)).astype("bool")
        )
        bias = ops.convert_to_tensor(
            rng.standard_normal((2, 1, 16, 16)).astype("float32")
        )

        for kwargs in (
            {},
            {"is_causal": True},
            {"mask": mask},
            {"mask": mask, "is_causal": True},
            {"bias": bias},
            {"scale": 0.125},
        ):

            def attend(kwargs=kwargs):
                return ops.dot_product_attention(query, key, value, **kwargs)

            self.assertAllClose(torch.compile(attend)(), attend())
