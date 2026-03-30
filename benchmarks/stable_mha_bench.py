"""Stable MHA comparison: old (einsum) vs new (matmul) path."""

import os

os.environ["KERAS_BACKEND"] = "torch"
import timeit

import torch

from keras import layers

# Build MHA
d_model, num_heads, seq_len, batch = 256, 4, 128, 2
mha_old = layers.MultiHeadAttention(num_heads, d_model // num_heads)
mha_new = layers.MultiHeadAttention(num_heads, d_model // num_heads)
x = torch.randn(batch, seq_len, d_model, device="mps")
mha_old(x, x)
mha_new(x, x)

# Disable matmul path on old to compare
for dense in [
    mha_old._query_dense,
    mha_old._key_dense,
    mha_old._value_dense,
    mha_old._output_dense,
]:
    dense._matmul_n_input_free = None

print("MHA sub-layer matmul path status:")
for name, dense in [
    ("query", mha_new._query_dense),
    ("key", mha_new._key_dense),
    ("value", mha_new._value_dense),
    ("output", mha_new._output_dense),
]:
    print(f"  {name}: _matmul_n_input_free={dense._matmul_n_input_free}")

# Stable benchmark (many iterations)
for _ in range(5000):
    mha_old(x, x)
    mha_new(x, x)

import statistics

N = 5000
times_old = [
    timeit.timeit(lambda: mha_old(x, x), number=1) * 1e3 for _ in range(N)
]
times_new = [
    timeit.timeit(lambda: mha_new(x, x), number=1) * 1e3 for _ in range(N)
]

# Sort and take median
times_old.sort()
times_new.sort()
mid = N // 2
print(
    f"\nMHA median: einsum={times_old[mid]:.3f}ms, matmul={times_new[mid]:.3f}ms"
)
print(f"Speedup: {times_old[mid] / times_new[mid]:.2f}x")
print(
    f"Mean  : einsum={statistics.mean(times_old):.3f}ms, matmul={statistics.mean(times_new):.3f}ms"
)
print(
    f"p5    : einsum={times_old[N // 20]:.3f}ms, matmul={times_new[N // 20]:.3f}ms"
)
print(
    f"p95   : einsum={times_old[int(N * 0.95)]:.3f}ms, matmul={times_new[int(N * 0.95)]:.3f}ms"
)
