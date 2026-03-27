"""Verify EinsumDense matmul fast path correctness and speed."""
import os
os.environ["KERAS_BACKEND"] = "torch"
import torch
import timeit
import keras
from keras.src.layers.core.einsum_dense import EinsumDense

print("=== Correctness Tests ===")

# Test abc,cde->abde (MHA Q/K/V projection)
ed1 = EinsumDense("abc,cde->abde", output_shape=(None, 4, 32), bias_axes="de")
x1 = torch.randn(2, 64, 128, device="mps")
ed1(x1)
print(f"abc,cde->abde: _matmul_n_input_free={ed1._matmul_n_input_free}")

# Test abcd,cde->abe (MHA output projection)
ed2 = EinsumDense("abcd,cde->abe", output_shape=(None, 128), bias_axes="e")
x2 = torch.randn(2, 64, 4, 32, device="mps")
ed2(x2)
print(f"abcd,cde->abe: _matmul_n_input_free={ed2._matmul_n_input_free}")

# Test simple ab,bc->ac (like Dense layer)
ed3 = EinsumDense("ab,bc->ac", output_shape=(64,), bias_axes="c")
x3 = torch.randn(4, 128, device="mps")
ed3(x3)
print(f"ab,bc->ac: _matmul_n_input_free={ed3._matmul_n_input_free}")

# Correctness: numerical comparison
ed_ref = EinsumDense("abc,cde->abde", output_shape=(None, 4, 32))
ed_fast = EinsumDense("abc,cde->abde", output_shape=(None, 4, 32))
x = torch.randn(2, 64, 128, device="mps")
ed_ref(x); ed_fast(x)
ed_fast.kernel.assign(ed_ref.kernel.value)
# Force einsum path on ref
ed_ref._matmul_n_input_free = None

out_ref = ed_ref(x)
out_fast = ed_fast(x)
print(f"\nabc,cde->abde outputs match: {torch.allclose(out_ref, out_fast, atol=1e-4)}")

# Test abcd,cde->abe correctness
ed_ref2 = EinsumDense("abcd,cde->abe", output_shape=(None, 128))
ed_fast2 = EinsumDense("abcd,cde->abe", output_shape=(None, 128))
x2 = torch.randn(2, 64, 4, 32, device="mps")
ed_ref2(x2); ed_fast2(x2)
ed_fast2.kernel.assign(ed_ref2.kernel.value)
ed_ref2._matmul_n_input_free = None
out_ref2 = ed_ref2(x2)
out_fast2 = ed_fast2(x2)
print(f"abcd,cde->abe outputs match: {torch.allclose(out_ref2, out_fast2, atol=1e-4)}")

print("\n=== Speed Tests ===")
# Speed: einsum vs matmul path for abc,cde->abde
ed_einsum = EinsumDense("abc,cde->abde", output_shape=(None, 4, 32))
ed_matmul = EinsumDense("abc,cde->abde", output_shape=(None, 4, 32))
x = torch.randn(2, 128, 256, device="mps")
ed_einsum(x); ed_matmul(x)
ed_einsum._matmul_n_input_free = None  # force einsum

for _ in range(5000):
    ed_einsum(x)
    ed_matmul(x)

N = 20000
t_einsum = timeit.timeit(lambda: ed_einsum(x), number=N) / N * 1e6
t_matmul = timeit.timeit(lambda: ed_matmul(x), number=N) / N * 1e6
print(f"abc,cde->abde (einsum path):  {t_einsum:.1f} us")
print(f"abc,cde->abde (matmul path):  {t_matmul:.1f} us")
print(f"Speedup: {t_einsum/t_matmul:.2f}x")

# Full MHA comparison
from keras import layers
mha_old = layers.MultiHeadAttention(4, 64)
mha_new = layers.MultiHeadAttention(4, 64)
x_mha = torch.randn(2, 128, 256, device="mps")
mha_old(x_mha, x_mha)
mha_new(x_mha, x_mha)
# Disable matmul path on old
for dense in [mha_old._query_dense, mha_old._key_dense, mha_old._value_dense, mha_old._output_dense]:
    dense._matmul_n_input_free = None

for _ in range(2000):
    mha_old(x_mha, x_mha)
    mha_new(x_mha, x_mha)

N = 10000
t_old = timeit.timeit(lambda: mha_old(x_mha, x_mha), number=N) / N * 1e3
t_new = timeit.timeit(lambda: mha_new(x_mha, x_mha), number=N) / N * 1e3
print(f"\nFull MHA (einsum path):  {t_old:.3f} ms")
print(f"Full MHA (matmul path):  {t_new:.3f} ms")
print(f"MHA speedup: {t_old/t_new:.2f}x")
