"""Compare Keras MHA vs raw torch MHA performance."""

import os

os.environ["KERAS_BACKEND"] = "torch"
import timeit

import torch

from keras import layers

d_model = 256
num_heads = 4
seq_len = 128
batch_size = 2

# Keras MHA
mha = layers.MultiHeadAttention(num_heads, d_model // num_heads)
x = torch.randn(batch_size, seq_len, d_model, device="mps")
mha(x, x)  # build

# Raw torch MHA
torch_mha = torch.nn.MultiheadAttention(
    d_model, num_heads, batch_first=True
).to("mps")


def f_keras():
    return mha(x, x)


def f_torch():
    with torch.no_grad():
        return torch_mha(x, x, x, need_weights=False)


# Warmup
for _ in range(2000):
    f_keras()
    f_torch()

N = 10000
t_keras = timeit.timeit(f_keras, number=N) / N * 1e3
t_torch = timeit.timeit(f_torch, number=N) / N * 1e3
print(f"Keras MHA (4 heads, d={d_model}):     {t_keras:.3f} ms")
print(f"Raw torch MHA (4 heads, d={d_model}): {t_torch:.3f} ms")
print(f"Keras overhead: {(t_keras / t_torch - 1) * 100:.1f}%")

# Also time individual components
q_dense = mha._query_dense  # EinsumDense
k_dense = mha._key_dense
v_dense = mha._value_dense
o_dense = mha._output_dense


def f_einsum():
    q = q_dense(x)
    k_dense(x)
    v_dense(x)
    return o_dense(q)


t_einsum = timeit.timeit(f_einsum, number=N) / N * 1e3
print(f"\nJust 4 EinsumDense calls:    {t_einsum:.3f} ms")
print(f"Rest of MHA (attn compute):  {t_keras - t_einsum:.3f} ms")
