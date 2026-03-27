import os
os.environ["KERAS_BACKEND"] = "torch"
from keras import layers
import torch
import timeit

d_model, num_heads, seq_len, batch = 256, 4, 128, 2
mha = layers.MultiHeadAttention(num_heads, d_model // num_heads)
x = torch.randn(batch, seq_len, d_model, device="mps")
mha(x, x)

q_dense = mha._query_dense
print("Query EinsumDense equation:", q_dense.equation)
print("Bias axes:", q_dense.bias_axes)

W = q_dense.kernel.value.detach()
eq = q_dense.equation  # typically 'abc,cde->abde' or similar

def f_einsum():
    return torch.einsum(eq, x, W)


# Equivalent with reshape+matmul
# x: (B, S, d_model) × W: (d_model, num_heads, head_dim) → (B, S, num_heads, head_dim)
def f_matmul():
    bs, sl, dm = x.shape
    return torch.matmul(x.reshape(-1, dm), W.reshape(dm, -1)).reshape(
        bs, sl, num_heads, -1
    )


# Check outputs match
o1 = f_einsum()
o2 = f_matmul()
print("Outputs match:", torch.allclose(o1, o2, atol=1e-5))

for _ in range(2000):
    f_einsum()
    f_matmul()
N = 20000
t1 = timeit.timeit(f_einsum, number=N) / N * 1e6
t2 = timeit.timeit(f_matmul, number=N) / N * 1e6
print(f"torch.einsum: {t1:.3f} us")
print(f"torch.matmul (reshape): {t2:.3f} us")
print(f"einsum overhead: {(t1/t2-1)*100:.1f}%")
