import os

os.environ["KERAS_BACKEND"] = "torch"
import timeit

import torch

import keras
from keras import layers

mha = layers.MultiHeadAttention(4, 32)
x = torch.randn(4, 64, 128, device="mps")
mha(x, x)  # build
print("MHA _fast_call:", mha._fast_call)


def f_call():
    return mha(x, x)


def f_direct():
    return mha.call(x, x)


for _ in range(2000):
    f_call()
t1 = timeit.timeit(f_call, number=10000) / 10000 * 1e3
t2 = timeit.timeit(f_direct, number=10000) / 10000 * 1e3
print(
    f"MHA.__call__(x,x): {t1:.3f} ms (overhead vs direct: {(t1 - t2) / t1 * 100:.1f}%)"
)
print(f"MHA.call(x,x):     {t2:.3f} ms")

# Single-arg path
dense = layers.Dense(64)
y = torch.randn(4, 128, device="mps")
dense(y)
print("Dense _fast_call:", dense._fast_call)


def fd():
    return dense(y)


td = timeit.timeit(fd, number=10000) / 10000 * 1e3
print(f"Dense.__call__(y): {td:.3f} ms")

# Functional model
inputs = keras.Input(shape=(128,))
out = layers.Dense(64)(inputs)
out = layers.Dense(32)(out)
model = keras.Model(inputs, out)
res = model(torch.randn(4, 128, device="mps"))
print("Functional model OK, shape:", res.shape)

# Masking flows through correctly (MHA with mask)
mask_in = torch.ones(4, 64, dtype=torch.bool, device="mps")
mask_in[:, 50:] = False
from keras.src.backend import set_keras_mask

set_keras_mask(x, mask_in)
out_masked = mha(x, x)
from keras.src.backend import get_keras_mask

print(
    "Masked MHA output mask:", get_keras_mask(out_masked)
)  # should be None (MHA doesn't produce mask by default)
