"""Measure Functional model __call__ overhead vs direct .call()"""
import time
import torch
import numpy as np
import keras
from keras import layers, ops

VOCAB = 1024; SEQ = 64; HDIM = 256; HEADS = 4; NLAYERS = 2; BATCH = 4

inp = keras.Input((None,), dtype="int32")
x = layers.Embedding(VOCAB, HDIM)(inp)
for _ in range(NLAYERS):
    r = x
    x = layers.LayerNormalization()(x)
    x = layers.MultiHeadAttention(HEADS, HDIM // HEADS)(x, x, use_causal_mask=True)
    x = x + r; r = x
    x = layers.LayerNormalization()(x)
    x = layers.Dense(HDIM * 4, activation="gelu")(x)
    x = layers.Dense(HDIM)(x) + r
x = layers.Dense(VOCAB)(layers.LayerNormalization()(x))
llm = keras.Model(inp, x)

ids = ops.convert_to_tensor(np.ones((BATCH, SEQ), dtype="int32"))

for _ in range(20):
    _ = llm(ids, training=False)
torch.mps.synchronize()

N = 200

# Time full __call__
t0 = time.perf_counter()
for _ in range(N):
    out = llm(ids, training=False)
torch.mps.synchronize()
full_call = (time.perf_counter() - t0) / N * 1e3
print(f"Full __call__:          {full_call:.3f} ms")

# Time .call() directly
t0 = time.perf_counter()
for _ in range(N):
    out = llm.call(ids, training=None, mask=None)
torch.mps.synchronize()
direct_call = (time.perf_counter() - t0) / N * 1e3
print(f"Direct .call():         {direct_call:.3f} ms")

overhead = full_call - direct_call
pct = overhead / full_call * 100
print(f"__call__ overhead:      {overhead:.3f} ms ({pct:.1f}%)")
