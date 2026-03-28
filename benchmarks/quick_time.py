"""Quick timing for both backends."""

import os
import time

import numpy as np

BACKEND = os.environ.get("KERAS_BACKEND", "torch")

import keras
from keras import layers
from keras import ops

VOCAB = 1024
SEQ = 64
HDIM = 256
HEADS = 4
NLAYERS = 2
BATCH = 4

inp = keras.Input((None,), dtype="int32")
x = layers.Embedding(VOCAB, HDIM)(inp)
for _ in range(NLAYERS):
    r = x
    x = layers.LayerNormalization()(x)
    x = layers.MultiHeadAttention(HEADS, HDIM // HEADS)(
        x, x, use_causal_mask=True
    )
    x = x + r
    r = x
    x = layers.LayerNormalization()(x)
    x = layers.Dense(HDIM * 4, activation="gelu")(x)
    x = layers.Dense(HDIM)(x) + r
x = layers.Dense(VOCAB)(layers.LayerNormalization()(x))
llm = keras.Model(inp, x)

ids = ops.convert_to_tensor(np.ones((BATCH, SEQ), dtype="int32"))


def sync():
    if BACKEND == "torch":
        import torch

        if torch.backends.mps.is_available():
            torch.mps.synchronize()
    elif BACKEND == "jax":
        import jax

        jax.effects_barrier()


for _ in range(30):
    llm(ids, training=False)
sync()

N = 100
t0 = time.perf_counter()
for _ in range(N):
    llm(ids, training=False)
sync()
ms = (time.perf_counter() - t0) / N * 1e3
print(f"BACKEND={BACKEND}  LLM forward: {ms:.2f} ms")
