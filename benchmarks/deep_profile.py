"""
Deep profiling script for both torch and JAX backends.
Identifies top Python-level hotspots in the Keras call stack.

Usage:
    KERAS_BACKEND=torch python benchmarks/deep_profile.py
    KERAS_BACKEND=jax   python benchmarks/deep_profile.py
"""
import cProfile
import pstats
import io
import os
import sys
import time
import numpy as np

BACKEND = os.environ.get("KERAS_BACKEND", "torch")

import keras
from keras import layers, ops

VOCAB = 1024; SEQ = 64; HDIM = 256; HEADS = 4; NLAYERS = 2; BATCH = 4

# ── Build LLM model ────────────────────────────────────────────────────
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


def sync():
    if BACKEND == "torch":
        import torch
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
    elif BACKEND == "jax":
        import jax
        jax.effects_barrier()


# Warmup
for _ in range(20):
    out = llm(ids, training=False)
sync()

# ── Profile 50 iters ───────────────────────────────────────────────────
pr = cProfile.Profile()
pr.enable()
for _ in range(50):
    out = llm(ids, training=False)
sync()
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
ps.print_stats(50)
full = s.getvalue()

print(f"\n{'='*70}")
print(f"  BACKEND={BACKEND}  LLM Forward  50 iters")
print(f"{'='*70}")

# Show full top-50 by tottime
print(full)

# Also print by cumulative to see call chains
print(f"\n{'='*70}")
print(f"  TOP 30 by cumtime (call chains)")
print(f"{'='*70}")
s2 = io.StringIO()
ps2 = pstats.Stats(pr, stream=s2).sort_stats("cumulative")
ps2.print_stats(30)
print(s2.getvalue())

# ── Wall time measurement ───────────────────────────────────────────────
N = 100
t0 = time.perf_counter()
for _ in range(N):
    out = llm(ids, training=False)
sync()
ms = (time.perf_counter() - t0) / N * 1e3
print(f"\n  Median wall time: ~{ms:.2f} ms / forward  ({N} iters)")
