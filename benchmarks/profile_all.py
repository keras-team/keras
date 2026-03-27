"""Deep cProfile of CNN and LLM for both torch and JAX."""
import os, cProfile, pstats, io, time

BACKEND = os.environ.get("KERAS_BACKEND", "torch")
import numpy as np
import keras
from keras import layers, ops
import torch

# ── models ──────────────────────────────────────────────────────────────
BATCH = 4

# CNN
ci = keras.Input((32, 32, 3))
cx = layers.Conv2D(64, 3, padding="same", activation="relu")(ci)
cx = layers.Conv2D(64, 3, padding="same", activation="relu")(cx)
cx = layers.MaxPooling2D()(cx)
cx = layers.Conv2D(128, 3, padding="same", activation="relu")(cx)
cx = layers.GlobalAveragePooling2D()(cx)
cx = layers.Dense(10)(cx)
cnn = keras.Model(ci, cx)
imgs = ops.convert_to_tensor(np.ones((BATCH, 32, 32, 3), dtype="float32"))

# LLM
VOCAB=256; SEQ=32; HDIM=128; HEADS=2; NLAYERS=1
li = keras.Input((None,), dtype="int32")
lx = layers.Embedding(VOCAB, HDIM)(li)
r = lx
lx = layers.LayerNormalization()(lx)
lx = layers.MultiHeadAttention(HEADS, HDIM//HEADS)(lx, lx, use_causal_mask=True)
lx = lx + r; r = lx
lx = layers.LayerNormalization()(lx)
lx = layers.Dense(HDIM*4, activation="gelu")(lx)
lx = layers.Dense(HDIM)(lx) + r
lx = layers.Dense(VOCAB)(layers.LayerNormalization()(lx))
llm = keras.Model(li, lx)
ids = ops.convert_to_tensor(np.ones((BATCH, SEQ), dtype="int32"))

def sync():
    if BACKEND == "torch":
        if torch.backends.mps.is_available(): torch.mps.synchronize()
    elif BACKEND == "jax":
        import jax; jax.effects_barrier()

# warmup
for _ in range(20): cnn(imgs, training=False)
for _ in range(20): llm(ids, training=False)
sync()

def profile(fn, label, n=200):
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(n): fn()
    pr.disable()
    sync()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(30)
    print(f"\n{'='*70}\n  {label}  ({n} iters)\n{'='*70}")
    print(s.getvalue())

profile(lambda: cnn(imgs, training=False), f"CNN  [{BACKEND}]", 200)
profile(lambda: llm(ids, training=False), f"LLM  [{BACKEND}]", 200)
