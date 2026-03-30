"""Profile training to find remaining hotspots."""

import os

os.environ["KERAS_BACKEND"] = "torch"
import cProfile
import pstats

import torch

import keras
from keras import layers

VOCAB = 1024
SEQ = 64
HDIM = 256
HEADS = 4
NLAYERS = 2
BATCH = 4
N = 30

dev = "mps" if torch.backends.mps.is_available() else "cpu"
data = torch.ones(BATCH, SEQ, dtype=torch.int32, device=dev)
labels = torch.ones(BATCH, SEQ, dtype=torch.int32, device=dev)

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
m = keras.Model(inp, x)
m.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

# Warmup
for _ in range(5):
    m.train_on_batch(data, labels)
torch.mps.synchronize()

# Profile
pr = cProfile.Profile()
pr.enable()
for _ in range(N):
    m.train_on_batch(data, labels)
torch.mps.synchronize()
pr.disable()

stats = pstats.Stats(pr)
stats.sort_stats("cumulative")
print("\n=== TOP 40 BY CUMULATIVE TIME ===")
stats.print_stats(40)
stats.sort_stats("tottime")
print("\n=== TOP 40 BY TOTAL TIME ===")
stats.print_stats(40)
print(f"\nTotal function calls: {stats.total_calls}")
