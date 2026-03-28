"""Trace torch._pytree.unflatten call sites."""

import os

os.environ["KERAS_BACKEND"] = "torch"
import traceback

import numpy as np
import torch.utils._pytree as pytree

orig_unflatten = pytree.TreeSpec.unflatten
hits = []


def traced(self, leaves):
    t = traceback.extract_stack()
    frame = "".join(traceback.format_list(t[-6:-1]))
    hits.append(frame)
    return orig_unflatten(self, leaves)


pytree.TreeSpec.unflatten = traced

import keras
from keras import layers
from keras import ops

VOCAB = 256
SEQ = 32
HDIM = 128
HEADS = 2
BATCH = 4
li = keras.Input((None,), dtype="int32")
lx = layers.Embedding(VOCAB, HDIM)(li)
r = lx
lx = layers.LayerNormalization()(lx)
lx = layers.MultiHeadAttention(HEADS, HDIM // HEADS)(
    lx, lx, use_causal_mask=True
)
lx = lx + r
r = lx
lx = layers.LayerNormalization()(lx)
lx = layers.Dense(HDIM * 4, activation="gelu")(lx)
lx = layers.Dense(HDIM)(lx) + r
lx = layers.Dense(VOCAB)(layers.LayerNormalization()(lx))
llm = keras.Model(li, lx)
ids = ops.convert_to_tensor(np.ones((BATCH, SEQ), dtype="int32"))
for _ in range(3):
    llm(ids, training=False)
hits.clear()
llm(ids, training=False)

from collections import Counter

unique = Counter()
for h in hits:
    lines = h.strip().split("\n")
    key = "\n".join(lines[-4:])
    unique[key] += 1

print(f"Total unflatten calls: {len(hits)}")
for k, v in unique.most_common(8):
    print(f"\n--- {v}x ---\n{k}")
