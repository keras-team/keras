"""Trace torch.as_tensor calls during one forward pass."""
import os, traceback
os.environ["KERAS_BACKEND"] = "torch"
import torch
import numpy as np

orig = torch.as_tensor
hits = []

def traced(*a, **kw):
    t = traceback.extract_stack()
    hits.append(''.join(traceback.format_list(t[-7:-1])))
    return orig(*a, **kw)

torch.as_tensor = traced

import keras
from keras import layers, ops

VOCAB=1024; SEQ=64; HDIM=256; HEADS=4; NLAYERS=2; BATCH=4
inp = keras.Input((None,), dtype="int32")
x = layers.Embedding(VOCAB, HDIM)(inp)
for _ in range(NLAYERS):
    r = x
    x = layers.LayerNormalization()(x)
    x = layers.MultiHeadAttention(HEADS, HDIM//HEADS)(x, x, use_causal_mask=True)
    x = x + r; r = x
    x = layers.LayerNormalization()(x)
    x = layers.Dense(HDIM*4, activation="gelu")(x)
    x = layers.Dense(HDIM)(x) + r
x = layers.Dense(VOCAB)(layers.LayerNormalization()(x))
llm = keras.Model(inp, x)

ids = ops.convert_to_tensor(np.ones((BATCH, SEQ), dtype="int32"))
hits.clear()

llm(ids, training=False)

print(f"=== {len(hits)} as_tensor calls ===")
for i, h in enumerate(hits):
    print(f"\n--- call {i+1} ---\n{h}")
