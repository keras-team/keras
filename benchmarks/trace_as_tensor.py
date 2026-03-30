"""Trace where torch.as_tensor is called from during training."""

import os
import traceback

os.environ["KERAS_BACKEND"] = "torch"

import torch

import keras
from keras import layers

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
model = keras.Model(inp, x)
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

data = torch.ones(BATCH, SEQ, dtype=torch.int32, device="mps")
labels = torch.ones(BATCH, SEQ, dtype=torch.int32, device="mps")

for _ in range(3):
    model.train_on_batch(data, labels)
torch.mps.synchronize()

callers = {}
orig = torch.as_tensor


def traced_as_tensor(*args, **kwargs):
    stack = traceback.extract_stack()
    key = []
    for f in stack[-6:-1]:
        fn = f.filename
        if "/keras/" in fn:
            fn = fn.split("/keras/")[-1]
        else:
            fn = fn.split("/")[-1]
        key.append((fn, f.lineno, f.name))
    callers[tuple(key)] = callers.get(tuple(key), 0) + 1
    return orig(*args, **kwargs)


torch.as_tensor = traced_as_tensor
model.train_on_batch(data, labels)
torch.as_tensor = orig

print("Total as_tensor calls:", sum(callers.values()))
for key, count in sorted(callers.items(), key=lambda x: -x[1]):
    print("\n  Count:", count)
    for fn, line, name in key:
        print("    {}:{} in {}".format(fn, line, name))
