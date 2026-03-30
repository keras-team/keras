"""Benchmark with memory tracking."""

import os

os.environ["KERAS_BACKEND"] = "torch"
import time

import torch
import torch.nn as nn

import keras
from keras import layers

VOCAB = 1024
SEQ = 64
HDIM = 256
HEADS = 4
NLAYERS = 2
BATCH = 4
N = 200


def sync():
    torch.mps.synchronize()


dev = "mps" if torch.backends.mps.is_available() else "cpu"
data = torch.ones(BATCH, SEQ, dtype=torch.int32, device=dev)
labels = torch.ones(BATCH, SEQ, dtype=torch.int32, device=dev)


def get_mem():
    if dev == "mps":
        return torch.mps.current_allocated_memory() / 1024 / 1024
    return 0


# --- Pure Torch ---
class PureLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HDIM)
        self.layers_list = nn.ModuleList()
        for _ in range(NLAYERS):
            self.layers_list.append(
                nn.TransformerEncoderLayer(
                    d_model=HDIM,
                    nhead=HEADS,
                    dim_feedforward=HDIM * 4,
                    activation="gelu",
                    batch_first=True,
                    dropout=0.0,
                )
            )
        self.ln = nn.LayerNorm(HDIM)
        self.head = nn.Linear(HDIM, VOCAB)

    def forward(self, x):
        x = self.embed(x)
        mask = nn.Transformer.generate_square_subsequent_mask(
            SEQ, device=x.device
        )
        for layer in self.layers_list:
            x = layer(x, src_mask=mask, is_causal=True)
        return self.head(self.ln(x))


pure = PureLLM().to(dev)
pure_opt = torch.optim.Adam(pure.parameters(), lr=1e-4)
ce = nn.CrossEntropyLoss()
sync()
pure_model_mem = get_mem()

for _ in range(10):
    pure_opt.zero_grad()
    out = pure(data)
    loss = ce(out.view(-1, VOCAB), labels.view(-1))
    loss.backward()
    pure_opt.step()
sync()
pure_warmup_mem = get_mem()

sync()
t0 = time.perf_counter()
for _ in range(N):
    pure_opt.zero_grad()
    out = pure(data)
    loss = ce(out.view(-1, VOCAB), labels.view(-1))
    loss.backward()
    pure_opt.step()
    loss.item()  # sync to match Keras train_on_batch which calls .item()
sync()
pure_ms = (time.perf_counter() - t0) / N * 1000
pure_peak_mem = get_mem()

del pure, pure_opt, ce, out, loss
sync()
import gc

gc.collect()
torch.mps.empty_cache() if dev == "mps" else None
sync()

# --- Keras ---
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
sync()
keras_model_mem = get_mem()

for _ in range(10):
    m.train_on_batch(data, labels)
sync()
keras_warmup_mem = get_mem()

sync()
t0 = time.perf_counter()
for _ in range(N):
    m.train_on_batch(data, labels)
sync()
keras_ms = (time.perf_counter() - t0) / N * 1000
keras_peak_mem = get_mem()

print(f"\n{'=' * 50}")
print(f"PERFORMANCE (N={N} steps)")
print(f"{'=' * 50}")
print(f"Pure torch:   {pure_ms:.2f} ms/step")
print(f"Keras[torch]: {keras_ms:.2f} ms/step")
print(
    f"Overhead:     {keras_ms - pure_ms:.2f} ms ({(keras_ms - pure_ms) / pure_ms * 100:.0f}%)"
)
print(f"\n{'=' * 50}")
print("MEMORY (MPS allocated)")
print(f"{'=' * 50}")
print(
    f"Pure torch:  model={pure_model_mem:.1f}MB  warmup={pure_warmup_mem:.1f}MB  peak={pure_peak_mem:.1f}MB"
)
print(
    f"Keras:       model={keras_model_mem:.1f}MB  warmup={keras_warmup_mem:.1f}MB  peak={keras_peak_mem:.1f}MB"
)
print(
    f"Keras/Pure ratio: {keras_peak_mem / pure_peak_mem:.2f}x"
    if pure_peak_mem > 0
    else ""
)
