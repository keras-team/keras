"""Precise overhead measurement: train_on_batch wrapper vs raw train_step."""

import os

os.environ["KERAS_BACKEND"] = "torch"
import time

import torch

import keras
from keras import layers

VOCAB = 1024
SEQ = 64
HDIM = 256
HEADS = 4
NLAYERS = 2
BATCH = 4
N = 200
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
for _ in range(10):
    m.train_on_batch(data, labels)
if dev == "mps":
    torch.mps.synchronize()

# Test 1: train_on_batch (full wrapper)
if dev == "mps":
    torch.mps.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    m.train_on_batch(data, labels)
if dev == "mps":
    torch.mps.synchronize()
full_ms = (time.perf_counter() - t0) / N * 1000

# Test 2: Raw train_step + .item()
if dev == "mps":
    torch.mps.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    logs = m.train_step((data, labels, None))
    # Simulate pythonify_logs for loss value
    for k, v in logs.items():
        if hasattr(v, "item"):
            v.item()
if dev == "mps":
    torch.mps.synchronize()
raw_ms = (time.perf_counter() - t0) / N * 1000

# Test 3: Raw train_step, NO .item()
if dev == "mps":
    torch.mps.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    logs = m.train_step((data, labels, None))
if dev == "mps":
    torch.mps.synchronize()
nosync_ms = (time.perf_counter() - t0) / N * 1000

# Test 4: Raw forward + backward + optimizer (minimal Keras)
if dev == "mps":
    torch.mps.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    y_pred = m(data, training=True)
    m.zero_grad()
    loss = torch.nn.functional.cross_entropy(
        y_pred.reshape(-1, VOCAB), labels.reshape(-1).long()
    )
    loss.backward()
    with torch.no_grad():
        m.optimizer.apply(
            [v._value.grad for v in m._cached_trainable_weights],
            m._cached_trainable_weights,
        )
    loss.item()
if dev == "mps":
    torch.mps.synchronize()
minimal_ms = (time.perf_counter() - t0) / N * 1000


# Test 5: Pure torch equivalent
class PureLLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, HDIM)
        self.layers_list = torch.nn.ModuleList()
        for _ in range(NLAYERS):
            self.layers_list.append(
                torch.nn.TransformerEncoderLayer(
                    d_model=HDIM,
                    nhead=HEADS,
                    dim_feedforward=HDIM * 4,
                    activation="gelu",
                    batch_first=True,
                    dropout=0.0,
                )
            )
        self.ln = torch.nn.LayerNorm(HDIM)
        self.head = torch.nn.Linear(HDIM, VOCAB)

    def forward(self, x):
        x = self.embed(x)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(
            SEQ, device=x.device
        )
        for layer in self.layers_list:
            x = layer(x, src_mask=mask, is_causal=True)
        return self.head(self.ln(x))


pure = PureLLM().to(dev)
pure_opt = torch.optim.Adam(pure.parameters(), lr=1e-4)
ce = torch.nn.CrossEntropyLoss()
for _ in range(10):
    pure_opt.zero_grad()
    out = pure(data)
    l = ce(out.view(-1, VOCAB), labels.view(-1))
    l.backward()
    pure_opt.step()
if dev == "mps":
    torch.mps.synchronize()

if dev == "mps":
    torch.mps.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    pure_opt.zero_grad()
    out = pure(data)
    l = ce(out.view(-1, VOCAB), labels.view(-1))
    l.backward()
    pure_opt.step()
    l.item()
if dev == "mps":
    torch.mps.synchronize()
pure_ms = (time.perf_counter() - t0) / N * 1000

print(f"\n{'=' * 55}")
print(f"OVERHEAD BREAKDOWN (N={N})")
print(f"{'=' * 55}")
print(f"Pure torch (with .item())  : {pure_ms:.2f} ms/step")
print(
    f"Keras minimal              : {minimal_ms:.2f} ms/step  (+{(minimal_ms - pure_ms) / pure_ms * 100:.0f}%)"
)
print(
    f"Keras train_step (no sync) : {nosync_ms:.2f} ms/step  (+{(nosync_ms - pure_ms) / pure_ms * 100:.0f}%)"
)
print(
    f"Keras train_step + .item() : {raw_ms:.2f} ms/step  (+{(raw_ms - pure_ms) / pure_ms * 100:.0f}%)"
)
print(
    f"Keras train_on_batch       : {full_ms:.2f} ms/step  (+{(full_ms - pure_ms) / pure_ms * 100:.0f}%)"
)
print("")
print(f"Breakdown of overhead beyond pure torch ({full_ms - pure_ms:.2f} ms):")
print(
    f"  Keras model overhead  : {minimal_ms - pure_ms:.2f} ms  (Keras forward/optimizer vs pure torch)"
)
print(
    f"  train_step overhead   : {nosync_ms - minimal_ms:.2f} ms  (loss wrapper, metrics, etc.)"
)
print(
    f"  .item() sync overhead : {raw_ms - nosync_ms:.2f} ms  (GPU pipeline deeper for Keras)"
)
print(
    f"  train_on_batch wrapper: {full_ms - raw_ms:.2f} ms  (_symbolic_build, pythonify, flatten)"
)
