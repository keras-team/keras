"""Full accounting of every component in train_on_batch."""

import os
import time

os.environ["KERAS_BACKEND"] = "torch"

import torch

import keras
from keras import layers
from keras.src.trainers.data_adapters import data_adapter_utils

VOCAB = 1024
SEQ = 64
HDIM = 256
HEADS = 4
NLAYERS = 2
BATCH = 4
N = 200


def sync():
    torch.mps.synchronize()


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

dev = "mps" if torch.backends.mps.is_available() else "cpu"
data = torch.ones(BATCH, SEQ, dtype=torch.int32, device=dev)
labels = torch.ones(BATCH, SEQ, dtype=torch.int32, device=dev)

# Warmup
for _ in range(5):
    model.train_on_batch(data, labels)
sync()

# Full train_on_batch
sync()
t0 = time.perf_counter()
for _ in range(N):
    model.train_on_batch(data, labels)
sync()
total = (time.perf_counter() - t0) / N * 1000
print(f"Full train_on_batch: {total:.2f} ms\n")

# Simulate every step of train_step exactly, measuring each piece
# From torch/trainer.py:train_step()

print("=== Step-by-step breakdown ===")

# 1. _symbolic_build
sync()
t0 = time.perf_counter()
for _ in range(N):
    model._symbolic_build(data_batch=(data, labels, None))
sync()
t_build = (time.perf_counter() - t0) / N * 1000
print(f"1. _symbolic_build:       {t_build:.3f} ms")

# 2. unpack_x_y_sample_weight
sync()
t0 = time.perf_counter()
for _ in range(N):
    x, y, sw = data_adapter_utils.unpack_x_y_sample_weight((data, labels))
sync()
t_unpack = (time.perf_counter() - t0) / N * 1000
print(f"2. unpack_x_y_sample_weight: {t_unpack:.3f} ms")

# 3. Forward pass (training=True)
sync()
t0 = time.perf_counter()
for _ in range(N):
    y_pred = model(data, training=True)
sync()
t_fwd = (time.perf_counter() - t0) / N * 1000
print(f"3. forward(training=True): {t_fwd:.3f} ms")

# 4. zero_grad
sync()
t0 = time.perf_counter()
for _ in range(N):
    model.zero_grad()
sync()
t_zero = (time.perf_counter() - t0) / N * 1000
print(f"4. zero_grad:             {t_zero:.3f} ms")

# 5. _compute_loss
y_pred = model(data, training=True)
sync()
t0 = time.perf_counter()
for _ in range(N):
    loss = model._compute_loss(
        x=data, y=labels, y_pred=y_pred, sample_weight=None, training=True
    )
sync()
t_loss = (time.perf_counter() - t0) / N * 1000
print(f"5. _compute_loss:         {t_loss:.3f} ms")

# 6. _loss_tracker.update_state
sync()
t0 = time.perf_counter()
for _ in range(N):
    model._loss_tracker.update_state(loss, sample_weight=BATCH)
sync()
t_loss_track = (time.perf_counter() - t0) / N * 1000
print(f"6. _loss_tracker.update:  {t_loss_track:.3f} ms")

# 7. optimizer.scale_loss
sync()
t0 = time.perf_counter()
for _ in range(N):
    loss_scaled = model.optimizer.scale_loss(loss)
sync()
t_scale = (time.perf_counter() - t0) / N * 1000
print(f"7. optimizer.scale_loss:  {t_scale:.3f} ms")

# 8. trainable_weights (property access)
sync()
t0 = time.perf_counter()
for _ in range(N):
    tw = model.trainable_weights
sync()
t_tw = (time.perf_counter() - t0) / N * 1000
print(f"8. trainable_weights:     {t_tw:.3f} ms")

# 9. loss.backward()
# Need to do fwd + zero + loss + bwd together for valid grads
sync()
t0 = time.perf_counter()
for _ in range(N):
    model.zero_grad()
    yp = model(data, training=True)
    l = model._compute_loss(
        x=data, y=labels, y_pred=yp, sample_weight=None, training=True
    )
    l.backward()
sync()
t_fwd_bwd = (time.perf_counter() - t0) / N * 1000
t_bwd = t_fwd_bwd - t_fwd - t_zero - t_loss
print(f"9. loss.backward():       {t_bwd:.3f} ms (estimated)")

# 10. grad extraction
tw = model.trainable_weights
sync()
t0 = time.perf_counter()
for _ in range(N):
    grads = [v.value.grad for v in tw]
sync()
t_grads = (time.perf_counter() - t0) / N * 1000
print(f"10. extract grads:        {t_grads:.3f} ms")

# 11. optimizer.apply (including no_grad context)
# Need valid grads first
model.zero_grad()
yp = model(data, training=True)
l = model._compute_loss(
    x=data, y=labels, y_pred=yp, sample_weight=None, training=True
)
l.backward()
grads = [v.value.grad for v in tw]
sync()
t0 = time.perf_counter()
for _ in range(N):
    with torch.no_grad():
        model.optimizer.apply(grads, tw)
sync()
t_opt = (time.perf_counter() - t0) / N * 1000
print(f"11. optimizer.apply:      {t_opt:.3f} ms")

# 12. compute_metrics
y_pred = model(data, training=True)
sync()
t0 = time.perf_counter()
for _ in range(N):
    result = model.compute_metrics(data, labels, y_pred, sample_weight=None)
sync()
t_metrics = (time.perf_counter() - t0) / N * 1000
print(f"12. compute_metrics:      {t_metrics:.3f} ms")

# Now account for train_on_batch overhead (the wrapper itself)
# train_on_batch does: callback handling, _symbolic_build, train_step, pythonify_logs
from keras.src.utils.python_utils import pythonify_logs

sync()
t0 = time.perf_counter()
for _ in range(N):
    pythonify_logs(result)
sync()
t_pylog = (time.perf_counter() - t0) / N * 1000
print(f"13. pythonify_logs:       {t_pylog:.3f} ms")

# Sum and compare
accounted = (
    t_build
    + t_unpack
    + t_fwd
    + t_zero
    + t_loss
    + t_loss_track
    + t_scale
    + t_tw
    + t_bwd
    + t_grads
    + t_opt
    + t_metrics
    + t_pylog
)
print("\n--- Summary ---")
print(f"Total train_on_batch:  {total:.2f} ms")
print(f"Sum of components:     {accounted:.2f} ms")
print(f"Unaccounted:           {total - accounted:.2f} ms")
print("\nTop 5 components:")
items = [
    ("forward", t_fwd),
    ("backward", t_bwd),
    ("optimizer", t_opt),
    ("_compute_loss", t_loss),
    ("_loss_tracker", t_loss_track),
    ("compute_metrics", t_metrics),
    ("zero_grad", t_zero),
    ("pythonify_logs", t_pylog),
    ("trainable_weights", t_tw),
    ("extract_grads", t_grads),
    ("scale_loss", t_scale),
    ("unpack", t_unpack),
    ("_symbolic_build", t_build),
]
items.sort(key=lambda x: -x[1])
for name, t in items[:8]:
    print(f"  {name:25s} {t:.3f} ms  ({t / total * 100:.1f}%)")

# Pure torch baseline
import torch.nn as nn


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
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            SEQ, device=x.device
        )
        for layer in self.layers_list:
            x = layer(x, src_mask=causal_mask, is_causal=True)
        return self.head(self.ln(x))


pure_model = PureLLM().to(dev)
pure_opt = torch.optim.Adam(pure_model.parameters(), lr=1e-4)
ce_loss = nn.CrossEntropyLoss()
for _ in range(5):
    pure_opt.zero_grad()
    logits = pure_model(data)
    l = ce_loss(logits.view(-1, VOCAB), labels.view(-1))
    l.backward()
    pure_opt.step()
sync()

sync()
t0 = time.perf_counter()
for _ in range(N):
    pure_opt.zero_grad()
    logits = pure_model(data)
    l = ce_loss(logits.view(-1, VOCAB), labels.view(-1))
    l.backward()
    pure_opt.step()
sync()
t_pure = (time.perf_counter() - t0) / N * 1000
print(f"\nPure torch train step: {t_pure:.2f} ms")
print(
    f"Keras overhead: {total - t_pure:.2f} ms ({(total - t_pure) / t_pure * 100:.0f}%)"
)
