"""Detailed timing breakdown of Keras train step components."""

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
torch.mps.synchronize() if dev == "mps" else None

# Monkey-patch to measure individual components
from keras.src.trainers.data_adapters import data_adapter_utils

timings = {
    k: 0.0
    for k in [
        "unpack",
        "forward",
        "zero_grad",
        "compute_loss",
        "loss_tracker",
        "scale_loss",
        "backward",
        "grad_extract",
        "optimizer_apply",
        "compute_metrics",
        "pythonify",
        "total",
    ]
}

original_train_step = m.train_step.__func__


def timed_train_step(self, data):
    t = time.perf_counter

    t0 = t()
    x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)
    t1 = t()
    timings["unpack"] += t1 - t0

    if self._call_has_training_arg:
        y_pred = self(x, training=True)
    else:
        y_pred = self(x)
    t2 = t()
    timings["forward"] += t2 - t1

    self.zero_grad()
    t3 = t()
    timings["zero_grad"] += t3 - t2

    loss = self._compute_loss(
        x=x, y=y, y_pred=y_pred, sample_weight=sample_weight, training=True
    )
    t4 = t()
    timings["compute_loss"] += t4 - t3

    from keras.src import tree

    self._loss_tracker.update_state(
        loss,
        sample_weight=next(i for i in tree.flatten(x) if i is not None).shape[
            0
        ],
    )
    t5 = t()
    timings["loss_tracker"] += t5 - t4

    if self.optimizer is not None:
        loss = self.optimizer.scale_loss(loss)
    t6 = t()
    timings["scale_loss"] += t6 - t5

    cached = getattr(self, "_cached_trainable_weights", None)
    if cached is not None:
        trainable_weights = cached
    else:
        trainable_weights = self.trainable_weights
        object.__setattr__(self, "_cached_trainable_weights", trainable_weights)
    if trainable_weights:
        loss.backward()
    t7 = t()
    timings["backward"] += t7 - t6

    gradients = [v._value.grad for v in trainable_weights]
    t8 = t()
    timings["grad_extract"] += t8 - t7

    with torch.no_grad():
        self.optimizer.apply(gradients, trainable_weights)
    t9 = t()
    timings["optimizer_apply"] += t9 - t8

    result = self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)
    t10 = t()
    timings["compute_metrics"] += t10 - t9

    return result


import types

m.train_step = types.MethodType(timed_train_step, m)

if dev == "mps":
    torch.mps.synchronize()

t_start = time.perf_counter()
for _ in range(N):
    logs = m.train_on_batch(data, labels)
if dev == "mps":
    torch.mps.synchronize()
total = time.perf_counter() - t_start
timings["total"] = total

print(f"\n{'=' * 55}")
print(f"TRAIN STEP BREAKDOWN (N={N}, {total / N * 1000:.2f} ms/step)")
print(f"{'=' * 55}")
for k, v in sorted(timings.items(), key=lambda x: -x[1]):
    pct = v / total * 100
    ms = v / N * 1000
    print(f"  {k:20s}: {ms:6.3f} ms/step  ({pct:5.1f}%)")

# The pythonify happens inside train_on_batch after train_step
pythonify_ms = (
    (total - sum(v for k, v in timings.items() if k != "total")) / N * 1000
)
print(
    f"  {'overhead(wrapping)':20s}: {pythonify_ms:6.3f} ms/step  ({pythonify_ms / (total / N * 1000) * 100:5.1f}%)"
)
