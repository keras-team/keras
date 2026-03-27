"""Profile per-step ops in the generate loop vs pure torch."""
import time
import torch
import numpy as np
import keras
from keras import layers, ops

VOCAB = 1024; SEQ = 64; HDIM = 256; HEADS = 4; NLAYERS = 2; BATCH = 4; GEN = 32

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

# Warmup
for _ in range(20):
    _ = llm(ids, training=False)
torch.mps.synchronize()

N = 50

# Time the full generate loop
def gen_keras(prompt):
    cur = prompt
    for _ in range(GEN):
        logits = llm(cur, training=False)
        nxt = ops.cast(ops.argmax(logits[:, -1:, :], axis=-1), "int32")
        cur = ops.concatenate([cur[:, 1:], nxt], axis=1)
    return cur

t0 = time.perf_counter()
for _ in range(N):
    out = gen_keras(ids)
torch.mps.synchronize()
full_gen = (time.perf_counter() - t0) / N * 1e3
print(f"Full generate loop:     {full_gen:.2f} ms  ({GEN} tokens)")

# Time the forward only (32 calls)
def fwd_only(prompt):
    cur = prompt
    for _ in range(GEN):
        logits = llm(cur, training=False)
    return logits

t0 = time.perf_counter()
for _ in range(N):
    out = fwd_only(ids)
torch.mps.synchronize()
fwd_only_time = (time.perf_counter() - t0) / N * 1e3
print(f"Forward-only loop:      {fwd_only_time:.2f} ms  ({GEN} static forwards without concat)")

# Time per-step ops only (no forward)
logits_fake = ops.zeros((BATCH, 1, VOCAB))
cur = ids

t0 = time.perf_counter()
for _ in range(N):
    c = ids
    for __ in range(GEN):
        nxt = ops.cast(ops.argmax(logits_fake, axis=-1), "int32")
        c = ops.concatenate([c[:, 1:], nxt], axis=1)
torch.mps.synchronize()
ops_overhead = (time.perf_counter() - t0) / N * 1e3
print(f"Per-step ops only:      {ops_overhead:.2f} ms  (argmax+cast+concat × {GEN})")
print(f"Per step:               {ops_overhead/GEN:.3f} ms avg")

# Time with pure torch ops
cur_pt = torch.ones(BATCH, SEQ, dtype=torch.int32, device="mps")
logits_pt = torch.zeros(BATCH, 1, VOCAB, device="mps")

t0 = time.perf_counter()
for _ in range(N):
    c = cur_pt
    for __ in range(GEN):
        nxt = torch.argmax(logits_pt, dim=-1).int()
        c = torch.cat([c[:, 1:], nxt], dim=1)
torch.mps.synchronize()
torch_ops = (time.perf_counter() - t0) / N * 1e3
print(f"Pure torch ops only:    {torch_ops:.2f} ms  (argmax+cat × {GEN})")
print(f"Keras ops overhead:     {ops_overhead - torch_ops:.2f} ms total, {(ops_overhead-torch_ops)/GEN:.3f} ms/step")
