"""Profile where the remaining 1.65ms of overhead is in the Keras[torch] LLM forward."""
import os
os.environ["KERAS_BACKEND"] = "torch"

import torch
import keras
from keras import layers

# Build benchmark LLM (matches bench.py)
vocab_size = 4096
seq_len = 128
d_model = 256
num_heads = 4
ff_dim = 512
num_layers = 4

inputs = keras.Input(shape=(seq_len,), dtype="int32")
x = layers.Embedding(vocab_size, d_model)(inputs)
for _ in range(num_layers):
    attn = layers.MultiHeadAttention(num_heads, d_model // num_heads)(x, x)
    x = layers.LayerNormalization()(x + attn)
    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(d_model)(ff)
    x = layers.LayerNormalization()(x + ff)
outputs = layers.Dense(vocab_size)(x)
model = keras.Model(inputs, outputs)

inp = torch.randint(0, vocab_size, (2, seq_len), device="mps")
for _ in range(5):
    model(inp)

# Profile with torch.profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    record_shapes=False,
    profile_memory=False,
    with_stack=False,
) as prof:
    for _ in range(20):
        model(inp)

# Show top Python-level functions by CPU time
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20, max_name_column_width=60))
