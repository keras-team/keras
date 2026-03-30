"""cProfile the Keras LLM forward to find Python overhead."""

import os

os.environ["KERAS_BACKEND"] = "torch"

import cProfile
import io
import pstats

import torch

import keras
from keras import layers

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
for _ in range(10):
    model(inp)

# Profile
pr = cProfile.Profile()
pr.enable()
for _ in range(100):
    model(inp)
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(30)
print(s.getvalue())
