import os

os.environ["KERAS_BACKEND"] = "torch"
import torch

import keras
import keras.src.backend.torch.nn as torch_nn
import keras.src.backend.torch.numpy as torch_numpy
from keras.src.backend.torch import core as torch_core
from keras.src.layers import Dense
from keras.src.layers import Embedding
from keras.src.layers import LayerNormalization
from keras.src.layers import MultiHeadAttention

call_count = 0
original = torch_core.convert_to_tensor


def counting_ctt(*args, **kwargs):
    global call_count
    call_count += 1
    return original(*args, **kwargs)


torch_core.convert_to_tensor = counting_ctt

torch_nn.convert_to_tensor = counting_ctt
torch_numpy.convert_to_tensor = counting_ctt

vocab_size, seq_len, d_model, num_heads = 1000, 64, 128, 4
inputs = keras.Input(shape=(seq_len,), dtype="int32")
x = Embedding(vocab_size, d_model)(inputs)
for _ in range(3):
    attn = MultiHeadAttention(num_heads, d_model // num_heads)(x, x)
    x = LayerNormalization()(x + attn)
    ff = Dense(d_model * 4, activation="relu")(x)
    ff = Dense(d_model)(ff)
    x = LayerNormalization()(x + ff)
outputs = Dense(vocab_size)(x)
model = keras.Model(inputs, outputs)

inp = torch.randint(0, vocab_size, (4, seq_len), device="mps")
for _ in range(3):
    model(inp)

call_count = 0
for _ in range(10):
    model(inp)
avg = call_count / 10
print(f"convert_to_tensor calls per LLM forward: {avg:.0f}")
print(f"at 0.295us each: {avg * 0.295 / 1000:.3f} ms saved per forward")
print(f"out of 2.77ms baseline: {avg * 0.295 / 2770 * 100:.1f}% improvement")
